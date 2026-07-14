"""
Two-stock basket option pricer — SINGLE FILE (engine + UI, no local imports).

Worst-of / best-of, calls and puts, buy or sell, on any two tickers.
Heston Monte Carlo calibrated to option-implied vols, validated against
closed forms (Stulz 1982 for options on the min; min/max identity for the max).

Deploy: put this file + requirements.txt in the repo root and point
Streamlit Cloud at basket_app.py. Nothing else is imported locally.

Run locally:  streamlit run basket_app.py
Self-test:    python basket_app.py --selftest
"""

APP_VERSION = "2.1 single-file, fast calibration"

import sys
import time
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import brentq, least_squares
from scipy.stats import multivariate_normal, norm


# =============================================================================
# ENGINE
# =============================================================================

from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.stats import multivariate_normal, norm


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------

@dataclass
class Asset:
    name: str
    spot: float
    div_yield: float = 0.0
    v0: float = 0.09
    kappa: float = 2.0
    theta: float = 0.09
    xi: float = 0.6
    rho_sv: float = -0.6

    def feller_ratio(self) -> float:
        return 2 * self.kappa * self.theta / max(self.xi**2, 1e-12)


@dataclass
class OptionSpec:
    basket_type: str = "worst"    # "worst" | "best"
    option_type: str = "put"      # "put"   | "call"
    strike_pct: float = 0.70
    tenor_years: float = 1.0
    r: float = 0.04
    notional: float = 1_000_000.0


@dataclass
class Result:
    premium_pct: float
    stderr_pct: float
    premium_cash: float
    prob_itm: float
    prob_driver_is: np.ndarray    # P(asset i sets the aggregate | ITM)
    exp_payoff_pct: float
    exp_payoff_given_itm: float
    max_payoff_pct: float
    var95_payoff_pct: float
    cvar95_payoff_pct: float
    breakeven_agg: float          # aggregate level where seller P&L = 0
    seller_pnl_pct: np.ndarray
    agg_T: np.ndarray
    perf_T: np.ndarray
    sample_paths: np.ndarray      # aggregate (worst/best) trajectories
    time_grid: np.ndarray
    deltas: np.ndarray
    vegas: np.ndarray
    corr_sens: float


# -----------------------------------------------------------------------------
# Black-Scholes closed forms for validation / flat-vol reference
# -----------------------------------------------------------------------------

def bs_vanilla(S, K, T, r, q, sigma, cp="call") -> float:
    if T <= 0 or sigma <= 0:
        f = S * np.exp(-q * T) - K * np.exp(-r * T)
        return max(f, 0.0) if cp == "call" else max(-f, 0.0)
    d1 = (np.log(S / K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if cp == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_vega(S, K, T, r, q, sigma) -> float:
    d1 = (np.log(S / K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def _bvn(a, b, rho):
    return float(multivariate_normal(mean=[0, 0],
                                     cov=[[1, rho], [rho, 1]]).cdf([a, b]))


def stulz_call_on_min(S1, S2, K, T, r, q1, q2, s1, s2, rho) -> float:
    if K <= 0:
        s = np.sqrt(s1**2 + s2**2 - 2 * rho * s1 * s2)
        d1 = (np.log(S2 / S1) + (q1 - q2 + s**2 / 2) * T) / (s * np.sqrt(T))
        d2 = d1 - s * np.sqrt(T)
        marg = S2 * np.exp(-q2 * T) * norm.cdf(d1) - S1 * np.exp(-q1 * T) * norm.cdf(d2)
        return S2 * np.exp(-q2 * T) - marg
    sq = np.sqrt(T)
    s12 = np.sqrt(s1**2 + s2**2 - 2 * rho * s1 * s2)
    d1 = (np.log(S1 / K) + (r - q1 + s1**2 / 2) * T) / (s1 * sq)
    d2 = (np.log(S2 / K) + (r - q2 + s2**2 / 2) * T) / (s2 * sq)
    d3 = (np.log(S2 / S1) + (q1 - q2 - s12**2 / 2) * T) / (s12 * sq)
    d4 = (np.log(S1 / S2) + (q2 - q1 - s12**2 / 2) * T) / (s12 * sq)
    r1 = (rho * s2 - s1) / s12
    r2 = (rho * s1 - s2) / s12
    return float(S1 * np.exp(-q1 * T) * _bvn(d1, d3, r1)
                 + S2 * np.exp(-q2 * T) * _bvn(d2, d4, r2)
                 - K * np.exp(-r * T) * _bvn(d1 - s1 * sq, d2 - s2 * sq, rho))


def bs_basket_option(basket_type, option_type, S1, S2, K, T, r,
                     q1, q2, s1, s2, rho) -> float:
    """Closed-form BS price for all four (worst/best) x (call/put) combos."""
    cmin = lambda k: stulz_call_on_min(S1, S2, k, T, r, q1, q2, s1, s2, rho)
    pv_min = cmin(0.0)
    pv_max = S1 * np.exp(-q1 * T) + S2 * np.exp(-q2 * T) - pv_min
    if basket_type == "worst":
        c = cmin(K)
        pv_agg = pv_min
    else:  # best: max(max-K,0) = max(S1-K,0)+max(S2-K,0)-max(min-K,0)
        c = (bs_vanilla(S1, K, T, r, q1, s1) + bs_vanilla(S2, K, T, r, q2, s2)
             - cmin(K))
        pv_agg = pv_max
    if option_type == "call":
        return float(c)
    return float(K * np.exp(-r * T) - pv_agg + c)   # put-call parity on agg


# -----------------------------------------------------------------------------
# Heston vanilla pricer (characteristic function) + implied vol + calibration
# -----------------------------------------------------------------------------

def _heston_cf_probs(u, wq, E, S0, K, T, r, q, v0, kappa, theta, xi, rho):
    """
    Vectorized Heston P1, P2 for a VECTOR of strikes at ONE maturity.

    Key speed insight: the characteristic function does not depend on the
    strike — only the Fourier factor exp(-i*u*ln K) does. So the CF is
    evaluated once per maturity (a length-n_quad vector) and all strikes are
    obtained with a single matrix product against a PRE-COMPUTED E matrix.
    This removes the O(n_strikes) inner loop that dominated calibration.

    E : (n_strikes, n_quad) complex array = exp(-i*u*lnK) / (i*u)
    """
    P = []
    for j in (1, 2):
        a = kappa * theta
        if j == 1:
            uu, b = 0.5, kappa - rho * xi
        else:
            uu, b = -0.5, kappa
        iu = 1j * u
        d = np.sqrt((rho * xi * iu - b) ** 2 - xi**2 * (2 * uu * iu - u**2))
        g = (b - rho * xi * iu + d) / (b - rho * xi * iu - d)
        ed = np.exp(d * T)
        C = (r - q) * iu * T + a / xi**2 * (
            (b - rho * xi * iu + d) * T - 2.0 * np.log((1 - g * ed) / (1 - g)))
        D = (b - rho * xi * iu + d) / xi**2 * (1 - ed) / (1 - g * ed)
        cf = np.exp(C + D * v0 + iu * np.log(S0))          # (n_quad,)
        integ = np.real(E * cf[None, :])                    # (n_strikes, n_quad)
        P.append(0.5 + (integ * wq[None, :]).sum(axis=1) / np.pi)
    return P[0], P[1]


def heston_prices_vec(S0, strikes, T, r, q, v0, kappa, theta, xi, rho,
                      cps=None, n_quad=96, u_max=150.0, cache={}):
    """All strikes at one maturity in a single vectorized pass."""
    strikes = np.atleast_1d(np.asarray(strikes, float))
    key = (n_quad, u_max)
    if key not in cache:
        x, w = np.polynomial.legendre.leggauss(n_quad)
        cache[key] = (0.5 * u_max * (x + 1.0), 0.5 * u_max * w)
    u, wq = cache[key]
    E = np.exp(-1j * u[None, :] * np.log(strikes)[:, None]) / (1j * u)[None, :]
    P1, P2 = _heston_cf_probs(u, wq, E, S0, strikes, T, r, q,
                              v0, kappa, theta, xi, rho)
    call = S0 * np.exp(-q * T) * P1 - strikes * np.exp(-r * T) * P2
    if cps is None:
        return call
    cps = np.asarray(cps)
    put = call - S0 * np.exp(-q * T) + strikes * np.exp(-r * T)
    return np.where(cps == "call", call, put)


def heston_vanilla(S0, K, T, r, q, v0, kappa, theta, xi, rho, cp="call",
                   n_quad=96, u_max=150.0) -> float:
    """Scalar convenience wrapper (kept for tests / single quotes)."""
    return float(heston_prices_vec(S0, [K], T, r, q, v0, kappa, theta, xi, rho,
                                   [cp], n_quad, u_max)[0])


def implied_vol(price, S, K, T, r, q, cp="call") -> float:
    intrinsic = bs_vanilla(S, K, T, r, q, 1e-9, cp)
    if price <= intrinsic + 1e-10:
        return np.nan
    try:
        return brentq(lambda s: bs_vanilla(S, K, T, r, q, s, cp) - price,
                      1e-4, 5.0, xtol=1e-8)
    except ValueError:
        return np.nan


def calibrate_heston(quotes: List[Tuple[float, float, float]],
                     S0: float, r: float, q: float,
                     x0: Optional[dict] = None,
                     n_quad: int = 128, max_nfev: int = 250) -> dict:
    """
    Fit (v0, kappa, theta, xi, rho) to option-implied vols.

    Residuals are price errors divided by Black-Scholes vega, which
    approximates implied-vol errors and conditions the problem far better than
    raw price errors across strikes.

    Speed: the model prices are grouped by maturity and priced with one
    vectorized CF pass each; all strike-dependent Fourier factors are
    pre-computed once, outside the optimizer. This is ~50-100x faster than
    pricing quote-by-quote.
    """
    quotes = [(t, k, iv) for (t, k, iv) in quotes
              if np.isfinite(iv) and 0.01 < iv < 4.0 and t > 1e-3]
    if len(quotes) < 5:
        raise ValueError("Need at least 5 valid option quotes to calibrate.")

    # ---- pre-compute everything that does not depend on the parameters ----
    x, w = np.polynomial.legendre.leggauss(n_quad)
    u = 0.5 * 150.0 * (x + 1.0)
    wq = 0.5 * 150.0 * w

    groups = {}                                   # T -> indices
    for i, (t, _, _) in enumerate(quotes):
        groups.setdefault(round(float(t), 6), []).append(i)

    n = len(quotes)
    mkt_px = np.empty(n); vegas = np.empty(n)
    cps = np.empty(n, dtype=object)
    for i, (t, k, iv) in enumerate(quotes):
        cp = "call" if k >= S0 else "put"         # always the OTM side
        cps[i] = cp
        mkt_px[i] = bs_vanilla(S0, k, t, r, q, iv, cp)
        vegas[i] = max(bs_vega(S0, k, t, r, q, iv), 1e-6)

    pre = {}                                      # T -> (strikes, E, idx, cps)
    for T, idx in groups.items():
        ks = np.array([quotes[i][1] for i in idx], float)
        E = np.exp(-1j * u[None, :] * np.log(ks)[:, None]) / (1j * u)[None, :]
        pre[T] = (ks, E, np.array(idx), np.array([cps[i] for i in idx]))

    x0 = x0 or {}
    p0 = np.array([x0.get("v0", 0.09), x0.get("kappa", 2.0),
                   x0.get("theta", 0.09), x0.get("xi", 0.8),
                   x0.get("rho_sv", -0.6)])
    lb = np.array([1e-4, 0.05, 1e-4, 0.05, -0.99])
    ub = np.array([4.0, 15.0, 4.0, 3.0, 0.50])
    p0 = np.clip(p0, lb + 1e-6, ub - 1e-6)

    def resid(p):
        v0, ka, th, xi_, rho = p
        out = np.empty(n)
        for T, (ks, E, idx, cp_g) in pre.items():
            try:
                P1, P2 = _heston_cf_probs(u, wq, E, S0, ks, T, r, q,
                                          v0, ka, th, xi_, rho)
                call = S0 * np.exp(-q * T) * P1 - ks * np.exp(-r * T) * P2
                put = call - S0 * np.exp(-q * T) + ks * np.exp(-r * T)
                px = np.where(cp_g == "call", call, put)
                px = np.where(np.isfinite(px), px, 1e3)
            except Exception:
                px = np.full(len(ks), 1e3)
            out[idx] = (px - mkt_px[idx]) / vegas[idx]
        return out

    sol = least_squares(resid, p0, bounds=(lb, ub), x_scale=np.array(
        [0.1, 2.0, 0.1, 0.5, 0.5]), xtol=1e-10, ftol=1e-10, max_nfev=max_nfev)
    v0, ka, th, xi_, rho = sol.x
    return {"v0": float(v0), "kappa": float(ka), "theta": float(th),
            "xi": float(xi_), "rho_sv": float(rho),
            "rmse_iv": float(np.sqrt(np.mean(sol.fun**2))),
            "n_quotes": n, "nfev": int(sol.nfev)}


def model_smile(S0, r, q, T, strikes, v0, kappa, theta, xi, rho) -> np.ndarray:
    """Heston implied-vol smile at tenor T (vectorized pricing)."""
    strikes = np.asarray(strikes, float)
    cps = np.where(strikes >= S0, "call", "put")
    px = heston_prices_vec(S0, strikes, T, r, q, v0, kappa, theta, xi, rho, cps)
    return np.array([implied_vol(p, S0, k, T, r, q, c)
                     for p, k, c in zip(px, strikes, cps)])


# -----------------------------------------------------------------------------
# Historical estimation (fallback when no option data)
# -----------------------------------------------------------------------------

def estimate_heston_from_returns(log_returns: np.ndarray,
                                 trading_days: int = 252) -> dict:
    lr = np.asarray(log_returns, float)
    lr = lr[np.isfinite(lr)]
    if lr.size < 60:
        raise ValueError("Need at least 60 return observations.")
    win = 21
    n = lr.size - win + 1
    rv = np.array([lr[i:i + win].var(ddof=1) for i in range(n)]) * trading_days
    rv = np.clip(rv, 1e-6, None)
    v0, theta = float(rv[-1]), float(np.median(rv))
    dt = 1.0 / trading_days
    dv, x = np.diff(rv), theta - rv[:-1]
    den = float(x @ x)
    kappa = float(np.clip((x @ dv) / (den * dt) if den > 0 else 2.0, 0.3, 10.0))
    resid = dv - kappa * x * dt
    xi = float(np.clip(np.std(resid, ddof=1)
                       / (np.sqrt(rv[:-1].mean()) * np.sqrt(dt)), 0.10, 2.5))
    dr = lr[win - 1:][1:n]
    rho = -0.6
    if dr.size == dv.size and dv.size > 10:
        c = np.corrcoef(dr, dv)[0, 1]
        if np.isfinite(c):
            rho = float(np.clip(c, -0.95, 0.2))
    return {"v0": v0, "theta": theta, "kappa": kappa, "xi": xi, "rho_sv": rho}


# -----------------------------------------------------------------------------
# Monte Carlo
# -----------------------------------------------------------------------------

def _simulate_terminal(a1: Asset, a2: Asset, rho_s: float, spec: OptionSpec,
                       n_paths: int, steps_per_year: int, seed: int,
                       n_saved: int = 0, basket_type: str = "worst"):
    T = spec.tenor_years
    n_steps = max(int(round(T * steps_per_year)), 2)
    dt, sqdt = T / n_steps, np.sqrt(T / n_steps)
    if n_paths % 2:
        n_paths += 1
    half = n_paths // 2
    rng = np.random.default_rng(seed)

    rho_s = float(np.clip(rho_s, -0.999, 0.999))
    L = np.linalg.cholesky(np.array([[1.0, rho_s], [rho_s, 1.0]]))
    S0 = np.array([a1.spot, a2.spot])
    q = np.array([a1.div_yield, a2.div_yield])
    kap = np.array([a1.kappa, a2.kappa]); th = np.array([a1.theta, a2.theta])
    xi = np.array([a1.xi, a2.xi]); rho = np.array([a1.rho_sv, a2.rho_sv])
    rho_c = np.sqrt(np.clip(1 - rho**2, 0, 1))

    S = np.tile(S0, (n_paths, 1))
    v = np.tile([a1.v0, a2.v0], (n_paths, 1))
    saved = np.ones((n_saved, n_steps + 1)) if n_saved else None
    tg = np.linspace(0, T, n_steps + 1)
    aggfun = np.min if basket_type == "worst" else np.max

    for t in range(1, n_steps + 1):
        Zs_h = rng.standard_normal((half, 2))
        Zw_h = rng.standard_normal((half, 2))
        Zs = np.vstack([Zs_h, -Zs_h]); Zw = np.vstack([Zw_h, -Zw_h])
        Zc = Zs @ L.T
        Zv = rho * Zc + rho_c * Zw
        vp = np.maximum(v, 0.0); sq = np.sqrt(vp)
        S = S * np.exp((spec.r - q - 0.5 * vp) * dt + sq * sqdt * Zc)
        v = v + kap * (th - vp) * dt + xi * sq * sqdt * Zv
        if n_saved:
            saved[:, t] = aggfun(S[:n_saved] / S0, axis=1)

    return S / S0, saved, tg


def _payoff(perf: np.ndarray, spec: OptionSpec) -> np.ndarray:
    agg = perf.min(axis=1) if spec.basket_type == "worst" else perf.max(axis=1)
    if spec.option_type == "put":
        return np.maximum(spec.strike_pct - agg, 0.0)
    return np.maximum(agg - spec.strike_pct, 0.0)


def price_basket_option(a1: Asset, a2: Asset, rho_s: float, spec: OptionSpec,
                        n_paths: int = 50_000, steps_per_year: int = 252,
                        seed: Optional[int] = 42, n_saved: int = 200,
                        greeks: bool = True) -> Result:
    perf, saved, tg = _simulate_terminal(a1, a2, rho_s, spec, n_paths,
                                         steps_per_year, seed, n_saved,
                                         spec.basket_type)
    n_paths = perf.shape[0]
    agg = perf.min(axis=1) if spec.basket_type == "worst" else perf.max(axis=1)
    K = spec.strike_pct
    df = np.exp(-spec.r * spec.tenor_years)

    payoff = _payoff(perf, spec)
    pv = df * payoff
    premium = float(pv.mean())
    stderr = float(pv.std(ddof=1) / np.sqrt(n_paths))

    itm = payoff > 0
    p_itm = float(itm.mean())
    if itm.any():
        which = (perf[itm].argmin(axis=1) if spec.basket_type == "worst"
                 else perf[itm].argmax(axis=1))
        p_which = np.array([(which == 0).mean(), (which == 1).mean()])
        e_pay_itm = float(payoff[itm].mean())
    else:
        p_which = np.array([0.5, 0.5]); e_pay_itm = 0.0

    seller_pnl = premium / df - payoff
    tail = np.sort(payoff)[::-1]
    var95 = float(np.percentile(payoff, 95))
    cvar95 = float(tail[: max(int(0.05 * n_paths), 1)].mean())
    be = (K - premium / df) if spec.option_type == "put" else (K + premium / df)

    pct = lambda x: 100.0 * x

    deltas = np.zeros(2); vegas = np.zeros(2); corr_sens = 0.0
    if greeks:
        ng = min(n_paths, 30_000)

        def px(aa1, aa2, rs, spot_mult=(1.0, 1.0)):
            p, _, _ = _simulate_terminal(aa1, aa2, rs, spec, ng,
                                         steps_per_year, seed, 0,
                                         spec.basket_type)
            return pct(df * _payoff(p * np.asarray(spot_mult), spec).mean())

        base = px(a1, a2, rho_s)
        for i, a in enumerate((a1, a2)):
            mu = [1.0, 1.0]; md = [1.0, 1.0]
            mu[i], md[i] = 1.01, 0.99
            deltas[i] = (px(a1, a2, rho_s, tuple(mu))
                         - px(a1, a2, rho_s, tuple(md))) / 2.0
            vol = np.sqrt(a.v0)
            vu = replace(a, v0=(vol + 0.01) ** 2,
                         theta=(np.sqrt(a.theta) + 0.01) ** 2)
            args_v = (vu, a2, rho_s) if i == 0 else (a1, vu, rho_s)
            vegas[i] = px(*args_v) - base
        corr_sens = px(a1, a2, min(rho_s + 0.05, 0.999)) - base

    return Result(
        premium_pct=pct(premium), stderr_pct=pct(stderr),
        premium_cash=premium * spec.notional,
        prob_itm=p_itm, prob_driver_is=p_which,
        exp_payoff_pct=pct(payoff.mean()),
        exp_payoff_given_itm=pct(e_pay_itm),
        max_payoff_pct=pct(payoff.max()),
        var95_payoff_pct=pct(var95), cvar95_payoff_pct=pct(cvar95),
        breakeven_agg=be, seller_pnl_pct=pct(seller_pnl),
        agg_T=agg, perf_T=perf, sample_paths=saved, time_grid=tg,
        deltas=deltas, vegas=vegas, corr_sens=corr_sens)


# -----------------------------------------------------------------------------
# Self-tests
# -----------------------------------------------------------------------------




# =============================================================================
# Self-test:  python basket_app.py --selftest
# =============================================================================

def _selftest():
    s1, s2, rho, T, r = 0.45, 0.55, 0.55, 1.0, 0.04
    a1 = Asset("A", 250.0, 0.0, v0=s1**2, kappa=1.0, theta=s1**2, xi=1e-4, rho_sv=0.0)
    a2 = Asset("B", 130.0, 0.0, v0=s2**2, kappa=1.0, theta=s2**2, xi=1e-4, rho_sv=0.0)
    for bt, ot, K in [("worst", "put", 0.70), ("worst", "call", 1.00),
                      ("best", "call", 1.30), ("best", "put", 1.00)]:
        spec = OptionSpec(bt, ot, K, T, r)
        ana = bs_basket_option(bt, ot, 1.0, 1.0, K, T, r, 0, 0, s1, s2, rho)
        res = price_basket_option(a1, a2, rho, spec, n_paths=200_000, seed=11,
                                  n_saved=0, greeks=False)
        mc = res.premium_pct / 100
        ok = abs(ana - mc) < 4 * res.stderr_pct / 100 + 3e-4
        print(f"{bt:>5}-of {ot:<4} K={K:.0%}: analytic={ana:.5f} MC={mc:.5f} "
              f"{'OK' if ok else 'FAIL'}")
        assert ok
    print(f"basket_app v{APP_VERSION}: all closed-form checks passed.")


if "--selftest" in sys.argv:
    _selftest()
    sys.exit(0)


# =============================================================================
# STREAMLIT UI
# =============================================================================

st.set_page_config(page_title="Basket Option Pricer", page_icon="🧾", layout="wide")

ACCENT, RED, GREEN, GRAY, AMBER = "#4b8bc4", "#d6564a", "#2aa574", "#8a8a85", "#cf9b2c"

# Theme-adaptive styling: translucent fills + inherited text color, so nothing
# renders as a white box hiding text in dark mode.
st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; max-width: 1250px;}
  div[data-testid="stMetric"] {
    background: rgba(128,128,128,0.10);
    border: 1px solid rgba(128,128,128,0.28);
    border-radius: 10px; padding: 14px 18px;
  }
  div[data-testid="stMetric"] * {color: inherit;}
  .note-box {background: rgba(75,139,196,0.14); border-left: 4px solid #4b8bc4;
    padding: 12px 16px; border-radius: 0 8px 8px 0; font-size: .92rem;
    margin: 6px 0 14px 0; color: inherit;}
  .warn-box {background: rgba(214,86,74,0.14); border-left: 4px solid #d6564a;
    padding: 12px 16px; border-radius: 0 8px 8px 0; font-size: .92rem;
    margin: 6px 0 14px 0; color: inherit;}
</style>""", unsafe_allow_html=True)

st.title("Two-stock basket option pricer")
st.caption(f"Worst-of / best-of · calls and puts · sell or buy · Heston Monte Carlo "
           f"calibrated to option-implied vols, validated against closed forms · v{APP_VERSION}")

# ----------------------------------------------------------------- data layer
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_hist(tickers: tuple, lookback: str):
    import yfinance as yf
    px = yf.download(list(tickers), period=lookback, auto_adjust=True,
                     progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(tickers[0])
    px = px[list(tickers)].dropna()
    if len(px) < 80:
        raise ValueError("Not enough overlapping history for these tickers.")
    lr = np.log(px / px.shift(1)).dropna()
    divs = {}
    for t in tickers:
        try:
            y = yf.Ticker(t).info.get("dividendYield", 0.0) or 0.0
            divs[t] = float(y) if y < 1 else float(y) / 100.0
        except Exception:
            divs[t] = 0.0
    return px, lr, float(lr.corr().iloc[0, 1]), divs


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_option_quotes(ticker: str, spot: float, tenor: float):
    """OTM implied-vol quotes from the listed option chain, on expiries
    bracketing the trade tenor. Returns list of (T, K, iv)."""
    import datetime as dtm
    import yfinance as yf
    tk = yf.Ticker(ticker)
    today = dtm.date.today()
    exps = []
    for e in tk.options:
        T = (dtm.date.fromisoformat(e) - today).days / 365.25
        if T > 0.05:
            exps.append((e, T))
    if not exps:
        raise ValueError("No listed expiries found.")
    # Two expiries bracketing the tenor is enough to pin the smile AND its
    # term structure; each extra expiry is another network round-trip.
    chosen = []
    for target in (tenor, 0.6 * tenor):
        e = min(exps, key=lambda x: abs(x[1] - target))
        if e not in chosen:
            chosen.append(e)
    quotes = []
    for e, T in chosen:
        ch = tk.option_chain(e)
        for df, side in ((ch.puts, "put"), (ch.calls, "call")):
            df = df.copy()
            df = df[(df["impliedVolatility"] > 0.02)
                    & (df["impliedVolatility"] < 4.0)]
            if side == "put":
                df = df[(df["strike"] >= 0.45 * spot) & (df["strike"] <= spot)]
            else:
                df = df[(df["strike"] > spot) & (df["strike"] <= 1.5 * spot)]
            df = df.sort_values("strike")
            if len(df) > 6:                       # thin to ~6 per side
                df = df.iloc[np.linspace(0, len(df) - 1, 6).astype(int)]
            for _, row in df.iterrows():
                quotes.append((float(T), float(row["strike"]),
                               float(row["impliedVolatility"])))
    if len(quotes) < 5:
        raise ValueError("Too few usable option quotes.")
    return quotes



@st.cache_data(ttl=1800, show_spinner=False)
def cached_calibrate(ticker: str, spot: float, tenor: float, r: float,
                     q: float, x0: dict):
    """Cached so changing an unrelated widget doesn't refit the smile."""
    quotes = fetch_option_quotes(ticker, spot, tenor)
    fit = calibrate_heston(quotes, spot, r, q, x0)
    return fit, quotes


# --------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Underlyings")
    t1 = st.text_input("Ticker 1", "NVDA").strip().upper()
    t2 = st.text_input("Ticker 2", "TSLA").strip().upper()
    tickers = (t1, t2)
    lookback = st.selectbox("History window", ["1y", "2y", "3y", "5y"], 1)
    use_live = st.toggle("Fetch live market data (yfinance)", True)
    calib_src = st.radio("Heston calibration",
                         ["Option-implied vols (recommended)",
                          "Historical returns"],
                         help="Implied calibration fits v0, kappa, theta, xi, rho "
                              "to the listed option smile — the market-consistent "
                              "choice. Falls back to historical if no chain data.")

    st.header("Trade")
    basket_type = st.radio("Basket", ["Worst-of", "Best-of"], horizontal=True)
    option_type = st.radio("Option", ["Put", "Call"], horizontal=True)
    position = st.radio("Your position", ["Sell", "Buy"], horizontal=True)
    strike_pct = st.slider("Strike (% of spot)", 40, 160, 70, 1) / 100
    tenor = st.slider("Tenor (years)", 0.25, 3.0, 1.0, 0.25)
    notional = st.number_input("Notional (USD)", 10_000, 1_000_000_000,
                               1_000_000, step=10_000)
    r = st.number_input("Risk-free rate (% p.a.)", 0.0, 15.0, 4.0, 0.1) / 100

    st.header("Simulation")
    n_paths = st.select_slider("Monte Carlo paths",
                               [10_000, 20_000, 50_000, 100_000, 200_000], 50_000)
    seed = st.number_input("Random seed", 0, 10_000, 42)
    do_greeks = st.toggle("Compute Greeks (slower)", True)

if t1 == t2 or not t1 or not t2:
    st.error("Enter two different, non-empty tickers.")
    st.stop()

bt = "worst" if basket_type == "Worst-of" else "best"
ot = option_type.lower()
selling = position == "Sell"
sgn = 1.0 if selling else -1.0     # premium flows to you if selling

# ------------------------------------------------------------ market data path
data_ok, err = False, ""
if use_live:
    try:
        with st.spinner("Fetching price history…"):
            hist, lr, rho_hist, divs = fetch_hist(tickers, lookback)
        spots = {t: float(hist[t].iloc[-1]) for t in tickers}
        data_ok = True
    except Exception as e:
        err = str(e)

if not data_ok:
    if use_live:
        reason = err or "no connection"
        st.markdown(f"<div class='warn-box'><b>Live data unavailable</b> "
                    f"({reason}). Enter market inputs manually — the pricer "
                    f"works identically.</div>", unsafe_allow_html=True)
    c = st.columns(2)
    spots, vols, divs = {}, {}, {}
    for i, t in enumerate(tickers):
        with c[i]:
            st.markdown(f"**{t}**")
            spots[t] = st.number_input(f"Spot {t}", 0.01, 1e6, 100.0, key=f"s{i}")
            vols[t] = st.number_input(f"Vol {t} (% p.a.)", 1.0, 300.0,
                                      45.0 if i == 0 else 55.0, key=f"v{i}") / 100
            divs[t] = st.number_input(f"Div yield {t} (%)", 0.0, 20.0, 0.0,
                                      key=f"d{i}") / 100
    rho_hist = st.slider("Spot correlation", -0.5, 0.99, 0.55, 0.01)
    hist, lr = None, None

# --------------------------------------------------- calibration per underlying
st.subheader("1 · Model calibration")

assets, calib_info, smiles = [], [], {}
cols = st.columns(2)
for i, t in enumerate(tickers):
    est, source = None, ""
    if data_ok and calib_src.startswith("Option"):
        try:
            with st.spinner(f"Calibrating {t} to its option smile…"):
                _t0 = time.time()
                x0 = estimate_heston_from_returns(lr[t].values)
                est, quotes = cached_calibrate(t, spots[t], tenor, r,
                                               divs.get(t, 0.0), x0)
                source = (f"option-implied · {est['n_quotes']} quotes · "
                          f"fit RMSE {est['rmse_iv']*100:.2f} vol pts · "
                          f"{time.time()-_t0:.1f}s")
                smiles[t] = quotes
        except Exception as e:
            est = None
            st.markdown(f"<div class='warn-box'>Implied calibration failed for "
                        f"<b>{t}</b> ({e}) — using historical fallback.</div>",
                        unsafe_allow_html=True)
    if est is None:
        if data_ok:
            est = estimate_heston_from_returns(lr[t].values)
            source = "historical returns (fallback)"
        else:
            vv = vols[t] ** 2
            est = {"v0": vv, "theta": vv, "kappa": 2.0, "xi": 0.8, "rho_sv": -0.6}
            source = "manual flat vol"

    with cols[i]:
        st.markdown(f"**{t}** · spot **{spots[t]:,.2f}** · strike "
                    f"**{spots[t]*strike_pct:,.2f}** ({strike_pct:.0%})")
        st.caption(f"Calibration: {source}")
        v0 = st.number_input("v₀", 0.0001, 4.0, float(round(est["v0"], 4)),
                             format="%.4f", key=f"v0{i}")
        th = st.number_input("θ", 0.0001, 4.0, float(round(est["theta"], 4)),
                             format="%.4f", key=f"th{i}")
        ka = st.number_input("κ", 0.05, 15.0, float(round(est["kappa"], 2)),
                             key=f"ka{i}")
        xv = st.number_input("ξ (vol of vol)", 0.01, 3.0,
                             float(round(est["xi"], 2)), key=f"xi{i}")
        rh = st.number_input("ρ spot-vol", -0.99, 0.5,
                             float(round(est["rho_sv"], 2)), key=f"rh{i}")
        st.caption(f"√v₀ = {np.sqrt(v0):.1%} · √θ = {np.sqrt(th):.1%}")
        assets.append(Asset(t, spots[t], divs.get(t, 0.0), v0, ka, th, xv, rh))

rho_s = st.slider("Spot correlation between the two stocks", -0.5, 0.99,
                  float(round(np.clip(rho_hist, -0.5, 0.99), 2)), 0.01)

# fit-quality smile charts
if smiles:
    cols = st.columns(len(smiles))
    for i, (t, quotes) in enumerate(smiles.items()):
        a = assets[tickers.index(t)]
        Ts = sorted({round(q[0], 3) for q in quotes})
        Tn = min(Ts, key=lambda x: abs(x - tenor))
        pts = [(k, iv) for (tt, k, iv) in quotes if round(tt, 3) == Tn]
        ks = np.array([p[0] for p in pts]); ivs = np.array([p[1] for p in pts])
        grid = np.linspace(ks.min(), ks.max(), 25)
        mdl = model_smile(spots[t], r, a.div_yield, Tn, grid,
                          a.v0, a.kappa, a.theta, a.xi, a.rho_sv)
        with cols[i]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ks / spots[t] * 100, y=ivs * 100,
                                     mode="markers", name="market IV",
                                     marker=dict(color=AMBER, size=7)))
            fig.add_trace(go.Scatter(x=grid / spots[t] * 100, y=mdl * 100,
                                     mode="lines", name="Heston fit",
                                     line=dict(color=ACCENT, width=2)))
            fig.add_vline(x=strike_pct * 100, line_dash="dot", line_color=RED,
                          annotation_text="trade strike")
            fig.update_layout(title=f"{t} implied-vol smile (T≈{Tn:.2f}y)",
                              height=280, margin=dict(l=10, r=10, t=45, b=10),
                              xaxis_title="Strike (% of spot)",
                              yaxis_title="Implied vol (%)",
                              legend=dict(x=0.55, y=0.98))
            st.plotly_chart(fig, use_container_width=True)
    st.markdown("<div class='note-box'>The Heston fit should hug the market "
                "points, especially near your trade strike (red line) — that's "
                "the region that prices this option. A poor fit there means "
                "adjust parameters manually.</div>", unsafe_allow_html=True)

if data_ok and hist is not None:
    norm_px = hist / hist.iloc[0] * 100
    fig = go.Figure()
    for t in tickers:
        fig.add_trace(go.Scatter(x=norm_px.index, y=norm_px[t], name=t,
                                 line=dict(width=1.6)))
    fig.update_layout(title=f"Normalised history ({lookback}) · realized corr "
                            f"= {rho_hist:.2f}", height=260,
                      margin=dict(l=10, r=10, t=45, b=10),
                      yaxis_title="Rebased to 100")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------- price
spec = OptionSpec(bt, ot, strike_pct, tenor, r, notional)
trade_name = f"{basket_type} {option_type.lower()}"

st.subheader(f"2 · {position} the {trade_name}: price & risk")
if not st.button(f"▶ Price the {trade_name}", type="primary",
                 use_container_width=True):
    st.info("Adjust terms in the sidebar, then price.")
    st.stop()

with st.spinner(f"Simulating {n_paths:,} Heston paths…"):
    res = price_basket_option(assets[0], assets[1], rho_s, spec,
                              n_paths=int(n_paths), seed=int(seed),
                              greeks=do_greeks)

bs_ref = bs_basket_option(bt, ot, 1.0, 1.0, strike_pct, tenor, r,
                          assets[0].div_yield, assets[1].div_yield,
                          float(np.sqrt(assets[0].theta)),
                          float(np.sqrt(assets[1].theta)), rho_s) * 100

driver = "worst performer" if bt == "worst" else "best performer"
prem_word = "receive" if selling else "pay"

m = st.columns(5)
m[0].metric(f"Premium you {prem_word}", f"{res.premium_pct:.2f}%",
            f"± {2*res.stderr_pct:.2f}% (95% CI)")
m[1].metric("Premium in cash", f"${res.premium_cash:,.0f}",
            f"on ${notional:,.0f}")
m[2].metric("P(exercise)", f"{res.prob_itm:.1%}",
            f"{driver} {'<' if ot=='put' else '>'} {strike_pct:.0%}",
            delta_color="off")
m[3].metric("E[payout] if exercised", f"{res.exp_payoff_given_itm:.1f}%",
            "of notional", delta_color="off")
m[4].metric("Breakeven level", f"{res.breakeven_agg:.1%}",
            f"of initial ({driver})", delta_color="off")

m = st.columns(5)
m[0].metric("95% tail payout (CVaR)", f"{res.cvar95_payoff_pct:.1f}%",
            "avg of worst 5% for the seller", delta_color="off")
m[1].metric("Max simulated payout", f"{res.max_payoff_pct:.1f}%", delta_color="off")
m[2].metric(f"P({tickers[0]} drives it | ITM)",
            f"{res.prob_driver_is[0]:.0%}")
m[3].metric("Heston vs flat-vol", f"{res.premium_pct - bs_ref:+.2f}%",
            f"BS/Stulz ref {bs_ref:.2f}%", delta_color="off")
m[4].metric("Annualized premium", f"{res.premium_pct/tenor:.2f}% p.a.")

if selling:
    msg = (f"You collect <b>${res.premium_cash:,.0f}</b> today. In "
           f"{res.prob_itm:.0%} of paths the {driver} finishes "
           f"{'below' if ot=='put' else 'above'} {strike_pct:.0%} and you pay "
           f"the difference — averaging {res.exp_payoff_given_itm:.1f}% of "
           f"notional when exercised. You stay profitable while the {driver} "
           f"stays {'above' if ot=='put' else 'below'} "
           f"<b>{res.breakeven_agg:.1%}</b>.")
else:
    msg = (f"You pay <b>${res.premium_cash:,.0f}</b> today. The option pays "
           f"off in {res.prob_itm:.0%} of paths, averaging "
           f"{res.exp_payoff_given_itm:.1f}% of notional when it does. You "
           f"profit if the {driver} finishes "
           f"{'below' if ot=='put' else 'above'} <b>{res.breakeven_agg:.1%}</b>.")
st.markdown(f"<div class='note-box'>{msg}</div>", unsafe_allow_html=True)

if do_greeks:
    st.markdown(f"**Greeks (per 100 notional, from your side as {position.lower()}er)**")
    your = -sgn  # seller is short the option's sensitivities
    gdf = pd.DataFrame({
        "Sensitivity": [f"Delta {tickers[0]} (+1% spot)",
                        f"Delta {tickers[1]} (+1% spot)",
                        f"Vega {tickers[0]} (+1 vol pt)",
                        f"Vega {tickers[1]} (+1 vol pt)",
                        "Correlation (+0.05)"],
        "Option value": [f"{res.deltas[0]:+.3f}", f"{res.deltas[1]:+.3f}",
                         f"{res.vegas[0]:+.3f}", f"{res.vegas[1]:+.3f}",
                         f"{res.corr_sens:+.3f}"],
        "Your P&L": [f"{your*res.deltas[0]:+.3f}", f"{your*res.deltas[1]:+.3f}",
                     f"{your*res.vegas[0]:+.3f}", f"{your*res.vegas[1]:+.3f}",
                     f"{your*res.corr_sens:+.3f}"]})
    st.dataframe(gdf, hide_index=True, use_container_width=True)
    corr_note = ("Worst-of options lose value as correlation rises (the basket "
                 "behaves more like one stock); best-of options gain from "
                 "dispersion the opposite way. Whoever is short the option is "
                 "on the other side of that correlation exposure.")
    st.caption(corr_note)

# --------------------------------------------------------------------- charts
your_pnl = sgn * res.seller_pnl_pct
cA, cB = st.columns(2)
with cA:
    fig = go.Figure(go.Histogram(x=your_pnl, nbinsx=90,
                                 marker_color=ACCENT, opacity=0.85))
    fig.add_vline(x=0, line_dash="dash", line_color=GRAY)
    fig.add_vline(x=float(np.mean(your_pnl)), line_color=GREEN,
                  annotation_text=f"mean {np.mean(your_pnl):+.1f}%")
    fig.update_layout(title=f"Your P&L at maturity ({position.lower()}er, % of "
                            f"notional, premium accrued)",
                      height=330, margin=dict(l=10, r=10, t=45, b=10),
                      xaxis_title="P&L %", yaxis_title="Paths")
    st.plotly_chart(fig, use_container_width=True)

with cB:
    x = np.linspace(0.05, 1.9, 300)
    prem_T = res.premium_pct / np.exp(-r * tenor) / 100
    opt_pay = np.maximum(strike_pct - x, 0) if ot == "put" \
        else np.maximum(x - strike_pct, 0)
    pnl = sgn * (prem_T - opt_pay) * 100
    fig = go.Figure(go.Scatter(x=x * 100, y=pnl,
                               line=dict(color=ACCENT, width=2.5)))
    fig.add_hline(y=0, line_color=GRAY, line_width=1)
    fig.add_vline(x=strike_pct * 100, line_dash="dot", line_color=AMBER,
                  annotation_text="strike")
    fig.add_vline(x=res.breakeven_agg * 100, line_dash="dot", line_color=RED,
                  annotation_text="breakeven")
    fig.update_layout(title=f"Payoff diagram vs final {driver}",
                      height=330, margin=dict(l=10, r=10, t=45, b=10),
                      xaxis_title=f"Final {driver} (% of initial)",
                      yaxis_title="Your P&L (% of notional)")
    st.plotly_chart(fig, use_container_width=True)

cA, cB = st.columns(2)
with cA:
    if res.sample_paths is not None:
        fig = go.Figure()
        for i in range(min(130, res.sample_paths.shape[0])):
            end = res.sample_paths[i, -1]
            itm = (end < strike_pct) if ot == "put" else (end > strike_pct)
            fig.add_trace(go.Scatter(
                x=res.time_grid, y=res.sample_paths[i] * 100, mode="lines",
                line=dict(width=0.8,
                          color=RED if itm else "rgba(75,139,196,0.30)"),
                showlegend=False, hoverinfo="skip"))
        fig.add_hline(y=strike_pct * 100, line_dash="dot", line_color=AMBER,
                      annotation_text="strike", annotation_position="right")
        fig.update_layout(title=f"Sample {driver} paths (red = exercised)",
                          height=330, margin=dict(l=10, r=10, t=45, b=10),
                          xaxis_title="Years",
                          yaxis_title=f"{basket_type} level (% of initial)")
        st.plotly_chart(fig, use_container_width=True)

with cB:
    fig = go.Figure(go.Histogram2d(
        x=res.perf_T[:, 0] * 100, y=res.perf_T[:, 1] * 100,
        nbinsx=60, nbinsy=60, colorscale="Blues"))
    fig.add_vline(x=strike_pct * 100, line_dash="dot", line_color=RED)
    fig.add_hline(y=strike_pct * 100, line_dash="dot", line_color=RED)
    if bt == "worst" and ot == "put":
        region = "either stock below its line (L-shape)"
    elif bt == "worst" and ot == "call":
        region = "both stocks above their lines (top-right box)"
    elif bt == "best" and ot == "call":
        region = "either stock above its line (reverse L)"
    else:
        region = "both stocks below their lines (bottom-left box)"
    fig.update_layout(title=f"Joint final performance — exercised when {region}",
                      height=330, margin=dict(l=10, r=10, t=45, b=10),
                      xaxis_title=f"{tickers[0]} final (%)",
                      yaxis_title=f"{tickers[1]} final (%)")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div class='warn-box'><b>Model Limitations.</b> With option-implied
calibration the single-name smiles are market-consistent, but the <i>correlation</i>
input is still historical — implied correlation from listed products typically
trades above realized, and correlation spikes in sell-offs. Constant rho between
the stocks, European exercise, flat rates, no credit/funding adjustments, and
discrete dividends approximated by a continuous yield. Educational tool, not
investment advice.</div>""", unsafe_allow_html=True)
