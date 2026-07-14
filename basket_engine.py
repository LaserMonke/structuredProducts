"""
Basket option engine: worst-of / best-of, call / put on two stocks.

Payoff at maturity T (performance convention, per 1.0 notional):
    agg = min(perf1, perf2)  ("worst")   or   max(perf1, perf2)  ("best")
    put : max(K_pct - agg, 0)        call: max(agg - K_pct, 0)

Dynamics per asset (risk-neutral) — Heston:
    dS_i = (r - q_i) S_i dt + sqrt(v_i) S_i dW_i^S
    dv_i = kappa_i (theta_i - v_i) dt + xi_i sqrt(v_i) dW_i^v
with d<W_1^S, W_2^S> = rho_S dt and d<W_i^S, W_i^v> = rho_i dt.
Full-truncation Euler, daily steps, antithetic variates.

Calibration: Heston parameters can be fit per stock to option-implied vols
(least squares in vega-normalized price space) — see calibrate_heston().

Validation (`python basket_engine.py`): all four payoff types are checked
against their Black-Scholes closed forms (Stulz 1982 for options on the min,
plus the min/max identity for options on the max) with degenerate Heston,
and the calibrator is checked to recover a known smile.
"""

from __future__ import annotations

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

def heston_vanilla(S0, K, T, r, q, v0, kappa, theta, xi, rho,
                   cp="call", n_quad=192, u_max=200.0) -> float:
    x, w = np.polynomial.legendre.leggauss(n_quad)
    u = 0.5 * u_max * (x + 1.0)
    wq = 0.5 * u_max * w

    def cf(phi, j):
        a = kappa * theta
        if j == 1:
            uu, b = 0.5, kappa - rho * xi
        else:
            uu, b = -0.5, kappa
        d = np.sqrt((rho * xi * 1j * phi - b) ** 2
                    - xi**2 * (2 * uu * 1j * phi - phi**2))
        g = (b - rho * xi * 1j * phi + d) / (b - rho * xi * 1j * phi - d)
        ed = np.exp(d * T)
        C = (r - q) * 1j * phi * T + a / xi**2 * (
            (b - rho * xi * 1j * phi + d) * T
            - 2.0 * np.log((1 - g * ed) / (1 - g)))
        D = (b - rho * xi * 1j * phi + d) / xi**2 * (1 - ed) / (1 - g * ed)
        return np.exp(C + D * v0 + 1j * phi * np.log(S0))

    P = []
    for j in (1, 2):
        integ = np.real(np.exp(-1j * u * np.log(K)) * cf(u, j) / (1j * u))
        P.append(0.5 + (wq * integ).sum() / np.pi)
    call = S0 * np.exp(-q * T) * P[0] - K * np.exp(-r * T) * P[1]
    if cp == "call":
        return float(call)
    return float(call - S0 * np.exp(-q * T) + K * np.exp(-r * T))


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
                     x0: Optional[dict] = None) -> dict:
    """
    Fit (v0, kappa, theta, xi, rho) to option-implied vols.

    quotes: list of (T, K, market_iv). Residuals are price errors normalized
    by Black-Scholes vega, which approximates implied-vol errors and is far
    better conditioned than raw price errors across strikes.
    """
    quotes = [(t, k, iv) for (t, k, iv) in quotes
              if np.isfinite(iv) and 0.01 < iv < 4.0 and t > 1e-3]
    if len(quotes) < 5:
        raise ValueError("Need at least 5 valid option quotes to calibrate.")

    mkt_px, vegas, cps = [], [], []
    for (t, k, iv) in quotes:
        cp = "call" if k >= S0 else "put"          # OTM side
        cps.append(cp)
        mkt_px.append(bs_vanilla(S0, k, t, r, q, iv, cp))
        vegas.append(max(bs_vega(S0, k, t, r, q, iv), 1e-6))
    mkt_px, vegas = np.array(mkt_px), np.array(vegas)

    x0 = x0 or {}
    p0 = np.array([x0.get("v0", 0.09), x0.get("kappa", 2.0),
                   x0.get("theta", 0.09), x0.get("xi", 0.8),
                   x0.get("rho_sv", -0.6)])
    lb = np.array([1e-4, 0.05, 1e-4, 0.05, -0.99])
    ub = np.array([4.0, 15.0, 4.0, 3.0, 0.50])
    p0 = np.clip(p0, lb + 1e-6, ub - 1e-6)

    def resid(p):
        v0, ka, th, xi, rho = p
        out = np.empty(len(quotes))
        for i, (t, k, _) in enumerate(quotes):
            try:
                mp = heston_vanilla(S0, k, t, r, q, v0, ka, th, xi, rho, cps[i])
            except Exception:
                mp = np.nan
            out[i] = (mp - mkt_px[i]) / vegas[i] if np.isfinite(mp) else 10.0
        return out

    sol = least_squares(resid, p0, bounds=(lb, ub), xtol=1e-8, ftol=1e-8,
                        max_nfev=300)
    v0, ka, th, xi, rho = sol.x
    rmse_iv = float(np.sqrt(np.mean(sol.fun**2)))     # approx in vol units
    return {"v0": float(v0), "kappa": float(ka), "theta": float(th),
            "xi": float(xi), "rho_sv": float(rho), "rmse_iv": rmse_iv,
            "n_quotes": len(quotes)}


def model_smile(S0, r, q, T, strikes, v0, kappa, theta, xi, rho) -> np.ndarray:
    """Heston implied-vol smile at tenor T for a fit-quality chart."""
    out = []
    for k in strikes:
        cp = "call" if k >= S0 else "put"
        px = heston_vanilla(S0, k, T, r, q, v0, kappa, theta, xi, rho, cp)
        out.append(implied_vol(px, S0, k, T, r, q, cp))
    return np.array(out)


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

if __name__ == "__main__":
    s1, s2, rho = 0.45, 0.55, 0.55
    T, r = 1.0, 0.04

    a1 = Asset("A", 250.0, 0.0, v0=s1**2, kappa=1.0, theta=s1**2, xi=1e-4, rho_sv=0.0)
    a2 = Asset("B", 130.0, 0.0, v0=s2**2, kappa=1.0, theta=s2**2, xi=1e-4, rho_sv=0.0)

    combos = [("worst", "put", 0.70), ("worst", "call", 1.00),
              ("best", "call", 1.30), ("best", "put", 1.00)]
    for bt, ot, K in combos:
        spec = OptionSpec(bt, ot, K, T, r)
        ana = bs_basket_option(bt, ot, 1.0, 1.0, K, T, r, 0, 0, s1, s2, rho)
        res = price_basket_option(a1, a2, rho, spec, n_paths=400_000, seed=11,
                                  n_saved=0, greeks=False)
        mc = res.premium_pct / 100
        print(f"[validation] {bt}-of {ot} K={K:.0%}: analytic={ana:.5f}  "
              f"MC={mc:.5f} (+/-{res.stderr_pct/100:.5f})")
        assert abs(ana - mc) < 4 * res.stderr_pct / 100 + 3e-4, f"{bt}/{ot} mismatch"

    # Calibration recovery: synthesize a smile from known Heston params
    true = dict(v0=0.20, kappa=2.5, theta=0.25, xi=0.9, rho=-0.65)
    S0, q = 100.0, 0.0
    quotes = []
    for t in (0.5, 1.0):
        for k in (60, 70, 80, 90, 100, 110, 120, 130):
            cp = "call" if k >= S0 else "put"
            px = heston_vanilla(S0, k, t, r, q, true["v0"], true["kappa"],
                                true["theta"], true["xi"], true["rho"], cp)
            iv = implied_vol(px, S0, k, t, r, q, cp)
            quotes.append((t, float(k), iv))
    fit = calibrate_heston(quotes, S0, r, q,
                           x0=dict(v0=0.10, kappa=1.0, theta=0.10,
                                   xi=0.5, rho_sv=-0.3))
    print(f"[validation] calibration RMSE (vol pts) = {fit['rmse_iv']*100:.3f} "
          f"on {fit['n_quotes']} quotes  ->  v0={fit['v0']:.3f} th={fit['theta']:.3f} "
          f"kap={fit['kappa']:.2f} xi={fit['xi']:.2f} rho={fit['rho_sv']:.2f}")
    assert fit["rmse_iv"] < 0.005, "calibration failed to recover the smile"

    print("All basket-engine self-tests passed.")
