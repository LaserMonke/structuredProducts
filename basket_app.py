"""
Two-stock basket structure pricer — SINGLE FILE (engine + UI, no local imports).

Worst-of / best-of · calls and puts · buy or sell · optional knock-in /
knock-out barriers (continuous or European) · optional autocall.
Heston Monte Carlo calibrated to option-implied vols; validated against
closed forms (Stulz 1982 for options on the min; min/max identity for the
max), in-out barrier parity, and autocall degeneracy checks.

Run locally:  streamlit run basket_app.py
Self-test:    python basket_app.py --selftest
"""

APP_VERSION = "3.2 after-hours quotes + widget-state fix"

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
# ENGINE — parameters
# =============================================================================

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
    basket_type: str = "worst"      # "worst" | "best"
    option_type: str = "put"        # "put"   | "call"
    strike_pct: float = 0.70
    tenor_years: float = 1.0
    r: float = 0.04
    notional: float = 1_000_000.0
    # ---- optional barrier on the option ----
    barrier_mode: str = "none"      # "none" | "ki" | "ko"
    barrier_level: float = 0.60     # fraction of initial fixing
    barrier_obs: str = "continuous" # "continuous" | "european"
    # ---- optional autocall (early termination of the structure) ----
    autocall: bool = False
    ac_freq_per_year: int = 4
    ac_trigger: float = 1.00        # structure ends when aggregate >= trigger
    ac_first_obs: int = 1


@dataclass
class Result:
    premium_pct: float
    stderr_pct: float
    premium_cash: float
    prob_itm: float
    prob_driver_is: np.ndarray
    exp_payoff_pct: float
    exp_payoff_given_itm: float
    max_payoff_pct: float
    var95_payoff_pct: float
    cvar95_payoff_pct: float
    breakeven_agg: float
    seller_pnl_pct: np.ndarray
    agg_T: np.ndarray
    perf_T: np.ndarray
    sample_paths: np.ndarray
    sample_exit_step: np.ndarray
    time_grid: np.ndarray
    deltas: np.ndarray
    vegas: np.ndarray
    corr_sens: float
    # barrier / autocall analytics
    prob_barrier_event: float       # P(knocked) for ki/ko, else 0
    prob_autocall: np.ndarray       # P(called at obs k)
    prob_alive_obs: np.ndarray      # P(alive at obs k, pre-call)
    obs_times: np.ndarray
    exp_life_years: float
    fair_coupon_pa: float           # coupon p.a. this premium would fund



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
    # Post-fit Feller guard: a calibrator can trade off a huge xi against theta
    # to shave listed-strike error while wrecking the deep-OTM wing that a 70%
    # worst-of put depends on. Cap xi to keep 2*kappa*theta/xi^2 >= 0.4.
    feller = 2.0 * ka * th / max(xi_**2, 1e-12)
    if feller < 0.4:
        xi_ = float(np.sqrt(2.0 * ka * th / 0.4))
    return {"v0": float(v0), "kappa": float(ka), "theta": float(th),
            "xi": float(xi_), "rho_sv": float(rho),
            "rmse_iv": float(np.sqrt(np.mean(sol.fun**2))),
            "n_quotes": n, "nfev": int(sol.nfev),
            "feller": float(2.0 * ka * th / max(xi_**2, 1e-12))}


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
    # kappa from AR(1); cap at 6 (10 just means we fit RV noise, not reversion)
    kappa = float(np.clip((x @ dv) / (den * dt) if den > 0 else 2.0, 0.5, 6.0))
    resid = dv - kappa * x * dt
    xi = float(np.clip(np.std(resid, ddof=1)
                       / (np.sqrt(rv[:-1].mean()) * np.sqrt(dt)), 0.10, 1.5))
    # Feller guard: keep 2*kappa*theta/xi^2 >= 0.5 so variance stays well-behaved
    # and deep-OTM tails are not artificially fattened by an oversized xi.
    feller = 2.0 * kappa * theta / max(xi**2, 1e-12)
    if feller < 0.5:
        xi = float(np.sqrt(2.0 * kappa * theta / 0.5))
    dr = lr[win - 1:][1:n]
    rho = -0.6
    if dr.size == dv.size and dv.size > 10:
        c = np.corrcoef(dr, dv)[0, 1]
        if np.isfinite(c):
            rho = float(np.clip(c, -0.95, 0.2))
    return {"v0": v0, "theta": theta, "kappa": kappa, "xi": xi, "rho_sv": rho}



# =============================================================================
# ENGINE — Monte Carlo with barriers and autocall
# =============================================================================

def _mc_run(a1: Asset, a2: Asset, rho_s: float, spec: OptionSpec,
            n_paths: int, steps_per_year: int, seed: int,
            spot_mult=(1.0, 1.0), n_saved: int = 0):
    """
    One Heston MC pass. Returns a dict of everything the pricer needs.

    spot_mult: post-inception spot bumps applied to each asset's performance
    at EVERY step (perf paths don't depend on the spot level under Heston, so
    a bumped-spot path is exactly mult x the base path). Barrier and autocall
    checks are done on the bumped performance, so Greeks see the features.
    """
    T = spec.tenor_years
    n_steps = max(int(round(T * steps_per_year)), 2)
    dt, sqdt = T / n_steps, np.sqrt(T / n_steps)
    if n_paths % 2:
        n_paths += 1
    half = n_paths // 2
    rng = np.random.default_rng(seed)

    rho_s = float(np.clip(rho_s, -0.999, 0.999))
    L = np.linalg.cholesky(np.array([[1.0, rho_s], [rho_s, 1.0]]))
    q = np.array([a1.div_yield, a2.div_yield])
    kap = np.array([a1.kappa, a2.kappa]); th = np.array([a1.theta, a2.theta])
    xi = np.array([a1.xi, a2.xi]); rho = np.array([a1.rho_sv, a2.rho_sv])
    rho_c = np.sqrt(np.clip(1 - rho**2, 0, 1))
    mult = np.asarray(spot_mult, float)

    worst = spec.basket_type == "worst"
    aggfun = (lambda p: p.min(axis=1)) if worst else (lambda p: p.max(axis=1))
    down = spec.option_type == "put"          # barrier direction follows payoff
    B = spec.barrier_level
    cont_barrier = (spec.barrier_mode != "none"
                    and spec.barrier_obs == "continuous")
    # Brownian bridge is exact for "any asset crosses" events:
    #   worst-of with a down barrier  (worst<=B  <=>  any perf<=B)
    #   best-of  with an up barrier   (best>=B   <=>  any perf>=B)
    bridge = cont_barrier and ((worst and down) or ((not worst) and not down))

    obs_steps = np.array([], dtype=int)
    if spec.autocall:
        n_obs = max(int(round(T * spec.ac_freq_per_year)), 1)
        obs_steps = np.unique(np.round(
            np.arange(1, n_obs + 1) * n_steps / n_obs).astype(int))
    obs_ptr = {int(s): k for k, s in enumerate(obs_steps)}
    obs_times = obs_steps * dt

    perf = np.tile(mult, (n_paths, 1))
    v = np.tile([a1.v0, a2.v0], (n_paths, 1))
    alive = np.ones(n_paths, dtype=bool)
    crossed = np.zeros(n_paths, dtype=bool)
    log_surv = np.zeros(n_paths)
    exit_step = np.full(n_paths, n_steps, dtype=int)
    prob_ac = np.zeros(len(obs_steps))
    prob_alive = np.zeros(len(obs_steps))

    n_save = min(n_saved, n_paths)
    saved = (np.tile(aggfun(perf[:n_save]), (n_steps + 1, 1)).T
             if n_save else None)
    tg = np.linspace(0, T, n_steps + 1)

    for t in range(1, n_steps + 1):
        Zs_h = rng.standard_normal((half, 2))
        Zw_h = rng.standard_normal((half, 2))
        Zs = np.vstack([Zs_h, -Zs_h]); Zw = np.vstack([Zw_h, -Zw_h])
        Zc = Zs @ L.T
        Zv = rho * Zc + rho_c * Zw
        vp = np.maximum(v, 0.0); sq = np.sqrt(vp)
        prev = perf
        perf = perf * np.exp((spec.r - q - 0.5 * vp) * dt + sq * sqdt * Zc)
        v = v + kap * (th - vp) * dt + xi * sq * sqdt * Zv
        agg = aggfun(perf)

        if cont_barrier:
            if bridge:
                hit = (perf <= B).any(axis=1) if down else (perf >= B).any(axis=1)
                crossed |= hit & alive
                ok = ~crossed & alive
                if ok.any():
                    var_step = np.maximum(vp, 1e-8) * dt
                    with np.errstate(divide="ignore", invalid="ignore"):
                        if down:
                            p = np.exp(-2.0 * np.log(prev / B)
                                       * np.log(perf / B) / var_step)
                            valid = (prev > B) & (perf > B)
                        else:
                            p = np.exp(-2.0 * np.log(B / prev)
                                       * np.log(B / perf) / var_step)
                            valid = (prev < B) & (perf < B)
                    p = np.where(valid, np.clip(p, 0.0, 1.0 - 1e-12), 0.0)
                    inc = np.log1p(-p).sum(axis=1)
                    log_surv = np.where(ok, log_surv + inc, log_surv)
            else:
                # "all assets" events: discrete daily monitoring of the aggregate
                hit = (agg <= B) if down else (agg >= B)
                crossed |= hit & alive

        if t in obs_ptr and t < n_steps:
            k = obs_ptr[t]
            prob_alive[k] = alive.mean()
            if (k + 1) >= spec.ac_first_obs:
                call = alive & (agg >= spec.ac_trigger)
                prob_ac[k] = call.mean()
                exit_step = np.where(call, t, exit_step)
                alive &= ~call
        elif t in obs_ptr:
            prob_alive[obs_ptr[t]] = alive.mean()

        if n_save:
            done = exit_step[:n_save] < t
            saved[:, t] = np.where(done, saved[:, t - 1], agg[:n_save])

    agg_T = aggfun(perf)
    if spec.barrier_mode == "none":
        surv = np.ones(n_paths)
    elif spec.barrier_obs == "european":
        hitT = (agg_T <= B) if down else (agg_T >= B)
        surv = (~hitT).astype(float)
    else:
        surv = np.where(crossed, 0.0,
                        np.exp(log_surv) if bridge else 1.0)

    return dict(perf_T=perf, agg_T=agg_T, alive=alive, surv=surv,
                prob_ac=prob_ac, prob_alive=prob_alive, obs_times=obs_times,
                exit_step=exit_step, n_steps=n_steps, saved=saved, tg=tg)


def _effective_payoff(sim: dict, spec: OptionSpec):
    """Per-path effective payoff at maturity (0 for autocalled paths)."""
    K = spec.strike_pct
    base = (np.maximum(K - sim["agg_T"], 0.0) if spec.option_type == "put"
            else np.maximum(sim["agg_T"] - K, 0.0))
    if spec.barrier_mode == "ki":
        base = base * (1.0 - sim["surv"])
    elif spec.barrier_mode == "ko":
        base = base * sim["surv"]
    return np.where(sim["alive"], base, 0.0)


def price_basket_option(a1: Asset, a2: Asset, rho_s: float, spec: OptionSpec,
                        n_paths: int = 50_000, steps_per_year: int = 252,
                        seed: Optional[int] = 42, n_saved: int = 150,
                        greeks: bool = True) -> Result:
    sim = _mc_run(a1, a2, rho_s, spec, n_paths, steps_per_year, seed,
                  n_saved=n_saved)
    n_paths = sim["perf_T"].shape[0]
    T, K = spec.tenor_years, spec.strike_pct
    df = np.exp(-spec.r * T)

    payoff = _effective_payoff(sim, spec)
    pv = df * payoff
    premium = float(pv.mean())
    stderr = float(pv.std(ddof=1) / np.sqrt(n_paths))

    itm = payoff > 1e-12
    p_itm = float(itm.mean())
    perf_T = sim["perf_T"]
    if itm.any():
        which = (perf_T[itm].argmin(axis=1) if spec.basket_type == "worst"
                 else perf_T[itm].argmax(axis=1))
        p_which = np.array([(which == 0).mean(), (which == 1).mean()])
        e_pay_itm = float(payoff[itm].mean())
    else:
        p_which = np.array([0.5, 0.5]); e_pay_itm = 0.0

    seller_pnl = premium / df - payoff
    tail = np.sort(payoff)[::-1]
    var95 = float(np.percentile(payoff, 95))
    cvar95 = float(tail[: max(int(0.05 * n_paths), 1)].mean())
    be = (K - premium / df) if spec.option_type == "put" else (K + premium / df)

    # ---- autocall analytics & the coupon this premium would fund ----
    exit_t = np.where(sim["exit_step"] < sim["n_steps"],
                      sim["exit_step"] / sim["n_steps"] * T, T)
    exp_life = float(exit_t.mean())
    if spec.autocall and len(sim["obs_times"]):
        annuity = float(np.sum(np.exp(-spec.r * sim["obs_times"])
                               * sim["prob_alive"]))
        fair_coupon = (premium / annuity * spec.ac_freq_per_year * 100.0
                       if annuity > 1e-9 else 0.0)
    else:
        f = 4
        ts = np.arange(1, int(round(T * f)) + 1) / f
        annuity = float(np.sum(np.exp(-spec.r * ts)))
        fair_coupon = premium / annuity * f * 100.0 if annuity > 0 else 0.0

    if spec.barrier_mode != "none":
        p_barrier = float(np.mean(1.0 - sim["surv"]))
    else:
        p_barrier = 0.0

    pct = lambda x: 100.0 * x
    deltas = np.zeros(2); vegas = np.zeros(2); corr_sens = 0.0
    if greeks:
        ng = min(n_paths, 30_000)

        def px(aa1, aa2, rs, sm=(1.0, 1.0)):
            s = _mc_run(aa1, aa2, rs, spec, ng, steps_per_year, seed,
                        spot_mult=sm)
            return pct(df * _effective_payoff(s, spec).mean())

        base = px(a1, a2, rho_s)
        for i, a in enumerate((a1, a2)):
            mu = [1.0, 1.0]; md = [1.0, 1.0]
            mu[i], md[i] = 1.01, 0.99
            deltas[i] = (px(a1, a2, rho_s, tuple(mu))
                         - px(a1, a2, rho_s, tuple(md))) / 2.0
            vol = np.sqrt(a.v0)
            vu = replace(a, v0=(vol + 0.01) ** 2,
                         theta=(np.sqrt(a.theta) + 0.01) ** 2)
            args = (vu, a2, rho_s) if i == 0 else (a1, vu, rho_s)
            vegas[i] = px(*args) - base
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
        agg_T=sim["agg_T"], perf_T=perf_T,
        sample_paths=sim["saved"], sample_exit_step=sim["exit_step"],
        time_grid=sim["tg"], deltas=deltas, vegas=vegas, corr_sens=corr_sens,
        prob_barrier_event=p_barrier, prob_autocall=sim["prob_ac"],
        prob_alive_obs=sim["prob_alive"], obs_times=sim["obs_times"],
        exp_life_years=exp_life, fair_coupon_pa=fair_coupon)


# =============================================================================
# Self-test:  python basket_app.py --selftest
# =============================================================================

def _selftest():
    s1, s2, rho, T, r = 0.45, 0.55, 0.55, 1.0, 0.04
    a1 = Asset("A", 250.0, 0.0, v0=s1**2, kappa=1.0, theta=s1**2, xi=1e-4, rho_sv=0.0)
    a2 = Asset("B", 130.0, 0.0, v0=s2**2, kappa=1.0, theta=s2**2, xi=1e-4, rho_sv=0.0)

    # 1) all four payoff types vs Black-Scholes closed forms (no features)
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

    # 2) in-out parity: KI + KO = vanilla (exact pathwise, same seed)
    van = OptionSpec("worst", "put", 0.70, T, r)
    ki = replace(van, barrier_mode="ki", barrier_level=0.60,
                 barrier_obs="continuous")
    ko = replace(ki, barrier_mode="ko")
    pv = price_basket_option(a1, a2, rho, van, 100_000, seed=7, greeks=False)
    pi = price_basket_option(a1, a2, rho, ki, 100_000, seed=7, greeks=False)
    po = price_basket_option(a1, a2, rho, ko, 100_000, seed=7, greeks=False)
    gap = abs(pi.premium_pct + po.premium_pct - pv.premium_pct)
    print(f"in-out parity: KI {pi.premium_pct:.3f} + KO {po.premium_pct:.3f} "
          f"= {pi.premium_pct+po.premium_pct:.3f} vs vanilla "
          f"{pv.premium_pct:.3f}  {'OK' if gap < 1e-6 else 'FAIL'}")
    assert gap < 1e-6

    # 3) unreachable autocall trigger reproduces the plain option exactly
    ac = replace(van, autocall=True, ac_trigger=50.0, ac_freq_per_year=4)
    pa = price_basket_option(a1, a2, rho, ac, 100_000, seed=7, greeks=False)
    gap = abs(pa.premium_pct - pv.premium_pct)
    print(f"autocall degeneracy: trigger=5000% -> {pa.premium_pct:.3f} vs "
          f"vanilla {pv.premium_pct:.3f}  {'OK' if gap < 1e-6 else 'FAIL'}")
    assert gap < 1e-6

    # 4) autocall must cheapen the option; KI must cheapen vs vanilla
    ac2 = replace(van, autocall=True, ac_trigger=1.00, ac_freq_per_year=4)
    pa2 = price_basket_option(a1, a2, rho, ac2, 100_000, seed=7, greeks=False)
    print(f"autocall@100% premium {pa2.premium_pct:.3f} < vanilla "
          f"{pv.premium_pct:.3f}: "
          f"{'OK' if pa2.premium_pct < pv.premium_pct else 'FAIL'}")
    assert pa2.premium_pct < pv.premium_pct and pi.premium_pct < pv.premium_pct

    # 5) clean flat-vol worst-of put matches closed form (guards don't distort)
    aM = Asset("MSFT", 500.0, 0.007, v0=0.25**2, kappa=1.5, theta=0.25**2,
               xi=1e-4, rho_sv=-0.5)
    aT = Asset("TSLA", 320.0, 0.0, v0=0.55**2, kappa=1.5, theta=0.55**2,
               xi=1e-4, rho_sv=-0.5)
    sp = OptionSpec("worst", "put", 0.70, 1.0, 0.04)
    rr = price_basket_option(aM, aT, 0.45, sp, n_paths=120_000, seed=1,
                             n_saved=0, greeks=False)
    an = bs_basket_option("worst", "put", 1.0, 1.0, 0.70, 1.0, 0.04,
                          0.007, 0.0, 0.25, 0.55, 0.45) * 100
    ok = abs(rr.premium_pct - an) < 0.25
    print(f"clean MSFT/TSLA worst-of put: MC={rr.premium_pct:.2f}% "
          f"closed-form={an:.2f}%  {'OK' if ok else 'FAIL'}")
    assert ok
    print(f"basket_app v{APP_VERSION}: all self-tests passed.")


if "--selftest" in sys.argv:
    _selftest()
    sys.exit(0)


# =============================================================================
# STREAMLIT UI
# =============================================================================

st.set_page_config(page_title="Basket Structure Pricer", page_icon="🧾",
                   layout="wide")

ACCENT, RED, GREEN, GRAY, AMBER = "#4b8bc4", "#d6564a", "#2aa574", "#8a8a85", "#cf9b2c"

# Theme-adaptive styling. Metric values are allowed to WRAP (no "..." ellipsis):
st.markdown("""
<style>
  .block-container {padding-top: 2.0rem; max-width: 1250px;}
  div[data-testid="stMetric"] {
    background: rgba(128,128,128,0.10);
    border: 1px solid rgba(128,128,128,0.28);
    border-radius: 10px; padding: 12px 14px;
  }
  div[data-testid="stMetric"] * {color: inherit;}
  div[data-testid="stMetricValue"] {font-size: 1.28rem; line-height: 1.2;}
  div[data-testid="stMetricValue"] > div {
    white-space: normal !important; overflow: visible !important;
    text-overflow: clip !important; overflow-wrap: anywhere;
  }
  div[data-testid="stMetricLabel"] {white-space: normal !important;}
  div[data-testid="stMetricLabel"] p {
    white-space: normal !important; overflow: visible !important;
    text-overflow: clip !important; font-size: 0.80rem;
  }
  div[data-testid="stMetricDelta"] > div {
    white-space: normal !important; overflow: visible !important;
    text-overflow: clip !important; font-size: 0.74rem;
  }
  .note-box {background: rgba(75,139,196,0.14); border-left: 4px solid #4b8bc4;
    padding: 12px 16px; border-radius: 0 8px 8px 0; font-size: .92rem;
    margin: 6px 0 14px 0; color: inherit;}
  .warn-box {background: rgba(214,86,74,0.14); border-left: 4px solid #d6564a;
    padding: 12px 16px; border-radius: 0 8px 8px 0; font-size: .92rem;
    margin: 6px 0 14px 0; color: inherit;}
</style>""", unsafe_allow_html=True)


def money(x: float) -> str:
    ax = abs(x)
    if ax >= 1e9:
        return f"${x/1e9:,.2f}B"
    if ax >= 1e6:
        return f"${x/1e6:,.2f}M"
    if ax >= 1e4:
        return f"${x/1e3:,.0f}k"
    return f"${x:,.0f}"


st.title("Two-stock basket structure pricer")
st.caption(f"Worst-of / best-of · calls & puts · optional barriers & autocall · "
           f"Heston MC calibrated to implied vols · v{APP_VERSION}")

# ----------------------------------------------------------------- data layer

def _robust_div_yield(tk, spot: float) -> float:
    """Continuous dividend yield, robust to yfinance unit changes.

    Primary: trailing-12M cash dividends / spot — unit-unambiguous.
    Fallback: parse info['dividendYield'], which has flipped between decimal
    (0.007) and percent (0.7) across yfinance versions; interpret defensively.
    Either way, clamp to [0, 12%] — no large-cap equity yields more, and an
    absurd q silently destroys every option price downstream.
    """
    try:
        d = tk.dividends
        if d is not None and len(d):
            d = d[d.index >= (d.index.max() - pd.Timedelta(days=365))]
            y = float(d.sum()) / spot
            if 0.0 <= y < 0.20:
                return min(y, 0.12)
    except Exception:
        pass
    try:
        y = float(tk.info.get("dividendYield", 0.0) or 0.0)
        if y > 1.0:
            y /= 100.0
        if y > 0.12:          # 0.7 almost certainly means 0.7%, not 70%
            y /= 100.0
        return float(min(max(y, 0.0), 0.12))
    except Exception:
        return 0.0


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
        divs[t] = _robust_div_yield(yf.Ticker(t), float(px[t].iloc[-1]))
    return px, lr, float(lr.corr().iloc[0, 1]), divs


def _rows_to_quotes(df, side, spot, T, r, q):
    """Convert option-chain rows to (T, K, iv) quotes, robust to market hours.

    Price source, in order of trust:
      1. bid/ask mid, when both sides are live and the spread is sane
      2. lastPrice, when there is evidence of trading (volume or OI)
         — outside US market hours Yahoo returns bid=ask=0 on EVERY contract,
           so a bid>0 requirement silently kills the whole chain
      3. Yahoo's impliedVolatility column, only if price inversion failed
    """
    quotes = []
    for _, row in df.iterrows():
        k = float(row["strike"])
        bid = float(row.get("bid", 0) or 0)
        ask = float(row.get("ask", 0) or 0)
        last = float(row.get("lastPrice", 0) or 0)
        traded = (float(row.get("volume", 0) or 0) > 0
                  or float(row.get("openInterest", 0) or 0) > 0)
        mid = None
        if bid > 0 and ask >= bid and (ask - bid) <= 0.6 * 0.5 * (bid + ask):
            mid = 0.5 * (bid + ask)
        elif last > 0 and traded:
            mid = last
        if mid is None:
            continue
        iv = implied_vol(mid, spot, k, T, r, q, side)
        if not (np.isfinite(iv) and 0.03 < iv < 2.0):
            yiv = float(row.get("impliedVolatility", np.nan) or np.nan)
            iv = yiv if (np.isfinite(yiv) and 0.05 < yiv < 1.8) else np.nan
        if np.isfinite(iv):
            quotes.append((float(T), k, float(iv)))
    return quotes


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_option_quotes(ticker: str, spot: float, tenor: float,
                        r: float, q: float):
    """OTM option quotes from two expiries bracketing the tenor."""
    import datetime as dtm
    import yfinance as yf
    tk = yf.Ticker(ticker)
    today = dtm.date.today()
    exps = [(e, (dtm.date.fromisoformat(e) - today).days / 365.25)
            for e in tk.options]
    exps = [x for x in exps if x[1] > 0.05]
    if not exps:
        raise ValueError("No listed expiries found.")
    chosen = []
    for target in (tenor, 0.6 * tenor):
        e = min(exps, key=lambda x: abs(x[1] - target))
        if e not in chosen:
            chosen.append(e)
    quotes, n_raw = [], 0
    for e, T in chosen:
        ch = tk.option_chain(e)
        for df, side in ((ch.puts, "put"), (ch.calls, "call")):
            df = df.copy()
            if side == "put":
                df = df[(df["strike"] >= 0.50 * spot) & (df["strike"] <= spot)]
            else:
                df = df[(df["strike"] > spot) & (df["strike"] <= 1.45 * spot)]
            df = df.sort_values("strike")
            n_raw += len(df)
            if len(df) > 8:
                df = df.iloc[np.linspace(0, len(df) - 1, 8).astype(int)]
            quotes += _rows_to_quotes(df, side, spot, T, r, q)
    if len(quotes) < 5:
        raise ValueError(
            f"only {len(quotes)} usable quotes from {n_raw} contracts — "
            f"if the market is closed, Yahoo often returns empty bid/ask "
            f"AND zero volume; try again during US market hours")
    return quotes


@st.cache_data(ttl=1800, show_spinner=False)
def cached_calibrate(ticker: str, spot: float, tenor: float, r: float,
                     q: float, x0: dict):
    quotes = fetch_option_quotes(ticker, spot, tenor, r, q)
    fit = calibrate_heston(quotes, spot, r, q, x0)
    return fit, quotes


# --------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Trade setup")

    with st.expander("Underlyings & data", expanded=True):
        t1 = st.text_input("Ticker 1", "NVDA").strip().upper()
        t2 = st.text_input("Ticker 2", "TSLA").strip().upper()
        lookback = st.selectbox("History window", ["1y", "2y", "3y", "5y"], 1)
        use_live = st.toggle("Live market data (yfinance)", True)
        calib_src = st.radio("Heston calibration",
                             ["Option-implied vols", "Historical returns"],
                             help="Implied fits v0, kappa, theta, xi, rho to "
                                  "the listed smile (market-consistent). Falls "
                                  "back to historical if no chain data.")

    with st.expander("Payoff", expanded=True):
        basket_type = st.radio("Basket", ["Worst-of", "Best-of"], horizontal=True)
        option_type = st.radio("Option", ["Put", "Call"], horizontal=True)
        position = st.radio("Your position", ["Sell", "Buy"], horizontal=True)
        strike_pct = st.slider("Strike (% of spot)", 40, 160, 70, 1) / 100
        tenor = st.slider("Tenor (years)", 0.25, 3.0, 1.0, 0.25)
        notional = st.number_input("Notional (USD)", 10_000, 1_000_000_000,
                                   1_000_000, step=10_000)

    with st.expander("Barrier (optional)"):
        b_mode_lbl = st.radio("Barrier type",
                              ["None", "Knock-in", "Knock-out"], index=0,
                              help="Knock-in: the option only pays if the "
                                   "barrier was crossed. Knock-out: it dies "
                                   "when crossed. Direction follows the "
                                   "payoff: down for puts, up for calls.")
        barrier_mode = {"None": "none", "Knock-in": "ki",
                        "Knock-out": "ko"}[b_mode_lbl]
        barrier_level = st.slider("Barrier level (% of initial)", 20, 200,
                                  60, 1,
                                  disabled=barrier_mode == "none") / 100
        barrier_obs_lbl = st.radio("Barrier observation",
                                   ["Continuous (American)",
                                    "At maturity (European)"],
                                   disabled=barrier_mode == "none")
        barrier_obs = ("continuous" if barrier_obs_lbl.startswith("Cont")
                       else "european")

    with st.expander("Autocall (optional)"):
        autocall = st.toggle("Autocallable structure", False,
                             help="At each observation date, if the basket "
                                  "aggregate closes at or above the trigger, "
                                  "the structure terminates early and the "
                                  "option is extinguished (worth 0 from then "
                                  "on) — this is what protects note "
                                  "investors' short puts.")
        ac_freq_lbl = st.selectbox("Observation frequency",
                                   ["Monthly", "Quarterly", "Semi-annual",
                                    "Annual"], 1, disabled=not autocall)
        ac_freq = {"Monthly": 12, "Quarterly": 4, "Semi-annual": 2,
                   "Annual": 1}[ac_freq_lbl]
        ac_trigger = st.slider("Autocall trigger (% of initial)", 70, 150,
                               100, 1, disabled=not autocall) / 100
        ac_first = st.number_input("First callable observation #", 1, 24, 1,
                                   disabled=not autocall)

    with st.expander("Market & simulation"):
        r = st.number_input("Risk-free rate (% p.a.)", 0.0, 15.0, 4.0,
                            0.1) / 100
        n_paths = st.select_slider("Monte Carlo paths",
                                   [10_000, 20_000, 50_000, 100_000, 200_000],
                                   50_000)
        seed = st.number_input("Random seed", 0, 10_000, 42)
        do_greeks = st.toggle("Compute Greeks (slower)", True)

tickers = (t1, t2)
if t1 == t2 or not t1 or not t2:
    st.error("Enter two different, non-empty tickers.")
    st.stop()

bt = "worst" if basket_type == "Worst-of" else "best"
ot = option_type.lower()
selling = position == "Sell"
sgn = 1.0 if selling else -1.0

feat = []
if barrier_mode != "none":
    feat.append(f"{b_mode_lbl.lower()} @ {barrier_level:.0%} "
                f"({'cont.' if barrier_obs=='continuous' else 'Euro.'})")
if autocall:
    feat.append(f"autocall {ac_freq_lbl.lower()} @ {ac_trigger:.0%}")
trade_name = f"{basket_type} {ot}"
trade_full = trade_name + (f" · {' · '.join(feat)}" if feat else "")

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
            spots[t] = st.number_input(f"Spot {t}", 0.01, 1e6, 100.0, key=f"s_{t}")
            vols[t] = st.number_input(f"Vol {t} (% p.a.)", 1.0, 300.0,
                                      45.0 if i == 0 else 55.0,
                                      key=f"v_{t}") / 100
            divs[t] = st.number_input(f"Div yield {t} (%)", 0.0, 20.0, 0.0,
                                      key=f"d_{t}") / 100
    rho_hist = st.slider("Spot correlation", -0.5, 0.99, 0.55, 0.01)
    hist, lr = None, None

# --------------------------------------------------- calibration per underlying
st.subheader("1 · Model calibration")

assets, smiles = [], {}
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
                          f"RMSE {est['rmse_iv']*100:.2f} vol pts · "
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
            est = {"v0": vv, "theta": vv, "kappa": 2.0, "xi": 0.8,
                   "rho_sv": -0.6}
            source = "manual flat vol"

    with cols[i]:
        st.markdown(f"**{t}** · spot **{spots[t]:,.2f}** · strike "
                    f"**{spots[t]*strike_pct:,.2f}** ({strike_pct:.0%})")
        st.caption(f"Calibration: {source}")
        st.caption(f"√v₀ {np.sqrt(est['v0']):.0%} · √θ "
                   f"{np.sqrt(est['theta']):.0%} · div yield "
                   f"{divs.get(t, 0.0):.2%}  ← sanity-check these")
        sig = f"{t}_{est['v0']:.4f}_{est['theta']:.4f}_{est['xi']:.2f}"
        with st.expander(f"Heston parameters — {t}", expanded=False):
            v0 = st.number_input("v₀", 0.0001, 4.0, float(round(est["v0"], 4)),
                                 format="%.4f", key=f"v0_{sig}")
            th = st.number_input("θ", 0.0001, 4.0,
                                 float(round(est["theta"], 4)),
                                 format="%.4f", key=f"th_{sig}")
            ka = st.number_input("κ", 0.05, 15.0,
                                 float(round(est["kappa"], 2)), key=f"ka_{sig}")
            xv = st.number_input("ξ (vol of vol)", 0.01, 3.0,
                                 float(round(est["xi"], 2)), key=f"xi_{sig}")
            rh = st.number_input("ρ spot-vol", -0.99, 0.5,
                                 float(round(est["rho_sv"], 2)), key=f"rh_{sig}")
        st.caption(f"√v₀ = {np.sqrt(v0):.1%} · √θ = {np.sqrt(th):.1%}")
        assets.append(Asset(t, spots[t], divs.get(t, 0.0), v0, ka, th, xv, rh))

rho_s = st.slider("Spot correlation between the two stocks", -0.5, 0.99,
                  float(round(np.clip(rho_hist, -0.5, 0.99), 2)), 0.01)

if smiles:
    cols = st.columns(len(smiles))
    for i, (t, quotes) in enumerate(smiles.items()):
        a = assets[tickers.index(t)]
        Ts = sorted({round(qq[0], 3) for qq in quotes})
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
                          annotation_text="strike")
            fig.update_layout(title=f"{t} smile (T≈{Tn:.2f}y)", height=270,
                              margin=dict(l=10, r=10, t=45, b=10),
                              xaxis_title="Strike (% of spot)",
                              yaxis_title="Implied vol (%)",
                              legend=dict(x=0.55, y=0.98))
            st.plotly_chart(fig, use_container_width=True)

if data_ok and hist is not None:
    with st.expander("Price history", expanded=False):
        norm_px = hist / hist.iloc[0] * 100
        fig = go.Figure()
        for t in tickers:
            fig.add_trace(go.Scatter(x=norm_px.index, y=norm_px[t], name=t,
                                     line=dict(width=1.6)))
        fig.update_layout(title=f"Normalised ({lookback}) · realized corr "
                                f"= {rho_hist:.2f}", height=260,
                          margin=dict(l=10, r=10, t=45, b=10),
                          yaxis_title="Rebased to 100")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------- price
spec = OptionSpec(bt, ot, strike_pct, tenor, r, notional,
                  barrier_mode, barrier_level, barrier_obs,
                  autocall, ac_freq, ac_trigger, int(ac_first))

st.subheader(f"2 · {position} the {trade_full}")
if not st.button(f"▶ Price it", type="primary", use_container_width=True):
    st.info("Adjust terms in the sidebar, then price.")
    st.stop()

with st.spinner(f"Simulating {n_paths:,} Heston paths…"):
    res = price_basket_option(assets[0], assets[1], rho_s, spec,
                              n_paths=int(n_paths), seed=int(seed),
                              greeks=do_greeks)

driver = "worst performer" if bt == "worst" else "best performer"
prem_word = "receive" if selling else "pay"

metrics = [
    (f"Premium you {prem_word}", f"{res.premium_pct:.2f}%",
     f"± {2*res.stderr_pct:.2f}%"),
    ("Premium in cash", money(res.premium_cash), f"on {money(notional)}"),
    ("P(exercise)", f"{res.prob_itm:.1%}", None),
    ("Breakeven level", f"{res.breakeven_agg:.0%}", f"final {driver}"),
    ("E[payout] if exercised", f"{res.exp_payoff_given_itm:.1f}%",
     "of notional"),
    ("Tail payout (CVaR 95)", f"{res.cvar95_payoff_pct:.1f}%",
     "worst 5% avg"),
    (f"P({tickers[0]} drives | ITM)", f"{res.prob_driver_is[0]:.0%}", None),
    ("Annualized premium", f"{res.premium_pct/tenor:.2f}%", "p.a."),
]
if barrier_mode != "none":
    metrics.append((f"P({b_mode_lbl.lower()} event)",
                    f"{res.prob_barrier_event:.1%}",
                    f"barrier {barrier_level:.0%}"))
if autocall:
    metrics += [
        ("P(autocalled early)", f"{res.prob_autocall.sum():.1%}", None),
        ("Expected life", f"{res.exp_life_years:.2f}y",
         f"of {tenor:.2f}y max"),
        ("Coupon this funds", f"{res.fair_coupon_pa:.2f}%",
         "p.a. while alive"),
    ]
else:
    bs_ref = bs_basket_option(bt, ot, 1.0, 1.0, strike_pct, tenor, r,
                              assets[0].div_yield, assets[1].div_yield,
                              float(np.sqrt(assets[0].theta)),
                              float(np.sqrt(assets[1].theta)), rho_s) * 100
    if barrier_mode == "none":
        metrics.append(("Heston vs flat-vol", f"{res.premium_pct-bs_ref:+.2f}%",
                        f"BS ref {bs_ref:.2f}%"))

for row_start in range(0, len(metrics), 4):
    row = metrics[row_start:row_start + 4]
    cols = st.columns(4)
    for c, (lbl, val, dl) in zip(cols, row):
        c.metric(lbl, val, dl, delta_color="off")

if selling:
    msg = (f"You collect <b>{money(res.premium_cash)}</b> today. The option is "
           f"exercised in {res.prob_itm:.0%} of paths, costing "
           f"{res.exp_payoff_given_itm:.1f}% of notional on average when it "
           f"happens.")
    if autocall:
        msg += (f" With autocall, the structure ends early in "
                f"{res.prob_autocall.sum():.0%} of paths (expected life "
                f"{res.exp_life_years:.2f}y) — this premium would fund a "
                f"coupon of about <b>{res.fair_coupon_pa:.2f}% p.a.</b> on an "
                f"equivalent autocallable note.")
else:
    msg = (f"You pay <b>{money(res.premium_cash)}</b> today; the option pays "
           f"off in {res.prob_itm:.0%} of paths, averaging "
           f"{res.exp_payoff_given_itm:.1f}% of notional when it does.")
st.markdown(f"<div class='note-box'>{msg}</div>", unsafe_allow_html=True)

if do_greeks:
    with st.expander("Greeks (per 100 notional)", expanded=True):
        your = -sgn
        gdf = pd.DataFrame({
            "Sensitivity": [f"Delta {tickers[0]} (+1% spot)",
                            f"Delta {tickers[1]} (+1% spot)",
                            f"Vega {tickers[0]} (+1 vol pt)",
                            f"Vega {tickers[1]} (+1 vol pt)",
                            "Correlation (+0.05)"],
            "Option value": [f"{res.deltas[0]:+.3f}", f"{res.deltas[1]:+.3f}",
                             f"{res.vegas[0]:+.3f}", f"{res.vegas[1]:+.3f}",
                             f"{res.corr_sens:+.3f}"],
            "Your P&L": [f"{your*res.deltas[0]:+.3f}",
                         f"{your*res.deltas[1]:+.3f}",
                         f"{your*res.vegas[0]:+.3f}",
                         f"{your*res.vegas[1]:+.3f}",
                         f"{your*res.corr_sens:+.3f}"]})
        st.dataframe(gdf, hide_index=True, use_container_width=True)

# --------------------------------------------------------------------- charts
your_pnl = sgn * res.seller_pnl_pct
cA, cB = st.columns(2)
with cA:
    fig = go.Figure(go.Histogram(x=your_pnl, nbinsx=90,
                                 marker_color=ACCENT, opacity=0.85))
    fig.add_vline(x=0, line_dash="dash", line_color=GRAY)
    fig.add_vline(x=float(np.mean(your_pnl)), line_color=GREEN,
                  annotation_text=f"mean {np.mean(your_pnl):+.1f}%")
    fig.update_layout(title=f"Your P&L at maturity ({position.lower()}er, % "
                            f"of notional)", height=320,
                      margin=dict(l=10, r=10, t=45, b=10),
                      xaxis_title="P&L %", yaxis_title="Paths")
    st.plotly_chart(fig, use_container_width=True)

with cB:
    x = np.linspace(0.05, 1.9, 300)
    prem_T = res.premium_pct / np.exp(-r * tenor) / 100
    opt_pay = (np.maximum(strike_pct - x, 0) if ot == "put"
               else np.maximum(x - strike_pct, 0))
    fig = go.Figure()
    if barrier_mode == "ki":
        fig.add_trace(go.Scatter(x=x*100, y=sgn*prem_T*100*np.ones_like(x),
                                 name="barrier never hit",
                                 line=dict(color=GREEN, width=2)))
        fig.add_trace(go.Scatter(x=x*100, y=sgn*(prem_T-opt_pay)*100,
                                 name="knocked in",
                                 line=dict(color=RED, width=2, dash="dash")))
    elif barrier_mode == "ko":
        fig.add_trace(go.Scatter(x=x*100, y=sgn*(prem_T-opt_pay)*100,
                                 name="never knocked out",
                                 line=dict(color=ACCENT, width=2)))
        fig.add_trace(go.Scatter(x=x*100, y=sgn*prem_T*100*np.ones_like(x),
                                 name="knocked out",
                                 line=dict(color=GREEN, width=2, dash="dash")))
    else:
        fig.add_trace(go.Scatter(x=x*100, y=sgn*(prem_T-opt_pay)*100,
                                 showlegend=False,
                                 line=dict(color=ACCENT, width=2.5)))
    fig.add_hline(y=0, line_color=GRAY, line_width=1)
    fig.add_vline(x=strike_pct*100, line_dash="dot", line_color=AMBER,
                  annotation_text="strike")
    if barrier_mode != "none":
        fig.add_vline(x=barrier_level*100, line_dash="dot", line_color=RED,
                      annotation_text="barrier")
    fig.update_layout(title=f"Payoff at maturity vs final {driver}",
                      height=320, margin=dict(l=10, r=10, t=45, b=10),
                      xaxis_title=f"Final {driver} (% of initial)",
                      yaxis_title="Your P&L (% of notional)",
                      legend=dict(x=0.02, y=0.98))
    st.plotly_chart(fig, use_container_width=True)

cA, cB = st.columns(2)
with cA:
    if res.sample_paths is not None:
        n_show = min(120, res.sample_paths.shape[0])
        n_steps = res.sample_paths.shape[1] - 1
        fig = go.Figure()
        for i in range(n_show):
            called = res.sample_exit_step[i] < n_steps
            end = res.sample_paths[i, -1]
            itm = ((end < strike_pct) if ot == "put"
                   else (end > strike_pct)) and not called
            color = (GREEN if called else
                     RED if itm else "rgba(75,139,196,0.28)")
            fig.add_trace(go.Scatter(x=res.time_grid,
                                     y=res.sample_paths[i]*100, mode="lines",
                                     line=dict(width=0.8, color=color),
                                     showlegend=False, hoverinfo="skip"))
        fig.add_hline(y=strike_pct*100, line_dash="dot", line_color=AMBER,
                      annotation_text="strike", annotation_position="right")
        if barrier_mode != "none":
            fig.add_hline(y=barrier_level*100, line_dash="dot",
                          line_color=RED, annotation_text="barrier",
                          annotation_position="right")
        if autocall:
            fig.add_hline(y=ac_trigger*100, line_dash="dot", line_color=GREEN,
                          annotation_text="autocall",
                          annotation_position="right")
        fig.update_layout(title=f"Sample {driver} paths (red = exercised"
                                + (", green = autocalled)" if autocall else ")"),
                          height=320, margin=dict(l=10, r=10, t=45, b=10),
                          xaxis_title="Years",
                          yaxis_title=f"{basket_type} level (% of initial)")
        st.plotly_chart(fig, use_container_width=True)

with cB:
    if autocall and len(res.obs_times):
        labels = [f"{t:.2f}y" for t in res.obs_times]
        surv = 1.0 - res.prob_autocall.sum()
        fig = go.Figure(go.Bar(x=labels + ["Maturity"],
                               y=list(res.prob_autocall*100) + [surv*100],
                               marker_color=[GREEN]*len(labels) + [GRAY]))
        fig.update_layout(title="When does the structure end? (%)",
                          height=320, margin=dict(l=10, r=10, t=45, b=10),
                          yaxis_title="% of paths")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = go.Figure(go.Histogram2d(x=res.perf_T[:, 0]*100,
                                       y=res.perf_T[:, 1]*100,
                                       nbinsx=60, nbinsy=60,
                                       colorscale="Blues"))
        fig.add_vline(x=strike_pct*100, line_dash="dot", line_color=RED)
        fig.add_hline(y=strike_pct*100, line_dash="dot", line_color=RED)
        fig.update_layout(title="Joint final performance",
                          height=320, margin=dict(l=10, r=10, t=45, b=10),
                          xaxis_title=f"{tickers[0]} final (%)",
                          yaxis_title=f"{tickers[1]} final (%)")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div class='warn-box'><b>Remaining model limitations.</b> Single-name smiles are
market-consistent under implied calibration, but the correlation input is
historical — implied correlation typically trades above realized and spikes in
sell-offs. Continuous barriers use a Brownian-bridge correction where the event
is "either stock crosses" (worst-of down, best-of up); "both stocks" events use
daily monitoring. European exercise, flat rates, no credit/funding adjustments.
Educational tool, not investment advice.</div>""", unsafe_allow_html=True)
