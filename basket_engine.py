"""
Worst-of put on two stocks — Heston Monte Carlo engine.

Payoff at maturity T (performance convention, per 1.0 notional):
    payoff = max(K_pct - min(S1_T/S1_0, S2_T/S2_0), 0)

The client SELLS this option: they receive the premium today and pay the
payoff (if any) at maturity.

Dynamics per asset (risk-neutral):
    dS_i = (r - q_i) S_i dt + sqrt(v_i) S_i dW_i^S
    dv_i = kappa_i (theta_i - v_i) dt + xi_i sqrt(v_i) dW_i^v
with d<W_1^S, W_2^S> = rho_S dt and d<W_i^S, W_i^v> = rho_i dt.
Full-truncation Euler, daily steps, antithetic variates.

Validation: with xi -> 0 and v0 = theta = sigma^2 the model degenerates to
Black-Scholes, where the put on the minimum of two assets has the closed form
of Stulz (1982). run `python basket_engine.py` to check MC vs analytic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
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
class PutSpec:
    strike_pct: float = 0.70      # strike as fraction of each initial spot
    tenor_years: float = 1.0
    r: float = 0.04
    notional: float = 1_000_000.0


@dataclass
class Result:
    premium_pct: float            # fair value, % of notional
    stderr_pct: float
    premium_cash: float
    prob_itm: float               # P(worst_T < strike)
    prob_worst_is: np.ndarray     # P(asset i is the worst | ITM), len 2
    exp_payoff_pct: float         # undiscounted E[payoff] %
    exp_payoff_given_itm: float   # % of notional
    max_loss_pct: float           # seller's worst simulated payoff %
    var95_payoff_pct: float       # 95th pct of payoff (seller tail)
    cvar95_payoff_pct: float
    breakeven_worst: float        # worst-of level where seller P&L = 0
    seller_pnl_pct: np.ndarray    # per-path seller P&L, % of notional
    worst_T: np.ndarray
    perf_T: np.ndarray            # (n_paths, 2) final performances
    sample_paths: np.ndarray      # (n_saved, n_steps+1) worst-of trajectories
    time_grid: np.ndarray
    deltas: np.ndarray            # dPrice%/d(1% spot move), per asset (seller sign: short)
    vegas: np.ndarray             # dPrice%/d(1 vol pt), per asset
    corr_sens: float              # dPrice%/d(+0.05 correlation)


# -----------------------------------------------------------------------------
# Stulz (1982) closed form: options on the minimum of two assets (BS world)
# -----------------------------------------------------------------------------

def _bvn(a, b, rho):
    return float(multivariate_normal(mean=[0, 0],
                                     cov=[[1, rho], [rho, 1]]).cdf([a, b]))


def stulz_call_on_min(S1, S2, K, T, r, q1, q2, s1, s2, rho) -> float:
    """European call on min(S1, S2) with common strike K (Stulz 1982)."""
    if K <= 0:
        # C_min(0) = PV of min(S1,S2) = S2 e^{-q2 T} - Margrabe(S2 -> S1)
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
    c = (S1 * np.exp(-q1 * T) * _bvn(d1, d3, r1)
         + S2 * np.exp(-q2 * T) * _bvn(d2, d4, r2)
         - K * np.exp(-r * T) * _bvn(d1 - s1 * sq, d2 - s2 * sq, rho))
    return float(c)


def stulz_put_on_min(S1, S2, K, T, r, q1, q2, s1, s2, rho) -> float:
    """Put on min via parity: P = K e^{-rT} - C_min(0) + C_min(K)."""
    c0 = stulz_call_on_min(S1, S2, 0.0, T, r, q1, q2, s1, s2, rho)
    ck = stulz_call_on_min(S1, S2, K, T, r, q1, q2, s1, s2, rho)
    return float(K * np.exp(-r * T) - c0 + ck)


# -----------------------------------------------------------------------------
# Heston parameter estimation from history (same heuristic as before)
# -----------------------------------------------------------------------------

def estimate_heston_from_returns(log_returns: np.ndarray, trading_days: int = 252) -> dict:
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
    xi = float(np.clip(np.std(resid, ddof=1) / (np.sqrt(rv[:-1].mean()) * np.sqrt(dt)),
                       0.10, 2.5))
    dr = lr[win - 1:][1:n]
    rho = -0.6
    if dr.size == dv.size and dv.size > 10:
        c = np.corrcoef(dr, dv)[0, 1]
        if np.isfinite(c):
            rho = float(np.clip(c, -0.95, 0.2))
    return {"v0": v0, "theta": theta, "kappa": kappa, "xi": xi, "rho_sv": rho,
            "hist_vol": float(np.sqrt(theta))}


# -----------------------------------------------------------------------------
# Monte Carlo core
# -----------------------------------------------------------------------------

def _simulate_terminal(a1: Asset, a2: Asset, rho_s: float, spec: PutSpec,
                       n_paths: int, steps_per_year: int, seed: int,
                       n_saved: int = 0):
    """Return final performances (n_paths, 2) and optional saved worst-of paths."""
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
            saved[:, t] = (S[:n_saved] / S0).min(axis=1)

    return S / S0, saved, tg


def price_worst_of_put(a1: Asset, a2: Asset, rho_s: float, spec: PutSpec,
                       n_paths: int = 50_000, steps_per_year: int = 252,
                       seed: Optional[int] = 42, n_saved: int = 200,
                       greeks: bool = True) -> Result:
    perf, saved, tg = _simulate_terminal(a1, a2, rho_s, spec, n_paths,
                                         steps_per_year, seed, n_saved)
    n_paths = perf.shape[0]
    worst = perf.min(axis=1)
    K = spec.strike_pct
    df = np.exp(-spec.r * spec.tenor_years)

    payoff = np.maximum(K - worst, 0.0)                    # per 1.0 notional
    pv = df * payoff
    premium = float(pv.mean())
    stderr = float(pv.std(ddof=1) / np.sqrt(n_paths))

    itm = payoff > 0
    p_itm = float(itm.mean())
    if itm.any():
        which = perf[itm].argmin(axis=1)
        p_which = np.array([(which == 0).mean(), (which == 1).mean()])
        e_pay_itm = float(payoff[itm].mean())
    else:
        p_which = np.array([0.5, 0.5]); e_pay_itm = 0.0

    seller_pnl = premium / df - payoff                     # both at maturity, per 1.0
    # seller receives premium today; compare like-for-like at T (premium accrued at r)

    tail = np.sort(payoff)[::-1]
    var95 = float(np.percentile(payoff, 95))
    cvar95 = float(tail[: max(int(0.05 * n_paths), 1)].mean())

    def pct(x): return 100.0 * x

    # ---- Greeks by bump-and-revalue with common random numbers ----
    # Delta: a post-inception spot move with the strike FIXED at its inception
    # level. In performance terms a +1% spot bump multiplies that asset's
    # performance (vs the inception fixing) by 1.01, so we bump the simulated
    # performance directly — Heston perf dynamics don't depend on spot level.
    deltas = np.zeros(2); vegas = np.zeros(2); corr_sens = 0.0
    if greeks:
        ng = min(n_paths, 30_000)

        def px(aa1, aa2, rs, spot_mult=(1.0, 1.0)):
            p, _, _ = _simulate_terminal(aa1, aa2, rs, spec, ng,
                                         steps_per_year, seed, 0)
            p = p * np.asarray(spot_mult)
            return 100.0 * df * np.maximum(K - p.min(axis=1), 0.0).mean()

        base = px(a1, a2, rho_s)
        from dataclasses import replace
        for i, a in enumerate((a1, a2)):
            mu = [1.0, 1.0]; md = [1.0, 1.0]
            mu[i], md[i] = 1.01, 0.99
            deltas[i] = (px(a1, a2, rho_s, tuple(mu))
                         - px(a1, a2, rho_s, tuple(md))) / 2.0   # per +1% spot
            vol = np.sqrt(a.v0)
            vu = replace(a, v0=(vol + 0.01) ** 2, theta=(np.sqrt(a.theta) + 0.01) ** 2)
            args_v = (vu, a2, rho_s) if i == 0 else (a1, vu, rho_s)
            vegas[i] = px(*args_v) - base                    # per +1 vol point
        corr_sens = px(a1, a2, min(rho_s + 0.05, 0.999)) - base

    return Result(
        premium_pct=pct(premium), stderr_pct=pct(stderr),
        premium_cash=premium * spec.notional,
        prob_itm=p_itm, prob_worst_is=p_which,
        exp_payoff_pct=pct(payoff.mean()),
        exp_payoff_given_itm=pct(e_pay_itm),
        max_loss_pct=pct(payoff.max()),
        var95_payoff_pct=pct(var95), cvar95_payoff_pct=pct(cvar95),
        breakeven_worst=K - premium / df,
        seller_pnl_pct=pct(seller_pnl),
        worst_T=worst, perf_T=perf,
        sample_paths=saved, time_grid=tg,
        deltas=deltas, vegas=vegas, corr_sens=corr_sens,
    )


# -----------------------------------------------------------------------------
# Self-tests:  python basket_engine.py
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Degenerate Heston (xi -> 0, v0 = theta = sigma^2)  ==  Black-Scholes,
    # so the MC must reproduce Stulz's closed-form put on the minimum.
    s1, s2, rho = 0.45, 0.55, 0.55
    S1, S2, K_pct, T, r = 250.0, 130.0, 0.70, 1.0, 0.04
    q1, q2 = 0.0, 0.0

    a1 = Asset("NVDA", S1, q1, v0=s1**2, kappa=1.0, theta=s1**2, xi=1e-4, rho_sv=0.0)
    a2 = Asset("TSLA", S2, q2, v0=s2**2, kappa=1.0, theta=s2**2, xi=1e-4, rho_sv=0.0)
    spec = PutSpec(strike_pct=K_pct, tenor_years=T, r=r)

    # Analytic price of put on min of *performances*: use unit spots, strike K_pct
    ana = stulz_put_on_min(1.0, 1.0, K_pct, T, r, q1, q2, s1, s2, rho)
    res = price_worst_of_put(a1, a2, rho, spec, n_paths=400_000, seed=11,
                             n_saved=0, greeks=False)
    mc = res.premium_pct / 100
    print(f"[validation] Stulz put-on-min analytic={ana:.5f}  "
          f"MC={mc:.5f} (+/-{res.stderr_pct/100:.5f})")
    assert abs(ana - mc) < 4 * res.stderr_pct / 100 + 2e-4, "MC != Stulz closed form"

    # Monotonicity: higher correlation => worst-of put cheaper
    lo = price_worst_of_put(a1, a2, 0.1, spec, n_paths=100_000, seed=2, greeks=False)
    hi = price_worst_of_put(a1, a2, 0.9, spec, n_paths=100_000, seed=2, greeks=False)
    print(f"[validation] rho=0.1 premium={lo.premium_pct:.3f}%  "
          f"rho=0.9 premium={hi.premium_pct:.3f}%")
    assert lo.premium_pct > hi.premium_pct

    # Worst-of put >= max(single-asset puts): price single by pairing asset with itself
    single = price_worst_of_put(a2, a2, 0.999, spec, n_paths=100_000, seed=2, greeks=False)
    both = price_worst_of_put(a1, a2, rho, spec, n_paths=100_000, seed=2, greeks=False)
    print(f"[validation] worst-of={both.premium_pct:.3f}%  >=  "
          f"single TSLA put={single.premium_pct:.3f}%")
    assert both.premium_pct >= single.premium_pct - 0.05

    # Full Heston run with Greeks
    a1h = Asset("NVDA", S1, 0.0, v0=0.20, kappa=2.0, theta=0.22, xi=0.9, rho_sv=-0.6)
    a2h = Asset("TSLA", S2, 0.0, v0=0.30, kappa=2.0, theta=0.32, xi=1.0, rho_sv=-0.6)
    resh = price_worst_of_put(a1h, a2h, 0.5, spec, n_paths=50_000, seed=5)
    print(f"[smoke] Heston worst-of put premium={resh.premium_pct:.2f}% "
          f"(+/-{resh.stderr_pct:.2f})  P(ITM)={resh.prob_itm:.1%}  "
          f"deltas={np.round(resh.deltas,3)}  vegas={np.round(resh.vegas,3)}  "
          f"dP/d(rho+0.05)={resh.corr_sens:+.3f}")
    assert resh.corr_sens < 0.05   # correlation up => premium down (allow noise)
    print("All basket-engine self-tests passed.")
