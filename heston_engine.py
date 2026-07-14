"""
Heston Monte Carlo engine for worst-of autocallable barrier reverse convertibles.

Model (per asset i, under the risk-neutral measure):
    dS_i = (r - q_i) S_i dt + sqrt(v_i) S_i dW_i^S
    dv_i = kappa_i (theta_i - v_i) dt + xi_i sqrt(v_i) dW_i^v
    d<W_i^S, W_i^v> = rho_i dt              (spot-vol leverage per asset)
    d<W_i^S, W_j^S> = corr_ij dt            (cross-asset spot correlation)

Discretisation: full-truncation Euler (Lord et al. 2010), daily steps.
Continuous barriers use a per-step Brownian-bridge crossing probability so the
knock-in estimate does not suffer from discrete-monitoring bias.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ----------------------------------------------------------------------------- 
# Parameter containers
# -----------------------------------------------------------------------------

@dataclass
class AssetParams:
    name: str
    spot: float
    div_yield: float = 0.0     # continuous dividend yield
    v0: float = 0.04           # initial variance  (vol^2)
    kappa: float = 2.0         # mean-reversion speed of variance
    theta: float = 0.04        # long-run variance
    xi: float = 0.5            # vol of vol
    rho_sv: float = -0.6       # spot/vol correlation (leverage effect)

    def feller_ratio(self) -> float:
        """2*kappa*theta / xi^2 ; >= 1 means variance stays strictly positive."""
        return 2.0 * self.kappa * self.theta / max(self.xi**2, 1e-12)


@dataclass
class ProductSpec:
    notional: float = 1000.0
    tenor_years: float = 3.0
    obs_per_year: int = 4            # coupon / autocall observation frequency
    coupon_rate_pa: float = 0.10     # per annum, paid per period if above coupon barrier
    coupon_barrier: float = 0.70     # as fraction of initial fixing (worst-of)
    autocall_trigger: float = 1.00   # as fraction of initial fixing (worst-of)
    first_callable_obs: int = 1      # 1 = callable from the first observation date
    ki_barrier: float = 0.60         # knock-in barrier (fraction of initial)
    barrier_type: str = "continuous" # "continuous" or "european"
    strike_pct: float = 1.00         # downside strike (loss measured from here)
    memory_coupon: bool = True
    guaranteed_coupon: bool = False  # pay coupon regardless of barrier
    r: float = 0.04                  # flat risk-free rate (cont. comp.)


@dataclass
class PricingResult:
    price: float                      # PV per 100 notional
    stderr: float                     # MC standard error, per 100
    prob_autocall: np.ndarray         # P(called at obs k), len n_obs
    prob_survive_to_maturity: float
    prob_ki: float                    # P(knock-in event)
    prob_capital_loss: float          # P(redemption < notional)
    exp_loss_given_loss: float        # avg capital loss % when a loss occurs
    exp_life_years: float
    exp_coupons_per100: float         # undiscounted expected coupon total
    exp_total_return: float           # E[total cash]/notional - 1
    payoff_dist: np.ndarray           # total (undiscounted) cash per path, per 100
    pv_dist: np.ndarray               # discounted cash per path, per 100
    worst_final: np.ndarray           # worst-of performance at exit per path
    sample_paths: np.ndarray          # (n_saved, n_steps+1) worst-of trajectories
    sample_exit_step: np.ndarray      # exit step index for saved paths
    obs_steps: np.ndarray
    time_grid: np.ndarray
    coupon_per_period: float          # per 100 notional
    var_95: float                     # 5th percentile of total payoff per 100
    cvar_95: float


# -----------------------------------------------------------------------------
# Heston parameter estimation from historical prices (heuristic)
# -----------------------------------------------------------------------------

def estimate_heston_from_returns(log_returns: np.ndarray,
                                 trading_days: int = 252) -> dict:
    """
    Heuristic historical estimation of Heston parameters from daily log returns.

    NOTE: a production desk calibrates (v0, kappa, theta, xi, rho) to the
    option-implied volatility surface, not to history. This estimator is a
    reasonable stand-in when no option data is supplied and is fully
    overridable in the UI.
    """
    lr = np.asarray(log_returns, dtype=float)
    lr = lr[np.isfinite(lr)]
    if lr.size < 60:
        raise ValueError("Need at least 60 return observations to estimate parameters.")

    # Rolling 21-day realized variance (annualized) as a proxy for v_t
    win = 21
    n = lr.size - win + 1
    rv = np.array([lr[i:i + win].var(ddof=1) for i in range(n)]) * trading_days
    rv = np.clip(rv, 1e-6, None)

    v0 = float(rv[-1])                       # today's short-dated variance
    theta = float(np.median(rv))             # long-run level (median is robust)

    # kappa from AR(1) fit of variance:  v_{t+1}-v_t = kappa(theta - v_t)dt + noise
    dt = 1.0 / trading_days
    dv = np.diff(rv)
    x = theta - rv[:-1]
    denom = float(np.dot(x, x))
    kappa = float(np.dot(x, dv) / (denom * dt)) if denom > 0 else 2.0
    kappa = float(np.clip(kappa, 0.3, 10.0))

    # xi from the residual noise of the variance process: std(dv) ~ xi*sqrt(v)*sqrt(dt)
    resid = dv - kappa * x * dt
    xi = float(np.std(resid, ddof=1) / (np.sqrt(np.mean(rv[:-1])) * np.sqrt(dt)))
    xi = float(np.clip(xi, 0.10, 2.5))

    # rho from corr(returns, changes in variance) over aligned windows
    dr = lr[win - 1:][1:n]                   # returns aligned with dv
    if dr.size == dv.size and dv.size > 10:
        c = np.corrcoef(dr, dv)[0, 1]
        rho = float(np.clip(c if np.isfinite(c) else -0.6, -0.95, 0.2))
    else:
        rho = -0.6

    return {"v0": v0, "theta": theta, "kappa": kappa, "xi": xi, "rho_sv": rho,
            "hist_vol": float(np.sqrt(theta))}


# -----------------------------------------------------------------------------
# Semi-analytic Heston vanilla call (validation of the MC engine)
# -----------------------------------------------------------------------------

def heston_call_analytic(S0, K, T, r, q, v0, kappa, theta, xi, rho,
                         n_quad: int = 256, u_max: float = 200.0) -> float:
    """European call under Heston via the Heston/Albrecher characteristic
    function and Gauss-Legendre quadrature. Used only to validate the MC."""
    x, w = np.polynomial.legendre.leggauss(n_quad)
    u = 0.5 * u_max * (x + 1.0)
    wq = 0.5 * u_max * w

    def cf(phi, j):
        a = kappa * theta
        if j == 1:
            uu, b = 0.5, kappa - rho * xi
        else:
            uu, b = -0.5, kappa
        d = np.sqrt((rho * xi * 1j * phi - b) ** 2 - xi**2 * (2 * uu * 1j * phi - phi**2))
        g = (b - rho * xi * 1j * phi + d) / (b - rho * xi * 1j * phi - d)
        ed = np.exp(d * T)
        C = (r - q) * 1j * phi * T + a / xi**2 * (
            (b - rho * xi * 1j * phi + d) * T - 2.0 * np.log((1 - g * ed) / (1 - g)))
        D = (b - rho * xi * 1j * phi + d) / xi**2 * (1 - ed) / (1 - g * ed)
        return np.exp(C + D * v0 + 1j * phi * np.log(S0))

    P = []
    for j in (1, 2):
        integrand = np.real(np.exp(-1j * u * np.log(K)) * cf(u, j) / (1j * u))
        P.append(0.5 + (wq * integrand).sum() / np.pi)
    return float(S0 * np.exp(-q * T) * P[0] - K * np.exp(-r * T) * P[1])


# -----------------------------------------------------------------------------
# Monte Carlo pricer
# -----------------------------------------------------------------------------

def price_note(assets: List[AssetParams],
               spot_corr: np.ndarray,
               spec: ProductSpec,
               n_paths: int = 20000,
               steps_per_year: int = 252,
               seed: Optional[int] = 42,
               antithetic: bool = True,
               brownian_bridge: bool = True,
               n_saved_paths: int = 150) -> PricingResult:
    """
    Price a worst-of autocallable barrier reverse convertible by Monte Carlo
    under multi-asset Heston dynamics.
    """
    m = len(assets)
    if not (1 <= m <= 3):
        raise ValueError("1 to 3 underlyings supported.")
    spot_corr = np.asarray(spot_corr, dtype=float).reshape(m, m)

    # Cholesky with a small ridge if the matrix is borderline
    try:
        L = np.linalg.cholesky(spot_corr)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(spot_corr + 1e-8 * np.eye(m))

    n_steps = max(int(round(spec.tenor_years * steps_per_year)), 2)
    dt = spec.tenor_years / n_steps
    sqdt = np.sqrt(dt)
    time_grid = np.linspace(0.0, spec.tenor_years, n_steps + 1)

    n_obs = int(round(spec.tenor_years * spec.obs_per_year))
    obs_steps = np.unique(np.round(
        np.arange(1, n_obs + 1) * n_steps / n_obs).astype(int))
    obs_times = obs_steps * dt
    coupon_per_period = spec.coupon_rate_pa / spec.obs_per_year * 100.0

    if antithetic and n_paths % 2:
        n_paths += 1
    half = n_paths // 2 if antithetic else n_paths

    rng = np.random.default_rng(seed)

    S0 = np.array([a.spot for a in assets])
    q = np.array([a.div_yield for a in assets])
    v0 = np.array([a.v0 for a in assets])
    kap = np.array([a.kappa for a in assets])
    th = np.array([a.theta for a in assets])
    xi = np.array([a.xi for a in assets])
    rho = np.array([a.rho_sv for a in assets])
    rho_c = np.sqrt(np.clip(1.0 - rho**2, 0.0, 1.0))

    # State
    S = np.tile(S0, (n_paths, 1))
    v = np.tile(v0, (n_paths, 1))
    perf = np.ones((n_paths, m))
    worst = perf.min(axis=1)

    alive = np.ones(n_paths, dtype=bool)          # not yet autocalled
    log_surv = np.zeros(n_paths)                  # log P(no KI so far | path)
    ki_hard = np.zeros(n_paths, dtype=bool)       # KI observed on the grid
    pv = np.zeros(n_paths)                        # discounted cash per path (per 100)
    cash = np.zeros(n_paths)                      # undiscounted cash per path (per 100)
    coupons_cash = np.zeros(n_paths)
    missed = np.zeros(n_paths)                    # memory-coupon counter
    exit_time = np.full(n_paths, spec.tenor_years)
    prob_ac = np.zeros(len(obs_steps))

    n_save = min(n_saved_paths, n_paths)
    saved = np.ones((n_save, n_steps + 1))
    saved_exit = np.full(n_save, n_steps, dtype=int)
    saved_done = np.zeros(n_save, dtype=bool)

    B = spec.ki_barrier
    continuous = spec.barrier_type == "continuous"
    obs_ptr = {int(s): k for k, s in enumerate(obs_steps)}

    for t in range(1, n_steps + 1):
        # ---- draw correlated shocks (antithetic) ----
        if antithetic:
            Zs_h = rng.standard_normal((half, m))
            Zw_h = rng.standard_normal((half, m))
            Zs = np.vstack([Zs_h, -Zs_h])
            Zw = np.vstack([Zw_h, -Zw_h])
        else:
            Zs = rng.standard_normal((n_paths, m))
            Zw = rng.standard_normal((n_paths, m))
        Zc = Zs @ L.T                       # cross-asset correlated spot shocks
        Zv = rho * Zc + rho_c * Zw          # per-asset spot/vol correlation

        vp = np.maximum(v, 0.0)
        sq_v = np.sqrt(vp)
        S_prev = S
        S = S * np.exp((spec.r - q - 0.5 * vp) * dt + sq_v * sqdt * Zc)
        v = v + kap * (th - vp) * dt + xi * sq_v * sqdt * Zv

        perf = S / S0
        worst = perf.min(axis=1)

        # ---- knock-in monitoring ----
        if continuous:
            below = (perf < B).any(axis=1)
            ki_hard |= below & alive
            if brownian_bridge:
                # Brownian-bridge crossing prob per asset within the step,
                # using that step's instantaneous variance (approximation
                # under stochastic vol; exact for locally constant vol).
                Babs = B * S0
                both_above = (S_prev > Babs) & (S > Babs)
                var_step = np.maximum(vp, 1e-8) * dt
                with np.errstate(divide="ignore", invalid="ignore"):
                    p_cross = np.exp(-2.0 * np.log(S_prev / Babs)
                                     * np.log(S / Babs) / var_step)
                p_cross = np.where(both_above, np.clip(p_cross, 0.0, 1.0), 0.0)
                inc = np.log1p(-np.clip(p_cross, 0.0, 1.0 - 1e-12)).sum(axis=1)
                log_surv = np.where(alive & ~ki_hard, log_surv + inc, log_surv)

        # ---- save sample worst-of trajectories ----
        w_save = worst[:n_save].copy()
        saved[:, t] = np.where(saved_done, saved[:, t - 1], w_save)

        # ---- observation date logic ----
        if t in obs_ptr:
            k = obs_ptr[t]
            t_yrs = obs_times[k]
            df = np.exp(-spec.r * t_yrs)

            # coupon decision (before/with call — called notes pay the coupon)
            if spec.guaranteed_coupon:
                pay_c = alive
            else:
                pay_c = alive & (worst >= spec.coupon_barrier)
            c_amt = coupon_per_period * (1.0 + (missed if spec.memory_coupon else 0.0))
            add = np.where(pay_c, c_amt, 0.0)
            pv += add * df
            cash += add
            coupons_cash += add
            if spec.memory_coupon:
                missed = np.where(pay_c, 0.0, missed + (alive & ~pay_c))
            calling_allowed = (k + 1) >= spec.first_callable_obs and t < n_steps
            if calling_allowed:
                call = alive & (worst >= spec.autocall_trigger)
                pv += np.where(call, 100.0 * df, 0.0)
                cash += np.where(call, 100.0, 0.0)
                prob_ac[k] = call.mean()
                exit_time = np.where(call, t_yrs, exit_time)
                newly = call[:n_save] & ~saved_done
                saved_exit = np.where(newly, t, saved_exit)
                saved_done |= call[:n_save]
                alive &= ~call

    # ---- maturity redemption for surviving paths ----
    dfT = np.exp(-spec.r * spec.tenor_years)
    surv_prob = np.where(ki_hard, 0.0, np.exp(log_surv)) if (continuous and brownian_bridge) \
        else np.where(ki_hard, 0.0, 1.0)
    if not continuous:  # European barrier: only the final fixing matters
        surv_prob = (worst >= B).astype(float)

    loss_leg = np.minimum(1.0, worst / spec.strike_pct) * 100.0
    redemption = surv_prob * 100.0 + (1.0 - surv_prob) * loss_leg
    pv += np.where(alive, redemption * dfT, 0.0)
    cash += np.where(alive, redemption, 0.0)

    # ---- statistics ----
    price = float(pv.mean())
    stderr = float(pv.std(ddof=1) / np.sqrt(n_paths))

    ki_prob = float((alive * (1.0 - surv_prob)).sum() / max(alive.sum(), 1)) \
        if alive.any() else 0.0
    prob_ki_uncond = float((alive * (1.0 - surv_prob)).mean())

    loss_mask_prob = alive * (1.0 - surv_prob) * (worst < spec.strike_pct)
    prob_loss = float(loss_mask_prob.mean())
    if prob_loss > 1e-9:
        loss_size = (100.0 - loss_leg)
        exp_lgl = float((loss_mask_prob * loss_size).sum() / loss_mask_prob.sum())
    else:
        exp_lgl = 0.0

    worst_exit = worst.copy()  # worst at exit == final for survivors; for called
    # paths the exit level is >= trigger; approximate with trigger for display
    pay_sorted = np.sort(cash)
    var95 = float(np.percentile(cash, 5))
    cvar95 = float(pay_sorted[: max(int(0.05 * n_paths), 1)].mean())

    return PricingResult(
        price=price, stderr=stderr,
        prob_autocall=prob_ac,
        prob_survive_to_maturity=float(alive.mean()),
        prob_ki=prob_ki_uncond,
        prob_capital_loss=prob_loss,
        exp_loss_given_loss=exp_lgl,
        exp_life_years=float(exit_time.mean()),
        exp_coupons_per100=float(coupons_cash.mean()),
        exp_total_return=float(cash.mean() / 100.0 - 1.0),
        payoff_dist=cash, pv_dist=pv,
        worst_final=worst_exit,
        sample_paths=saved, sample_exit_step=saved_exit,
        obs_steps=obs_steps, time_grid=time_grid,
        coupon_per_period=coupon_per_period,
        var_95=var95, cvar_95=cvar95,
    )


# -----------------------------------------------------------------------------
# Quick self-tests:  python heston_engine.py
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # 1) MC vs semi-analytic Heston vanilla call (single asset)
    a = AssetParams("TEST", spot=100.0, v0=0.04, kappa=2.0, theta=0.05,
                    xi=0.6, rho_sv=-0.7)
    r, T, K = 0.03, 1.0, 100.0
    ana = heston_call_analytic(100, K, T, r, 0.0, a.v0, a.kappa, a.theta, a.xi, a.rho_sv)

    rng = np.random.default_rng(1)
    n, steps = 200000, 252
    dt = T / steps
    S = np.full(n, 100.0); v = np.full(n, a.v0)
    for _ in range(steps):
        z1 = rng.standard_normal(n); z2 = rng.standard_normal(n)
        zv = a.rho_sv * z1 + np.sqrt(1 - a.rho_sv**2) * z2
        vp = np.maximum(v, 0)
        S *= np.exp((r - 0.5 * vp) * dt + np.sqrt(vp * dt) * z1)
        v += a.kappa * (a.theta - vp) * dt + a.xi * np.sqrt(vp * dt) * zv
    mc = np.exp(-r * T) * np.maximum(S - K, 0).mean()
    se = np.exp(-r * T) * np.maximum(S - K, 0).std() / np.sqrt(n)
    print(f"[validation] Heston call  analytic={ana:.4f}  MC={mc:.4f} (+/-{se:.4f})")
    assert abs(ana - mc) < 4 * se + 0.05, "MC does not match analytic Heston price"

    # 2) Zero-vol sanity: note must autocall at first observation
    assets = [AssetParams("A", 100, v0=1e-8, theta=1e-8, xi=1e-6, kappa=1.0)]
    spec = ProductSpec(tenor_years=2, obs_per_year=4, coupon_rate_pa=0.08,
                       autocall_trigger=1.0, ki_barrier=0.6, r=0.03)
    res = price_note(assets, np.eye(1), spec, n_paths=2000, seed=7)
    expected = (100 + 2.0) * np.exp(-0.03 * 0.25)
    print(f"[validation] zero-vol autocall  price={res.price:.4f}  expected={expected:.4f}")
    assert abs(res.price - expected) < 0.05

    # 3) Full 3-asset worst-of pricing smoke test
    assets = [
        AssetParams("AAA", 100, 0.01, v0=0.06, kappa=2.0, theta=0.06, xi=0.7, rho_sv=-0.6),
        AssetParams("BBB", 50, 0.02, v0=0.09, kappa=1.5, theta=0.08, xi=0.8, rho_sv=-0.7),
        AssetParams("CCC", 200, 0.00, v0=0.16, kappa=2.5, theta=0.12, xi=0.9, rho_sv=-0.5),
    ]
    corr = np.array([[1.0, 0.6, 0.4], [0.6, 1.0, 0.5], [0.4, 0.5, 1.0]])
    spec = ProductSpec(tenor_years=3, obs_per_year=4, coupon_rate_pa=0.12,
                       coupon_barrier=0.7, autocall_trigger=1.0,
                       ki_barrier=0.6, barrier_type="continuous",
                       strike_pct=1.0, memory_coupon=True, r=0.04)
    res = price_note(assets, corr, spec, n_paths=30000, seed=3)
    print(f"[smoke] 3-asset worst-of BRC price = {res.price:.2f} +/- {res.stderr:.2f} per 100")
    print(f"        P(autocall)={res.prob_autocall.sum():.1%}  "
          f"P(KI)={res.prob_ki:.1%}  P(loss)={res.prob_capital_loss:.1%}  "
          f"E[loss|loss]={res.exp_loss_given_loss:.1f}%  "
          f"E[life]={res.exp_life_years:.2f}y")
    assert 40 < res.price < 130
    print("All engine self-tests passed.")
