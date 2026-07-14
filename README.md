# Worst-of autocallable note pricer

Prices barrier reverse convertibles / autocallables on up to 3 stocks using a
multi-asset **Heston stochastic-volatility model** and **Monte Carlo**, with
live market data from Yahoo Finance and a Streamlit UI.

## Quick start

```bash
pip install numpy pandas scipy plotly streamlit yfinance
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

Run the engine's built-in validation suite anytime with:

```bash
python heston_engine.py
```

It checks the Monte Carlo against the semi-analytic Heston vanilla price
(characteristic-function quadrature) and a zero-volatility autocall identity.

## What you can configure

- **Underlyings**: 1–3 tickers, history window for calibration, or fully manual
  spot/vol/correlation inputs if you have no internet
- **Terms**: tenor, observation frequency, coupon rate, coupon barrier,
  memory/guaranteed coupons, autocall trigger and first callable date,
  knock-in barrier (continuous *or* European), downside strike, risk-free rate
- **Model**: every Heston parameter (v0, kappa, theta, xi, rho) per asset is
  editable; historical estimates are pre-filled
- **Simulation**: number of paths (antithetic variates on by default), seed

## Outputs

Fair value per 100 with a 95% confidence interval, embedded margin vs par,
probability of autocall by date, knock-in and capital-loss probabilities,
expected loss given loss, expected life, expected coupons, CVaR of the payoff,
plus charts: payoff distribution, autocall timing, sample worst-of paths
against the barriers, and the maturity payoff diagram.

## Methodology notes

- Dynamics: Heston per asset; cross-asset spot correlation via Cholesky;
  per-asset spot–vol correlation (leverage). Full-truncation Euler, daily steps.
- Continuous barriers use a per-step **Brownian-bridge** crossing probability
  to remove discrete-monitoring bias (approximate under stochastic vol).
- Heston parameters are estimated from *historical* returns (rolling realized
  variance, AR(1) mean-reversion fit). Dealers calibrate to the option-implied
  surface instead — implied vols and skew are usually higher, so historical
  calibration tends to make notes look *more* valuable than dealer marks.
  Override the parameters with implied levels for market-consistent pricing.
- Not modeled: issuer credit spread, stochastic rates, discrete cash dividends,
  correlation skew (correlations rising in crashes). All of these make real
  notes worth *less* than this model shows.

Educational tool — not investment advice or a dealer valuation.
