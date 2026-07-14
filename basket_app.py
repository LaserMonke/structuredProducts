"""
Worst-of put seller — pricing and risk dashboard.

The client SELLS a European put on the worst performer of 2 stocks
(default: NVDA / TSLA, strike 70% of spot, 1-year tenor).

Run with:   streamlit run basket_app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from basket_engine import (Asset, PutSpec, price_worst_of_put,
                           estimate_heston_from_returns, stulz_put_on_min)

st.set_page_config(page_title="Worst-of Put Seller", page_icon="🧾", layout="wide")

ACCENT, RED, GREEN, GRAY, AMBER = "#2a5d8f", "#c0392b", "#1e8e5a", "#8a8a85", "#b8860b"

st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; max-width: 1250px;}
  div[data-testid="stMetric"] {background:#f7f6f2; border:1px solid #e5e3da;
    border-radius:10px; padding:14px 18px;}
  .note-box {background:#f2f5f9; border-left:4px solid #2a5d8f; padding:12px 16px;
    border-radius:0 8px 8px 0; font-size:.92rem; margin:6px 0 14px 0;}
  .warn-box {background:#faf3ee; border-left:4px solid #c0392b; padding:12px 16px;
    border-radius:0 8px 8px 0; font-size:.92rem; margin:6px 0 14px 0;}
</style>""", unsafe_allow_html=True)

st.title("Selling a worst-of put on two stocks")
st.caption("You (the client) sell a European put on the worst performer of two "
           "stocks — you collect the premium today and pay the worst-of shortfall "
           "below the strike at maturity, if any. Heston Monte Carlo, validated "
           "against the Stulz (1982) closed form.")

UNIVERSE = ["NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AMD",
            "NFLX", "JPM", "XOM", "COIN", "PLTR", "SPY", "QQQ"]

# ---------------------------------------------------------------- market data
@st.cache_data(ttl=3600, show_spinner=False)
def fetch(tickers: tuple, lookback: str):
    import yfinance as yf
    px = yf.download(list(tickers), period=lookback, auto_adjust=True,
                     progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(tickers[0])
    px = px[list(tickers)].dropna()
    if len(px) < 80:
        raise ValueError("Not enough overlapping history.")
    lr = np.log(px / px.shift(1)).dropna()
    divs = {}
    for t in tickers:
        try:
            y = yf.Ticker(t).info.get("dividendYield", 0.0) or 0.0
            divs[t] = float(y) if y < 1 else float(y) / 100.0
        except Exception:
            divs[t] = 0.0
    return px, lr, float(lr.corr().iloc[0, 1]), divs


with st.sidebar:
    st.header("Underlyings")
    t1 = st.selectbox("Stock 1", UNIVERSE, index=0)
    t2 = st.selectbox("Stock 2", [u for u in UNIVERSE if u != t1], index=0)
    tickers = (t1, t2)
    lookback = st.selectbox("Calibration history", ["1y", "2y", "3y", "5y"], 1)
    use_live = st.toggle("Fetch live market data (yfinance)", True)

    st.header("Trade terms")
    strike_pct = st.slider("Strike (% of spot)", 40, 110, 70, 1) / 100
    tenor = st.slider("Tenor (years)", 0.25, 3.0, 1.0, 0.25)
    notional = st.number_input("Notional (USD)", 10_000, 1_000_000_000,
                               1_000_000, step=10_000)
    r = st.number_input("Risk-free rate (% p.a.)", 0.0, 15.0, 4.0, 0.1) / 100

    st.header("Simulation")
    n_paths = st.select_slider("Monte Carlo paths",
                               [10_000, 20_000, 50_000, 100_000, 200_000], 50_000)
    seed = st.number_input("Random seed", 0, 10_000, 42)
    do_greeks = st.toggle("Compute Greeks (slower)", True)

# ---- data or manual fallback ----
data_ok, err = False, ""
if use_live:
    try:
        with st.spinner("Fetching market data…"):
            hist, lr, rho_hist, divs = fetch(tickers, lookback)
        spots = {t: float(hist[t].iloc[-1]) for t in tickers}
        data_ok = True
    except Exception as e:
        err = str(e)

if not data_ok:
    if use_live:
        reason = err or "no connection"
        st.markdown(f"<div class='warn-box'><b>Live data unavailable</b> "
                    f"({reason}). Enter market inputs manually.</div>",
                    unsafe_allow_html=True)
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

# ---------------------------------------------------- Heston params per asset
st.subheader("1 · Market data & model parameters")
st.markdown("<div class='note-box'>Parameters are estimated from historical "
            "returns and fully overridable. For a dealer-consistent price, set "
            "√v₀ ≈ the 1-year <i>implied</i> vol of each name (implied usually "
            "runs above historical, especially with downside skew — which is "
            "exactly the region a 70% put lives in).</div>", unsafe_allow_html=True)

assets = []
cols = st.columns(2)
for i, t in enumerate(tickers):
    if data_ok:
        est = estimate_heston_from_returns(lr[t].values)
    else:
        vv = vols[t] ** 2
        est = {"v0": vv, "theta": vv, "kappa": 2.0, "xi": 0.8, "rho_sv": -0.6}
    with cols[i]:
        st.markdown(f"**{t}** · spot **{spots[t]:,.2f}** · strike "
                    f"**{spots[t]*strike_pct:,.2f}** ({strike_pct:.0%})")
        v0 = st.number_input("v₀", 0.0001, 4.0, float(round(est["v0"], 4)),
                             format="%.4f", key=f"v0{i}")
        th = st.number_input("θ", 0.0001, 4.0, float(round(est["theta"], 4)),
                             format="%.4f", key=f"th{i}")
        ka = st.number_input("κ", 0.05, 15.0, float(round(est["kappa"], 2)), key=f"ka{i}")
        xv = st.number_input("ξ (vol of vol)", 0.01, 3.0, float(round(est["xi"], 2)),
                             key=f"xi{i}")
        rh = st.number_input("ρ spot-vol", -0.99, 0.5, float(round(est["rho_sv"], 2)),
                             key=f"rh{i}")
        st.caption(f"√v₀ = {np.sqrt(v0):.1%} · √θ = {np.sqrt(th):.1%}")
        assets.append(Asset(t, spots[t], divs.get(t, 0.0), v0, ka, th, xv, rh))

rho_s = st.slider("Spot correlation between the two stocks", -0.5, 0.99,
                  float(round(np.clip(rho_hist, -0.5, 0.99), 2)), 0.01)

if data_ok and hist is not None:
    norm_px = hist / hist.iloc[0] * 100
    fig = go.Figure()
    for t in tickers:
        fig.add_trace(go.Scatter(x=norm_px.index, y=norm_px[t], name=t,
                                 line=dict(width=1.6)))
    fig.update_layout(title=f"Normalised history ({lookback}) · realized corr "
                            f"= {rho_hist:.2f}", height=270,
                      margin=dict(l=10, r=10, t=45, b=10), yaxis_title="Rebased to 100")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------- pricing
spec = PutSpec(strike_pct=strike_pct, tenor_years=tenor, r=r, notional=notional)

st.subheader("2 · Price & seller risk")
if not st.button("▶ Price the worst-of put", type="primary", use_container_width=True):
    st.info("Adjust terms in the sidebar, then price.")
    st.stop()

with st.spinner(f"Simulating {n_paths:,} Heston paths…"):
    res = price_worst_of_put(assets[0], assets[1], rho_s, spec,
                             n_paths=int(n_paths), seed=int(seed),
                             greeks=do_greeks)

# Black-Scholes/Stulz reference using √θ as flat vols
bs_ref = stulz_put_on_min(1.0, 1.0, strike_pct, tenor, r,
                          assets[0].div_yield, assets[1].div_yield,
                          float(np.sqrt(assets[0].theta)),
                          float(np.sqrt(assets[1].theta)), rho_s) * 100

m = st.columns(5)
m[0].metric("Premium you receive", f"{res.premium_pct:.2f}%",
            f"± {2*res.stderr_pct:.2f}% (95% CI)")
m[1].metric("Premium in cash", f"${res.premium_cash:,.0f}",
            f"on ${notional:,.0f}")
m[2].metric("Annualized yield", f"{res.premium_pct/tenor:.2f}% p.a.")
m[3].metric("P(put exercised)", f"{res.prob_itm:.1%}",
            f"worst-of < {strike_pct:.0%}", delta_color="off")
m[4].metric("Breakeven worst-of", f"{res.breakeven_worst:.1%}",
            "of initial spot", delta_color="off")

m = st.columns(5)
m[0].metric("E[payout] if exercised", f"−{res.exp_payoff_given_itm:.1f}%",
            "of notional", delta_color="off")
m[1].metric("95% tail payout (CVaR)", f"−{res.cvar95_payoff_pct:.1f}%",
            "avg of worst 5% paths", delta_color="off")
m[2].metric("Worst simulated payout", f"−{res.max_loss_pct:.1f}%",
            f"max possible: −{strike_pct:.0%}", delta_color="off")
m[3].metric(f"P({tickers[0]} is the worst | ITM)", f"{res.prob_worst_is[0]:.0%}")
m[4].metric("Heston vs flat-vol (Stulz)", f"{res.premium_pct - bs_ref:+.2f}%",
            f"BS ref {bs_ref:.2f}%", delta_color="off")

st.markdown(f"<div class='note-box'>Seller economics: you collect "
            f"<b>${res.premium_cash:,.0f}</b> today. In {res.prob_itm:.0%} of paths "
            f"the worst stock finishes below {strike_pct:.0%} and you pay the "
            f"shortfall — averaging {res.exp_payoff_given_itm:.1f}% of notional when "
            f"it happens. Your P&L is positive as long as the worst performer stays "
            f"above <b>{res.breakeven_worst:.1%}</b> of its initial level.</div>",
            unsafe_allow_html=True)

if do_greeks:
    st.markdown("**Greeks (per 100 notional, seller = short these)**")
    gdf = pd.DataFrame({
        "Sensitivity": [f"Delta {tickers[0]} (+1% spot)", f"Delta {tickers[1]} (+1% spot)",
                        f"Vega {tickers[0]} (+1 vol pt)", f"Vega {tickers[1]} (+1 vol pt)",
                        "Correlation (+0.05)"],
        "Option value change": [f"{res.deltas[0]:+.3f}", f"{res.deltas[1]:+.3f}",
                                f"{res.vegas[0]:+.3f}", f"{res.vegas[1]:+.3f}",
                                f"{res.corr_sens:+.3f}"],
        "Your P&L (short)": [f"{-res.deltas[0]:+.3f}", f"{-res.deltas[1]:+.3f}",
                             f"{-res.vegas[0]:+.3f}", f"{-res.vegas[1]:+.3f}",
                             f"{-res.corr_sens:+.3f}"]})
    st.dataframe(gdf, hide_index=True, use_container_width=True)
    st.caption("As the seller you are short vol on both names and LONG correlation "
               "— you profit if the stocks move more in lockstep, lose if they "
               "decorrelate. That short-dispersion position is the defining risk "
               "of worst-of structures.")

# --------------------------------------------------------------------- charts
cA, cB = st.columns(2)
with cA:
    fig = go.Figure(go.Histogram(x=res.seller_pnl_pct, nbinsx=90,
                                 marker_color=ACCENT, opacity=0.85))
    fig.add_vline(x=0, line_dash="dash", line_color=GRAY)
    fig.add_vline(x=float(np.mean(res.seller_pnl_pct)), line_color=GREEN,
                  annotation_text=f"mean {np.mean(res.seller_pnl_pct):+.1f}%")
    fig.update_layout(title="Your P&L at maturity (% of notional, premium accrued)",
                      height=330, margin=dict(l=10, r=10, t=45, b=10),
                      xaxis_title="Seller P&L %", yaxis_title="Paths")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Classic short-option shape: a tall spike of small wins (keep the "
               "premium) and a long left tail of large losses.")

with cB:
    x = np.linspace(0.1, 1.4, 200)
    prem_T = res.premium_pct / np.exp(-r * tenor) / 100
    pnl = (prem_T - np.maximum(strike_pct - x, 0)) * 100
    fig = go.Figure(go.Scatter(x=x * 100, y=pnl, line=dict(color=ACCENT, width=2.5)))
    fig.add_hline(y=0, line_color=GRAY, line_width=1)
    fig.add_vline(x=strike_pct * 100, line_dash="dot", line_color=AMBER,
                  annotation_text="strike")
    fig.add_vline(x=res.breakeven_worst * 100, line_dash="dot", line_color=RED,
                  annotation_text="breakeven")
    fig.update_layout(title="Seller payoff diagram vs final worst-of",
                      height=330, margin=dict(l=10, r=10, t=45, b=10),
                      xaxis_title="Final worst-of (% of initial)",
                      yaxis_title="P&L (% of notional)")
    st.plotly_chart(fig, use_container_width=True)

cA, cB = st.columns(2)
with cA:
    if res.sample_paths is not None:
        fig = go.Figure()
        for i in range(min(130, res.sample_paths.shape[0])):
            itm = res.sample_paths[i, -1] < strike_pct
            fig.add_trace(go.Scatter(
                x=res.time_grid, y=res.sample_paths[i] * 100, mode="lines",
                line=dict(width=0.8, color=RED if itm else "rgba(42,93,143,0.25)"),
                showlegend=False, hoverinfo="skip"))
        fig.add_hline(y=strike_pct * 100, line_dash="dot", line_color=AMBER,
                      annotation_text="strike", annotation_position="right")
        fig.update_layout(title="Sample worst-of paths (red = exercised against you)",
                          height=330, margin=dict(l=10, r=10, t=45, b=10),
                          xaxis_title="Years", yaxis_title="Worst-of (% of initial)")
        st.plotly_chart(fig, use_container_width=True)

with cB:
    fig = go.Figure(go.Histogram2d(
        x=res.perf_T[:, 0] * 100, y=res.perf_T[:, 1] * 100,
        nbinsx=60, nbinsy=60, colorscale="Blues"))
    fig.add_vline(x=strike_pct * 100, line_dash="dot", line_color=RED)
    fig.add_hline(y=strike_pct * 100, line_dash="dot", line_color=RED)
    fig.update_layout(title=f"Joint final performance — you lose in the L-shaped "
                            f"region", height=330,
                      margin=dict(l=10, r=10, t=45, b=10),
                      xaxis_title=f"{tickers[0]} final (%)",
                      yaxis_title=f"{tickers[1]} final (%)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Either stock crossing its red line puts you in the payout zone — "
               "with two stocks there are two independent ways to lose. That's "
               "why the worst-of premium exceeds any single-stock put premium.")

st.markdown("""
<div class='warn-box'><b>Model limitations.</b> Historical Heston calibration
typically understates option-implied vol and skew, so the model premium is a
<i>floor</i> relative to where a dealer would quote. Constant correlation is
assumed — realized correlation spikes in sell-offs, precisely when the put pays.
European exercise, flat rates, no credit/funding adjustments. Educational tool,
not investment advice.</div>""", unsafe_allow_html=True)
