"""
Basket option pricer — worst-of / best-of, calls / puts on any two stocks.

Run with:   streamlit run basket_app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from basket_engine import (Asset, OptionSpec, price_basket_option,
                           estimate_heston_from_returns, calibrate_heston,
                           model_smile, bs_basket_option)

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
st.caption("Worst-of / best-of · calls and puts · sell or buy · Heston Monte Carlo "
           "calibrated to option-implied vols, validated against closed forms")

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
    # pick up to 3 expiries closest to 0.5x, 1.0x, 1.5x tenor
    chosen = []
    for target in (0.5 * tenor, tenor, 1.5 * tenor):
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
            if len(df) > 7:                       # thin to ~7 per side
                df = df.iloc[np.linspace(0, len(df) - 1, 7).astype(int)]
            for _, row in df.iterrows():
                quotes.append((float(T), float(row["strike"]),
                               float(row["impliedVolatility"])))
    if len(quotes) < 5:
        raise ValueError("Too few usable option quotes.")
    return quotes


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
                quotes = fetch_option_quotes(t, spots[t], tenor)
                x0 = estimate_heston_from_returns(lr[t].values)
                est = calibrate_heston(quotes, spots[t], r, divs.get(t, 0.0), x0)
                source = (f"option-implied · {est['n_quotes']} quotes · "
                          f"fit RMSE {est['rmse_iv']*100:.2f} vol pts")
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
<div class='warn-box'><b>Remaining model limitations.</b> With option-implied
calibration the single-name smiles are market-consistent, but the <i>correlation</i>
input is still historical — implied correlation from listed products typically
trades above realized, and correlation spikes in sell-offs. Constant rho between
the stocks, European exercise, flat rates, no credit/funding adjustments, and
discrete dividends approximated by a continuous yield. Educational tool, not
investment advice.</div>""", unsafe_allow_html=True)
