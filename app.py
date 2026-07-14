"""
Worst-of Autocallable / Barrier Reverse Convertible pricer.

Run with:   streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heston_engine import (AssetParams, ProductSpec, price_note,
                           estimate_heston_from_returns)

st.set_page_config(page_title="Structured Note Pricer", page_icon="📉",
                   layout="wide")

ACCENT = "#2a5d8f"
RED = "#c0392b"
GREEN = "#1e8e5a"
GRAY = "#8a8a85"

st.markdown("""
<style>

/* ---------- Main Layout ---------- */

.block-container{
    max-width:1280px;
    padding-top:2rem;
    padding-bottom:2rem;
}

/* ---------- Metric Cards ---------- */

div[data-testid="stMetric"]{
    background:rgba(127,127,127,0.10);
    backdrop-filter:blur(12px);
    border:1px solid rgba(127,127,127,0.25);
    border-radius:14px;
    padding:18px;
    box-shadow:0 2px 8px rgba(0,0,0,.18);
}

div[data-testid="stMetricLabel"]{
    font-size:0.95rem;
    font-weight:600;
}

div[data-testid="stMetricValue"]{
    font-size:2rem;
    font-weight:700;
}

div[data-testid="stMetricDelta"]{
    font-size:0.9rem;
}

/* ---------- Buttons ---------- */

.stButton>button{
    width:100%;
    border-radius:10px;
    font-weight:600;
}

/* ---------- Inputs ---------- */

.stSelectbox,
.stTextInput,
.stNumberInput{
    border-radius:10px;
}

/* ---------- Sidebar ---------- */

section[data-testid="stSidebar"]{
    border-right:1px solid rgba(127,127,127,.18);
}

/* ---------- Note Boxes ---------- */

.note-box{
    background:rgba(42,93,143,.12);
    border-left:5px solid #2a5d8f;
    padding:14px 18px;
    border-radius:10px;
    margin:10px 0 18px 0;
    font-size:0.95rem;
}

.warn-box{
    background:rgba(192,57,43,.12);
    border-left:5px solid #c0392b;
    padding:14px 18px;
    border-radius:10px;
    margin:10px 0 18px 0;
    font-size:0.95rem;
}

/* ---------- Tables ---------- */

[data-testid="stDataFrame"]{
    border-radius:12px;
    overflow:hidden;
    border:1px solid rgba(127,127,127,.2);
}

/* ---------- Expanders ---------- */

details{
    border-radius:12px;
    overflow:hidden;
}

/* ---------- Headers ---------- */

h1{
    font-weight:700;
    margin-bottom:.2rem;
}

h2,h3{
    margin-top:1rem;
    margin-bottom:.6rem;
}

/* ---------- Plotly ---------- */

.js-plotly-plot{
    border-radius:12px;
}

/* ---------- Remove White Backgrounds ---------- */

div[data-testid="stMarkdownContainer"]{
    background:transparent !important;
}

/* ---------- General Cards ---------- */

div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stMetric"]){
    margin-bottom:0.5rem;
}

</style>
""", unsafe_allow_html=True)

st.title("Worst-of autocallable note pricer")
st.caption("Multi-asset Heston · Monte Carlo · barrier reverse convertibles, "
           "autocallables and reverse convertibles on up to three underlyings")

# =============================================================================
# Market data
# =============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_data(tickers: tuple, lookback: str):
    import yfinance as yf
    data = yf.download(list(tickers), period=lookback, auto_adjust=True,
                       progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(tickers[0])
    data = data[list(tickers)].dropna()
    if len(data) < 80:
        raise ValueError("Not enough overlapping history for these tickers.")
    spots = data.iloc[-1]
    lr = np.log(data / data.shift(1)).dropna()
    corr = lr.corr().values
    divs = {}
    for t in tickers:
        try:
            y = yf.Ticker(t).info.get("dividendYield", 0.0) or 0.0
            divs[t] = float(y) if y < 1 else float(y) / 100.0
        except Exception:
            divs[t] = 0.0
    return data, spots, lr, corr, divs


with st.sidebar:
    st.header("Underlyings")
    n_assets = st.radio("Number of stocks", [1, 2, 3], index=2, horizontal=True)
    default_tk = ["AAPL", "NVDA", "TSLA"]
    tickers = []
    for i in range(n_assets):
        tickers.append(st.text_input(f"Ticker {i+1}", default_tk[i]).strip().upper())
    lookback = st.selectbox("History for calibration", ["1y", "2y", "3y", "5y"], index=1)
    use_live = st.toggle("Fetch live market data (yfinance)", value=True)

    st.header("Product terms")
    tenor = st.slider("Tenor (years)", 0.5, 5.0, 3.0, 0.25)
    freq = st.selectbox("Observation frequency", ["Monthly", "Quarterly", "Semi-annual", "Annual"], 1)
    obs_per_year = {"Monthly": 12, "Quarterly": 4, "Semi-annual": 2, "Annual": 1}[freq]
    coupon = st.slider("Coupon (% p.a.)", 0.0, 30.0, 10.0, 0.25) / 100
    coupon_barrier = st.slider("Coupon barrier (% of initial)", 40, 100, 70, 1) / 100
    guaranteed = st.toggle("Guaranteed coupon (ignore coupon barrier)", value=False)
    memory = st.toggle("Memory coupon", value=True, disabled=guaranteed)

    autocallable = st.toggle("Autocallable", value=True)
    ac_trigger = st.slider("Autocall trigger (% of initial)", 70, 120, 100, 1) / 100
    first_call = st.number_input("First callable observation #", 1, 20, 1)

    ki_barrier = st.slider("Knock-in barrier (% of initial)", 30, 100, 60, 1) / 100
    barrier_type = st.selectbox("Barrier observation",
                                ["Continuous (American)", "At maturity only (European)"])
    strike = st.slider("Downside strike (% of initial)", 50, 110, 100, 1) / 100

    st.header("Rates & simulation")
    r = st.number_input("Risk-free rate (% p.a.)", 0.0, 15.0, 4.0, 0.1) / 100
    n_paths = st.select_slider("Monte Carlo paths", [5000, 10000, 20000, 50000, 100000], 20000)
    seed = st.number_input("Random seed", 0, 10_000, 42)

# ---- get market data or manual fallback ------------------------------------

data_ok, err_msg = False, ""
if use_live:
    try:
        with st.spinner("Fetching market data…"):
            hist, spots, lr, corr, divs = fetch_market_data(tuple(tickers), lookback)
        data_ok = True
    except Exception as e:
        err_msg = str(e)

if not data_ok:
    if use_live:
        st.markdown(f"<div class='warn-box'><b>Live data unavailable</b> "
                    f"({err_msg or 'no connection'}). Enter market inputs manually below — "
                    f"the pricer works identically.</div>", unsafe_allow_html=True)
    with st.expander("Manual market inputs", expanded=True):
        cols = st.columns(n_assets)
        spots, vols, dys = {}, {}, {}
        for i, t in enumerate(tickers):
            with cols[i]:
                st.markdown(f"**{t}**")
                spots[t] = st.number_input(f"Spot {t}", 0.01, 1e6, 100.0, key=f"sp{i}")
                vols[t] = st.number_input(f"Vol {t} (% p.a.)", 1.0, 300.0, 30.0, key=f"vv{i}") / 100
                dys[t] = st.number_input(f"Div yield {t} (%)", 0.0, 20.0, 0.5, key=f"dy{i}") / 100
        rho_in = st.slider("Pairwise correlation (applied to all pairs)", -0.5, 1.0, 0.5, 0.05)
        corr = np.full((n_assets, n_assets), rho_in); np.fill_diagonal(corr, 1.0)
        spots = pd.Series(spots); divs = dys
        hist, lr = None, None

# ---- Heston parameters per asset --------------------------------------------

st.subheader("1 · Market data & Heston calibration")
st.markdown("<div class='note-box'>Heston parameters below are estimated from "
            "historical returns (rolling realized variance → v₀, θ; AR(1) fit → κ; "
            "residual noise → ξ; return/variance correlation → ρ). A trading desk "
            "would calibrate these to the option-implied vol surface instead — "
            "override any value to match implied levels for a more market-consistent "
            "price.</div>", unsafe_allow_html=True)

assets, prows = [], []
pcols = st.columns(n_assets)
for i, t in enumerate(tickers):
    if data_ok:
        est = estimate_heston_from_returns(lr[t].values)
    else:
        v = vols[t] ** 2
        est = {"v0": v, "theta": v, "kappa": 2.0, "xi": 0.6, "rho_sv": -0.6,
               "hist_vol": vols[t]}
    with pcols[i]:
        st.markdown(f"**{t}** · spot {spots[t]:,.2f}")
        v0 = st.number_input("v₀ (initial var)", 0.0001, 4.0, float(round(est["v0"], 4)),
                             format="%.4f", key=f"v0{i}")
        th = st.number_input("θ (long-run var)", 0.0001, 4.0, float(round(est["theta"], 4)),
                             format="%.4f", key=f"th{i}")
        ka = st.number_input("κ (mean reversion)", 0.05, 15.0, float(round(est["kappa"], 2)), key=f"ka{i}")
        xv = st.number_input("ξ (vol of vol)", 0.01, 3.0, float(round(est["xi"], 2)), key=f"xi{i}")
        rh = st.number_input("ρ (spot-vol corr)", -0.99, 0.5, float(round(est["rho_sv"], 2)), key=f"rh{i}")
        a = AssetParams(t, float(spots[t]), float(divs.get(t, 0.0)), v0, ka, th, xv, rh)
        assets.append(a)
        prows.append({"Asset": t, "Spot": f"{spots[t]:,.2f}",
                      "√v₀": f"{np.sqrt(v0):.1%}", "√θ": f"{np.sqrt(th):.1%}",
                      "Div yield": f"{divs.get(t, 0.0):.2%}",
                      "Feller 2κθ/ξ²": f"{a.feller_ratio():.2f}"})

c1, c2 = st.columns([1.2, 1])
with c1:
    st.dataframe(pd.DataFrame(prows), hide_index=True, use_container_width=True)
    if any(a.feller_ratio() < 1 for a in assets):
        st.caption("⚠ Feller ratio < 1 for at least one asset: variance can touch "
                   "zero (common for equities; the full-truncation scheme handles it).")
with c2:
    fig = go.Figure(go.Heatmap(z=corr, x=tickers, y=tickers, zmin=-1, zmax=1,
                    colorscale="RdBu", text=np.round(corr, 2), texttemplate="%{text}"))
    fig.update_layout(title="Spot correlation (historical)", height=260,
                      margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

if data_ok and hist is not None:
    norm = hist / hist.iloc[0] * 100
    fig = go.Figure()
    for t in tickers:
        fig.add_trace(go.Scatter(x=norm.index, y=norm[t], name=t, line=dict(width=1.6)))
    fig.update_layout(title=f"Normalised price history ({lookback})", height=280,
                      margin=dict(l=10, r=10, t=40, b=10), yaxis_title="Rebased to 100")
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# Pricing
# =============================================================================

spec = ProductSpec(
    notional=1000.0, tenor_years=tenor, obs_per_year=obs_per_year,
    coupon_rate_pa=coupon, coupon_barrier=coupon_barrier,
    autocall_trigger=ac_trigger if autocallable else 10.0,
    first_callable_obs=int(first_call),
    ki_barrier=ki_barrier,
    barrier_type="continuous" if barrier_type.startswith("Cont") else "european",
    strike_pct=strike, memory_coupon=memory, guaranteed_coupon=guaranteed, r=r)

st.subheader("2 · Pricing results")
run = st.button("▶ Price the note", type="primary", use_container_width=True)
if not run:
    st.info("Set your terms in the sidebar, adjust Heston parameters if desired, "
            "then press **Price the note**.")
    st.stop()

with st.spinner(f"Simulating {n_paths:,} Heston paths…"):
    res = price_note(assets, corr, spec, n_paths=int(n_paths), seed=int(seed))

fair = res.price
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Fair value (per 100)", f"{fair:.2f}", f"± {2*res.stderr:.2f} (95% CI)")
m2.metric("Issuer margin at par", f"{100-fair:+.2f}",
          "you pay 100" if fair < 100 else "cheap at par", delta_color="inverse")
m3.metric("P(any capital loss)", f"{res.prob_capital_loss:.1%}")
m4.metric("Avg loss when it hits", f"−{res.exp_loss_given_loss:.1f}%")
m5.metric("Expected life", f"{res.exp_life_years:.2f} y",
          f"of {tenor:.2f} y max")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("P(autocalled early)", f"{res.prob_autocall.sum():.1%}")
m2.metric("P(knock-in)", f"{res.prob_ki:.1%}")
m3.metric("Expected coupons (per 100)", f"{res.exp_coupons_per100:.2f}")
m4.metric("Expected total return", f"{res.exp_total_return:+.2%}")
m5.metric("5% worst outcomes avg (CVaR)", f"{res.cvar_95:.1f} per 100")

if fair < 100:
    st.markdown(f"<div class='warn-box'>At an issue price of 100, this note is worth "
                f"<b>{fair:.2f}</b> — an embedded cost of <b>{100-fair:.2f}%</b> "
                f"(covers issuer hedging profit, distribution fees and credit spread). "
                f"Typical street levels are 1–3%. A fair coupon for this structure at "
                f"these model inputs would be higher than {coupon:.2%}.</div>",
                unsafe_allow_html=True)
else:
    st.markdown(f"<div class='note-box'>Model value <b>{fair:.2f}</b> ≥ 100: at these "
                f"inputs the terms are generous relative to the risk (or your Heston "
                f"parameters understate volatility vs option-implied levels).</div>",
                unsafe_allow_html=True)

# ---- charts ------------------------------------------------------------------

cA, cB = st.columns(2)

with cA:
    fig = go.Figure(go.Histogram(x=res.payoff_dist, nbinsx=80,
                                 marker_color=ACCENT, opacity=0.85))
    fig.add_vline(x=100, line_dash="dash", line_color=GRAY,
                  annotation_text="capital", annotation_position="top")
    fig.add_vline(x=res.payoff_dist.mean(), line_color=GREEN,
                  annotation_text=f"mean {res.payoff_dist.mean():.1f}")
    fig.update_layout(title="Distribution of total cash received (per 100, undiscounted)",
                      height=330, margin=dict(l=10, r=10, t=45, b=10),
                      xaxis_title="Total coupons + redemption", yaxis_title="Paths")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The tall spike is the 'nothing bad happened' cluster — capital plus "
               "coupons. The left tail is what you sold to earn them.")

with cB:
    labels = [f"Obs {k+1}" for k in range(len(res.prob_autocall))]
    surv = res.prob_survive_to_maturity
    fig = go.Figure(go.Bar(x=labels + ["Maturity"],
                           y=list(res.prob_autocall * 100) + [surv * 100],
                           marker_color=[ACCENT] * len(labels) + [GRAY]))
    fig.update_layout(title="When does the note end? (autocall by date, %)",
                      height=330, margin=dict(l=10, r=10, t=45, b=10),
                      yaxis_title="% of paths")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Autocallables tend to die early in good markets and survive to "
               "maturity in bad ones — you keep the risk longest when you least want it.")

cA, cB = st.columns(2)

with cA:
    fig = go.Figure()
    idx = np.arange(res.sample_paths.shape[0])
    for i in idx[:120]:
        stop = res.sample_exit_step[i]
        col = RED if res.sample_paths[i, min(stop, res.sample_paths.shape[1]-1)] < ki_barrier \
            else "rgba(42,93,143,0.25)"
        fig.add_trace(go.Scatter(x=res.time_grid[:stop+1],
                                 y=res.sample_paths[i, :stop+1] * 100,
                                 mode="lines", line=dict(width=0.8, color=col),
                                 showlegend=False, hoverinfo="skip"))
    for lvl, nm, cc in [(ki_barrier, "KI barrier", RED),
                        (coupon_barrier, "coupon barrier", "#b8860b"),
                        (ac_trigger if autocallable else None, "autocall", GREEN)]:
        if lvl:
            fig.add_hline(y=lvl * 100, line_dash="dot", line_color=cc,
                          annotation_text=nm, annotation_position="right")
    fig.update_layout(title="Sample worst-of paths (red = knocked in)",
                      height=340, margin=dict(l=10, r=10, t=45, b=10),
                      xaxis_title="Years", yaxis_title="Worst-of level (% of initial)")
    st.plotly_chart(fig, use_container_width=True)

with cB:
    x = np.linspace(0.2, 1.5, 200)
    n_c = int(round(tenor * obs_per_year))
    full_c = res.coupon_per_period * n_c
    note = np.where(x >= ki_barrier, 100.0, np.minimum(1, x / strike) * 100)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x*100, y=note + full_c, name="Note (all coupons paid)",
                             line=dict(color=ACCENT, width=2.5)))
    fig.add_trace(go.Scatter(x=x*100, y=x*100, name="Direct worst-of stock",
                             line=dict(color=GRAY, dash="dash")))
    fig.add_vline(x=ki_barrier*100, line_dash="dot", line_color=RED,
                  annotation_text="barrier")
    fig.update_layout(title="Payoff at maturity vs final worst-of (illustrative)",
                      height=340, margin=dict(l=10, r=10, t=45, b=10),
                      xaxis_title="Final worst-of (% of initial)",
                      yaxis_title="Total per 100", legend=dict(x=0.02, y=0.98))
    st.plotly_chart(fig, use_container_width=True)

# ---- product summary ----------------------------------------------------------

st.subheader("3 · Term-sheet summary")
bt = "continuously observed (American)" if spec.barrier_type == "continuous" \
     else "observed at final valuation only (European)"
ac_txt = (f"Callable from observation {int(first_call)} if the worst performer closes at or above "
          f"{ac_trigger:.0%} of its initial level — redeems at 100% plus the due coupon."
          if autocallable else "Not autocallable.")
mem_txt = "with memory (missed coupons caught up later)" if memory and not guaranteed else \
          ("guaranteed" if guaranteed else "without memory")
st.markdown(f"""
| Term | Value |
|---|---|
| Underlyings | {" / ".join(tickers)} — payoff on the **worst performer** |
| Tenor | {tenor:.2f} years, {freq.lower()} observations |
| Coupon | {coupon:.2%} p.a. ({res.coupon_per_period:.3f} per 100 per period), paid if worst-of ≥ {coupon_barrier:.0%}, {mem_txt} |
| Autocall | {ac_txt} |
| Knock-in barrier | {ki_barrier:.0%} of initial, {bt} |
| Downside | If knocked in and worst-of finishes below {strike:.0%}: redemption = worst-of / {strike:.0%} (1:1 loss from the strike, not the barrier) |
| Discounting | Flat {r:.2%} risk-free; **issuer credit risk not included** |
""")

st.markdown("""
<div class='warn-box'><b>Model limitations — read before trusting the number.</b>
Heston parameters here are estimated from <i>historical</i> returns; dealers calibrate to the
option-implied surface, which usually carries higher vol and steeper skew — historical
calibration therefore tends to <b>overstate</b> the note's value (understate its risk).
Constant correlation is assumed (real correlations spike in crashes, hurting worst-of notes).
No issuer credit spread, no discrete dividends, flat rates, daily-step Euler discretisation
with Brownian-bridge barrier correction. This is an educational tool, not investment advice
or a substitute for a dealer valuation.</div>
""", unsafe_allow_html=True)
