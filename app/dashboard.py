"""
Options Pricing Engine — Streamlit Dashboard v2
Run with: streamlit run app/dashboard.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime

from options_pricer.core import black_scholes as bs
from options_pricer.core import binomial_tree as bt
from options_pricer.core import monte_carlo as mc
from options_pricer.data.market_data import get_stock_data, get_risk_free_rate, time_to_expiry
from ticker_search import search_tickers
from streamlit_searchbox import st_searchbox

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Options Pricing Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        border-radius: 6px 6px 0 0;
        font-weight: 500;
    }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 12px 16px;
    }
    .section-header {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.45);
        margin-bottom: 8px;
    }
    .greek-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 14px 16px;
        text-align: center;
    }
    .info-box {
        background: rgba(99,110,250,0.08);
        border: 1px solid rgba(99,110,250,0.25);
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.88rem;
        line-height: 1.6;
    }
    .interp-box {
        background: rgba(0,204,150,0.07);
        border: 1px solid rgba(0,204,150,0.2);
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 0.86rem;
        line-height: 1.6;
        margin-top: 8px;
    }
    .badge-itm { background: rgba(38,194,129,0.15); border: 1px solid #26C281; color: #26C281; padding: 2px 10px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; }
    .badge-otm { background: rgba(231,76,60,0.12);  border: 1px solid #E74C3C; color: #E74C3C; padding: 2px 10px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; }
    .badge-atm { background: rgba(255,161,90,0.12); border: 1px solid #FFA15A; color: #FFA15A; padding: 2px 10px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; }
    .iv-result { background: rgba(171,99,250,0.1); border: 1px solid rgba(171,99,250,0.3); border-radius: 8px; padding: 14px 18px; margin-top: 12px; }
    .help-formula { background: rgba(0,0,0,0.3); border-radius: 6px; padding: 10px 14px; font-family: monospace; font-size: 0.9rem; margin: 8px 0; }
    .chain-above { color: #26C281; }
    .chain-below { color: #E74C3C; }
    .fetch-time { font-size: 0.72rem; color: rgba(255,255,255,0.35); margin-top: 4px; }

    /* Smooth draw-in animation for all Plotly line traces */
    .stPlotlyChart svg .lines path,
    .stPlotlyChart svg .scatter path {
        stroke-dasharray: 4000;
        stroke-dashoffset: 4000;
        animation: drawLine 3.5s cubic-bezier(0.25, 0.46, 0.45, 0.94) infinite alternate;
    }
    .stPlotlyChart svg .lines path:nth-child(1) { animation-delay: 0s; }
    .stPlotlyChart svg .lines path:nth-child(2) { animation-delay: 0.2s; }
    .stPlotlyChart svg .lines path:nth-child(3) { animation-delay: 0.4s; }
    .stPlotlyChart svg .lines path:nth-child(4) { animation-delay: 0.6s; }
    .stPlotlyChart svg .lines path:nth-child(5) { animation-delay: 0.8s; }
    .stPlotlyChart svg .lines path:nth-child(6) { animation-delay: 1.0s; }
    @keyframes drawLine {
        0%   { stroke-dashoffset: 4000; }
        80%  { stroke-dashoffset: 0; }
        100% { stroke-dashoffset: 0; }
    }

    /* Fade-in for bar charts */
    .stPlotlyChart svg .bars .point path {
        opacity: 0;
        animation: fadeInBar 0.9s ease forwards;
    }
    .stPlotlyChart svg .bars .point:nth-child(1) path { animation-delay: 0.0s; }
    .stPlotlyChart svg .bars .point:nth-child(2) path { animation-delay: 0.25s; }
    .stPlotlyChart svg .bars .point:nth-child(3) path { animation-delay: 0.50s; }
    @keyframes fadeInBar {
        from { opacity: 0; transform: scaleY(0); transform-origin: bottom; }
        to   { opacity: 1; transform: scaleY(1); }
    }

    /* Dot markers fade in */
    .stPlotlyChart svg .scatter .point path {
        opacity: 0;
        animation: fadeIn 0.6s ease 3.0s forwards;
    }
    @keyframes fadeIn {
        to { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# ─── IV Solver ────────────────────────────────────────────────────────────────

def implied_volatility(market_price, S, K, T, r, option_type, tol=1e-6, max_iter=200):
    """Solve for IV using Brent's method — robust bracketed root-finding."""
    if T <= 0:
        return None
    intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
    if market_price < intrinsic - tol:
        return None  # price below intrinsic — arbitrage
    try:
        def objective(sigma):
            return bs.price(S, K, T, r, sigma, option_type) - market_price
        iv = brentq(objective, 1e-6, 10.0, xtol=tol, maxiter=max_iter)
        return iv
    except (ValueError, RuntimeError):
        return None

# ─── Sidebar ──────────────────────────────────────────────────────────────────

def _ticker_search_fn(query: str):
    """Search function for st_searchbox — returns (label, value) tuples."""
    if not query or len(query.strip()) < 1:
        return []
    results = search_tickers(query, max_results=8)
    return [(f"{r['symbol']} — {r['name']}", r["symbol"]) for r in results]

def _do_fetch(symbol: str):
    """Fetch stock data and store in session state."""
    if not symbol:
        return
    symbol = symbol.strip().upper()
    with st.spinner(f"Fetching {symbol}..."):
        try:
            data = get_stock_data(symbol)
            st.session_state["live_data"]   = data
            st.session_state["live_ticker"] = symbol
            st.session_state["fetch_time"]  = datetime.now().strftime("%H:%M:%S")
        except Exception as e:
            st.session_state["fetch_error"] = str(e)

with st.sidebar:
    st.markdown("## ⚙️ Parameters")

    # ── Market Data ──────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Market Data</p>', unsafe_allow_html=True)
    use_live = st.toggle("Fetch live data", value=False)

    if use_live:
        # Live-search-as-you-type searchbox
        selected_ticker = st_searchbox(
            _ticker_search_fn,
            placeholder="Search ticker or company…",
            label="",
            key="ticker_searchbox",
            clear_on_submit=False,
            rerun_on_update=True,
        )

        # Auto-fetch when a suggestion is selected
        if selected_ticker and selected_ticker != st.session_state.get("last_fetched"):
            _do_fetch(selected_ticker)
            st.session_state["last_fetched"] = selected_ticker
            st.rerun()

        # Quick pick buttons
        st.markdown('<p class="section-header" style="margin-top:8px">Quick Pick</p>', unsafe_allow_html=True)
        qp_cols = st.columns(4)
        for i, qt in enumerate(["AAPL", "TSLA", "SPY", "NVDA"]):
            if qp_cols[i].button(qt, use_container_width=True, key=f"qp_{qt}"):
                _do_fetch(qt)
                st.session_state["last_fetched"] = qt
                st.rerun()

        # Show error if any
        if "fetch_error" in st.session_state:
            st.error(f"Could not fetch: {st.session_state.pop('fetch_error')}")

        # Show current loaded ticker info
        if st.session_state.get("live_ticker"):
            live_d = st.session_state.get("live_data", {})
            st.success(
                f"**{st.session_state['live_ticker']}** — "
                f"${live_d.get('current_price', 0):.2f}  |  "
                f"Vol(30d): {live_d.get('hist_volatility_30d', 0):.1%}"
            )
            if "fetch_time" in st.session_state:
                st.markdown(
                    f'<p class="fetch-time">Last fetched: {st.session_state["fetch_time"]}</p>',
                    unsafe_allow_html=True,
                )

    st.divider()
    st.markdown('<p class="section-header">Option Parameters</p>', unsafe_allow_html=True)

    live = st.session_state.get("live_data", {})
    default_S     = live.get("current_price", 100.0)
    default_sigma = live.get("hist_volatility_30d", 0.25)

    S = st.number_input("Stock Price (S) $", value=float(default_S), min_value=0.01, step=1.0,
                        help="Current market price of the underlying stock")
    K = st.number_input("Strike Price (K) $", value=float(round(default_S, 0)), min_value=0.01, step=1.0,
                        help="The price at which the option can be exercised")
    T_days = st.slider("Days to Expiry", min_value=1, max_value=730, value=90,
                       help="Calendar days until expiry")
    T = T_days / 365

    option_type = st.radio("Option Type", ["call", "put"], horizontal=True,
                           help="Call = right to buy | Put = right to sell")

    st.divider()
    st.markdown('<p class="section-header">Market Parameters</p>', unsafe_allow_html=True)

    r_pct = st.slider("Risk-Free Rate (%)", min_value=0.0, max_value=15.0, value=5.0, step=0.25,
                      help="Annualised risk-free rate (e.g. US T-bill yield)")
    r = r_pct / 100

    sigma_pct = st.slider("Volatility / IV (%)", min_value=1.0, max_value=150.0,
                          value=float(round(default_sigma * 100, 1)), step=0.5,
                          help="Annualised volatility. Use historical vol as a proxy.")
    sigma = sigma_pct / 100

    st.divider()
    st.markdown('<p class="section-header">Model Settings</p>', unsafe_allow_html=True)

    bt_steps = st.slider("Binomial Steps", 50, 500, 200, step=50,
                         help="More steps = more accurate, slower")
    mc_sims = st.select_slider("MC Simulations", [10_000, 50_000, 100_000, 500_000], value=100_000,
                               help="More simulations = lower standard error")

# ─── Header ───────────────────────────────────────────────────────────────────

ticker_label = st.session_state.get("live_ticker", "")
title_suffix = f" — {ticker_label}" if ticker_label else ""
st.markdown(f"## 📈 Options Pricing Engine{title_suffix}")

moneyness = S / K
is_itm = (moneyness > 1 and option_type == "call") or (moneyness < 1 and option_type == "put")
is_atm = abs(moneyness - 1) < 0.005
if is_atm:
    badge = '<span class="badge-atm">At the Money</span>'
elif is_itm:
    badge = '<span class="badge-itm">In the Money</span>'
else:
    badge = '<span class="badge-otm">Out of the Money</span>'

opt_color = "#636EFA" if option_type == "call" else "#EF553B"
opt_bg    = "rgba(99,110,250,0.12)" if option_type == "call" else "rgba(231,76,60,0.12)"
opt_border = "rgba(99,110,250,0.35)" if option_type == "call" else "rgba(231,76,60,0.35)"

info_html = (
    "<div style='display:flex;align-items:center;gap:14px;flex-wrap:wrap;padding:4px 0'>"
    f"<span style='background:{opt_bg};border:1px solid {opt_border};border-radius:6px;"
    f"padding:3px 12px;font-weight:700;color:{opt_color};font-size:0.85rem'>{option_type.upper()}</span>"
    "<span style='color:rgba(255,255,255,0.35)'>|</span>"
    f"<span><span style='color:rgba(255,255,255,0.4);font-size:0.85rem'>Stock&nbsp;</span>"
    f"<strong>${S:.2f}</strong></span>"
    "<span style='color:rgba(255,255,255,0.2)'>·</span>"
    f"<span><span style='color:rgba(255,255,255,0.4);font-size:0.85rem'>Strike&nbsp;</span>"
    f"<strong>${K:.2f}</strong></span>"
    "<span style='color:rgba(255,255,255,0.2)'>·</span>"
    f"<span><span style='color:rgba(255,255,255,0.4);font-size:0.85rem'>Expiry&nbsp;</span>"
    f"<strong>{T_days} days</strong></span>"
    "<span style='color:rgba(255,255,255,0.2)'>·</span>"
    f"<span><span style='color:rgba(255,255,255,0.4);font-size:0.85rem'>Vol&nbsp;</span>"
    f"<strong>{sigma_pct:.1f}%</strong></span>"
    "<span style='color:rgba(255,255,255,0.2)'>·</span>"
    f"<span><span style='color:rgba(255,255,255,0.4);font-size:0.85rem'>Rate&nbsp;</span>"
    f"<strong>{r_pct:.1f}%</strong></span>"
    f"&nbsp;&nbsp;{badge}"
    "</div>"
)
st.markdown(info_html, unsafe_allow_html=True)

# ─── Stock Overview Card ───────────────────────────────────────────────────────

if ticker_label and use_live:
    live_data = st.session_state.get("live_data", {})
    hist = live_data.get("history")

    if hist is not None and not hist.empty:
        import yfinance as yf

        # Key price stats
        current_price = live_data.get("current_price", S)
        prev_close    = float(hist["Close"].iloc[-2]) if len(hist) > 1 else current_price
        day_change    = current_price - prev_close
        day_change_pct = (day_change / prev_close) * 100 if prev_close else 0
        is_up = day_change >= 0
        chg_color = "#26C281" if is_up else "#E74C3C"
        chg_arrow = "▲" if is_up else "▼"

        day_high  = float(hist["High"].iloc[-1])
        day_low   = float(hist["Low"].iloc[-1])
        vol_30    = live_data.get("hist_volatility_30d", 0)
        vol_252   = live_data.get("hist_volatility_252d", 0)

        # Try to get extra info
        try:
            tkr_info = yf.Ticker(ticker_label).info
            mkt_cap  = tkr_info.get("marketCap")
            pe_ratio = tkr_info.get("trailingPE")
            week_52_high = tkr_info.get("fiftyTwoWeekHigh")
            week_52_low  = tkr_info.get("fiftyTwoWeekLow")
            company_name = tkr_info.get("shortName", ticker_label)
            sector       = tkr_info.get("sector", "")
        except Exception:
            mkt_cap = pe_ratio = week_52_high = week_52_low = None
            company_name = ticker_label
            sector = ""

        def fmt_cap(v):
            if v is None: return "—"
            if v >= 1e12: return f"${v/1e12:.2f}T"
            if v >= 1e9:  return f"${v/1e9:.2f}B"
            if v >= 1e6:  return f"${v/1e6:.2f}M"
            return f"${v:,.0f}"

        # ── Layout: price block left, chart right ──────────────────────────
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        ov_left, ov_right = st.columns([1, 2.6])

        with ov_left:
            st.markdown(
                f"""<div style="padding:4px 0 12px 0">
                    <div style="font-size:0.85rem;color:rgba(255,255,255,0.5);font-weight:600;letter-spacing:0.05em">{company_name.upper()}{(" · " + sector) if sector else ""}</div>
                    <div style="font-size:2.6rem;font-weight:800;line-height:1.15;margin:4px 0">${current_price:.2f}</div>
                    <div style="font-size:1.05rem;font-weight:600;color:{chg_color}">
                        {chg_arrow} ${abs(day_change):.2f} ({abs(day_change_pct):.2f}%) today
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

            # Stats grid
            def stat_row(label, value):
                return (
                    f"<div style='display:flex;justify-content:space-between;"
                    f"padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.06)'>"
                    f"<span style='color:rgba(255,255,255,0.45);font-size:0.82rem'>{label}</span>"
                    f"<span style='font-size:0.82rem;font-weight:600'>{value}</span></div>"
                )

            stats_html = "".join([
                stat_row("Open",        f"${float(hist['Open'].iloc[-1]):.2f}"),
                stat_row("Day High",    f"${day_high:.2f}"),
                stat_row("Day Low",     f"${day_low:.2f}"),
                stat_row("Prev Close",  f"${prev_close:.2f}"),
                stat_row("52w High",    f"${week_52_high:.2f}" if week_52_high else "—"),
                stat_row("52w Low",     f"${week_52_low:.2f}"  if week_52_low  else "—"),
                stat_row("Mkt Cap",     fmt_cap(mkt_cap)),
                stat_row("P/E Ratio",   f"{pe_ratio:.1f}" if pe_ratio else "—"),
                stat_row("Vol (30d)",   f"{vol_30:.1%}"),
                stat_row("Vol (252d)",  f"{vol_252:.1%}"),
            ])
            st.markdown(f"<div style='margin-top:8px'>{stats_html}</div>", unsafe_allow_html=True)

        with ov_right:
            # Timeframe selector
            tf_options = {"1D": "1d", "5D": "5d", "1M": "1mo", "6M": "6mo", "1Y": "1y", "5Y": "5y"}
            tf_cols = st.columns(len(tf_options))
            selected_tf = st.session_state.get("chart_tf", "6M")
            for i, (label_tf, period) in enumerate(tf_options.items()):
                style = "primary" if label_tf == selected_tf else "secondary"
                if tf_cols[i].button(label_tf, key=f"tf_{label_tf}", type=style, use_container_width=True):
                    st.session_state["chart_tf"] = label_tf
                    st.rerun()

            # Fetch chart data for selected timeframe
            tf_period = tf_options[selected_tf]
            try:
                chart_hist = yf.Ticker(ticker_label).history(period=tf_period)
            except Exception:
                chart_hist = hist

            if chart_hist is not None and not chart_hist.empty:
                chart_close  = chart_hist["Close"]
                chart_open   = chart_hist["Open"]
                chart_high   = chart_hist["High"]
                chart_low    = chart_hist["Low"]
                chart_volume = chart_hist["Volume"]
                chart_dates  = chart_hist.index

                first_price = float(chart_close.iloc[0])
                last_price  = float(chart_close.iloc[-1])
                chart_up    = last_price >= first_price
                line_color  = "#26C281" if chart_up else "#E74C3C"
                fill_color  = "rgba(38,194,129,0.08)" if chart_up else "rgba(231,76,60,0.08)"

                use_candles = selected_tf in ("1D", "5D")

                fig_stock = go.Figure()

                if use_candles:
                    fig_stock.add_trace(go.Candlestick(
                        x=chart_dates,
                        open=chart_open, high=chart_high,
                        low=chart_low,   close=chart_close,
                        increasing_line_color="#26C281",
                        decreasing_line_color="#E74C3C",
                        name=ticker_label,
                        showlegend=False,
                    ))
                else:
                    fig_stock.add_trace(go.Scatter(
                        x=chart_dates, y=chart_close,
                        mode="lines",
                        line=dict(color=line_color, width=2),
                        fill="tozeroy",
                        fillcolor=fill_color,
                        name=ticker_label,
                        showlegend=False,
                        hovertemplate="<b>%{x|%b %d, %Y}</b><br>$%{y:.2f}<extra></extra>",
                    ))

                # Strike line on chart
                if K >= float(chart_low.min()) and K <= float(chart_high.max()):
                    fig_stock.add_hline(
                        y=K,
                        line_dash="dot",
                        line_color="rgba(255,161,90,0.6)",
                        annotation_text=f"Strike K=${K:.2f}",
                        annotation_font_size=10,
                        annotation_font_color="rgba(255,161,90,0.8)",
                    )

                fig_stock.update_layout(
                    height=280,
                    margin=dict(t=8, b=8, l=0, r=0),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(
                        showgrid=False,
                        showticklabels=True,
                        tickfont=dict(size=10, color="rgba(255,255,255,0.4)"),
                        rangeslider=dict(visible=False),
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor="rgba(255,255,255,0.05)",
                        tickfont=dict(size=10, color="rgba(255,255,255,0.4)"),
                        tickprefix="$",
                        side="right",
                    ),
                    hovermode="x unified",
                )

                st.plotly_chart(fig_stock, use_container_width=True, config={
                    "displayModeBar": True,
                    "modeBarButtonsToRemove": ["autoScale2d", "lasso2d", "select2d"],
                    "displaylogo": False,
                })

        st.divider()

st.divider()

# ─── Compute prices ───────────────────────────────────────────────────────────

bs_price  = bs.price(S, K, T, r, sigma, option_type)
bs_greeks = bs.greeks(S, K, T, r, sigma, option_type)
bt_price  = bt.price(S, K, T, r, sigma, option_type, style="european", steps=bt_steps)
mc_result = mc.price(S, K, T, r, sigma, option_type, n_simulations=mc_sims)

intrinsic   = max(float(S - K), 0.0) if option_type == "call" else max(float(K - S), 0.0)
time_value  = max(bs_price - intrinsic, 0.0)

# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "💰 Pricing", "🔢 Greeks", "🌲 Binomial Tree",
    "🎲 Monte Carlo", "📊 Sensitivity", "🔭 IV Solver", "❓ Help"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PRICING
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Black-Scholes** *(Analytical)*")
        st.metric("Price", f"${bs_price:.4f}")
        st.caption("Closed-form formula. Assumes constant vol, no dividends, European exercise only.")

    with col2:
        diff_bt = bt_price - bs_price
        st.markdown(f"**Binomial Tree** *({bt_steps} steps, CRR)*")
        st.metric("Price", f"${bt_price:.4f}", delta=f"{diff_bt:+.4f} vs BS")
        st.caption("Lattice model. Converges to Black-Scholes as steps → ∞. Supports American exercise.")

    with col3:
        diff_mc = mc_result["price"] - bs_price
        ci = mc_result["confidence_interval"]
        st.markdown(f"**Monte Carlo** *({mc_sims:,} paths)*")
        st.metric("Price", f"${mc_result['price']:.4f}", delta=f"{diff_mc:+.4f} vs BS")
        st.caption(f"GBM with antithetic variates. 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}] | SE: {mc_result['std_error']:.5f}")

    st.divider()

    # Intrinsic vs time value
    st.markdown("#### Price Decomposition")
    d1, d2, d3 = st.columns(3)
    d1.metric("Total Premium", f"${bs_price:.4f}", help="What you pay for the option")
    d2.metric("Intrinsic Value", f"${intrinsic:.4f}",
              help="What you'd get if you exercised right now: max(S−K, 0) for calls")
    d3.metric("Time Value", f"${time_value:.4f}",
              help="The extra amount above intrinsic — reflects uncertainty and time remaining")

    # Decomposition bar
    if bs_price > 0:
        fig_decomp = go.Figure()
        fig_decomp.add_trace(go.Bar(
            name="Intrinsic Value",
            x=["Option Price"],
            y=[intrinsic],
            marker_color="#26C281",
            text=f"${intrinsic:.4f}",
            textposition="inside",
        ))
        fig_decomp.add_trace(go.Bar(
            name="Time Value",
            x=["Option Price"],
            y=[time_value],
            marker_color="#636EFA",
            text=f"${time_value:.4f}",
            textposition="inside",
        ))
        fig_decomp.update_layout(
            barmode="stack",
            height=180,
            margin=dict(t=10, b=10, l=10, r=10),
            showlegend=True,
            legend=dict(orientation="h", y=1.3),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showticklabels=False),
            yaxis_title="Price ($)",
        )
        st.plotly_chart(fig_decomp, use_container_width=True)

    st.divider()

    # Model comparison bar chart
    fig_compare = go.Figure()
    models = ["Black-Scholes", f"Binomial ({bt_steps} steps)", f"Monte Carlo ({mc_sims//1000}k)"]
    prices = [bs_price, bt_price, mc_result["price"]]
    colors = ["#636EFA", "#EF553B", "#00CC96"]

    fig_compare.add_trace(go.Bar(
        x=models, y=prices,
        marker_color=colors,
        text=[f"${p:.4f}" for p in prices],
        textposition="outside",
        width=0.4,
    ))
    fig_compare.add_hline(y=bs_price, line_dash="dash",
                          line_color="rgba(255,255,255,0.25)",
                          annotation_text="BS benchmark", annotation_font_size=11)
    fig_compare.update_layout(
        title=dict(text="Model Price Comparison", font=dict(size=15)),
        yaxis_title="Option Price ($)",
        yaxis=dict(range=[min(prices) * 0.9, max(prices) * 1.15]),
        showlegend=False,
        height=320,
        margin=dict(t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    with st.expander("🔍 Put-Call Parity Verification"):
        call_p = bs.price(S, K, T, r, sigma, "call")
        put_p  = bs.price(S, K, T, r, sigma, "put")
        lhs = call_p - put_p
        rhs = S - K * np.exp(-r * T)
        err = abs(lhs - rhs)
        c1, c2, c3 = st.columns(3)
        c1.metric("C − P", f"${lhs:.8f}")
        c2.metric("S − Ke⁻ʳᵀ", f"${rhs:.8f}")
        c3.metric("Error", f"{err:.2e}", delta="✓ Machine precision" if err < 1e-10 else "⚠ Check")
        st.caption("Put-call parity: C − P = S − Ke^(−rT). Should hold to ~1e-14 for analytical BS.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GREEKS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Black-Scholes Greeks")
    st.caption("All Greeks computed analytically — no finite differences.")

    greek_meta = {
        "delta": ("Δ Delta",  "#636EFA", "Price sensitivity to $1 stock move"),
        "gamma": ("Γ Gamma",  "#00CC96", "Delta sensitivity to $1 stock move"),
        "vega":  ("ν Vega",   "#AB63FA", "Price sensitivity to 1% vol change"),
        "theta": ("Θ Theta",  "#FFA15A", "Price decay per calendar day"),
        "rho":   ("ρ Rho",    "#19D3F3", "Price sensitivity to 1% rate change"),
    }

    cols = st.columns(5)
    for i, (key, (symbol, color, desc)) in enumerate(greek_meta.items()):
        val = bs_greeks[key]
        val_str = f"{val:+.5f}"
        with cols[i]:
            st.markdown(
                f"""<div class="greek-card">
                    <div style="font-size:1.1rem;font-weight:700;color:{color}">{symbol}</div>
                    <div style="font-size:1.6rem;font-weight:800;color:{color};margin:6px 0">{val_str}</div>
                    <div style="font-size:0.75rem;color:rgba(255,255,255,0.5);line-height:1.4">{desc}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    # Dynamic interpretation
    delta_val = bs_greeks["delta"]
    gamma_val = bs_greeks["gamma"]
    vega_val  = bs_greeks["vega"]
    theta_val = bs_greeks["theta"]

    ticker_str = ticker_label if ticker_label else "the stock"
    direction  = "gains" if delta_val >= 0 else "loses"

    st.markdown(
        f"""<div class="interp-box">
        <b>What this means right now:</b><br>
        • A <b>Δ Delta of {delta_val:+.4f}</b> means if {ticker_str} moves +$1, your option {direction} ~<b>${abs(delta_val):.4f}</b>.<br>
        • A <b>Γ Gamma of {gamma_val:.5f}</b> means Delta itself shifts by {gamma_val:.5f} for each $1 stock move — {"high sensitivity near ATM." if abs(moneyness - 1) < 0.05 else "lower sensitivity at this moneyness."}<br>
        • A <b>ν Vega of {vega_val:+.4f}</b> means a 1% rise in volatility {"increases" if vega_val > 0 else "decreases"} your option value by ~<b>${abs(vega_val):.4f}</b>.<br>
        • A <b>Θ Theta of {theta_val:+.5f}</b> means you lose ~<b>${abs(theta_val):.5f}</b> of value per day just from time passing.
        </div>""",
        unsafe_allow_html=True,
    )

    st.divider()

    # Greeks charts
    spot_range = np.linspace(S * 0.5, S * 1.5, 100)
    greek_vals = {k: [] for k in ["delta", "gamma", "vega", "theta", "rho"]}
    bs_prices_range = []

    for s_val in spot_range:
        g = bs.greeks(s_val, K, T, r, sigma, option_type)
        bs_prices_range.append(bs.price(s_val, K, T, r, sigma, option_type))
        for k in greek_vals:
            greek_vals[k].append(g[k])

    plot_configs = [
        ("Option Price", bs_prices_range,    "#636EFA"),
        ("Δ Delta",      greek_vals["delta"],"#EF553B"),
        ("Γ Gamma",      greek_vals["gamma"],"#00CC96"),
        ("ν Vega",       greek_vals["vega"], "#AB63FA"),
        ("Θ Theta",      greek_vals["theta"],"#FFA15A"),
        ("ρ Rho",        greek_vals["rho"],  "#19D3F3"),
    ]

    fig_greeks = make_subplots(
        rows=3, cols=2,
        subplot_titles=[p[0] for p in plot_configs],
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )

    for idx, (title, y_vals, color) in enumerate(plot_configs):
        row = idx // 2 + 1
        col = idx % 2 + 1
        interp_y = float(np.interp(S, spot_range, y_vals))

        fig_greeks.add_trace(go.Scatter(
            x=spot_range, y=y_vals, name=title,
            line=dict(color=color, width=2), showlegend=False,
        ), row=row, col=col)

        fig_greeks.add_trace(go.Scatter(
            x=[S], y=[interp_y], mode="markers",
            marker=dict(color=color, size=9, symbol="circle",
                        line=dict(color="white", width=1.5)),
            showlegend=False,
            hovertemplate=f"S=${S:.2f}<br>{title}: {interp_y:.5f}<extra></extra>",
        ), row=row, col=col)

        fig_greeks.add_vline(x=K, line_dash="dot",
                             line_color="rgba(255,255,255,0.2)", row=row, col=col)

    fig_greeks.update_layout(
        height=750, showlegend=False,
        margin=dict(t=60, b=40, l=50, r=30),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig_greeks.update_xaxes(title_text="Stock Price ($)", title_font=dict(size=11))

    # Inject CSS draw-in animation for all SVG paths in this chart
    st.markdown("""
    <style>
    .stPlotlyChart svg .lines path {
        stroke-dasharray: 3000;
        stroke-dashoffset: 3000;
        animation: drawLine 1.6s cubic-bezier(0.4, 0, 0.2, 1) forwards;
    }
    @keyframes drawLine {
        to { stroke-dashoffset: 0; }
    }
    </style>
    """, unsafe_allow_html=True)
    st.plotly_chart(fig_greeks, use_container_width=True)
    st.caption(f"Dotted vertical line = strike K=${K:.2f}. Dot on each curve = current spot S=${S:.2f}.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BINOMIAL TREE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Binomial Tree (Cox-Ross-Rubinstein)")
    st.caption("A discrete lattice model. At each step the stock can move up or down. "
               "Option value is computed by backward induction from expiry.")

    euro_p   = bt.price(S, K, T, r, sigma, option_type, style="european", steps=bt_steps)
    amer_p   = bt.price(S, K, T, r, sigma, option_type, style="american", steps=bt_steps)
    premium  = amer_p - euro_p

    c1, c2, c3 = st.columns(3)
    c1.metric("European Price", f"${euro_p:.4f}")
    c2.metric("American Price", f"${amer_p:.4f}")
    c3.metric("Early Exercise Premium", f"${premium:.4f}",
              help="Extra value from being able to exercise before expiry. Always ≥ 0.")

    st.divider()

    step_counts = [5, 10, 25, 50, 100, 200, 300, 500]
    converged = [bt.price(S, K, T, r, sigma, option_type, steps=n) for n in step_counts]

    fig_conv = go.Figure()
    fig_conv.add_trace(go.Scatter(
        x=step_counts, y=converged, mode="lines+markers",
        name="Binomial Price",
        line=dict(color="#EF553B", width=2),
        marker=dict(size=6),
    ))
    fig_conv.add_hline(y=bs_price, line_dash="dash", line_color="#636EFA",
                       annotation_text=f"BS = ${bs_price:.4f}", annotation_font_size=11)
    fig_conv.update_layout(
        title="Convergence to Black-Scholes as Steps Increase",
        xaxis_title="Number of Steps",
        yaxis_title="Option Price ($)",
        height=320,
        margin=dict(t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_conv, use_container_width=True)

    st.markdown("#### Option Value at Each Node (10-step tree)")
    st.caption("Green = high value, Red = low/zero. Time flows left → right.")
    _, price_tree = bt.price_grid(S, K, T, r, sigma, option_type, "european", 10)
    mask   = np.tril(np.ones_like(price_tree, dtype=bool))
    masked = np.where(mask, price_tree, np.nan)

    fig_tree = go.Figure(data=go.Heatmap(
        z=masked,
        colorscale="RdYlGn",
        text=[[f"${v:.2f}" if not np.isnan(v) else "" for v in row] for row in masked],
        texttemplate="%{text}",
        textfont={"size": 9},
        colorbar=dict(title="Value ($)", thickness=15),
    ))
    fig_tree.update_layout(
        xaxis_title="Time Step",
        yaxis_title="Node",
        height=380,
        margin=dict(t=20, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_tree, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MONTE CARLO
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Monte Carlo Simulation")
    st.caption("Simulates thousands of stock price paths under GBM. "
               "Option price = discounted average payoff across all paths.")

    pct_itm = (mc.simulate_paths(S, T, r, sigma, n_paths=10000, n_steps=1)[1][-1] > K).mean() * 100 \
              if option_type == "call" else \
              (mc.simulate_paths(S, T, r, sigma, n_paths=10000, n_steps=1)[1][-1] < K).mean() * 100

    mc_c1, mc_c2, mc_c3, mc_c4, mc_c5 = st.columns(5)
    mc_c1.metric("MC Price",     f"${mc_result['price']:.4f}")
    mc_c2.metric("Std Error",    f"{mc_result['std_error']:.5f}")
    mc_c3.metric("95% CI Lower", f"${mc_result['confidence_interval'][0]:.4f}")
    mc_c4.metric("95% CI Upper", f"${mc_result['confidence_interval'][1]:.4f}")
    mc_c5.metric("Prob. of Profit", f"{pct_itm:.1f}%",
                 help="% of simulated paths that finish in the money")

    st.divider()

    times, paths = mc.simulate_paths(S, T, r, sigma, n_paths=100, n_steps=252)
    n_show = min(60, paths.shape[1])

    fig_paths = go.Figure()
    for i in range(n_show):
        alpha = 0.12 + 0.22 * (i / n_show)
        fig_paths.add_trace(go.Scatter(
            x=times * 365, y=paths[:, i], mode="lines",
            line=dict(width=0.7, color=f"rgba(99,110,250,{alpha:.2f})"),
            showlegend=False,
        ))
    fig_paths.add_hline(y=K, line_dash="dash", line_color="#EF553B",
                        annotation_text=f"Strike K=${K:.2f}", annotation_font_size=11)
    fig_paths.add_hline(y=S, line_dash="dot", line_color="rgba(255,255,255,0.4)",
                        annotation_text=f"S₀=${S:.2f}", annotation_font_size=11)
    fig_paths.update_layout(
        title="Simulated GBM Price Paths (60 shown)",
        xaxis_title="Days",
        yaxis_title="Stock Price ($)",
        height=340,
        margin=dict(t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_paths, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Terminal Price Distribution")
        _, term_paths = mc.simulate_paths(S, T, r, sigma, n_paths=mc_sims // 252, n_steps=1)
        ST_dist = term_paths[-1]

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=ST_dist, nbinsx=70,
            marker_color="rgba(99,110,250,0.65)",
            marker_line=dict(color="rgba(99,110,250,0.9)", width=0.3),
        ))
        fig_dist.add_vline(x=K, line_dash="dash", line_color="#EF553B",
                           annotation_text=f"K=${K:.2f}", annotation_font_size=11)
        fig_dist.add_vline(x=S, line_dash="dot", line_color="rgba(255,255,255,0.5)",
                           annotation_text=f"S₀=${S:.2f}", annotation_font_size=11)
        fig_dist.update_layout(
            xaxis_title="Stock Price at Expiry ($)",
            yaxis_title="Frequency",
            height=300,
            margin=dict(t=20, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        st.caption(f"**{pct_itm:.1f}%** of paths finish {'above' if option_type == 'call' else 'below'} the strike — these are the profitable scenarios.")

    with col2:
        st.markdown("#### Convergence vs Simulations")
        sim_counts = [500, 1_000, 5_000, 10_000, 50_000, 100_000]
        mc_conv = [mc.price(S, K, T, r, sigma, option_type, n_simulations=n)["price"] for n in sim_counts]

        fig_mc_conv = go.Figure()
        fig_mc_conv.add_trace(go.Scatter(
            x=sim_counts, y=mc_conv, mode="lines+markers",
            line=dict(color="#00CC96", width=2),
            marker=dict(size=6),
        ))
        fig_mc_conv.add_hline(y=bs_price, line_dash="dash", line_color="#636EFA",
                              annotation_text=f"BS=${bs_price:.4f}", annotation_font_size=11)
        fig_mc_conv.update_layout(
            xaxis_title="Number of Simulations",
            yaxis_title="Option Price ($)",
            xaxis_type="log",
            height=300,
            margin=dict(t=20, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_mc_conv, use_container_width=True)
        st.caption("Antithetic variates used to reduce variance without doubling compute.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### Sensitivity Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Price Surface — Volatility × Time")
        vol_range = np.linspace(0.05, 1.0, 40)
        t_range   = np.linspace(0.05, 2.0, 40)
        Z = np.array([[bs.price(S, K, t_v, r, v_v, option_type)
                       for v_v in vol_range] for t_v in t_range])

        # Build rotation frames — smooth 360° loop
        n_rot_frames = 120
        rot_frames = []
        for i in range(n_rot_frames):
            angle = (i / n_rot_frames) * 360
            rot_frames.append(go.Frame(
                layout=dict(scene_camera=dict(
                    eye=dict(
                        x=1.6 * np.cos(np.radians(angle)),
                        y=1.6 * np.sin(np.radians(angle)),
                        z=0.8,
                    )
                )),
                name=str(i),
            ))

        fig_surface = go.Figure(
            data=go.Surface(
                x=vol_range * 100, y=t_range * 365, z=Z,
                colorscale="Viridis",
                colorbar=dict(title="Price ($)", thickness=15),
                contours=dict(z=dict(show=True, usecolormap=True,
                                     highlightcolor="white", project_z=True)),
            ),
            frames=rot_frames,
        )
        fig_surface.update_layout(
            scene=dict(
                xaxis_title="Volatility (%)",
                yaxis_title="Days to Expiry",
                zaxis_title="Price ($)",
                bgcolor="rgba(0,0,0,0)",
                camera=dict(eye=dict(x=1.6, y=0.0, z=0.8)),
            ),
            height=430,
            margin=dict(t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                visible=False,   # hidden — auto-triggered by JS below
                buttons=[dict(
                    label="Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=80, redraw=False),
                        transition=dict(duration=80, easing="linear"),
                        fromcurrent=True,
                        mode="immediate",
                        loop=True,
                    )],
                )],
            )],
        )

        # Render chart then auto-click Play via JS
        st.plotly_chart(fig_surface, use_container_width=True, key="surface_chart")
        st.components.v1.html("""
        <script>
        (function() {
            function autoPlay() {
                // Find the hidden play button in the most recent Plotly chart and click it
                var btns = window.parent.document.querySelectorAll('.plotly .updatemenu-item-rect');
                if (btns.length > 0) {
                    btns[btns.length - 1].dispatchEvent(new MouseEvent('click', {bubbles: true}));
                } else {
                    setTimeout(autoPlay, 300);
                }
            }
            setTimeout(autoPlay, 800);
        })();
        </script>
        """, height=0)
        st.caption("Auto-rotating. Drag to take control, it will resume. Higher vol and longer expiry increase price.")

    with col2:
        st.markdown("#### P&L Diagram")
        spot_pnl = np.linspace(S * 0.5, S * 1.5, 300)
        payoffs_at_expiry, current_prices_pnl = [], []
        for s_val in spot_pnl:
            payoffs_at_expiry.append(
                float(max(s_val - K, 0) if option_type == "call" else max(K - s_val, 0)) - bs_price
            )
            current_prices_pnl.append(bs.price(float(s_val), K, T, r, sigma, option_type) - bs_price)

        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Scatter(
            x=spot_pnl, y=payoffs_at_expiry,
            name="At Expiry",
            line=dict(color="#EF553B", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(231,76,60,0.06)",
        ))
        fig_pnl.add_trace(go.Scatter(
            x=spot_pnl, y=current_prices_pnl,
            name=f"Now ({T_days}d remaining)",
            line=dict(color="#636EFA", width=2, dash="dash"),
        ))
        fig_pnl.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
        fig_pnl.add_vline(x=K, line_dash="dot", line_color="rgba(255,255,255,0.3)",
                          annotation_text=f"K=${K:.2f}", annotation_font_size=11)
        fig_pnl.add_vline(x=S, line_dash="dot", line_color="rgba(0,204,150,0.5)",
                          annotation_text=f"S=${S:.2f}", annotation_font_size=11)
        fig_pnl.update_layout(
            xaxis_title="Stock Price ($)",
            yaxis_title="Profit / Loss ($)",
            legend=dict(orientation="h", y=1.05),
            height=430,
            margin=dict(t=40, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_pnl, use_container_width=True)
        breakeven = K + bs_price if option_type == "call" else K - bs_price
        st.caption(f"Break-even at expiry: **${breakeven:.2f}** | Max loss: **${bs_price:.4f}** (premium paid)")

    st.divider()
    st.markdown("#### Scenario Analysis")

    scenarios = {
        "Deep OTM": S * 0.7 if option_type == "call" else S * 1.3,
        "OTM":      S * 0.9 if option_type == "call" else S * 1.1,
        "ATM":      S,
        "ITM":      S * 1.1 if option_type == "call" else S * 0.9,
        "Deep ITM": S * 1.3 if option_type == "call" else S * 0.7,
    }

    rows = []
    for label, spot in scenarios.items():
        p = bs.price(spot, K, T, r, sigma, option_type)
        g = bs.greeks(spot, K, T, r, sigma, option_type)
        rows.append({
            "Scenario":    label,
            "Spot ($)":    f"${spot:.2f}",
            "Price ($)":   f"${p:.4f}",
            "Δ Delta":     f"{g['delta']:+.4f}",
            "Γ Gamma":     f"{g['gamma']:.5f}",
            "ν Vega":      f"{g['vega']:+.4f}",
            "Θ Theta/day": f"{g['theta']:+.5f}",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — IV SOLVER
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("### Implied Volatility Solver")
    st.caption(
        "Implied volatility (IV) is the volatility the market is *pricing in* — "
        "it's what you get when you back-solve Black-Scholes from an observed market price. "
        "IV is the language traders use to quote options."
    )

    iv_col1, iv_col2 = st.columns([1, 1])

    with iv_col1:
        st.markdown("#### Single Option IV")
        market_price_input = st.number_input(
            "Market Price of Option ($)",
            min_value=0.001, value=round(bs_price, 2), step=0.01,
            help="The price you observe in the market for this option",
        )
        iv_type = st.radio("Option type for IV", ["call", "put"], horizontal=True, key="iv_type")

        if st.button("Solve for IV", type="primary"):
            iv_result = implied_volatility(market_price_input, S, K, T, r, iv_type)
            if iv_result is not None:
                st.markdown(
                    f"""<div class="iv-result">
                    <div style="font-size:0.8rem;color:rgba(255,255,255,0.5);margin-bottom:4px">IMPLIED VOLATILITY</div>
                    <div style="font-size:2.8rem;font-weight:800;color:#AB63FA">{iv_result*100:.2f}%</div>
                    <div style="font-size:0.85rem;color:rgba(255,255,255,0.6);margin-top:6px">
                        Historical vol (30d): {default_sigma*100:.1f}% &nbsp;·&nbsp;
                        IV premium: {(iv_result - default_sigma)*100:+.1f}pp
                    </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                # Verify round-trip
                verify_price = bs.price(S, K, T, r, iv_result, iv_type)
                st.caption(f"✓ Round-trip check: BS(IV) = ${verify_price:.6f} vs market ${market_price_input:.6f} | error = {abs(verify_price - market_price_input):.2e}")
            else:
                st.error("Could not solve for IV. Check that the market price is above intrinsic value and within no-arbitrage bounds.")

    with iv_col2:
        st.markdown("#### IV vs Strike (Volatility Smile)")
        st.caption("Shows how IV varies across strike prices for a given expiry — "
                   "the 'vol smile' or 'skew' visible in real markets.")

        smile_strikes = np.linspace(S * 0.7, S * 1.3, 25)
        # Compute IV for each strike using the current model price as proxy
        smile_ivs = []
        for k_val in smile_strikes:
            model_p = bs.price(S, k_val, T, r, sigma, option_type)
            iv_val  = implied_volatility(model_p, S, k_val, T, r, option_type)
            smile_ivs.append(iv_val * 100 if iv_val is not None else np.nan)

        fig_smile = go.Figure()
        fig_smile.add_trace(go.Scatter(
            x=smile_strikes, y=smile_ivs,
            mode="lines+markers",
            line=dict(color="#AB63FA", width=2),
            marker=dict(size=5),
            name="Implied Vol",
        ))
        fig_smile.add_vline(x=K, line_dash="dot", line_color="rgba(255,255,255,0.3)",
                            annotation_text=f"K=${K:.2f}", annotation_font_size=11)
        fig_smile.add_vline(x=S, line_dash="dot", line_color="rgba(0,204,150,0.4)",
                            annotation_text=f"S=${S:.2f}", annotation_font_size=11)
        fig_smile.update_layout(
            xaxis_title="Strike Price ($)",
            yaxis_title="Implied Volatility (%)",
            height=340,
            margin=dict(t=20, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_smile, use_container_width=True)
        st.caption("Under BS assumptions, this should be flat (constant vol). In real markets it curves — that's the vol smile/skew.")

    st.divider()

    # Live options chain with IV
    if ticker_label and use_live:
        st.markdown(f"#### Live Options Chain — {ticker_label}")
        st.caption("Real market prices vs Black-Scholes theoretical price. IV back-solved from market prices.")

        try:
            from options_pricer.data.market_data import get_options_chain
            chain_data = get_options_chain(ticker_label)
            expiry_dates = chain_data["expiry_dates"]

            selected_expiry = st.selectbox("Expiry date", expiry_dates[:8], key="chain_expiry")
            chain_type      = st.radio("Chain", ["calls", "puts"], horizontal=True, key="chain_type")

            import yfinance as yf
            tkr_obj = yf.Ticker(ticker_label)
            opt_chain = tkr_obj.option_chain(selected_expiry)
            df_chain = opt_chain.calls if chain_type == "calls" else opt_chain.puts

            T_chain = time_to_expiry(selected_expiry)

            chain_rows = []
            for _, row in df_chain.iterrows():
                k_val  = row["strike"]
                mid    = (row["bid"] + row["ask"]) / 2 if row["bid"] > 0 else row["lastPrice"]
                if mid <= 0:
                    continue
                bs_p   = bs.price(S, k_val, T_chain, r, sigma, chain_type[:-1])
                iv_val = implied_volatility(mid, S, k_val, T_chain, r, chain_type[:-1])
                diff   = mid - bs_p
                chain_rows.append({
                    "Strike":      f"${k_val:.2f}",
                    "Bid":         f"${row['bid']:.2f}",
                    "Ask":         f"${row['ask']:.2f}",
                    "Mid":         f"${mid:.2f}",
                    "BS Price":    f"${bs_p:.2f}",
                    "Diff":        f"{diff:+.2f}",
                    "IV":          f"{iv_val*100:.1f}%" if iv_val else "—",
                    "Volume":      int(row["volume"]) if not pd.isna(row["volume"]) else 0,
                    "OI":          int(row["openInterest"]) if not pd.isna(row["openInterest"]) else 0,
                })

            if chain_rows:
                st.dataframe(
                    pd.DataFrame(chain_rows),
                    use_container_width=True,
                    hide_index=True,
                    height=400,
                )
                st.caption(f"Expiry: {selected_expiry} | T = {T_chain:.4f}y | Spot = ${S:.2f} | σ used = {sigma_pct:.1f}%")
            else:
                st.info("No valid option prices returned for this expiry.")

        except Exception as e:
            st.info(f"Live chain unavailable: {e}")
    elif use_live and not ticker_label:
        st.info("Fetch a ticker in the sidebar to see the live options chain.")
    else:
        st.info("Enable **Fetch live data** in the sidebar and fetch a ticker to see the live options chain with real IV.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — HELP
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.markdown("### How to Use This Tool")
    st.caption("A practical guide to option pricing, the models, and interpreting the outputs.")

    st.markdown("#### 🚀 Quick Start")
    st.markdown("""
<div class="info-box">
<b>1.</b> In the sidebar, click a <b>Quick Pick</b> ticker or toggle <b>Fetch live data</b> and type your own, then hit <b>Fetch</b>.<br>
<b>2.</b> Stock price and volatility auto-fill from live market data.<br>
<b>3.</b> Set your <b>strike price</b> and <b>days to expiry</b>.<br>
<b>4.</b> Choose <b>call</b> (right to buy) or <b>put</b> (right to sell).<br>
<b>5.</b> Explore the tabs — Pricing, Greeks, Monte Carlo, Sensitivity, and the IV Solver.
</div>
""", unsafe_allow_html=True)

    col_h1, col_h2 = st.columns(2)

    with col_h1:
        st.markdown("#### 📐 The Parameters")
        params_data = {
            "Parameter": ["S — Stock Price", "K — Strike Price", "T — Time to Expiry", "σ — Volatility", "r — Risk-Free Rate"],
            "What it means": [
                "Current market price of the underlying stock",
                "The fixed price at which you can buy (call) or sell (put)",
                "How long until the option expires, in years",
                "Annualised std deviation of stock returns — higher = more expensive options",
                "Return on a risk-free asset (e.g. US T-bill). Typically 4–5% currently",
            ],
        }
        st.dataframe(pd.DataFrame(params_data), use_container_width=True, hide_index=True)

        st.markdown("#### 💰 Moneyness")
        st.markdown("""
| Term | Call | Put |
|------|------|-----|
| **In the Money (ITM)** | S > K | S < K |
| **At the Money (ATM)** | S ≈ K | S ≈ K |
| **Out of the Money (OTM)** | S < K | S > K |
""")

    with col_h2:
        st.markdown("#### 🔢 The Greeks")
        greeks_data = {
            "Greek": ["Δ Delta", "Γ Gamma", "ν Vega", "Θ Theta", "ρ Rho"],
            "Measures": [
                "Option price move per $1 stock move",
                "Delta change per $1 stock move",
                "Price move per 1% vol increase",
                "Price decay per calendar day",
                "Price move per 1% rate increase",
            ],
            "Sign": [
                "0→1 call / −1→0 put",
                "Always positive",
                "Always positive (long)",
                "Always negative (long)",
                "+ call / − put",
            ],
        }
        st.dataframe(pd.DataFrame(greeks_data), use_container_width=True, hide_index=True)

    st.divider()

    st.markdown("#### 🧮 The Three Pricing Models")
    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown("**Black-Scholes (1973)**")
        st.markdown("The industry-standard closed-form formula. Fast and exact under its assumptions: constant vol, no dividends, European exercise, log-normal returns.")
        st.markdown('<div class="help-formula">C = S·N(d₁) − Ke⁻ʳᵀ·N(d₂)</div>', unsafe_allow_html=True)

    with m2:
        st.markdown("**Binomial Tree (CRR)**")
        st.markdown("Discrete lattice — stock moves up or down at each step. More flexible than BS. Handles American options (early exercise). Converges to BS as steps → ∞.")
        st.markdown('<div class="help-formula">u = e^(σ√Δt) &nbsp; d = 1/u</div>', unsafe_allow_html=True)

    with m3:
        st.markdown("**Monte Carlo**")
        st.markdown("Simulates thousands of random GBM price paths. Option price = average discounted payoff. Uses antithetic variates for variance reduction.")
        st.markdown('<div class="help-formula">ST = S·exp((r−σ²/2)T + σ√T·Z)</div>', unsafe_allow_html=True)

    st.divider()

    st.markdown("#### 🔭 Implied Volatility")
    st.markdown("""
Implied volatility is the vol that makes the BS formula equal a *market-observed* option price.
It's what the market is "pricing in" about future uncertainty.

- **IV > Historical Vol** → market expects more volatility than usual (e.g. before earnings)
- **IV < Historical Vol** → market is calm; options are cheap
- **Vol smile** → in reality IV varies by strike, which violates BS assumptions. This is the "vol smile" or "skew."
""")

    st.divider()

    st.markdown("#### ❓ Common Questions")
    with st.expander("Why do the three models give slightly different prices?"):
        st.write("Black-Scholes is exact. Binomial Tree approximates it with a discrete grid — more steps = closer. Monte Carlo is statistical — more paths = lower error. All converge to the same value at the limit.")
    with st.expander("What volatility should I use?"):
        st.write("Two choices: historical vol (from past returns — what yfinance provides) or implied vol (backed out from market prices — use the IV Solver tab). Traders quote and trade in IV terms.")
    with st.expander("What does the early exercise premium mean?"):
        st.write("The extra value of being able to exercise before expiry (American vs European). For puts it can be significant when deep ITM and rates are high. For calls on non-dividend stocks it's always zero — early exercise is never optimal.")
    with st.expander("What does the P&L diagram show?"):
        st.write("Profit/loss relative to premium paid. Red curve = at expiry. Blue dashed = now (includes time value). The gap narrows as expiry approaches — that's Theta decay.")
    with st.expander("Why is the vol smile flat in the IV Solver tab?"):
        st.write("Because we're computing IV by back-solving BS prices that were themselves generated by BS. In real market data, IV curves — that curvature is the vol smile/skew, and it exists because real returns aren't perfectly log-normal.")

    st.divider()
    st.markdown(
        "<div style='text-align:center;color:rgba(255,255,255,0.3);font-size:0.8rem'>"
        "Built by Kanishka Sarkar &nbsp;·&nbsp; Black-Scholes · Binomial Tree (CRR) · Monte Carlo GBM · IV Solver"
        "</div>",
        unsafe_allow_html=True,
    )