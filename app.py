from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh

from model import (
    fetch_btc_hourly,
    fetch_btc_ohlc,
    load_backtest_metrics,
    run_backtest,
    run_live_prediction,
    run_previous_bar_validation,
    save_jsonl,
)
from lgbm_model import (
    run_lgbm_backtest,
    run_lgbm_live_prediction,
    run_lgbm_previous_bar_validation,
)
import db

try:
    db.init_db()
except Exception as _db_err:
    import sys
    print(f"[app] DB init failed, predictions won't be saved: {_db_err}", file=sys.stderr)

st.set_page_config(
    page_title="BTC Forecast Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

AUTO_REFRESH_SECONDS = 30 * 60
refresh_count = st_autorefresh(interval=AUTO_REFRESH_SECONDS * 1000, key="btc_refresh_30m")

st.markdown(
    """
<style>
:root {
  --bg: #070d1a;
  --card: #111a2b;
  --panel: #0e1727;
  --border: #243249;
  --text: #e8eefc;
  --muted: #97a7c2;
  --accent: #3ba3ff;
  --green: #22c55e;
  --orange: #f97316;
  --red: #ef4444;
}
html, body, [class*="css"] {
  background-color: var(--bg);
  color: var(--text);
  font-size: 14px;
}
[data-testid="stAppViewContainer"] {
  background-color: var(--bg);
}
[data-testid="stHeader"] {
  background: transparent;
}
[data-testid="stSidebar"] {
  background-color: var(--panel);
}
.block-container {
  padding-top: 0.8rem;
  padding-left: 1.2rem;
  padding-right: 1.2rem;
  max-width: 100%;
}
.main-title {
  font-size: 1.95rem;
  font-weight: 760;
  letter-spacing: 0.01em;
  margin-bottom: 0.45rem;
  color: var(--text);
}
.metric-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 12px 14px;
  min-height: 88px;
}
.metric-label {
  color: var(--muted);
  letter-spacing: 0.08em;
  font-size: 0.64rem;
  text-transform: uppercase;
  margin-bottom: 6px;
}
.metric-value {
  color: var(--text);
  font-size: 1.28rem;
  font-weight: 720;
  line-height: 1.2;
}
.panel {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 12px 14px;
}
.section-title {
  font-size: 1.04rem;
  font-weight: 700;
  margin: 0.2rem 0 0.45rem 0;
}
.muted {
  color: var(--muted);
}
.status-ok {
  color: var(--green);
  font-weight: 700;
}
.status-bad {
  color: var(--red);
  font-weight: 700;
}
.small-table [data-testid="stDataFrame"] * {
  font-size: 0.79rem !important;
}
</style>
""",
    unsafe_allow_html=True,
)


def fmt_usd(value: float) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "N/A"
    return f"${value:,.2f}"


def fmt_pct(value: float) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "N/A"
    return f"{value * 100:.1f}%"


def fmt_score(value: float) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "N/A"
    return f"{value:,.2f}"


def render_countdown(seconds_left: int) -> None:
    html_code = f"""
    <div style="padding:6px 10px;border:1px solid #243249;border-radius:8px;background:#111a2b;color:#e8eefc;font-size:12px;font-family:sans-serif;">
      Next auto-check in <span id="timer" style="font-weight:700;">--:--</span>
    </div>
    <script>
    let remaining = {max(0, int(seconds_left))};
    const timerEl = document.getElementById("timer");
    function tick() {{
      if (remaining <= 0) {{
        timerEl.textContent = "00:00";
        window.parent.location.reload();
        return;
      }}
      const m = Math.floor(remaining / 60).toString().padStart(2, "0");
      const s = Math.floor(remaining % 60).toString().padStart(2, "0");
      timerEl.textContent = `${{m}}:${{s}}`;
      remaining -= 1;
    }}
    tick();
    setInterval(tick, 1000);
    </script>
    """
    st.html(html_code)


def _safe_window(series: pd.Series, train_window: int = 500) -> tuple[pd.Series, pd.Series, int]:
    if len(series) < 120:
        raise ValueError("Not enough bars available for model fitting.")
    tw = min(train_window, len(series) - 2)
    tw = max(tw, 100)
    pred_slice = series.iloc[-tw:]
    val_slice = series.iloc[-(tw + 2) :]
    return pred_slice, val_slice, tw


@st.cache_data(ttl=AUTO_REFRESH_SECONDS)
def load_hourly_bundle(profile: str) -> tuple[pd.DataFrame, dict, dict, int]:
    ohlc = fetch_btc_ohlc(n_bars=620, interval="1h")
    pred_slice, val_slice, tw = _safe_window(ohlc["close"], train_window=500)
    
    if profile == "lgbm":
        pred = run_lgbm_live_prediction(ohlc.iloc[-502:], train_window=tw)
        validation = run_lgbm_previous_bar_validation(ohlc.iloc[-503:], train_window=tw)
    else:
        pred = run_live_prediction(pred_slice, n_sims=10_000, profile=profile)
        validation = run_previous_bar_validation(val_slice, train_window=tw, n_sims=3000, profile=profile)
        
    # Database Logging — wrapped so DB errors never kill the dashboard
    try:
        now_utc = pd.Timestamp.now(tz="UTC").isoformat().replace("+00:00", "Z")
        target_time = (ohlc.index[-1] + pd.Timedelta(hours=1)).isoformat().replace("+00:00", "Z")
        db.save_prediction(
            fetched_at=now_utc,
            target_time=target_time,
            low_95=pred["low_95"],
            high_95=pred["high_95"],
            current_price=pred["current_price"],
            profile=profile
        )
        db.update_actual_price(
            target_time=validation.get("actual_time", ""),
            actual_price=validation.get("actual", 0.0)
        )
    except Exception as _db_err:
        import sys
        print(f"[app] DB write failed (non-fatal): {_db_err}", file=sys.stderr)
    
    return ohlc, pred, validation, tw


@st.cache_data(ttl=AUTO_REFRESH_SECONDS)
def load_daily_bundle(profile: str) -> tuple[pd.DataFrame, dict, dict, int]:
    ohlc = fetch_btc_ohlc(n_bars=560, interval="1d")
    pred_slice, val_slice, tw = _safe_window(ohlc["close"], train_window=500)
    
    if profile == "lgbm":
        pred = run_lgbm_live_prediction(ohlc.iloc[-502:], train_window=tw)
        validation = run_lgbm_previous_bar_validation(ohlc.iloc[-503:], train_window=tw)
    else:
        pred = run_live_prediction(pred_slice, n_sims=7000, profile=profile)
        validation = run_previous_bar_validation(val_slice, train_window=tw, n_sims=2000, profile=profile)
    return ohlc, pred, validation, tw


@st.cache_data(ttl=AUTO_REFRESH_SECONDS)
def load_backtest_chart_frame(bt_rows: pd.DataFrame) -> pd.DataFrame:
    if bt_rows.empty:
        return pd.DataFrame()
    now_utc = pd.Timestamp.now(tz="UTC")
    oldest = bt_rows["timestamp"].min()
    span_hrs = int((now_utc - oldest).total_seconds() / 3600) + 72
    bars = min(5000, max(1500, span_hrs))
    ohlc = fetch_btc_ohlc(n_bars=bars, interval="1h")
    bt_for_merge = bt_rows.set_index("timestamp")[["low_95", "high_95", "actual", "coverage_95"]]
    merged = ohlc.join(bt_for_merge, how="inner")
    return merged


def build_hourly_chart(ohlc: pd.DataFrame, low_95: float, high_95: float) -> go.Figure:
    tail = ohlc.tail(50)
    last_ts = tail.index[-1]
    step = tail.index[-1] - tail.index[-2] if len(tail) > 1 else timedelta(hours=1)
    next_ts = last_ts + step

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=tail.index,
            open=tail["open"],
            high=tail["high"],
            low=tail["low"],
            close=tail["close"],
            name="BTCUSDT 1h",
            increasing_line_color="#22c55e",
            decreasing_line_color="#f97316",
            increasing_fillcolor="#22c55e",
            decreasing_fillcolor="#f97316",
            opacity=0.95,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[last_ts, next_ts],
            y=[low_95, low_95],
            mode="lines",
            name="Pred Low 95%",
            line={"color": "#f59e0b", "width": 1.8, "dash": "dot"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[last_ts, next_ts],
            y=[high_95, high_95],
            mode="lines",
            name="Pred High 95%",
            line={"color": "#38bdf8", "width": 1.8, "dash": "dot"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[next_ts, next_ts],
            y=[low_95, high_95],
            mode="lines+markers",
            name="Next-Hour Range",
            line={"color": "#3ba3ff", "width": 2.2},
            marker={"size": 6, "color": "#3ba3ff"},
            showlegend=True,
        )
    )

    fig.add_vline(x=last_ts, line_dash="dash", line_color="#64748b", line_width=1)
    fig.update_layout(
        title="Hourly Candles with Next-Hour Predicted Band",
        paper_bgcolor="#111a2b",
        plot_bgcolor="#111a2b",
        font={"color": "#e8eefc", "size": 12},
        xaxis={"showgrid": True, "gridcolor": "#243249", "title": ""},
        yaxis={"showgrid": True, "gridcolor": "#243249", "title": "Price (USD)", "tickformat": ",.2f"},
        xaxis_rangeslider_visible=False,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 18, "r": 18, "t": 58, "b": 20},
    )
    return fig


def build_backtest_chart(merged: pd.DataFrame, bars_to_show: int = 220) -> go.Figure:
    view = merged.tail(bars_to_show)
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=view.index,
            open=view["open"],
            high=view["high"],
            low=view["low"],
            close=view["close"],
            name="Actual Candles",
            increasing_line_color="#22c55e",
            decreasing_line_color="#f97316",
            increasing_fillcolor="#22c55e",
            decreasing_fillcolor="#f97316",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=view.index,
            y=view["low_95"],
            mode="lines",
            name="Pred Low 95%",
            line={"color": "#f59e0b", "width": 1.8},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=view.index,
            y=view["high_95"],
            mode="lines",
            name="Pred High 95%",
            line={"color": "#38bdf8", "width": 1.8},
        )
    )

    misses = view[view["coverage_95"] == 0]
    if not misses.empty:
        fig.add_trace(
            go.Scatter(
                x=misses.index,
                y=misses["actual"],
                mode="markers",
                name="Miss",
                marker={"color": "#ef4444", "size": 6, "symbol": "x"},
            )
        )

    fig.update_layout(
        title="Backtest Candles vs Predicted Low/High 95%",
        paper_bgcolor="#111a2b",
        plot_bgcolor="#111a2b",
        font={"color": "#e8eefc", "size": 12},
        xaxis={"showgrid": True, "gridcolor": "#243249", "title": ""},
        yaxis={"showgrid": True, "gridcolor": "#243249", "title": "Price (USD)", "tickformat": ",.2f"},
        xaxis_rangeslider_visible=False,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 18, "r": 18, "t": 58, "b": 20},
    )
    return fig


if "refresh_anchor_utc" not in st.session_state:
    st.session_state.refresh_anchor_utc = pd.Timestamp.now(tz="UTC")
if "last_refresh_count" not in st.session_state:
    st.session_state.last_refresh_count = refresh_count
if refresh_count != st.session_state.last_refresh_count:
    st.session_state.refresh_anchor_utc = pd.Timestamp.now(tz="UTC")
    st.session_state.last_refresh_count = refresh_count

elapsed = (pd.Timestamp.now(tz="UTC") - st.session_state.refresh_anchor_utc).total_seconds()
seconds_left = max(0, int(AUTO_REFRESH_SECONDS - elapsed))

st.markdown('<div class="main-title">BTC Forecast Dashboard</div>', unsafe_allow_html=True)

profile = st.sidebar.selectbox("Profile", ["precision", "lgbm", "tuned", "challenge"], index=0)
manual_refresh = st.sidebar.button("Run Simulation Now", use_container_width=True)
st.sidebar.markdown(f"Auto-check interval: {AUTO_REFRESH_SECONDS // 60} minutes")
st.sidebar.markdown(f"Last app run: {pd.Timestamp.now(tz='Asia/Kolkata').strftime('%Y-%m-%d %H:%M:%S IST')}")

with st.spinner("Running simulation"):
    if manual_refresh:
        load_hourly_bundle.clear()
        load_daily_bundle.clear()
        
    hourly_ohlc, hourly_pred, hourly_val, hourly_tw = load_hourly_bundle(profile)
    daily_ohlc, daily_pred, daily_val, daily_tw = load_daily_bundle(profile)

backtest_path = Path(__file__).parent / f"backtest_results_{profile}.jsonl"
bt = load_backtest_metrics(backtest_path)
has_backtest = not bt.rows.empty

coverage_text = fmt_pct(bt.coverage) if has_backtest else "N/A"
winkler_text = fmt_score(bt.mean_winkler) if has_backtest else "N/A"

top_left, top_right = st.columns([4.0, 1.2], vertical_alignment="center")
with top_left:
    hour_status = "WITHIN RANGE" if hourly_val["in_range"] else "OUT OF RANGE"
    st.caption(
        f"Hourly previous-bar check | Prediction @ {hourly_val['prediction_time']} | "
        f"Actual @ {hourly_val['actual_time']} | Status: {hour_status}"
    )
with top_right:
    render_countdown(seconds_left)

m1, m2, m3, m4 = st.columns(4)
m1.markdown(
    f"""
<div class="metric-card">
  <div class="metric-label">Current BTC Price (Hourly)</div>
  <div class="metric-value">{fmt_usd(hourly_pred["current_price"])}</div>
</div>
""",
    unsafe_allow_html=True,
)
m2.markdown(
    f"""
<div class="metric-card">
  <div class="metric-label">Hourly 95% Predicted Range</div>
  <div class="metric-value">{fmt_usd(hourly_pred["low_95"])} - {fmt_usd(hourly_pred["high_95"])}</div>
</div>
""",
    unsafe_allow_html=True,
)
m3.markdown(
    f"""
<div class="metric-card">
  <div class="metric-label">Backtest Coverage</div>
  <div class="metric-value">{coverage_text}</div>
</div>
""",
    unsafe_allow_html=True,
)
m4.markdown(
    f"""
<div class="metric-card">
  <div class="metric-label">Mean Winkler Score</div>
  <div class="metric-value">{winkler_text}</div>
</div>
""",
    unsafe_allow_html=True,
)

v1, v2 = st.columns(2)
hour_class = "status-ok" if hourly_val["in_range"] else "status-bad"
day_class = "status-ok" if daily_val["in_range"] else "status-bad"

v1.markdown(
    f"""
<div class="panel" style="margin-top:8px;">
  <div class="metric-label">Hourly Validation</div>
  <div><span class="muted">Actual:</span> <b>{fmt_usd(hourly_val["actual"])}</b></div>
  <div><span class="muted">Predicted:</span> <b>{fmt_usd(hourly_val["low_95"])} - {fmt_usd(hourly_val["high_95"])}</b></div>
  <div><span class="{hour_class}">{'WITHIN RANGE' if hourly_val['in_range'] else 'OUT OF RANGE'}</span></div>
</div>
""",
    unsafe_allow_html=True,
)
v2.markdown(
    f"""
<div class="panel" style="margin-top:8px;">
  <div class="metric-label">Daily Validation</div>
  <div><span class="muted">Actual:</span> <b>{fmt_usd(daily_val["actual"])}</b></div>
  <div><span class="muted">Predicted:</span> <b>{fmt_usd(daily_val["low_95"])} - {fmt_usd(daily_val["high_95"])}</b></div>
  <div><span class="{day_class}">{'WITHIN RANGE' if daily_val['in_range'] else 'OUT OF RANGE'}</span></div>
</div>
""",
    unsafe_allow_html=True,
)

st.plotly_chart(
    build_hourly_chart(hourly_ohlc, low_95=hourly_pred["low_95"], high_95=hourly_pred["high_95"]),
    width="stretch",
    config={"displayModeBar": False},
)

dleft, dright = st.columns([1.35, 0.95])
with dleft:
    st.markdown('<div class="section-title">Daily Forecast Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="panel">
  <div><span class="muted">Current BTC Daily Close:</span> <b>{fmt_usd(daily_pred["current_price"])}</b></div>
  <div><span class="muted">Next-Day 95% Range:</span> <b>{fmt_usd(daily_pred["low_95"])} - {fmt_usd(daily_pred["high_95"])}</b></div>
  <div><span class="muted">Training Window Used:</span> <b>{daily_tw} daily bars</b></div>
</div>
""",
        unsafe_allow_html=True,
    )
    day_tail = daily_ohlc.tail(120)
    day_fig = go.Figure()
    day_fig.add_trace(
        go.Candlestick(
            x=day_tail.index,
            open=day_tail["open"],
            high=day_tail["high"],
            low=day_tail["low"],
            close=day_tail["close"],
            name="BTCUSDT 1d",
            increasing_line_color="#22c55e",
            decreasing_line_color="#f97316",
            increasing_fillcolor="#22c55e",
            decreasing_fillcolor="#f97316",
        )
    )
    day_fig.update_layout(
        paper_bgcolor="#111a2b",
        plot_bgcolor="#111a2b",
        font={"color": "#e8eefc", "size": 12},
        xaxis={"showgrid": True, "gridcolor": "#243249"},
        yaxis={"showgrid": True, "gridcolor": "#243249", "tickformat": ",.2f"},
        xaxis_rangeslider_visible=False,
        margin={"l": 18, "r": 18, "t": 24, "b": 18},
    )
    st.plotly_chart(day_fig, use_container_width=True, config={"displayModeBar": False})

with dright:
    st.markdown('<div class="section-title">Actions and Parameters</div>', unsafe_allow_html=True)
    with st.form("backtest_form"):
        bars = st.number_input("Fetch bars", min_value=800, max_value=5000, value=1300, step=100)
        train_window = st.number_input("Train window", min_value=300, max_value=1500, value=500, step=25)
        test_bars = st.number_input("Test bars", min_value=100, max_value=2000, value=720, step=20)
        n_sims_bt = st.number_input("Backtest n_sims", min_value=500, max_value=5000, value=2000, step=250)
        rerun_bt = st.form_submit_button("Run Full Backtest Now", use_container_width=True)

    if rerun_bt:
        with st.spinner("Running simulation"):
            if profile == "lgbm":
                bt_ohlc = fetch_btc_ohlc(n_bars=int(bars))
                bt_rows = run_lgbm_backtest(
                    ohlc=bt_ohlc,
                    train_window=int(train_window),
                    test_bars=int(test_bars),
                )
            else:
                bt_prices = fetch_btc_hourly(n_bars=int(bars))
                bt_rows = run_backtest(
                    prices=bt_prices,
                    train_window=int(train_window),
                    test_bars=int(test_bars),
                    n_sims=int(n_sims_bt),
                    profile=profile,
                )
            save_jsonl(bt_rows, backtest_path)
        st.success(f"Backtest completed: {len(bt_rows)} rows saved to {backtest_path.name}")

    st.markdown(
        f"""
<div class="panel">
  <div><span class="muted">Model:</span> Cyber-GBM / FIGARCH / Student-t ({profile})</div>
  <div><span class="muted">Hourly Training Window:</span> {hourly_tw} bars</div>
  <div><span class="muted">Daily Training Window:</span> {daily_tw} bars</div>
  <div><span class="muted">Auto-check:</span> Every 30 minutes</div>
  <div><span class="muted">Data Source:</span> Binance data-api.binance.vision</div>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-title">Backtest Analytics (Candles + Predicted Low/High)</div>', unsafe_allow_html=True)

if has_backtest:
    date_min = bt.rows["timestamp"].min()
    date_max = bt.rows["timestamp"].max()
    now_utc = pd.Timestamp.now(tz="UTC")
    stale_hours = (now_utc - date_max).total_seconds() / 3600

    st.markdown(
        f"""
<div class="panel" style="margin-bottom:8px;">
  <div><span class="muted">Rows:</span> <b>{len(bt.rows):,}</b></div>
  <div><span class="muted">Date range in file:</span> <b>{date_min.strftime("%Y-%m-%d %H:%M UTC")} to {date_max.strftime("%Y-%m-%d %H:%M UTC")}</b></div>
  <div><span class="muted">Coverage:</span> <b>{coverage_text}</b> |
       <span class="muted">Mean Width:</span> <b>{fmt_usd(bt.avg_width)}</b> |
       <span class="muted">Mean Winkler:</span> <b>{winkler_text}</b></div>
</div>
""",
        unsafe_allow_html=True,
    )

    if stale_hours > 6:
        st.warning(
            f"Backtest data is stale by {stale_hours:.1f} hours. Click 'Run Full Backtest Now' to refresh to latest bars."
        )

    bars_to_show = st.slider("Backtest candles to display", min_value=80, max_value=720, value=220, step=20)
    merged_bt = load_backtest_chart_frame(bt.rows)

    if merged_bt.empty:
        st.warning("Could not align backtest rows with fetched OHLC candles for plotting.")
    else:
        aligned_ratio = len(merged_bt) / max(1, len(bt.rows))
        if aligned_ratio < 0.95:
            st.info(
                f"Showing {len(merged_bt)} / {len(bt.rows)} rows on chart. "
                "If this is low, rerun backtest now to sync with latest fetched history."
            )
        st.plotly_chart(
            build_backtest_chart(merged_bt, bars_to_show=bars_to_show),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    hist = bt.rows.sort_values("timestamp", ascending=False).copy()
    hist["Timestamp"] = hist["timestamp"].dt.tz_convert("Asia/Kolkata").dt.strftime("%Y-%m-%d %H:%M:%S IST")
    hist["Actual Price"] = hist["actual"].map(fmt_usd)
    hist["Low 95%"] = hist["low_95"].map(fmt_usd)
    hist["High 95%"] = hist["high_95"].map(fmt_usd)
    hist["Width"] = hist["width_95"].map(fmt_usd)
    hist["Hit"] = np.where(hist["coverage_95"] == 1, "YES", "NO")
    hist = hist[["Timestamp", "Actual Price", "Low 95%", "High 95%", "Width", "Hit"]]

    def _hit_style(value: str) -> str:
        return "color: #22c55e; font-weight: 700;" if value == "YES" else "color: #ef4444; font-weight: 700;"

    st.markdown('<div class="section-title">Backtest Full History Table</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-table">', unsafe_allow_html=True)
    st.dataframe(hist.style.map(_hit_style, subset=["Hit"]), use_container_width=True, hide_index=True, height=420)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info(f"No {backtest_path.name} found. Run backtest from the Actions panel.")

# --- LIVE PREDICTION HISTORY ---
st.markdown("---")
st.markdown('<div class="section-title">Live Prediction History (Database)</div>', unsafe_allow_html=True)
st.markdown('<div class="metric-card">', unsafe_allow_html=True)
st.caption("Tracking live performance stored in Supabase/SQLite.")

live_df = db.get_prediction_history_df(limit=20)
if not live_df.empty:
    display_df = live_df.copy()
    if "target_time" in display_df.columns:
        display_df["Target Time"] = pd.to_datetime(display_df["target_time"], utc=True).dt.tz_convert("Asia/Kolkata").dt.strftime("%H:%M IST (%d %b)")
    
    for col, new_name in [("current_price", "Entry Price"), ("low_95", "Lower Band"), ("high_95", "Upper Band"), ("actual_price", "Actual Price")]:
        if col in display_df.columns:
            display_df[new_name] = display_df[col].apply(lambda x: f"${x:,.2f}" if x and x > 0 else "---")
    
    # Summary Metrics
    if "Result" in display_df.columns:
        valid_rows = display_df[display_df["Result"] != "Pending"]
        if len(valid_rows) > 0:
            hits = (valid_rows["Result"] == "HIT").sum()
            acc = (hits / len(valid_rows)) * 100
            st.metric("Live Accuracy", f"{acc:.1f}%")

    cols_to_show = ["Target Time", "Entry Price", "Lower Band", "Upper Band", "Actual Price", "Result", "profile"]
    available_cols = [c for c in cols_to_show if c in display_df.columns]
    
    def _result_style(val):
        if val == "HIT": return "background-color: rgba(34, 197, 94, 0.1); color: #22c55e; font-weight: bold;"
        if val == "MISS": return "background-color: rgba(239, 68, 68, 0.1); color: #ef4444; font-weight: bold;"
        return ""

    if "Result" in display_df.columns:
        st.dataframe(display_df[available_cols].style.map(_result_style, subset=["Result"]), use_container_width=True, hide_index=True)
    else:
        st.dataframe(display_df[available_cols], use_container_width=True, hide_index=True)
else:
    st.info("No live predictions logged yet.")
st.markdown("</div>", unsafe_allow_html=True)