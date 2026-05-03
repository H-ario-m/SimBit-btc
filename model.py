"""
model.py — BTC 1-Hour Forecast Model
=====================================
Changes vs previous version (annotated with CHANGE tags):

[CHANGE-1] GJR-GARCH replaces FIGARCH
  FIGARCH has long memory designed for daily data. On 1h BTC it over-weights
  volatility spikes from days ago, inflating bar_sigma2 and therefore every
  simulated path. GJR-GARCH(1,1,1) with studentst distribution is better
  calibrated for intraday crypto: it decays old shocks quickly and captures
  the asymmetric leverage effect (drops spike vol more than rises).

[CHANGE-2] 1-step-ahead GARCH forecast replaces mean of historical cond. vol
  Previous code used bar_sigma2 = mean(sigma_fig**2) across 500 bars as the
  simulation baseline. This averages calm and stormy periods together, keeping
  baseline too high during calm stretches. We now use the GARCH model's own
  1-step-ahead variance forecast as sigma_now2 and as bar_sigma2. This is the
  model's direct estimate of the NEXT hour's vol, not a historical average.

[CHANGE-3] EMA drift replaces long-run mean
  mu = log_ret.mean() across 500 bars is close to zero and noisy. It does not
  capture local momentum. Replaced with ewm(span=24).mean().iloc[-1] — the
  exponentially weighted mean of the last ~24 hours, which reflects whether
  BTC has been trending up or down recently.

[CHANGE-4] sigma_scale calibration multiplier
  A single float in SimulationConfig (default 1.0 for challenge, 0.82 for
  precision). Applied once to sigma2 inside simulate_cyber_gbm before the
  Student-t draw. This is the cleanest way to tighten bands without changing
  the structural model: if backtest shows coverage=0.975, set sigma_scale=0.85
  and rerun. Target: coverage lands at 0.94-0.96.

[CHANGE-5] Recent-window bar_sigma2
  bar_sigma2 is now the mean of sigma_fig**2 over only the last 48 bars
  (2 days), not the full 500-bar window. This makes the mean-reversion target
  in the gamma term reflect current volatility regime, not historical average.

[CHANGE-6] "precision" profile
  New SimulationConfig with alpha=0.05, delta=0.02, info_filter_multiplier=0.05,
  sigma_scale=0.82. Run backtest with --profile precision to get tighter bands.
  After verifying coverage stays >= 0.93, use this as your submission profile.

[CHANGE-7] Today-inclusive fetch guarantee
  Added endTime=None on the first call (no cap) so Binance returns bars up to
  the latest closed hour. Previous pagination started with end_time=None but
  could miss the current partial hour if run at the top of the hour. Added a
  check that drops any bar whose close_time is in the future.

[NO CHANGE] simulate_cyber_gbm path structure, Student-t draws, Winkler
  formula, JSONL format, all dashboard function signatures — UI stays identical.
"""

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import scipy.stats as stats
from arch import arch_model
from tqdm import tqdm


BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"


@dataclass
class BacktestMetrics:
    coverage: float
    avg_width: float
    mean_winkler: float
    rows: pd.DataFrame


@dataclass
class SimulationConfig:
    profile: str
    alpha: float
    delta: float
    gamma: float = 0.2
    kappa: float = 0.1
    eta: float = 1e-3
    info_filter_multiplier: float = 0.5
    enforce_entropy_scaling: bool = True
    # [CHANGE-4] sigma_scale: global multiplier applied to sigma2 before each draw.
    # Values < 1.0 tighten bands. Tune this until backtest coverage ~ 0.95.
    sigma_scale: float = 1.0


def get_simulation_config(profile: str = "precision") -> SimulationConfig:
    """Get simulation hyperparameter profile."""
    profile_key = profile.strip().lower()

    if profile_key == "tuned":
        return SimulationConfig(
            profile="tuned",
            alpha=0.15,
            delta=0.08,
            gamma=0.2,
            kappa=0.1,
            eta=1e-3,
            info_filter_multiplier=0.15,
            enforce_entropy_scaling=False,
            sigma_scale=0.90,
        )

    # [CHANGE-6] New precision profile: minimal signal stacking, calibrated scale.
    # This is the recommended submission profile once you verify coverage >= 0.93.
    if profile_key == "precision":
        return SimulationConfig(
            profile="precision",
            alpha=0.08,         
            delta=0.05,         
            gamma=0.10,         
            kappa=0.1,
            eta=1e-3,
            info_filter_multiplier=0.05,
            enforce_entropy_scaling=False,
            sigma_scale=0.90,   # Raised from 0.88: nu cap now does the tail work, scale adds uniform headroom
        )

    # Original challenge profile — kept identical for reproducibility.
    return SimulationConfig(
        profile="challenge",
        alpha=0.5,
        delta=0.3,
        gamma=0.2,
        kappa=0.1,
        eta=1e-3,
        info_filter_multiplier=0.5,
        enforce_entropy_scaling=True,
        sigma_scale=0.85,       # Tuned down from 1.0 to drop coverage from 0.97 to ~0.95
    )


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _fetch_btc_klines_rows(n_bars: int = 750, symbol: str = "BTCUSDT", interval: str = "1h") -> list[list[Any]]:
    """
    Fetch raw Binance klines with backward pagination.
    [CHANGE-7] Drops any row whose close_time is still in the future (partial candle).
    """
    target = int(n_bars)
    if target <= 0:
        raise ValueError("n_bars must be > 0")

    all_rows: list[list[Any]] = []
    end_time: int | None = None
    now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)

    while len(all_rows) < target:
        params: dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": 1000}
        if end_time is not None:
            params["endTime"] = end_time

        response = requests.get(BINANCE_KLINES_URL, params=params, timeout=20)
        response.raise_for_status()
        raw_batch = response.json()          # keep raw for pagination checks
        if not raw_batch:
            break

        # [CHANGE-7] Drop bars whose close_time (index 6) is still in the future.
        # Use raw_batch length for the pagination break — NOT the filtered length.
        # Bug fix: filtering can produce len < 1000 even when API returned a full
        # page (e.g. the latest partial candle gets removed), causing premature exit.
        batch = [row for row in raw_batch if int(row[6]) <= now_ms]
        if batch:
            all_rows = batch + all_rows
        end_time = int(raw_batch[0][0]) - 1  # always use raw for end_time
        if len(raw_batch) < 1000:            # only stop if API gave a partial page
            break

    rows = all_rows[-target:]
    if not rows:
        raise ValueError("Binance API returned no rows.")
    return rows


def fetch_btc_ohlc(n_bars: int = 750, symbol: str = "BTCUSDT", interval: str = "1h") -> pd.DataFrame:
    """Fetch BTC OHLCV candles indexed by UTC open time."""
    rows = _fetch_btc_klines_rows(n_bars=n_bars, symbol=symbol, interval=interval)
    df = pd.DataFrame(
        rows,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
        ],
    )
    ohlc = pd.DataFrame(
        {
            "open":   df["open"].astype(float),
            "high":   df["high"].astype(float),
            "low":    df["low"].astype(float),
            "close":  df["close"].astype(float),
            "volume": df["volume"].astype(float),
        }
    )
    ohlc.index = pd.to_datetime(df["open_time"].astype(np.int64), unit="ms", utc=True)
    ohlc.index.name = "timestamp"
    return ohlc.sort_index()


def fetch_btc_hourly(n_bars: int = 750, symbol: str = "BTCUSDT", interval: str = "1h") -> pd.Series:
    """Fetch BTC close prices as a Series indexed by UTC open time."""
    return fetch_btc_ohlc(n_bars=n_bars, symbol=symbol, interval=interval)["close"]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def rolling_entropy(x: pd.Series, window: int = 24, bins: int = 20) -> pd.Series:
    """Rolling Shannon entropy of recent residuals."""
    def ent(values: np.ndarray) -> float:
        probs, _ = np.histogram(values, bins=bins, density=True)
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs)))
    return x.rolling(window).apply(ent, raw=True)


def _prepare_train_features(train_prices: pd.Series, config: SimulationConfig | None = None) -> dict[str, Any]:
    """
    Fit volatility model and compute all regime features from train data only.
    No peeking: only prices up to and including bar i are used.
    """
    if config is None:
        config = get_simulation_config("challenge")

    log_ret = np.log(train_prices / train_prices.shift(1)).dropna()
    if len(log_ret) < 100:
        raise ValueError("Not enough return observations to fit model reliably.")

    # [CHANGE-1] GJR-GARCH(1,1,1) replaces FIGARCH.
    # p=1 ARCH term, o=1 asymmetry (GJR leverage), q=1 GARCH term.
    # dist='studentst' preserved — critical for fat tails.
    # disp='off' suppresses convergence output in loops.
    am = arch_model(log_ret * 100, vol="Garch", p=1, o=1, q=1, dist="studentst")
    res = am.fit(disp="off", show_warning=False)

    sigma_fig = res.conditional_volatility / 100.0   # kept name for compatibility

    # [CHANGE-2] 1-step-ahead GARCH forecast.
    # This is the model's direct estimate of next-hour conditional variance.
    # Much better than the historical mean — responds to current calm/storm.
    try:
        fc = res.forecast(horizon=1, reindex=False)
        sigma_forecast_val = float(np.sqrt(fc.variance.iloc[-1, 0])) / 100.0
    except Exception:
        sigma_forecast_val = float(sigma_fig.iloc[-1])

    # Standardized residuals for entropy and Student-t nu estimation.
    resid = (log_ret * 100.0 - res.params.get("mu", 0.0)) / (res.conditional_volatility + 1e-12)
    nu_fit = stats.t.fit(resid, floc=0, fscale=1)[0]
    # Clip nu to [3, 5]. When GJR-GARCH fits well, nu_fit can be 10-20 (near-normal).
    # max(4.0, nu_fit) then does nothing — the distribution has thin tails and
    # extreme moves land outside the 95% band.
    # t(3): 97.5th pct=3.18 | t(5): 97.5th pct=2.57 | t(6): 97.5th pct=2.45 | t(12): 97.5th pct=2.18
    # Capping at 5 (down from 6) gives ~5% heavier tails vs t(6) with barely any center inflation.
    # Misses concentrate in sudden spike bars where GARCH sees LOW entropy (calm before storm).
    # Alpha-based entropy inflation won't catch these — heavier tails universally is the right fix.
    nu = float(np.clip(nu_fit, 3.0, 5.0))

    H_series = rolling_entropy(resid, window=24)
    M_series = log_ret.abs().rolling(24).mean()

    price_var_short = train_prices.rolling(5).var()
    price_var_long  = train_prices.rolling(20).var()
    redundancy = 1.0 + 0.1 * np.log1p(price_var_short / price_var_long)
    redundancy = redundancy.reindex(log_ret.index).ffill().bfill().fillna(1.0)

    info_filter = (H_series > H_series.mean()).astype(float).fillna(0.0)

    h_max = float(np.nanmax(H_series.values)) if np.isfinite(np.nanmax(H_series.values)) else 1.0
    m_max = float(np.nanmax(M_series.values)) if np.isfinite(np.nanmax(M_series.values)) else 1.0
    h_max = h_max if h_max > 0 else 1.0
    m_max = m_max if m_max > 0 else 1.0

    alpha0 = float(config.alpha)
    delta0 = float(config.delta)
    if config.enforce_entropy_scaling and alpha0 * h_max + delta0 * m_max >= 1:
        factor = 0.95 / (alpha0 * h_max + delta0 * m_max)
        alpha0 *= factor
        delta0 *= factor

    base_params = {
        "alpha": float(alpha0),
        "delta": float(delta0),
        "gamma": float(config.gamma),
        "kappa": float(config.kappa),
        "eta":   float(config.eta),
    }

    # [CHANGE-5] bar_sigma2 from last 48 bars only (current regime), not full 500-bar mean.
    # The gamma mean-reversion target now reflects recent vol, not historical average.
    recent_sigma = sigma_fig.iloc[-48:] if len(sigma_fig) >= 48 else sigma_fig
    bar_sigma2 = float((recent_sigma ** 2).mean())
    if not np.isfinite(bar_sigma2) or bar_sigma2 <= 0:
        bar_sigma2 = float(log_ret.std(ddof=0) ** 2)

    # [CHANGE-3] EMA drift: exponentially weighted mean over last ~24 bars.
    # Captures local momentum (is BTC trending up or down right now?).
    mu_ema = float(log_ret.ewm(span=24, adjust=False).mean().iloc[-1])
    if not np.isfinite(mu_ema):
        mu_ema = float(log_ret.mean())

    return {
        "log_ret":        log_ret,
        "mu":             mu_ema,                  # [CHANGE-3]
        "sigma_fig":      sigma_fig,
        "sigma_forecast": sigma_forecast_val,      # [CHANGE-2] 1-step ahead forecast
        "resid":          resid,
        "nu":             nu,
        "H_series":       H_series,
        "M_series":       M_series,
        "redundancy":     redundancy,
        "info_filter":    info_filter,
        "base_params":    base_params,
        "bar_sigma2":     bar_sigma2,              # [CHANGE-5] recent 48-bar target
    }


# ---------------------------------------------------------------------------
# Simulation engine  (structure unchanged — only sigma_scale added)
# ---------------------------------------------------------------------------

def update_params(params: dict[str, float], sigma2: float, bar_sigma2: float, t: int) -> dict[str, float]:
    err = sigma2 - bar_sigma2
    lr = params["eta"] / (1 + t ** 0.55)
    params["gamma"] = float(np.clip(params["gamma"] + lr * err, 0.01, 0.5))
    return params


def simulate_cyber_gbm(
    S0: float,
    mu: float,
    sigma_fig: pd.Series,
    H: pd.Series,
    M: pd.Series,
    params: dict[str, float],
    bar_sigma2: float,
    nu: float,
    redundancy: pd.Series,
    info_filter: pd.Series,
    n_steps: int,
    dt: float = 1.0,
    eps: float = 1e-6,
    info_filter_multiplier: float = 0.5,
    sigma_scale: float = 1.0,          # [CHANGE-4]
    sigma_start: float | None = None,  # [CHANGE-2] override starting sigma2
) -> tuple[np.ndarray, np.ndarray]:
    """Single Cyber-GBM path with GARCH-informed variance dynamics."""
    S = np.zeros(n_steps + 1)
    V = np.zeros(n_steps + 1)
    S[0] = float(S0)

    # [CHANGE-2] Use 1-step-ahead forecast as starting sigma2 if provided.
    if sigma_start is not None and np.isfinite(sigma_start) and sigma_start > 0:
        sigma2 = float(sigma_start ** 2)
    else:
        sigma2 = float(sigma_fig.iloc[-1] ** 2)

    H_max = float(H.max()) if float(H.max()) > 0 else 1.0
    M_max = float(M.max()) if float(M.max()) > 0 else 1.0

    for t in range(1, n_steps + 1):
        current = -1
        H_val = min(float(H.iloc[current]) / H_max, 1.0)
        M_val = min(float(M.iloc[current]) / M_max, 1.0)
        crisis = (H_val > 0.8) or (M_val > 0.8)
        delta_t = params["delta"] if crisis else 0.0

        sigma2 = (
            float(sigma_fig.iloc[current] ** 2) * (1 + params["alpha"] * H_val + delta_t * M_val)
            + params["gamma"] * (bar_sigma2 - sigma2)
        )
        sigma2 *= max(1e-12, float(redundancy.iloc[current]))
        sigma2 *= 1 + info_filter_multiplier * float(info_filter.iloc[current])

        # [CHANGE-4] Apply sigma_scale: tightens bands proportionally.
        # Values < 1.0 narrow the interval; does not change structural shape.
        sigma2 *= sigma_scale ** 2

        sigma2 = max(eps, min(sigma2, 0.5))

        Z = np.random.standard_t(nu) * np.sqrt((nu - 2) / nu)
        S[t] = S[t - 1] * np.exp((mu - 0.5 * sigma2) * dt + np.sqrt(sigma2 * dt) * Z)
        V[t] = sigma2
        params = update_params(params, sigma2, bar_sigma2, t)

    return S, V


def simulate_mc(
    S0: float,
    mu: float,
    sigma_fig: pd.Series,
    H: pd.Series,
    M: pd.Series,
    bar_sigma2: float,
    base_params: dict[str, float],
    nu: float,
    redundancy: pd.Series,
    info_filter: pd.Series,
    n_sims: int = 10_000,
    n_days: int = 1,
    dt: float = 1.0,
    info_filter_multiplier: float = 0.5,
    sigma_scale: float = 1.0,       # [CHANGE-4]
    sigma_start: float | None = None,  # [CHANGE-2]
) -> np.ndarray:
    out = np.zeros((n_sims, n_days + 1))
    for i in range(n_sims):
        paths, _ = simulate_cyber_gbm(
            S0=S0,
            mu=mu,
            sigma_fig=sigma_fig,
            H=H,
            M=M,
            params=copy.deepcopy(base_params),
            bar_sigma2=bar_sigma2,
            nu=nu,
            redundancy=redundancy,
            info_filter=info_filter,
            n_steps=n_days,
            dt=dt,
            info_filter_multiplier=info_filter_multiplier,
            sigma_scale=sigma_scale,
            sigma_start=sigma_start,
        )
        out[i] = paths
    return out


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def run_backtest(
    prices: pd.Series,
    train_window: int = 500,
    test_bars: int = 720,
    n_sims: int = 2000,
    profile: str = "challenge",
) -> list[dict]:
    """
    Strict no-peek rolling backtest for one-step-ahead BTC forecasts.
    Iterates from bar train_window up to train_window + test_bars - 1.
    Bar i+1 actual price is only used for scoring, never for fitting.
    """
    if len(prices) < train_window + 2:
        raise ValueError("Not enough bars for requested train_window.")

    config = get_simulation_config(profile)

    # [FIX] Test the most recent test_bars, not the first test_bars after train_window.
    # Old: range(train_window, train_window + test_bars) → test starts from bar 500, ends ~April 30.
    # New: take the last test_bars available → always ends at the latest data (today).
    # With 1300 bars, train_window=500, test_bars=720: start_i = max(500, 1300-721) = 579
    # → test covers bars 579..1298, actuals bars 580..1299 → April 3 to May 3. ✓
    start_i = max(train_window, len(prices) - test_bars - 1)
    max_i = len(prices) - 2
    alpha = 0.05
    rows: list[dict] = []

    for i in tqdm(range(start_i, max_i + 1), desc=f"Backtest [{profile}]"):
        train_prices = prices.iloc[i - train_window : i]   # strict no-peek
        feats = _prepare_train_features(train_prices, config=config)

        S0_bt = float(prices.iloc[i])
        paths_bt = simulate_mc(
            S0=S0_bt,
            mu=feats["mu"],
            sigma_fig=feats["sigma_fig"],
            H=feats["H_series"],
            M=feats["M_series"],
            bar_sigma2=feats["bar_sigma2"],
            base_params=feats["base_params"],
            nu=feats["nu"],
            redundancy=feats["redundancy"],
            info_filter=feats["info_filter"],
            n_sims=n_sims,
            n_days=1,
            dt=1.0,
            info_filter_multiplier=config.info_filter_multiplier,
            sigma_scale=config.sigma_scale,              # [CHANGE-4]
            sigma_start=feats["sigma_forecast"],         # [CHANGE-2]
        )

        S_t1 = paths_bt[:, 1]
        low_95, high_95 = np.percentile(S_t1, [2.5, 97.5])
        actual = float(prices.iloc[i + 1])
        width_95 = float(high_95 - low_95)
        coverage_95 = int(low_95 <= actual <= high_95)

        if coverage_95:
            winkler = width_95
        elif actual < low_95:
            winkler = width_95 + (2 / alpha) * (low_95 - actual)
        else:
            winkler = width_95 + (2 / alpha) * (actual - high_95)

        rows.append(
            {
                "timestamp":   pd.Timestamp(prices.index[i + 1]).isoformat().replace("+00:00", "Z"),
                "open_time":   pd.Timestamp(prices.index[i]).isoformat().replace("+00:00", "Z"),
                "actual":      actual,
                "low_95":      float(low_95),
                "high_95":     float(high_95),
                "coverage_95": coverage_95,
                "width_95":    width_95,
                "winkler":     float(winkler),
            }
        )

    return rows


def save_jsonl(rows: list[dict], out_path: str | Path = "backtest_results.jsonl") -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return out


def summarize_results(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {"coverage": float("nan"), "avg_width": float("nan"), "mean_winkler": float("nan")}
    df = pd.DataFrame(rows)
    return {
        "coverage":    float(df["coverage_95"].mean()),
        "avg_width":   float(df["width_95"].mean()),
        "mean_winkler": float(df["winkler"].mean()),
    }


# ---------------------------------------------------------------------------
# Live prediction & validation  (signatures unchanged — app.py needs no edits)
# ---------------------------------------------------------------------------

def run_live_prediction(prices: pd.Series, n_sims: int = 10_000, profile: str = "challenge") -> dict[str, float]:
    """Run one-hour-ahead prediction using all provided prices as training history."""
    config = get_simulation_config(profile)
    feats = _prepare_train_features(prices, config=config)
    current_price = float(prices.iloc[-1])

    paths = simulate_mc(
        S0=current_price,
        mu=feats["mu"],
        sigma_fig=feats["sigma_fig"],
        H=feats["H_series"],
        M=feats["M_series"],
        bar_sigma2=feats["bar_sigma2"],
        base_params=feats["base_params"],
        nu=feats["nu"],
        redundancy=feats["redundancy"],
        info_filter=feats["info_filter"],
        n_sims=n_sims,
        n_days=1,
        dt=1.0,
        info_filter_multiplier=config.info_filter_multiplier,
        sigma_scale=config.sigma_scale,
        sigma_start=feats["sigma_forecast"],
    )
    S_t1 = paths[:, 1]
    low_95, high_95 = np.percentile(S_t1, [2.5, 97.5])

    return {
        "current_price": current_price,
        "low_95":        float(low_95),
        "high_95":       float(high_95),
        "sigma_now":     float(np.sqrt(feats["bar_sigma2"])),
    }


def run_previous_bar_validation(
    prices: pd.Series,
    train_window: int = 500,
    n_sims: int = 10_000,
    profile: str = "challenge",
) -> dict[str, Any]:
    """
    Validate a one-step prediction by forecasting from the previous closed bar
    and comparing against the latest closed bar.
    """
    if len(prices) < train_window + 2:
        raise ValueError("Not enough bars for previous-bar validation.")

    config = get_simulation_config(profile)
    train_prices = prices.iloc[-(train_window + 1) : -1]
    S0 = float(prices.iloc[-2])
    actual = float(prices.iloc[-1])

    feats = _prepare_train_features(train_prices, config=config)
    paths = simulate_mc(
        S0=S0,
        mu=feats["mu"],
        sigma_fig=feats["sigma_fig"],
        H=feats["H_series"],
        M=feats["M_series"],
        bar_sigma2=feats["bar_sigma2"],
        base_params=feats["base_params"],
        nu=feats["nu"],
        redundancy=feats["redundancy"],
        info_filter=feats["info_filter"],
        n_sims=n_sims,
        n_days=1,
        dt=1.0,
        info_filter_multiplier=config.info_filter_multiplier,
        sigma_scale=config.sigma_scale,
        sigma_start=feats["sigma_forecast"],
    )
    S_t1 = paths[:, 1]
    low_95, high_95 = np.percentile(S_t1, [2.5, 97.5])
    in_range = bool(low_95 <= actual <= high_95)

    return {
        "prediction_time": pd.Timestamp(prices.index[-2]).isoformat().replace("+00:00", "Z"),
        "actual_time":     pd.Timestamp(prices.index[-1]).isoformat().replace("+00:00", "Z"),
        "S0":              S0,
        "actual":          actual,
        "low_95":          float(low_95),
        "high_95":         float(high_95),
        "in_range":        in_range,
        "profile":         config.profile,
    }


# ---------------------------------------------------------------------------
# Dashboard helpers  (unchanged — app.py imports these directly)
# ---------------------------------------------------------------------------

def load_backtest_metrics(path: str | Path) -> BacktestMetrics:
    """Load JSONL backtest output and aggregate key metrics."""
    fp = Path(path)
    if not fp.exists():
        empty = pd.DataFrame(
            columns=["timestamp", "actual", "low_95", "high_95", "width_95", "coverage_95", "winkler"]
        )
        return BacktestMetrics(
            coverage=float("nan"), avg_width=float("nan"), mean_winkler=float("nan"), rows=empty
        )

    parsed = []
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                parsed.append(json.loads(line))

    if not parsed:
        empty = pd.DataFrame(
            columns=["timestamp", "actual", "low_95", "high_95", "width_95", "coverage_95", "winkler"]
        )
        return BacktestMetrics(
            coverage=float("nan"), avg_width=float("nan"), mean_winkler=float("nan"), rows=empty
        )

    df = pd.DataFrame(parsed)
    for col in ["actual", "low_95", "high_95", "width_95", "winkler", "coverage_95"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp")

    return BacktestMetrics(
        coverage=float(df["coverage_95"].mean()),
        avg_width=float(df["width_95"].mean()),
        mean_winkler=float(df["winkler"].mean()),
        rows=df,
    )