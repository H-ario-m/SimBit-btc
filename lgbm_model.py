import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm
from ta.momentum import RSIIndicator

from model import rolling_entropy

def build_lgbm_features(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Build features for LightGBM including realized volatility."""
    prices = ohlc["close"]
    log_ret = np.log(prices / prices.shift(1))
    
    feats = pd.DataFrame(index=ohlc.index)
    
    # Realized volatility proxy (High-Low range)
    feats["realized_vol"] = np.log(ohlc["high"] / ohlc["low"])
    for lag in [1, 2, 4, 12, 24]:
        feats[f"realized_vol_lag_{lag}"] = feats["realized_vol"].shift(lag)
        
    # Moving averages of realized volatility
    for window in [6, 12, 24, 48]:
        feats[f"realized_vol_ma_{window}h"] = feats["realized_vol"].rolling(window).mean()
    
    # Lagged returns
    for lag in [1, 2, 4, 8, 12, 24]:
        feats[f"ret_lag_{lag}"] = log_ret.shift(lag)
    
    # Rolling vol at multiple horizons
    for window in [6, 12, 24, 48]:
        feats[f"vol_{window}h"] = log_ret.rolling(window).std()
        
    # Volatility ratios
    feats["vol_ratio_6_48"] = feats["vol_6h"] / (feats["vol_48h"] + 1e-9)
    feats["vol_ratio_12_24"] = feats["vol_12h"] / (feats["vol_24h"] + 1e-9)
        
    # Volatility of volatility
    feats["vol_of_vol_24h"] = feats["vol_6h"].rolling(24).std()
    
    # RSI
    feats["rsi_14"] = RSIIndicator(prices, window=14).rsi()
    
    # Entropy
    feats["entropy_24"] = rolling_entropy(log_ret.dropna().reindex(log_ret.index), window=24)
    
    # Calendar features (crypto has volume patterns during US/Asia open)
    feats["hour_of_day"] = ohlc.index.hour
    feats["day_of_week"] = ohlc.index.dayofweek
    
    # Drop NaNs
    return feats.dropna()

def train_lgbm_quantile(X_train: pd.DataFrame, y_train: np.ndarray, alpha: float) -> lgb.LGBMRegressor:
    """Train one quantile model."""
    params = {
        'objective': 'quantile',
        'alpha': alpha,
        'metric': 'quantile',
        'n_estimators': 80,         # Restored to 80
        'learning_rate': 0.05,
        'num_leaves': 7,            # Restored to 7
        'min_child_samples': 15,
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'verbose': -1,
        'random_state': 42
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    return model

def run_lgbm_backtest(
    ohlc: pd.DataFrame, 
    train_window: int = 500, 
    test_bars: int = 720
) -> list[dict]:
    """Rolling backtest for LightGBM quantile regression."""
    feats_all = build_lgbm_features(ohlc)
    
    prices_aligned = ohlc["close"].reindex(feats_all.index)
    log_ret_aligned = np.log(prices_aligned / prices_aligned.shift(1)).dropna()
    
    # Re-align everything to log_ret_aligned
    feats_aligned = feats_all.reindex(log_ret_aligned.index)
    prices_aligned = prices_aligned.reindex(log_ret_aligned.index)
    
    results = []
    available = len(prices_aligned) - train_window - 1
    run_limit = min(test_bars, available)
    
    model_lo, model_hi = None, None
    
    for i in tqdm(range(train_window, train_window + run_limit), desc="LGBM Backtest"):
        X_train = feats_aligned.iloc[i - train_window : i]
        y_train = log_ret_aligned.iloc[i - train_window + 1 : i + 1].values
        
        X_pred = feats_aligned.iloc[i : i + 1]
        S0 = float(prices_aligned.iloc[i])
        actual = float(prices_aligned.iloc[i + 1])
        
        if (i - train_window) % 24 == 0 or model_lo is None:
            # We target slightly wider in-sample bounds (0.005/0.995) to achieve ~0.95 out-of-sample coverage
            model_lo = train_lgbm_quantile(X_train, y_train, alpha=0.005)
            model_hi = train_lgbm_quantile(X_train, y_train, alpha=0.995)
            
        ret_lo = model_lo.predict(X_pred)[0]
        ret_hi = model_hi.predict(X_pred)[0]
        
        low_95 = float(S0 * np.exp(ret_lo))
        high_95 = float(S0 * np.exp(ret_hi))
        
        if low_95 > high_95:
            low_95, high_95 = high_95, low_95
            
        width_95 = high_95 - low_95
        coverage_95 = int(low_95 <= actual <= high_95)
        alpha_w = 0.05
        
        if coverage_95:
            winkler = width_95
        elif actual < low_95:
            winkler = width_95 + (2 / alpha_w) * (low_95 - actual)
        else:
            winkler = width_95 + (2 / alpha_w) * (actual - high_95)
            
        results.append({
            "timestamp": pd.Timestamp(prices_aligned.index[i + 1]).isoformat().replace("+00:00", "Z"),
            "open_time": pd.Timestamp(prices_aligned.index[i]).isoformat().replace("+00:00", "Z"),
            "actual": actual,
            "low_95": low_95,
            "high_95": high_95,
            "coverage_95": coverage_95,
            "width_95": width_95,
            "winkler": float(winkler)
        })
        
    return results

def run_lgbm_live_prediction(ohlc: pd.DataFrame, train_window: int = 500) -> dict:
    """Run live prediction for the next unseen bar using LGBM."""
    feats_all = build_lgbm_features(ohlc)
    prices = ohlc["close"].reindex(feats_all.index)
    log_ret = np.log(prices / prices.shift(1)).dropna()
    
    feats_aligned = feats_all.reindex(log_ret.index)
    prices_aligned = prices.reindex(log_ret.index)
    
    # Train on the most recent `train_window` rows
    X_train = feats_aligned.iloc[-train_window:]
    y_train = log_ret.iloc[-train_window:].values
    
    model_lo = train_lgbm_quantile(X_train, y_train, alpha=0.005)
    model_hi = train_lgbm_quantile(X_train, y_train, alpha=0.995)
    
    # Predict using the very last known feature state
    X_pred = feats_aligned.iloc[-1:]
    S0 = float(prices_aligned.iloc[-1])
    
    ret_lo = model_lo.predict(X_pred)[0]
    ret_hi = model_hi.predict(X_pred)[0]
    
    low_95 = float(S0 * np.exp(ret_lo))
    high_95 = float(S0 * np.exp(ret_hi))
    
    if low_95 > high_95:
        low_95, high_95 = high_95, low_95
        
    return {
        "current_price": S0,
        "low_95": low_95,
        "high_95": high_95
    }

def run_lgbm_previous_bar_validation(ohlc: pd.DataFrame, train_window: int = 500) -> dict:
    """Validate the LGBM model on the most recently closed bar."""
    feats_all = build_lgbm_features(ohlc)
    prices = ohlc["close"].reindex(feats_all.index)
    log_ret = np.log(prices / prices.shift(1)).dropna()
    
    feats_aligned = feats_all.reindex(log_ret.index)
    prices_aligned = prices.reindex(log_ret.index)
    
    # Train up to the *second to last* bar
    X_train = feats_aligned.iloc[-(train_window + 1):-1]
    y_train = log_ret.iloc[-(train_window + 1):-1].values
    
    model_lo = train_lgbm_quantile(X_train, y_train, alpha=0.005)
    model_hi = train_lgbm_quantile(X_train, y_train, alpha=0.995)
    
    # Predict using the second to last bar's features (to forecast the last bar)
    X_pred = feats_aligned.iloc[-2:-1]
    S0 = float(prices_aligned.iloc[-2])
    actual = float(prices_aligned.iloc[-1])
    
    ret_lo = model_lo.predict(X_pred)[0]
    ret_hi = model_hi.predict(X_pred)[0]
    
    low_95 = float(S0 * np.exp(ret_lo))
    high_95 = float(S0 * np.exp(ret_hi))
    
    if low_95 > high_95:
        low_95, high_95 = high_95, low_95
        
    in_range = low_95 <= actual <= high_95
    
    return {
        "prediction_time": pd.Timestamp(prices_aligned.index[-2]).isoformat().replace("+00:00", "Z"),
        "actual_time": pd.Timestamp(prices_aligned.index[-1]).isoformat().replace("+00:00", "Z"),
        "actual": actual,
        "low_95": low_95,
        "high_95": high_95,
        "in_range": in_range,
    }
