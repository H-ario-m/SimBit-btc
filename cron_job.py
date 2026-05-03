import os
import sys
import pandas as pd

# Fallback profile (same as default in dashboard)
PROFILE = "precision"

def main():
    if not os.environ.get("DATABASE_URL"):
        print("Error: DATABASE_URL environment variable is not set.")
        sys.exit(1)
        
    print("Initializing environment...")
    import db
    from model import fetch_btc_ohlc, run_live_prediction, run_previous_bar_validation

    def _safe_window(series: pd.Series, train_window: int = 500) -> tuple[pd.Series, pd.Series, int]:
        if len(series) < 120:
            raise ValueError("Not enough bars available for model fitting.")
        tw = min(train_window, len(series) - 2)
        tw = max(tw, 100)
        pred_slice = series.iloc[-tw:]
        val_slice = series.iloc[-(tw + 2) :]
        return pred_slice, val_slice, tw

    print("Fetching latest BTC OHLC data...")
    ohlc = fetch_btc_ohlc(n_bars=620, interval="1h")
    pred_slice, val_slice, tw = _safe_window(ohlc["close"], train_window=500)
    
    print(f"Running previous bar validation for profile: {PROFILE}...")
    validation = run_previous_bar_validation(val_slice, train_window=tw, n_sims=3000, profile=PROFILE)
    
    print(f"Running live prediction for profile: {PROFILE}...")
    pred = run_live_prediction(pred_slice, n_sims=10000, profile=PROFILE)
    
    now_utc = pd.Timestamp.now(tz="UTC").isoformat().replace("+00:00", "Z")
    # Target time is the end of the next candle
    target_time = (ohlc.index[-1] + pd.Timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    
    print(f"Saving new prediction for target: {target_time}...")
    db.save_prediction(
        fetched_at=now_utc,
        target_time=target_time,
        low_95=pred["low_95"],
        high_95=pred["high_95"],
        current_price=pred["current_price"],
        profile=PROFILE
    )
    
    print(f"Updating actual price for previous target: {validation.get('actual_time')}...")
    db.update_actual_price(
        target_time=validation.get("actual_time", ""),
        actual_price=validation.get("actual", 0.0)
    )
    
    print("Cron job execution completed successfully!")

if __name__ == "__main__":
    main()
