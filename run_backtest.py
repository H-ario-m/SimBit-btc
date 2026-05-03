import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from model import fetch_btc_hourly, fetch_btc_ohlc, run_backtest, save_jsonl, summarize_results
from lgbm_model import run_lgbm_backtest

def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict no-peek BTC 1h backtest and save JSONL results.")
    parser.add_argument("--bars",         type=int,   default=1300,       help="Number of hourly bars to fetch from Binance.")
    parser.add_argument("--train-window", type=int,   default=500,        help="Training window size in bars.")
    parser.add_argument("--test-bars",    type=int,   default=720,        help="Requested number of rolling test predictions.")
    parser.add_argument("--n-sims",       type=int,   default=2000,       help="Monte Carlo paths per test step.")
    parser.add_argument("--lgbm",         action="store_true",            help="Run LightGBM Quantile Regression instead of GBM.")
    parser.add_argument(
        "--profile", type=str, default="precision",
        # [CHANGE-6] Added "precision" to choices.
        choices=["challenge", "tuned", "precision"],
        help=(
            "Simulation profile.\n"
            "  challenge  — original settings (coverage ~0.97-0.99, wide bands)\n"
            "  tuned      — moderate tightening (coverage ~0.95-0.97)\n"
            "  precision  — GJR-GARCH + calibrated sigma_scale (target ~0.94-0.96)\n"
        ),
    )
    parser.add_argument("--out",  type=str, default="backtest_results.jsonl", help="Output JSONL path.")
    parser.add_argument("--plot", action="store_true",                         help="Plot actual vs predicted 95% bands.")
    args = parser.parse_args()

    print(f"Fetching {args.bars} bars from Binance...")
    if args.lgbm:
        ohlc = fetch_btc_ohlc(n_bars=args.bars)
        prices = ohlc["close"]
        print(f"Fetched {len(ohlc)} bars | Latest: {ohlc.index[-1]} | Price: {prices.iloc[-1]:.2f}")
        rows = run_lgbm_backtest(
            ohlc=ohlc,
            train_window=args.train_window,
            test_bars=args.test_bars,
        )
    else:
        prices = fetch_btc_hourly(n_bars=args.bars)
        print(f"Fetched {len(prices)} bars | Latest: {prices.index[-1]} | Price: {prices.iloc[-1]:.2f}")
        rows = run_backtest(
            prices=prices,
            train_window=args.train_window,
            test_bars=args.test_bars,
            n_sims=args.n_sims,
            profile=args.profile,
        )
        
    out_path = save_jsonl(rows, out_path=args.out)
    summary = summarize_results(rows)

    print(f"\n--- Results ---")
    print(f"Profile:      {args.profile}")
    print(f"Backtest rows:{len(rows)}")
    print(f"Saved:        {out_path.resolve()}")
    print(f"Coverage:     {summary['coverage']:.4f}  (target: 0.94-0.96)")
    print(f"Avg Width:    {summary['avg_width']:.2f} USDT")
    print(f"Mean Winkler: {summary['mean_winkler']:.2f}  (lower is better)")

    # Sigma_scale guidance: if coverage is too high, lower sigma_scale.
    cov = summary["coverage"]
    if cov > 0.97:
        print("\n  Coverage too high — reduce sigma_scale in get_simulation_config('precision') and rerun.")
        print("  Suggested: sigma_scale = 0.78")
    elif cov < 0.92:
        print("\n  Coverage too low — increase sigma_scale in get_simulation_config('precision') and rerun.")
        print("  Suggested: sigma_scale = 0.88")
    else:
        print(f"\n  Coverage {cov:.3f} is within target range. This profile is submission-ready.")

    if args.plot and rows:
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df["timestamp"], df["actual"],  label="Actual Price", color="#1e90ff", linewidth=1.5)
        ax.fill_between(df["timestamp"], df["low_95"], df["high_95"],
                        alpha=0.2, color="#1e90ff", label="95% Band")
        misses = df[df["coverage_95"] == 0]
        if not misses.empty:
            ax.scatter(misses["timestamp"], misses["actual"],
                       color="#e05252", s=22, label=f"Miss ({len(misses)})", zorder=3)

        ax.set_title(
            f"BTCUSDT Backtest [{args.profile}] — "
            f"Coverage: {summary['coverage']:.3f} | "
            f"Avg Width: ${summary['avg_width']:.0f} | "
            f"Winkler: {summary['mean_winkler']:.0f}"
        )
        ax.set_xlabel("Timestamp (UTC)")
        ax.set_ylabel("Price (USDT)")
        ax.legend()
        ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()