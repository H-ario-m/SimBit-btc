import argparse

from model import fetch_btc_hourly, run_previous_bar_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate previous-hour forecast against latest closed BTC 1h bar.")
    parser.add_argument("--bars", type=int, default=505, help="Number of bars to fetch.")
    parser.add_argument("--train-window", type=int, default=500, help="Training bars used for validation.")
    parser.add_argument("--n-sims", type=int, default=10000, help="Monte Carlo paths for validation.")
    parser.add_argument("--profile", type=str, default="challenge", choices=["challenge", "tuned"], help="Simulation profile.")
    args = parser.parse_args()

    prices = fetch_btc_hourly(n_bars=args.bars)
    out = run_previous_bar_validation(
        prices=prices,
        train_window=args.train_window,
        n_sims=args.n_sims,
        profile=args.profile,
    )

    print("--- REAL-TIME VALIDATION ---")
    print(f"Profile: {out['profile']}")
    print(f"Prediction Time (S0): {out['prediction_time']}")
    print(f"Actual Price Time:    {out['actual_time']}")
    print(f"Actual Price:         {out['actual']:.2f} USDT")
    print(f"Predicted Low (95%):  {out['low_95']:.2f} USDT")
    print(f"Predicted High (95%): {out['high_95']:.2f} USDT")
    print(f"Status: {'WITHIN RANGE' if out['in_range'] else 'OUT OF BOUNDS'}")


if __name__ == "__main__":
    main()
