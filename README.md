# BTC Forecast Dashboard

## 1) Install dependencies

```bash
pip install -r requirements.txt
```

## 2) Run backtest and generate JSONL

```bash
python run_backtest.py --bars 1300 --train-window 500 --test-bars 720 --n-sims 2000 --profile challenge --out backtest_results.jsonl --plot
```

Notes:
- `--bars 1300` gives enough history to support `500` train + `720` test + next-bar scoring.
- Output is one JSON object per line in `backtest_results.jsonl`.
- Available profiles: `challenge` (original settings) and `tuned` (narrower intervals).

## 3) Run previous-bar real-time validation

```bash
python realtime_validation.py --bars 505 --train-window 500 --n-sims 10000 --profile challenge
```

## 4) Launch dashboard

```bash
streamlit run app.py
```

The app auto-refreshes every 5 minutes and fetches BTC hourly bars from Binance data API.
