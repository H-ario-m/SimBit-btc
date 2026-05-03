# BTC Forecast Dashboard — AlphaI × Polaris Challenge

A high-precision Bitcoin price range forecasting system built to optimize the **Winkler Score**.

##  Deployed URL
[Your Streamlit URL Here]

##  Performance (Precision Profile)
- **Coverage**: 94.7% (Target: 95%)
- **Avg Width**: ~1179 USDT
- **Mean Winkler Score**: 1659

---

## How it Works

The system predicts the 95% probability range for BTC price 1-hour from now. It uses two primary engines:

1.  **GJR-GARCH + Cyber Monte Carlo (Default)**: 
    - Uses GJR-GARCH to model asymmetric volatility (leverage effects).
    - Runs 10,000 Monte Carlo paths using a Student-t distribution to account for "Fat Tails."
    - Dynamically widens bands based on "Entropy" (Information complexity) and Momentum sensors.
2.  **LightGBM Quantile Regression**:
    - A machine learning approach that directly minimizes pinball loss.
    - Features include realized volatility (High-Low range), multi-horizon rolling volume, RSI, and temporal regimes.

### No-Peek Guarantee
The backtest engine uses a strict **rolling window of 500 bars**. For every prediction at hour `T`, the model is re-fitted using only data from `T-500` to `T`. No future data is ever used for feature scaling or model training.

---

##  Profiles

You can switch between these in the sidebar to see how they behave:

-   **`precision` (DEFAULT)**: The most balanced profile. Tuned to hit ~95% coverage while minimizing the Winkler score. It uses a "Cyber" scaling logic that reacts to sudden momentum shifts.
-   **`lgbm`**: The machine learning model. Often produces the tightest bands but can be more sensitive to regime changes.
-   **`tuned`**: A slightly more conservative version of the GARCH model.
-   **`challenge`**: The baseline configuration.

---

##  Deployment & Persistence

-   **Framework**: Streamlit
-   **Database**: Supabase (PostgreSQL)
-   **Persistence**: Live predictions and actual outcomes are saved permanently to the cloud DB, ensuring history survives app reboots.

## 🏃‍♂️ Local Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `streamlit run app.py`
3. (Optional) Run a manual backtest: `python run_backtest.py --profile precision`
