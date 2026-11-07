# AI Derivatives Terminal

NIFTY + MCX research, backtesting and paper-trading platform.

## Instruments
- NIFTY
- CRUDEOIL
- NATURALGAS
- COPPER
- SILVER
- GOLD

## Timeframes
- 5m
- 15m

## Safety
Live trading is disabled by default. The Market Analysis page is signal/context only and does not place orders.

## Setup

### Windows
1. Run `py -3.11 -m venv .venv`
2. Activate: `.\.venv\Scripts\Activate.ps1`
3. Run `python -m pip install --upgrade pip`
4. Run `python -m pip install -r requirements.txt`
5. Copy `.env.example` to `.env`
6. Run `python scripts/run_uat.py`
7. Run `python -m streamlit run app/main.py`

Or double-click `run_windows.bat` from the repository root. It creates the local environment, installs dependencies, runs UAT and launches Streamlit.

### macOS / Linux
1. Run `python3 -m venv .venv`
2. Activate: `source .venv/bin/activate`
3. Run `python -m pip install --upgrade pip`
4. Run `python -m pip install -r requirements.txt`
5. Copy `.env.example` to `.env`
6. Run `python scripts/run_uat.py`
7. Run `python -m streamlit run app/main.py`

Or run `chmod +x run_linux_mac.sh && ./run_linux_mac.sh`.

Keep the project root as the current directory when launching Streamlit so imports such as `config`, `strategy`, `backtest` and `marketdata` resolve correctly.

## Market Analysis page

After launching the Streamlit app, select **Market Analysis** from the left sidebar. The page provides tabs for:

- **Overview:** combines independent evidence groups and returns a directional bias only when at least two inputs agree; otherwise it returns WAIT.
- **Price Action:** checks recent OHLC structure.
- **Option Positioning:** pairs option premium change with OI change. OI alone is not treated as proof of buyer/writer intent.
- **NIFTY 50 Breadth:** compares advancing and declining constituents; optional `above_vwap` column adds participation context.
- **Validation Notes:** practical safeguards against overfitting.

### Market Analysis CSV formats

1. **NIFTY candles:** CSV with `open, high, low, close`; `timestamp` and `volume` are recommended.
2. **Option chain:** CSV with `option_type` (CE/PE), `ltp_change`, `oi_change`; optional `strike`, `ltp`, `volume`.
3. **NIFTY 50 breadth:** CSV with `symbol`, `change_pct`; optional `above_vwap` (true/false).

The page analyses uploaded data only. It does not fabricate live market data, provide guaranteed predictions, or execute trades. Do not interpret a directional bias as an entry instruction; require a completed entry trigger, defined invalidation and acceptable risk/reward.

## Historical validation in the existing UI

Open the Streamlit app and use **Historical Rule Backtest**:

1. Upload a CSV with required columns: `timestamp, open, high, low, close, volume`.
2. Keep earlier candles in the file before the dates you want to score. These provide indicator warm-up for EMA(50), RSI, MACD, VWAP and ATR.
3. Select the historical date range to score.
4. Click **Run Historical Backtest**.
5. Review closed trades, win rate, wins/losses, net P&L, profit factor, average trade, maximum drawdown, BUY vs SELL win rate, equity/P&L curve and individual rule performance.

The win rate is the percentage of closed simulated trades that were profitable. It is not ML prediction accuracy and is not a guarantee of future returns.

The UI backtest is an OHLCV/rule-engine validation. It is not a historical option-chain replay unless timestamped historical option-contract data is supplied and processed by a dedicated replay engine. A single option-chain snapshot cannot support a 3-minute option-premium backtest.

## Recommended validation

Do not judge the strategy from synthetic demo data. Use real historical data and compare:
- in-sample period
- separate, untouched out-of-sample period
- multiple market regimes
- multiple dates/years where available
- brokerage, taxes, spread and slippage

Compare expectancy, profit factor, drawdown and stability—not win rate alone. Avoid repeatedly tuning thresholds on the same data.

## Important

This repository starts in PAPER mode.

Do not enable live trading until:
1. Data validation passes
2. Backtesting passes
3. Walk-forward testing passes
4. UAT passes
5. Paper trading has been monitored
6. Broker integration has been independently tested

Kotak Neo live integration depends on valid account access, supported market-data APIs and current broker SDK behaviour. A successful dashboard launch does not prove that the broker feed is connected or validated.
