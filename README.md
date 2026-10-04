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

Live trading is disabled by default.

## Setup

Create virtual environment:

python -m venv .venv

Activate:

.venv\Scripts\activate

Install:

pip install -r requirements.txt

Copy environment:

copy .env.example .env

Run tests:

python scripts/run_uat.py

Run UI:

streamlit run app/main.py

## Important

This repository starts in PAPER mode.

Do not enable live trading until:

1. Data validation passes
2. Backtesting passes
3. Walk-forward testing passes
4. UAT passes
5. Paper trading has been monitored
6. Broker integration has been independently tested


## Historical validation in the UI

Open the Streamlit app and use **Historical Rule Backtest**:

1. Upload a CSV with these required columns:
   `timestamp, open, high, low, close, volume`
2. Keep earlier candles in the file before the dates you want to score. These provide indicator warm-up for EMA(50), RSI, MACD, VWAP and ATR.
3. Select the historical date range to score.
4. Click **Run Historical Backtest**.
5. Review:
   - Closed trades
   - Win rate (historical trade accuracy)
   - Wins / losses
   - Net P&L and return
   - Profit factor
   - Average trade
   - Maximum drawdown
   - BUY vs SELL win rate
   - Equity/P&L curve
   - Individual rule performance
   - Full historical trade list

**Important:** the win rate is the percentage of closed simulated trades that were profitable. It is not ML prediction accuracy and is not a guarantee of future returns.

For rules that require market-context fields, include these optional CSV columns when available:

- `sentiment` or `news_sentiment`
- `bid`
- `ask`
- `spread`

Without those fields, the corresponding sentiment/order-book rules cannot be evaluated from the historical file.

### Recommended validation

Do not judge the strategy from synthetic demo data. Use real historical data and compare:

- in-sample period
- separate out-of-sample period
- multiple market regimes
- multiple years/dates where available

The UI backtest is an OHLCV/rule-engine validation. It is not a historical option-chain replay unless the repository is extended with historical option-contract data.

## Windows setup

From PowerShell:

```powershell
cd C:\Users\srini\Downloads\ai_derivatives_terminal\nifty

deactivate 2>$null
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -c "import sys; print(sys.executable); print(sys.version)"

python scripts/run_uat.py

python -m streamlit run app/main.py
```

Keep the project root as the current directory when launching Streamlit so imports such as `config`, `strategy`, `backtest` and `marketdata` resolve correctly.


### Option-chain snapshot CSV

The application also accepts the wide CE/PE option-chain export format containing fields such as:

- Calls/Puts OI and change in OI
- Calls/Puts volume
- Calls/Puts IV
- Calls/Puts delta, theta and vega
- Calls/Puts built-up
- Calls/Puts LTP change percentage
- Strike

This is analyzed as an **option-chain snapshot**, not converted into OHLCV candles. A snapshot does not contain candle open/high/low/close history, timestamps, or bid/ask spread history, so it cannot legitimately be used as the historical seven-rule candle backtest.

Rules 1–5 and 7 need candle/context history; Rule 6 needs bid/ask and spread history. The UI therefore keeps snapshot analysis separate from historical backtesting rather than inventing missing data.

## Kotak Neo production decision source

The production/paper decision path uses **Kotak Neo**. Yahoo Finance is intentionally isolated to historical research and independent validation; it is not a silent live fallback.

Before running the production decision panel, configure:

- `NEO_CONSUMER_KEY`
- `NEO_MOBILE`
- `NEO_UCC`
- `NEO_MPIN`
- `NEO_NIFTY_NEOSYMBOL` — the current `exchange_segment|instrument_token` used by Kotak Neo historical candles.

Kotak Neo's current SDK requires the historical API to receive a Neo symbol in this form, and the 5-minute historical endpoint has a 30-day request limit. citeturn1search2turn2search0

## Dashboard ChatGPT assistant

The dashboard includes a bottom-right **Ask ChatGPT** popover. It can explain the current decision, rule/ML/ensemble state, data-quality gates, option-chain observations, risk controls, backtests, and general platform questions.

Configure:

```
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-6-luna
```

The assistant receives only a sanitized dashboard snapshot and is explicitly prevented from overriding trading gates or exposing broker credentials. The integration uses OpenAI's current Responses API rather than the retired Assistants API. citeturn0search0turn0search2

**Important:** the ChatGPT assistant is explanatory/advisory. It does not place or bypass orders.

