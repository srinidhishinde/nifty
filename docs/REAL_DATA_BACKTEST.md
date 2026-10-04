# Real-data backtesting

Use **real NIFTY futures 5-minute OHLCV** for the strict rule-engine validation.

## Required CSV

Columns:

`timestamp,open,high,low,close,volume`

Optional raw-data columns:

`contract_symbol,expiry,neo_symbol,oi`

The rule engine requires non-negative, non-zero OHLC and non-negative volume. Keep timezone-aware IST timestamps when possible.

## Download real data from Kotak Neo

The repository contains:

`scripts/download_kotak_nifty_futures.py`

Example:

```powershell
python scripts/download_kotak_nifty_futures.py --start 2026-05-01 --end 2026-10-04
```

The downloader:
- uses the configured `NEO_CONSUMER_KEY`;
- discovers NIFTY futures from Kotak Neo;
- downloads 5-minute candles in API-safe chunks;
- filters 09:15–15:30 IST;
- validates OHLCV;
- removes duplicates;
- writes raw contract metadata;
- writes a clean backtest CSV only from validated observations;
- never fabricates missing history.

**Important:** Kotak historical data is not a guaranteed expired-contract archive. If the requested period cannot be fully covered by actual returned contracts, the manifest must be treated as incomplete rather than filling the gap with another contract.

## Strict ₹1 lakh test

Run:

```powershell
python scripts/run_backtest_validation.py \
  --csv data/backtest/kotak_nifty/nifty_futures_5m_backtest.csv \
  --capital 100000 \
  --risk 1000 \
  --lot-size 65 \
  --slippage 0.25 \
  --brokerage 10
```

The strict engine:
- generates signals at bar close;
- enters on the next bar open;
- uses futures price movement × quantity for P&L;
- sizes in whole lots;
- applies slippage and brokerage;
- limits daily trades;
- applies a 3% daily loss lockout;
- forces intraday exits;
- records an equity curve and validation manifest.

Do not treat a backtest as evidence of future profitability. Use multiple regimes and out-of-sample periods.
