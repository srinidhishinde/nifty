# Evidence-grade validation standard

The system's acceptance test is deliberately stricter than “the backtest made money”.

## ₹1 lakh / 100-trading-day gate

The default research profile uses:

- starting capital: ₹100,000
- risk budget: ₹1,000 per trade
- daily loss lock: ₹2,000
- maximum 5 trades/day
- whole-lot sizing
- next-bar execution
- explicit slippage and brokerage
- no synthetic data for a real acceptance result

A dataset must contain at least **100 distinct trading dates**. The report fails rather than silently treating fewer days as a 100-day study.

## Required acceptance checks

1. OHLCV structural quality.
2. At least 100 trading days.
3. At least 30 closed trades.
4. Maximum drawdown ≤ 12%.
5. Profit factor ≥ 1.10.
6. Positive expectancy.
7. Positive P&L after doubled recorded costs plus 10 bps adverse notional slippage.
8. Bootstrap Monte Carlo probability of ending below starting capital ≤ 5%.

These thresholds are research gates, not guarantees of profitability.

## Walk-forward

The default windows are 60 training days, 20 validation days and 20 chronological out-of-sample days, with a documented one-bar purge. The next window advances chronologically; there is no random train/test shuffle.

For a production study, repeat across multiple windows and market regimes. A single 100-day PASS is not sufficient evidence for live capital.

## Intrabar ambiguity

When OHLC data show both a stop and target touched in one candle, the conservative execution convention is **stop first** unless lower-timeframe/tick data proves otherwise. This avoids optimistic sequencing.

## Stress testing

The report applies doubled recorded costs and an adverse notional slippage charge. Additional scenarios should be run for 1.5×/2× costs, delayed fills, missed trades and volatility shocks.

## Option/BTST/closing-session evidence

BTST and 15:15–15:40 recommendations must use timestamped real option-chain observations. No option contract, bid/ask, OI or IV is invented.

NSE currently lists the regular equity-derivatives session as 09:15–15:40. The 15:40–16:00 closing session shown on NSE's market-timings page is a cash-market session, so the application does not label the derivatives research window as a CAS auction. citeturn0search1

NSE's NIFTY derivatives documentation states that an option contract's closing price uses the last-half-hour weighted average when traded in that period; this is why the final half-hour is treated as a research window, not as a special auction. citeturn0search10

Contract metadata such as expiry and lot size must come from the applicable NSE contract information rather than hard-coded assumptions. citeturn0search0turn0search4


### Canonical decision-path requirement
Backtests must use the same `generate_signal()` decision gates as live evaluation, including evidence-group, confidence, volatility, signal-separation and reward/risk checks. Lower-level rule evaluation is for diagnostics only.
