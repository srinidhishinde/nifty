# Integrated failure prediction and prevention

The `reliability.failure_guard` module provides deterministic health checks for
broker connectivity, feed freshness, OHLCV integrity/cadence, forecast evidence,
out-of-sample accuracy versus a predeclared baseline, and explicit risk flags.

## Fail-closed policy

- Missing evidence is UNKNOWN and blocks live order permission.
- Default forecast gate is 80% accuracy on at least 500 scored unseen candles,
  and the result must beat a predeclared baseline.
- Stale/missing market data, invalid candles, disconnected broker, disabled
  execution configuration, or risk flags block order permission.
- Findings can be appended as sanitized JSONL incident records.
- This module does not send orders, auto-authenticate the broker, or claim that
  failures can always be predicted in advance. It detects known unsafe states
  and prevents execution when evidence is inadequate.

## Integrating at an order boundary

Before every order submission, call `assess_health(...)` with fresh broker and
market evidence. Reject the order unless `report.live_orders_allowed` is true.
The UI health panel is diagnostic; the actual order-submission function must
enforce this check too. Keep LIVE disabled until broker-specific order-path UAT
and independent operator approval have passed.
