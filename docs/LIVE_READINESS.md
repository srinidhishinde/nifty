# Live Readiness Gates

This application is not considered live-ready merely because it starts or because unit tests pass. Live readiness requires passing the software gates below and an authenticated, read-only validation against the user's Kotak Neo session. Live order submission must remain disabled until separately reviewed and explicitly authorized.

## Non-negotiable data-source rules

- Kotak Neo is the authoritative live broker/data source for the existing live paths.
- Never synthesize option contracts or silently substitute Yahoo/research data for a required live input.
- A saved prior-session snapshot must be labeled HISTORICAL and must not authorize a live trade.
- Keep WebSocket feed lifecycle independent from REST quote enrichment.
- Preserve existing option-chain refresh and strike-selection behavior unless a separately reviewed change explicitly targets it.
- On missing, stale, malformed, or contradictory required market data, fail closed: show unavailable/blocked and do not authorize an order.

## Readiness gates

### Gate 1 — Reproducible local validation

Run from the project root in the project's active virtual environment:

```powershell
python -m pytest tests\marketdata\test_kotak_neo_provider.py -q
python -m compileall -q app broker config marketdata prediction tests
git diff --check
```

Then run the relevant broker, forecast, option-chain, persistence, and full regression suites. Record the exact command and result. Do not describe unexecuted tests as passing.

### Gate 2 — Broker SDK compatibility

- Confirm the installed `kotakneoapi` version and inspect the actual `NeoAPI` constructor and REST behavior.
- Use the real installed SDK in a no-order smoke test; mocks alone do not validate SDK constructor compatibility.
- Keep credentials, TOTP, MPIN, tokens, and account identifiers out of logs and committed files.
- Treat network, timeout, authentication, and HTTP 4xx errors as distinct failures. Do not blindly retry order submissions.

### Gate 3 — Read-only live market data

With valid user-authenticated credentials and while the relevant exchange data is available, verify:
- NIFTY index quote is positive, recent, and sourced from Kotak.
- NIFTY and MCX WebSocket subscriptions receive real ticks and reconnect safely.
- Option-chain response is non-empty, exchange/underlying/expiry match the request, and CE/PE contracts have valid tokens and strikes.
- Quote-enrichment requests contain at most 40 symbols; each failure is visible and never represented as a fresh quote.
- MCX five-minute candles are built from the real Kotak SFeed pipeline and persisted without duplicate candle keys or cumulative-volume spikes.
- A closed-market or stale-data view is explicitly HISTORICAL/STALE, never LIVE.

This gate is read-only. Do not submit orders as part of a connectivity smoke test.

### Gate 4 — Strategy and forecast validation

- Verify time ordering, duplicate handling, missing bars, timezone/session boundaries, and incomplete candles.
- Use chronological/walk-forward splits; never train or select features using future target data.
- Record forecast input timestamp, target horizon, model/version, and actual outcome after the target period closes.
- Evaluate forecast error and trading costs separately. Forecast accuracy is not proof of profitability.
- Compare any refactor against existing outputs before replacing the established path.

### Gate 5 — Risk and order controls

- Paper mode is the default; live order submission stays disabled by default.
- A trade must be blocked when required market data is stale/invalid, risk limits fail, or account/order state cannot be reconciled.
- Check position size, available funds, max loss per trade, daily loss, exposure, and trade-count limits before order creation.
- Use explicit order states and idempotency keys. After an ambiguous timeout, reconcile broker order state before any retry.
- Require a separate reviewed enablement change before live order submission can be turned on.

## Quote batching

The provider uses a conservative batch size of 40 symbols. The broker has returned HTTP 400 for requests containing exactly 50 symbols, despite the response text saying the maximum is 50. Regression coverage asserts that 81 contracts are requested in batches of 40, 40, and 1.

## Current status

This checklist is a gate, not a certification. A change is not production-approved until local automated tests and the authenticated read-only checks above have actual recorded results. Never infer live readiness from source review alone.
