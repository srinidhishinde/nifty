# Advanced trading design

## Decision hierarchy

1. Data quality gate
2. 15-minute regime
3. 10-minute market structure
4. 5-minute setup
5. 1-minute execution when reliable data exists
6. Technical + price-action evidence
7. Options/OI/PCR evidence
8. Cross-market and event/news context
9. Calibrated probability and expected value
10. Risk sizing and execution simulation

Lower timeframes cannot override a higher-timeframe risk block.

## BTST

BTST is a separate workflow. It requires a real option-chain snapshot, liquidity and spread checks, an explicit underlying bias, and overnight-gap risk disclosure. Entry is based on the ask when available; the system abstains when data is insufficient.

## NIFTY 15:15–15:40

The dashboard provides a closing-session research engine. NSE states that the normal derivatives market closes at 15:40. NSE also documents a 15:40–16:00 post-close session for the cash normal-market segment. Therefore the dashboard does not call 15:15–15:40 derivatives trading a CAS auction. It treats this period as a derivatives closing decision window and uses a hard 15:40 exit. The futures settlement methodology also references the last half-hour weighted average price for index futures, making the closing window economically relevant.

## Option selection

No fixed PCR threshold is sufficient. PCR is computed as Put OI / Call OI and is interpreted with price, OI change, liquidity, IV and structure. OI alone is not directional.

## Safety

The system may return WAIT even when a directional prediction exists. A recommendation is not an order. Production execution remains disabled unless all explicit broker/environment safeguards are enabled.
