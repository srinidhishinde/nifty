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
