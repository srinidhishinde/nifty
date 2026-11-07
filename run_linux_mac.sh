#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

test -f requirements.txt || { echo "Run this script from the repository root."; exit 1; }

if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/run_uat.py
exec .venv/bin/python -m streamlit run app/main.py
