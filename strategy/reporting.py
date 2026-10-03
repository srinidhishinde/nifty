from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


def build_daily_report(
    signals: pd.DataFrame,
    trades: pd.DataFrame,
    report_date: str | None = None,
) -> dict:
    """Create a serializable daily signal/trade/ROI report."""
    date_value = report_date or datetime.now().strftime("%Y-%m-%d")
    signal_count = len(signals)
    trade_count = len(trades)
    pnl = float(trades["pnl"].sum()) if not trades.empty and "pnl" in trades else 0.0
    wins = int((trades["pnl"] > 0).sum()) if not trades.empty and "pnl" in trades else 0
    losses = int((trades["pnl"] < 0).sum()) if not trades.empty and "pnl" in trades else 0
    roi = float(trades["roi_pct"].sum()) if not trades.empty and "roi_pct" in trades else 0.0

    return {
        "date": date_value,
        "signals": signal_count,
        "trades": trade_count,
        "wins": wins,
        "losses": losses,
        "net_pnl": round(pnl, 2),
        "roi_pct": round(roi, 4),
    }


def write_report(report: dict, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([report]).to_csv(output, index=False)
    return output
