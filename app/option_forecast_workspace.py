"""Compact option selector with 1m/5m high-low forecasts and closed-market validation."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import streamlit as st

from marketdata.daily_store import DailyMarketStore, load_captured_candles
from prediction.next_minute_forecast import forecast_next_minute


ROOT = Path(__file__).resolve().parents[1]
HISTORY_ROOT = ROOT / "data" / "backtest" / "kotak"


def _read_history(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    aliases = {"datetime": "timestamp", "date": "timestamp", "time": "timestamp",
               "ltp": "close", "price": "close", "vol": "volume"}
    raw = raw.rename(columns={k: v for k, v in aliases.items() if k in raw.columns and v not in raw.columns})
    required = ["timestamp", "open", "high", "low", "close"]
    missing = [c for c in required if c not in raw]
    if missing:
        raise ValueError(f"Historical file missing columns: {missing}")
    if "volume" not in raw:
        raw["volume"] = 0.0
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    if raw["timestamp"].dt.tz is None:
        raw["timestamp"] = raw["timestamp"].dt.tz_localize("Asia/Kolkata")
    else:
        raw["timestamp"] = raw["timestamp"].dt.tz_convert("Asia/Kolkata")
    for col in ("open", "high", "low", "close", "volume"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    return (raw.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
            .sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True))


def _five_minute_bars(one_minute: pd.DataFrame) -> pd.DataFrame:
    """Build only complete 5x1m candles, separately per session."""
    if one_minute.empty:
        return pd.DataFrame()
    frame = one_minute.copy()
    ts = pd.to_datetime(frame["timestamp"], errors="coerce")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("Asia/Kolkata")
    else:
        ts = ts.dt.tz_convert("Asia/Kolkata")
    frame["timestamp"] = ts
    frame = frame.dropna(subset=["timestamp"]).set_index("timestamp")
    parts = []
    for _, session in frame.groupby(frame.index.date, sort=True):
        bars = session[["open", "high", "low", "close", "volume"]].resample(
            "5min", origin="start_day", offset="9h15min", label="left", closed="left"
        ).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        counts = session["close"].resample(
            "5min", origin="start_day", offset="9h15min", label="left", closed="left"
        ).count()
        parts.append(bars[counts == 5].dropna(subset=["open", "high", "low", "close"]).reset_index())
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _historical_comparison(frame: pd.DataFrame, minutes: int, symbol: str) -> pd.DataFrame:
    """Strictly walk forward: forecast each historical candle using earlier bars only."""
    rows = []
    # Keep the display small; still compute chronologically and never train on target bars.
    first = max(3, len(frame) - 80)
    for target_idx in range(first, len(frame)):
        history = frame.iloc[:target_idx].copy()
        actual = frame.iloc[target_idx]
        try:
            pred = forecast_next_minute(
                history, underlying=symbol, lookback=min(30, len(history)),
                quantile=0.80, range_method="adaptive", timeframe_minutes=minutes,
                generated_at=history["timestamp"].iloc[-1],
            )
        except (ValueError, TypeError, KeyError, OverflowError):
            continue
        rows.append({
            "Candle time": pd.Timestamp(actual["timestamp"]).strftime("%Y-%m-%d %H:%M"),
            "Predicted low": pred.predicted_low,
            "Actual low": round(float(actual["low"]), 2),
            "Low hit": float(actual["low"]) >= pred.predicted_low,
            "Actual high": round(float(actual["high"]), 2),
            "Predicted high": pred.predicted_high,
            "High hit": float(actual["high"]) <= pred.predicted_high,
        })
    return pd.DataFrame(rows)


@st.fragment(run_every="15s")
def render_option_forecast_workspace(instrument: str) -> None:
    with st.container(border=True):
        st.subheader("Option High / Low Forecast")
        st.caption("Select a contract, then view the next 1-minute or 5-minute candle range. No order execution.")
        interval_label = st.radio("Forecast interval", ["1 minute", "5 minutes"], horizontal=True, key="compact_option_interval")
        minutes = 1 if interval_label == "1 minute" else 5
        now = pd.Timestamp.now(tz="Asia/Kolkata")
        store = DailyMarketStore()
        chain = pd.DataFrame()
        chain_ts = None
        try:
            chain, chain_ts = store.load_latest_option_chain_snapshot(instrument, before=now)
        except Exception:
            pass

        # A cached snapshot is never labelled live merely because it exists.
        market_open = now.weekday() < 5 and (
            ((now.hour, now.minute) >= (9, 15) and (now.hour, now.minute) < (15, 30))
            if instrument == "NIFTY" else
            ((now.hour, now.minute) >= (9, 0) and (now.hour, now.minute) < (23, 30))
        )
        broker = st.session_state.get("neo_broker")
        connected = False
        if broker is not None:
            try:
                connected = bool(broker.connection_status().connected)
            except Exception:
                pass
        chain_age = float("inf")
        if chain_ts is not None:
            chain_local = chain_ts.tz_convert("Asia/Kolkata") if chain_ts.tzinfo else chain_ts.tz_localize("Asia/Kolkata")
            chain_age = max(0.0, (now - chain_local).total_seconds())
        if market_open and connected and (chain.empty or chain_age > 90):
            last_attempt = st.session_state.get("compact_chain_refresh_attempt")
            attempt_age = (now - last_attempt).total_seconds() if isinstance(last_attempt, pd.Timestamp) else float("inf")
            if attempt_age >= 60:
                st.session_state["compact_chain_refresh_attempt"] = now
                try:
                    from marketdata.providers.kotak_neo import KotakNeoProvider
                    exchange = "nse_fo" if instrument == "NIFTY" else "mcx_fo"
                    refreshed = KotakNeoProvider(broker.client).get_option_chain(
                        underlying=instrument, exchange=exchange, count=40,
                        enrich_quotes=True,
                    )
                    if refreshed:
                        store.save_option_chain_snapshot(instrument, refreshed, captured_at=now)
                        chain, chain_ts = store.load_latest_option_chain_snapshot(instrument, before=now)
                        chain_age = 0.0
                except Exception as exc:
                    st.warning(f"Live option-chain refresh failed; saved history will be used only as historical data. {exc}")
        live = bool(market_open and connected and not chain.empty and chain_age <= 90)

        historical_files: dict[tuple[str, float, str], list[tuple[pd.Timestamp, Path]]] = {}
        invalid_history_filenames: list[str] = []
        if HISTORY_ROOT.exists():
            for path in HISTORY_ROOT.glob("*_1min.csv"):
                match = re.match(r"^(?P<symbol>.+?)(?P<strike>\d+(?:\.\d+)?)(?P<side>CE|PE)_(?P<day>\d{4}-\d{2}-\d{2})_1min\.csv$", path.name, re.I)
                if not match or not match.group("symbol").rstrip("_").upper().startswith(instrument.upper()):
                    continue
                # The filename regex checks date shape only; validate the actual
                # calendar date before constructing a Timestamp so a malformed
                # filename (for example, 2026-10-78) cannot crash the dashboard.
                day = pd.to_datetime(match.group("day"), format="%Y-%m-%d", errors="coerce")
                if pd.isna(day):
                    invalid_history_filenames.append(path.name)
                    continue
                key = (match.group("symbol").rstrip("_"), float(match.group("strike")), match.group("side").upper())
                historical_files.setdefault(key, []).append((pd.Timestamp(day), path))
        if invalid_history_filenames:
            examples = ", ".join(invalid_history_filenames[:3])
            suffix = " ..." if len(invalid_history_filenames) > 3 else ""
            st.warning(
                f"Ignored {len(invalid_history_filenames)} saved candle file(s) with invalid dates in their filenames: "
                f"{examples}{suffix}. Rename or recapture those files; their dates were not guessed."
            )

        historical_mode = not live
        if historical_mode and historical_files:
            records = []
            for (prefix, strike_value, side_value), paths in historical_files.items():
                latest_day, latest_path = max(paths, key=lambda item: item[0])
                records.append({"symbol": f"{prefix}_{strike_value:g}{side_value}", "strike": strike_value,
                                "option_type": side_value, "expiry": f"Saved through {latest_day.date()}",
                                "history_path": latest_path})
            chain = pd.DataFrame(records)
        if chain.empty:
            st.warning("No option-chain snapshot or saved option-candle history is available for this instrument. Capture real Kotak data first.")
            return
        for col in ("symbol", "strike", "option_type"):
            if col not in chain:
                st.warning("Option data is missing contract-selection fields.")
                return
        chain = chain.copy()
        chain["strike"] = pd.to_numeric(chain["strike"], errors="coerce")
        chain["option_type"] = chain["option_type"].astype(str).str.upper().str.strip()
        chain = chain.dropna(subset=["strike"])
        if "expiry" not in chain:
            chain["expiry"] = ""
        expiries = sorted(str(v) for v in chain["expiry"].dropna().unique() if str(v).strip())
        if not expiries:
            st.warning("No expiry/session label is available.")
            return
        expiry = st.selectbox("Expiry / saved session", expiries, key="compact_option_expiry")
        exp_chain = chain[chain["expiry"].astype(str) == expiry]
        sides = [s for s in ("CE", "PE") if s in set(exp_chain["option_type"])]
        if not sides:
            st.warning("No CE/PE contracts for the selected expiry.")
            return
        side = st.selectbox("Option type", sides, key="compact_option_side")
        side_chain = exp_chain[exp_chain["option_type"] == side]
        strikes = sorted(float(v) for v in side_chain["strike"].dropna().unique())
        strike = st.selectbox("Strike", strikes, key="compact_option_strike")
        selected = side_chain[side_chain["strike"] == strike]
        if selected.empty:
            st.warning("Selected option contract is unavailable.")
            return
        contract = selected.iloc[0]
        symbol = str(contract["symbol"]).strip()
        st.caption(f"Contract: {symbol} · {side} {strike:g} · {expiry}")
        try:
            if historical_mode and "history_path" in selected.columns:
                path = contract["history_path"]
                one_min = _read_history(Path(path))
                source = f"Saved historical CSV: {Path(path).name}"
            else:
                one_min, source = load_captured_candles(
                    symbol, start=now.date() - pd.Timedelta(days=120), end=now.date()
                )
                one_min = one_min.copy()
                if not one_min.empty:
                    one_min["timestamp"] = pd.to_datetime(one_min["timestamp"], errors="coerce")
                    if one_min["timestamp"].dt.tz is None:
                        one_min["timestamp"] = one_min["timestamp"].dt.tz_localize("Asia/Kolkata")
                    else:
                        one_min["timestamp"] = one_min["timestamp"].dt.tz_convert("Asia/Kolkata")
        except Exception as exc:
            st.warning(f"Could not load candles for this contract: {exc}")
            return
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        if one_min.empty or not set(required).issubset(one_min.columns):
            st.warning("No usable OHLCV candles for this contract. Forecast is withheld rather than fabricated.")
            return
        one_min = one_min[required].copy()
        one_min["timestamp"] = pd.to_datetime(one_min["timestamp"], errors="coerce")
        if one_min["timestamp"].dt.tz is None:
            one_min["timestamp"] = one_min["timestamp"].dt.tz_localize("Asia/Kolkata")
        else:
            one_min["timestamp"] = one_min["timestamp"].dt.tz_convert("Asia/Kolkata")
        for col in required[1:]:
            one_min[col] = pd.to_numeric(one_min[col], errors="coerce")
        one_min = one_min.dropna(subset=required).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        if minutes == 5:
            frame = _five_minute_bars(one_min)
            source += " · complete 5x1-minute buckets"
        else:
            frame = one_min
        if frame.empty or len(frame) < 3:
            st.warning(f"Need at least 3 valid completed {minutes}-minute candles; found {len(frame)}.")
            return
        last_start = pd.Timestamp(frame["timestamp"].iloc[-1])
        completed_at = last_start + pd.Timedelta(minutes=minutes)
        candle_age = max(0.0, (now - completed_at).total_seconds())
        data_is_live = bool(live and candle_age <= (180 if minutes == 1 else 420))
        status = "LIVE DATA" if data_is_live else "HISTORICAL / MARKET CLOSED"
        st.write(f"**Data status:** {status} · **Source:** {source}")
        st.write(f"**Last actual candle:** {last_start.strftime('%Y-%m-%d %H:%M %Z')} · close ₹{float(frame['close'].iloc[-1]):,.2f}")
        try:
            pred = forecast_next_minute(
                frame, underlying=symbol, lookback=min(30, len(frame)),
                quantile=0.80, range_method="adaptive", timeframe_minutes=minutes,
            )
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            st.warning(f"Forecast unavailable: {exc}")
            return
        a, b, c = st.columns(3)
        a.metric("Next candle predicted low", f"₹{pred.predicted_low:,.2f}")
        b.metric("Next candle predicted high", f"₹{pred.predicted_high:,.2f}")
        c.metric("Forecast target", pd.Timestamp(pred.target_candle_time).strftime("%Y-%m-%d %H:%M"))
        st.caption("High/low are estimates, not guarantees. This range method is not a calibrated probability or a promise of trade accuracy.")
        if not data_is_live:
            st.info("Market is closed or data is stale. The forecast is anchored to the latest saved candle and is historical-context only—not a current live quote.")
            st.markdown("#### Actual vs predicted (walk-forward history)")
            comparison = _historical_comparison(frame, minutes, symbol)
            if comparison.empty:
                st.info("Not enough earlier candles to calculate a walk-forward comparison.")
            else:
                cols = st.columns(2)
                cols[0].metric("Low inside predicted range", f"{comparison['Low hit'].mean() * 100:.1f}%", f"{int(comparison['Low hit'].sum())}/{len(comparison)} candles")
                cols[1].metric("High inside predicted range", f"{comparison['High hit'].mean() * 100:.1f}%", f"{int(comparison['High hit'].sum())}/{len(comparison)} candles")
                st.dataframe(comparison.iloc[::-1], use_container_width=True, hide_index=True, height=360)
        else:
            st.caption("Actual high/low for the target candle become known only after that candle closes; they are not displayed as if already observed.")
