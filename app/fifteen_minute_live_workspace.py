"""Dedicated 15-minute-ahead option forecast workspace; advisory only."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from marketdata.daily_store import DailyMarketStore, load_captured_candles
from prediction.live_direction_forecast import predict_live_direction
from prediction.next_minute_forecast import forecast_next_minute


@st.fragment(run_every="15s")
def render_fifteen_minute_live_workspace() -> None:
    """Forecast the next 15-minute candle from completed, timeframe-isolated bars."""
    with st.container(border=True):
        st.subheader("15-Minute-Ahead Live Forecast")
        st.caption(
            "Uses completed 15-minute candles only · updates every 15 seconds · "
            "advisory forecast only; broker order execution remains disabled."
        )
        now = pd.Timestamp.now(tz="Asia/Kolkata")
        market_open = (
            now.weekday() < 5
            and (now.hour, now.minute) >= (9, 15)
            and (now.hour, now.minute) < (15, 30)
        )
        store = DailyMarketStore()
        try:
            chain, chain_ts = store.load_latest_option_chain_snapshot("NIFTY", before=now)
        except Exception as exc:
            st.error(f"Could not load the saved Kotak option chain: {exc}")
            return

        broker = st.session_state.get("neo_broker")
        broker_connected = False
        if broker is not None:
            try:
                broker_connected = bool(broker.connection_status().connected)
            except Exception:
                broker_connected = False

        chain_age = (
            max(0.0, (now - (chain_ts.tz_convert("Asia/Kolkata") if chain_ts.tzinfo else chain_ts.tz_localize("Asia/Kolkata"))).total_seconds())
            if chain_ts is not None else float("inf")
        )
        live_chain_available = bool(
            not chain.empty and market_open and broker_connected and chain_age <= 90
        )
        historical_files_mode = False
        if not live_chain_available:
            historical_root = Path(__file__).resolve().parents[1] / "data" / "backtest" / "kotak"
            found: dict[tuple[str, float, str], list[tuple[pd.Timestamp, Path]]] = {}
            if historical_root.exists():
                for path in historical_root.glob("*_1min.csv"):
                    match = re.match(r"^(?P<symbol>.+?)(?P<strike>\d+(?:\.\d+)?)(?P<type>CE|PE)_(?P<day>\d{4}-\d{2}-\d{2})_1min\.csv$", path.name, re.I)
                    if not match:
                        continue
                    key = (match.group("symbol").rstrip("_"), float(match.group("strike")), match.group("type").upper())
                    found.setdefault(key, []).append((pd.Timestamp(match.group("day")), path))
            if found:
                rows = []
                for (prefix, saved_strike, saved_side), paths in found.items():
                    latest_day, latest_path = max(paths, key=lambda item: item[0])
                    rows.append({
                        "symbol": f"{prefix}_{saved_strike:g}{saved_side}",
                        "strike": saved_strike, "option_type": saved_side,
                        "expiry": f"Saved data through {latest_day.date()}",
                        "history_path": latest_path,
                    })
                chain = pd.DataFrame(rows)
                historical_files_mode = True
            elif chain.empty:
                st.warning("No fresh option-chain snapshot or backdated option-candle CSVs are available.")
                return
        chain = chain.copy()
        if "option_type" not in chain or "strike" not in chain or "symbol" not in chain:
            st.warning("Saved option-chain snapshot lacks contract selection fields.")
            return
        chain["option_type"] = chain["option_type"].astype(str).str.upper().str.strip()
        chain["strike"] = pd.to_numeric(chain["strike"], errors="coerce")
        chain = chain.dropna(subset=["strike"])
        if "expiry" not in chain:
            chain["expiry"] = ""
        expiries = sorted(str(v) for v in chain["expiry"].dropna().unique() if str(v).strip())
        if not expiries:
            st.warning("No usable expiry is available in the saved option chain.")
            return
        expiry = st.selectbox("Expiry", expiries, key="fifteen_minute_expiry")
        by_expiry = chain[chain["expiry"].astype(str) == expiry]
        sides = [s for s in ("CE", "PE") if s in set(by_expiry["option_type"])]
        if not sides:
            st.warning("No CE/PE contracts for the selected expiry.")
            return
        side = st.selectbox("Option type", sides, key="fifteen_minute_side")
        by_side = by_expiry[by_expiry["option_type"] == side]
        strikes = sorted(float(x) for x in by_side["strike"].dropna().unique())
        if not strikes:
            st.warning("No strikes available for the selected side.")
            return
        strike = st.selectbox("Strike", strikes, key="fifteen_minute_strike")
        selected = by_side[by_side["strike"] == strike]
        if selected.empty:
            st.warning("Selected contract is not in the saved chain.")
            return
        symbol = str(selected.iloc[0]["symbol"]).strip()
        if not symbol:
            st.warning("Selected contract has no canonical symbol; forecast withheld.")
            return
        st.caption(f"Selected contract: {symbol} · {side} {strike:g} · expiry {expiry}")

        try:
            if historical_files_mode:
                history_path = selected.iloc[0]["history_path"]
                raw = pd.read_csv(history_path)
                raw.columns = [str(column).strip().lower() for column in raw.columns]
                aliases = {"datetime": "timestamp", "date": "timestamp", "time": "timestamp", "ltp": "close", "price": "close", "vol": "volume"}
                raw = raw.rename(columns={k: v for k, v in aliases.items() if k in raw.columns and v not in raw.columns})
                required_raw = {"timestamp", "open", "high", "low", "close"}
                if not required_raw.issubset(raw.columns):
                    raise ValueError(f"Historical file is missing columns: {sorted(required_raw - set(raw.columns))}")
                if "volume" not in raw.columns:
                    raw["volume"] = 0.0
                raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
                if raw["timestamp"].dt.tz is None:
                    raw["timestamp"] = raw["timestamp"].dt.tz_localize("Asia/Kolkata")
                else:
                    raw["timestamp"] = raw["timestamp"].dt.tz_convert("Asia/Kolkata")
                for col in ("open", "high", "low", "close", "volume"):
                    raw[col] = pd.to_numeric(raw[col], errors="coerce")
                raw = raw.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp")
                # Resample one session at a time and accept only complete 15x1m buckets.
                raw = raw.drop_duplicates("timestamp", keep="last")
                indexed = raw.set_index("timestamp")
                grouped = indexed.groupby(indexed.index.date, group_keys=False)
                parts = []
                for _, session in grouped:
                    bars = session[["open", "high", "low", "close", "volume"]].resample(
                        "15min", origin="start_day", offset="9h15min", label="left", closed="left"
                    ).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
                    counts = session["close"].resample(
                        "15min", origin="start_day", offset="9h15min", label="left", closed="left"
                    ).count()
                    bars = bars[counts == 15].dropna(subset=["open", "high", "low", "close"]).reset_index()
                    parts.append(bars)
                candles = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
                source = f"BACKDATED CSV · {history_path.name} · complete 15-minute bars resampled from 1-minute data"
            else:
                start = now.date() - pd.Timedelta(days=120)
                candles, source = load_captured_candles(
                    symbol, start=start, end=now.date(), timeframe_minutes=15
                )
        except Exception as exc:
            st.error(f"Could not load 15-minute candles for {symbol}: {exc}")
            return
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        if candles.empty or not set(required).issubset(candles.columns):
            st.warning(
                f"No usable 15-minute candles or complete historical 15-minute buckets exist for {symbol}. "
                "Start scripts/capture_option_15min.py for this exact expiry/side/strike, or provide a saved 1-minute historical session."
            )
            return

        frame = candles[required].copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        if frame["timestamp"].dt.tz is None:
            frame["timestamp"] = frame["timestamp"].dt.tz_localize("Asia/Kolkata")
        else:
            frame["timestamp"] = frame["timestamp"].dt.tz_convert("Asia/Kolkata")
        for col in required[1:]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=required).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        if frame.empty:
            st.warning("No valid 15-minute OHLCV rows remain after validation.")
            return

        latest_start = frame["timestamp"].iloc[-1]
        latest_completed_at = latest_start + pd.Timedelta(minutes=15)
        age_seconds = max(0.0, (now - latest_completed_at).total_seconds())
        live = bool(market_open and broker_connected and chain_age <= 90 and age_seconds <= 180)
        metrics = st.columns(4)
        metrics[0].metric("Data mode", "LIVE" if live else "HISTORICAL")
        metrics[1].metric("Completed 15m candles", f"{len(frame):,}")
        metrics[2].metric("Latest close", f"₹{float(frame['close'].iloc[-1]):,.2f}")
        metrics[3].metric("Last candle completion age", f"{age_seconds / 60:.1f} min")
        if not live:
            st.info("HISTORICAL / STALE MODE: forecast is anchored to the latest saved completed candle, not the current market.")

        chart = go.Figure(data=[go.Candlestick(
            x=frame.tail(120)["timestamp"], open=frame.tail(120)["open"],
            high=frame.tail(120)["high"], low=frame.tail(120)["low"],
            close=frame.tail(120)["close"], name=f"{symbol} 15m",
        )])
        chart.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10),
                            xaxis_title="Time (Asia/Kolkata)", yaxis_title="Option price (₹)",
                            xaxis_rangeslider_visible=False)
        st.plotly_chart(chart, use_container_width=True)

        if len(frame) < 3:
            st.warning(f"At least 3 completed 15-minute candles are required; currently {len(frame)}.")
            return
        try:
            direction = predict_live_direction(
                frame, estimator="logistic", min_train=80,
                tolerance=0.01, timeframe_minutes=15,
            )
            range_forecast = forecast_next_minute(
                frame, underlying=symbol, lookback=min(30, len(frame)),
                quantile=0.90, range_method="adaptive", timeframe_minutes=15,
            )
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            st.warning(f"15-minute forecast unavailable: {exc}")
            return
        cols = st.columns(4)
        cols[0].metric("Next 15-minute direction", direction["direction"])
        cols[1].metric("Forecast method", direction["estimator"])
        cols[2].metric("Training examples", f"{direction['training_rows']:,}")
        cols[3].metric("Estimated next-candle range", f"₹{range_forecast.predicted_low:,.2f} – ₹{range_forecast.predicted_high:,.2f}")
        st.caption(
            f"Forecast origin: {latest_start.strftime('%Y-%m-%d %H:%M %Z')} · "
            f"target candle starts at {direction['target_candle_time']} · source: {source}. "
            "Raw classifier score is not a calibrated probability; range is an estimate, not a guarantee."
        )
        if direction["method"] == "MOMENTUM_FALLBACK":
            st.warning("Momentum fallback is active because there is insufficient eligible training data; this is not a trained classifier output.")
        st.info(
            ("LIVE DATA FORECAST" if live else "HISTORICAL DATA FORECAST")
            + " · advisory/paper use only · no broker orders submitted."
        )
