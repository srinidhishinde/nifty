"""Dedicated 15-minute-ahead option forecast workspace; advisory only."""
from __future__ import annotations

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
        if chain.empty:
            st.warning("No saved Kotak option-chain snapshot is available. Refresh the main dashboard while connected to Kotak.")
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
            start = now.date() - pd.Timedelta(days=120)
            candles, source = load_captured_candles(
                symbol, start=start, end=now.date(), timeframe_minutes=15
            )
        except Exception as exc:
            st.error(f"Could not load isolated 15-minute candles for {symbol}: {exc}")
            return
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        if candles.empty or not set(required).issubset(candles.columns):
            st.warning(
                f"No saved 15-minute candles exist for {symbol}. Start scripts/capture_option_15min.py "
                "for this exact expiry/side/strike. The 1-minute file will not be reused as 15-minute data."
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
                frame, underlying=symbol, lookback=min(30, len(frame) - 1),
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
