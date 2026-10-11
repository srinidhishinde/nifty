"""Dedicated live one-minute workspace using only persisted Kotak SFeed candles."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from marketdata.daily_store import DailyMarketStore, load_captured_candles
from prediction.live_direction_forecast import predict_live_direction
from prediction.next_minute_forecast import forecast_next_minute


@st.fragment(run_every="10s")
def render_one_minute_live_workspace() -> None:
    """Refresh the live 1-minute panel without refreshing the entire dashboard."""
    with st.container(border=True):
        st.subheader("1-Minute Live Workspace")
        st.caption(
            "Independent live-candle layout · Kotak SFeed capture only · "
            "refreshes every 10 seconds · forecasts are advisory, not order instructions."
        )
        left, right = st.columns([1, 4])
        refresh = left.button("Refresh 1-minute data", key="refresh_one_minute_live")
        history_rows = right.selectbox(
            "Chart history",
            [60, 120, 180, 300],
            index=1,
            key="one_minute_live_chart_rows",
        )
        if refresh:
            st.session_state["one_minute_live_force_refresh"] = True

        now = pd.Timestamp.now(tz="Asia/Kolkata")
        st.markdown("#### Live mode · select an option contract")
        try:
            chain, chain_timestamp = DailyMarketStore().load_latest_option_chain_snapshot(
                "NIFTY", before=now
            )
        except Exception as exc:
            st.error(f"Could not load the latest real Kotak option chain: {exc}")
            return

        if chain.empty:
            st.warning(
                "No real Kotak NIFTY option-chain snapshot is available. Refresh the broker option chain "
                "in the main dashboard while connected; live prediction is withheld until a real chain is available."
            )
            return

        chain = chain.copy()
        chain["option_type"] = chain["option_type"].astype(str).str.upper().str.strip()
        chain["strike"] = pd.to_numeric(chain.get("strike"), errors="coerce")
        chain = chain.dropna(subset=["strike"])
        if "expiry" not in chain.columns:
            chain["expiry"] = ""
        if "symbol" not in chain.columns:
            st.warning("The captured option chain does not contain contract symbols; cannot select a live contract.")
            return

        expiries = sorted(str(v) for v in chain["expiry"].dropna().unique() if str(v).strip())
        if not expiries:
            st.warning("The real option-chain snapshot has no usable expiry field.")
            return
        expiry = st.selectbox("Expiry", expiries, key="one_minute_live_expiry")
        expiry_chain = chain[chain["expiry"].astype(str) == expiry].copy()
        types = [t for t in ["CE", "PE"] if t in set(expiry_chain["option_type"])]
        if not types:
            st.warning("No CE/PE contracts are available for the selected expiry.")
            return
        option_type = st.selectbox("Option type", types, key="one_minute_live_option_type")
        type_chain = expiry_chain[expiry_chain["option_type"] == option_type].copy()
        strikes = sorted(float(v) for v in type_chain["strike"].dropna().unique())
        if not strikes:
            st.warning("No strikes are available for the selected option type.")
            return
        strike = st.selectbox("Strike", strikes, key="one_minute_live_strike")
        matches = type_chain[type_chain["strike"] == strike]
        if matches.empty:
            st.warning("Selected contract is no longer present in the latest option-chain snapshot.")
            return
        contract = matches.iloc[0]
        contract_symbol = str(contract["symbol"]).strip()
        if not contract_symbol:
            st.warning("Selected option contract has no broker symbol; prediction is withheld.")
            return

        st.caption(
            f"Selected contract: {contract_symbol} · {option_type} {strike:g} · expiry {expiry} · "
            f"option-chain snapshot: {chain_timestamp if chain_timestamp is not None else 'timestamp unavailable'}"
        )
        try:
            # Keep a broad history window so the panel remains useful outside
            # market hours and during feed outages. This is still broker-captured
            # data only; no synthetic or third-party fallback is allowed.
            start = now.date() - pd.Timedelta(days=90)
            frame, source = load_captured_candles(
                contract_symbol,
                start=start,
                end=now.date(),
            )
        except Exception as exc:
            st.error(f"Could not load captured one-minute candles for {contract_symbol}: {exc}")
            return

        required = ["timestamp", "open", "high", "low", "close", "volume"]
        if frame.empty or not set(required).issubset(frame.columns):
            st.warning(
                f"No captured one-minute OHLCV candles are available for {contract_symbol} in the last 90 days. "
                "Start the Kotak SFeed recorder or select a contract with saved history; "
                "synthetic or Yahoo data will not be substituted."
            )
            return

        frame = frame[required].copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
        for column in required[1:]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = (
            frame.dropna(subset=required)
            .sort_values("timestamp")
            .drop_duplicates("timestamp", keep="last")
            .reset_index(drop=True)
        )
        if frame.empty:
            st.warning("Captured candle file has no valid OHLCV rows.")
            return

        # The capture recorder stores one-minute bars. Refuse to label coarser
        # or irregular data as a 1-minute feed.
        deltas = frame["timestamp"].diff().dt.total_seconds().div(60)
        normal_deltas = deltas[(deltas > 0) & (deltas <= 10)]
        if not normal_deltas.empty and float(normal_deltas.median()) > 1.5:
            st.warning(
                f"Captured candles appear coarser than one minute (median interval "
                f"{float(normal_deltas.median()):.1f} minutes). This layout requires genuine 1-minute capture."
            )
            return

        latest_ts = frame["timestamp"].iloc[-1]
        age_seconds = max(0.0, (pd.Timestamp.now(tz="UTC") - latest_ts).total_seconds())
        local_now = pd.Timestamp.now(tz="Asia/Kolkata")
        is_weekday = local_now.weekday() < 5
        market_clock_open = (
            is_weekday
            and (local_now.hour, local_now.minute) >= (9, 15)
            and (local_now.hour, local_now.minute) < (15, 30)
        )
        is_live = market_clock_open and age_seconds <= 180
        mode_label = "LIVE" if is_live else "HISTORICAL"
        metrics = st.columns(4)
        metrics[0].metric("Data mode", mode_label)
        metrics[1].metric("Captured candles", f"{len(frame):,}")
        metrics[2].metric("Latest close", f"₹{float(frame['close'].iloc[-1]):,.2f}")
        metrics[3].metric("Latest candle age", f"{age_seconds / 60:.1f} min")
        if not is_live:
            st.info(
                "No fresh in-session one-minute feed detected. Showing the latest saved historical "
                "candles for this selected contract; forecasts refer to the next candle after the "
                "historical data endpoint, not to the current market."
            )

        chart_frame = frame.tail(int(history_rows)).copy()
        chart = go.Figure(
            data=[
                go.Candlestick(
                    x=chart_frame["timestamp"].dt.tz_convert("Asia/Kolkata"),
                    open=chart_frame["open"],
                    high=chart_frame["high"],
                    low=chart_frame["low"],
                    close=chart_frame["close"],
                    name=f"{contract_symbol} 1m",
                )
            ]
        )
        chart.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="Time (Asia/Kolkata)",
            yaxis_title=f"{contract_symbol} price (₹)",
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(chart, use_container_width=True)

        if is_live and age_seconds > 180:
            # Defensive consistency check; stale feeds are shown as historical above.
            st.warning("The live feed is stale; historical mode is being used instead.")
        if len(frame) < 100:
            st.info(f"Need more history before forecasting reliably; currently have {len(frame)} candles.")
            return

        try:
            direction = predict_live_direction(
                frame,
                estimator="logistic",
                min_train=80,
                tolerance=0.01,
                timeframe_minutes=1,
            )
            range_forecast = forecast_next_minute(
                frame,
                underlying=contract_symbol,
                lookback=min(30, max(5, len(frame) - 1)),
                quantile=0.90,
                range_method="adaptive",
            )
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            st.warning(f"One-minute forecast is currently unavailable: {exc}")
            return

        forecast_cols = st.columns(4)
        forecast_cols[0].metric("Next 1-minute direction", direction["direction"])
        forecast_cols[1].metric("Forecast method", direction["estimator"])
        forecast_cols[2].metric("Training examples", f"{direction['training_rows']:,}")
        forecast_cols[3].metric(
            "Estimated next-candle range",
            f"₹{range_forecast.predicted_low:,.2f} – ₹{range_forecast.predicted_high:,.2f}",
        )
        st.caption(
            f"Last candle: {latest_ts.tz_convert('Asia/Kolkata').strftime('%Y-%m-%d %H:%M:%S %Z')} · "
            f"Target: {direction['target_candle_time']} · source label: {source}. "
            "The range is an estimate, not a guaranteed bound; raw classifier score is not calibrated probability."
        )
        if direction["method"] == "MOMENTUM_FALLBACK":
            st.warning("Momentum fallback is active; this is not a trained classifier forecast.")
        if is_live:
            st.info("LIVE DATA FORECAST for the selected option contract. Advisory/paper use only; no broker orders are submitted.")
        else:
            st.info("HISTORICAL DATA FORECAST for the selected option contract. This is a backdated analysis, not a live-market signal; no broker orders are submitted.")
