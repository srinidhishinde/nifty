"""Live-first option selector with explicit data freshness and safe historical fallback."""
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


@st.fragment(run_every="30s")
def render_option_forecast_workspace(instrument: str) -> None:
    with st.container(border=True):
        st.subheader("Option High / Low Forecast")
        st.caption("Live-first workspace · Kotak Neo option-chain refresh every 30 seconds while connected and the market is open. Saved data is clearly marked historical. Forecasting only; orders remain disabled.")
        interval_label = st.radio("Forecast interval", ["1 minute", "5 minutes"], horizontal=True, key="compact_option_interval")
        minutes = 1 if interval_label == "1 minute" else 5
        now = pd.Timestamp.now(tz="Asia/Kolkata")
        force_refresh = st.button("Refresh market data now", key="compact_option_refresh_now", type="secondary")
        store = DailyMarketStore()
        chain = pd.DataFrame()
        chain_ts = None
        saved_dates = store.available_option_chain_snapshot_dates(instrument)
        date_labels = [day.isoformat() for day in saved_dates]
        default_date = date_labels[0] if date_labels else "Latest available"
        selected_date_label = st.selectbox(
            "Snapshot / trading date",
            options=date_labels if date_labels else ["Latest available"],
            index=0,
            key="compact_option_snapshot_date",
            help="Choose the date whose saved Kotak option-chain snapshot and quotes you want to inspect. The expiry is selected separately below.",
        )
        selected_snapshot_date = pd.Timestamp(selected_date_label).date() if selected_date_label != "Latest available" else None
        try:
            chain, chain_ts = store.load_latest_option_chain_snapshot(
                instrument, before=now, session_date=selected_snapshot_date
            )
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
        exchange = "nse_fo" if instrument == "NIFTY" else "mcx_fo"
        provider = None
        broker_expiries: list[str] = []
        if connected:
            try:
                from marketdata.providers.kotak_neo import KotakNeoProvider
                provider = KotakNeoProvider(broker.client)
                expiry_underlying = (
                    provider.resolve_option_underlying(instrument, exchange)
                    if exchange == "mcx_fo" else instrument.upper()
                )
                expiry_response = broker.client.expiries(
                    exchange=exchange, underlying=expiry_underlying, instrument_type="option"
                )
                if isinstance(expiry_response, dict):
                    expiry_payload = expiry_response.get("data")
                    if not isinstance(expiry_payload, dict):
                        expiry_payload = expiry_response
                    expiry_values = expiry_response.get("expiries") or expiry_payload.get("expiries") or []
                    if isinstance(expiry_values, list):
                        broker_expiries = sorted({str(value).strip() for value in expiry_values if str(value).strip()})
            except Exception:
                # Saved expiry labels remain usable when the broker expiry endpoint is unavailable.
                provider = provider or None
        chain_age = float("inf")
        if chain_ts is not None:
            chain_local = chain_ts.tz_convert("Asia/Kolkata") if chain_ts.tzinfo else chain_ts.tz_localize("Asia/Kolkata")
            chain_age = max(0.0, (now - chain_local).total_seconds())
        # A historical snapshot date is an explicit backtrace request: never overwrite
        # it with a current live refresh. The date selector defaults to the newest saved
        # session, so live refresh remains enabled for normal use.
        historical_date_selected = selected_snapshot_date is not None and selected_snapshot_date < now.date()
        # Fetch from Neo even when no local snapshot exists and the market is closed:
        # the broker may still return its latest available chain. Such data remains
        # historical outside market hours and is never labelled live.
        should_fetch_broker_chain = connected and not historical_date_selected and (
            force_refresh or chain.empty or (market_open and chain_age > 30)
        )
        if should_fetch_broker_chain:
            last_attempt = st.session_state.get("compact_chain_refresh_attempt")
            attempt_age = (now - last_attempt).total_seconds() if isinstance(last_attempt, pd.Timestamp) else float("inf")
            if force_refresh or chain.empty or attempt_age >= 30:
                st.session_state["compact_chain_refresh_attempt"] = now
                try:
                    if provider is None:
                        from marketdata.providers.kotak_neo import KotakNeoProvider
                        provider = KotakNeoProvider(broker.client)
                    requested_expiry = st.session_state.get("compact_option_expiry")
                    refreshed = provider.get_option_chain(
                        underlying=instrument, exchange=exchange,
                        expiry=str(requested_expiry) if requested_expiry in broker_expiries else None,
                        count=40, enrich_quotes=True,
                    )
                    if refreshed:
                        store.save_option_chain_snapshot(instrument, refreshed, captured_at=now)
                        chain, chain_ts = store.load_latest_option_chain_snapshot(instrument, before=now)
                        chain_age = 0.0
                        refresh_map = dict(st.session_state.get("compact_live_refresh_at", {}))
                        refresh_map[instrument] = now
                        st.session_state["compact_live_refresh_at"] = refresh_map
                    else:
                        st.warning("Kotak Neo returned an empty option chain. The saved snapshot, if any, remains historical.")
                except Exception as exc:
                    st.warning(f"Kotak Neo option-chain fetch failed; any saved data remains historical. Check the broker session, expiry availability, and API response. {exc}")
        refresh_map = st.session_state.get("compact_live_refresh_at", {})
        last_live_refresh = refresh_map.get(instrument) if isinstance(refresh_map, dict) else None
        refresh_age = (now - last_live_refresh).total_seconds() if isinstance(last_live_refresh, pd.Timestamp) else float("inf")
        live = bool(market_open and connected and not chain.empty and chain_age <= 90 and refresh_age <= 90)

        # Always show the provenance and timestamp of the option-chain data,
        # including when the market is closed or Neo is temporarily disconnected.
        if chain_ts is not None:
            snapshot_local = chain_ts.tz_convert("Asia/Kolkata") if chain_ts.tzinfo else chain_ts.tz_localize("Asia/Kolkata")
            snapshot_label = snapshot_local.strftime("%Y-%m-%d %H:%M:%S %Z")
            age_minutes = max(0, int(chain_age // 60))
            if live:
                st.success(f"LIVE · Kotak Neo option chain · Updated {snapshot_label} · refresh age {int(refresh_age)}s")
            else:
                st.info(f"HISTORICAL SNAPSHOT · Last saved {snapshot_label} · {age_minutes} minute(s) old · not a live quote")
        elif not chain.empty:
            st.warning("Option-chain data is available, but its capture timestamp could not be determined; treat it as historical.")

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

        if market_open and connected and not live:
            st.warning("Market hours are active, but a successful fresh Kotak Neo refresh has not been confirmed in this session. Displayed snapshot data is not labelled live.")
        elif market_open and not connected:
            st.warning("Market is open, but Kotak Neo is disconnected. Connect to Neo to refresh live market data.")
        historical_mode = not live
        if historical_mode and historical_files:
            # Keep the real saved option-chain snapshot as the primary selector.
            # Attach the best matching saved candle file to each snapshot contract,
            # then add file-only contracts so users can still inspect captured history
            # when the snapshot did not include that strike.
            history_by_strike_side: dict[tuple[float, str], list[tuple[str, pd.Timestamp, Path]]] = {}
            history_by_symbol: dict[str, list[tuple[pd.Timestamp, Path]]] = {}
            for (prefix, strike_value, side_value), paths in historical_files.items():
                history_by_strike_side.setdefault((strike_value, side_value), []).extend(
                    (prefix, day, path) for day, path in paths
                )
                history_by_symbol.setdefault(f"{prefix}_{strike_value:g}{side_value}".upper(), []).extend(paths)

            snapshot_chain = chain.copy() if not chain.empty else pd.DataFrame()
            if not snapshot_chain.empty:
                snapshot_chain["strike"] = pd.to_numeric(snapshot_chain.get("strike"), errors="coerce")
                snapshot_chain["option_type"] = snapshot_chain["option_type"].astype(str).str.upper().str.strip()
                if "expiry" not in snapshot_chain:
                    snapshot_chain["expiry"] = ""
                attached_paths = []
                for _, row in snapshot_chain.iterrows():
                    symbol_key = str(row.get("symbol", "")).strip().upper()
                    exact = history_by_symbol.get(symbol_key, [])
                    candidates = exact
                    if not candidates:
                        candidates = [
                            (day, path)
                            for (strike_value, side_value), entries in history_by_strike_side.items()
                            if pd.notna(row.get("strike")) and abs(float(row["strike"]) - strike_value) < 1e-6
                            and str(row.get("option_type", "")).upper() == side_value
                            for _, day, path in entries
                        ]
                    if selected_snapshot_date is not None:
                        candidates = [
                            item for item in candidates
                            if (item[1] if len(item) == 3 else item[0]).date() <= selected_snapshot_date
                        ]
                    if candidates:
                        if len(candidates[0]) == 3:
                            _, day, path = max(candidates, key=lambda item: item[1])
                        else:
                            day, path = max(candidates, key=lambda item: item[0])
                        attached_paths.append(path)
                    else:
                        attached_paths.append(None)
                snapshot_chain["history_path"] = attached_paths

            existing_keys = set()
            if not snapshot_chain.empty:
                existing_keys = {
                    (round(float(row["strike"]), 6), str(row["option_type"]).upper())
                    for _, row in snapshot_chain.dropna(subset=["strike"]).iterrows()
                }
            history_records = []
            for (prefix, strike_value, side_value), paths in historical_files.items():
                if (round(strike_value, 6), side_value) in existing_keys:
                    continue
                eligible_paths = [
                    item for item in paths
                    if selected_snapshot_date is None or item[0].date() <= selected_snapshot_date
                ]
                if not eligible_paths:
                    continue
                latest_day, latest_path = max(eligible_paths, key=lambda item: item[0])
                history_records.append({
                    "symbol": f"{prefix}_{strike_value:g}{side_value}",
                    "strike": strike_value,
                    "option_type": side_value,
                    "expiry": f"Saved through {latest_day.date()}",
                    "history_path": latest_path,
                })
            if not snapshot_chain.empty and history_records:
                chain = pd.concat([snapshot_chain, pd.DataFrame(history_records)], ignore_index=True, sort=False)
            elif not snapshot_chain.empty:
                chain = snapshot_chain
            elif history_records:
                chain = pd.DataFrame(history_records)

        if chain.empty:
            if connected:
                st.warning("No saved snapshot exists and Kotak Neo did not return an option chain. The app tried the broker directly; inspect the fetch error above and confirm the Neo session and available expiry.")
            else:
                st.warning("No saved snapshot exists and Kotak Neo is disconnected. Connect Neo to request the latest available expiry; no prices will be fabricated.")
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
        snapshot_expiries = [str(v) for v in chain["expiry"].dropna().unique() if str(v).strip()]
        expiries = sorted(set(snapshot_expiries) | set(broker_expiries))
        if not expiries:
            st.warning("No expiry/session label is available from Kotak Neo or saved data.")
            return
        selector_cols = st.columns(3)
        expiry = selector_cols[0].selectbox("Expiry date", expiries, key="compact_option_expiry")
        exp_chain = chain[chain["expiry"].astype(str) == expiry]
        if exp_chain.empty and market_open and connected and expiry in broker_expiries:
            try:
                if provider is None:
                    from marketdata.providers.kotak_neo import KotakNeoProvider
                    provider = KotakNeoProvider(broker.client)
                selected_chain = provider.get_option_chain(
                    underlying=instrument, exchange=exchange, expiry=expiry,
                    count=40, enrich_quotes=True,
                )
                if selected_chain:
                    store.save_option_chain_snapshot(instrument, selected_chain, captured_at=now)
                    refresh_map = dict(st.session_state.get("compact_live_refresh_at", {}))
                    refresh_map[instrument] = now
                    st.session_state["compact_live_refresh_at"] = refresh_map
                    st.rerun()
                st.warning(f"Kotak Neo returned no contracts for expiry {expiry}.")
            except Exception as exc:
                st.warning(f"Could not load selected expiry {expiry} from Kotak Neo: {exc}")
        if exp_chain.empty:
            st.info(
                f"Expiry {expiry} is not present in the saved option-chain snapshot for {selected_date_label}. "
                "Choose another Snapshot / trading date to backtrace a prior session, or connect to Kotak Neo during market hours. "
                "Historical data is never fabricated."
            )
            return
        sides = [s for s in ("CE", "PE") if s in set(exp_chain["option_type"])]
        if not sides:
            st.warning("No CE/PE contracts for the selected expiry.")
            return
        side = selector_cols[1].selectbox("Option type", sides, key="compact_option_side")
        side_chain = exp_chain[exp_chain["option_type"] == side]
        strikes = sorted(float(v) for v in side_chain["strike"].dropna().unique())
        strike = selector_cols[2].selectbox("Strike", strikes, key="compact_option_strike")
        selected = side_chain[side_chain["strike"] == strike]
        if selected.empty:
            st.warning("Selected option contract is unavailable.")
            return
        contract = selected.iloc[0]
        symbol = str(contract["symbol"]).strip()
        st.caption(f"Selected contract: {symbol} · {side} · strike {strike:g} · expiry/session {expiry}")

        # Show quote fields from the same selected Kotak snapshot, if present.
        # These values retain the snapshot timestamp above and are never implied
        # to be current when the market is closed or the feed is stale.
        quote_fields = [
            ("LTP", "ltp", "₹"),
            ("Bid", "bid", "₹"),
            ("Ask", "ask", "₹"),
            ("Volume", "volume", ""),
            ("Open interest", "open_interest", ""),
            ("OI change", "oi_change", ""),
        ]
        available_quotes = []
        for label, column, prefix in quote_fields:
            if column not in contract.index or pd.isna(contract.get(column)):
                continue
            try:
                value = float(contract.get(column))
            except (TypeError, ValueError):
                continue
            if value == 0 and label in {"LTP", "Bid", "Ask"}:
                continue
            display = f"{prefix}{value:,.2f}" if prefix else f"{value:,.0f}"
            available_quotes.append((label, display))
        if available_quotes:
            quote_cols = st.columns(min(3, len(available_quotes)))
            for index, (label, display) in enumerate(available_quotes):
                quote_cols[index % len(quote_cols)].metric(label, display)
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
