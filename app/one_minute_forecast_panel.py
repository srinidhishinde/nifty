"""Streamlit UI for historical one-minute option candle forecasting.

CSV uploads are historical research only; they are never represented as live
broker prices and are not used for order execution.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from prediction.next_minute_forecast import forecast_next_minute

_REQUIRED = ["timestamp", "open", "high", "low", "close", "volume"]


def _normalize_csv(upload) -> pd.DataFrame:
    frame = pd.read_csv(upload)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    missing = [column for column in _REQUIRED if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    frame = frame[_REQUIRED].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    for column in _REQUIRED[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.isna().any().any():
        raise ValueError("CSV has blank or invalid timestamps/OHLCV values.")
    if frame.empty:
        raise ValueError("CSV contains no candle rows.")
    if (frame["high"] < frame[["open", "low", "close"]].max(axis=1)).any():
        raise ValueError("Invalid OHLCV row: high is below open, low, or close.")
    if (frame["low"] > frame[["open", "high", "close"]].min(axis=1)).any():
        raise ValueError("Invalid OHLCV row: low is above open, high, or close.")
    if (frame["volume"] < 0).any():
        raise ValueError("Volume cannot be negative.")
    return frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _walk_forward(frame: pd.DataFrame, *, quantile: float = 0.90, lookback: int = 30, range_method: str = "adaptive") -> pd.DataFrame:
    """Forecast row i from rows strictly before i; never pass the target row in."""
    results = []
    for target_index in range(3, len(frame)):
        history = frame.iloc[:target_index]
        actual = frame.iloc[target_index]
        forecast = forecast_next_minute(history, underlying="NIFTY", lookback=lookback, quantile=quantile, range_method=range_method)
        results.append({
            "timestamp": actual["timestamp"],
            "predicted_low": forecast.predicted_low,
            "predicted_high": forecast.predicted_high,
            "actual_low": float(actual["low"]),
            "actual_high": float(actual["high"]),
            "low_covered": forecast.predicted_low <= float(actual["low"]),
            "high_covered": forecast.predicted_high >= float(actual["high"]),
            "full_range_covered": (
                forecast.predicted_low <= float(actual["low"])
                and forecast.predicted_high >= float(actual["high"])
            ),
            "low_abs_error": abs(forecast.predicted_low - float(actual["low"])),
            "high_abs_error": abs(forecast.predicted_high - float(actual["high"])),
            "low_signed_error": forecast.predicted_low - float(actual["low"]),
            "high_signed_error": forecast.predicted_high - float(actual["high"]),
            "predicted_range_width": forecast.predicted_high - forecast.predicted_low,
            "actual_range_width": float(actual["high"]) - float(actual["low"]),
        })
    return pd.DataFrame(results)


def _comparison_summary(frame: pd.DataFrame, *, quantile: float, lookback: int) -> pd.DataFrame:
    """Evaluate both range methods on identical walk-forward target candles."""
    summaries = []
    for method in ("adaptive", "baseline"):
        replay = _walk_forward(frame, quantile=quantile, lookback=lookback, range_method=method)
        if replay.empty:
            continue
        predicted_width = float(replay["predicted_range_width"].mean())
        actual_width = float(replay["actual_range_width"].mean())
        summaries.append({
            "Model": "Adaptive (previous-close excursions)" if method == "adaptive" else "Baseline (open-to-extreme)",
            "Forecasts": len(replay),
            "Low coverage %": 100.0 * float(replay["low_covered"].mean()),
            "High coverage %": 100.0 * float(replay["high_covered"].mean()),
            "Full-range coverage %": 100.0 * float(replay["full_range_covered"].mean()),
            "Low MAE (₹)": float(replay["low_abs_error"].mean()),
            "High MAE (₹)": float(replay["high_abs_error"].mean()),
            "Avg predicted width (₹)": predicted_width,
            "Avg actual width (₹)": actual_width,
            "Width ratio (×)": predicted_width / actual_width if actual_width > 0 else float("nan"),
        })
    return pd.DataFrame(summaries)


def render_one_minute_forecast_panel() -> None:
    st.divider()
    st.header("One-minute High / Low Forecast")
    st.caption(
        "Historical CSV research panel. Upload completed option OHLCV candles. "
        "Forecasts are estimates—not guaranteed candle extremes, trade signals, or live Kotak prices."
    )
    uploaded = st.file_uploader(
        "Upload one-minute option OHLCV CSV",
        type=["csv"],
        key="one_minute_forecast_csv",
        help="Required columns: timestamp, open, high, low, close, volume.",
    )
    if uploaded is None:
        st.info("Upload data/backtest/kotak/NIFTY_22700CE_2026-10-09_1min.csv to view the chart and forecast.")
        return

    try:
        frame = _normalize_csv(uploaded)
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"Could not read this OHLCV CSV: {exc}")
        return

    if len(frame) < 4:
        st.warning("At least four valid one-minute candles are required for a forecast and walk-forward check.")
        return

    st.markdown("#### Forecast range sensitivity")
    quantile_pct = st.slider(
        "Range width / historical excursion quantile",
        min_value=60,
        max_value=95,
        value=90,
        step=5,
        help="Higher values usually widen the estimated range and may improve candle-range coverage, but increase the range width. This is not a probability guarantee.",
        key="one_minute_forecast_quantile",
    )
    quantile = quantile_pct / 100.0
    range_method_label = st.selectbox(
        "Forecast model",
        options=["Adaptive (previous-close excursions)", "Baseline (open-to-extreme)"],
        index=0,
        key="one_minute_forecast_model",
        help="Adaptive ranges include gaps relative to the previous close. The walk-forward panel compares only past data; check results rather than assuming one model is better.",
    )
    range_method = "adaptive" if range_method_label.startswith("Adaptive") else "baseline"
    lookback = st.slider(
        "Recent candles used",
        min_value=10,
        max_value=385,
        value=385,
        step=5,
        key="one_minute_forecast_lookback",
        help="Use up to 385 completed candles when the uploaded file contains enough history. Forecasts still use only candles available before the target candle.",
    )
    forecast = forecast_next_minute(
        frame, underlying="NIFTY", lookback=lookback, quantile=quantile, range_method=range_method
    )
    target_time = pd.Timestamp(forecast.target_candle_time)
    latest = frame.iloc[-1]

    st.caption(
        f"File: {uploaded.name} · {len(frame):,} candles · "
        f"{frame['timestamp'].iloc[0]} to {frame['timestamp'].iloc[-1]} · HISTORICAL"
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Reference premium", f"₹{forecast.reference_price:.2f}")
    m2.metric("Next-minute estimated low", f"₹{forecast.predicted_low:.2f}")
    m3.metric("Next-minute estimated high", f"₹{forecast.predicted_high:.2f}")
    m4.metric("Direction context", forecast.direction)
    st.write(f"**Target candle:** {target_time}  |  **Method:** {forecast.method}  |  **Model:** {range_method_label}  |  **Range quantile:** {quantile_pct}%  |  **History:** {lookback} bars")
    st.caption(forecast.disclaimer)

    visible = frame.tail(385)
    chart = go.Figure(data=[go.Candlestick(
        x=visible["timestamp"],
        open=visible["open"],
        high=visible["high"],
        low=visible["low"],
        close=visible["close"],
        name="Historical option premium",
    )])
    chart.add_trace(go.Scatter(
        x=[target_time, target_time],
        y=[forecast.predicted_low, forecast.predicted_high],
        mode="markers+lines",
        name="Next-minute estimated range",
        line={"width": 5},
        marker={"size": 9},
    ))
    chart.add_trace(go.Scatter(
        x=[latest["timestamp"], target_time],
        y=[forecast.reference_price, forecast.reference_price],
        mode="lines",
        name="Reference close",
        line={"dash": "dot", "width": 1},
    ))
    chart.update_layout(
        title="Option premium — last 385 candles and next-minute estimate",
        xaxis_title="Candle time",
        yaxis_title="Premium (₹)",
        xaxis_rangeslider_visible=False,
        height=520,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
    )
    st.plotly_chart(chart, width="stretch")

    with st.expander("Adaptive vs baseline — same-candle comparison", expanded=True):
        comparison = _comparison_summary(frame, quantile=quantile, lookback=lookback)
        if comparison.empty:
            st.info("Not enough candles to compare the two range methods.")
        else:
            st.caption(
                "Both methods are tested against exactly the same target candles, using the same range quantile "
                f"({quantile_pct}%) and lookback ({lookback} completed candles). This is a historical comparison, not a live-performance guarantee."
            )
            st.dataframe(
                comparison.round({
                    "Low coverage %": 2,
                    "High coverage %": 2,
                    "Full-range coverage %": 2,
                    "Low MAE (₹)": 3,
                    "High MAE (₹)": 3,
                    "Avg predicted width (₹)": 3,
                    "Avg actual width (₹)": 3,
                    "Width ratio (×)": 3,
                }),
                width="stretch",
                hide_index=True,
            )
            if len(comparison) == 2:
                adaptive_row = comparison.iloc[0]
                baseline_row = comparison.iloc[1]
                st.write(
                    "**Adaptive minus baseline:** "
                    f"full-range coverage {adaptive_row['Full-range coverage %'] - baseline_row['Full-range coverage %']:+.2f} pp; "
                    f"low MAE ₹{adaptive_row['Low MAE (₹)'] - baseline_row['Low MAE (₹)']:+.3f}; "
                    f"high MAE ₹{adaptive_row['High MAE (₹)'] - baseline_row['High MAE (₹)']:+.3f}; "
                    f"predicted width ₹{adaptive_row['Avg predicted width (₹)'] - baseline_row['Avg predicted width (₹)']:+.3f}."
                )
                st.caption("Higher coverage is not automatically better if it is achieved by much wider ranges. Review coverage, both MAEs, and width together.")
            st.download_button(
                "Download adaptive-vs-baseline comparison (CSV)",
                data=comparison.to_csv(index=False).encode("utf-8"),
                file_name="nifty_adaptive_vs_baseline_comparison.csv",
                mime="text/csv",
                key="one_minute_model_comparison_download",
            )

    with st.expander("Selected model — walk-forward historical accuracy", expanded=True):
        replay = _walk_forward(frame, quantile=quantile, lookback=lookback, range_method=range_method)
        if replay.empty:
            st.info("Not enough candles to calculate walk-forward metrics.")
            return
        total = len(replay)
        low_rate = 100 * replay["low_covered"].mean()
        high_rate = 100 * replay["high_covered"].mean()
        range_rate = 100 * replay["full_range_covered"].mean()
        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric("Forecasts checked", f"{total:,}")
        e2.metric("Low bound coverage", f"{low_rate:.1f}%")
        e3.metric("High bound coverage", f"{high_rate:.1f}%")
        e4.metric("Full candle range covered", f"{range_rate:.1f}%")
        e5.metric("Low / high MAE", f"₹{replay['low_abs_error'].mean():.2f} / ₹{replay['high_abs_error'].mean():.2f}")

        predicted_width = float(replay["predicted_range_width"].mean())
        actual_width = float(replay["actual_range_width"].mean())
        width_ratio = predicted_width / actual_width if actual_width > 0 else float("inf")
        b1, b2, b3 = st.columns(3)
        b1.metric("Average predicted range", f"₹{predicted_width:.2f}")
        b2.metric("Average actual candle range", f"₹{actual_width:.2f}")
        b3.metric("Predicted / actual width", f"{width_ratio:.2f}×" if pd.notna(width_ratio) else "N/A")
        st.caption(
            f"Mean signed low/high error: ₹{replay['low_signed_error'].mean():+.2f} / ₹{replay['high_signed_error'].mean():+.2f}. "
            "Negative low error means the predicted low was below the actual low; positive high error means the predicted high was above the actual high."
        )
        if total < 100:
            st.warning(
                f"INSUFFICIENT DATA: only {total} walk-forward forecasts are available. "
                "Do not treat 100% coverage as evidence of reliability; collect at least 100 predictions across multiple sessions before judging the model."
            )
        if width_ratio >= 2.5:
            st.warning(
                f"RANGE TOO WIDE: the average predicted range is {width_ratio:.2f}× the actual candle range. "
                "Coverage may be high because the bounds are broad; compare MAE and width on unseen sessions before using for scalping."
            )
        st.caption(
            f"Each historical forecast used only candles before its target candle, with {range_method_label} and the selected {quantile_pct}% excursion quantile. "
            "Higher range quantiles can increase coverage by widening bounds; they do not improve exact high/low timing, directional accuracy, or profitability. This one-session sample may not generalize."
        )
        st.caption(f"Showing all {total:,} walk-forward forecast rows. The table is scrollable; use the CSV download for the complete history.")
        export_df = replay.rename(columns={
                "timestamp": "Target time",
                "predicted_low": "Predicted low",
                "predicted_high": "Predicted high",
                "actual_low": "Actual low",
                "actual_high": "Actual high",
                "low_covered": "Low covered",
                "high_covered": "High covered",
                "full_range_covered": "Full range covered",
                "low_abs_error": "Low abs error",
                "high_abs_error": "High abs error",
            })
        st.download_button(
            "Download complete candle forecast history (CSV)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="nifty_one_minute_forecast_history.csv",
            mime="text/csv",
            key="one_minute_forecast_history_download",
        )
        st.dataframe(
            export_df,
            width="stretch",
            hide_index=True,
            height=600,
        )
