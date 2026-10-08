    )


# ============================================================
# Cross-market trend radar
# ============================================================
def _real_radar_snapshot(name: str, connected: bool = False) -> TrendSnapshot:
    from marketdata.daily_store import load_captured_candles
    frame, _source = load_captured_candles(name)
    if frame.empty:
        return TrendSnapshot(
            name, "DATA_UNAVAILABLE", 0.0, 0.0, "UNKNOWN", "UNKNOWN", "UNKNOWN",
            None, 0, False,
            "No real captured Kotak candles are available; synthetic radar data is disabled.",
        )
    # Captured files are historical evidence, not a live feed merely because Neo is connected.\n    return calculate_trend(name, frame, is_live=False)

st.subheader("Market Radar")
st.caption("Directional context for NIFTY, Crude Oil, Natural Gas and Copper. Research cards are explicitly marked when a live feed is unavailable.")
radar_names = ["NIFTY", "CRUDE", "NATGAS", "COPPER"]
radar_cols = st.columns(4)