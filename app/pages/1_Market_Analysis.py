import pandas as pd
import streamlit as st

from analysis.market_context import (
    analyze_breadth,
    analyze_option_positioning,
    analyze_price_action,
    combine_evidence,
)

st.set_page_config(page_title="Market Analysis", page_icon="📊", layout="wide")
st.title("NIFTY Market Analysis")
st.caption("Independent evidence dashboard • signal-only • no order placement")

st.warning(
    "This page analyses only the data you upload. It does not invent live quotes, "
    "infer buyers or writers from OI alone, or guarantee a trade outcome."
)

spot_file = st.file_uploader(
    "NIFTY OHLC candle CSV (3-minute or 5-minute)",
    type=["csv"],
    key="spot_candles",
    help="Required columns: open, high, low, close. Timestamp and volume are recommended.",
)
chain_file = st.file_uploader(
    "Option-chain CSV",
    type=["csv"],
    key="option_chain",
    help="Required columns: option_type (CE/PE), ltp_change, oi_change. Optional: strike, ltp, volume.",
)
breadth_file = st.file_uploader(
    "NIFTY 50 constituent breadth CSV",
    type=["csv"],
    key="breadth",
    help="Required columns: symbol, change_pct. Optional: above_vwap (true/false).",
)

candles = pd.read_csv(spot_file) if spot_file else pd.DataFrame()
chain = pd.read_csv(chain_file) if chain_file else pd.DataFrame()
breadth = pd.read_csv(breadth_file) if breadth_file else pd.DataFrame()

tabs = st.tabs(["Overview", "Price Action", "Option Positioning", "NIFTY 50 Breadth", "Validation Notes"])

price_evidence = analyze_price_action(candles)
option_evidence, classified_chain = analyze_option_positioning(chain)
breadth_evidence = analyze_breadth(breadth)
evidence = [price_evidence, option_evidence, breadth_evidence]
decision, explanation = combine_evidence(evidence)

with tabs[0]:
    st.subheader("Evidence-based market bias")
    st.metric("Current assessment", decision)
    st.write(explanation)
    cols = st.columns(3)
    for col, item in zip(cols, evidence):
        with col:
            st.markdown(f"**{item.label}**")
            st.write("Bullish" if item.direction == 1 else "Bearish" if item.direction == -1 else "Neutral / unavailable")
            st.caption(item.detail)
    st.info("A bias is not an entry signal. Wait for a completed candle, defined invalidation and acceptable risk/reward.")

with tabs[1]:
    st.subheader("Price structure")
    st.write(price_evidence.detail)
    if not candles.empty:
        st.dataframe(candles.tail(30), use_container_width=True, hide_index=True)
    else:
        st.info("Upload NIFTY candle data to assess recent structure.")

with tabs[2]:
    st.subheader("CE/PE positioning")
    st.write(option_evidence.detail)
    if not classified_chain.empty:
        view_cols = [c for c in ["option_type", "strike", "ltp", "ltp_change", "oi_change", "volume", "position_read"] if c in classified_chain.columns]
        st.dataframe(classified_chain[view_cols], use_container_width=True, hide_index=True)
        summary = classified_chain.groupby(["option_type", "position_read"], dropna=False).size().rename("contracts").reset_index()
        st.markdown("**Positioning distribution**")
        st.dataframe(summary, use_container_width=True, hide_index=True)
    else:
        st.info("Upload a timestamped or current option-chain export with paired price and OI changes.")

with tabs[3]:
    st.subheader("Constituent participation")
    st.write(breadth_evidence.detail)
    if not breadth.empty and {"symbol", "change_pct"}.issubset(breadth.columns):
        st.dataframe(breadth.sort_values("change_pct", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("Upload a constituent CSV. Index movement alone does not show market breadth.")

with tabs[4]:
    st.subheader("How to avoid overfitting")
    st.markdown(
        "- Keep price structure, derivatives positioning and constituent breadth as the three primary evidence groups.\n"
        "- Treat RSI, VWAP, EMA and ADX as context, not independent votes when they describe the same price movement.\n"
        "- Do not tune thresholds repeatedly on the same dates. Keep a chronological, untouched holdout period.\n"
        "- Use walk-forward validation across trending, sideways and volatile sessions.\n"
        "- Include brokerage, taxes, spread, slippage and realistic option fills.\n"
        "- Compare expectancy, profit factor, drawdown and stability—not win rate alone.\n"
        "- A single option-chain snapshot is not enough for a historical options backtest."
    )
    st.caption("The current page uses transparent rules and no fitted ML probability.")
