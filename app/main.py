import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.settings import settings
from broker.kotak_neo import KotakNeoBroker
from features.option_chain import (
    OptionAnalysis,
    OptionContract,
    analyze_option,
)
from marketdata.providers.kotak_neo import KotakNeoProvider
from strategy.ce_pe_selector import (
    CEPESelector,
    MarketContext,
)
from strategy.strike_selector import StrikeSelector
from backtest.rule_engine import RuleBacktestEngine
from features.technical.indicators import add_indicators
from strategy.rules import StrategyConfig, evaluate_rules
from marketdata.option_chain_csv import is_option_chain_snapshot, parse_option_chain_csv


# ============================================================
# Research signal
# ============================================================

def build_research_signal(
    instrument: str,
    timeframe: str,
    seed: int = 42,
):
    """
    Generate deterministic synthetic research data.

    This function does not connect to a broker.
    It is used for research and paper-trading UI development.

    Returns:

        (
            MarketContext,
            CE OptionAnalysis,
            PE OptionAnalysis,
            Signal,
            ML probability,
        )
    """

    rng = random.Random(seed)

    trend = rng.choice(
        [
            "BULLISH",
            "BEARISH",
            "NEUTRAL",
        ]
    )

    momentum = rng.choice(
        [
            "POSITIVE",
            "NEGATIVE",
            "NEUTRAL",
        ]
    )

    price_vs_vwap = rng.choice(
        [
            "ABOVE",
            "BELOW",
            "AT",
        ]
    )

    volatility_regime = rng.choice(
        [
            "LOW",
            "NORMAL",
            "HIGH",
        ]
    )

    context = MarketContext(
        timeframe=timeframe,
        trend=trend,
        momentum=momentum,
        price_vs_vwap=price_vs_vwap,
        volatility_regime=volatility_regime,
    )

    ce_score = rng.uniform(
        40.0,
        75.0,
    )

    pe_score = rng.uniform(
        40.0,
        75.0,
    )

    ce = OptionAnalysis(
        option_type="CE",
        strike=25000.0,
        ltp=100.0,
        volume=10000.0,
        open_interest=20000.0,
        oi_change=rng.uniform(
            -5000.0,
            5000.0,
        ),
        implied_volatility=rng.uniform(
            12.0,
            45.0,
        ),
        score=round(
            ce_score,
            2,
        ),
        reasons=(
            "Research-mode synthetic option data",
        ),
    )

    pe = OptionAnalysis(
        option_type="PE",
        strike=25000.0,
        ltp=100.0,
        volume=10000.0,
        open_interest=20000.0,
        oi_change=rng.uniform(
            -5000.0,
            5000.0,
        ),
        implied_volatility=rng.uniform(
            12.0,
            45.0,
        ),
        score=round(
            pe_score,
            2,
        ),
        reasons=(
            "Research-mode synthetic option data",
        ),
    )

    ml_probability = round(
        rng.uniform(
            0.35,
            0.80,
        ),
        4,
    )

    selector = CEPESelector(
        minimum_confidence=(
            settings.minimum_signal_confidence
        ),
        minimum_edge=(
            settings.minimum_signal_edge
        ),
    )

    signal = selector.generate(
        context=context,
        ce=ce,
        pe=pe,
    )

    return (
        context,
        ce,
        pe,
        signal,
        ml_probability,
    )


# ============================================================
# Synthetic option chain
# ============================================================

def build_research_option_chain(
    instrument: str,
    spot: float,
    seed: int,
    strike_step: float,
) -> list[OptionContract]:
    """
    Build a deterministic synthetic option chain.

    This is research-only data. It must not be interpreted
    as live market data.
    """

    rng = random.Random(
        f"{instrument}:{seed}:{spot}"
    )

    atm_strike = round(
        spot / strike_step
    ) * strike_step

    strikes = [
        atm_strike
        + (
            offset * strike_step
        )
        for offset in range(-10, 11)
    ]

    contracts: list[OptionContract] = []

    for strike in strikes:

        distance = abs(
            strike - spot
        )

        # Synthetic option premium.
        intrinsic_ce = max(
            0.0,
            spot - strike,
        )

        intrinsic_pe = max(
            0.0,
            strike - spot,
        )

        time_value = max(
            8.0,
            120.0
            - distance * 0.45,
        )

        ce_ltp = max(
            1.0,
            intrinsic_ce
            + time_value
            + rng.uniform(-3.0, 3.0),
        )

        pe_ltp = max(
            1.0,
            intrinsic_pe
            + time_value
            + rng.uniform(-3.0, 3.0),
        )

        ce_spread = max(
            0.5,
            ce_ltp * rng.uniform(
                0.003,
                0.012,
            ),
        )

        pe_spread = max(
            0.5,
            pe_ltp * rng.uniform(
                0.003,
                0.012,
            ),
        )

        ce_volume = rng.randint(
            5000,
            80000,
        )

        pe_volume = rng.randint(
            5000,
            80000,
        )

        ce_oi = rng.randint(
            10000,
            150000,
        )

        pe_oi = rng.randint(
            10000,
            150000,
        )

        ce_oi_change = rng.randint(
            -30000,
            30000,
        )

        pe_oi_change = rng.randint(
            -30000,
            30000,
        )

        ce_iv = rng.uniform(
            12.0,
            38.0,
        )

        pe_iv = rng.uniform(
            12.0,
            38.0,
        )

        expiry = (
            pd.Timestamp.now()
            .normalize()
            + pd.Timedelta(days=7)
        ).strftime(
            "%Y-%m-%d"
        )

        contracts.append(
            OptionContract(
                symbol=(
                    f"{instrument}"
                    f"{int(strike)}CE"
                ),
                expiry=expiry,
                strike=float(strike),
                option_type="CE",
                ltp=round(
                    ce_ltp,
                    2,
                ),
                bid=round(
                    max(
                        0.05,
                        ce_ltp
                        - ce_spread,
                    ),
                    2,
                ),
                ask=round(
                    ce_ltp
                    + ce_spread,
                    2,
                ),
                volume=float(
                    ce_volume
                ),
                open_interest=float(
                    ce_oi
                ),
                oi_change=float(
                    ce_oi_change
                ),
                implied_volatility=round(
                    ce_iv,
                    2,
                ),
            )
        )

        contracts.append(
            OptionContract(
                symbol=(
                    f"{instrument}"
                    f"{int(strike)}PE"
                ),
                expiry=expiry,
                strike=float(strike),
                option_type="PE",
                ltp=round(
                    pe_ltp,
                    2,
                ),
                bid=round(
                    max(
                        0.05,
                        pe_ltp
                        - pe_spread,
                    ),
                    2,
                ),
                ask=round(
                    pe_ltp
                    + pe_spread,
                    2,
                ),
                volume=float(
                    pe_volume
                ),
                open_interest=float(
                    pe_oi
                ),
                oi_change=float(
                    pe_oi_change
                ),
                implied_volatility=round(
                    pe_iv,
                    2,
                ),
            )
        )

    return contracts


# ============================================================
# Option-chain dataframe
# ============================================================

def build_option_chain_dataframe(
    contracts: list[OptionContract],
    spot: float,
) -> pd.DataFrame:

    rows = {}

    for contract in contracts:

        analysis = analyze_option(
            contract
        )

        row = rows.setdefault(
            contract.strike,
            {
                "Strike": contract.strike,
            },
        )

        prefix = contract.option_type

        row[
            f"{prefix} LTP"
        ] = contract.ltp

        row[
            f"{prefix} Volume"
        ] = int(contract.volume)

        row[
            f"{prefix} OI"
        ] = int(contract.open_interest)

        row[
            f"{prefix} OI Chg"
        ] = int(contract.oi_change)

        row[
            f"{prefix} IV"
        ] = contract.implied_volatility

        row[
            f"{prefix} Score"
        ] = analysis.score

    if not rows:
        return pd.DataFrame(columns=[
            "Strike", "CE LTP", "CE Volume", "CE OI", "CE OI Chg",
            "CE IV", "CE Score", "PE LTP", "PE Volume", "PE OI",
            "PE OI Chg", "PE IV", "PE Score", "Distance", "ATM",
        ])

    dataframe = pd.DataFrame(
        list(rows.values())
    )

    option_columns = [
        "CE LTP", "CE Volume", "CE OI", "CE OI Chg", "CE IV", "CE Score",
        "PE LTP", "PE Volume", "PE OI", "PE OI Chg", "PE IV", "PE Score",
    ]
    for column in option_columns:
        if column not in dataframe:
            dataframe[column] = np.nan

    dataframe[
        "Distance"
    ] = (
        dataframe["Strike"]
        - spot
    ).abs()

    dataframe[
        "ATM"
    ] = np.where(
        dataframe["Distance"]
        == dataframe["Distance"].min(),
        "ATM",
        "",
    )

    dataframe = dataframe.sort_values(
        "Strike"
    ).reset_index(
        drop=True
    )

    return dataframe


# ============================================================
# Strike configuration
# ============================================================

def get_strike_step(
    instrument: str,
) -> float:

    if instrument == "NIFTY":
        return 50.0

    if instrument == "BANKNIFTY":
        return 100.0

    if instrument in {
        "CRUDEOIL",
        "NATURALGAS",
    }:
        return 10.0

    return 100.0


# ============================================================
# Streamlit
# ============================================================

st.set_page_config(
    page_title="AI Derivatives Terminal",
    page_icon="AI",
    layout="wide",
)

st.title(
    "AI Derivatives Terminal"
)

st.caption(
    "NIFTY + MCX | Research / Backtest / Paper Trading"
)

st.sidebar.header(
    "Configuration"
)

instrument = st.sidebar.selectbox(
    "Instrument",
    [
        "NIFTY",
        "CRUDEOIL",
        "NATURALGAS",
        "COPPER",
        "SILVER",
        "GOLD",
    ],
)

timeframe = st.sidebar.selectbox(
    "Timeframe",
    [
        "5m",
        "15m",
    ],
)

environment = st.sidebar.selectbox(
    "Environment",
    [
        "RESEARCH",
        "BACKTEST",
        "UAT",
        "PAPER",
    ],
)

seed = st.sidebar.number_input(
    "Research Seed",
    min_value=1,
    max_value=1_000_000,
    value=42,
    step=1,
)

(
    context,
    ce,
    pe,
    signal,
    ml_probability,
) = build_research_signal(
    instrument=instrument,
    timeframe=timeframe,
    seed=int(seed),
)



st.sidebar.divider()

st.sidebar.subheader("Kotak Neo")

neo_totp = st.sidebar.text_input(
    "6-digit TOTP",
    type="password",
    max_chars=6,
    placeholder="Enter current TOTP",
)

neo_connect = st.sidebar.button(
    "Connect to Kotak Neo",
    use_container_width=True,
)

if "neo_broker" not in st.session_state:
    st.session_state["neo_broker"] = KotakNeoBroker()

neo_broker = st.session_state["neo_broker"]

if neo_connect:
    if not neo_totp.isdigit() or len(neo_totp) != 6:
        st.session_state["neo_authenticated"] = False
        st.sidebar.error("Enter the current 6-digit TOTP.")
    else:
        connection = neo_broker.authenticate(neo_totp)
        st.session_state["neo_authenticated"] = connection.connected
        if connection.connected:
            st.sidebar.success(connection.message)
        else:
            st.sidebar.error(connection.message)

neo_status = neo_broker.connection_status()
st.sidebar.caption(
    "Kotak Neo: " + ("CONNECTED" if neo_status.connected else "NOT CONNECTED")
)


# ============================================================
# Research spot
# ============================================================

# Live underlying price when Kotak Neo is connected.
# Research-mode synthetic spot is retained only when the broker is not connected.
spot = None
if neo_status.connected:
    try:
        provider = KotakNeoProvider(neo_broker.client)
        if instrument == "NIFTY":
            spot = provider.get_index_quote("Nifty 50").ltp
        else:
            st.info(
                f"Live underlying quote integration for {instrument} is not wired yet; "
                "no synthetic price is used for live mode."
            )
    except Exception as exc:
        st.error(f"Live underlying quote request failed: {exc}")

if spot is None and not neo_status.connected:
    spot_rng = random.Random(f"spot:{instrument}:{seed}")
    base_spot = {
        "NIFTY": 25040.0,
        "CRUDEOIL": 6500.0,
        "NATURALGAS": 300.0,
        "COPPER": 950.0,
        "SILVER": 95000.0,
        "GOLD": 125000.0,
    }.get(instrument, 25000.0)
    spot = base_spot + spot_rng.uniform(-100, 100)

if spot is None:
    spot = 0.0


# ============================================================
# Risk summary
# ============================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Capital",
    f"Rs {settings.starting_capital:,.0f}",
)

col2.metric(
    "Max Risk / Trade",
    f"Rs {settings.max_loss_per_trade:,.0f}",
)

col3.metric(
    "Daily Loss Limit",
    f"Rs {settings.max_daily_loss:,.0f}",
)

col4.metric(
    "Trades / Day",
    f"0 / {settings.max_trades_per_day}",
)

st.divider()


# ============================================================
# Signal
# ============================================================

st.subheader(
    f"{instrument} - AI Signal"
)

c1, c2, c3 = st.columns(3)

c1.metric(
    "15m Regime",
    context.trend,
)

c2.metric(
    f"{timeframe} Signal",
    signal.decision,
)

c3.metric(
    "ML Probability",
    f"{ml_probability * 100:.1f}%",
)

st.divider()


# ============================================================
# Market context
# ============================================================

st.subheader(
    "Market Context"
)

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Spot",
    f"{spot:,.2f}",
)

m2.metric(
    "Trend",
    context.trend,
)

m3.metric(
    "Momentum",
    context.momentum,
)

m4.metric(
    "Volatility",
    context.volatility_regime,
)

st.divider()


# ============================================================
# CE / WAIT / PE
# ============================================================

left, middle, right = st.columns(3)

with left:

    st.subheader("CE")

    st.metric(
        "Score",
        f"{signal.ce_score:.1f}",
    )

    if signal.decision == "CE":
        st.success("Selected")
    else:
        st.write("Not selected")


with middle:

    st.subheader("WAIT")

    st.metric(
        "Edge",
        f"{signal.edge:.1f}",
    )

    if signal.decision == "WAIT":
        st.warning(
            "Insufficient confirmation"
        )
    else:
        st.write(
            "Available when evidence conflicts"
        )


with right:

    st.subheader("PE")

    st.metric(
        "Score",
        f"{signal.pe_score:.1f}",
    )

    if signal.decision == "PE":
        st.success("Selected")
    else:
        st.write("Not selected")


st.divider()


# ============================================================
# Option chain
# ============================================================

authenticated = neo_status.connected
live_contracts: list[OptionContract] = []

if instrument == "NIFTY" and authenticated:
    now = pd.Timestamp.now()
    market_open = (
        now.weekday() < 5
        and now.time() >= pd.Timestamp("09:15").time()
        and now.time() <= pd.Timestamp("15:30").time()
    )
    if not market_open:
        st.info(
            "NIFTY market is currently closed. Live option-chain polling is paused; "
            "the research chain below is clearly labelled and is not live data."
        )
    else:
        try:
            provider = KotakNeoProvider(neo_broker.client)
            live_contracts = provider.get_option_chain(
                underlying="NIFTY",
                exchange="nse_fo",
                count=40,
                enrich_quotes=True,
            )
            if not live_contracts:
                st.warning(
                    "Kotak Neo returned no NIFTY option contracts. "
                    "The research chain is shown separately."
                )
        except Exception as exc:
            st.warning(
                f"Live NIFTY option-chain unavailable: {exc}. "
                "Showing clearly labelled research data instead."
            )
elif instrument != "NIFTY":
    st.info(
        "Live option-chain display is currently implemented for NIFTY. "
        "MCX instruments use futures/spot market data rather than an option chain."
    )
else:
    st.warning(
        "Kotak Neo is not connected. Connect with TOTP to load live option-chain data."
    )

st.subheader("Option Chain")
st.caption(
    "Live Kotak Neo data is shown when connected. "
    "Synthetic data is never substituted for a failed live request."
)

chain_col1, chain_col2, chain_col3 = st.columns(3)

with chain_col1:

    strike_view = st.selectbox(
        "Strike View",
        [
            "Full Chain",
            "2 ATM + 5 OTM",
        ],
    )

with chain_col2:

    chain_side = st.selectbox(
        "Option Side",
        [
            "CE + PE",
            "CE",
            "PE",
        ],
    )

with chain_col3:

    st.metric(
        "Underlying",
        f"{spot:,.2f}",
    )


strike_step = get_strike_step(
    instrument
)

if live_contracts:
    # Convert broker-native contracts into the UI's normalized analysis model.
    contracts = [
        OptionContract(
            symbol=x.symbol,
            expiry=x.expiry,
            strike=x.strike,
            option_type=x.option_type,
            ltp=float(x.ltp or 0.0),
            bid=float(x.bid or 0.0),
            ask=float(x.ask or 0.0),
            volume=float(x.volume or 0.0),
            open_interest=float(x.open_interest or 0.0),
            oi_change=float(x.oi_change or 0.0),
            implied_volatility=0.0,
        )
        for x in live_contracts
    ]
else:
    contracts = build_research_option_chain(
        instrument=instrument,
        spot=spot,
        seed=int(seed),
        strike_step=strike_step,
    )

chain_df = build_option_chain_dataframe(
    contracts=contracts,
    spot=spot,
)


# ------------------------------------------------------------
# 2 ATM + 5 OTM selection
# ------------------------------------------------------------

if strike_view == "2 ATM + 5 OTM":

    strikes = sorted(
        chain_df["Strike"]
        .astype(float)
        .tolist()
    )

    selected = StrikeSelector.select(
        strikes=strikes,
        spot=spot,
        atm_count=2,
        otm_count=5,
    )

    selected_strikes = (
        selected["atm"]
        + selected["otm"]
    )

    selected_strikes = list(
        dict.fromkeys(
            selected_strikes
        )
    )

    chain_df = chain_df[
        chain_df["Strike"].isin(
            selected_strikes
        )
    ].copy()

    chain_df = chain_df.sort_values(
        "Strike"
    )


# ------------------------------------------------------------
# Side filter
# ------------------------------------------------------------

if chain_side == "CE":

    visible_columns = [
        "Strike",
        "CE LTP",
        "CE Volume",
        "CE OI",
        "CE OI Chg",
        "CE IV",
        "CE Score",
        "ATM",
    ]

elif chain_side == "PE":

    visible_columns = [
        "Strike",
        "PE LTP",
        "PE Volume",
        "PE OI",
        "PE OI Chg",
        "PE IV",
        "PE Score",
        "ATM",
    ]

else:

    visible_columns = [
        "CE Score",
        "CE IV",
        "CE OI Chg",
        "CE OI",
        "CE Volume",
        "CE LTP",
        "Strike",
        "PE LTP",
        "PE Volume",
        "PE OI",
        "PE OI Chg",
        "PE IV",
        "PE Score",
        "ATM",
    ]


display_df = chain_df[
    visible_columns
].copy()


# ------------------------------------------------------------
# Highlight ATM
# ------------------------------------------------------------

def highlight_atm(
    row: pd.Series,
):

    styles = [
        ""
        for _ in row
    ]

    if row.get("ATM") == "ATM":

        styles = [
            "background-color: #173f5f; color: white"
            for _ in row
        ]

    return styles


if not live_contracts:
    st.caption("RESEARCH DATA — deterministic synthetic option chain; not a broker feed.")

st.dataframe(
    display_df.style.apply(
        highlight_atm,
        axis=1,
    ),
    use_container_width=True,
    hide_index=True,
)


st.caption(
    "ATM is the strike closest to the displayed underlying price. "
    "2 ATM + 5 OTM is a research selection view, not an order instruction."
)

st.divider()


# ============================================================
# Selected contract summary
# ============================================================

st.subheader(
    "Selected Option Candidates"
)

candidate_col1, candidate_col2 = st.columns(2)

ce_candidates = chain_df[
    "CE Score"
].sort_values(
    ascending=False
)

pe_candidates = chain_df[
    "PE Score"
].sort_values(
    ascending=False
)

with candidate_col1:

    st.markdown(
        "### CE Candidate"
    )

    if not ce_candidates.empty:

        best_ce_strike = float(
            ce_candidates.index[
                0
            ]
            if False
            else chain_df.loc[
                ce_candidates.idxmax(),
                "Strike",
            ]
        )

        best_ce_row = chain_df[
            chain_df["Strike"]
            == best_ce_strike
        ].iloc[0]

        st.metric(
            "Strike",
            f"{best_ce_strike:,.0f}",
        )

        st.metric(
            "Score",
            f"{best_ce_row['CE Score']:.1f}",
        )

        st.write(
            f"LTP: Rs {best_ce_row['CE LTP']:.2f}"
        )

        st.write(
            f"OI: {best_ce_row['CE OI']:,.0f}"
        )

        st.write(
            f"OI Change: "
            f"{best_ce_row['CE OI Chg']:,.0f}"
        )

with candidate_col2:

    st.markdown(
        "### PE Candidate"
    )

    if not pe_candidates.empty:

        best_pe_strike = float(
            chain_df.loc[
                pe_candidates.idxmax(),
                "Strike",
            ]
        )

        best_pe_row = chain_df[
            chain_df["Strike"]
            == best_pe_strike
        ].iloc[0]

        st.metric(
            "Strike",
            f"{best_pe_strike:,.0f}",
        )

        st.metric(
            "Score",
            f"{best_pe_row['PE Score']:.1f}",
        )

        st.write(
            f"LTP: Rs {best_pe_row['PE LTP']:.2f}"
        )

        st.write(
            f"OI: {best_pe_row['PE OI']:,.0f}"
        )

        st.write(
            f"OI Change: "
            f"{best_pe_row['PE OI Chg']:,.0f}"
        )


st.divider()


# ============================================================
# Signal reasons
# ============================================================

st.subheader(
    "Signal Reasons"
)

for reason in signal.reasons:

    st.write(
        f"- {reason}"
    )


st.divider()


# ============================================================
# Research chart
# ============================================================

st.subheader(
    "Research Price Chart"
)

chart_rng = np.random.default_rng(
    int(seed)
)

dates = pd.date_range(
    end=pd.Timestamp.now(),
    periods=100,
    freq="5min",
)

prices = (
    spot
    + np.cumsum(
        chart_rng.normal(
            0,
            10,
            100,
        )
    )
)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=dates,
        y=prices,
        mode="lines",
        name="Price",
    )
)

fig.update_layout(
    height=450,
    template="plotly_dark",
)

st.plotly_chart(
    fig,
    use_container_width=True,
)


st.divider()


# ============================================================
# Option-chain CSV snapshot analysis
# ============================================================

st.subheader("Option-Chain Snapshot Import")
st.caption(
    "Your Calls/ Puts OI, IV, volume, delta, theta, vega and built-up export is an option-chain snapshot. "
    "It is suitable for option-chain analysis, but it is not OHLCV candle history and cannot be used directly "
    "for the seven-rule candle backtest."
)

option_csv = st.file_uploader(
    "Upload option-chain snapshot CSV",
    type=["csv"],
    key="option_chain_csv",
    help="Supports Strike plus Calls/ Puts OI, volume, IV, delta, theta, vega and built-up columns.",
)

if option_csv is not None:
    try:
        option_snapshot = pd.read_csv(option_csv)
        if not is_option_chain_snapshot(option_snapshot.columns):
            st.error(
                "This file is not recognized as the supported option-chain snapshot format. "
                "Expected a Strike column and Calls OI or Puts OI."
            )
        else:
            snapshot_contracts = parse_option_chain_csv(option_snapshot)
            snapshot_rows = []
            for contract in snapshot_contracts:
                snapshot_rows.append({
                    "Side": contract.option_type,
                    "Strike": contract.strike,
                    "LTP Change %": contract.ltp_change_pct,
                    "IV": contract.implied_volatility,
                    "OI": contract.open_interest,
                    "OI Change": contract.oi_change,
                    "Volume": contract.volume,
                    "Built Up": contract.built_up,
                    "Delta": contract.delta,
                    "Theta": contract.theta,
                    "Vega": contract.vega,
                    "Score": analyze_option(contract).score,
                })
            snapshot_df = pd.DataFrame(snapshot_rows).sort_values(["Strike", "Side"])
            st.success(f"Loaded {len(snapshot_contracts):,} option contracts from {option_csv.name}.")
            st.dataframe(snapshot_df, use_container_width=True, hide_index=True)
            st.info(
                "Rules 1–5 and 7 require candle/context history. Rule 6 additionally requires bid/ask and spread history. "
                "This snapshot has none of those fields, so it is not silently used as candle backtest input."
            )
    except Exception as exc:
        st.error(f"Option-chain snapshot import failed: {exc}")

st.divider()

# ============================================================
# Backtest
# ============================================================

st.subheader("Historical Rule Backtest")
st.caption(
    "Use real historical OHLCV data to measure how the seven rules would have performed "
    "on past candles. Synthetic demo data is for UI smoke-testing only and must not be "
    "used to judge strategy accuracy."
)

uploaded = st.file_uploader(
    "Upload historical OHLCV CSV",
    type=["csv"],
    help=(
        "Required columns: timestamp, open, high, low, close, volume. "
        "For trustworthy results, include warm-up candles before the period you want to score."
    ),
)

run_demo = st.button(
    "Run Synthetic Demo",
    use_container_width=True,
    help="UI/engine smoke test only. Do not treat synthetic results as evidence of profitability.",
)

bt_data = None
bt_source = None

if uploaded is not None:
    try:
        bt_data = pd.read_csv(uploaded)
        bt_source = f"Uploaded historical CSV: {uploaded.name}"
    except Exception as exc:
        st.error(f"Could not read backtest CSV: {exc}")
elif run_demo:
    rng = np.random.default_rng(int(seed))
    bt_ts = pd.date_range(
        end=pd.Timestamp.now().normalize() - pd.Timedelta(days=1),
        periods=800,
        freq="5min",
    )
    volatility = max(abs(float(spot)) * 0.0008, 1.0)
    base = float(spot) + np.cumsum(
        rng.normal(0, volatility, len(bt_ts))
    )
    bt_data = pd.DataFrame({
        "timestamp": bt_ts,
        "open": base,
        "high": base + rng.uniform(0, volatility * 2, len(bt_ts)),
        "low": base - rng.uniform(0, volatility * 2, len(bt_ts)),
        "close": base + rng.normal(0, volatility * 0.6, len(bt_ts)),
        "volume": rng.integers(10000, 100000, len(bt_ts)),
    })
    bt_source = "Synthetic demo data"

if bt_data is not None:
    required_bt = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
    missing_bt = required_bt - set(bt_data.columns)

    if missing_bt:
        if is_option_chain_snapshot(bt_data.columns):
            st.error(
                "This CSV is an option-chain snapshot, not historical OHLCV candle data. "
                "Use the 'Option-Chain Snapshot Import' section above for CE/PE analysis. "
                "For the seven-rule backtest, upload timestamp, open, high, low, close and volume."
            )
        else:
            st.error(
                f"Backtest data is missing required columns: {sorted(missing_bt)}"
            )
    else:
        try:
            bt_data = bt_data.copy()
            bt_data["timestamp"] = pd.to_datetime(
                bt_data["timestamp"],
                errors="coerce",
            )

            invalid_timestamps = int(bt_data["timestamp"].isna().sum())
            if invalid_timestamps:
                st.warning(
                    f"Dropped {invalid_timestamps:,} rows with invalid timestamps."
                )
                bt_data = bt_data.dropna(subset=["timestamp"])

            for column in [
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]:
                bt_data[column] = pd.to_numeric(
                    bt_data[column],
                    errors="coerce",
                )

            bt_data = bt_data.dropna(
                subset=[
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]
            )
            bt_data = (
                bt_data
                .sort_values("timestamp")
                .drop_duplicates("timestamp")
                .reset_index(drop=True)
            )

            if bt_data.empty:
                st.error("No valid historical candles remain after cleaning.")
            else:
                data_min = bt_data["timestamp"].min().date()
                data_max = bt_data["timestamp"].max().date()

                st.info(
                    f"Source: {bt_source} | Available data: "
                    f"{data_min} to {data_max} | Rows: {len(bt_data):,}"
                )

                date_range = st.date_input(
                    "Backtest date range",
                    value=(data_min, data_max),
                    min_value=data_min,
                    max_value=data_max,
                    help=(
                        "The selected dates are the scored period. "
                        "Keep earlier warm-up candles in the CSV so EMA/RSI/MACD/VWAP "
                        "have enough history."
                    ),
                )

                if isinstance(date_range, tuple) and len(date_range) == 2:
                    start_date, end_date = date_range
                else:
                    start_date = data_min
                    end_date = data_max

                selected = bt_data[
                    (bt_data["timestamp"].dt.date >= start_date)
                    & (bt_data["timestamp"].dt.date <= end_date)
                ].copy()

                st.write(
                    f"Selected period: **{start_date} → {end_date}** "
                    f"({len(selected):,} candles)"
                )

                run_historical = st.button(
                    "Run Historical Backtest",
                    type="primary",
                    use_container_width=True,
                )

                if run_historical:
                    if len(selected) < 60:
                        st.error(
                            "At least 60 candles are recommended for a meaningful "
                            "EMA(50)/indicator warm-up."
                        )
                    else:
                        try:
                            evaluation_start = pd.Timestamp(start_date)
                            evaluation_end = (
                                pd.Timestamp(end_date)
                                + pd.Timedelta(days=1)
                                - pd.Timedelta(microseconds=1)
                            )

                            result = RuleBacktestEngine(
                                starting_capital=settings.starting_capital,
                                risk_per_trade=settings.max_loss_per_trade,
                                instrument=instrument,
                            ).run(
                                bt_data,
                                symbol=instrument,
                                evaluation_start=evaluation_start,
                                evaluation_end=evaluation_end,
                            )
                            metrics = result.metrics
                            trades = result.trades.copy()

                            st.success(
                                "Historical backtest completed. "
                                "These results are historical simulation results, not a guarantee of future performance."
                            )

                            total_trades = int(metrics.total_trades)
                            wins = int(metrics.winning_trades)
                            losses = int(metrics.losing_trades)

                            top = st.columns(5)
                            top[0].metric(
                                "Closed Trades",
                                total_trades,
                            )
                            top[1].metric(
                                "Win Rate / Accuracy",
                                f"{metrics.win_rate_pct:.1f}%",
                            )
                            top[2].metric(
                                "Net P&L",
                                f"Rs {metrics.net_pnl:,.0f}",
                            )
                            top[3].metric(
                                "Return",
                                f"{metrics.return_pct:.2f}%",
                            )
                            top[4].metric(
                                "Profit Factor",
                                (
                                    "∞"
                                    if metrics.profit_factor == float("inf")
                                    else f"{metrics.profit_factor:.2f}"
                                ),
                            )

                            detail = st.columns(5)
                            detail[0].metric(
                                "Wins",
                                wins,
                            )
                            detail[1].metric(
                                "Losses",
                                losses,
                            )
                            detail[2].metric(
                                "Avg Trade",
                                f"Rs {metrics.average_trade:,.0f}",
                            )
                            detail[3].metric(
                                "Max Drawdown",
                                f"Rs {metrics.max_drawdown:,.0f}",
                            )
                            detail[4].metric(
                                "Max DD %",
                                f"{metrics.max_drawdown_pct:.2f}%",
                            )

                            if total_trades:
                                buy_trades = trades[
                                    trades["direction"] == "BUY"
                                ]
                                sell_trades = trades[
                                    trades["direction"] == "SELL"
                                ]

                                buy_accuracy = (
                                    (buy_trades["pnl"] > 0).mean() * 100
                                    if not buy_trades.empty
                                    else 0.0
                                )
                                sell_accuracy = (
                                    (sell_trades["pnl"] > 0).mean() * 100
                                    if not sell_trades.empty
                                    else 0.0
                                )

                                st.markdown("#### Direction Accuracy")
                                direction_cols = st.columns(2)
                                direction_cols[0].metric(
                                    "BUY Win Rate",
                                    f"{buy_accuracy:.1f}%",
                                    f"{len(buy_trades)} trades",
                                )
                                direction_cols[1].metric(
                                    "SELL Win Rate",
                                    f"{sell_accuracy:.1f}%",
                                    f"{len(sell_trades)} trades",
                                )

                                if "exit_time" in trades.columns:
                                    trades["entry_time"] = pd.to_datetime(
                                        trades["entry_time"],
                                        errors="coerce",
                                    )
                                    trades["exit_time"] = pd.to_datetime(
                                        trades["exit_time"],
                                        errors="coerce",
                                    )

                                if "pnl" in trades.columns:
                                    trades["cumulative_pnl"] = (
                                        pd.to_numeric(
                                            trades["pnl"],
                                            errors="coerce",
                                        ).fillna(0.0).cumsum()
                                    )

                                st.markdown("#### Equity / P&L Curve")
                                if not trades.empty and "exit_time" in trades.columns:
                                    equity_chart = trades[
                                        ["exit_time", "cumulative_pnl"]
                                    ].dropna()
                                    if not equity_chart.empty:
                                        st.line_chart(
                                            equity_chart.set_index("exit_time"),
                                            y="cumulative_pnl",
                                            width="stretch",
                                            height=300,
                                        )

                            if not result.rule_performance.empty:
                                st.markdown("#### Rule-by-Rule Accuracy")
                                rule_view = result.rule_performance.copy()
                                rule_view = rule_view.rename(
                                    columns={
                                        "win_rate_pct": "accuracy_pct",
                                    }
                                )
                                st.dataframe(
                                    rule_view,
                                    use_container_width=True,
                                    hide_index=True,
                                )

                            if not trades.empty:
                                st.markdown("#### Historical Trades")
                                trade_columns = [
                                    column
                                    for column in [
                                        "entry_time",
                                        "exit_time",
                                        "direction",
                                        "entry",
                                        "exit_price",
                                        "stop_loss",
                                        "target",
                                        "quantity",
                                        "rule",
                                        "pnl",
                                        "roi_pct",
                                        "reason",
                                    ]
                                    if column in trades.columns
                                ]
                                st.dataframe(
                                    trades[trade_columns],
                                    use_container_width=True,
                                    hide_index=True,
                                )
                            else:
                                st.warning(
                                    "No trades were generated in this period. "
                                    "That is not the same as 0% accuracy."
                                )

                            st.caption(
                                "Accuracy here means the percentage of closed simulated trades "
                                "that ended profitable. It is not ML prediction accuracy. "
                                "Use a sufficiently large, out-of-sample historical sample before "
                                "drawing conclusions about strategy quality."
                            )
                        except Exception as exc:
                            st.error(f"Historical backtest failed: {exc}")

        except Exception as exc:
            st.error(f"Backtest data preparation failed: {exc}")


st.divider()


# ============================================================
# Safety
# ============================================================

if settings.paper_trading:

    st.warning(
        "PAPER TRADING MODE - Live trading is disabled."
    )

else:

    st.info(
        "Paper trading is disabled, but live execution "
        "still requires explicit production configuration."
    )

st.write(
    f"Environment: **{environment}**"
)

st.write(
    f"Live trading allowed: "
    f"**{settings.live_trading_allowed()}**"
)
