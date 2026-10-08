import random
import sys
from pathlib import Path
from types import SimpleNamespace

# Streamlit executes this file with app/ as the script directory. Add the
# repository root so top-level packages (strategy, broker, marketdata, etc.)
# resolve the same way they do under pytest and normal Python execution.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from strategy.market_specs import get_option_strike_step, round_to_strike
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
from backtest.capital_aware import CapitalAwareRuleBacktestEngine
from backtest.signal_research import run_signal_research
from features.technical.indicators import add_indicators
from strategy.rules import StrategyConfig, StrategySignal, _generate_signal_from_enriched, evaluate_rules
from marketdata.option_chain_csv import (
    is_option_chain_snapshot, parse_option_chain_csv,
    is_nse_option_chain_export, parse_nse_option_chain_export,
)
from features.option_signal_engine import generate_option_chain_signal
from strategy.buy_today_sell_tomorrow import run_buy_today_sell_tomorrow
from strategy.btst_option_selector import rank_btst_options
from strategy.closing_session import evaluate_closing_session
from analysis.data_quality import assess_ohlcv
from prediction.nifty_315_340 import predict_315_340, evaluate_next_day_accuracy
from prediction.nifty_model import walk_forward_predict
from news.global_news import fetch_global_news
from marketdata.nifty_csv import normalize_nifty_csv
from marketdata.option_chain_replay import replay_option_chain_csv
from strategy.cross_market_trend import calculate_trend, TrendSnapshot, aggregate_context
from marketdata.yahoo_finance import fetch_yahoo_ohlcv
from marketdata.yahoo_window import fetch_yahoo_rolling_window
from ml.engine import MLConfig, train as train_ml, predict as predict_ml
from ensemble.signal import build_ensemble
from marketdata.decision_source import load_kotak_decision_snapshot
from assistant.chatbot import answer as chatbot_answer
from alerts.recipient_store import WhatsAppRecipientStore
from alerts.runtime_settings import WhatsAppSettingsStore
from alerts.whatsapp import WhatsAppAlertService
from analytics.signal_journal import SignalJournal


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

    research_base = {
        "NIFTY": 25040.0,
        "BANKNIFTY": 58000.0,
        "CRUDEOIL": 6500.0,
        "NATURALGAS": 300.0,
        "COPPER": 950.0,
        "SILVER": 95000.0,
        "GOLD": 125000.0,
    }.get(instrument.upper(), 25040.0)
    research_strike = round_to_strike(
        research_base,
        get_strike_step(instrument),
    )

    ce = OptionAnalysis(
        option_type="CE",
        strike=research_strike,
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
        strike=research_strike,
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

    atm_strike = round_to_strike(spot, strike_step)

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

def get_strike_step(instrument: str) -> float:
    """Compatibility wrapper around the centralized market specification."""
    return get_option_strike_step(instrument)


# ============================================================
# Streamlit
# ============================================================

st.set_page_config(
    page_title="AI Derivatives Terminal",
    page_icon="AI",
    layout="wide",
)

# Modern terminal styling: information-dense, high-contrast and status-oriented.
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #07111f; }
[data-testid="stHeader"] { background: rgba(7,17,31,0.85); }
.block-container { padding-top: 1.2rem; max-width: 1500px; }
[data-testid="stMetric"] {
  background: linear-gradient(135deg, rgba(18,35,58,.96), rgba(10,22,38,.96));
  border: 1px solid rgba(91,151,255,.22);
  border-radius: 14px;
  padding: 12px 14px;
  box-shadow: 0 8px 24px rgba(0,0,0,.18);
}
.market-radar { border:1px solid rgba(91,151,255,.22); border-radius:16px; padding:14px; background:linear-gradient(135deg,#0c1b2f,#091525); }
.radar-title { font-size:1.05rem; font-weight:700; margin-bottom:8px; }
.radar-pill { display:inline-block; padding:7px 11px; margin:3px; border-radius:999px; font-weight:700; font-size:.82rem; }
.radar-up { background:#063b2a; color:#54e39a; }
.radar-down { background:#45171d; color:#ff7785; }
.radar-range { background:#403512; color:#ffd86b; }
.radar-na { background:#263244; color:#b9c5d6; }
.st-key-dashboard_chat { position: fixed; right: 24px; bottom: 24px; z-index: 9999; }
.st-key-dashboard_chat button { border-radius: 999px; box-shadow: 0 10px 28px rgba(0,0,0,.35); }
</style>
""", unsafe_allow_html=True)

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
        "LIVE",
    ],
)

if environment == "LIVE" and not settings.live_trading_allowed():
    st.sidebar.warning(
        "LIVE selected, but live order submission is LOCKED. "
        "Set LIVE_TRADING_ENABLED=true, PAPER_TRADING=false and ALLOW_ORDER_SUBMISSION=true "
        "after completing broker/risk/UAT validation. LIVE never bypasses data-quality or strategy gates."
    )

seed = st.sidebar.number_input(
    "Research Seed",
    min_value=1,
    max_value=1_000_000,
    value=42,
    step=1,
)

# Research compatibility placeholders. The active dashboard decision path
# below is always driven by Kotak/persisted real data; synthetic research
# selectors are not used for PAPER/UAT/LIVE decisions.
context = SimpleNamespace(
    trend="UNAVAILABLE",
    momentum="UNAVAILABLE",
    price_vs_vwap="UNAVAILABLE",
    volatility_regime="UNAVAILABLE",
)
ce = pe = None
signal = SimpleNamespace(ce_score=0.0, pe_score=0.0, edge=0.0, decision="WAIT")
ml_probability = None



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
    width="stretch",
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
        # Authentication changes the production data source state. Any
        # previous RED/WAIT snapshot may have been generated while Neo was
        # disconnected, so it must never survive a new authentication attempt.
        st.session_state["kotak_decision_snapshot"] = None
        st.session_state.pop("last_signal_journal_key", None)
        if connection.connected:
            st.sidebar.success(connection.message)
        else:
            st.sidebar.error(connection.message)

neo_status = neo_broker.connection_status()
st.sidebar.caption(
    "Kotak Neo: " + ("CONNECTED" if neo_status.connected else "NOT CONNECTED")
)


st.sidebar.divider()
st.sidebar.subheader("WhatsApp Alerts")
st.sidebar.caption("Manage alert recipients here. Numbers are stored locally and do not require .env edits.")

recipient_store = WhatsAppRecipientStore()
whatsapp_runtime = WhatsAppSettingsStore()
wa_service = WhatsAppAlertService(recipient_store)
wa_recipients = recipient_store.load()
wa_enabled = whatsapp_runtime.enabled() or bool(getattr(settings, "whatsapp_alerts_enabled", False))

with st.sidebar.expander("Recipients", expanded=True):

    if wa_recipients:
        for recipient in wa_recipients:
            row = st.columns([4, 1])
            row[0].caption(f"+{recipient}")
            if row[1].button("Remove", key=f"wa_remove_{recipient}"):
                recipient_store.remove(recipient)
                st.rerun()
    else:
        st.info("No recipients configured.")

    with st.form("whatsapp_recipient_form", clear_on_submit=True):
        new_recipient = st.text_input(
            "Add mobile number",
            placeholder="+91 9876543210",
            help="Use international format. Spaces, brackets and a leading + are accepted.",
        )
        add_recipient = st.form_submit_button("Add recipient", width="stretch")
        if add_recipient:
            try:
                saved = recipient_store.add(new_recipient)
                st.success(f"Recipient +{recipient_store.normalize(new_recipient)} added.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    if wa_recipients:
        st.caption(f"{len(wa_recipients)} recipient(s) saved locally.")

    enable_whatsapp = st.toggle(
        "Enable WhatsApp sending",
        value=wa_enabled,
        key="whatsapp_dashboard_enabled",
        help="This controls alert sending from the dashboard. API credentials remain in .env.",
    )
    if enable_whatsapp != wa_enabled:
        whatsapp_runtime.set_enabled(enable_whatsapp)
        wa_enabled = enable_whatsapp
        st.rerun()

    if not wa_recipients:
        st.warning("Add at least one recipient before sending alerts.")
    elif not wa_service.token or not wa_service.phone_number_id:
        st.warning("WhatsApp API credentials are missing. Add the Meta access token and phone number ID to .env once; recipients remain dashboard-managed.")
    elif wa_enabled:
        st.success("WhatsApp sending is enabled.")
    else:
        st.info("WhatsApp sending is disabled. Turn on the toggle above to enable alerts.")


# ============================================================
# Global news
# ============================================================

news_snapshot = fetch_global_news()
global_news_score = news_snapshot.sentiment

# ============================================================
# Production decision source: Kotak Neo
# ============================================================

st.subheader("Decision Data")
signal_journal = SignalJournal()

st.caption(
    "KOTAK NEO is the primary production/paper decision source. "
    "Yahoo Finance is kept below as an independent historical research/validation source only. "
    "The live decision path never silently falls back to Yahoo."
)

decision_cols = st.columns(4)
decision_timeframe = decision_cols[0].selectbox(
    "Decision timeframe", ["5m", "15m"], index=0, key="neo_decision_timeframe"
)
decision_history_days = decision_cols[1].number_input(
    "Kotak history days", min_value=3, max_value=30, value=10, step=1,
    key="neo_decision_history_days"
)
decision_refresh = decision_cols[2].button(
    "Refresh Kotak decision", type="primary", width="stretch", key="neo_decision_refresh"
)
decision_cols[3].metric(
    "Decision source",
    "KOTAK NEO",
)

if "kotak_decision_snapshot" not in st.session_state:
    st.session_state["kotak_decision_snapshot"] = None

cached_snapshot = st.session_state["kotak_decision_snapshot"]
snapshot_connection_mismatch = (
    cached_snapshot is not None
    and (
        (cached_snapshot.status == "RED" and neo_status.connected)
        or (cached_snapshot.status != "RED" and not neo_status.connected)
    )
)

if decision_refresh or cached_snapshot is None or snapshot_connection_mismatch:
    snapshot = load_kotak_decision_snapshot(
        neo_broker,
        instrument=instrument,
        timeframe=decision_timeframe,
        history_days=int(decision_history_days),
        config=StrategyConfig(require_option_confirmation=True),
    )
    st.session_state["kotak_decision_snapshot"] = snapshot
else:
    snapshot = cached_snapshot

prediction_frame = snapshot.frame if snapshot is not None else None
prediction_source = "Kotak Neo production decision data" if snapshot and snapshot.frame is not None else None
canonical_signal = snapshot.signal if snapshot is not None else StrategySignal(
    "WAIT", (), ("Kotak Neo decision snapshot is not available",),
    0.0, 0.0, 0.0, 0.0, False
)
prediction_quality = SimpleNamespace(
    status=snapshot.quality_status if snapshot else "RED",
    reasons=list(snapshot.quality_reasons) if snapshot else ["Kotak Neo decision snapshot unavailable"],
)

dcols = st.columns(5)
dcols[0].metric("Source", "KOTAK NEO")
dcols[1].metric("Mode", snapshot.mode if snapshot else "LIVE")
dcols[2].metric("Quality", prediction_quality.status)
dcols[3].metric("Rule", canonical_signal.direction)
dcols[4].metric("Options", f"{snapshot.option_count} contracts" if snapshot else "0")

if snapshot:
    st.caption(
        f"Last decision snapshot: {snapshot.timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')} · "
        f"{snapshot.message}"
    )
    if prediction_quality.reasons:
        st.caption("Data-quality notes: " + " | ".join(prediction_quality.reasons))

if canonical_signal.valid:
    st.success("Kotak Neo data passed the canonical precision rule gate.")
else:
    st.warning(canonical_signal.reasons[0] if canonical_signal.reasons else "WAIT — no qualified production signal.")

# ------------------------------------------------------------
# Yahoo research / validation — explicitly non-production
# ------------------------------------------------------------

with st.expander("Yahoo historical research / independent validation", expanded=False):
    st.caption(
        "Yahoo is intentionally isolated from the production decision path. "
        "Use it to compare historical behavior, train/research models, or detect data discrepancies."
    )
    ycols = st.columns(4)
    research_interval = ycols[0].selectbox(
        "Research interval", ["5m", "15m", "30m", "60m", "1d"], index=0,
        key="research_yahoo_interval"
    )
    research_days = ycols[1].number_input(
        "Research days", min_value=5, max_value=90, value=30, step=5,
        key="research_yahoo_days"
    )
    research_symbol = ycols[2].text_input(
        "Yahoo symbol", value="^NSEI", key="research_yahoo_symbol"
    )
    research_pull = ycols[3].button(
        "Load research data", key="research_yahoo_pull", width="stretch"
    )

    if research_pull:
        try:
            research_end = pd.Timestamp.now(tz="Asia/Kolkata").date()
            research_start = research_end - pd.Timedelta(days=int(research_days))
            result = fetch_yahoo_rolling_window(
                research_start, research_end,
                symbol=research_symbol.strip() or "^NSEI",
                interval=research_interval,
            )
            st.session_state["yahoo_research_result"] = result
        except Exception as exc:
            st.session_state["yahoo_research_result"] = None
            st.error(f"Yahoo research load failed: {exc}")

    yahoo_result = st.session_state.get("yahoo_research_result")
    if yahoo_result is not None and yahoo_result.status == "OK":
        st.success(
            f"Yahoo research loaded: {len(yahoo_result.data):,} rows · "
            f"{yahoo_result.provider_start} → {yahoo_result.provider_end}. "
            "Research only."
        )
    elif yahoo_result is not None:
        st.warning(f"Yahoo research unavailable: {yahoo_result.message}")
    else:
        st.info("No Yahoo research dataset loaded.")

# ============================================================
# Decision Center
# ============================================================

st.markdown("## Decision Center")
st.caption(
    "Kotak Neo → data quality → technical/options rules → regime → ML advisory → ensemble. "
    "No AI/chat layer can override the trading gates."
)

if prediction_frame is not None and snapshot is not None and prediction_quality.status != "RED":
    ml_input = prediction_frame.copy()
    if snapshot.pcr_oi is not None:
        ml_input["PCR"] = snapshot.pcr_oi
        ml_input["PCR_OI"] = snapshot.pcr_oi
    if snapshot.pcr_volume is not None:
        ml_input["PCR_VOLUME"] = snapshot.pcr_volume
    ml_input["SENTIMENT"] = float(global_news_score) if global_news_score is not None else np.nan
    ml_input["SENTIMENT_CHANGE"] = ml_input["SENTIMENT"].diff()
    ml_input["NEWS_COUNT"] = np.nan

    coverage_rows = [
        ["Technical", "6/6 available", "READY"],
        ["Derivatives", f"{1 if snapshot.pcr_oi is not None else 0}/5 available", "PARTIAL" if snapshot.pcr_oi is not None else "BLOCKED"],
        ["Sentiment", "2/3 available", "PARTIAL"],
    ]
    with st.expander("Model input coverage", expanded=False):
        st.dataframe(
            pd.DataFrame(coverage_rows, columns=["Feature group", "Coverage", "Status"]),
            width="stretch", hide_index=True
        )
        st.caption(
            "Only real Kotak/news values are supplied. Missing derivative fields are not synthesized."
        )

    # Current global news is a point-in-time value. It must never be copied
    # across historical rows and used as a training feature, because that would
    # leak present information into the past. Keep it for the latest prediction
    # only; historical ML training uses only features actually present per row.
    ml_training_input = ml_input.copy()
    for _leaky_column in ("SENTIMENT", "SENTIMENT_CHANGE", "NEWS_COUNT"):
        if _leaky_column in ml_training_input.columns:
            ml_training_input[_leaky_column] = np.nan

    ml_context_key = (
        instrument.upper(),
        decision_timeframe,
        str(snapshot.timestamp) if snapshot is not None else "",
    )
    if st.session_state.get("ml_context_key") != ml_context_key:
        st.session_state.pop("final_ml_result", None)
        st.session_state.pop("final_ml_artifacts", None)
        st.session_state["ml_context_key"] = ml_context_key

    train_col, status_col = st.columns([1, 3])
    train_clicked = train_col.button(
        "Train / Retrain ML", type="primary", width="stretch", key="final_ml_train"
    )
    if train_clicked:
        try:
            with st.spinner("Training advisory models on the available Kotak decision dataset..."):
                ml_result, ml_artifacts = train_ml(
                    ml_training_input,
                    MLConfig(window_days=90, refresh_minutes=5),
                    model_dir=f"models/ml_advisory/{instrument.upper()}_{decision_timeframe}",
                )
            st.session_state["final_ml_result"] = ml_result
            st.session_state["final_ml_artifacts"] = ml_artifacts
            st.success(
                "ML advisory models trained. Historical current-news values were excluded "
                "from training to prevent temporal leakage."
            )
        except Exception as exc:
            st.error(f"ML training failed: {exc}")

    ml_result = st.session_state.get("final_ml_result")
    ml_artifacts = st.session_state.get("final_ml_artifacts")
    if ml_result:
        status_col.caption(
            f"Model {ml_result.model_version} · {ml_result.rows:,} rows · "
            f"{ml_result.window_start} → {ml_result.window_end}"
        )
        train_table = pd.DataFrame([
            {
                "Horizon": f"{h}m",
                "Train": f"{m['training_accuracy']*100:.1f}%",
                "Validation": f"{m['validation_accuracy']*100:.1f}%",
                "Balanced": f"{m['validation_balanced_accuracy']*100:.1f}%",
                "Time-series CV": f"{m['cv_accuracy']*100:.1f}%",
                "Samples": m["validation_samples"],
                "Accuracy CI": f"{m['confidence_interval_pct'][0]:.1f}%–{m['confidence_interval_pct'][1]:.1f}%",
            }
            for h, m in ml_result.horizons.items()
        ])
        st.dataframe(train_table, width="stretch", hide_index=True)

    ml_predictions = []
    if ml_artifacts:
        try:
            ml_predictions = predict_ml(ml_input, ml_artifacts, MLConfig())
        except Exception as exc:
            st.warning(f"ML prediction unavailable: {exc}")

    if ml_predictions:
        try:
            micro = None
            micro_required = {"bid_size", "ask_size"}
            if micro_required.issubset(set(ml_input.columns)):
                from features.microstructure import microstructure_features, AdaptiveMicroWeight
                latest = ml_input.iloc[-1]
                depth = (
                    pd.to_numeric(ml_input["bid_size"], errors="coerce").fillna(0)
                    + pd.to_numeric(ml_input["ask_size"], errors="coerce").fillna(0)
                ) * pd.to_numeric(ml_input["close"], errors="coerce").fillna(0)
                last_week = depth.tail(min(len(depth), 7 * 78))
                liquidity_threshold = float(last_week.quantile(0.95)) if len(last_week) >= 20 else float("inf")
                tick_atr_14 = pd.to_numeric(ml_input.get("ATR_TICK_14", pd.Series(dtype=float)), errors="coerce")
                tick_atr_20 = pd.to_numeric(ml_input.get("ATR_TICK_20", pd.Series(dtype=float)), errors="coerce")
                if not tick_atr_14.empty and not tick_atr_20.empty and tick_atr_14.notna().any() and tick_atr_20.notna().any():
                    volatility_band = float(tick_atr_14.iloc[-1] / max(float(tick_atr_20.iloc[-1]), 1e-9))
                    ticks = ml_input.get("LAST_TRADE_PRICE", pd.Series(dtype=float)).dropna().tail(50).tolist()
                    micro = microstructure_features(
                        bid_volume=float(latest.get("bid_size", 0) or 0),
                        ask_volume=float(latest.get("ask_size", 0) or 0),
                        liquidity_threshold=liquidity_threshold,
                        imbalance_history=ml_input.get("IMBALANCE_RATIO", pd.Series(dtype=float)).dropna().tail(20).tolist(),
                        last_trade_price=float(latest.get("LAST_TRADE_PRICE", latest.get("close", 0)) or 0),
                        last_trade_direction=latest.get("LAST_TRADE_DIRECTION"),
                        best_bid=float(latest.get("best_bid", 0) or 0),
                        best_ask=float(latest.get("best_ask", 0) or 0),
                        last_ticks=ticks,
                        spread_width=float(latest.get("spread", 0) or 0),
                        volatility_band=volatility_band,
                    )
                    micro["micro_weight"] = AdaptiveMicroWeight().load()
            regime, ensemble_rows = build_ensemble(
                ml_input,
                ml_predictions,
                StrategyConfig(require_option_confirmation=(instrument.upper() == "NIFTY")),
                micro=micro,
            )
            rcols = st.columns(4)
            rcols[0].metric("Market Regime", regime.name)
            rcols[1].metric("Rule Weight", f"{regime.rule_weight*100:.0f}%")
            rcols[2].metric("ML Weight", f"{regime.ml_weight*100:.0f}%")
            rcols[3].caption(regime.reason)

            ensemble_table = pd.DataFrame([
                {
                    "Horizon": f"{x.horizon_minutes} min",
                    "CE Reliability": f"{x.final_ce:.1f}%",
                    "PE Reliability": f"{x.final_pe:.1f}%",
                    "Stronger Side": "🟢 CE" if x.stronger_side == "CE" else "🔴 PE" if x.stronger_side == "PE" else "⚪ WAIT",
                    "Validation CI": f"{x.confidence_low:.1f}%–{x.confidence_high:.1f}%",
                }
                for x in ensemble_rows
            ])
            st.dataframe(ensemble_table, width="stretch", hide_index=True)

            first = ensemble_rows[0]
            if first.final_ce < 55 and first.final_pe < 55:
                st.warning("Weak signal: neither side reaches the 55% reliability threshold.")
            elif abs(first.final_ce - first.final_pe) < 10:
                st.warning("Marginal edge: CE/PE reliability separation is below 10 points.")
            else:
                stronger = "CE" if first.final_ce > first.final_pe else "PE"
                st.success(
                    f"Stronger side: {stronger} · 5-minute ensemble reliability "
                    f"is {max(first.final_ce, first.final_pe):.1f}%."
                )
            st.caption(
                "ML is advisory only. A trade still requires the canonical rule, "
                "data-quality, EV, risk, and execution gates."
            )
            # Streamlit reruns are frequent. Journal a Kotak snapshot only once,
            # while still allowing every genuinely new 5m/15m decision to be stored.
            snapshot_key = str(snapshot.timestamp) if snapshot is not None else ""
            last_journal_key = st.session_state.get("last_signal_journal_key")
            if snapshot_key and snapshot_key != last_journal_key:
                signal_journal.append_decision(
                    instrument=instrument,
                    timeframe=decision_timeframe,
                    source="KOTAK_NEO",
                    status=str(snapshot.status),
                    direction=str(canonical_signal.direction if canonical_signal.valid else "WAIT"),
                    confidence=max(float(first.final_ce), float(first.final_pe)) if canonical_signal.valid else 0.0,
                    reliability=max(float(first.final_ce), float(first.final_pe)) if canonical_signal.valid else 0.0,
                    reason="; ".join(canonical_signal.reasons),
                    regime=regime.name,
                    pcr=snapshot.pcr_oi,
                    imbalance_ratio=(micro or {}).get("imbalance_ratio") if isinstance(micro, dict) else None,
                    aggressor=(micro or {}).get("aggressor", "") if isinstance(micro, dict) else "",
                    micro_weight=(micro or {}).get("micro_weight") if isinstance(micro, dict) else None,
                    threshold=(micro or {}).get("imbalance_trigger") if isinstance(micro, dict) else None,
                    ml_ce=float(first.ml_ce),
                    ml_pe=float(first.ml_pe),
                    rule_ce=float(first.rule_ce),
                    rule_pe=float(first.rule_pe),
                    entry=float(prediction_frame.iloc[-1]["close"]) if prediction_frame is not None and canonical_signal.valid else None,
                    stop_loss=float(canonical_signal.stop_loss) if canonical_signal.valid and canonical_signal.stop_loss > 0 else None,
                    target=float(canonical_signal.target) if canonical_signal.valid and canonical_signal.target > 0 else None,
                )
                st.session_state["last_signal_journal_key"] = snapshot_key
        except Exception as exc:
            st.warning(f"Ensemble calculation unavailable: {exc}")
    else:
        st.info("Train the ML layer to activate ensemble reliability. Rule Engine remains independent.")
else:
    st.info(
        "Kotak Neo production data is not decision-ready. The system remains WAIT; "
        "Yahoo cannot be used as a silent live fallback."
    )

with st.expander("Signal Journal — weekly review", expanded=False):
    journal_path = "logs/signal_journal.jsonl"
    if __import__("pathlib").Path(journal_path).exists():
        try:
            journal_rows = [
                __import__("json").loads(line)
                for line in __import__("pathlib").Path(journal_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            journal_df = pd.DataFrame(journal_rows)
            st.caption(f"Persistent decision journal: {len(journal_df):,} records")
            if not journal_df.empty:
                st.dataframe(journal_df.tail(200), width="stretch", hide_index=True)
                st.download_button(
                    "Download signal journal",
                    data=__import__("pathlib").Path(journal_path).read_bytes(),
                    file_name="signal_journal.jsonl",
                    mime="application/json",
                    width="stretch",
                )
        except Exception as exc:
            st.warning(f"Signal journal could not be read: {exc}")
    else:
        st.info("No signal journal records yet. The first evaluated Kotak Neo decision will create it.")

with st.expander("Trading system health & safety gates", expanded=False):
    health = pd.DataFrame([
        ["Data source", "KOTAK NEO", "Primary production/paper decision source"],
        ["Data quality", prediction_quality.status, "RED blocks production decisions"],
        ["Rule Engine", canonical_signal.direction, "Deterministic gate"],
        ["ML", "TRAINED" if st.session_state.get("final_ml_artifacts") else "NOT TRAINED", "Advisory only"],
        ["Risk", "SEPARATE", "Runtime equity/risk fraction; no hardcoded capital"],
        ["Execution", "LOCKED" if not settings.live_trading_allowed() else "ENABLED", "Separate broker safety gate"],
        ["Yahoo", "RESEARCH ONLY", "No production fallback"],
    ], columns=["Layer", "Status", "Safety"])
    st.dataframe(health, width="stretch", hide_index=True)

# ============================================================
# Dashboard ChatGPT assistant
# ============================================================

chat_anchor = st.columns([8, 2])
with chat_anchor[1]:
    with st.popover("💬 Ask ChatGPT", type="primary", width=420, key="dashboard_chat"):
        st.markdown("### AI Terminal Assistant")
        st.caption(
            "Ask about the dashboard, rules, ML, backtests, option chain, "
            "risk gates, or why the current decision is WAIT."
        )
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        for message in st.session_state["chat_history"][-8:]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.text_input(
            "Question",
            key="dashboard_chat_question",
            placeholder="Why is the system WAIT?",
        )
        send = st.button("Send", type="primary", key="dashboard_chat_send")

        if send and question.strip():
            context = {
                "instrument": instrument,
                "environment": environment,
                "data_source": "KOTAK_NEO",
                "data_mode": snapshot.mode if snapshot else "LIVE",
                "data_status": snapshot.status if snapshot else "RED",
                "data_timestamp": str(snapshot.timestamp) if snapshot else None,
                "quality_status": prediction_quality.status,
                "quality_reasons": list(prediction_quality.reasons),
                "rule_signal": canonical_signal.direction,
                "rule_confidence": canonical_signal.confidence,
                "option_count": snapshot.option_count if snapshot else 0,
                "pcr_oi": snapshot.pcr_oi if snapshot else None,
                "pcr_volume": snapshot.pcr_volume if snapshot else None,
                "regime": regime.name if "regime" in locals() else "NOT AVAILABLE",
                "rule_weight": regime.rule_weight if "regime" in locals() else None,
                "ml_weight": regime.ml_weight if "regime" in locals() else None,
                "ensemble_ce": ensemble_rows[0].final_ce if "ensemble_rows" in locals() and ensemble_rows else None,
                "ensemble_pe": ensemble_rows[0].final_pe if "ensemble_rows" in locals() and ensemble_rows else None,
                "stronger_side": ensemble_rows[0].stronger_side if "ensemble_rows" in locals() and ensemble_rows else None,
                "no_trade_reason": canonical_signal.reasons[0] if not canonical_signal.valid and canonical_signal.reasons else None,
                "historical_research_source": "Yahoo Finance (research only)",
            }
            with st.spinner("ChatGPT is answering..."):
                reply = chatbot_answer(
                    question.strip(),
                    history=st.session_state["chat_history"],
                    dashboard_context=context,
                )
            st.session_state["chat_history"].append({"role": "user", "content": question.strip()})
            st.session_state["chat_history"].append({"role": "assistant", "content": reply})
            st.rerun()

# ============================================================
# Research spot
# ============================================================


# ============================================================

# Live underlying price when Kotak Neo is connected.
# Research-mode synthetic spot is retained only when the broker is not connected.
spot = None
if neo_status.connected:
    try:
        provider = KotakNeoProvider(neo_broker.client)
        if instrument == "NIFTY":
            spot = provider.get_index_quote("Nifty 50").ltp
        elif instrument in {"CRUDEOIL", "NATURALGAS", "COPPER", "SILVER", "GOLD"}:
            # A current MCX futures quote is safe for display/context only. It
            # must not be mistaken for the completed 5m candle history required
            # by the canonical decision engine.
            contract = provider.resolve_mcx_futures(instrument)
            quote = provider.get_mcx_quote(contract)
            spot = float(quote["ltp"])
        elif prediction_frame is not None and not prediction_frame.empty:
            spot = float(prediction_frame.iloc[-1]["close"])
        else:
            st.info(
                f"Live {instrument} underlying is unavailable because the canonical "
                "Kotak Neo decision frame is not ready."
            )
    except Exception as exc:
        st.error(f"Live underlying quote request failed: {exc}")

if spot is None and environment == "RESEARCH":
    from marketdata.daily_store import load_captured_candles
    research_frame, _ = load_captured_candles(instrument.upper())
    if not research_frame.empty:
        latest_research = research_frame.sort_values("timestamp").iloc[-1]
        spot = float(latest_research["close"])
        st.caption(
            "Research spot: latest persisted KOTAK_CAPTURED close at "
            f"{latest_research['timestamp']}."
        )

if spot is None:
    st.warning(
        f"{environment} underlying price is unavailable for {instrument}. "
        "No synthetic price is used outside RESEARCH."
    )


# ============================================================
# Cross-market trend radar
# ============================================================
def _real_radar_snapshot(name: str, connected: bool) -> TrendSnapshot:
    from marketdata.daily_store import load_captured_candles
    frame, _source = load_captured_candles(name)
    if frame.empty:
        return TrendSnapshot(
            name, "DATA_UNAVAILABLE", 0.0, 0.0, "UNKNOWN", "UNKNOWN", "UNKNOWN",
            None, 0, False,
            "No real captured Kotak candles are available; synthetic radar data is disabled.",
        )
    return calculate_trend(name, frame, is_live=connected)

st.subheader("Market Radar")
st.caption("Directional context for NIFTY, Crude Oil, Natural Gas and Copper. Research cards are explicitly marked when a live feed is unavailable.")
radar_names = ["NIFTY", "CRUDE", "NATGAS", "COPPER"]
radar_cols = st.columns(4)
radar_snapshots: dict[str, TrendSnapshot] = {}
for name, col in zip(radar_names, radar_cols):
    snap = _real_radar_snapshot(name, neo_status.connected)
    radar_snapshots[name] = snap
    if snap.direction in {"STRONG_UP", "UP"}:
        cls = "radar-up"
    elif snap.direction in {"STRONG_DOWN", "DOWN"}:
        cls = "radar-down"
    elif snap.direction == "RANGE":
        cls = "radar-range"
    else:
        cls = "radar-na"
    source = "LIVE" if snap.is_live else "RESEARCH"
    col.markdown(f'<div class="market-radar"><div class="radar-title">{name}</div><span class="radar-pill {cls}">{snap.direction}</span><br><small>{source} · score {snap.score:.0f} · {snap.volatility}</small><br><small>{snap.reason}</small></div>', unsafe_allow_html=True)

radar_context = aggregate_context(radar_snapshots)
st.caption(f"Cross-market context score: {radar_context:+.1f}. Only real captured Kotak candles are eligible; unavailable markets contribute no score.")

# ============================================================
# System readiness dashboard
# ============================================================
from strategy.readiness import assess_readiness
from strategy.regime import classify_regime

st.subheader("System Readiness")
st.caption("Design-time gate dashboard. Automated test status must be confirmed by the repository-local UAT before merge.")
regime_snapshot = classify_regime(pd.Series({
    "ADX": 20.0, "ATR_PCT": 0.01, "EMA_SPREAD": 1.0, "VWAP_DEV": 0.0,
}), global_news_score)
readiness = assess_readiness(
    tests_passed=False,  # Repository-local UAT must certify this; never hardcode PASS.
    warmup_ready=bool(prediction_frame is not None and not prediction_frame.empty),
    risk_engine_ready=True,
    ml_available=bool(st.session_state.get("final_ml_artifacts")),
    live_order_enabled=settings.live_trading_allowed(),
    realistic_backtest_available=True,
    option_premium_history_available=bool(getattr(snapshot, "option_chain", ())) if snapshot is not None else False,
)
rc = st.columns(4)
rc[0].metric("Readiness", f"{readiness.score:.0f}/100")
rc[1].metric("Status", readiness.status)
rc[2].metric("Regime", regime_snapshot.regime)
rc[3].metric("Live Orders", "ENABLED" if settings.live_trading_allowed() else "LOCKED")
with st.expander("Readiness gates", expanded=False):
    st.dataframe(pd.DataFrame([{
        "Gate": g.name, "Passed": g.passed, "Priority": g.severity, "Detail": g.detail
    } for g in readiness.gates]), width="stretch", hide_index=True)

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
    f"{settings.risk_fraction:.2%} of equity",
)

col3.metric(
    "Daily Loss Limit",
    f"{settings.max_daily_loss_fraction:.2%} of equity",
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

c1, c2, c3, c4 = st.columns(4)

if canonical_signal.valid:
    display_prediction = canonical_signal.direction
    prediction_status = "READY"
else:
    display_prediction = "WAIT"
    prediction_status = "WAIT"

c1.metric("Prediction", display_prediction)
c2.metric("Confidence", f"{canonical_signal.confidence:.1f}%")
c3.metric("Status", prediction_status)
c4.metric("R:R", "—" if not canonical_signal.valid else f"{abs(canonical_signal.target - float(prediction_frame.iloc[-1]['close'])) / max(abs(float(prediction_frame.iloc[-1]['close']) - canonical_signal.stop_loss), 1e-9):.2f}")

if canonical_signal.valid:
    st.success(f"{canonical_signal.direction} signal passed the canonical precision gate.")
    pc = st.columns(3)
    pc[0].metric("Entry", f"{float(prediction_frame.iloc[-1]['close']):,.2f}")
    pc[1].metric("Stop Loss", f"{canonical_signal.stop_loss:,.2f}")
    pc[2].metric("Target", f"{canonical_signal.target:,.2f}")
else:
    st.warning(canonical_signal.reasons[0] if canonical_signal.reasons else "No qualified signal.")

if canonical_signal.rules:
    st.caption("Rules: " + ", ".join(canonical_signal.rules))

st.divider()


# ============================================================
# Market context
# ============================================================

if environment != "RESEARCH" and prediction_frame is not None and not prediction_frame.empty:
    latest = prediction_frame.iloc[-1]
    ema9 = float(latest.get("EMA9", np.nan))
    ema21 = float(latest.get("EMA21", np.nan))
    rsi = float(latest.get("RSI", np.nan))
    vwap_dev = float(latest.get("VWAP_DEV", np.nan))
    atr_pct = float(latest.get("ATR_PCT", np.nan))
    live_trend = (
        "BULLISH" if np.isfinite(ema9) and np.isfinite(ema21) and ema9 > ema21
        else "BEARISH" if np.isfinite(ema9) and np.isfinite(ema21) and ema9 < ema21
        else "NEUTRAL"
    )
    live_momentum = (
        "POSITIVE" if np.isfinite(rsi) and rsi >= 55
        else "NEGATIVE" if np.isfinite(rsi) and rsi <= 45
        else "NEUTRAL"
    )
    live_price_vs_vwap = (
        "ABOVE" if np.isfinite(vwap_dev) and vwap_dev > 0
        else "BELOW" if np.isfinite(vwap_dev) and vwap_dev < 0
        else "AT"
    )
    live_volatility = (
        "HIGH" if np.isfinite(atr_pct) and atr_pct >= 0.025
        else "NORMAL" if np.isfinite(atr_pct) and atr_pct >= 0.01
        else "LOW"
    )
    context = SimpleNamespace(
        trend=live_trend,
        momentum=live_momentum,
        price_vs_vwap=live_price_vs_vwap,
        volatility_regime=live_volatility,
    )
    st.caption("Market context is derived from the canonical Kotak Neo decision frame.")
elif environment != "RESEARCH":
    context = SimpleNamespace(
        trend="UNAVAILABLE",
        momentum="UNAVAILABLE",
        price_vs_vwap="UNAVAILABLE",
        volatility_regime="UNAVAILABLE",
    )
    st.caption("Live market context is unavailable because the canonical Kotak Neo decision frame is blocked.")
else:
    st.caption("Research context is synthetic and is available only in RESEARCH environment.")

st.subheader(
    "Market Context"
)

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Spot",
    "Unavailable" if spot is None else f"{spot:,.2f}",
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
# CE / WAIT / PE decision view
# ============================================================

# Keep broker candidate variables defined in every environment. The Streamlit
# module executes top-to-bottom during import, including RESEARCH/test runs.
best_ce = None
best_pe = None

if environment == "RESEARCH":
    st.subheader("Research CE / WAIT / PE")
    st.caption(
        "Research uses the latest persisted real Kotak option-chain snapshot. "
        "No synthetic selector is used. Trade direction is WAIT until real "
        "market context and the option-chain engine produce a qualified signal."
    )
    ce_score = 0.0
    pe_score = 0.0
    wait_edge = 0.0
    selected_side = "WAIT"
else:
    st.subheader(f"{instrument} CE / WAIT / PE")
    st.caption(
        "Real Kotak Neo CE/PE candidates remain visible in PAPER/UAT/LIVE. "
        "They are read-only: the canonical signal, data-quality, risk and "
        "execution gates control the decision."
    )
    broker_contracts = list(snapshot.option_chain) if snapshot is not None else []
    ce_contracts = [x for x in broker_contracts if getattr(x, "option_type", "") == "CE"]
    pe_contracts = [x for x in broker_contracts if getattr(x, "option_type", "") == "PE"]
    ce_analyses = [analyze_option(x) for x in ce_contracts]
    pe_analyses = [analyze_option(x) for x in pe_contracts]
    best_ce = max(ce_analyses, key=lambda x: float(x.score)) if ce_analyses else None
    best_pe = max(pe_analyses, key=lambda x: float(x.score)) if pe_analyses else None
    ce_score = float(best_ce.score) if best_ce is not None else 0.0
    pe_score = float(best_pe.score) if best_pe is not None else 0.0
    wait_edge = abs(ce_score - pe_score)
    selected_side = (
        "CE" if canonical_signal.direction == "CE"
        else "PE" if canonical_signal.direction == "PE"
        else "WAIT"
    )

left, middle, right = st.columns(3)
with left:
    st.subheader("CE")
    st.metric("Candidate score", f"{ce_score:.1f}")
    if selected_side == "CE":
        st.success("Canonical side")
    elif best_ce is not None:
        st.write(f"Broker candidate · {getattr(best_ce, 'strike', '—')}")
with middle:
    st.subheader("WAIT")
    st.metric("CE/PE edge", f"{wait_edge:.1f}")
    if selected_side == "WAIT":
        st.warning(f"Canonical decision: {canonical_signal.direction}")
    else:
        st.write("Available when evidence conflicts")
with right:
    st.subheader("PE")
    st.metric("Candidate score", f"{pe_score:.1f}")
    if selected_side == "PE":
        st.success("Canonical side")
    elif best_pe is not None:
        st.write(f"Broker candidate · {getattr(best_pe, 'strike', '—')}")

if environment != "RESEARCH":
    st.info(
        f"{environment} safety rule: CE/PE selection is read-only. "
        "No manual selector can override the canonical signal, data-quality gate, "
        "risk gate, or execution lock."
    )

st.divider()


# ============================================================
# Option chain
# ============================================================

authenticated = neo_status.connected
# Always initialize the displayed option-chain container before any environment
# branch. Streamlit executes this module during test collection, so a missing
# initialization here becomes an import-time NameError.
contracts: list[OptionContract] = []
live_contracts: list[OptionContract] = []

# The option-chain display is independent from the canonical trade decision:
# - OPEN market: poll Kotak now and display the current broker chain.
# - CLOSED market: do not poll; display the latest Kotak snapshot already held
#   by the decision source, explicitly labelled as the last snapshot.
# - Never replace a failed broker chain with Yahoo or synthetic data.
option_chain_config = {
    "NIFTY": ("nse_fo", "NIFTY", 40),
    "CRUDEOIL": ("mcx_fo", "CRUDEOIL", 40),
    "NATURALGAS": ("mcx_fo", "NATURALGAS", 40),
    "COPPER": ("mcx_fo", "COPPER", 40),
    "SILVER": ("mcx_fo", "SILVER", 40),
    "GOLD": ("mcx_fo", "GOLD", 40),
}
chain_config = option_chain_config.get(instrument)

chain_market_open = False
chain_status_label = "BROKER DATA UNAVAILABLE"
chain_status_detail = ""

if chain_config:
    chain_exchange, chain_underlying, chain_count = chain_config
    now = pd.Timestamp.now(tz="Asia/Kolkata")

    # NSE index options: 09:15-15:30 IST.
    # MCX commodity derivatives: 09:00-23:30 IST.
    if instrument == "NIFTY":
        chain_market_open = (
            now.weekday() < 5
            and pd.Timestamp("09:15").time() <= now.time() <= pd.Timestamp("15:30").time()
        )
    else:
        chain_market_open = (
            now.weekday() < 5
            and pd.Timestamp("09:00").time() <= now.time() <= pd.Timestamp("23:30").time()
        )

    snapshot_contracts = (
        list(snapshot.option_chain)
        if snapshot is not None and getattr(snapshot, "option_chain", None)
        else []
    )

    # Recover the newest persisted real Kotak chain as a UI failover cache.
    # This survives Streamlit reruns/restarts and is never treated as current
    # live data unless the broker refresh succeeds in this same run.
    cached_chain_timestamp = None
    try:
        from marketdata.daily_store import DailyMarketStore
        cached_chain_frame, cached_chain_timestamp = DailyMarketStore().load_latest_option_chain_snapshot(
            instrument,
            before=now,
        )
        if not cached_chain_frame.empty:
            cached_contracts = []
            for row in cached_chain_frame.to_dict("records"):
                try:
                    cached_contracts.append(
                        OptionContract(
                            symbol=str(row.get("symbol") or ""),
                            expiry=str(row.get("expiry") or ""),
                            strike=float(row.get("strike") or 0),
                            option_type=str(row.get("option_type") or "").upper(),
                            ltp=float(row.get("ltp") or 0),
                            bid=float(row["bid"]) if pd.notna(row.get("bid")) else None,
                            ask=float(row["ask"]) if pd.notna(row.get("ask")) else None,
                            volume=float(row.get("volume") or 0),
                            open_interest=float(row.get("open_interest") or 0),
                            oi_change=float(row.get("oi_change") or 0),
                            implied_volatility=float(row.get("implied_volatility") or 0),
                            built_up=str(row.get("built_up") or ""),
                            delta=float(row["delta"]) if pd.notna(row.get("delta")) else None,
                            theta=float(row["theta"]) if pd.notna(row.get("theta")) else None,
                            vega=float(row["vega"]) if pd.notna(row.get("vega")) else None,
                            gamma=float(row["gamma"]) if pd.notna(row.get("gamma")) else None,
                            ltp_change_pct=float(row.get("ltp_change_pct") or 0),
                        )
                    )
                except (TypeError, ValueError):
                    continue
            if cached_contracts:
                # Prefer the in-memory decision snapshot when it is newer;
                # otherwise recover the persisted real broker snapshot.
                if not snapshot_contracts or snapshot is None or cached_chain_timestamp > snapshot.timestamp:
                    snapshot_contracts = cached_contracts
    except Exception:
        # Cache recovery is deliberately best-effort. It must never make the
        # broker/live path fail or introduce synthetic data.
        cached_chain_timestamp = None

    if chain_market_open and authenticated:
        try:
            provider = KotakNeoProvider(neo_broker.client)
            live_contracts = provider.get_option_chain(
                underlying=chain_underlying,
                exchange=chain_exchange,
                count=chain_count,
                expiry=None,
                # Refresh broker quotes so displayed LTP is current,
                # not the option-chain snapshot's potentially stale quote.
                enrich_quotes=True,
            )
            if live_contracts:
                # Persist only a successful real broker refresh. This becomes
                # the fallback shown when the next refresh fails or the market
                # is closed; it is never used to claim a live refresh succeeded.
                persistence_error = None
                try:
                    from marketdata.daily_store import DailyMarketStore
                    DailyMarketStore().save_option_chain_snapshot(
                        instrument,
                        live_contracts,
                        captured_at=now,
                    )
                except Exception as exc:
                    # Never hide a persistence failure. The live chain can be
                    # displayed in-memory, but Research must not be told that
                    # this refresh was captured if the durable write failed.
                    persistence_error = str(exc)
                chain_status_label = "LIVE KOTAK NEO"
                if persistence_error:
                    chain_status_detail = (
                        f"{chain_status_detail} Persistence FAILED: {persistence_error}"
                    )
                chain_status_detail = (
                    f"Current broker chain · {len(live_contracts)} contracts · "
                    f"refreshed {now.strftime('%H:%M:%S IST')}"
                )
            else:
                chain_status_label = "LIVE REFRESH EMPTY"
                chain_status_detail = (
                    "Kotak returned no option contracts. The trade path remains WAIT."
                )
                live_contracts = snapshot_contracts
        except Exception as exc:
            # A transient refresh failure must not destroy a valid broker snapshot.
            # Never substitute synthetic/Yahoo data.
            live_contracts = snapshot_contracts
            chain_status_label = "LIVE REFRESH FAILED"
            chain_status_detail = (
                f"{exc}. Showing the last updated real Kotak snapshot."
                if snapshot_contracts
                else f"{exc}. No broker chain is available."
            )
    elif chain_market_open and not authenticated:
        live_contracts = snapshot_contracts
        chain_status_label = "KOTAK NOT CONNECTED"
        chain_status_detail = (
            "Connect to Kotak Neo to refresh the current option chain."
        )
    else:
        # Closed market: the chain is not considered "unavailable" merely
        # because polling is paused. Show the most recent real Kotak snapshot.
        live_contracts = snapshot_contracts
        chain_status_label = "MARKET CLOSED"
        chain_status_detail = (
            "Live polling paused. "
            + (
                f"Showing the last updated real Kotak snapshot from "
                f"{(cached_chain_timestamp or snapshot.timestamp).strftime('%Y-%m-%d %H:%M:%S %Z')}."
                if snapshot_contracts and (cached_chain_timestamp is not None or snapshot is not None)
                else "No Kotak option-chain snapshot is available yet."
            )
        )

if environment != "RESEARCH":
    if chain_status_label == "LIVE KOTAK NEO":
        st.success(f"{chain_status_label} · {chain_status_detail}")
    elif chain_status_label == "MARKET CLOSED":
        st.info(f"{chain_status_label} · {chain_status_detail}")
    elif live_contracts:
        st.warning(f"{chain_status_label} · {chain_status_detail}")
    else:
        st.warning(f"{chain_status_label} · {chain_status_detail}")

st.subheader("Option Chain")
st.caption(
    "Live Kotak Neo data is shown when connected. "
    "Synthetic option data is permitted only in RESEARCH environment; PAPER never substitutes a failed broker request."
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
        "Unavailable" if spot is None else f"{spot:,.2f}",
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
    contracts = []
    if environment == "RESEARCH":
        from marketdata.daily_store import DailyMarketStore
        research_frame, captured_at = DailyMarketStore().load_latest_option_chain_snapshot(
            instrument.upper()
        )
        for row in research_frame.to_dict("records"):
            try:
                contracts.append(
                    OptionContract(
                        symbol=str(row.get("symbol") or ""),
                        expiry=str(row.get("expiry") or ""),
                        strike=float(row["strike"]),
                        option_type=str(row.get("option_type") or "").upper(),
                        ltp=float(row.get("ltp") or 0),
                        bid=float(row["bid"]) if pd.notna(row.get("bid")) else None,
                        ask=float(row["ask"]) if pd.notna(row.get("ask")) else None,
                        volume=float(row.get("volume") or 0),
                        open_interest=float(row.get("open_interest") or 0),
                        oi_change=float(row.get("oi_change") or 0),
                        implied_volatility=float(row.get("implied_volatility") or 0),
                        built_up=str(row.get("built_up") or ""),
                        delta=float(row["delta"]) if pd.notna(row.get("delta")) else None,
                        theta=float(row["theta"]) if pd.notna(row.get("theta")) else None,
                        vega=float(row["vega"]) if pd.notna(row.get("vega")) else None,
                        gamma=float(row["gamma"]) if pd.notna(row.get("gamma")) else None,
                        ltp_change_pct=float(row.get("ltp_change_pct") or 0),
                    )
                )
            except (TypeError, ValueError):
                continue
        if contracts:
            stamp = captured_at.strftime("%Y-%m-%d %H:%M:%S %Z") if captured_at is not None else "timestamp unavailable"
            st.success(
                f"RESEARCH DATA — latest persisted real KOTAK_CAPTURED option chain "
                f"({len(contracts)} contracts, captured {stamp})."
            )
        else:
            st.warning(
                "RESEARCH DATA UNAVAILABLE — no real captured option chain exists. "
                "Synthetic option-chain data is disabled."
            )
    else:
        st.warning(
            f"No live {instrument} option-chain data is available. "
            f"{environment} mode will not substitute synthetic contracts."
        )

chain_is_live = bool(live_contracts)
if contracts and spot is not None:
    chain_df = build_option_chain_dataframe(
        contracts=contracts,
        spot=spot,
    )
else:
    chain_df = pd.DataFrame(columns=[
        "Strike", "CE LTP", "CE Volume", "CE OI", "CE OI Chg", "CE IV", "CE Score",
        "PE LTP", "PE Volume", "PE OI", "PE OI Chg", "PE IV", "PE Score", "ATM",
    ])

# Only the current broker refresh is tradable in PAPER/UAT/LIVE.
# A stale persisted snapshot may be displayed, but it can never generate a
# fresh trade plan while the market is open.
tradable_contracts = contracts
if environment != "RESEARCH" and chain_market_open:
    tradable_contracts = (
        live_contracts
        if chain_status_label == "LIVE KOTAK NEO" and live_contracts
        else []
    )

if tradable_contracts and spot is not None:
    direction_hint = None
    if environment != "RESEARCH" and canonical_signal.valid:
        direction_hint = (
            "CE" if canonical_signal.direction == "BUY"
            else "PE" if canonical_signal.direction == "SELL"
            else "WAIT"
        )
    chain_signal, chain_signal_rows = generate_option_chain_signal(
        tradable_contracts,
        spot=spot,
        global_news_score=global_news_score,
        direction_hint=direction_hint,
        require_two_sided_quote=(environment != "RESEARCH"),
        require_verified_broker_quote=(environment != "RESEARCH"),
    )
else:
    chain_signal, chain_signal_rows = None, []

if chain_signal is not None:
    st.markdown("#### Option-chain signal levels")
    signal_cols = st.columns(6)
    signal_cols[0].metric(
        "Signal",
        chain_signal.direction if chain_is_live else "RESEARCH",
    )
    signal_cols[1].metric(
        "Confidence",
        f"{chain_signal.confidence:.1f}%" if chain_is_live else "N/A",
    )
    signal_cols[2].metric(
        "Entry",
        f"Rs {chain_signal.entry_price:.2f}"
        if chain_is_live and chain_signal.entry_price is not None else "Unavailable",
    )
    signal_cols[3].metric(
        "Stop Loss",
        f"Rs {chain_signal.stop_loss:.2f}"
        if chain_is_live and chain_signal.stop_loss is not None else "Unavailable",
    )
    signal_cols[4].metric(
        "Take Profit",
        f"Rs {chain_signal.take_profit:.2f}"
        if chain_is_live and chain_signal.take_profit is not None else "Unavailable",
    )
    signal_cols[5].metric("Global News", f"{global_news_score:+.2f}")


# ------------------------------------------------------------
# 2 ATM + 5 OTM selection
# ------------------------------------------------------------

if strike_view == "2 ATM + 5 OTM" and spot is not None and not chain_df.empty:

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


display_df = chain_df[visible_columns].copy()

# Trade-plan columns: every visible option gets a concrete plan when an actual
# option LTP exists; otherwise premium fields remain unavailable rather than
# being fabricated from LTP-change percentages.
# Normalize the option trade-plan schema before rendering.  Older signal
# frames used Max Gain % / Max Loss %, while the current engine exposes
# Target Gain % / Stop Risk %.  Missing premium-derived values must remain
# unavailable rather than being fabricated.
# Signal rows can be returned as a DataFrame by the research/live
# generator or as a plain list of dictionaries. Normalize once at the UI
# boundary so an empty/blocked PAPER chain can never crash the dashboard.
if isinstance(chain_signal_rows, pd.DataFrame):
    plan = chain_signal_rows.copy()
elif isinstance(chain_signal_rows, list):
    plan = pd.DataFrame(chain_signal_rows)
elif chain_signal_rows is None:
    plan = pd.DataFrame()
else:
    try:
        plan = pd.DataFrame(chain_signal_rows)
    except (TypeError, ValueError):
        plan = pd.DataFrame()

for column in [
    "Side", "Strike", "Signal", "Confidence",
    "Entry Price", "Stop Loss", "Take Profit",
]:
    if column not in plan.columns:
        plan[column] = np.nan

if "Target Gain %" not in plan.columns:
    plan["Target Gain %"] = np.nan
if "Stop Risk %" not in plan.columns:
    plan["Stop Risk %"] = np.nan

plan["Target Gain %"] = pd.to_numeric(plan["Target Gain %"], errors="coerce")
plan["Stop Risk %"] = pd.to_numeric(plan["Stop Risk %"], errors="coerce")

# Canonicalize the join key before *any* merge. Neo option payloads,
# research generators and older CSV/Yahoo artifacts can represent Strike as
# strings, ints or floats. Pandas refuses an object/float merge and silently
# coercing only one side is unsafe.
def _normalize_strike_column(frame: pd.DataFrame, *, required: bool = False) -> pd.DataFrame:
    frame = frame.copy()
    if "Strike" not in frame.columns:
        if required:
            raise ValueError("Option-chain data is missing required Strike column.")
        return frame
    frame["Strike"] = pd.to_numeric(frame["Strike"], errors="coerce")
    frame = frame.dropna(subset=["Strike"]).copy()
    frame["Strike"] = frame["Strike"].astype("float64")
    return frame

display_df = _normalize_strike_column(display_df, required=True)
plan = _normalize_strike_column(plan, required=True)

# Remove duplicate join keys from the plan. One CE/PE trade-plan row per
# strike is the UI contract; duplicates otherwise create Cartesian expansion.
plan = plan.drop_duplicates(subset=["Side", "Strike"], keep="last").reset_index(drop=True)

# Backward-compatible display aliases for any downstream UI/test code that
# still expects the previous names.
plan["Max Gain %"] = plan["Target Gain %"]
plan["Max Loss %"] = plan["Stop Risk %"]
display_df = display_df.merge(plan, on=["Side", "Strike"], how="left") if "Side" in display_df.columns else display_df
ce_plan = plan[plan["Side"]=="CE"].rename(columns={
    "Signal":"CE Signal","Confidence":"CE Confidence","Entry Price":"CE Entry",
    "Stop Loss":"CE SL","Take Profit":"CE TP","Max Gain %":"CE Max Gain %",
    "Max Loss %":"CE Max Loss %"
}).drop(columns=["Side"])
pe_plan = plan[plan["Side"]=="PE"].rename(columns={
    "Signal":"PE Signal","Confidence":"PE Confidence","Entry Price":"PE Entry",
    "Stop Loss":"PE SL","Take Profit":"PE TP","Max Gain %":"PE Max Gain %",
    "Max Loss %":"PE Max Loss %"
}).drop(columns=["Side"])

# Attach the trade plan to every view, including CE-only and PE-only views.
# Premium levels remain blank when actual option LTP is unavailable.
display_df = display_df.drop(
    columns=[
        c for c in [
            "Signal","Confidence","Entry Price","Stop Loss",
            "Take Profit","Max Gain %","Max Loss %",
        ] if c in display_df.columns
    ],
    errors="ignore",
)
if chain_side == "CE":
    display_df = display_df.merge(ce_plan, on="Strike", how="left")
elif chain_side == "PE":
    display_df = display_df.merge(pe_plan, on="Strike", how="left")
else:
    display_df = display_df.merge(ce_plan, on="Strike", how="left").merge(
        pe_plan, on="Strike", how="left"
    )


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


if environment == "RESEARCH" and contracts:
    st.caption("RESEARCH DATA — latest persisted real KOTAK_CAPTURED option chain. Synthetic option-chain data is disabled.")

st.markdown("#### Option Chain — Trade Plan")
st.caption("Entry / SL / TP and Max Gain are premium-based only when actual option LTP is available. Snapshot files containing only LTP-change % will show unavailable premium levels.")
st.dataframe(
    display_df.style.apply(
        highlight_atm,
        axis=1,
    ),
    width="stretch",
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
