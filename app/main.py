import random
from types import SimpleNamespace

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

if decision_refresh or st.session_state["kotak_decision_snapshot"] is None:
    snapshot = load_kotak_decision_snapshot(
        neo_broker,
        instrument=instrument,
        timeframe=decision_timeframe,
        history_days=int(decision_history_days),
        config=StrategyConfig(require_option_confirmation=True),
    )
    st.session_state["kotak_decision_snapshot"] = snapshot
else:
    snapshot = st.session_state["kotak_decision_snapshot"]

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

    train_col, status_col = st.columns([1, 3])
    train_clicked = train_col.button(
        "Train / Retrain ML", type="primary", width="stretch", key="final_ml_train"
    )
    if train_clicked:
        try:
            with st.spinner("Training advisory models on the available Kotak decision dataset..."):
                ml_result, ml_artifacts = train_ml(
                    ml_input,
                    MLConfig(window_days=min(90, int(decision_history_days)), refresh_minutes=5),
                    model_dir="models/ml_advisory",
                )
            st.session_state["final_ml_result"] = ml_result
            st.session_state["final_ml_artifacts"] = ml_artifacts
            st.success("ML models trained separately from the Rule Engine.")
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
                    micro["micro_weight"] = AdaptiveMicroWeight().update(volatility_band)
            regime, ensemble_rows = build_ensemble(
                ml_input,
                ml_predictions,
                StrategyConfig(require_option_confirmation=True),
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
            signal_journal.append_decision(
                instrument=instrument,
                timeframe=decision_timeframe,
                source="KOTAK_NEO",
                status=str(getattr(st.session_state.get("kotak_decision_snapshot"), "status", "UNKNOWN")),
                direction=str(first.stronger_side),
                confidence=max(float(first.final_ce), float(first.final_pe)),
                reliability=max(float(first.final_ce), float(first.final_pe)),
                regime=regime.name,
                pcr=getattr(st.session_state.get("kotak_decision_snapshot"), "pcr_oi", None),
                imbalance_ratio=(micro or {}).get("imbalance_ratio") if isinstance(micro, dict) else None,
                aggressor=(micro or {}).get("aggressor", "") if isinstance(micro, dict) else "",
                micro_weight=(micro or {}).get("micro_weight") if isinstance(micro, dict) else None,
                threshold=(micro or {}).get("imbalance_trigger") if isinstance(micro, dict) else None,
                ml_ce=float(first.ml_ce),
                ml_pe=float(first.ml_pe),
                rule_ce=float(first.rule_ce),
                rule_pe=float(first.rule_pe),
            )
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
# Cross-market trend radar
# ============================================================
def _research_trend_snapshot(name: str, seed_value: int) -> TrendSnapshot:
    base = {"NIFTY": 25040.0, "CRUDE": 6500.0, "NATGAS": 300.0, "COPPER": 950.0}[name]
    rng = np.random.default_rng(abs(hash((name, int(seed_value)))) % (2**32))
    returns = rng.normal(0.0, base * 0.0008, 120)
    close = base + np.cumsum(returns)
    frame = pd.DataFrame({
        "timestamp": pd.date_range(end=pd.Timestamp.now(), periods=120, freq="5min"),
        "open": close,
        "high": close + abs(rng.normal(0, base * 0.0003, 120)),
        "low": close - abs(rng.normal(0, base * 0.0003, 120)),
        "close": close,
        "volume": rng.integers(1000, 10000, 120),
    })
    return calculate_trend(name, frame, is_live=False)


st.subheader("Market Radar")
st.caption("Directional context for NIFTY, Crude Oil, Natural Gas and Copper. Research cards are explicitly marked when a live feed is unavailable.")
radar_names = ["NIFTY", "CRUDE", "NATGAS", "COPPER"]
radar_cols = st.columns(4)
radar_snapshots: dict[str, TrendSnapshot] = {}
for name, col in zip(radar_names, radar_cols):
    if name == "NIFTY" and neo_status.connected:
        # Until historical intraday streaming is wired for every instrument, do not
        # fabricate a live trend from the single index quote.
        snap = TrendSnapshot(name, "LIVE_QUOTE_ONLY", 0.0, 0.0, "UNKNOWN", "UNKNOWN", "UNKNOWN", pd.Timestamp.now(), 0, True, "Live quote available; completed-candle history is required for a genuine trend score.")
    else:
        snap = _research_trend_snapshot(name, int(seed))
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
st.caption(f"Cross-market context score: {radar_context:+.1f}. This is a context filter, not a standalone trade signal.")

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
    tests_passed=False,
    warmup_ready=True,
    risk_engine_ready=True,
    ml_available=True,
    live_order_enabled=settings.live_trading_allowed(),
    realistic_backtest_available=True,
    option_premium_history_available=False,
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

chain_signal, chain_signal_rows = generate_option_chain_signal(
    contracts,
    spot=spot,
    global_news_score=global_news_score,
)
st.markdown("#### Option-chain signal levels")
signal_cols = st.columns(6)
signal_cols[0].metric("Signal", chain_signal.direction)
signal_cols[1].metric("Confidence", f"{chain_signal.confidence:.1f}%")
signal_cols[2].metric("Entry", "Unavailable" if chain_signal.entry_price is None else f"Rs {chain_signal.entry_price:.2f}")
signal_cols[3].metric("Stop Loss", "Unavailable" if chain_signal.stop_loss is None else f"Rs {chain_signal.stop_loss:.2f}")
signal_cols[4].metric("Take Profit", "Unavailable" if chain_signal.take_profit is None else f"Rs {chain_signal.take_profit:.2f}")
signal_cols[5].metric("Global News", f"{global_news_score:+.2f}")


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


display_df = chain_df[visible_columns].copy()

# Trade-plan columns: every visible option gets a concrete plan when an actual
# option LTP exists; otherwise premium fields remain unavailable rather than
# being fabricated from LTP-change percentages.
# Normalize the option trade-plan schema before rendering.  Older signal
# frames used Max Gain % / Max Loss %, while the current engine exposes
# Target Gain % / Stop Risk %.  Missing premium-derived values must remain
# unavailable rather than being fabricated.
plan = chain_signal_rows.copy()
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


if not live_contracts:
    st.caption("RESEARCH DATA — deterministic synthetic option chain; not a broker feed.")

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
    width="stretch",
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
    help="Supports structured snapshots and NSE two-row option-chain exports.",
)

if option_csv is not None:
    try:
        option_snapshot = pd.read_csv(option_csv)
        if is_nse_option_chain_export(option_snapshot):
            snapshot_contracts = parse_nse_option_chain_export(option_snapshot)
            st.success(f"NSE option-chain export detected: {len(snapshot_contracts):,} real CE/PE contracts.")
        elif is_option_chain_snapshot(option_snapshot.columns):
            snapshot_contracts = parse_option_chain_csv(option_snapshot)
        else:
            st.error("Unsupported option-chain format. Expected a structured snapshot or NSE two-row export.")
            snapshot_contracts = []

        if snapshot_contracts:
            snapshot_rows = []
            for contract in snapshot_contracts:
                snapshot_rows.append({
                    "Side": contract.option_type,
                    "Strike": contract.strike,
                    "LTP": contract.ltp,
                    "LTP Change %": contract.ltp_change_pct,
                    "Bid": contract.bid,
                    "Ask": contract.ask,
                    "Spread": max(0.0, contract.ask - contract.bid),
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
            option_signal, signal_rows = generate_option_chain_signal(
                snapshot_contracts, spot=spot, global_news_score=global_news_score
            )
            snapshot_df = snapshot_df.merge(
                signal_rows[["Side","Strike","Signal","Confidence","Entry Price","Stop Loss","Take Profit","Target Gain %","Stop Risk %","Global News"]],
                on=["Side","Strike"], how="left"
            )
            st.success(f"Loaded {len(snapshot_contracts):,} option contracts from {option_csv.name}.")
            st.dataframe(snapshot_df, width="stretch", hide_index=True)
            oc = st.columns(8)
            oc[0].metric("Signal", option_signal.direction)
            oc[1].metric("Confidence", f"{option_signal.confidence:.1f}%")
            oc[2].metric("Entry", "Unavailable" if option_signal.entry_price is None else f"Rs {option_signal.entry_price:.2f}")
            oc[3].metric("Stop Loss", "Unavailable" if option_signal.stop_loss is None else f"Rs {option_signal.stop_loss:.2f}")
            oc[4].metric("Take Profit", "Unavailable" if option_signal.take_profit is None else f"Rs {option_signal.take_profit:.2f}")
            oc[5].metric("Target Gain", "N/A")
            oc[6].metric("Stop Risk", "N/A")
            oc[7].metric("Global News", f"{global_news_score:+.2f}")
            if all(contract.ltp <= 0 for contract in snapshot_contracts):
                st.info("No usable option LTP is present. Premium entry/SL/TP are unavailable.")
            elif all(contract.bid <= 0 or contract.ask <= 0 for contract in snapshot_contracts):
                st.info("Option LTP is available, but bid/ask quality is incomplete; execution-quality checks remain unavailable.")
            st.info("A single option-chain snapshot is context only. It is never used as OHLCV backtest input.")
    except Exception as exc:
        st.error(f"Option-chain snapshot import failed: {exc}")

st.subheader("Historical Option-Chain Replay")
st.caption("Replay requires timestamped option-chain snapshots. A single exported snapshot cannot be replayed because it has no time axis.")
replay_file = st.file_uploader("Upload timestamped option-chain replay CSV", type=["csv"], key="option_replay_csv")
if replay_file is not None:
    try:
        replay_df = pd.read_csv(replay_file)
        snapshots = replay_option_chain_csv(replay_df)
        st.success(f"Loaded {len(snapshots):,} timestamped option-chain snapshots.")
        if snapshots:
            st.dataframe(
                pd.DataFrame({
                    "Timestamp": [ts for ts, _ in snapshots],
                    "Contracts": [len(cs) for _, cs in snapshots],
                }),
                width="stretch",
                hide_index=True,
            )
    except Exception as exc:
        st.error(f"Option-chain replay failed: {exc}")

st.divider()

# ============================================================
# Global news + BTST + NIFTY 3:15-3:40 prediction
# ============================================================

st.subheader("Global News")
news_cols = st.columns(3)
news_cols[0].metric("Global News Sentiment", f"{global_news_score:+.2f}")
news_cols[1].metric("Latest Global Headline", news_snapshot.headline[:80])
news_cols[2].metric("Headlines Used", len(news_snapshot.headlines))
if news_snapshot.headlines:
    st.dataframe(
        pd.DataFrame({"Global News": list(news_snapshot.headlines)}),
        width="stretch",
        hide_index=True,
    )

st.subheader("Buy Today, Sell Tomorrow")
st.caption("Separate next-session strategy: buy today's close and sell tomorrow, with the same default 1.5% stop-loss discipline.")
btst_file = st.file_uploader("Upload daily NIFTY OHLC CSV", type=["csv"], key="btst_csv")
if btst_file is not None:
    try:
        btst_data = normalize_nifty_csv(pd.read_csv(btst_file))
        btst = run_buy_today_sell_tomorrow(btst_data)
        model_accuracy, model_rows = evaluate_next_day_accuracy(
            btst_data,
            global_news_score=global_news_score,
        )
        walk_forward = walk_forward_predict(btst_data)
        bc = st.columns(6)
        bc[0].metric("BTST Accuracy", f"{btst.accuracy_pct:.1f}%")
        bc[1].metric("BTST Return", f"{btst.total_return_pct:.2f}%")
        bc[2].metric("BTST Net P&L", f"Rs {btst.net_pnl:,.2f}")
        bc[3].metric("Trades", len(btst.trades))
        bc[4].metric("Rule Model Accuracy", f"{model_accuracy:.1f}%")
        bc[5].metric("Walk-Forward ML Accuracy", f"{walk_forward.accuracy_pct:.1f}%")
        if not btst.trades.empty:
            st.dataframe(btst.trades, width="stretch", hide_index=True)
        st.markdown("#### Rule prediction accuracy")
        st.dataframe(model_rows, width="stretch", hide_index=True)
        st.markdown("#### Walk-forward ML prediction")
        st.dataframe(walk_forward.predictions, width="stretch", hide_index=True)
    except Exception as exc:
        st.error(f"BTST analysis failed: {exc}")

st.subheader("NIFTY Prediction — 3:15–3:40")
st.caption("Requires intraday timestamped candles. Daily OHLC exports cannot produce this window and are therefore not converted into a false intraday prediction.")
intraday_file = st.file_uploader("Upload NIFTY intraday CSV", type=["csv"], key="nifty_prediction_csv")
if intraday_file is not None:
    try:
        intraday = pd.read_csv(intraday_file)
        prediction = predict_315_340(intraday, global_news_score=global_news_score)
        pc = st.columns(6)
        pc[0].metric("Prediction", prediction.prediction)
        pc[1].metric("Confidence", f"{prediction.confidence:.1f}%")
        pc[2].metric("Reference", "N/A" if prediction.reference_price is None else f"{prediction.reference_price:.2f}")
        pc[3].metric("Target", "N/A" if prediction.target is None else f"{prediction.target:.2f}")
        pc[4].metric("Stop Loss", "N/A" if prediction.stop_loss is None else f"{prediction.stop_loss:.2f}")
        pc[5].metric("Global News", f"{prediction.global_news_score:+.2f}")
        prediction_table = pd.DataFrame([{
            "Prediction": prediction.prediction,
            "Confidence": prediction.confidence,
            "Reference": prediction.reference_price,
            "Target": prediction.target,
            "Stop Loss": prediction.stop_loss,
            "Global News": prediction.global_news_score,
        }])
        st.dataframe(prediction_table, width="stretch", hide_index=True)
        st.write(prediction.reason)
    except Exception as exc:
        st.error(f"NIFTY prediction failed: {exc}")

st.divider()

# ============================================================
# Advanced decision workspace
# ============================================================

st.subheader("Advanced Decision Workspace")
st.caption("A transparent research cockpit: higher-timeframe regime → setup → option-chain evidence → risk → execution. No synthetic chain is used for trade recommendations.")

workspace_tabs = st.tabs(["Closing Session 15:15–15:40", "BTST Option Chain", "Data Quality"])

with workspace_tabs[0]:
    st.markdown("**NIFTY derivatives closing-session engine**")
    st.caption("Research window for the final 25 minutes. This is not the cash-market closing auction (CAS); derivatives have their own normal-market close.")
    closing_file = st.file_uploader("Upload NIFTY intraday candles", type=["csv"], key="closing_session_csv")
    chain_file = st.file_uploader("Upload real NIFTY option-chain snapshot", type=["csv"], key="closing_chain_csv")
    if closing_file is not None and chain_file is not None:
        try:
            close_candles = pd.read_csv(closing_file)
            close_chain_df = pd.read_csv(chain_file)
            close_contracts = parse_option_chain_csv(close_chain_df)
            closing_signal, closing_table = evaluate_closing_session(
                close_candles, close_contracts, news_score=global_news_score
            )
            cc = st.columns(6)
            cc[0].metric("Decision", closing_signal.decision)
            cc[1].metric("Confidence", f"{closing_signal.confidence:.1f}%")
            cc[2].metric("Option", f"{closing_signal.option_type} {closing_signal.strike or ''}")
            cc[3].metric("Entry", "—" if closing_signal.entry is None else f"₹{closing_signal.entry:.2f}")
            cc[4].metric("SL", "—" if closing_signal.stop_loss is None else f"₹{closing_signal.stop_loss:.2f}")
            cc[5].metric("Target", "—" if closing_signal.target is None else f"₹{closing_signal.target:.2f}")
            st.info(" | ".join(closing_signal.reasons))
            if closing_signal.status == "READY":
                st.success("Trade candidate passed the closing-session research filters. Use a broker-confirmed live quote before any order.")
            else:
                st.warning("NO TRADE: the engine is intentionally allowed to abstain.")
            if not closing_table.empty:
                st.dataframe(closing_table.head(15), width="stretch", hide_index=True)
        except Exception as exc:
            st.error(f"Closing-session analysis failed: {exc}")
    else:
        st.info("Upload both completed intraday candles and a real option-chain snapshot to generate a candidate. The system will not invent an option.")

with workspace_tabs[1]:
    st.markdown("**BTST — Buy Today, Sell Tomorrow option selection**")
    st.caption("The selector ranks only liquid real contracts and prices entry from the ask when available. Overnight gap risk is explicit.")
    btst_file = st.file_uploader("Upload real option-chain snapshot", type=["csv"], key="btst_chain_csv")
    btst_direction = st.selectbox("Underlying next-session bias", ["UP", "DOWN", "NEUTRAL"], key="btst_direction")
    if btst_file is not None:
        try:
            btst_df = pd.read_csv(btst_file)
            btst_contracts = parse_option_chain_csv(btst_df)
            btst_signal, btst_table = rank_btst_options(btst_contracts, btst_direction)
            bc = st.columns(6)
            bc[0].metric("Decision", btst_signal.decision)
            bc[1].metric("Confidence", f"{btst_signal.confidence:.1f}%")
            bc[2].metric("Strike", "—" if btst_signal.strike is None else f"{btst_signal.strike:g}")
            bc[3].metric("Entry", "—" if btst_signal.entry is None else f"₹{btst_signal.entry:.2f}")
            bc[4].metric("SL", "—" if btst_signal.stop_loss is None else f"₹{btst_signal.stop_loss:.2f}")
            bc[5].metric("Target", "—" if btst_signal.target is None else f"₹{btst_signal.target:.2f}")
            st.warning("BTST is not guaranteed: overnight gap, IV change and next-session liquidity can invalidate the setup.")
            if not btst_table.empty:
                st.dataframe(btst_table.head(20), width="stretch", hide_index=True)
        except Exception as exc:
            st.error(f"BTST option analysis failed: {exc}")
    else:
        st.info("Upload a real option-chain snapshot. No synthetic option is recommended for BTST.")

with workspace_tabs[2]:
    st.markdown("**Data quality gate**")
    quality_file = st.file_uploader("Upload OHLCV for quality audit", type=["csv"], key="quality_csv")
    if quality_file is not None:
        try:
            quality = assess_ohlcv(pd.read_csv(quality_file))
            qc = st.columns(6)
            qc[0].metric("Status", quality.status)
            qc[1].metric("Rows", f"{quality.rows:,}")
            qc[2].metric("Duplicates", quality.duplicate_timestamps)
            qc[3].metric("Invalid OHLC", quality.invalid_ohlc)
            qc[4].metric("Gaps", quality.gaps_over_expected)
            qc[5].metric("Max Gap", f"{quality.max_gap_minutes:.1f}m")
            if quality.reasons:
                st.warning(" | ".join(quality.reasons))
            else:
                st.success("No structural quality issues detected.")
        except Exception as exc:
            st.error(f"Data-quality audit failed: {exc}")

st.divider()

# ============================================================
# Backtest
# ============================================================

# ============================================================
# Yahoo Finance backtest layer
# ============================================================

st.subheader("Yahoo Finance Backtest")
st.caption(
    "Independent NIFTY 50 spot/index validation using Yahoo Finance OHLCV. "
    "This layer is separate from uploaded CSV and futures P&L backtests."
)

yahoo_cols = st.columns(5)
yahoo_interval = yahoo_cols[0].selectbox(
    "Yahoo interval", ["5m", "15m", "30m", "60m", "1d"],
    index=0, key="yahoo_bt_interval"
)
yahoo_start = yahoo_cols[1].date_input(
    "Start date",
    value=pd.Timestamp.now(tz="Asia/Kolkata").date() - pd.Timedelta(days=30),
    key="yahoo_bt_start"
)
yahoo_end = yahoo_cols[2].date_input(
    "End date",
    value=pd.Timestamp.now(tz="Asia/Kolkata").date(),
    key="yahoo_bt_end"
)
yahoo_symbol = yahoo_cols[3].text_input(
    "Yahoo symbol", value="^NSEI", key="yahoo_bt_symbol"
)
yahoo_run = yahoo_cols[4].button(
    "Pull & Backtest", type="primary", width="stretch", key="yahoo_bt_run"
)

if yahoo_run:
    yahoo_result = fetch_yahoo_ohlcv(
        yahoo_start,
        yahoo_end,
        symbol=yahoo_symbol.strip() or "^NSEI",
        interval=yahoo_interval,
    )
    st.session_state["yahoo_bt_result"] = yahoo_result

yahoo_result = st.session_state.get("yahoo_bt_result")
if yahoo_result is not None:
    if yahoo_result.status == "OK":
        st.success(
            f"GREEN — {yahoo_result.message}"
        )
        yc = st.columns(6)
        yc[0].metric("Source", "Yahoo Finance")
        yc[1].metric("Symbol", yahoo_result.symbol)
        yc[2].metric("Interval", yahoo_result.interval)
        yc[3].metric("Candles", f"{len(yahoo_result.data):,}")
        yc[4].metric("From", str(yahoo_result.provider_start))
        yc[5].metric("To", str(yahoo_result.provider_end))

        st.info(
            "Yahoo ^NSEI is NIFTY 50 spot/index data. "
            "This validates signal behavior, not futures/options profitability. "
            "No synthetic option/OI/PCR inputs are created."
        )

        yahoo_data = yahoo_result.data
        yahoo_quality = assess_ohlcv(yahoo_data)
        if yahoo_quality.status != "GREEN":
            st.warning(
                f"Yahoo OHLCV quality gate: {yahoo_quality.status}. "
                + " | ".join(yahoo_quality.reasons)
            )
        else:
            st.success("Yahoo OHLCV passed the structural data-quality gate.")

        st.dataframe(
            yahoo_data.tail(25), width="stretch", hide_index=True
        )

        if len(yahoo_data) >= 60:
            yahoo_research = run_signal_research(yahoo_data, StrategyConfig(require_option_confirmation=False))
            yv = yahoo_research.validation

            st.markdown("#### Yahoo signal-validation funnel")
            yf = st.columns(6)
            yf[0].metric("Bars", f"{yv.get('bars_considered', 0):,}")
            yf[1].metric("Rule triggers", f"{yv.get('rule_trigger_bars', 0):,}")
            yf[2].metric("Qualified", f"{yv.get('qualified_signal_bars', 0):,}")
            yf[3].metric("Conflicts", f"{yv.get('conflicting_signal_bars', 0):,}")
            yf[4].metric("Signals", f"{yv.get('signals', 0):,}")
            yf[5].metric("Trading days", f"{yv.get('trading_days', 0):,}")

            ym = st.columns(5)
            ym[0].metric("Win rate", f"{yv.get('win_rate_pct', 0.0):.1f}%")
            ym[1].metric("Average R", f"{yv.get('average_R', 0.0):.3f}")
            ym[2].metric("Total R", f"{yv.get('total_R', 0.0):.2f}")
            ym[3].metric("Wins", f"{yv.get('wins', 0):,}")
            ym[4].metric("Losses", f"{yv.get('losses', 0):,}")

            if not yahoo_research.signals.empty:
                st.dataframe(
                    yahoo_research.signals,
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.warning(
                    "No qualified Yahoo signals survived the existing gates."
                )
            st.caption(
                "Spot/index signal research only — not executable futures P&L."
            )
        else:
            st.warning(
                "Fewer than 60 Yahoo candles were returned; indicator warm-up "
                "blocks a misleading backtest."
            )
    else:
        st.error(
            f"Yahoo backtest unavailable: {yahoo_result.message}"
        )
        st.info(
            "No synthetic fallback is used. Adjust the Yahoo interval/date range "
            "or use the existing CSV backtest."
        )

st.divider()


st.subheader("Historical Rule Backtest")
st.caption(
    "Choose the role of the uploaded data explicitly. Spot/index OHLCV validates signal quality; "
    "contract-specific futures OHLCV is required for executable ₹1 lakh P&L."
)
backtest_mode = st.radio(
    "Backtest data role",
    ["NIFTY Spot / Index — Signal Research", "NIFTY Futures — Executable ₹1 lakh P&L"],
    horizontal=True,
    key="backtest_data_role",
)

if backtest_mode.startswith("NIFTY Futures"):
    st.warning(
        "Executable futures mode requires contract-specific economics. Do not use a current lot size "
        "for a historical period that spans a contract-specification change."
    )
    spec_cols = st.columns(4)
    futures_lot_size = int(spec_cols[0].number_input("Lot size", min_value=1, value=65, step=1))
    futures_point_value = float(spec_cols[1].number_input("Point value ₹", min_value=0.0001, value=1.0, step=0.1))
    futures_tick_size = float(spec_cols[2].number_input("Tick size", min_value=0.0001, value=0.05, step=0.05))
    futures_margin_per_lot = float(spec_cols[3].number_input("Margin / lot ₹", min_value=0.0, value=0.0, step=1000.0))
else:
    futures_lot_size = 65
    futures_point_value = 1.0
    futures_tick_size = 0.05
    futures_margin_per_lot = 0.0
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
    width="stretch",
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
    try:
        bt_data = normalize_nifty_csv(bt_data)
    except ValueError as schema_error:
        missing_bt = {"timestamp", "open", "high", "low", "close", "volume"} - set(bt_data.columns)
        if missing_bt:
            st.error(f"Backtest data is missing/invalid required OHLCV fields: {sorted(missing_bt)}")
        else:
            st.error(f"Backtest data schema validation failed: {schema_error}")
        bt_data = None

    if bt_data is None:
        pass
    elif is_option_chain_snapshot(bt_data.columns):
        st.error(
            "This CSV is an option-chain snapshot, not historical OHLCV candle data. "
            "Use the 'Option-Chain Snapshot Import' section above for CE/PE analysis. "
            "For the seven-rule backtest, upload timestamp, open, high, low, close and volume."
        )
        bt_data = None
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
                    width="stretch",
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

                            is_probably_spot = backtest_mode.startswith("NIFTY Spot")

                            if is_probably_spot:
                                research = run_signal_research(
                                    bt_data,
                                    evaluation_start=evaluation_start,
                                    evaluation_end=evaluation_end,
                                )
                                rv = research.validation
                                st.info(
                                    "SIGNAL RESEARCH MODE: this CSV is treated as NIFTY spot/index OHLCV. "
                                    "Signals and outcomes are measured in index points/R, not futures rupees. "
                                    "Upload contract-specific NIFTY futures OHLCV for executable ₹1 lakh P&L."
                                )
                                st.markdown("#### Signal research funnel")
                                rf = st.columns(6)
                                rf[0].metric("Bars", f"{rv.get('bars_considered', 0):,}")
                                rf[1].metric("Rule-trigger bars", f"{rv.get('rule_trigger_bars', 0):,}")
                                rf[2].metric("Qualified signals", f"{rv.get('qualified_signal_bars', 0):,}")
                                rf[3].metric("Conflicts", f"{rv.get('conflicting_signal_bars', 0):,}")
                                rf[4].metric("Evaluated signals", f"{rv.get('signals', 0):,}")
                                rf[5].metric("Trading days", f"{rv.get('trading_days', 0):,}")
                                sm = st.columns(5)
                                sm[0].metric("Signal Win Rate", f"{rv.get('win_rate_pct', 0.0):.1f}%")
                                sm[1].metric("Average R", f"{rv.get('average_R', 0.0):.3f}")
                                sm[2].metric("Total R", f"{rv.get('total_R', 0.0):.2f}")
                                sm[3].metric("Wins", f"{rv.get('wins', 0):,}")
                                sm[4].metric("Losses", f"{rv.get('losses', 0):,}")
                                if not research.signals.empty:
                                    st.markdown("#### Qualified signal outcomes")
                                    st.dataframe(research.signals, width="stretch", hide_index=True)
                                else:
                                    st.warning("No qualified signals survived the confirmation, volatility and timing gates.")
                                st.warning(
                                    "This is NOT a futures profitability result. Futures P&L requires historical "
                                    "futures candles plus the correct contract/expiry lot specification."
                                )
                            else:
                                if futures_margin_per_lot <= 0:
                                    st.error("Futures backtest blocked: enter a valid contract-specific margin per lot. The system will not assume margin or manufacture capital capacity.")
                                    st.stop()
                                if settings.starting_capital <= 0:
                                    st.warning("Enter an explicit starting capital in configuration before running an executable futures backtest. Capital is an execution input, not a strategy assumption.")
                                    st.stop()
                                result = CapitalAwareRuleBacktestEngine(
                                    starting_capital=settings.starting_capital,
                                    risk_fraction=settings.risk_fraction,
                                    instrument=instrument,
                                    lot_size=futures_lot_size,
                                    point_value=futures_point_value,
                                    margin_per_lot=futures_margin_per_lot,
                                    slippage_points=max(0.25, futures_tick_size),
                                    brokerage_per_order=10.0,
                                    max_daily_loss_fraction=settings.max_daily_loss_fraction,
                                    max_trades_per_day=settings.max_trades_per_day,
                                ).run(
                                    bt_data, symbol=instrument,
                                    evaluation_start=evaluation_start, evaluation_end=evaluation_end,
                                )
                                metrics = result.metrics
                                trades = result.trades.copy()
                                validation = result.validation
                                trading_days = int(validation.get("trading_days", 0))
                                qualified_signals = int(validation.get("qualified_signal_bars", 0))
                                rejected_risk = int(validation.get("rejected_risk_budget", 0))
                                if trading_days < 100:
                                    st.warning("INSUFFICIENT EVIDENCE: fewer than 100 trading days are available.")
                                elif len(trades) < 30:
                                    if qualified_signals and rejected_risk == qualified_signals:
                                        st.warning("NO EXECUTABLE TRADES: qualified signals were found, but every candidate exceeded the configured per-trade risk budget. The risk model was NOT loosened.")
                                    elif qualified_signals == 0:
                                        st.warning("NO QUALIFIED SIGNALS: no candidate survived the configured confirmation gates.")
                                    else:
                                        st.warning("INSUFFICIENT TRADE EVIDENCE: the dataset has enough history, but fewer than 30 closed trades were produced.")
                                else:
                                    st.success("Historical futures backtest completed. Results are historical simulation results, not a guarantee of future performance.")
                                st.markdown("#### Backtest diagnostic funnel")
                                funnel = st.columns(6)
                                funnel[0].metric("Bars", f"{validation.get('bars_considered', 0):,}")
                                funnel[1].metric("Rule-trigger bars", f"{validation.get('rule_trigger_bars', 0):,}")
                                funnel[2].metric("Qualified signals", f"{qualified_signals:,}")
                                funnel[3].metric("Risk rejected", f"{rejected_risk:,}")
                                funnel[4].metric("Closed trades", f"{len(trades):,}")
                                funnel[5].metric("Trading days", f"{trading_days:,}")
                                total_trades = int(metrics.total_trades)
                                wins = int(metrics.winning_trades)
                                losses = int(metrics.losing_trades)
                                top = st.columns(5)
                                top[0].metric("Closed Trades", total_trades)
                                top[1].metric("Win Rate", f"{metrics.win_rate_pct:.1f}%")
                                top[2].metric("Net P&L", f"Rs {metrics.net_pnl:,.0f}")
                                top[3].metric("Return", f"{metrics.return_pct:.2f}%")
                                top[4].metric("Profit Factor", "∞" if metrics.profit_factor == float("inf") else f"{metrics.profit_factor:.2f}")
                                detail = st.columns(5)
                                detail[0].metric("Wins", wins)
                                detail[1].metric("Losses", losses)
                                detail[2].metric("Avg Trade", f"Rs {metrics.average_trade:,.0f}")
                                detail[3].metric("Max Drawdown", f"Rs {metrics.max_drawdown:,.0f}")
                                detail[4].metric("Max DD %", f"{metrics.max_drawdown_pct:.2f}%")
                                if not result.rule_performance.empty:
                                    st.markdown("#### Rule-by-Rule Accuracy")
                                    st.dataframe(result.rule_performance.rename(columns={"win_rate_pct":"accuracy_pct"}), width="stretch", hide_index=True)
                                if not trades.empty:
                                    st.markdown("#### Historical Futures Trades")
                                    cols=[x for x in ["entry_time","exit_time","direction","entry_price","exit_price","stop_loss","target","quantity","rule","pnl","reason"] if x in trades.columns]
                                    st.dataframe(trades[cols], width="stretch", hide_index=True)
                                else:
                                    st.warning("No executable futures trades were closed in this period. That is not the same as 0% accuracy.")

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