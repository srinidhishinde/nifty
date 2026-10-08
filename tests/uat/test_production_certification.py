from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from analytics.signal_journal import SignalJournal
from features.option_chain import OptionContract
from marketdata.decision_source import load_kotak_decision_snapshot
from strategy.production_hardening import (
    DecisionEvidence,
    validate_option_contract,
    option_expiry_valid,
    source_policy_ok,
    live_permission,
    validate_contract_identity,
    runtime_clock_ok,
)


def _contract(ts=None, expiry=None):
    now = ts or datetime.now().astimezone()
    return OptionContract(
        symbol="NIFTY26O1320250CE", exchange="NSE_FO", underlying="NIFTY",
        expiry=expiry or (now.date() + timedelta(days=4)).isoformat(), strike=20250,
        option_type="CE", instrument_token="nse_fo|123", ltp=100, bid=99, ask=101,
        volume=1000, open_interest=10000, quote_timestamp=now, quote_source="KOTAK_NEO",
    )


def test_cert_1_broker_outage_fails_closed():
    class Broker:
        client = None
        def connection_status(self):
            return type("Status", (), {"connected": False})()
    snapshot = load_kotak_decision_snapshot(Broker(), instrument="NIFTY", now=pd.Timestamp.now(tz="Asia/Kolkata"))
    assert snapshot.status == "RED"
    assert snapshot.signal.direction == "WAIT"


def test_cert_2_stale_quote_is_not_executable():
    stale = _contract(datetime.now().astimezone() - timedelta(minutes=5))
    ok, reasons = validate_option_contract(stale, max_quote_age_seconds=60)
    assert not ok
    assert any("stale" in reason.lower() for reason in reasons)


def test_cert_3_restart_journal_is_idempotent(tmp_path):
    path = tmp_path / "signal.jsonl"
    record = {"instrument":"NIFTY", "timeframe":"5m", "direction":"BUY CE", "candle_timestamp":"2026-10-09T09:20:00+05:30", "option_snapshot_timestamp":"2026-10-09T09:20:02+05:30", "contract_symbol":"NIFTY26O1320250CE"}
    SignalJournal(path).append(record)
    SignalJournal(path).append(record)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_cert_4_expiry_and_source_policy_are_fail_closed():
    assert option_expiry_valid((datetime.now().date() + timedelta(days=1)).isoformat())
    assert not option_expiry_valid((datetime.now().date() - timedelta(days=1)).isoformat())
    assert source_policy_ok(environment="RESEARCH", source="KOTAK_NEO")
    assert not source_policy_ok(environment="RESEARCH", source="YAHOO", yahoo=True)
    assert source_policy_ok(environment="BACKTEST", source="YAHOO", yahoo=True)


def test_cert_5_live_permission_requires_every_mandatory_gate():
    fields = dict(
        instrument="NIFTY", timeframe="5m", source="KOTAK_NEO",
        candle_timestamp=pd.Timestamp.now(tz="Asia/Kolkata"),
        option_snapshot_timestamp=pd.Timestamp.now(tz="Asia/Kolkata"),
        underlying_quote_timestamp=pd.Timestamp.now(tz="Asia/Kolkata"),
        candle_valid=True, option_chain_valid=True, option_quotes_valid=True,
        liquidity_valid=True, synchronized=True, regime_valid=True, ml_validated=True,
        ensemble_valid=True, trade_plan_valid=True, account_valid=True, risk_valid=True,
        journal_valid=True, reconciliation_valid=True, tests_valid=True, backtest_valid=True,
        contract_valid=True, expiry_valid=True, clock_valid=True, schema_valid=True,
        source_policy_valid=True, reasons=(),
    )
    evidence = DecisionEvidence(**fields)
    assert live_permission(evidence, config_enabled=True)
    blocked = DecisionEvidence(**{**fields, "reconciliation_valid": False})
    assert not live_permission(blocked, config_enabled=True)


def test_cert_6_malformed_numeric_quote_fails_closed():
    bad = _contract()
    object.__setattr__(bad, "bid", "N/A")
    ok, reasons = validate_option_contract(bad)
    assert not ok
    assert any("numeric" in reason.lower() for reason in reasons)


def test_cert_7_contract_identity_is_cross_checked():
    contract = _contract()
    ok, reasons = validate_contract_identity(contract, expected_instrument="CRUDEOIL", expected_exchange="mcx_fo")
    assert not ok
    assert any("underlying" in reason.lower() for reason in reasons)
    assert any("exchange" in reason.lower() for reason in reasons)


def test_cert_8_clock_gate_rejects_future_broker_timestamp():
    future = pd.Timestamp.now(tz="Asia/Kolkata") + pd.Timedelta(minutes=1)
    assert not runtime_clock_ok(reference_timestamp=future)


def test_cert_9_execution_config_cannot_authorize_invalid_evidence():
    fields = dict(
        instrument="NIFTY", timeframe="5m", source="KOTAK_NEO",
        candle_timestamp=pd.Timestamp.now(tz="Asia/Kolkata"),
        option_snapshot_timestamp=pd.Timestamp.now(tz="Asia/Kolkata"),
        underlying_quote_timestamp=pd.Timestamp.now(tz="Asia/Kolkata"),
        candle_valid=True, option_chain_valid=True, option_quotes_valid=True,
        liquidity_valid=True, synchronized=True, regime_valid=True, ml_validated=True,
        ensemble_valid=True, trade_plan_valid=True, account_valid=True, risk_valid=True,
        journal_valid=True, reconciliation_valid=False, tests_valid=True,
        backtest_valid=True, contract_valid=True, expiry_valid=True,
        clock_valid=True, schema_valid=True, source_policy_valid=True, reasons=(),
    )
    evidence = DecisionEvidence(**fields)
    assert not live_permission(evidence, config_enabled=True)