from execution.execution_ledger import ExecutionLedger

def test_ambiguous_order_cannot_be_retried(tmp_path):
    ledger=ExecutionLedger(tmp_path/"ledger.jsonl")
    ledger.begin("d1","NIFTYCE","BUY",50,"2026-10-09T10:00:00+05:30")
    ledger.mark_ambiguous("d1","timeout","2026-10-09T10:00:01+05:30")
    assert ledger.can_submit("d1") is False
