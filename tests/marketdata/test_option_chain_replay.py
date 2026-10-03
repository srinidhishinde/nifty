import pandas as pd
import pytest

from marketdata.option_chain_replay import replay_option_chain_csv

def test_single_snapshot_requires_timestamp():
    data = pd.DataFrame({"Strike": [25000], "Calls OI": [100], "Puts OI": [200]})
    with pytest.raises(ValueError, match="requires a timestamp"):
        replay_option_chain_csv(data)

def test_timestamped_snapshot_replays():
    data = pd.DataFrame({
        "timestamp": ["2026-10-01 09:15:00", "2026-10-01 09:20:00"],
        "Strike": [25000, 25000],
        "Calls OI": [100, 110],
        "Puts OI": [200, 210],
        "Calls Volume": [10, 12],
        "Puts Volume": [20, 22],
    })
    snapshots = replay_option_chain_csv(data)
    assert len(snapshots) == 2
    assert all(len(contracts) == 2 for _, contracts in snapshots)
