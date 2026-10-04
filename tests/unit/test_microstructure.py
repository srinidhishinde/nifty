import pandas as pd

from features.microstructure import (
    AdaptiveMicroWeight,
    dynamic_imbalance_threshold,
    imbalance_ratio,
    infer_aggressor,
)


def test_imbalance_is_zero_below_liquidity_threshold():
    assert imbalance_ratio(100, 100, 250, []) == 0.0


def test_imbalance_is_bounded_and_median_smoothed():
    value = imbalance_ratio(900, 100, 100, [-0.2, 0.2, 0.4, 0.1], 5)
    assert -1.0 <= value <= 1.0
    assert value == 0.4


def test_aggressor_falls_back_to_last_trade_direction_with_sparse_ticks():
    assert infer_aggressor(101, "buy", 100, 102, [100, 101]) == "buyer"


def test_aggressor_uses_midpoint_tolerance_with_dense_ticks():
    ticks = [100 + (i % 3) * 0.01 for i in range(50)]
    assert infer_aggressor(101.5, None, 100, 100.1, ticks, 0.1) == "buyer"


def test_dynamic_threshold_needs_history():
    pct, trigger = dynamic_imbalance_threshold([0.1, 0.2], 1.0)
    assert 70 <= pct <= 90
    assert trigger == 1.0


def test_micro_weight_stays_bounded_and_change_limited(tmp_path):
    store = AdaptiveMicroWeight(tmp_path / "weight.json")
    first = store.update(1.0)
    second = store.update(2.0)
    assert 0.10 <= first <= 0.20
    assert 0.10 <= second <= 0.20
    assert abs(second - first) <= 0.05
