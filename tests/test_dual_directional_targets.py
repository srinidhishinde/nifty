from __future__ import annotations

from scripts.benchmark_dual_directional_targets import summarize_target


def test_dual_target_summaries_keep_accuracy_and_baseline_separate(monkeypatch, tmp_path):
    import scripts.benchmark_dual_directional_targets as dual

    def fake_evaluate(path, *, min_train, estimator, tolerance):
        if "nifty" in str(path):
            return {
                "file": str(path), "evaluated_targets": 100, "correct_predictions": 82,
                "coverage_pct": 100.0, "momentum_baseline_correct_predictions": 60,
            }
        return {
            "file": str(path), "evaluated_targets": 100, "correct_predictions": 55,
            "coverage_pct": 100.0, "momentum_baseline_correct_predictions": 52,
        }

    monkeypatch.setattr(dual, "evaluate", fake_evaluate)
    nifty = summarize_target(
        [tmp_path / "nifty-day1.csv", tmp_path / "nifty-day2.csv"],
        min_train=80, estimator="logistic", tolerance=0.01,
        min_accuracy=80, min_coverage=80, min_targets=100, min_sessions=2,
    )
    option = summarize_target(
        [tmp_path / "option-day1.csv", tmp_path / "option-day2.csv"],
        min_train=80, estimator="logistic", tolerance=0.01,
        min_accuracy=80, min_coverage=80, min_targets=100, min_sessions=2,
    )

    assert nifty["aggregate"]["accuracy_pct"] == 82.0
    assert nifty["status"] == "PASS"
    assert option["aggregate"]["accuracy_pct"] == 55.0
    assert option["status"] == "FAIL"
    assert nifty["aggregate"]["evaluated_targets"] == option["aggregate"]["evaluated_targets"] == 200
