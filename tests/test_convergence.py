import math

from worldgen.convergence import ConvergenceThresholds, ConvergenceTracker


def test_tracker_requires_history_and_full_fidelity():
    tracker = ConvergenceTracker(
        ConvergenceThresholds(temperature_c=0.2, precipitation_mm_year=20.0, elevation_m=3.0)
    )
    first = tracker.evaluate(
        1,
        temperature_change_c=None,
        precipitation_change_mm_year=None,
        elevation_change_m=0.0,
        full_fidelity=True,
    )
    assert not first.stop
    assert first.reason == "insufficient_history"

    predictor = tracker.evaluate(
        2,
        temperature_change_c=0.01,
        precipitation_change_mm_year=1.0,
        elevation_change_m=0.2,
        full_fidelity=False,
    )
    assert predictor.converged
    assert not predictor.stop
    assert predictor.reason == "predictor_pass"


def test_tracker_stops_after_required_consecutive_converged_passes():
    tracker = ConvergenceTracker(
        ConvergenceThresholds(
            temperature_c=0.2,
            precipitation_mm_year=20.0,
            elevation_m=3.0,
            required_consecutive=2,
            minimum_passes=2,
        )
    )
    a = tracker.evaluate(
        2,
        temperature_change_c=0.1,
        precipitation_change_mm_year=5.0,
        elevation_change_m=1.0,
    )
    b = tracker.evaluate(
        3,
        temperature_change_c=0.05,
        precipitation_change_mm_year=4.0,
        elevation_change_m=0.5,
    )
    assert a.converged and not a.stop
    assert a.consecutive_converged == 1
    assert b.stop and b.reason == "converged"
    assert b.consecutive_converged == 2


def test_failed_pass_resets_consecutive_counter():
    tracker = ConvergenceTracker(
        ConvergenceThresholds(0.2, 20.0, required_consecutive=2)
    )
    tracker.evaluate(1, temperature_change_c=0.1, precipitation_change_mm_year=10.0)
    bad = tracker.evaluate(2, temperature_change_c=0.5, precipitation_change_mm_year=10.0)
    good = tracker.evaluate(3, temperature_change_c=0.1, precipitation_change_mm_year=10.0)
    assert bad.consecutive_converged == 0
    assert not bad.converged
    assert good.consecutive_converged == 1
    assert not good.stop


def test_normalized_residual_is_maximum_threshold_ratio():
    tracker = ConvergenceTracker(ConvergenceThresholds(0.2, 20.0, elevation_m=2.0))
    decision = tracker.evaluate(
        1,
        temperature_change_c=0.1,
        precipitation_change_mm_year=30.0,
        elevation_change_m=0.5,
    )
    assert math.isclose(decision.temperature_ratio, 0.5)
    assert math.isclose(decision.precipitation_ratio, 1.5)
    assert math.isclose(decision.elevation_ratio, 0.25)
    assert math.isclose(decision.normalized_residual, 1.5)
    assert not decision.converged


def test_zero_tolerance_requires_exact_zero_change():
    tracker = ConvergenceTracker(ConvergenceThresholds(0.0, 0.0))
    exact = tracker.evaluate(1, temperature_change_c=0.0, precipitation_change_mm_year=0.0)
    assert exact.stop
    tracker = ConvergenceTracker(ConvergenceThresholds(0.0, 0.0))
    nonzero = tracker.evaluate(1, temperature_change_c=1e-12, precipitation_change_mm_year=0.0)
    assert math.isinf(nonzero.normalized_residual)
    assert not nonzero.stop
