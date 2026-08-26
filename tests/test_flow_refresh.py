import numpy as np

from worldgen.flow_refresh import FlowRefreshState, decide_flow_refresh


def _state(shape=(8, 16)):
    state = FlowRefreshState()
    z = np.ones(shape, dtype=np.float32)
    land = np.ones(shape, dtype=bool)
    state.mark_refreshed(0, z, land)
    return state, z, land


def test_interval_mode_preserves_legacy_schedule():
    state, z, land = _state()
    d1 = decide_flow_refresh(state, iteration=1, elevation_km=z, land=land, mode="interval", interval=2)
    d2 = decide_flow_refresh(state, iteration=2, elevation_km=z, land=land, mode="interval", interval=2)
    assert not d1.refresh and d1.reason == "reuse"
    assert d2.refresh and d2.reason == "fixed_interval"


def test_adaptive_refreshes_after_accumulated_elevation_change():
    state, z, land = _state()
    changed = z.copy()
    changed[2:6, 4:12] += 0.010  # ten metres
    decision = decide_flow_refresh(
        state,
        iteration=1,
        elevation_km=changed,
        land=land,
        mode="adaptive",
        max_interval=10,
        elevation_threshold_m=5.0,
        land_change_fraction_threshold=1.0,
        delta_threshold_m=100.0,
    )
    assert decision.refresh
    assert decision.reason == "elevation_change"
    assert decision.elevation_change_m >= 9.9


def test_adaptive_refreshes_when_coastline_moves():
    state, z, land = _state()
    changed_land = land.copy()
    changed_land[:, 0] = False
    weights = np.ones_like(z, dtype=np.float64)
    decision = decide_flow_refresh(
        state,
        iteration=1,
        elevation_km=z,
        land=changed_land,
        mode="adaptive",
        max_interval=10,
        elevation_threshold_m=100.0,
        land_change_fraction_threshold=0.01,
        delta_threshold_m=100.0,
        area_weights=weights,
    )
    assert decision.refresh
    assert decision.reason == "coastline_change"
    assert decision.land_change_fraction > 0.01


def test_adaptive_refreshes_after_large_delta_aggradation():
    state, z, land = _state()
    decision = decide_flow_refresh(
        state,
        iteration=1,
        elevation_km=z,
        land=land,
        mode="adaptive",
        max_interval=10,
        elevation_threshold_m=100.0,
        land_change_fraction_threshold=1.0,
        delta_threshold_m=2.0,
        previous_delta_max_m=3.5,
    )
    assert decision.refresh and decision.reason == "delta_aggradation"


def test_adaptive_max_interval_prevents_indefinite_reuse():
    state, z, land = _state()
    decision = decide_flow_refresh(
        state,
        iteration=3,
        elevation_km=z,
        land=land,
        mode="adaptive",
        max_interval=3,
        elevation_threshold_m=100.0,
        land_change_fraction_threshold=1.0,
        delta_threshold_m=100.0,
    )
    assert decision.refresh and decision.reason == "max_interval"


def test_mark_refreshed_resets_baseline_and_age():
    state, z, land = _state()
    changed = z + 0.02
    state.mark_refreshed(4, changed, land)
    decision = decide_flow_refresh(
        state,
        iteration=5,
        elevation_km=changed,
        land=land,
        mode="adaptive",
        max_interval=3,
        elevation_threshold_m=1.0,
        land_change_fraction_threshold=1.0,
        delta_threshold_m=100.0,
    )
    assert not decision.refresh
    assert decision.iterations_since_refresh == 1
    assert decision.elevation_change_m == 0.0
