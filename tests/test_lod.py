from __future__ import annotations

import math

import pytest

from worldgen.lod import (
    CameraLODRequest,
    camera_footprint_angular_radius_deg,
    parent_chain,
    required_fallback_tiles,
    select_camera_lod,
)
from worldgen.planet_tiles import CUBE_FACES, TileKey


EARTH_RADIUS_M = 6_371_000.0


def _request(**overrides) -> CameraLODRequest:
    values = dict(
        latitude_deg=23.5,
        longitude_deg=31.25,
        altitude_m=500_000.0,
        viewport_width_px=1280,
        viewport_height_px=720,
        vertical_fov_deg=60.0,
        maximum_screen_error_px=2.0,
        maximum_level=14,
        maximum_tiles=4096,
    )
    values.update(overrides)
    return CameraLODRequest(**values)


def _select(request: CameraLODRequest, *, tile_size: int = 64):
    return select_camera_lod(
        planet_radius_m=EARTH_RADIUS_M,
        tile_size=tile_size,
        request=request,
    )


def _is_ancestor(ancestor: TileKey, descendant: TileKey) -> bool:
    if ancestor.face != descendant.face or ancestor.level > descendant.level:
        return False
    shift = descendant.level - ancestor.level
    return (
        descendant.x >> shift == ancestor.x
        and descendant.y >> shift == ancestor.y
    )


def _covered_by(keys: tuple[TileKey, ...], key: TileKey) -> bool:
    return any(_is_ancestor(candidate, key) for candidate in keys)


def test_camera_request_validation_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="latitude"):
        _request(latitude_deg=91.0).validate()
    with pytest.raises(ValueError, match="altitude"):
        _request(altitude_m=0.0).validate()
    with pytest.raises(ValueError, match="screen_error"):
        _request(maximum_screen_error_px=0.0).validate()
    with pytest.raises(ValueError, match="maximum_level"):
        _request(maximum_level=31).validate()
    with pytest.raises(ValueError, match="maximum_tiles"):
        _request(maximum_tiles=0).validate()


def test_camera_footprint_is_finite_positive_and_bounded_by_horizon():
    altitude = 1_000_000.0
    footprint = camera_footprint_angular_radius_deg(
        planet_radius_m=EARTH_RADIUS_M,
        altitude_m=altitude,
        vertical_fov_deg=60.0,
        viewport_width_px=1920,
        viewport_height_px=1080,
    )
    horizon = math.degrees(
        math.acos(EARTH_RADIUS_M / (EARTH_RADIUS_M + altitude))
    )
    assert math.isfinite(footprint)
    assert 0.0 < footprint <= horizon + 1e-12


def test_selection_is_deterministic_and_sorted():
    request = _request()
    a = _select(request)
    b = _select(request)
    assert a == b
    assert a.keys == tuple(sorted(a.keys))
    assert len(a.keys) > 0


def test_selected_tiles_form_a_true_quadtree_leaf_set():
    result = _select(_request())
    keys = result.keys
    for i, key in enumerate(keys):
        for other in keys[i + 1 :]:
            assert not _is_ancestor(key, other)
            assert not _is_ancestor(other, key)


def test_lower_camera_altitude_never_reduces_maximum_selected_level():
    high = _select(_request(altitude_m=2_000_000.0))
    medium = _select(_request(altitude_m=500_000.0))
    low = _select(_request(altitude_m=100_000.0))
    assert medium.maximum_level >= high.maximum_level
    assert low.maximum_level >= medium.maximum_level
    assert min(t.meters_per_sample_approx for t in low.tiles) <= min(
        t.meters_per_sample_approx for t in high.tiles
    )


def test_stricter_screen_error_never_reduces_selected_detail():
    coarse = _select(_request(maximum_screen_error_px=6.0))
    medium = _select(_request(maximum_screen_error_px=2.0))
    fine = _select(_request(maximum_screen_error_px=0.75))
    assert medium.maximum_level >= coarse.maximum_level
    assert fine.maximum_level >= medium.maximum_level


def test_higher_pixel_density_never_reduces_selected_detail():
    low_dpi = _select(
        _request(viewport_width_px=960, viewport_height_px=540)
    )
    high_dpi = _select(
        _request(viewport_width_px=1920, viewport_height_px=1080)
    )
    assert high_dpi.maximum_level >= low_dpi.maximum_level


def test_finer_selection_remains_covered_by_coarser_parent_leaf_set():
    coarse = _select(_request(maximum_screen_error_px=8.0))
    fine = _select(_request(maximum_screen_error_px=1.0))
    assert fine.maximum_level >= coarse.maximum_level
    assert all(_covered_by(coarse.keys, key) for key in fine.keys)


def test_coarse_selection_is_replaced_only_by_equal_or_descendant_fine_tiles():
    coarse = _select(_request(maximum_screen_error_px=8.0))
    fine = _select(_request(maximum_screen_error_px=1.0))
    for coarse_key in coarse.keys:
        assert any(_is_ancestor(coarse_key, key) for key in fine.keys)


def test_maximum_level_is_a_hard_refinement_cap():
    result = _select(
        _request(
            altitude_m=20_000.0,
            maximum_screen_error_px=0.05,
            maximum_level=5,
        )
    )
    assert result.maximum_level <= 5
    assert all(key.level <= 5 for key in result.keys)


def test_resident_height_estimate_is_exact_and_bounded_by_tile_budget():
    tile_size = 64
    request = _request(maximum_tiles=1024)
    result = _select(request, tile_size=tile_size)
    per_tile = (tile_size + 1) ** 2 * 4
    assert result.estimated_resident_height_bytes == len(result.tiles) * per_tile
    assert result.estimated_resident_height_bytes <= request.maximum_tiles * per_tile


def test_pathological_budget_fails_instead_of_returning_unbounded_residency():
    with pytest.raises(RuntimeError, match="maximum_tiles|safety budget"):
        _select(
            _request(
                altitude_m=500_000.0,
                maximum_screen_error_px=0.25,
                maximum_level=12,
                maximum_tiles=1,
            )
        )


def test_parent_chain_and_required_fallbacks_are_root_to_parent_and_unique():
    key = TileKey("pz", 4, 11, 6)
    chain = parent_chain(key)
    assert chain == (
        TileKey("pz", 0, 0, 0),
        TileKey("pz", 1, 1, 0),
        TileKey("pz", 2, 2, 1),
        TileKey("pz", 3, 5, 3),
    )
    sibling = TileKey("pz", 4, 10, 6)
    fallbacks = required_fallback_tiles((key, sibling))
    assert fallbacks == tuple(sorted(set(parent_chain(key)) | set(parent_chain(sibling))))


def test_polar_and_dateline_camera_positions_select_only_valid_cube_tiles():
    requests = (
        _request(latitude_deg=89.999, longitude_deg=179.999, altitude_m=150_000.0),
        _request(latitude_deg=-89.999, longitude_deg=-179.999, altitude_m=150_000.0),
        _request(latitude_deg=0.0, longitude_deg=179.999, altitude_m=150_000.0),
        _request(latitude_deg=0.0, longitude_deg=-179.999, altitude_m=150_000.0),
    )
    for request in requests:
        result = _select(request)
        assert result.tiles
        for key in result.keys:
            key.validate()
            assert key.face in CUBE_FACES


def test_extremely_high_camera_still_returns_a_bounded_valid_leaf_set():
    result = _select(
        _request(
            altitude_m=50_000_000.0,
            maximum_screen_error_px=2.0,
            maximum_level=12,
        )
    )
    assert result.tiles
    assert len(result.tiles) <= 4096
    assert result.minimum_level <= result.maximum_level <= 12
