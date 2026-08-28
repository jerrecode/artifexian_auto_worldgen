from __future__ import annotations

import json

import numpy as np
import pytest

from worldgen.planet_tiles import PlanetTilePyramid, TilePyramidSpec
from worldgen.precompute import (
    PrecomputeLimitError,
    PrecomputeProducts,
    complete_pyramid_tile_count,
    enforce_precompute_limits,
    iter_complete_pyramid,
    make_precompute_plan,
    precompute_complete_prefix,
)


def _write_world(root):
    h, w = 8, 16
    lat = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    lon = -180.0 + (np.arange(w) + 0.5) * 360.0 / w
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    elevation = (
        0.8 * np.sin(2.0 * np.pi * xx / w)
        + 0.25 * np.cos(np.pi * (yy + 0.5) / h)
    ).astype(np.float32)
    np.savez(
        root / "world_arrays.npz",
        lat=lat,
        lon=lon,
        elevation_km=elevation,
        plate_id=((xx + yy) % 4).astype(np.int16),
    )
    (root / "world.json").write_text(
        json.dumps({"seed": 1234, "astronomy": {"planet": {"radius_earth": 1.0}}}),
        encoding="utf-8",
    )


def test_complete_prefix_tile_count_and_iterator_are_exact():
    assert complete_pyramid_tile_count(0) == 6
    assert complete_pyramid_tile_count(1) == 30
    assert complete_pyramid_tile_count(2) == 126
    keys = tuple(iter_complete_pyramid(2))
    assert len(keys) == 126
    assert len(set(keys)) == 126
    assert {key.level for key in keys} == {0, 1, 2}
    assert sum(key.level == 0 for key in keys) == 6
    assert sum(key.level == 1 for key in keys) == 24
    assert sum(key.level == 2 for key in keys) == 96


def test_precompute_materializes_every_tile_and_is_resumable(tmp_path):
    _write_world(tmp_path)
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=16, elevation_detail_strength=0.1, maximum_level=5),
    )
    products = PrecomputeProducts(fields=("elevation_m", "plate_id"))
    first = precompute_complete_prefix(
        pyramid,
        1,
        products=products,
        workers=2,
        maximum_tiles=100,
        maximum_estimated_bytes=256 * 1024**2,
    )
    assert first.total_tiles == 30
    assert first.completed_tiles == 30
    assert first.base_generated_tiles == 30
    assert first.base_cache_hits == 0
    for key in iter_complete_pyramid(1):
        assert pyramid._field_path(key, "elevation_m").exists()
        assert pyramid._field_path(key, "plate_id").exists()
        assert pyramid._metadata_path(key).exists()

    second = precompute_complete_prefix(
        pyramid,
        1,
        products=products,
        workers=2,
        maximum_tiles=100,
        maximum_estimated_bytes=256 * 1024**2,
    )
    assert second.completed_tiles == 30
    assert second.base_cache_hits == 30
    assert second.base_generated_tiles == 0
    status = json.loads(open(second.status_path, encoding="utf-8").read())
    assert status["state"] == "complete"
    assert status["progress"]["completed_tiles"] == 30


def test_plan_estimates_storage_and_guards_large_prefix(tmp_path):
    _write_world(tmp_path)
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=16, maximum_level=10),
    )
    plan = make_precompute_plan(pyramid, 3, PrecomputeProducts())
    assert plan.tile_count == 510
    assert plan.estimated_uncompressed_bytes > plan.tile_count * 16 * 16
    with pytest.raises(PrecomputeLimitError):
        enforce_precompute_limits(plan, maximum_tiles=100, maximum_estimated_bytes=1 << 50)
    # Explicit opt-in is allowed for users who intentionally provisioned storage.
    enforce_precompute_limits(
        plan,
        maximum_tiles=100,
        maximum_estimated_bytes=1,
        force_large=True,
    )


def test_depth_cannot_exceed_tileset_maximum(tmp_path):
    _write_world(tmp_path)
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=16, maximum_level=2),
    )
    with pytest.raises(ValueError):
        make_precompute_plan(pyramid, 3)
