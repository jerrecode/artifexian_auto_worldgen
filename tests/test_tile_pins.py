from __future__ import annotations

import json
import os

import numpy as np

from worldgen.planet_tiles import PlanetTilePyramid, TileKey, TilePyramidSpec
from worldgen.tile_pins import (
    clear_pinned_prefix,
    path_is_pinned,
    path_tile_level,
    pin_complete_prefix,
    read_pinned_prefix,
)
from worldgen.tile_runtime import PersistentTileFileLRU


def _write_world(root):
    h, w = 6, 12
    lat = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    lon = -180.0 + (np.arange(w) + 0.5) * 360.0 / w
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    elevation = (np.sin(2.0 * np.pi * xx / w) + 0.1 * yy).astype(np.float32)
    np.savez(root / "world_arrays.npz", lat=lat, lon=lon, elevation_km=elevation)
    (root / "world.json").write_text(
        json.dumps({"seed": 19, "astronomy": {"planet": {"radius_earth": 1.0}}}),
        encoding="utf-8",
    )


def test_pin_manifest_is_monotonic_and_path_levels_are_detected(tmp_path):
    _write_world(tmp_path)
    pyramid = PlanetTilePyramid(tmp_path, spec=TilePyramidSpec(tile_size=16, maximum_level=8))
    first = pin_complete_prefix(pyramid, 2)
    assert first.maximum_level == 2
    second = pin_complete_prefix(pyramid, 1)
    assert second.maximum_level == 2
    loaded = read_pinned_prefix(pyramid.root)
    assert loaded is not None and loaded.maximum_level == 2

    z2 = pyramid._field_path(TileKey("px", 2, 1, 1), "elevation_m")
    z3 = pyramid._field_path(TileKey("px", 3, 1, 1), "elevation_m")
    assert path_tile_level(pyramid.root, z2) == 2
    assert path_tile_level(pyramid.root, z3) == 3
    assert path_is_pinned(pyramid.root, z2, loaded)
    assert not path_is_pinned(pyramid.root, z3, loaded)
    assert clear_pinned_prefix(pyramid.root)
    assert read_pinned_prefix(pyramid.root) is None


def test_runtime_lru_preserves_pinned_prefix_and_evicts_deeper_tiles(tmp_path):
    _write_world(tmp_path)
    pyramid = PlanetTilePyramid(tmp_path, spec=TilePyramidSpec(tile_size=16, maximum_level=8))
    coarse_key = TileKey("px", 0, 0, 0)
    deep_key = TileKey("px", 3, 2, 2)
    coarse = pyramid.generate_tile(coarse_key, ("elevation_m",))
    deep = pyramid.generate_tile(deep_key, ("elevation_m",))

    # Make the pinned files older so ordinary LRU ordering would choose them first.
    old = 1_700_000_000_000_000_000
    new = old + 1_000_000_000
    for path in (*coarse.fields.values(), coarse.metadata_path):
        os.utime(path, ns=(old, old))
    for path in (*deep.fields.values(), deep.metadata_path):
        os.utime(path, ns=(new, new))

    pin_complete_prefix(pyramid, 0)
    cache = PersistentTileFileLRU(pyramid.root, max_bytes=1)
    removed = cache.prune()

    assert coarse.fields["elevation_m"].exists()
    assert coarse.metadata_path.exists()
    assert removed
    assert not (
        deep.fields["elevation_m"].exists() and deep.metadata_path.exists()
    )
    stats = cache.stats()
    assert stats.pinned_maximum_level == 0
    assert stats.pinned_files >= 2
    assert stats.pinned_bytes > 0
    # It is valid to remain above a soft quota when the explicitly pinned archive
    # alone exceeds the quota.
    assert stats.bytes_used >= stats.pinned_bytes > stats.max_bytes


def test_precompute_control_manifests_are_not_managed_payload(tmp_path):
    _write_world(tmp_path)
    pyramid = PlanetTilePyramid(tmp_path, spec=TilePyramidSpec(tile_size=16, maximum_level=4))
    pin_complete_prefix(pyramid, 0)
    extra = pyramid.root / "precompute" / "status.json"
    extra.write_text("{}", encoding="utf-8")
    cache = PersistentTileFileLRU(pyramid.root, max_bytes=0)
    cache.prune()
    assert extra.exists()
    assert (pyramid.root / "precompute" / "pinned_prefix.json").exists()
