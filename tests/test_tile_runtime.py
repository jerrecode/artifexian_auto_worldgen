from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from threading import Lock
import time

import numpy as np

from worldgen.planet_tiles import PlanetTilePyramid, TileKey, TilePyramidSpec
from worldgen.tile_runtime import (
    KeyedRequestCoalescer,
    PersistentTileFileLRU,
    PlanetTileRuntime,
)


def _write_world(root: Path) -> None:
    h, w = 6, 12
    lat = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    lon = -180.0 + (np.arange(w) + 0.5) * 360.0 / w
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    elevation = (
        np.sin(2.0 * np.pi * xx / w) + 0.25 * np.cos(np.pi * (yy + 0.5) / h)
    ).astype(np.float32)
    np.savez(root / "world_arrays.npz", lat=lat, lon=lon, elevation_km=elevation)
    (root / "world.json").write_text(
        json.dumps({"seed": 42, "astronomy": {"planet": {"radius_earth": 1.0}}}),
        encoding="utf-8",
    )


def test_keyed_request_coalescer_serializes_same_key():
    coalescer = KeyedRequestCoalescer()
    state_lock = Lock()
    active = 0
    maximum_active = 0

    def worker() -> None:
        nonlocal active, maximum_active
        with coalescer.hold(("px", 3, 2, 1)):
            with state_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.01)
            with state_lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _: worker(), range(8)))

    assert maximum_active == 1
    assert coalescer.active_keys() == 0


def test_keyed_request_coalescer_does_not_globally_serialize_unrelated_keys():
    coalescer = KeyedRequestCoalescer()
    state_lock = Lock()
    active = 0
    maximum_active = 0

    def worker(key: int) -> None:
        nonlocal active, maximum_active
        with coalescer.hold(key):
            with state_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.03)
            with state_lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(worker, range(4)))

    assert maximum_active > 1
    assert coalescer.active_keys() == 0


def test_persistent_file_lru_evicts_oldest_and_never_tileset(tmp_path):
    root = tmp_path / "tiles"
    root.mkdir()
    tileset = root / "tileset.json"
    tileset.write_text("{}", encoding="utf-8")
    paths = [root / f"field_{i}.bin" for i in range(3)]
    for i, path in enumerate(paths):
        path.write_bytes(bytes([i]) * 10)
        stamp = 1_700_000_000_000_000_000 + i * 1_000_000
        os.utime(path, ns=(stamp, stamp))

    cache = PersistentTileFileLRU(root, max_bytes=20)
    removed = cache.prune()
    assert removed == (paths[0],)
    assert not paths[0].exists()
    assert paths[1].exists() and paths[2].exists()
    assert tileset.exists()
    stats = cache.stats()
    assert stats.files == 2
    assert stats.bytes_used == 20
    assert stats.evictions == 1


def test_persistent_file_lru_touch_changes_eviction_priority(tmp_path):
    root = tmp_path / "tiles"
    root.mkdir()
    a = root / "a.bin"; b = root / "b.bin"; c = root / "c.bin"
    for i, path in enumerate((a, b, c)):
        path.write_bytes(b"x" * 10)
        stamp = 1_700_000_000_000_000_000 + i * 1_000_000
        os.utime(path, ns=(stamp, stamp))
    cache = PersistentTileFileLRU(root, max_bytes=20)
    cache.touch((a,))
    removed = cache.prune()
    assert b in removed
    assert a.exists()


def test_persistent_file_lru_protects_current_response_even_if_over_budget(tmp_path):
    root = tmp_path / "tiles"
    root.mkdir()
    old = root / "old.bin"
    current = root / "current.bin"
    old.write_bytes(b"o" * 20)
    current.write_bytes(b"c" * 30)
    cache = PersistentTileFileLRU(root, max_bytes=10)
    removed = cache.prune(protected=(current,))
    assert old in removed
    assert current.exists()
    assert cache.stats().bytes_used == 30


def test_planet_tile_runtime_coalesces_concurrent_missing_tile_generation(tmp_path):
    _write_world(tmp_path)
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=16, maximum_level=8),
    )
    runtime = PlanetTileRuntime(pyramid, disk_cache_max_bytes=10_000_000)
    key = TileKey("px", 4, 3, 5)

    original = pyramid._generate_field
    counter_lock = Lock()
    generation_calls = 0

    def counted_generate(key_arg, field, geom):
        nonlocal generation_calls
        with counter_lock:
            generation_calls += 1
        time.sleep(0.02)
        return original(key_arg, field, geom)

    pyramid._generate_field = counted_generate  # type: ignore[method-assign]

    def worker(_):
        return runtime.generate_tile(key, ("elevation_m",))

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(worker, range(8)))

    assert generation_calls == 1
    assert results[0].fields["elevation_m"].exists()
    assert sum(result.cache_hit for result in results) == 7
    assert runtime.coalescer.active_keys() == 0


def test_runtime_quota_prunes_older_products_but_keeps_returned_tile(tmp_path):
    _write_world(tmp_path)
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=16, maximum_level=8),
    )
    # Large enough for one small response, deliberately too small for many tiles.
    runtime = PlanetTileRuntime(pyramid, disk_cache_max_bytes=3_000)
    first = runtime.generate_tile(TileKey("px", 3, 1, 1), ("elevation_m",))
    time.sleep(0.01)
    second = runtime.generate_tile(TileKey("px", 3, 2, 1), ("elevation_m",))

    assert second.fields["elevation_m"].exists()
    assert second.metadata_path.exists()
    stats = runtime.cache_stats()
    assert stats.evictions >= 1
    # The first response is the eviction candidate once a newer protected response
    # pushes the cache over quota. At least one of its managed files should be gone.
    assert not (
        first.fields["elevation_m"].exists() and first.metadata_path.exists()
    )
