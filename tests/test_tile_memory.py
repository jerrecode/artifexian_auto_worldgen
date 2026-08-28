from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from worldgen.planet_tiles import PlanetTilePyramid, TileKey, TilePyramidSpec
from worldgen.tile_memory import ResidentTileMemoryCache
from worldgen.tile_pins import pin_complete_prefix
from worldgen.tile_runtime import PlanetTileRuntime


def _write_world(root: Path) -> None:
    h, w = 8, 16
    lat = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    lon = -180.0 + (np.arange(w) + 0.5) * 360.0 / w
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    elevation = (
        0.9 * np.sin(2.0 * np.pi * xx / w)
        + 0.2 * np.cos(np.pi * (yy + 0.5) / h)
    ).astype(np.float32)
    np.savez(root / "world_arrays.npz", lat=lat, lon=lon, elevation_km=elevation)
    (root / "world.json").write_text(
        json.dumps({"seed": 123, "astronomy": {"planet": {"radius_earth": 1.0}}}),
        encoding="utf-8",
    )


def test_resident_memory_cache_is_byte_bounded_lru():
    cache = ResidentTileMemoryCache(max_bytes=20)
    a = b"a" * 10
    b = b"b" * 10
    c = b"c" * 10
    assert cache.put("a", a)
    assert cache.put("b", b)
    assert cache.get("a") is a  # a becomes most recently used
    assert cache.put("c", c)
    assert cache.contains("a")
    assert not cache.contains("b")
    assert cache.contains("c")
    stats = cache.stats()
    assert stats.bytes_used == 20
    assert stats.evictions == 1
    assert stats.hits == 1


def test_oversize_resident_object_is_returnable_but_not_cached():
    cache = ResidentTileMemoryCache(max_bytes=4)
    payload = b"12345"
    assert not cache.put("large", payload)
    assert not cache.contains("large")
    assert cache.stats().rejected_oversize == 1


def test_runtime_loads_static_npy_into_ram_and_release_keeps_file(tmp_path):
    _write_world(tmp_path)
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=16, maximum_level=6),
    )
    key = TileKey("px", 2, 1, 2)
    generated = pyramid.generate_tile(key, ("elevation_m",))
    path = generated.fields["elevation_m"]
    assert path.exists()

    runtime = PlanetTileRuntime(
        pyramid,
        disk_cache_max_bytes=None,
        memory_cache_max_bytes=8 * 1024 * 1024,
    )
    first = runtime.load_field(key, "elevation_m", generate=False)
    second = runtime.load_field(key, "elevation_m", generate=False)
    assert first is second
    assert isinstance(first, np.ndarray)
    assert not isinstance(first, np.memmap)
    assert first.flags.writeable is False
    stats = runtime.memory_cache_stats()
    assert stats.items == 1
    assert stats.hits >= 1
    assert stats.bytes_used == first.nbytes

    assert runtime.release_field(key, "elevation_m")
    assert runtime.memory_cache_stats().items == 0
    assert path.exists(), "unloading RAM must not remove the static tile file"

    third = runtime.load_field(key, "elevation_m", generate=False)
    np.testing.assert_array_equal(third, first)
    assert third is not first
    assert path.exists()


def test_ram_lru_can_evict_precomputed_tile_while_disk_pin_preserves_file(tmp_path):
    _write_world(tmp_path)
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=16, maximum_level=6),
    )
    a = TileKey("px", 0, 0, 0)
    b = TileKey("nx", 0, 0, 0)
    a_result = pyramid.generate_tile(a, ("elevation_m",))
    b_result = pyramid.generate_tile(b, ("elevation_m",))
    pin_complete_prefix(pyramid, 0)

    one_array_bytes = (pyramid.spec.tile_size + 1) ** 2 * np.dtype(np.float32).itemsize
    runtime = PlanetTileRuntime(
        pyramid,
        disk_cache_max_bytes=1,
        memory_cache_max_bytes=one_array_bytes,
    )
    runtime.load_field(a, "elevation_m", generate=False)
    assert runtime.memory_cache_stats().items == 1
    runtime.load_field(b, "elevation_m", generate=False)
    stats = runtime.memory_cache_stats()
    assert stats.items == 1
    assert stats.evictions >= 1

    # RAM residency is disposable even for pinned precompute content; persistence is not.
    assert a_result.fields["elevation_m"].exists()
    assert b_result.fields["elevation_m"].exists()
    disk = runtime.cache_stats()
    assert disk.pinned_maximum_level == 0
    assert disk.pinned_files >= 2


def test_clear_memory_cache_unloads_many_tiles_without_touching_static_files(tmp_path):
    _write_world(tmp_path)
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=16, maximum_level=6),
    )
    runtime = PlanetTileRuntime(pyramid, memory_cache_max_bytes=16 * 1024 * 1024)
    keys = [TileKey("pz", 1, 0, 0), TileKey("pz", 1, 1, 0)]
    paths = []
    for key in keys:
        result = pyramid.generate_tile(key, ("elevation_m", "ocean_depth_m"))
        paths.extend(result.fields.values())
        runtime.load_field(key, "elevation_m", generate=False)
        runtime.load_field(key, "ocean_depth_m", generate=False)
    assert runtime.memory_cache_stats().items == 4
    assert runtime.clear_memory_cache() == 4
    assert runtime.memory_cache_stats().items == 0
    assert all(path.exists() for path in paths)


def test_static_binary_product_is_loaded_and_unloaded_through_same_ram_cache(tmp_path):
    _write_world(tmp_path)
    pyramid = PlanetTilePyramid(tmp_path, spec=TilePyramidSpec(tile_size=16))
    runtime = PlanetTileRuntime(pyramid, memory_cache_max_bytes=1024)
    product = pyramid.root / "viewer" / "z00" / "px" / "x00000000" / "y00000000.bin"
    product.parent.mkdir(parents=True, exist_ok=True)
    product.write_bytes(b"static-map-payload")

    first = runtime.load_product_bytes(product)
    second = runtime.load_product_bytes(product)
    assert first is second
    assert first == b"static-map-payload"
    assert runtime.memory_cache_stats().items == 1
    assert runtime.release_product(product)
    assert runtime.memory_cache_stats().items == 0
    assert product.exists()
    assert product.read_bytes() == b"static-map-payload"
