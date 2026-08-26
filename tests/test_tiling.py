import numpy as np

from worldgen.grid import SphereGrid
from worldgen.tiling import SphericalTiler, apply_tiled, auto_tile_shape


def test_auto_tile_shape_is_genuinely_two_dimensional_for_large_world():
    ch, cw = auto_tile_shape((768, 1536), np.float32, target_mb=8.0, arrays_in_flight=8)
    assert 1 <= ch < 768
    assert 1 <= cw < 1536


def test_spherical_tile_halo_wraps_longitude_seam():
    grid = SphereGrid(16, 8)
    field = np.arange(8 * 16, dtype=np.int32).reshape(8, 16)
    tile = next(iter(SphericalTiler(grid, chunk_shape=(4, 4), halo_cells=1)))
    expanded = tile.extract(field)
    # First tile starts at x=0, so its left halo is global longitude column -1.
    assert expanded[1, 0] == field[0, -1]


def test_spherical_tile_halo_crosses_north_pole_antipodally():
    grid = SphereGrid(16, 8)
    field = np.arange(8 * 16, dtype=np.int32).reshape(8, 16)
    tile = next(iter(SphericalTiler(grid, chunk_shape=(2, 4), halo_cells=1)))
    expanded = tile.extract(field)
    # Virtual row -1 reflects to row 0 and rotates longitude by 180 degrees.
    assert expanded[0, 1] == field[0, 8]


def test_physical_halo_is_at_least_requested_north_south_radius():
    grid = SphereGrid(128, 64)
    radius_km = 1.7 * grid.dy_km
    tile = next(iter(SphericalTiler(grid, chunk_shape=(16, 16), halo_km=radius_km)))
    assert tile.halo_y >= 2
    assert tile.halo_x >= 1


def test_tile_order_and_coverage_are_deterministic():
    grid = SphereGrid(20, 10)
    tiler = SphericalTiler(grid, chunk_shape=(4, 6))
    tiles = list(tiler)
    assert [t.index for t in tiles] == list(range(len(tiles)))
    cover = np.zeros((10, 20), dtype=np.int16)
    for tile in tiles:
        cover[tile.core] += 1
    assert np.all(cover == 1)
    assert tiler.count() == len(tiles)


def test_apply_tiled_identity_preserves_array_with_seam_and_pole_halos():
    grid = SphereGrid(32, 16)
    rng = np.random.default_rng(123)
    field = rng.normal(size=(16, 32)).astype(np.float32)
    tiler = SphericalTiler(grid, chunk_shape=(5, 7), halo_cells=(2, 3))
    result = apply_tiled(field, tiler, lambda a: a)
    assert np.array_equal(result, field)


def test_apply_tiled_local_kernel_matches_global_for_pointwise_operation():
    grid = SphereGrid(40, 20)
    field = np.arange(800, dtype=np.float32).reshape(20, 40)
    tiler = SphericalTiler(grid, chunk_shape=(6, 9), halo_cells=2)
    result = apply_tiled(field, tiler, lambda a: a * a + 3.0)
    expected = field * field + 3.0
    assert np.array_equal(result, expected)
