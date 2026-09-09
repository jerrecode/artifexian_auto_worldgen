from __future__ import annotations

import heapq
import json

import numpy as np

from worldgen.local_hydrology import (
    _D16,
    LocalHydrologySolver,
    LocalHydrologySpec,
    _flow_d16_open,
    _patch_geometry,
    _priority_flood_open,
    _resolved_elevation_patch,
)
from worldgen.planet_tiles import (
    PlanetTilePyramid,
    TileKey,
    TilePyramidSpec,
    tile_geometry,
)


def _world(root):
    h, w = 18, 36
    lat = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    lon = -180.0 + (np.arange(w) + 0.5) * 360.0 / w
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    elevation = (
        1.3
        + 0.9 * np.cos(2.0 * np.pi * xx / w)
        + 0.35 * np.cos(np.pi * (yy + 0.5) / h)
    ).astype(np.float32)
    runoff = (
        250.0
        + 800.0 * np.clip(np.cos(np.deg2rad(lat))[:, None], 0.0, 1.0)
        + 40.0 * np.sin(2 * np.pi * xx / w)
    ).astype(np.float32)
    rivers = np.zeros((h, w), dtype=bool)
    rivers[:, w // 2] = True
    precipitation = (runoff * 1.7).astype(np.float32)
    annual_temperature = (22.0 - 0.32 * np.abs(lat)[:, None]).astype(np.float32)
    np.savez(
        root / "world_arrays.npz",
        lat=lat,
        lon=lon,
        elevation_km=elevation,
        runoff_mm_year=runoff,
        annual_precipitation_mm=precipitation,
        annual_temperature_c=annual_temperature,
        rivers=rivers,
    )
    (root / "world.json").write_text(
        json.dumps({"seed": 1234, "astronomy": {"planet": {"radius_earth": 1.0}}}),
        encoding="utf-8",
    )


def test_open_priority_flood_never_wraps_or_changes_perimeter_seed_heights():
    z = np.array(
        [
            [9.0, 8.0, 7.0, 6.0, 5.0],
            [8.0, 4.0, 4.0, 4.0, 6.0],
            [7.0, 4.0, 1.0, 4.0, 7.0],
            [6.0, 4.0, 4.0, 4.0, 8.0],
            [5.0, 6.0, 7.0, 8.0, 9.0],
        ]
    )
    filled = _priority_flood_open(z, np.zeros_like(z, dtype=bool), epsilon_m=0.01)
    np.testing.assert_array_equal(filled[0], z[0])
    np.testing.assert_array_equal(filled[-1], z[-1])
    np.testing.assert_array_equal(filled[:, 0], z[:, 0])
    np.testing.assert_array_equal(filled[:, -1], z[:, -1])
    assert filled[2, 2] > z[2, 2]


def test_halo_resolved_elevation_core_matches_authoritative_tile(tmp_path):
    _world(tmp_path)
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=20, elevation_detail_strength=0.35),
    )
    key = TileKey("px", 4, 7, 6)
    halo = 5
    geom = _patch_geometry(key, pyramid.spec.tile_size, halo)
    patch = _resolved_elevation_patch(pyramid, key, geom)
    core = patch[halo : halo + 21, halo : halo + 21]
    tile = np.asarray(pyramid.load_field(key, "elevation_m"), dtype=float)
    np.testing.assert_allclose(core, tile, rtol=0.0, atol=2e-4)


def test_d16_router_uses_intermediate_directions_and_remains_downhill():
    h = w = 21
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float64),
        np.arange(w, dtype=np.float64),
        indexing="ij",
    )
    scale = 1.0e-4
    xyz = np.stack(
        (
            (xx - w // 2) * scale,
            (yy - h // 2) * scale,
            np.ones((h, w), dtype=np.float64),
        ),
        axis=-1,
    )
    xyz /= np.linalg.norm(xyz, axis=-1, keepdims=True)
    # Continuous downhill direction is approximately (+2 rows,+1 col), which
    # D8 cannot represent but the queen+knight stencil can.
    z = 5000.0 - 100.0 * yy - 50.0 * xx
    ocean = np.zeros((h, w), dtype=bool)
    receiver, code16, _slope = _flow_d16_open(
        z, ocean, xyz, 1.0e6
    )
    center = (h // 2, w // 2)
    direction = int(code16[center])
    assert direction >= 8
    assert _D16[direction] == (2, 1)

    flat_z = z.ravel()
    active = receiver >= 0
    sources = np.flatnonzero(active)
    assert np.all(flat_z[receiver[active]] < flat_z[sources])


def test_local_hydrology_returns_bounded_d16_and_inherited_runoff(tmp_path):
    _world(tmp_path)
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=24, elevation_detail_strength=0.4),
    )
    solver = LocalHydrologySolver(
        pyramid, spec=LocalHydrologySpec(halo_cells=8, stream_quantile=0.96)
    )
    result = solver.solve(TileKey("px", 5, 15, 14))
    expected = (25, 25)
    assert result.filled_elevation_m.shape == expected
    assert result.flow_direction_d8.shape == expected
    assert result.flow_direction_d16.shape == expected
    assert result.flow_angle_rad.shape == expected
    assert result.meander_potential.shape == expected
    assert result.runoff_mm_year.shape == expected
    assert result.drainage_area_km2.shape == expected
    assert result.discharge_index.shape == expected
    assert result.streams.shape == expected
    assert np.all((result.flow_direction_d8 >= -1) & (result.flow_direction_d8 <= 7))
    assert np.all((result.flow_direction_d16 >= -1) & (result.flow_direction_d16 < len(_D16)))
    assert np.all((result.meander_potential >= 0.0) & (result.meander_potential <= 1.0 + 1e-6))
    assert np.all(result.runoff_mm_year >= 0.0)
    assert np.all(result.drainage_area_km2 >= 0.0)
    assert np.all((result.discharge_index >= 0.0) & (result.discharge_index <= 1.0 + 1e-6))
    assert result.metadata["runoff_semantics"] == "inherited global runoff_mm_year"
    flow_meta = result.metadata["flow_direction_semantics"]
    assert flow_meta["not_global_flow_to"] is True
    assert flow_meta["strictly_downhill_receivers"] is True
    assert flow_meta["long_move_ridge_jump_guard"] is True
    assert "D16" in flow_meta["type"]


def test_inherited_major_river_is_soft_corridor_not_stamped_centerline(tmp_path):
    _world(tmp_path)
    pyramid = PlanetTilePyramid(tmp_path, spec=TilePyramidSpec(tile_size=32))
    solver = LocalHydrologySolver(
        pyramid,
        spec=LocalHydrologySpec(
            halo_cells=6,
            stream_quantile=0.94,
            major_river_corridor_cells=7,
        ),
    )
    # Root +X includes longitude around 0 degrees where the synthetic parent
    # river lies. The refined channel only needs to remain in its corridor.
    result = solver.solve(TileKey("px", 0, 0, 0))
    inherited = np.asarray(result.inherited_major_river, dtype=bool)
    streams = np.asarray(result.streams, dtype=bool)
    assert np.any(inherited)
    assert np.any(streams)
    from scipy import ndimage

    corridor = ndimage.binary_dilation(inherited, iterations=7)
    assert np.count_nonzero(streams & corridor) > 0
    assert result.metadata["major_river_guide_cells"] >= np.count_nonzero(inherited)
    assert "soft corridor" in result.metadata["boundary_semantics"]


def test_local_hydrology_cache_is_sparse_and_reusable(tmp_path):
    _world(tmp_path)
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=16, elevation_detail_strength=0.2),
    )
    solver = LocalHydrologySolver(
        pyramid, spec=LocalHydrologySpec(halo_cells=4, stream_quantile=0.95)
    )
    key = TileKey("pz", 7, 61, 70)
    first = solver.solve(key)
    assert solver._metadata_path(key).exists()
    sibling = TileKey("pz", 7, 62, 70)
    assert not solver._metadata_path(sibling).exists()
    second = solver.solve(key)
    np.testing.assert_array_equal(first.discharge_index, second.discharge_index)
    np.testing.assert_array_equal(first.flow_direction_d8, second.flow_direction_d8)
    np.testing.assert_array_equal(first.flow_direction_d16, second.flow_direction_d16)
    np.testing.assert_array_equal(first.streams, second.streams)


def test_final_terrain_reroute_uses_halo_not_core_perimeter_outlets(tmp_path):
    from worldgen.planet_tiles import tile_geometry

    _world(tmp_path)
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(
            tile_size=32,
            elevation_detail_strength=0.0,
            maximum_level=4,
        ),
    )
    solver = LocalHydrologySolver(
        pyramid,
        spec=LocalHydrologySpec(
            halo_cells=6,
            stream_quantile=0.94,
        ),
    )
    key = TileKey("px", 1, 0, 0)
    geom = tile_geometry(key, 32)
    final_elevation = np.asarray(
        pyramid._sample_source_field("elevation_m", geom),
        dtype=np.float64,
    )
    result = solver.solve_elevation(key, final_elevation)

    assert result.metadata["patch_shape"] == [45, 45]
    assert result.metadata["core_shape"] == [33, 33]
    assert "not an artificial outlet" in result.metadata["boundary_semantics"]

    code = np.asarray(result.flow_direction_d16)
    perimeter = np.concatenate(
        (code[0, :], code[-1, :], code[1:-1, 0], code[1:-1, -1])
    )
    # Ocean cells may remain outlets, but a land tile edge is no longer forced
    # wholesale to -1 merely because it is the exported core boundary.
    assert np.any(perimeter >= 0)


def test_routing_metrics_active_mask_counts_only_selected_source_cells():
    z = np.array(
        [
            [9.0, 8.0, 7.0, 6.0],
            [8.0, 7.0, 6.0, 5.0],
            [7.0, 6.0, 5.0, 4.0],
            [6.0, 5.0, 4.0, 3.0],
        ]
    )
    yy, xx = np.mgrid[:4, :4]
    xyz = np.stack(
        (
            (xx - 1.5) * 1e-4,
            (yy - 1.5) * 1e-4,
            np.ones((4, 4)),
        ),
        axis=-1,
    )
    xyz /= np.linalg.norm(xyz, axis=-1, keepdims=True)
    ocean = np.zeros((4, 4), dtype=bool)
    receiver, code16, _slope = _flow_d16_open(z, ocean, xyz, 1e6)
    streams = code16 >= 0
    discharge = np.ones((4, 4), dtype=np.float64)
    active_mask = np.zeros((4, 4), dtype=bool)
    active_mask[1:3, 1:3] = True

    from worldgen.local_hydrology import _routing_metrics

    metrics = _routing_metrics(
        receiver,
        code16,
        streams,
        discharge,
        xyz,
        1e6,
        active_mask=active_mask,
    )
    assert metrics["stream_direction_count"] <= 4


def test_routing_metrics_eighth_moment_catches_balanced_axis_diagonal_lattice():
    from worldgen.local_hydrology import _routing_metrics

    h = w = 24
    code = np.full((h, w), -1, dtype=np.int8)
    streams = np.zeros((h, w), dtype=bool)

    # Equal populations of east (0 degrees) and southeast (45 degrees).
    code[2:10, 2:-2] = 4
    code[14:22, 2:-2] = 7
    streams[code >= 0] = True

    receiver = np.full(h * w, -1, dtype=np.int64)
    discharge = np.ones((h, w), dtype=np.float64)
    yy, xx = np.mgrid[:h, :w]
    xyz = np.stack(
        (
            (xx - w / 2) * 1e-5,
            (yy - h / 2) * 1e-5,
            np.ones((h, w)),
        ),
        axis=-1,
    )
    xyz /= np.linalg.norm(xyz, axis=-1, keepdims=True)

    metrics = _routing_metrics(
        receiver,
        code,
        streams,
        discharge,
        xyz,
        1e6,
    )
    assert metrics["directional_fourfold_anisotropy"] < 0.05
    assert metrics["directional_eighth_anisotropy"] > 0.95


def test_accumulation_backend_matches_python_recurrence_when_numba_available():
    from worldgen.local_hydrology import (
        _accumulate_topological_numba,
        _accumulate_topological_python,
    )

    # 0->2, 1->2, 2->3, 3 outlet. Order is upstream to downstream.
    order = np.array([0, 1, 2, 3], dtype=np.int64)
    receiver = np.array([2, 2, 3, -1], dtype=np.int64)
    drainage_py = np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float64)
    discharge_py = np.array([10.0, 20.0, 40.0, 80.0], dtype=np.float64)
    _accumulate_topological_python(
        order,
        receiver,
        drainage_py,
        discharge_py,
    )
    np.testing.assert_allclose(drainage_py, [1.0, 2.0, 7.0, 15.0])
    np.testing.assert_allclose(discharge_py, [10.0, 20.0, 70.0, 150.0])

    if _accumulate_topological_numba is not None:
        drainage_nb = np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float64)
        discharge_nb = np.array([10.0, 20.0, 40.0, 80.0], dtype=np.float64)
        _accumulate_topological_numba(
            order,
            receiver,
            drainage_nb,
            discharge_nb,
        )
        np.testing.assert_array_equal(drainage_nb, drainage_py)
        np.testing.assert_array_equal(discharge_nb, discharge_py)


def test_priority_flood_optional_numba_matches_independent_python_reference():
    def reference(elevation, ocean, epsilon):
        z = np.asarray(elevation, dtype=np.float64).copy()
        oc = np.asarray(ocean, dtype=bool)
        h, w = z.shape
        visited = oc.copy()

        seed = np.zeros_like(oc)
        land = ~oc
        # Coastal land.
        for dy, dx in (
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1),
        ):
            sy0 = max(0, -dy)
            sy1 = min(h, h - dy)
            sx0 = max(0, -dx)
            sx1 = min(w, w - dx)
            ty0, ty1 = sy0 + dy, sy1 + dy
            tx0, tx1 = sx0 + dx, sx1 + dx
            seed[sy0:sy1, sx0:sx1] |= oc[ty0:ty1, tx0:tx1]
        seed &= land
        seed[0, :] |= land[0, :]
        seed[-1, :] |= land[-1, :]
        seed[:, 0] |= land[:, 0]
        seed[:, -1] |= land[:, -1]

        heap = []
        ys, xs = np.where(seed & ~visited)
        for y, x in zip(ys.tolist(), xs.tolist()):
            visited[y, x] = True
            heapq.heappush(heap, (float(z[y, x]), y, x))

        while heap:
            cur, y, x = heapq.heappop(heap)
            for dy, dx in (
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1), (0, 1),
                (1, -1), (1, 0), (1, 1),
            ):
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w or visited[ny, nx]:
                    continue
                visited[ny, nx] = True
                nz = float(z[ny, nx])
                if nz <= cur:
                    nz = cur + epsilon
                    z[ny, nx] = nz
                heapq.heappush(heap, (nz, ny, nx))
        return z

    elevation = np.array(
        [
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            [10.0,  8.0,  8.0,  8.0,  8.0, 10.0],
            [10.0,  8.0,  2.0,  2.0,  8.0, 10.0],
            [10.0,  8.0,  2.0,  1.0,  8.0, 10.0],
            [10.0,  8.0,  8.0,  8.0,  8.0, 10.0],
            [ 5.0,  6.0,  7.0,  8.0,  9.0, 10.0],
        ],
        dtype=np.float64,
    )
    ocean = np.zeros_like(elevation, dtype=bool)
    expected = reference(elevation, ocean, 0.01)
    actual = _priority_flood_open(elevation, ocean, epsilon_m=0.01)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)



def test_d16_meander_steering_breaks_long_axis_lock_without_uphill_flow():
    from worldgen.local_hydrology import _routing_metrics

    h = w = 96
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float64),
        np.arange(w, dtype=np.float64),
        indexing="ij",
    )
    scale = 2.0e-5
    xyz = np.stack(
        (
            (xx - w / 2.0) * scale,
            (yy - h / 2.0) * scale,
            np.ones((h, w), dtype=np.float64),
        ),
        axis=-1,
    )
    xyz /= np.linalg.norm(xyz, axis=-1, keepdims=True)

    # Primarily eastward descent, with a much smaller southward component.  Both
    # east and southeast remain downhill, allowing deterministic steering to
    # choose a gently oscillating path instead of an axis-locked reach.
    z = 2000.0 - 2.0 * xx - 0.30 * yy
    ocean = np.zeros((h, w), dtype=bool)
    preferred = 0.62 * np.sin(2.0 * np.pi * xx / 18.0)
    steer = np.full((h, w), 0.84, dtype=np.float64)

    receiver, code16, _slope = _flow_d16_open(
        z,
        ocean,
        xyz,
        1.0e6,
        preferred_angle_rad=preferred,
        steering_weight=steer,
    )
    streams = code16 >= 0
    metrics = _routing_metrics(
        receiver,
        code16,
        streams,
        np.ones((h, w), dtype=np.float64),
        xyz,
        1.0e6,
    )

    flat_z = z.ravel()
    active = receiver >= 0
    sources = np.flatnonzero(active)
    assert np.all(flat_z[receiver[active]] < flat_z[sources])
    assert metrics["max_straight_run_cells"] < 80
    assert metrics["stream_turn_fraction_gt10deg"] > 0.02



def test_adaptive_meander_variants_are_deterministic_bounded_and_distinct():
    from worldgen.local_hydrology import _meander_phase

    h, w = 40, 48
    yy, xx = np.meshgrid(
        np.linspace(-0.08, 0.08, h),
        np.linspace(-0.10, 0.10, w),
        indexing="ij",
    )
    xyz = np.stack((xx, yy, np.ones_like(xx)), axis=-1)
    xyz /= np.linalg.norm(xyz, axis=-1, keepdims=True)
    q = np.clip((xx + 0.10) / 0.20, 0.0, 1.0)

    kwargs = dict(
        radius_m=6.4e6,
        discharge_index=q,
        seed=2026090707,
        minimum_wavelength_m=12_000.0,
    )
    base = _meander_phase(xyz, variant=0, **kwargs)
    v1 = _meander_phase(xyz, variant=1, **kwargs)
    v1_again = _meander_phase(xyz, variant=1, **kwargs)
    v2 = _meander_phase(xyz, variant=2, **kwargs)

    np.testing.assert_array_equal(v1, v1_again)
    assert np.isfinite(base).all()
    assert np.isfinite(v1).all()
    assert np.isfinite(v2).all()
    assert float(np.max(np.abs(v1))) <= 1.0 + 1e-12
    assert float(np.max(np.abs(v2))) <= 1.0 + 1e-12
    assert not np.array_equal(base, v1)
    assert not np.array_equal(v1, v2)


def test_routing_metadata_reports_resolved_adaptive_wavelength_floor(tmp_path):
    from worldgen.planet_tiles import approximate_meters_per_sample

    _world(tmp_path)
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(
            tile_size=32,
            elevation_detail_strength=0.0,
            maximum_level=4,
        ),
    )
    solver = LocalHydrologySolver(
        pyramid,
        spec=LocalHydrologySpec(halo_cells=6, stream_quantile=0.94),
    )
    key = TileKey("px", 1, 0, 0)
    result = solver.solve(key)
    expected = 4.5 * approximate_meters_per_sample(
        pyramid.planet_radius_m,
        key.level,
        pyramid.spec.tile_size,
    )
    assert result.metadata["adaptive_meander_min_wavelength_m"] == pytest.approx(
        expected
    )
    assert result.metadata["adaptive_meander_attempt"] in (0, 1, 2)
    assert result.metadata["routing_candidate_metrics"]



def test_d16_corrective_steering_floor_preserves_strict_downhill_flow():
    h = w = 64
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float64),
        np.arange(w, dtype=np.float64),
        indexing="ij",
    )
    scale = 2.0e-5
    xyz = np.stack(
        (
            (xx - w / 2.0) * scale,
            (yy - h / 2.0) * scale,
            np.ones((h, w), dtype=np.float64),
        ),
        axis=-1,
    )
    xyz /= np.linalg.norm(xyz, axis=-1, keepdims=True)
    z = 2500.0 - 2.8 * xx - 0.22 * yy
    preferred = 0.72 * np.sin(2.0 * np.pi * xx / 16.0)
    steer = np.full((h, w), 0.92, dtype=np.float64)

    receiver, _code, _slope = _flow_d16_open(
        z,
        np.zeros((h, w), dtype=bool),
        xyz,
        1.0e6,
        preferred_angle_rad=preferred,
        steering_weight=steer,
        steering_score_floor=0.018,
    )
    flat = z.ravel()
    active = receiver >= 0
    sources = np.flatnonzero(active)
    assert np.all(flat[receiver[active]] < flat[sources])

    with pytest.raises(ValueError):
        _flow_d16_open(
            z,
            np.zeros((h, w), dtype=bool),
            xyz,
            1.0e6,
            preferred_angle_rad=preferred,
            steering_weight=steer,
            steering_score_floor=-0.01,
        )
