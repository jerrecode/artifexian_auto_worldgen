from __future__ import annotations

import json

import numpy as np

from worldgen.local_orography import (
    LocalOrographicDownscaler,
    OrographicDownscalingSpec,
    cube_vertex_area_weights,
    downscale_wind,
    edge_anchor_taper,
    redistribute_precipitation,
    terrain_frame,
)
from worldgen.planet_tiles import PlanetTilePyramid, TileKey, TilePyramidSpec, tile_geometry


def _write_world(root) -> None:
    h, w = 8, 16
    lat = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    lon = -180.0 + (np.arange(w) + 0.5) * 360.0 / w
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    elevation = (
        1.1 * np.sin(2.0 * np.pi * xx / w)
        + 0.35 * np.cos(np.pi * (yy + 0.5) / h)
    ).astype(np.float32)
    wind_u = np.full((12, h, w), 7.0, dtype=np.float32)
    wind_v = np.zeros((12, h, w), dtype=np.float32)
    precipitation = np.empty((12, h, w), dtype=np.float32)
    for month in range(12):
        precipitation[month] = 70.0 + 5.0 * np.cos(2.0 * np.pi * month / 12.0)
    np.savez(
        root / "world_arrays.npz",
        lat=lat,
        lon=lon,
        elevation_km=elevation,
        wind_u_monthly=wind_u,
        wind_v_monthly=wind_v,
        precipitation_mm_monthly=precipitation,
    )
    (root / "world.json").write_text(
        json.dumps({"seed": 314, "astronomy": {"planet": {"radius_earth": 1.0}}}),
        encoding="utf-8",
    )


def test_edge_anchor_taper_is_zero_on_perimeter_and_one_interior():
    taper = edge_anchor_taper((17, 17), 3)
    assert np.all(taper[0] == 0.0)
    assert np.all(taper[-1] == 0.0)
    assert np.all(taper[:, 0] == 0.0)
    assert np.all(taper[:, -1] == 0.0)
    assert taper[8, 8] == 1.0
    assert np.all((0.0 <= taper) & (taper <= 1.0))


def test_cube_vertex_area_weights_are_positive_normalized_and_seam_intrinsic():
    geom = tile_geometry(TileKey("px", 3, 2, 3), 16)
    weights = cube_vertex_area_weights(geom.xyz)
    assert weights.shape == (17, 17)
    assert np.all(weights > 0)
    assert abs(float(weights.sum()) - 1.0) < 1e-14


def test_constant_elevation_sphere_has_nearly_radial_central_normal():
    geom = tile_geometry(TileKey("pz", 3, 3, 3), 32)
    elevation = np.zeros((33, 33), dtype=np.float64)
    normal, slope, grad_e, grad_s = terrain_frame(geom.xyz, elevation, 6_371_000.0)
    centre = (16, 16)
    assert np.dot(normal[centre], geom.xyz[centre]) > 0.9999
    assert slope[centre] < 0.5
    assert abs(grad_e[centre]) < 0.01
    assert abs(grad_s[centre]) < 0.01


def test_wind_blocking_reduces_upslope_cross_barrier_component_only_interior():
    shape = (17, 17)
    base_u = np.full((12, *shape), 10.0)
    base_v = np.zeros_like(base_u)
    grad_e = np.full(shape, 0.35)
    grad_s = np.zeros(shape)
    taper = edge_anchor_taper(shape, 3)
    u, v = downscale_wind(
        base_u,
        base_v,
        grad_e,
        grad_s,
        taper,
        spec=OrographicDownscalingSpec(),
    )
    assert np.allclose(u[:, 0, :], base_u[:, 0, :])
    assert np.allclose(u[:, -1, :], base_u[:, -1, :])
    assert np.all(u[:, 8, 8] < base_u[:, 8, 8])
    assert np.allclose(v, 0.0)


def test_precipitation_redistribution_preserves_weighted_total_and_edges():
    shape = (17, 17)
    base = np.full((12, *shape), 100.0)
    wind_u = np.full_like(base, 8.0)
    wind_v = np.zeros_like(base)
    x = np.linspace(-0.25, 0.25, shape[1])
    grad_e = np.broadcast_to(x[None, :], shape).copy()
    grad_s = np.zeros(shape)
    taper = edge_anchor_taper(shape, 3)
    weights = np.ones(shape, dtype=np.float64)
    weights /= weights.sum()
    local = redistribute_precipitation(
        base,
        wind_u,
        wind_v,
        grad_e,
        grad_s,
        taper,
        weights,
        spec=OrographicDownscalingSpec(),
    )
    np.testing.assert_allclose(local[:, 0, :], base[:, 0, :], rtol=0, atol=0)
    np.testing.assert_allclose(local[:, -1, :], base[:, -1, :], rtol=0, atol=0)
    assert float(local[:, :, 12:].mean()) > float(local[:, :, :5].mean())
    before = np.sum(base * weights[None, :, :], axis=(-2, -1))
    after = np.sum(local * weights[None, :, :], axis=(-2, -1))
    np.testing.assert_allclose(after, before, rtol=2e-7, atol=2e-5)


def test_local_orographic_downscaler_generates_mass_neutral_boundary_anchored_fields(tmp_path):
    _write_world(tmp_path)
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=16, elevation_detail_strength=0.3, maximum_level=8),
    )
    key = TileKey("px", 4, 6, 7)
    downscaler = LocalOrographicDownscaler(pyramid)
    paths = downscaler.generate(key)
    assert set(paths) == {
        "terrain_normal_xyz",
        "slope_deg",
        "wind_u_monthly",
        "wind_v_monthly",
        "precipitation_mm_monthly",
        "annual_precipitation_mm",
    }
    for path in paths.values():
        assert path.exists()

    geom = tile_geometry(key, 16)
    weights = cube_vertex_area_weights(geom.xyz)
    base_p = np.asarray(pyramid._sample_source_field("precipitation_mm_monthly", geom), dtype=float)
    base_u = np.asarray(pyramid._sample_source_field("wind_u_monthly", geom), dtype=float)
    local_p = np.load(paths["precipitation_mm_monthly"])
    local_u = np.load(paths["wind_u_monthly"])

    np.testing.assert_array_equal(local_p[:, 0, :], base_p[:, 0, :].astype(np.float32))
    np.testing.assert_array_equal(local_p[:, -1, :], base_p[:, -1, :].astype(np.float32))
    np.testing.assert_array_equal(local_u[:, 0, :], base_u[:, 0, :].astype(np.float32))
    before = np.sum(base_p * weights[None, :, :], axis=(-2, -1))
    after = np.sum(local_p * weights[None, :, :], axis=(-2, -1))
    np.testing.assert_allclose(after, before, rtol=3e-7, atol=1e-4)
    annual = np.load(paths["annual_precipitation_mm"])
    np.testing.assert_allclose(annual, local_p.sum(axis=0), rtol=1e-6, atol=1e-4)

    # Repeated generation must be a pure cache hit and preserve bytes.
    before_bytes = paths["precipitation_mm_monthly"].read_bytes()
    again = downscaler.generate(key)
    assert again == paths
    assert paths["precipitation_mm_monthly"].read_bytes() == before_bytes
