from __future__ import annotations

import numpy as np
import pytest

from worldgen.grid import SphereGrid
from worldgen.resources import _near
from worldgen.spatial_naturalism import (
    irregular_blob_field,
    irregular_near,
    major_water_mask,
    marine_thermal_distance,
)
from worldgen.tectonics import _blob_field


def _azimuthal_boundary_variation(
    grid: SphereGrid, mask: np.ndarray, center: np.ndarray
) -> float:
    pts = np.asarray(grid.xyz[mask], float)
    c = np.asarray(center, float)
    c /= np.linalg.norm(c)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(c, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    a = np.cross(ref, c)
    a /= np.linalg.norm(a)
    b = np.cross(c, a)
    b /= np.linalg.norm(b)
    dot = np.clip(pts @ c, -1.0, 1.0)
    radius = np.arccos(dot)
    az = np.arctan2(pts @ b, pts @ a)
    bins = np.linspace(-np.pi, np.pi, 25)
    outer = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        rr = radius[(az >= lo) & (az < hi)]
        if rr.size:
            outer.append(float(np.max(rr)))
    outer = np.asarray(outer)
    return float(np.std(outer) / max(np.mean(outer), 1e-12))


def test_hotspot_province_is_not_a_circular_gaussian_disc():
    grid = SphereGrid(240, 120)
    center = np.array([1.0, 0.0, 0.0])
    field = _blob_field(
        grid,
        np.asarray([center]),
        np.asarray([7.0]),
        np.asarray([1.0]),
    )
    footprint = field >= 0.42 * float(np.max(field))
    assert int(np.sum(footprint)) > 20
    assert _azimuthal_boundary_variation(grid, footprint, center) > 0.055
    np.testing.assert_allclose(
        field,
        irregular_blob_field(
            grid, np.asarray([center]), np.asarray([7.0]), np.asarray([1.0])
        ),
    )


def test_resource_proximity_uses_irregular_geodesic_margin():
    grid = SphereGrid(240, 120)
    source = np.zeros(grid.shape, dtype=bool)
    source[grid.height // 2, grid.width // 2] = True
    near = _near(source, grid, 1200.0)
    expected = irregular_near(source, grid, 1200.0)
    np.testing.assert_array_equal(near, expected)
    center = np.asarray(grid.xyz[grid.height // 2, grid.width // 2], float)
    assert int(np.sum(near)) > 20
    assert _azimuthal_boundary_variation(grid, near, center) > 0.025


def test_tiny_lake_is_not_promoted_to_planetary_marine_reservoir():
    grid = SphereGrid(160, 80)
    water = np.zeros(grid.shape, dtype=bool)
    water[grid.height // 2, grid.width // 2] = True
    assert not np.any(major_water_mask(grid, water))

    distance = marine_thermal_distance(
        grid,
        water,
        np.zeros(grid.shape, dtype=np.float32),
        inland_scale_km=1800.0,
    )
    far = np.ones(grid.shape, dtype=bool)
    y, x = grid.height // 2, grid.width // 2
    far[max(0, y - 2):y + 3, x - 2:x + 3] = False
    assert float(np.percentile(distance[far], 5)) > 4500.0


def test_large_ocean_is_retained_as_marine_thermal_source():
    grid = SphereGrid(160, 80)
    water = np.zeros(grid.shape, dtype=bool)
    water[:, : grid.width // 2] = True
    major = major_water_mask(grid, water)
    assert np.array_equal(major, water)

    distance = marine_thermal_distance(
        grid,
        water,
        np.zeros(grid.shape, dtype=np.float32),
        inland_scale_km=1800.0,
    )
    assert float(np.max(distance[water])) == 0.0
    assert float(np.max(distance[~water])) > 0.0


def test_spatial_naturalism_is_deterministic():
    grid = SphereGrid(96, 48)
    centers = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    sigmas = np.asarray([6.0, 10.0])
    weights = np.asarray([1.0, 0.7])
    a = irregular_blob_field(grid, centers, sigmas, weights)
    b = irregular_blob_field(grid, centers, sigmas, weights)
    np.testing.assert_array_equal(a, b)



def test_irregular_near_rejects_wrong_shape_even_when_source_is_empty():
    grid = SphereGrid(32, 16)
    with pytest.raises(ValueError, match="mask.*shape"):
        irregular_near(np.zeros((8, 64), dtype=bool), grid, 100.0)


@pytest.mark.parametrize("radius", [np.nan, np.inf, -1.0])
def test_irregular_near_rejects_invalid_radius(radius):
    grid = SphereGrid(32, 16)
    source = np.zeros(grid.shape, dtype=bool)
    source[8, 16] = True
    with pytest.raises(ValueError, match="km.*finite and non-negative"):
        irregular_near(source, grid, radius)


def test_zero_radius_irregular_near_is_exact_source_mask():
    grid = SphereGrid(32, 16)
    source = np.zeros(grid.shape, dtype=bool)
    source[8, 16] = True
    np.testing.assert_array_equal(irregular_near(source, grid, 0.0), source)


@pytest.mark.parametrize(
    "centers,sigmas,weights",
    [
        (np.asarray([[0.0, 0.0, 0.0]]), np.asarray([2.0]), np.asarray([1.0])),
        (np.asarray([[np.nan, 0.0, 1.0]]), np.asarray([2.0]), np.asarray([1.0])),
        (np.asarray([[1.0, 0.0, 0.0]]), np.asarray([0.0]), np.asarray([1.0])),
        (np.asarray([[1.0, 0.0, 0.0]]), np.asarray([np.nan]), np.asarray([1.0])),
        (np.asarray([[1.0, 0.0, 0.0]]), np.asarray([2.0]), np.asarray([np.nan])),
    ],
)
def test_irregular_blob_rejects_invalid_geometry(centers, sigmas, weights):
    grid = SphereGrid(32, 16)
    with pytest.raises(ValueError, match="center|sigma|weight"):
        irregular_blob_field(grid, centers, sigmas, weights)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_global_fraction": np.nan},
        {"min_global_fraction": -0.1},
        {"min_global_fraction": 1.1},
        {"min_component_fraction_of_water": np.nan},
        {"min_component_fraction_of_water": -0.1},
        {"min_component_fraction_of_water": 1.1},
    ],
)
def test_major_water_mask_rejects_invalid_fraction_thresholds(kwargs):
    grid = SphereGrid(32, 16)
    water = np.zeros(grid.shape, dtype=bool)
    water[:, :16] = True
    with pytest.raises(ValueError, match="fraction.*\[0, 1\]"):
        major_water_mask(grid, water, **kwargs)


@pytest.mark.parametrize("scale", [np.nan, np.inf, -np.inf, 0.0, -1.0])
def test_marine_thermal_distance_rejects_invalid_inland_scale(scale):
    grid = SphereGrid(32, 16)
    water = np.zeros(grid.shape, dtype=bool)
    with pytest.raises(ValueError, match="inland_scale_km.*finite and positive"):
        marine_thermal_distance(
            grid, water, np.zeros(grid.shape), inland_scale_km=scale
        )


def test_marine_thermal_distance_rejects_nonfinite_elevation():
    grid = SphereGrid(32, 16)
    water = np.zeros(grid.shape, dtype=bool)
    elevation = np.zeros(grid.shape)
    elevation[3, 5] = np.nan
    with pytest.raises(ValueError, match="elevation_km.*finite"):
        marine_thermal_distance(
            grid, water, elevation, inland_scale_km=1000.0
        )
