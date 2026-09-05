import numpy as np
import pytest

from worldgen.grid import SphereGrid, distance_to, spherical_voronoi_ids
from worldgen.hydrology_natural_routing import flow_directions_continuous_local


@pytest.mark.parametrize(
    "width, height",
    [
        (0, 16),
        (-1, 16),
        (16, 0),
        (16, -1),
        (1.5, 16),
        (16, 2.5),
        (True, 16),
        (16, False),
    ],
)
def test_sphere_grid_rejects_invalid_dimensions(width, height):
    with pytest.raises((TypeError, ValueError), match="width|height|dimensions"):
        SphereGrid(width, height)


@pytest.mark.parametrize("radius", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_sphere_grid_rejects_nonfinite_or_nonpositive_radius(radius):
    with pytest.raises(ValueError, match="radius_km.*finite and positive"):
        SphereGrid(16, 8, radius_km=radius)


def test_weighted_fraction_rejects_broadcastable_but_wrong_shape():
    grid = SphereGrid(16, 8)
    # A 1-D longitude mask used to broadcast over all latitude rows silently.
    with pytest.raises(ValueError, match="shape"):
        grid.weighted_fraction(np.ones(16, dtype=bool))


def test_weighted_quantile_rejects_wrong_shape_even_with_same_element_count():
    grid = SphereGrid(16, 8)
    values = np.arange(128, dtype=float).reshape(16, 8)
    with pytest.raises(ValueError, match="shape"):
        grid.weighted_quantile(values, 0.5)


def test_distance_to_rejects_wrong_shape_before_cache_lookup():
    grid = SphereGrid(16, 8)
    with pytest.raises(ValueError, match="shape"):
        distance_to(np.zeros((4, 32), dtype=bool), grid)


@pytest.mark.parametrize(
    "seeds",
    [
        np.empty((0, 3)),
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([[np.nan, 0.0, 1.0]]),
        np.asarray([[1.0, 0.0]]),
    ],
)
def test_spherical_voronoi_rejects_invalid_seed_vectors(seeds):
    grid = SphereGrid(16, 8)
    with pytest.raises(ValueError, match="seed"):
        spherical_voronoi_ids(grid, seeds)


def test_spherical_voronoi_normalizes_finite_nonunit_seed_vectors():
    grid = SphereGrid(32, 16)
    unit = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    scaled = unit * np.asarray([[7.0], [0.25]])
    expected = spherical_voronoi_ids(grid, unit)
    actual = spherical_voronoi_ids(grid, scaled)
    np.testing.assert_array_equal(actual, expected)



@pytest.mark.parametrize("cache_bytes", [-1, 1.5, True])
def test_sphere_grid_rejects_invalid_distance_cache_budget(cache_bytes):
    with pytest.raises((TypeError, ValueError), match="distance_cache_max_bytes"):
        SphereGrid(16, 8, distance_cache_max_bytes=cache_bytes)


@pytest.mark.parametrize("q", [np.nan, np.inf, -0.01, 1.01])
def test_weighted_quantile_rejects_invalid_quantile(q):
    grid = SphereGrid(16, 8)
    with pytest.raises(ValueError, match="q.*finite.*\[0, 1\]"):
        grid.weighted_quantile(np.arange(128, dtype=float).reshape(grid.shape), q)


@pytest.mark.parametrize("chunk", [0, -1, 1.5, True])
def test_spherical_voronoi_rejects_invalid_chunk_size(chunk):
    grid = SphereGrid(16, 8)
    with pytest.raises(ValueError, match="chunk.*positive integer"):
        spherical_voronoi_ids(grid, np.asarray([[1.0, 0.0, 0.0]]), chunk=chunk)


def test_natural_routing_rejects_nonfinite_elevation():
    grid = SphereGrid(16, 8)
    elevation = np.zeros(grid.shape)
    elevation[2, 3] = np.nan
    with pytest.raises(ValueError, match="elevation.*finite"):
        flow_directions_continuous_local(
            elevation, np.zeros(grid.shape, dtype=bool), grid
        )


def test_natural_routing_rejects_grid_shape_mismatch():
    grid = SphereGrid(16, 8)
    with pytest.raises(ValueError, match="shape.*grid"):
        flow_directions_continuous_local(
            np.zeros((4, 32)), np.zeros((4, 32), dtype=bool), grid
        )


def test_natural_routing_rejects_nonfinite_near_tie_fraction():
    grid = SphereGrid(16, 8)
    with pytest.raises(ValueError, match="near_tie_fraction.*finite"):
        flow_directions_continuous_local(
            np.zeros(grid.shape),
            np.zeros(grid.shape, dtype=bool),
            grid,
            near_tie_fraction=np.nan,
        )


def test_natural_routing_has_no_invalid_fractional_power_evaluation():
    grid = SphereGrid(48, 24)
    yy, xx = np.indices(grid.shape)
    elevation = (
        2.0
        - 0.03 * yy
        + 0.08 * np.sin(xx / 3.0)
        + 0.04 * np.cos((xx + yy) / 5.0)
    )
    ocean = np.zeros(grid.shape, dtype=bool)
    ocean[-1] = True
    with np.errstate(invalid="raise", divide="raise", over="raise"):
        receiver = flow_directions_continuous_local(elevation, ocean, grid)
    assert receiver.shape == (grid.width * grid.height,)
