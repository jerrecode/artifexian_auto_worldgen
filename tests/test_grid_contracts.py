import numpy as np
import pytest

from worldgen.grid import SphereGrid, distance_to, spherical_voronoi_ids


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
