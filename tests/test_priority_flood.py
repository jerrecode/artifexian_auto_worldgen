import numpy as np
import pytest

from worldgen.grid import SphereGrid
from worldgen.priority_flood import (
    numba_priority_flood_available,
    priority_flood,
    priority_flood_reference,
)


def _assert_backend_equivalent(elev, ocean):
    grid = SphereGrid(elev.shape[1], elev.shape[0])
    ref = priority_flood_reference(elev, ocean, grid)
    auto = priority_flood(elev, ocean, grid, backend="auto")
    assert np.array_equal(auto, ref)
    if numba_priority_flood_available():
        compiled = priority_flood(elev, ocean, grid, backend="numba")
        assert np.array_equal(compiled, ref)
    assert np.isfinite(ref).all()
    assert np.all(ref[~ocean] >= elev[~ocean])


def test_priority_flood_artificial_basin():
    elev = np.ones((16, 32), dtype=np.float64)
    ocean = np.zeros_like(elev, dtype=bool)
    ocean[7:9, 0] = True
    elev[4:12, 8:24] = 0.2
    elev[7:9, 1:8] = np.linspace(0.3, 0.9, 7)[None, :]
    _assert_backend_equivalent(elev, ocean)


def test_priority_flood_random_terrain():
    rng = np.random.default_rng(908172)
    elev = rng.normal(0.8, 0.35, (24, 48))
    ocean = elev < 0.25
    _assert_backend_equivalent(elev, ocean)


def test_priority_flood_plateau_and_ties_are_deterministic():
    elev = np.full((20, 40), 1.0, dtype=np.float64)
    ocean = np.zeros_like(elev, dtype=bool)
    ocean[9:11, 0] = True
    elev[6:14, 10:30] = 0.5
    a = priority_flood(elev, ocean, SphereGrid(40, 20), backend="auto")
    b = priority_flood(elev, ocean, SphereGrid(40, 20), backend="auto")
    assert np.array_equal(a, b)
    _assert_backend_equivalent(elev, ocean)


def test_priority_flood_crosses_longitude_seam():
    elev = np.full((18, 36), 2.0, dtype=np.float64)
    ocean = np.zeros_like(elev, dtype=bool)
    ocean[8:10, 0] = True
    elev[8:10, -4:] = 0.25
    elev[8:10, 1:5] = 0.25
    _assert_backend_equivalent(elev, ocean)


def test_priority_flood_crosses_poles_with_antipodal_shift():
    elev = np.full((18, 36), 2.0, dtype=np.float64)
    ocean = np.zeros_like(elev, dtype=bool)
    ocean[0, 0] = True
    elev[0, 18] = 0.2
    elev[1, 18] = 0.3
    _assert_backend_equivalent(elev, ocean)


def test_priority_flood_tiny_grid():
    elev = np.array([[1.0, 0.2, 0.8, 0.7], [1.1, 0.1, 0.9, 0.6]], dtype=np.float64)
    ocean = np.array([[False, True, False, False], [False, True, False, False]])
    _assert_backend_equivalent(elev, ocean)


def test_explicit_numba_backend_fails_cleanly_without_numba():
    if numba_priority_flood_available():
        pytest.skip("Numba installed")
    grid = SphereGrid(8, 4)
    elev = np.ones((4, 8), dtype=np.float64)
    ocean = np.zeros((4, 8), dtype=bool)
    with pytest.raises(RuntimeError, match="Numba"):
        priority_flood(elev, ocean, grid, backend="numba")
