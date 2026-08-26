import numpy as np

from worldgen.grid import SphereGrid, smooth_periodic
from worldgen.topology import prepare_spherical_bilinear_sampler, apply_bilinear_sampler


def test_spherical_bilinear_sampler_crosses_north_pole_antipodally():
    h, w = 16, 32
    a = np.arange(h * w, dtype=np.float64).reshape(h, w)
    sampler = prepare_spherical_bilinear_sampler(np.array([[-1.0]]), np.array([[0.0]]), (h, w))
    got = apply_bilinear_sampler(a, sampler)
    assert got.shape == (1, 1)
    assert got[0, 0] == a[0, w // 2]


def test_spherical_bilinear_sampler_crosses_south_pole_antipodally():
    h, w = 16, 32
    a = np.arange(h * w, dtype=np.float64).reshape(h, w)
    sampler = prepare_spherical_bilinear_sampler(np.array([[float(h)]]), np.array([[3.0]]), (h, w))
    got = apply_bilinear_sampler(a, sampler)
    assert got[0, 0] == a[h - 1, (3 + w // 2) % w]


def test_spherical_bilinear_sampler_wraps_longitude():
    h, w = 12, 24
    a = np.arange(h * w, dtype=np.float64).reshape(h, w)
    sampler = prepare_spherical_bilinear_sampler(np.array([[5.0]]), np.array([[float(w)]]), (h, w))
    got = apply_bilinear_sampler(a, sampler)
    assert got[0, 0] == a[5, 0]


def test_pole_aware_gaussian_filter_diffuses_to_antipodal_longitude():
    h, w = 32, 64
    impulse = np.zeros((h, w), dtype=np.float64)
    impulse[0, 0] = 1.0
    smoothed = smooth_periodic(impulse, (1.5, 1.5))
    assert smoothed[0, w // 2] > 0.0
    assert np.isfinite(smoothed).all()


def test_pole_aware_gaussian_filter_preserves_constant_field():
    grid = SphereGrid(64, 32)
    a = np.full((grid.height, grid.width), 3.25, dtype=np.float64)
    out = grid.ops.gaussian_filter(a, (2.0, 3.0))
    assert np.allclose(out, 3.25, atol=1e-12)
