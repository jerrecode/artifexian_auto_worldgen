import numpy as np

from worldgen.grid import SphereGrid


def test_spherical_divergence_and_curl_for_potential_harmonic():
    grid = SphereGrid(720, 360)
    lam = np.deg2rad(grid.lon)
    phi = np.deg2rad(grid.lat)
    amp = 20.0

    # Tangential gradient of chi = amp * R * cos(phi) * cos(lambda).
    # In the project's east/south convention this is
    # u=-A sin(lambda), v=A sin(phi) cos(lambda).
    u = -amp * np.sin(lam)
    v = amp * np.sin(phi) * np.cos(lam)
    expected_div = -2.0 * amp * np.cos(phi) * np.cos(lam) / grid.radius_km

    div = grid.ops.divergence(u, v)
    vort = grid.ops.curl(u, v)

    assert np.isfinite(div).all()
    assert np.isfinite(vort).all()
    assert np.max(np.abs(div - expected_div)) < 1.0e-4
    assert np.sqrt(np.mean((div - expected_div) ** 2)) < 6.0e-6
    assert np.max(np.abs(vort)) < 4.0e-5


def test_solid_body_rotation_has_zero_divergence_and_known_vorticity():
    grid = SphereGrid(720, 360)
    phi = np.deg2rad(grid.lat)
    amp = 25.0
    u = amp * np.cos(phi)
    v = np.zeros_like(u)

    div = grid.ops.divergence(u, v)
    vort = grid.ops.curl(u, v)
    expected_vort = 2.0 * amp * np.sin(phi) / grid.radius_km

    assert np.max(np.abs(div)) < 1.0e-12
    assert np.max(np.abs(vort - expected_vort)) < 3.0e-6


def test_vector_calculus_is_continuous_across_longitude_seam():
    grid = SphereGrid(360, 180)
    lam = np.deg2rad(grid.lon)
    phi = np.deg2rad(grid.lat)
    u = -12.0 * np.sin(lam)
    v = 12.0 * np.sin(phi) * np.cos(lam)

    div = grid.ops.divergence(u, v)
    expected = -24.0 * np.cos(phi) * np.cos(lam) / grid.radius_km
    seam_error = np.max(np.abs(div[:, [0, -1]] - expected[:, [0, -1]]))
    interior_error = np.max(np.abs(div[:, 1:-1] - expected[:, 1:-1]))
    assert seam_error <= max(1.2e-4, 1.5 * interior_error)


def test_vector_calculus_remains_finite_and_accurate_near_poles():
    grid = SphereGrid(720, 360)
    lam = np.deg2rad(grid.lon)
    phi = np.deg2rad(grid.lat)
    u = -15.0 * np.sin(lam)
    v = 15.0 * np.sin(phi) * np.cos(lam)
    expected = -30.0 * np.cos(phi) * np.cos(lam) / grid.radius_km

    div = grid.ops.divergence(u, v)
    polar = np.abs(grid.lat) > 85.0
    assert np.isfinite(div[polar]).all()
    assert np.max(np.abs(div[polar] - expected[polar])) < 1.0e-4


def test_physical_distance_morphology_crosses_seam():
    grid = SphereGrid(128, 64, distance_cache_max_bytes=8 * 1024 * 1024)
    mask = np.zeros((64, 128), dtype=bool)
    mask[32, 0] = True
    radius = 1.25 * float(grid.dx_km[32, 0])
    expanded = grid.ops.binary_dilation_km(mask, radius)
    assert expanded[32, -1]
    assert expanded[32, 1]


def test_cell_morphology_crosses_pole_antipodally():
    grid = SphereGrid(64, 32)
    mask = np.zeros((32, 64), dtype=bool)
    mask[0, 0] = True
    expanded = grid.ops.binary_dilation(mask, 1)
    assert expanded[0, 32]
    assert expanded[0, 31]
    assert expanded[0, 33]
