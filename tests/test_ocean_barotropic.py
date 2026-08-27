import numpy as np

from worldgen.config import OceanConfig
from worldgen.grid import SphereGrid
from worldgen.ocean_barotropic import build_barotropic_currents, velocity_from_streamfunction


def test_streamfunction_velocity_is_nearly_divergence_free_interior():
    grid = SphereGrid(128, 64)
    phi = np.deg2rad(grid.lat)
    lam = np.deg2rad(grid.lon)
    psi = np.cos(phi) ** 2 * np.sin(2.0 * lam)

    u, v = velocity_from_streamfunction(grid, psi)
    div = grid.ops.divergence(u, v)

    # Exclude the first/last two cell-centred polar bands where the equirectangular
    # basis is singular and the operator intentionally switches to one-sided fluxes.
    interior = np.zeros(grid.shape, dtype=bool)
    interior[2:-2] = True
    scale = np.sqrt(np.mean(u[interior] ** 2 + v[interior] ** 2)) / grid.radius_km
    rms = np.sqrt(np.mean(div[interior] ** 2))
    assert rms <= max(5.0e-3 * scale, 2.0e-10)


def test_barotropic_currents_are_deterministic_finite_and_land_masked():
    grid = SphereGrid(96, 48)
    ocean = np.ones(grid.shape, dtype=bool)
    ocean[:, :18] = False
    ocean[18:30, 18:29] = False
    depth = np.where(ocean, 4200.0, 0.0)
    coast_distance = np.where(ocean, 900.0, 0.0)
    coast_distance[:, 18:23] = np.where(ocean[:, 18:23], 80.0, 0.0)

    months = 12
    lat = np.deg2rad(grid.lat)
    phase = np.arange(months, dtype=float)[:, None, None] * (2.0 * np.pi / months)
    wind_u = np.broadcast_to(np.cos(lat)[None, ...], (months, *grid.shape)).copy()
    wind_u += 0.15 * np.sin(phase)
    wind_v = np.broadcast_to(0.25 * np.sin(2.0 * lat)[None, ...], (months, *grid.shape)).copy()
    wind_v += 0.08 * np.cos(phase)

    cfg = OceanConfig()
    a_u, a_v, a_diag = build_barotropic_currents(
        grid, ocean, depth, coast_distance, wind_u, wind_v, cfg
    )
    b_u, b_v, b_diag = build_barotropic_currents(
        grid, ocean, depth, coast_distance, wind_u, wind_v, cfg
    )

    assert a_u.shape == (months, *grid.shape)
    assert a_v.shape == (months, *grid.shape)
    assert np.isfinite(a_u).all()
    assert np.isfinite(a_v).all()
    assert np.array_equal(a_u, b_u)
    assert np.array_equal(a_v, b_v)
    assert a_diag == b_diag
    assert np.all(a_u[:, ~ocean] == 0.0)
    assert np.all(a_v[:, ~ocean] == 0.0)
    assert a_diag.interior_divergence_rms_per_km >= 0.0
    assert a_diag.kinetic_energy_index > 0.0


def test_zero_wind_still_produces_bounded_background_gyres():
    grid = SphereGrid(64, 32)
    ocean = np.ones(grid.shape, dtype=bool)
    depth = np.full(grid.shape, 4000.0)
    coast_distance = np.full(grid.shape, 5000.0)
    wind_u = np.zeros((12, *grid.shape), dtype=float)
    wind_v = np.zeros_like(wind_u)

    u, v, diag = build_barotropic_currents(
        grid, ocean, depth, coast_distance, wind_u, wind_v, OceanConfig()
    )

    assert np.isfinite(u).all() and np.isfinite(v).all()
    assert float(np.max(np.hypot(u, v))) <= 3.0
    assert diag.kinetic_energy_index > 0.0
