from __future__ import annotations

import numpy as np

import worldgen  # noqa: F401 - package import installs compatibility initializer
from worldgen.config import TectonicsConfig
from worldgen.grid import SphereGrid
from worldgen import tectonics


def _centers(seed: int, count: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return tectonics.random_unit_vectors(rng, count)


def _equivalent_speed_cm_yr(omega: np.ndarray, radius_km: float) -> np.ndarray:
    # tectonics stores angular rates in rad/Myr proxy: speed_cm_yr * 10 / radius_km
    return np.linalg.norm(np.asarray(omega, dtype=np.float64), axis=1) * radius_km / 10.0


def test_titan_scale_plate_speed_does_not_reverse_uniform_interval() -> None:
    cfg = TectonicsConfig(max_plate_speed_cm_yr=0.03)
    grid = SphereGrid(64, 32, radius_km=2574.73)
    macro = _centers(1001, cfg.plate_count)
    rng = np.random.default_rng(2002)

    centers, parent, desired, is_continental = tectonics._initial_subplates(
        macro, cfg, grid, rng
    )

    assert centers.shape == desired.shape
    assert centers.shape[1] == 3
    assert parent.shape == (len(centers),)
    assert is_continental.shape == (len(centers),)
    assert np.isfinite(desired).all()

    speeds = _equivalent_speed_cm_yr(desired, grid.radius_km)
    # Subplate dispersion and the final 0.82..1.18 multiplier can move a child
    # modestly beyond its parent speed, but should stay in the configured sluggish
    # regime rather than inheriting the historical 0.8 cm/yr floor.
    assert float(np.max(speeds)) < 0.10
    assert float(np.median(speeds)) < 0.05


def test_zero_maximum_plate_speed_remains_exactly_stationary() -> None:
    cfg = TectonicsConfig(max_plate_speed_cm_yr=0.0)
    grid = SphereGrid(64, 32, radius_km=2574.73)
    macro = _centers(3003, cfg.plate_count)
    rng = np.random.default_rng(4004)

    _centres, _parent, desired, _continental = tectonics._initial_subplates(
        macro, cfg, grid, rng
    )

    assert np.array_equal(desired, np.zeros_like(desired))
