from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from worldgen.config import ProceduralErosionConfig
from worldgen.erosion_forcing import build_erosion_forcing
from worldgen.grid import SphereGrid


def test_forcing_uses_runoff_lithology_soil_and_phase_crossings_without_nan():
    grid = SphereGrid(64, 32, 6371.0)
    shape = grid.shape
    land = np.ones(shape, dtype=bool)
    terrain = SimpleNamespace(
        elevation_km=np.sin(np.deg2rad(grid.lat)).astype(np.float32),
        land=land,
        ocean=~land,
        shelf=np.zeros(shape, dtype=bool),
    )
    ocean = SimpleNamespace(current_speed=np.zeros(shape, dtype=np.float32))
    temp = np.empty((12, *shape), dtype=np.float32)
    for month in range(12):
        temp[month] = -4.0 + 10.0 * np.sin(2.0 * np.pi * month / 12.0)
    climate = SimpleNamespace(
        temperature_c=temp,
        annual_temperature_c=temp.mean(axis=0),
        annual_precipitation_mm=np.full(shape, 900.0, dtype=np.float32),
        precipitation_mm=np.full((12, *shape), 75.0, dtype=np.float32),
        snow_fraction=np.full(shape, 0.25, dtype=np.float32),
        continentality_index_c=np.full(shape, 20.0, dtype=np.float32),
    )
    hydro = SimpleNamespace(
        surface_runoff_mm_year=np.full(shape, 450.0, dtype=np.float32),
        runoff=np.full(shape, 600.0, dtype=np.float32),
        discharge_index=np.full(shape, 0.35, dtype=np.float32),
        storminess_index=np.full(shape, 0.4, dtype=np.float32),
        soil_water_storage_mm=np.full(shape, 90.0, dtype=np.float32),
        subgrid_drainage_density_km_per_km2=np.full(shape, 0.6, dtype=np.float32),
        topographic_wetness_index=np.full(shape, 0.5, dtype=np.float32),
        height_above_nearest_drainage_m=np.full(shape, 30.0, dtype=np.float32),
        channel_class=np.zeros(shape, dtype=np.uint8),
    )
    geology = SimpleNamespace(rock_code=np.full(shape, 3, dtype=np.uint8))
    astronomy = SimpleNamespace(planet={"surface_gravity_m_s2": 9.80665})
    cfg = ProceduralErosionConfig(enabled=True)
    forcing = build_erosion_forcing(
        grid, terrain, ocean, climate, hydro, geology, astronomy, cfg
    )
    assert forcing.strength.shape == shape
    assert np.isfinite(forcing.strength).all()
    assert np.any(forcing.fluvial_activity > 0)
    assert np.any(forcing.freeze_thaw_activity > 0)
    assert np.all(forcing.preferred_scale_km >= cfg.min_wavelength_km)
    assert np.all(forcing.preferred_scale_km <= cfg.max_wavelength_km)
