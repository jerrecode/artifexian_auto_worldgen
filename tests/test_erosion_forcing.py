from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from worldgen.config import ProceduralErosionConfig
from worldgen.erosion_forcing import build_erosion_forcing
from worldgen.grid import SphereGrid


def _forcing_fixture(*, land: bool = True, temperature_c: float = 15.0,
                     precipitation_mm_year: float = 0.0, snow_fraction: float = 0.0):
    grid = SphereGrid(64, 32, 6371.0)
    shape = grid.shape
    land_mask = np.full(shape, land, dtype=bool)
    terrain = SimpleNamespace(
        elevation_km=np.sin(np.deg2rad(grid.lat)).astype(np.float32),
        land=land_mask,
        ocean=~land_mask,
        shelf=np.zeros(shape, dtype=bool),
    )
    ocean = SimpleNamespace(current_speed=np.zeros(shape, dtype=np.float32))
    climate = SimpleNamespace(
        temperature_c=np.full((12, *shape), temperature_c, dtype=np.float32),
        annual_temperature_c=np.full(shape, temperature_c, dtype=np.float32),
        annual_precipitation_mm=np.full(
            shape, precipitation_mm_year, dtype=np.float32
        ),
        snow_fraction=np.full(shape, snow_fraction, dtype=np.float32),
        continentality_index_c=np.zeros(shape, dtype=np.float32),
    )
    hydrology = SimpleNamespace(
        surface_runoff_mm_year=np.zeros(shape, dtype=np.float32),
        discharge_index=np.zeros(shape, dtype=np.float32),
        storminess_index=np.zeros(shape, dtype=np.float32),
        soil_water_storage_mm=np.zeros(shape, dtype=np.float32),
        subgrid_drainage_density_km_per_km2=np.zeros(shape, dtype=np.float32),
        topographic_wetness_index=np.zeros(shape, dtype=np.float32),
        height_above_nearest_drainage_m=np.zeros(shape, dtype=np.float32),
        channel_class=np.zeros(shape, dtype=np.uint8),
    )
    geology = SimpleNamespace(bedrock_code=np.full(shape, 3, dtype=np.uint8))
    astronomy = SimpleNamespace(planet={"surface_gravity_m_s2": 9.80665})
    return grid, terrain, ocean, climate, hydrology, geology, astronomy


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



def test_dry_temperate_land_has_no_rain_snow_or_weathering_forcing():
    args = _forcing_fixture(
        land=True,
        temperature_c=15.0,
        precipitation_mm_year=0.0,
        snow_fraction=0.0,
    )
    cfg = ProceduralErosionConfig(enabled=True)
    forcing = build_erosion_forcing(*args, cfg)
    assert np.count_nonzero(forcing.fluvial_activity) == 0
    assert np.count_nonzero(forcing.pluvial_activity) == 0
    assert np.count_nonzero(forcing.glacial_activity) == 0
    assert np.count_nonzero(forcing.chemical_weathering) == 0
    assert np.count_nonzero(forcing.freeze_thaw_activity) == 0
    assert np.count_nonzero(forcing.strength) == 0


def test_cold_snowy_land_selects_glacial_regime_without_liquid_rain():
    args = _forcing_fixture(
        land=True,
        temperature_c=-12.0,
        precipitation_mm_year=1000.0,
        snow_fraction=1.0,
    )
    cfg = ProceduralErosionConfig(enabled=True)
    forcing = build_erosion_forcing(*args, cfg)
    assert np.count_nonzero(forcing.pluvial_activity) == 0
    assert np.count_nonzero(forcing.chemical_weathering) == 0
    assert np.all(forcing.glacial_activity > 0.0)
    assert np.all(forcing.strength > 0.0)


def test_shelf_marine_activity_exceeds_matching_deep_ocean_activity():
    grid, terrain, ocean, climate, hydrology, geology, astronomy = _forcing_fixture(
        land=False,
        temperature_c=5.0,
        precipitation_mm_year=0.0,
        snow_fraction=0.0,
    )
    shelf = np.zeros(grid.shape, dtype=bool)
    shelf[:, ::2] = True
    terrain.shelf = shelf
    cfg = ProceduralErosionConfig(enabled=True)
    forcing = build_erosion_forcing(
        grid, terrain, ocean, climate, hydrology, geology, astronomy, cfg
    )
    assert np.count_nonzero(forcing.fluvial_activity) == 0
    assert np.count_nonzero(forcing.pluvial_activity) == 0
    assert np.count_nonzero(forcing.glacial_activity) == 0
    assert float(np.mean(forcing.marine_activity[shelf])) > (
        5.0 * float(np.mean(forcing.marine_activity[~shelf]))
    )


def test_modern_and_legacy_geology_codes_are_equivalent_and_shape_checked():
    grid, terrain, ocean, climate, hydrology, geology, astronomy = _forcing_fixture(
        land=True,
        precipitation_mm_year=500.0,
    )
    cfg = ProceduralErosionConfig(enabled=True)
    modern = build_erosion_forcing(
        grid, terrain, ocean, climate, hydrology, geology, astronomy, cfg
    )
    legacy = build_erosion_forcing(
        grid,
        terrain,
        ocean,
        climate,
        hydrology,
        SimpleNamespace(rock_code=geology.bedrock_code.copy()),
        astronomy,
        cfg,
    )
    assert np.array_equal(modern.strength, legacy.strength)

    import pytest
    with pytest.raises(ValueError, match="shape"):
        build_erosion_forcing(
            grid,
            terrain,
            ocean,
            climate,
            hydrology,
            SimpleNamespace(bedrock_code=np.zeros((1, 1), dtype=np.uint8)),
            astronomy,
            cfg,
        )
