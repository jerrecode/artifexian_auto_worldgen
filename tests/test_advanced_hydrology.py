from types import SimpleNamespace

import numpy as np

from worldgen.config import HydrologyConfig, AppearanceConfig
from worldgen.grid import SphereGrid
from worldgen.drainage import DrainageGraph
from worldgen.watersheds import build_watershed_hierarchy
from worldgen.hydrology_advanced import build_water_balance, transport_sediment_topological
from worldgen import hydrology_advanced as hadv
from worldgen.tides import build_tides
from worldgen.appearance_advanced import attenuate_deep_bathymetry


def test_outlet_watersheds_distinguish_rivers_entering_same_ocean_system():
    grid = SphereGrid(4, 2, 6371.0)
    # Two separate land catchments. Both terminate into ocean cells, but must never be
    # merged merely because those ocean cells belong to one connected ocean.
    receiver = np.asarray([1, 2, 3, -1, 5, 6, 7, -1], dtype=np.int64)
    graph = DrainageGraph.from_receiver(receiver, (2, 4))
    land = np.asarray([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=bool)
    elevation = np.asarray([[3.0, 2.0, 1.0, -1.0], [3.2, 2.1, 1.1, -1.2]])
    drainage = np.asarray([[100.0, 300.0, 700.0, 0.0], [120.0, 350.0, 800.0, 0.0]])
    slope = np.where(land, 0.01, 0.0)
    channel = land.copy()
    w = build_watershed_hierarchy(
        grid, graph, land, elevation, drainage, slope, channel,
        subbasin_thresholds_km2=(600.0, 300.0, 100.0),
    )
    assert np.all(w.basin_id[land] > 0)
    assert w.basin_id[0, 0] == w.basin_id[0, 2]
    assert w.basin_id[1, 0] == w.basin_id[1, 2]
    assert w.basin_id[0, 0] != w.basin_id[1, 0]
    assert np.all(w.exorheic[land])
    assert np.all(w.distance_to_outlet_km[land] > 0)


def test_monthly_water_balance_closes_and_generates_groundwater_baseflow():
    cfg = HydrologyConfig()
    shape = (4, 8)
    climate = SimpleNamespace(
        precipitation_mm=np.full((12, *shape), 90.0, dtype=np.float32),
        temperature_c=np.full((12, *shape), 14.0, dtype=np.float32),
    )
    land = np.ones(shape, dtype=bool)
    wb = build_water_balance(climate, land, None, cfg)
    assert float(np.max(np.abs(wb.closure_residual_mm_year))) < 0.02
    assert float(np.mean(wb.total_runoff_mm_year)) > 0.0
    assert float(np.mean(wb.baseflow_mm_year)) > 0.0
    assert np.all(wb.total_runoff_mm_year >= wb.baseflow_mm_year)
    assert float(np.mean(wb.groundwater_recharge_mm_year)) > 0.0


def test_topological_sediment_router_reaches_distant_outlet_and_closes_mass():
    cfg = HydrologyConfig()
    cfg.deposition_strength = 0.08
    shape = (1, 8)
    flow = np.asarray([1, 2, 3, 4, 5, 6, 7, -1], dtype=np.int64)
    land = np.asarray([[1, 1, 1, 1, 1, 1, 1, 0]], dtype=bool)
    erosion = np.zeros(shape, dtype=float); erosion[0, 0] = 10.0
    area = np.ones(shape, dtype=float)
    routing = np.asarray([[8., 7., 6., 5., 4., 3., 2., 0.]])
    discharge = np.linspace(0.1, 1.0, 8, dtype=float)[None, :]
    slope = np.full(shape, 0.01, dtype=float)
    dep, _, exported = transport_sediment_topological(
        routing, flow, erosion, discharge, slope, area, land, cfg
    )
    assert float(exported.sum()) > 0.0
    assert float(dep.sum()) > 0.0
    assert abs(float(hadv._LAST_SEDIMENT_LEDGER["relative_residual"])) < 1e-12


def test_multi_moon_and_stellar_tides_are_positive_and_spatial():
    grid = SphereGrid(16, 8, 9000.0)
    astronomy = SimpleNamespace(
        planet={"mass_earth": 5.0, "radius_earth": 9000.0 / 6371.0, "semimajor_axis_au": 1.0},
        star={"mass_solar": 1.0},
        moons=[
            {"name": "A", "mass_earth": 0.01, "orbit_km": 400000.0, "sidereal_period_days": 20.0},
            {"name": "B", "mass_earth": 0.003, "orbit_km": 650000.0, "sidereal_period_days": 41.0},
        ],
        calendar={"local_year_days": 365.0},
    )
    land = np.zeros(grid.shape, dtype=bool); land[:, :3] = True
    terrain = SimpleNamespace(land=land, ocean=~land)
    ocean = SimpleNamespace(depth_m=np.where(~land, 1800.0, 0.0))
    tides = build_tides(grid, astronomy, terrain, ocean)
    assert tides.constituent_count == 3
    assert float(np.max(tides.tidal_range_m)) > 0.0
    assert float(np.std(tides.tidal_range_m[~land])) > 0.0
    assert float(np.max(tides.tidal_current_index)) <= 1.0 + 1e-6


def test_deep_ocean_optical_correction_preserves_uint8_contract():
    cfg = AppearanceConfig()
    shape = (3, 4)
    water = np.zeros(shape, dtype=bool); water[:, 2:] = True
    terrain = SimpleNamespace(ocean=water)
    ocean = SimpleNamespace(depth_m=np.where(water, 5000.0, 0.0))
    weather = SimpleNamespace(coral_reef=np.zeros(shape, bool), sea_ice_max=np.zeros(shape, bool))
    appearance = SimpleNamespace(
        water_turbidity=np.zeros(shape, np.float32),
        true_color_rgb=np.full((*shape, 3), 127, np.uint8),
        true_color_january_rgb=np.full((*shape, 3), 127, np.uint8),
        true_color_july_rgb=np.full((*shape, 3), 127, np.uint8),
        true_color_with_clouds_rgb=np.full((*shape, 3), 127, np.uint8),
        true_color_january_with_clouds_rgb=np.full((*shape, 3), 127, np.uint8),
        true_color_july_with_clouds_rgb=np.full((*shape, 3), 127, np.uint8),
        cloud_fraction_annual=np.zeros(shape, np.float32),
        cloud_fraction_monthly=np.zeros((12, *shape), np.float32),
        metadata={},
    )
    corrected = attenuate_deep_bathymetry(appearance, terrain, ocean, weather, cfg)
    assert corrected.true_color_rgb.dtype == np.uint8
    assert corrected.true_color_with_clouds_rgb.dtype == np.uint8
    # Land pixels are not rewritten by the optical ocean correction.
    assert np.all(corrected.true_color_rgb[:, :2] == 127)
    # Uniform deep water no longer contains an arbitrary relief-driven pattern.
    assert np.unique(corrected.true_color_rgb[:, 2:].reshape(-1, 3), axis=0).shape[0] == 1
