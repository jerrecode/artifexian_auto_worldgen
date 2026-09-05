from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from worldgen.diagnostics import world_diagnostics
from worldgen.grid import SphereGrid


def _world(*, wet: bool) -> dict:
    grid = SphereGrid(8, 4, 6371.0)
    shape = grid.shape
    liquid_mask = np.full(shape, wet, dtype=bool)
    land = ~liquid_mask
    ocean = liquid_mask.copy()
    elevation = np.full(shape, -1.0 if wet else 1.0, dtype=np.float32)

    terrain = SimpleNamespace(
        elevation_km=elevation,
        land=land,
        ocean=ocean,
        metadata={"actual_land_fraction": float(grid.weighted_fraction(land))},
    )
    climate = SimpleNamespace(
        annual_temperature_c=np.zeros(shape, dtype=np.float32),
        annual_precipitation_mm=np.zeros(shape, dtype=np.float32),
        precipitation_mm=np.zeros((12, *shape), dtype=np.float32),
    )
    ocean_state = SimpleNamespace(
        current_u=np.zeros(shape, dtype=np.float32),
        current_v=np.zeros(shape, dtype=np.float32),
    )
    hydro = SimpleNamespace(
        runoff=np.zeros(shape, dtype=np.float32),
        flow_to=np.full(shape, -1, dtype=np.int64),
        rivers=np.zeros(shape, dtype=bool),
        lakes=np.zeros(shape, dtype=bool),
        stream_order=np.zeros(shape, dtype=np.uint8),
        discharge_index=np.zeros(shape, dtype=np.float32),
        metadata={},
    )
    liquids = SimpleNamespace(
        total_liquid_volume_m3=1.0e12 if wet else 0.0,
        total_liquid_mass_kg=1.0e15 if wet else 0.0,
        volume_residual_m3=0.0,
        liquid_mask=liquid_mask,
    )
    return {
        "grid": grid,
        "terrain": terrain,
        "climate": climate,
        "ocean": ocean_state,
        "hydrology": hydro,
        "surface_liquids": liquids,
    }


def _invariant(result: dict, name: str) -> dict:
    return next(item for item in result["invariants"] if item["name"] == name)


def test_fully_frozen_volatile_inventory_allows_all_land_surface():
    result = world_diagnostics(_world(wet=False))
    assert _invariant(result, "terrain:land_fraction_valid")["passed"] is True
    assert _invariant(result, "surface_liquid:wet_mask_matches_terrain_ocean")["passed"] is True


def test_mobile_liquid_allows_fully_oceanic_surface():
    result = world_diagnostics(_world(wet=True))
    assert _invariant(result, "terrain:land_fraction_valid")["passed"] is True
    assert _invariant(result, "surface_liquid:wet_mask_matches_terrain_ocean")["passed"] is True


def test_mobile_liquid_mask_must_match_canonical_terrain_ocean():
    world = _world(wet=True)
    world["terrain"].ocean = np.zeros(world["grid"].shape, dtype=bool)
    result = world_diagnostics(world)
    assert _invariant(result, "surface_liquid:wet_mask_matches_terrain_ocean")["passed"] is False
    assert result["all_invariants_passed"] is False
