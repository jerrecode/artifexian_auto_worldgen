from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from worldgen.config import ProceduralErosionConfig
from worldgen.erosion_forcing import ErosionForcing
from worldgen.grid import SphereGrid
from worldgen.procedural_erosion import apply_procedural_erosion


def _forcing(grid: SphereGrid, strength: float = 1.0) -> ErosionForcing:
    shape = grid.shape
    ones = np.ones(shape, dtype=np.float32)
    zeros = np.zeros(shape, dtype=np.float32)
    return ErosionForcing(
        strength=ones * strength,
        preferred_scale_km=ones * 600.0,
        detail=ones * 0.8,
        ridge_valley_target=zeros,
        orientation_south=zeros,
        orientation_east=ones,
        ridge_rounding=ones * 0.25,
        crease_rounding=ones * 0.15,
        fluvial_activity=ones,
        pluvial_activity=zeros,
        glacial_activity=zeros,
        marine_activity=zeros,
        chemical_weathering=zeros,
        freeze_thaw_activity=zeros,
        soil_saturation=zeros,
        fluid_mechanical_factor=ones,
        metadata={},
    )


def test_procedural_erosion_is_deterministic_finite_and_bounded():
    grid = SphereGrid(64, 32, 6371.0)
    terrain = SimpleNamespace(elevation_km=np.zeros(grid.shape, dtype=np.float32))
    cfg = ProceduralErosionConfig(
        enabled=True,
        octaves=3,
        base_amplitude_m=20.0,
        max_displacement_m=35.0,
    )
    a = apply_procedural_erosion(grid, terrain, _forcing(grid), cfg, seed=1234)
    b = apply_procedural_erosion(grid, terrain, _forcing(grid), cfg, seed=1234)
    assert np.array_equal(a.delta_height_m, b.delta_height_m)
    assert np.isfinite(a.delta_height_m).all()
    assert float(np.max(np.abs(a.delta_height_m))) <= 35.0 + 1e-6
    assert np.isfinite(a.phase_coherence).all()
    assert np.all((a.phase_coherence >= 0.0) & (a.phase_coherence <= 1.0))


def test_zero_strength_is_exact_identity():
    grid = SphereGrid(64, 32, 6371.0)
    terrain = SimpleNamespace(elevation_km=np.zeros(grid.shape, dtype=np.float32))
    cfg = ProceduralErosionConfig(enabled=True, octaves=2)
    result = apply_procedural_erosion(grid, terrain, _forcing(grid, 0.0), cfg, seed=5)
    assert np.count_nonzero(result.delta_height_m) == 0
