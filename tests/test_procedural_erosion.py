from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import worldgen.procedural_erosion as procedural_erosion_module

from worldgen.config import ProceduralErosionConfig
from worldgen.erosion_forcing import ErosionForcing
from worldgen.grid import SphereGrid
from worldgen.procedural_erosion import (
    apply_procedural_erosion,
    phase_cell_octave_xyz,
)


def _forcing(
    grid: SphereGrid,
    strength: float = 1.0,
    preferred_scale_km: float = 4000.0,
) -> ErosionForcing:
    shape = grid.shape
    ones = np.ones(shape, dtype=np.float32)
    zeros = np.zeros(shape, dtype=np.float32)
    return ErosionForcing(
        strength=ones * strength,
        preferred_scale_km=ones * preferred_scale_km,
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


def _terrain(grid: SphereGrid):
    lat = np.deg2rad(grid.lat)
    lon = np.deg2rad(grid.lon)
    elevation = (
        1.8 * np.sin(2.0 * lat)
        + 0.9 * np.cos(3.0 * lon) * np.cos(lat)
        + 0.35 * np.sin(5.0 * lon + lat)
    )
    return SimpleNamespace(elevation_km=elevation.astype(np.float32))


def test_procedural_erosion_is_deterministic_finite_and_bounded():
    grid = SphereGrid(64, 32, 6371.0)
    terrain = _terrain(grid)
    cfg = ProceduralErosionConfig(
        enabled=True,
        octaves=3,
        base_amplitude_m=20.0,
        max_displacement_m=35.0,
        slope_reference=0.002,
    )
    a = apply_procedural_erosion(grid, terrain, _forcing(grid), cfg, seed=1234)
    b = apply_procedural_erosion(grid, terrain, _forcing(grid), cfg, seed=1234)
    assert np.array_equal(a.delta_height_m, b.delta_height_m)
    assert np.isfinite(a.delta_height_m).all()
    assert np.any(np.abs(a.delta_height_m) > 0.0)
    assert float(np.max(np.abs(a.delta_height_m))) <= 35.0 + 1e-6
    assert np.isfinite(a.phase_coherence).all()
    assert np.all((a.phase_coherence >= 0.0) & (a.phase_coherence <= 1.0))
    assert np.all((a.ridge_map >= 0.0) & (a.ridge_map <= 1.0))
    assert np.all((a.crease_map >= 0.0) & (a.crease_map <= 1.0))


def test_zero_strength_is_exact_identity():
    grid = SphereGrid(64, 32, 6371.0)
    terrain = _terrain(grid)
    cfg = ProceduralErosionConfig(enabled=True, octaves=2, slope_reference=0.002)
    result = apply_procedural_erosion(grid, terrain, _forcing(grid, 0.0), cfg, seed=5)
    assert np.count_nonzero(result.delta_height_m) == 0


def test_different_seed_changes_resolved_morphology():
    grid = SphereGrid(64, 32, 6371.0)
    terrain = _terrain(grid)
    cfg = ProceduralErosionConfig(
        enabled=True,
        octaves=2,
        base_amplitude_m=12.0,
        slope_reference=0.002,
    )
    a = apply_procedural_erosion(grid, terrain, _forcing(grid), cfg, seed=5)
    b = apply_procedural_erosion(grid, terrain, _forcing(grid), cfg, seed=6)
    assert not np.array_equal(a.delta_height_m, b.delta_height_m)


def test_partial_phase_normalization_is_bounded_and_monotone():
    grid = SphereGrid(64, 32, 6371.0)
    lon = np.deg2rad(grid.lon)
    east_tangent = np.stack(
        (-np.sin(lon), np.cos(lon), np.zeros_like(lon)),
        axis=-1,
    )
    wavelength = np.full(grid.shape, 3000.0, dtype=np.float64)

    raw_c, raw_s, raw_q = phase_cell_octave_xyz(
        grid.xyz,
        grid.radius_km,
        wavelength,
        east_tangent,
        cell_scale=0.72,
        seed=77,
        octave=0,
        normalization=0.0,
    )
    half_c, half_s, half_q = phase_cell_octave_xyz(
        grid.xyz,
        grid.radius_km,
        wavelength,
        east_tangent,
        cell_scale=0.72,
        seed=77,
        octave=0,
        normalization=0.5,
    )
    full_c, full_s, full_q = phase_cell_octave_xyz(
        grid.xyz,
        grid.radius_km,
        wavelength,
        east_tangent,
        cell_scale=0.72,
        seed=77,
        octave=0,
        normalization=1.0,
    )

    raw_amp = np.hypot(raw_c, raw_s)
    half_amp = np.hypot(half_c, half_s)
    full_amp = np.hypot(full_c, full_s)
    assert np.all(raw_amp <= half_amp + 1e-12)
    assert np.all(half_amp <= full_amp + 1e-12)
    assert np.all(full_amp <= 1.0 + 1e-12)
    assert np.array_equal(raw_q, half_q)
    assert np.array_equal(half_q, full_q)


def test_area_zero_mean_survives_displacement_clipping():
    grid = SphereGrid(64, 32, 6371.0)
    terrain = _terrain(grid)
    cfg = ProceduralErosionConfig(
        enabled=True,
        octaves=3,
        base_amplitude_m=120.0,
        max_displacement_m=3.0,
        slope_reference=0.001,
        zero_mean_displacement=True,
    )
    result = apply_procedural_erosion(grid, terrain, _forcing(grid), cfg, seed=123)
    active = result.effective_strength > 1.0e-6
    weighted_mean = np.average(
        result.delta_height_m[active],
        weights=grid.cell_area_weights[active],
    )
    assert abs(float(weighted_mean)) < 2.0e-6
    assert float(np.max(np.abs(result.delta_height_m))) <= 3.0 + 1e-6
    assert result.metadata["displacement_limiter"] == "uniform_rescale"
    assert result.metadata["displacement_scale_factor"] < 1.0
    assert result.metadata["preconstraint_max_absolute_displacement_m"] > 3.0
    assert abs(result.metadata["area_weighted_mean_displacement_m"]) < 1.0e-9


def test_unresolved_wavelength_is_exact_identity():
    grid = SphereGrid(64, 32, 6371.0)
    terrain = _terrain(grid)
    cfg = ProceduralErosionConfig(
        enabled=True,
        octaves=4,
        min_samples_per_wavelength=5.0,
        slope_reference=0.001,
    )
    result = apply_procedural_erosion(
        grid,
        terrain,
        _forcing(grid, preferred_scale_km=10.0),
        cfg,
        seed=42,
    )
    assert result.metadata["octaves_executed"] == 0
    assert np.count_nonzero(result.delta_height_m) == 0


def test_nonfinite_forcing_is_rejected_at_public_operator_boundary():
    grid = SphereGrid(64, 32, 6371.0)
    terrain = _terrain(grid)
    forcing = _forcing(grid)
    forcing.strength[0, 0] = np.nan
    cfg = ProceduralErosionConfig(enabled=True, slope_reference=0.002)
    with pytest.raises(ValueError, match="finite"):
        apply_procedural_erosion(grid, terrain, forcing, cfg, seed=1)


@pytest.mark.parametrize("normalization", [-0.01, 1.01, np.nan, np.inf])
def test_phase_normalization_rejects_invalid_values(normalization):
    grid = SphereGrid(64, 32, 6371.0)
    lon = np.deg2rad(grid.lon)
    east_tangent = np.stack(
        (-np.sin(lon), np.cos(lon), np.zeros_like(lon)),
        axis=-1,
    )
    with pytest.raises(ValueError, match="normalization"):
        phase_cell_octave_xyz(
            grid.xyz,
            grid.radius_km,
            np.full(grid.shape, 3000.0),
            east_tangent,
            cell_scale=0.72,
            seed=1,
            octave=0,
            normalization=normalization,
        )


def test_default_phase_normalization_preserves_existing_low_level_contract():
    grid = SphereGrid(64, 32, 6371.0)
    lon = np.deg2rad(grid.lon)
    east_tangent = np.stack(
        (-np.sin(lon), np.cos(lon), np.zeros_like(lon)),
        axis=-1,
    )
    wavelength = np.full(grid.shape, 3000.0, dtype=np.float64)
    implicit = phase_cell_octave_xyz(
        grid.xyz,
        grid.radius_km,
        wavelength,
        east_tangent,
        cell_scale=0.72,
        seed=99,
        octave=2,
    )
    explicit = phase_cell_octave_xyz(
        grid.xyz,
        grid.radius_km,
        wavelength,
        east_tangent,
        cell_scale=0.72,
        seed=99,
        octave=2,
        normalization=1.0,
    )
    for a, b in zip(implicit, explicit, strict=True):
        assert np.array_equal(a, b)


@pytest.mark.parametrize(
    "radius, cell_scale, wavelength_value, message",
    [
        (0.0, 0.72, 3000.0, "radius_km"),
        (6371.0, 0.0, 3000.0, "cell_scale"),
        (6371.0, 0.72, 0.0, "wavelength_km"),
        (6371.0, 0.72, np.nan, "wavelength_km"),
    ],
)
def test_phase_cell_public_geometry_contract_rejects_invalid_inputs(
    radius, cell_scale, wavelength_value, message
):
    grid = SphereGrid(64, 32, 6371.0)
    lon = np.deg2rad(grid.lon)
    east_tangent = np.stack(
        (-np.sin(lon), np.cos(lon), np.zeros_like(lon)),
        axis=-1,
    )
    with pytest.raises(ValueError, match=message):
        phase_cell_octave_xyz(
            grid.xyz,
            radius,
            np.full(grid.shape, wavelength_value),
            east_tangent,
            cell_scale=cell_scale,
            seed=1,
            octave=0,
        )


def test_phase_cell_row_chunking_is_bit_identical_to_unchunked():
    grid = SphereGrid(64, 32, 6371.0)
    lon = np.deg2rad(grid.lon)
    east_tangent = np.stack(
        (-np.sin(lon), np.cos(lon), np.zeros_like(lon)),
        axis=-1,
    )
    wavelength = np.full(grid.shape, 3000.0, dtype=np.float64)
    reference = phase_cell_octave_xyz(
        grid.xyz,
        grid.radius_km,
        wavelength,
        east_tangent,
        cell_scale=0.72,
        seed=123,
        octave=2,
        normalization=0.5,
        chunk_rows=0,
    )
    chunked = phase_cell_octave_xyz(
        grid.xyz,
        grid.radius_km,
        wavelength,
        east_tangent,
        cell_scale=0.72,
        seed=123,
        octave=2,
        normalization=0.5,
        chunk_rows=7,
    )
    for expected, actual in zip(reference, chunked, strict=True):
        assert np.array_equal(expected, actual)


def test_high_level_chunking_preserves_complete_erosion_result():
    grid = SphereGrid(64, 32, 6371.0)
    terrain = _terrain(grid)
    base = dict(
        enabled=True,
        octaves=3,
        base_amplitude_m=20.0,
        max_displacement_m=35.0,
        slope_reference=0.002,
    )
    reference = apply_procedural_erosion(
        grid,
        terrain,
        _forcing(grid),
        ProceduralErosionConfig(**base, phase_chunk_rows=0),
        seed=773,
    )
    chunked = apply_procedural_erosion(
        grid,
        terrain,
        _forcing(grid),
        ProceduralErosionConfig(**base, phase_chunk_rows=7),
        seed=773,
    )
    for name in (
        "delta_height_m",
        "phase_coherence",
        "ridge_map",
        "crease_map",
        "effective_strength",
        "effective_scale_km",
    ):
        assert np.array_equal(getattr(reference, name), getattr(chunked, name))


@pytest.mark.parametrize("chunk_rows", [-1, 1.5, True])
def test_phase_cell_chunk_rows_reject_invalid_values(chunk_rows):
    grid = SphereGrid(64, 32, 6371.0)
    lon = np.deg2rad(grid.lon)
    east_tangent = np.stack(
        (-np.sin(lon), np.cos(lon), np.zeros_like(lon)),
        axis=-1,
    )
    with pytest.raises((TypeError, ValueError), match="chunk_rows"):
        phase_cell_octave_xyz(
            grid.xyz,
            grid.radius_km,
            np.full(grid.shape, 3000.0),
            east_tangent,
            cell_scale=0.72,
            seed=1,
            octave=0,
            chunk_rows=chunk_rows,
        )


def test_precomputed_tangent_geometry_matches_reference_path_exactly():
    grid = SphereGrid(64, 32, 6371.0)
    south = np.sin(np.deg2rad(grid.lat)) + 0.25 * np.cos(np.deg2rad(grid.lon))
    east = np.cos(np.deg2rad(grid.lat)) - 0.15 * np.sin(np.deg2rad(grid.lon))
    normal = procedural_erosion_module._unit_positions(grid)
    south_basis, east_basis = procedural_erosion_module._tangent_bases(grid)

    reference = procedural_erosion_module._tangent_perpendicular(grid, south, east)
    precomputed = procedural_erosion_module._tangent_perpendicular_precomputed(
        normal, south_basis, east_basis, south, east
    )
    assert np.array_equal(reference, precomputed)


def test_high_level_erosion_builds_invariant_spherical_geometry_once(monkeypatch):
    grid = SphereGrid(64, 32, 6371.0)
    terrain = _terrain(grid)
    cfg = ProceduralErosionConfig(
        enabled=True,
        octaves=3,
        base_amplitude_m=12.0,
        slope_reference=0.002,
        phase_chunk_rows=7,
    )

    calls = {"positions": 0, "bases": 0}
    original_positions = procedural_erosion_module._unit_positions
    original_bases = procedural_erosion_module._tangent_bases

    def counted_positions(target_grid):
        calls["positions"] += 1
        return original_positions(target_grid)

    def counted_bases(target_grid):
        calls["bases"] += 1
        return original_bases(target_grid)

    monkeypatch.setattr(procedural_erosion_module, "_unit_positions", counted_positions)
    monkeypatch.setattr(procedural_erosion_module, "_tangent_bases", counted_bases)

    result = apply_procedural_erosion(grid, terrain, _forcing(grid), cfg, seed=7721)
    assert result.metadata["octaves_executed"] >= 2
    assert result.metadata["spherical_geometry_precomputed"] is True
    assert calls == {"positions": 1, "bases": 1}
