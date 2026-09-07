from __future__ import annotations

import json
import math

import numpy as np

from worldgen.planet_tiles import PlanetTilePyramid, TilePyramidSpec
from worldgen.ultrares import (
    UltraResolutionSpec,
    _merge_children_downsample,
    derive_scale_aware_geomorphology_spec,
    make_ultra_resolution_plan,
)


def _write_source(root) -> None:
    h, w = 32, 64
    lat = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    lon = -180.0 + (np.arange(w) + 0.5) * 360.0 / w
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    elevation = (
        0.8 * np.sin(2.0 * np.pi * xx / w)
        + 0.4 * np.cos(np.pi * (yy + 0.5) / h)
    ).astype(np.float32)
    runoff = np.maximum(500.0 + 150.0 * np.cos(np.deg2rad(lat))[:, None], 0.0)
    runoff = np.broadcast_to(runoff, (h, w)).astype(np.float32)
    np.savez(
        root / "world_arrays.npz",
        lat=lat,
        lon=lon,
        elevation_km=elevation,
        runoff_mm_year=runoff,
        annual_precipitation_mm=(1.8 * runoff).astype(np.float32),
        rivers=np.zeros((h, w), dtype=bool),
        stream_order=np.zeros((h, w), dtype=np.uint8),
        discharge_index=np.zeros((h, w), dtype=np.float32),
        river_width_proxy=np.zeros((h, w), dtype=np.float32),
    )
    (root / "world.json").write_text(
        json.dumps(
            {
                "seed": 1234,
                "astronomy": {"planet": {"radius_earth": 1.0}},
                "config": {
                    "hydrology": {
                        "stream_power_m": 0.5,
                        "stream_power_n": 1.0,
                        "max_fluvial_erosion_m_per_iteration": 15.0,
                        "deposition_strength": 0.54,
                        "hillslope_diffusion_strength": 0.028,
                    },
                    "procedural_erosion": {
                        "base_wavelength_km": 420.0,
                        "base_amplitude_m": 24.0,
                        "gain": 0.52,
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_ultrares_plan_is_four_times_base_then_two_times_subsection(tmp_path):
    _write_source(tmp_path)
    cfg = UltraResolutionSpec(
        base_linear_multiplier=4.0,
        subsection_linear_multiplier=2.0,
        tile_size=64,
    )
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(
            tile_size=64,
            elevation_detail_strength=0.0,
            maximum_level=6,
        ),
    )
    plan = make_ultra_resolution_plan(pyramid, cfg)

    assert (plan.source_width, plan.source_height) == (64, 32)
    assert (plan.fullview_width, plan.fullview_height) == (256, 128)
    assert plan.actual_base_multiplier >= 4.0 - 1.0e-12
    assert plan.actual_subsection_multiplier >= 2.0 - 1.0e-12
    assert plan.base_level == 0
    assert plan.finest_level == 1


def test_scale_aware_erosion_band_fills_only_newly_resolvable_frequencies(tmp_path):
    _write_source(tmp_path)
    cfg = UltraResolutionSpec(
        base_linear_multiplier=4.0,
        subsection_linear_multiplier=2.0,
        tile_size=64,
        min_samples_per_wavelength=4.0,
        procedural_lacunarity=2.0,
    )
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(
            tile_size=64,
            elevation_detail_strength=0.0,
            maximum_level=6,
        ),
    )
    plan = make_ultra_resolution_plan(pyramid, cfg)
    geom = derive_scale_aware_geomorphology_spec(pyramid, plan, cfg)

    ratio = plan.source_equatorial_m_per_sample / plan.finest_m_per_sample
    assert math.isclose(ratio, 8.0, rel_tol=1.0e-12)
    assert math.isclose(
        geom.procedural_base_wavelength_samples,
        cfg.min_samples_per_wavelength * ratio,
        rel_tol=2.0e-9,
    )
    # 32, 16, 8 and 4 samples/wavelength: the last octave reaches the safe
    # resolution boundary rather than stopping at an interpolated-looking scale.
    assert geom.procedural_octaves == 4
    finest_samples = geom.procedural_base_wavelength_samples / (
        geom.procedural_lacunarity ** (geom.procedural_octaves - 1)
    )
    assert math.isclose(finest_samples, 4.0, rel_tol=2.0e-9)
    assert geom.max_fluvial_erosion_m < 15.0
    assert geom.procedural_amplitude_m < 24.0


def test_parent_reconstruction_is_exact_bottom_up_decimation():
    n = 8
    fine = np.arange((2 * n + 1) ** 2, dtype=np.float32).reshape(2 * n + 1, 2 * n + 1)
    children = (
        fine[: n + 1, : n + 1],
        fine[: n + 1, n:],
        fine[n:, : n + 1],
        fine[n:, n:],
    )
    parent = _merge_children_downsample(children)
    np.testing.assert_array_equal(parent, fine[::2, ::2])
