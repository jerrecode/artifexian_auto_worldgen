from __future__ import annotations

from dataclasses import asdict
import json
import math

import numpy as np

from worldgen.local_geomorphology import LOCAL_GEOMORPHOLOGY_ALGORITHM_REVISION
from worldgen.planet_tiles import PlanetTilePyramid, TileKey, TilePyramidSpec, tile_geometry
from worldgen.ultrares import (
    ULTRARES_RESUME_FIELDS,
    UltraResolutionSpec,
    UltraResolutionTilePyramid,
    _geomorph_metadata_path,
    _geomorph_path,
    _merge_children_downsample,
    _select_shard_keys,
    _semantic_authority_sha256,
    _tile_checkpoint_path,
    _tile_resume_valid,
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
        mountain_strength=np.full((h, w), 0.72, dtype=np.float32),
        ruggedness=np.full((h, w), 0.55, dtype=np.float32),
        convergence_strength=np.full((h, w), 0.48, dtype=np.float32),
        strain_field=np.full((h, w), 0.36, dtype=np.float32),
        paleo_convergence=np.full((h, w), 0.25, dtype=np.float32),
        orogen_age_myr=np.full((h, w), 140.0, dtype=np.float32),
        stress_field=np.full((h, w), 0.42, dtype=np.float32),
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


def test_fractional_subsection_request_advances_to_discrete_lod(tmp_path):
    _write_source(tmp_path)
    cfg = UltraResolutionSpec(
        base_linear_multiplier=4.0,
        subsection_linear_multiplier=3.0,
        tile_size=64,
        terrain_detail_strength=1.0,
    )
    pyramid = UltraResolutionTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(
            tile_size=64,
            elevation_detail_strength=1.0,
            maximum_level=6,
        ),
    )
    plan = make_ultra_resolution_plan(pyramid, cfg)
    assert plan.base_level == 0
    # Cube-sphere LODs halve ground spacing, so a requested 3x refinement
    # necessarily advances by two levels and yields 4x native sampling.
    assert plan.finest_level == 2
    assert plan.actual_subsection_multiplier >= 3.0
    assert math.isclose(plan.actual_subsection_multiplier, 4.0, rel_tol=1e-12)


def test_xyz_microrelief_is_nontrivial_and_exactly_shared_across_tile_edge(tmp_path):
    _write_source(tmp_path)
    pyramid = UltraResolutionTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(
            tile_size=64,
            elevation_detail_strength=1.0,
            detail_hurst_exponent=0.65,
            maximum_level=6,
        ),
    )
    left = TileKey("px", 2, 1, 1)
    right = TileKey("px", 2, 2, 1)
    g_left = tile_geometry(left, 64)
    g_right = tile_geometry(right, 64)
    d_left = pyramid._spectral_detail(g_left.xyz, left.level)
    d_right = pyramid._spectral_detail(g_right.xyz, right.level)

    assert float(np.std(d_left)) > 0.5
    assert float(np.max(np.abs(d_left))) > 1.0
    np.testing.assert_allclose(
        d_left[:, -1],
        d_right[:, 0],
        rtol=0.0,
        atol=1.0e-9,
    )



def test_ultrares_shards_partition_keys_exactly_once():
    keys = tuple(TileKey("px", 3, i % 8, i // 8) for i in range(37))
    shards = [
        _select_shard_keys(keys, shard_index=index, shard_count=7)
        for index in range(7)
    ]
    flattened = [key for shard in shards for key in shard]

    assert len(flattened) == len(keys)
    assert len(set(flattened)) == len(keys)
    assert set(flattened) == set(keys)
    for index, shard in enumerate(shards):
        assert shard == tuple(
            key for ordinal, key in enumerate(keys) if ordinal % 7 == index
        )


def test_ultrares_resume_requires_matching_retained_authority(tmp_path):
    _write_source(tmp_path)
    cfg = UltraResolutionSpec(
        base_linear_multiplier=4.0,
        subsection_linear_multiplier=2.0,
        tile_size=64,
        terrain_detail_strength=1.0,
    )
    pyramid = UltraResolutionTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(
            tile_size=64,
            elevation_detail_strength=1.0,
            maximum_level=6,
        ),
    )
    plan = make_ultra_resolution_plan(pyramid, cfg)
    geom = derive_scale_aware_geomorphology_spec(pyramid, plan, cfg)
    key = TileKey("px", plan.finest_level, 0, 0)
    shape = (65, 65)

    for field in ULTRARES_RESUME_FIELDS:
        path = _geomorph_path(pyramid, key, field)
        path.parent.mkdir(parents=True, exist_ok=True)
        if field == "final_streams":
            values = np.zeros(shape, dtype=np.bool_)
        elif field == "elevation_m":
            values = np.zeros(shape, dtype=np.float64)
        else:
            values = np.zeros(shape, dtype=np.float32)
        np.save(path, values, allow_pickle=False)

    metadata_path = _geomorph_metadata_path(pyramid, key)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "key": asdict(key),
                "source_sha256": pyramid._source_hash(),
                "authority_sampling_revision": pyramid.authority_sampling_revision,
                "algorithm_revision": LOCAL_GEOMORPHOLOGY_ALGORITHM_REVISION,
                "spec": asdict(geom),
            }
        ),
        encoding="utf-8",
    )

    assert _tile_resume_valid(pyramid, key, geom)
    marker = _tile_checkpoint_path(tmp_path, key)
    assert marker.exists()

    _geomorph_path(pyramid, key, "erosion_m").unlink()
    assert not _tile_resume_valid(pyramid, key, geom)


def test_ultrares_resume_rejects_stale_spec_even_with_marker(tmp_path):
    _write_source(tmp_path)
    cfg = UltraResolutionSpec(
        base_linear_multiplier=4.0,
        subsection_linear_multiplier=2.0,
        tile_size=64,
        terrain_detail_strength=1.0,
    )
    pyramid = UltraResolutionTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(
            tile_size=64,
            elevation_detail_strength=1.0,
            maximum_level=6,
        ),
    )
    plan = make_ultra_resolution_plan(pyramid, cfg)
    geom = derive_scale_aware_geomorphology_spec(pyramid, plan, cfg)
    key = TileKey("px", plan.finest_level, 0, 0)
    shape = (65, 65)

    for field in ULTRARES_RESUME_FIELDS:
        path = _geomorph_path(pyramid, key, field)
        path.parent.mkdir(parents=True, exist_ok=True)
        dtype = (
            np.bool_ if field == "final_streams"
            else np.float64 if field == "elevation_m"
            else np.float32
        )
        np.save(path, np.zeros(shape, dtype=dtype), allow_pickle=False)

    metadata = {
        "schema_version": 2,
        "key": asdict(key),
        "source_sha256": pyramid._source_hash(),
        "authority_sampling_revision": pyramid.authority_sampling_revision,
        "algorithm_revision": LOCAL_GEOMORPHOLOGY_ALGORITHM_REVISION,
        "spec": asdict(geom),
    }
    metadata["spec"]["max_fluvial_erosion_m"] += 1.0
    metadata_path = _geomorph_metadata_path(pyramid, key)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    marker = _tile_checkpoint_path(tmp_path, key)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"state": "complete"}), encoding="utf-8")

    assert not _tile_resume_valid(pyramid, key, geom)



def test_semantic_authority_fingerprint_is_order_stable_and_content_sensitive():
    arrays_a = {
        "elevation": np.arange(12, dtype=np.float64).reshape(3, 4),
        "streams": np.array([[False, True], [True, False]], dtype=np.bool_),
    }
    arrays_b = {
        "streams": arrays_a["streams"].copy(),
        "elevation": arrays_a["elevation"].copy(),
    }
    digest_a = _semantic_authority_sha256(arrays_a)
    digest_b = _semantic_authority_sha256(arrays_b)
    assert digest_a == digest_b

    arrays_b["elevation"][1, 2] += 1.0
    assert _semantic_authority_sha256(arrays_b) != digest_a


def test_ultrares_pyramid_prefers_semantic_compaction_fingerprint(tmp_path):
    _write_source(tmp_path)
    semantic = "ab" * 32
    report = tmp_path / "ultrares" / "authority_compaction.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps({"semantic_sha256": semantic}),
        encoding="utf-8",
    )

    pyramid = UltraResolutionTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(
            tile_size=64,
            elevation_detail_strength=1.0,
            maximum_level=6,
        ),
    )
    assert pyramid._source_hash() == semantic



def test_metric_isotropic_elevation_prefilter_suppresses_polar_zonal_aliasing(tmp_path):
    _write_source(tmp_path)
    source_path = tmp_path / "world_arrays.npz"
    with np.load(source_path, allow_pickle=False) as z:
        arrays = {name: np.asarray(z[name]) for name in z.files}

    elevation = np.zeros_like(arrays["elevation_km"], dtype=np.float32)
    x = np.arange(elevation.shape[1], dtype=np.float64)
    stripe = (8.0 * np.sin(2.0 * np.pi * x / 4.0)).astype(np.float32)
    elevation[0, :] = stripe
    elevation[-1, :] = stripe
    elevation[elevation.shape[0] // 2, :] = stripe
    arrays["elevation_km"] = elevation
    np.savez(source_path, **arrays)

    pyramid = UltraResolutionTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(
            tile_size=64,
            elevation_detail_strength=1.0,
            maximum_level=6,
        ),
    )
    filtered = pyramid._metric_isotropic_elevation_source()

    raw_polar_std = float(np.std(elevation[0]))
    filtered_polar_std = float(np.std(filtered[0]))
    raw_equatorial_std = float(np.std(elevation[elevation.shape[0] // 2]))
    filtered_equatorial_std = float(np.std(filtered[elevation.shape[0] // 2]))

    assert filtered_polar_std < 0.10 * raw_polar_std
    assert filtered_equatorial_std > 0.95 * raw_equatorial_std
    np.testing.assert_allclose(
        np.mean(filtered, axis=1),
        np.mean(elevation, axis=1),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_resume_rejects_pre_sampling_revision_checkpoint(tmp_path):
    _write_source(tmp_path)
    cfg = UltraResolutionSpec(
        base_linear_multiplier=4.0,
        subsection_linear_multiplier=2.0,
        tile_size=64,
        terrain_detail_strength=1.0,
    )
    pyramid = UltraResolutionTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(
            tile_size=64,
            elevation_detail_strength=1.0,
            maximum_level=6,
        ),
    )
    plan = make_ultra_resolution_plan(pyramid, cfg)
    geom = derive_scale_aware_geomorphology_spec(pyramid, plan, cfg)
    key = TileKey("px", plan.finest_level, 0, 0)
    shape = (65, 65)

    for field in ULTRARES_RESUME_FIELDS:
        path = _geomorph_path(pyramid, key, field)
        path.parent.mkdir(parents=True, exist_ok=True)
        dtype = (
            np.bool_ if field == "final_streams"
            else np.float64 if field == "elevation_m"
            else np.float32
        )
        np.save(path, np.zeros(shape, dtype=dtype), allow_pickle=False)

    metadata_path = _geomorph_metadata_path(pyramid, key)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "key": asdict(key),
                "source_sha256": pyramid._source_hash(),
                "spec": asdict(geom),
            }
        ),
        encoding="utf-8",
    )

    assert not _tile_resume_valid(pyramid, key, geom)
