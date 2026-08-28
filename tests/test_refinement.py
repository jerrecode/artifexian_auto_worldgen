from __future__ import annotations

import json
import time

import numpy as np
import pytest

from worldgen.diagnostics import receiver_graph_is_acyclic
from worldgen.refinement import RefinementEngine, RefinementSpec
from worldgen.topology import spherical_resize


def _write_base_world(root, *, seed: int = 12345, elevation_offset: float = 0.0):
    h, w = 8, 16
    lat = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    lon = -180.0 + (np.arange(w) + 0.5) * 360.0 / w
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    elevation = (
        1.4 * np.sin(2.0 * np.pi * xx / w)
        + 0.7 * np.cos(np.pi * (yy + 0.5) / h)
        - 0.2
        + elevation_offset
    ).astype(np.float32)
    temp = (24.0 - 0.45 * np.abs(lat)[:, None] + 0.2 * np.cos(2.0 * np.pi * xx / w)).astype(np.float32)
    plate = ((xx // 4 + yy // 3) % 5).astype(np.int16)
    rgb = np.stack(
        (
            np.clip((elevation + 2.5) * 45.0, 0, 255),
            np.clip((temp + 20.0) * 4.0, 0, 255),
            np.full((h, w), 90.0),
        ),
        axis=-1,
    ).astype(np.uint8)
    # Deliberately invalid/self-referential base receiver data. Refinement must not
    # interpolate or preserve this address-space-dependent graph.
    flow_to = np.arange(h * w, dtype=np.int64)
    runoff = (420.0 + 180.0 * np.cos(np.deg2rad(lat))[:, None] + 25.0 * np.sin(2.0 * np.pi * xx / w)).astype(np.float32)
    runoff = np.maximum(runoff, 0.0)
    precip = (1.75 * runoff).astype(np.float32)
    baseflow = (0.22 * runoff).astype(np.float32)
    storminess = np.clip(0.25 + 0.18 * np.sin(2.0 * np.pi * xx / w) ** 2, 0, 1).astype(np.float32)
    np.savez(
        root / "world_arrays.npz",
        lat=lat,
        lon=lon,
        elevation_km=elevation,
        ocean_depth_m=np.maximum(-elevation * 1000.0, 0.0).astype(np.float32),
        annual_temperature_c=temp,
        annual_precipitation_mm=precip,
        runoff_mm_year=runoff,
        baseflow_mm_year=baseflow,
        storminess_index=storminess,
        plate_id=plate,
        true_color_rgb=rgb,
        flow_to=flow_to,
        subplate_parent=np.arange(5, dtype=np.int16),
    )
    (root / "world.json").write_text(
        json.dumps({
            "seed": seed,
            "astronomy": {"planet": {"radius_earth": 1.1}},
            "config": {
                "hydrology": {
                    "subbasin_thresholds_km2": [5.0e5, 5.0e4, 5.0e3],
                    "lake_min_depth_m": 3.0,
                    "lake_min_catchment_km2": 100.0,
                }
            },
        }),
        encoding="utf-8",
    )
    return elevation


def test_refinement_composes_children_and_recomputes_global_hydrology(tmp_path):
    base = _write_base_world(tmp_path)
    spec = RefinementSpec(
        scale=2,
        sections_y=2,
        sections_x=2,
        halo_cells=3,
        elevation_detail_strength=0.0,
        keep_sections=True,
    )
    manifest = RefinementEngine(tmp_path, spec=spec).refine(1)
    assert manifest["deepest_complete_level"] == 1
    assert manifest["levels"]["1"]["resolution"] == [32, 16]
    assert manifest["levels"]["1"]["node_count"] == 4

    level = tmp_path / "refinement" / "levels" / "level_0001"
    refined = np.load(level / "arrays" / "elevation_km.npy", mmap_mode="r")
    expected = spherical_resize(base, (16, 32), order=1)
    np.testing.assert_allclose(refined, expected, rtol=0.0, atol=2e-6)

    depth = np.load(level / "arrays" / "ocean_depth_m.npy", mmap_mode="r")
    np.testing.assert_allclose(depth, np.maximum(-refined * 1000.0, 0.0), atol=1e-4)

    flow = np.load(level / "arrays" / "flow_to.npy", mmap_mode="r")
    basin = np.load(level / "arrays" / "basin_id.npy", mmap_mode="r")
    streams = np.load(level / "arrays" / "stream_order.npy", mmap_mode="r")
    assert flow.shape == (16 * 32,)
    assert receiver_graph_is_acyclic(flow)
    land = refined > 0.0
    assert np.all(basin[land] > 0)
    assert int(streams.max()) >= 1
    # The old self-referential base receiver graph cannot survive recomputation.
    assert not np.array_equal(flow, np.arange(flow.size, dtype=flow.dtype))

    index = json.loads((level / "index.json").read_text(encoding="utf-8"))
    assert index["hydrology_recomputed"] is True
    assert index["entries"]["flow_to"]["recomputed_after_composition"] is True
    assert "flow_to" not in index["omitted_fields"]
    assert index["hydrology_metadata"]["all_land_assigned"] is True
    assert (level / "hydrology_refinement.json").exists()
    assert (level / "maps" / "height_grayscale_16bit.png").exists()


def test_repeated_invocation_creates_sections_of_sections_with_fresh_hydrology(tmp_path):
    _write_base_world(tmp_path)
    spec = RefinementSpec(scale=2, sections_y=2, sections_x=2, halo_cells=2, elevation_detail_strength=0.0)
    engine = RefinementEngine(tmp_path, spec=spec)
    first = engine.refine(1)
    assert first["deepest_complete_level"] == 1
    second = engine.refine(1)
    assert second["deepest_complete_level"] == 2
    assert second["levels"]["2"]["resolution"] == [64, 32]
    assert second["levels"]["2"]["node_count"] == 16
    level = tmp_path / "refinement" / "levels" / "level_0002"
    index = json.loads((level / "index.json").read_text(encoding="utf-8"))
    ids = [node["node_id"] for node in index["nodes"]]
    assert any(node_id.count("/") >= 2 for node_id in ids)
    assert index["hydrology_recomputed"] is True
    flow = np.load(level / "arrays" / "flow_to.npy")
    assert flow.shape == (32 * 64,)
    assert receiver_graph_is_acyclic(flow)


def test_interrupted_refinement_reuses_completed_atomic_node_fields(tmp_path):
    _write_base_world(tmp_path)
    spec = RefinementSpec(
        scale=2,
        sections_y=2,
        sections_x=2,
        halo_cells=2,
        elevation_detail_strength=0.0,
        keep_sections=True,
    )
    raised = False

    def interrupt(event):
        nonlocal raised
        if event.get("event") == "field_done" and not raised:
            raised = True
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        RefinementEngine(tmp_path, spec=spec, progress=interrupt).refine(1)

    completed = (
        tmp_path
        / "refinement"
        / "levels"
        / "level_0001"
        / "nodes"
        / "root__r0c0"
        / "arrays"
        / "elevation_km.npy"
    )
    assert completed.exists()
    before = completed.stat().st_mtime_ns
    time.sleep(0.01)
    result = RefinementEngine(tmp_path, spec=spec, resume=True).refine(1)
    assert result["deepest_complete_level"] == 1
    assert completed.stat().st_mtime_ns == before
    assert (tmp_path / "refinement" / "levels" / "level_0001" / "arrays" / "flow_to.npy").exists()


def test_interrupted_depth_rejects_changed_refinement_spec_on_resume(tmp_path):
    _write_base_world(tmp_path)
    original = RefinementSpec(
        scale=2,
        sections_y=2,
        sections_x=2,
        halo_cells=2,
        elevation_detail_strength=0.0,
        keep_sections=True,
    )
    interrupted = False

    def stop_after_first(event):
        nonlocal interrupted
        if event.get("event") == "field_done" and not interrupted:
            interrupted = True
            raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        RefinementEngine(tmp_path, spec=original, progress=stop_after_first).refine(1)

    changed = RefinementSpec(
        scale=3,
        sections_y=2,
        sections_x=2,
        halo_cells=4,
        elevation_detail_strength=0.0,
        keep_sections=True,
    )
    with pytest.raises(RuntimeError, match="different parent/specification"):
        RefinementEngine(tmp_path, spec=changed, resume=True).refine(1)

    rebuilt = RefinementEngine(tmp_path, spec=changed, resume=False).refine(1)
    assert rebuilt["deepest_complete_level"] == 1
    assert rebuilt["levels"]["1"]["resolution"] == [48, 24]


def test_changed_base_world_invalidates_existing_refinement_provenance(tmp_path):
    _write_base_world(tmp_path, seed=99)
    spec = RefinementSpec(scale=2, sections_y=2, sections_x=2, elevation_detail_strength=0.0)
    first = RefinementEngine(tmp_path, spec=spec).refine(1)
    assert first["deepest_complete_level"] == 1

    time.sleep(0.01)
    _write_base_world(tmp_path, seed=99, elevation_offset=0.125)
    with pytest.raises(RuntimeError, match="world_arrays.npz changed"):
        RefinementEngine(tmp_path, spec=spec, resume=True).refine(1)

    rebuilt = RefinementEngine(tmp_path, spec=spec, resume=False).refine(1)
    assert rebuilt["deepest_complete_level"] == 1
    level0 = json.loads(
        (tmp_path / "refinement" / "levels" / "level_0000" / "index.json").read_text(encoding="utf-8")
    )
    assert level0["source_fingerprint"]["sha256"]


def test_deterministic_detail_and_refined_hydrology_are_partition_independent(tmp_path):
    a = tmp_path / "a"; b = tmp_path / "b"
    a.mkdir(); b.mkdir()
    _write_base_world(a, seed=777)
    _write_base_world(b, seed=777)
    RefinementEngine(
        a,
        spec=RefinementSpec(scale=2, sections_y=2, sections_x=2, halo_cells=3, elevation_detail_strength=0.25),
    ).refine(1)
    RefinementEngine(
        b,
        spec=RefinementSpec(scale=2, sections_y=1, sections_x=4, halo_cells=3, elevation_detail_strength=0.25),
    ).refine(1)
    la = a / "refinement" / "levels" / "level_0001" / "arrays"
    lb = b / "refinement" / "levels" / "level_0001" / "arrays"
    np.testing.assert_array_equal(np.load(la / "elevation_km.npy"), np.load(lb / "elevation_km.npy"))
    for name in (
        "flow_to",
        "drainage_area_km2",
        "basin_id",
        "subbasin_level_1",
        "channel_class",
        "stream_order",
        "rivers",
    ):
        np.testing.assert_array_equal(np.load(la / f"{name}.npy"), np.load(lb / f"{name}.npy"))
