from __future__ import annotations

import json
import time

import numpy as np
import pytest

from worldgen.refinement import RefinementEngine, RefinementSpec
from worldgen.topology import spherical_resize


def _write_base_world(root, *, seed: int = 12345):
    h, w = 8, 16
    lat = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    lon = -180.0 + (np.arange(w) + 0.5) * 360.0 / w
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    elevation = (
        1.4 * np.sin(2.0 * np.pi * xx / w)
        + 0.7 * np.cos(np.pi * (yy + 0.5) / h)
        - 0.2
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
    flow_to = np.arange(h * w, dtype=np.int64).reshape(h, w)
    np.savez(
        root / "world_arrays.npz",
        lat=lat,
        lon=lon,
        elevation_km=elevation,
        ocean_depth_m=np.maximum(-elevation * 1000.0, 0.0).astype(np.float32),
        annual_temperature_c=temp,
        plate_id=plate,
        true_color_rgb=rgb,
        flow_to=flow_to,
        subplate_parent=np.arange(5, dtype=np.int16),
    )
    (root / "world.json").write_text(json.dumps({"seed": seed}), encoding="utf-8")
    return elevation


def test_refinement_composes_children_to_exact_spherical_parent_resample_without_detail(tmp_path):
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
    assert not (level / "arrays" / "flow_to.npy").exists()
    index = json.loads((level / "index.json").read_text(encoding="utf-8"))
    assert "flow_to" in index["omitted_fields"]
    assert (level / "maps" / "height_grayscale_16bit.png").exists()


def test_repeated_invocation_creates_sections_of_sections(tmp_path):
    _write_base_world(tmp_path)
    spec = RefinementSpec(scale=2, sections_y=2, sections_x=2, halo_cells=2, elevation_detail_strength=0.0)
    engine = RefinementEngine(tmp_path, spec=spec)
    first = engine.refine(1)
    assert first["deepest_complete_level"] == 1
    second = engine.refine(1)
    assert second["deepest_complete_level"] == 2
    assert second["levels"]["2"]["resolution"] == [64, 32]
    assert second["levels"]["2"]["node_count"] == 16
    index = json.loads(
        (tmp_path / "refinement" / "levels" / "level_0002" / "index.json").read_text(encoding="utf-8")
    )
    ids = [node["node_id"] for node in index["nodes"]]
    assert any(node_id.count("/") >= 2 for node_id in ids)


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


def test_deterministic_spherical_detail_is_independent_of_section_partition(tmp_path):
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
    ea = np.load(a / "refinement" / "levels" / "level_0001" / "arrays" / "elevation_km.npy")
    eb = np.load(b / "refinement" / "levels" / "level_0001" / "arrays" / "elevation_km.npy")
    np.testing.assert_array_equal(ea, eb)
