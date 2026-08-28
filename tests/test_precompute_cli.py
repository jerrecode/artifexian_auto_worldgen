from __future__ import annotations

import json

import numpy as np
import pytest

from worldgen.tile_cli import main


def _write_world(root):
    h, w = 8, 16
    lat = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    lon = -180.0 + (np.arange(w) + 0.5) * 360.0 / w
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    elevation = (0.5 * np.sin(2 * np.pi * xx / w) + 0.1 * yy).astype(np.float32)
    np.savez(root / "world_arrays.npz", lat=lat, lon=lon, elevation_km=elevation)
    (root / "world.json").write_text(
        json.dumps({"seed": 42, "astronomy": {"planet": {"radius_earth": 1.0}}}),
        encoding="utf-8",
    )


def test_cli_can_plan_complete_prefix_without_generating_tiles(tmp_path, capsys):
    _write_world(tmp_path)
    assert main([
        "--world", str(tmp_path),
        "--tile-size", "16",
        "--maximum-level", "5",
        "--precompute-depth", "2",
        "--precompute-plan-only",
        "--json",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["precompute_plan"]["tile_count"] == 126
    assert payload["precompute_plan"]["maximum_level"] == 2
    assert payload["tiles"] == []
    fields_root = tmp_path / "tiles" / "cubesphere_v1" / "fields"
    assert not fields_root.exists()


def test_cli_precomputes_all_root_faces_and_resumes(tmp_path, capsys):
    _write_world(tmp_path)
    argv = [
        "--world", str(tmp_path),
        "--tile-size", "16",
        "--maximum-level", "5",
        "--precompute-depth", "0",
        "--precompute-workers", "2",
        "--precompute-max-tiles", "20",
        "--precompute-max-gib", "1",
        "--json",
    ]
    assert main(argv) == 0
    first_io = capsys.readouterr()
    first = json.loads(first_io.out)
    assert first["precompute"]["total_tiles"] == 6
    assert first["precompute"]["base_generated_tiles"] == 6
    assert "precompute 6/6" in first_io.err

    assert main(argv) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["precompute"]["base_cache_hits"] == 6
    assert second["precompute"]["base_generated_tiles"] == 0


def test_cli_requires_explicit_override_for_large_prefix(tmp_path):
    _write_world(tmp_path)
    with pytest.raises(SystemExit):
        main([
            "--world", str(tmp_path),
            "--tile-size", "16",
            "--maximum-level", "5",
            "--precompute-depth", "2",
            "--precompute-max-tiles", "10",
            "--precompute-max-gib", "100",
        ])
