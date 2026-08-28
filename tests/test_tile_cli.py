from __future__ import annotations

import json

import numpy as np

from worldgen.tile_cli import main


def _world(root):
    h, w = 8, 16
    lat = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    lon = -180.0 + (np.arange(w) + 0.5) * 360.0 / w
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    elevation = (0.6 * np.sin(2 * np.pi * xx / w) + 0.2 * np.cos(np.pi * yy / h)).astype(np.float32)
    np.savez(root / "world_arrays.npz", lat=lat, lon=lon, elevation_km=elevation)
    (root / "world.json").write_text(
        json.dumps({"seed": 7, "astronomy": {"planet": {"radius_earth": 1.0}}}),
        encoding="utf-8",
    )


def test_cli_generates_addressed_tile_and_mesh(tmp_path, capsys):
    _world(tmp_path)
    rc = main(
        [
            "--world",
            str(tmp_path),
            "--tile-size",
            "16",
            "--tile",
            "px/4/6/7",
            "--field",
            "elevation_m",
            "--mesh",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tiles"][0]["key"] == {"face": "px", "level": 4, "x": 6, "y": 7}
    assert payload["tiles"][0]["mesh"].endswith(".npz")


def test_cli_resolves_zoom_from_requested_ground_resolution(tmp_path, capsys):
    _world(tmp_path)
    rc = main(
        [
            "--world",
            str(tmp_path),
            "--tile-size",
            "32",
            "--at",
            "48.2,16.37",
            "--meters-per-sample",
            "1000",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tiles"]
    assert payload["tiles"][0]["meters_per_sample_approx"] <= 1000.0


def test_cli_visible_request_materializes_only_selected_cap(tmp_path, capsys):
    _world(tmp_path)
    rc = main(
        [
            "--world",
            str(tmp_path),
            "--tile-size",
            "16",
            "--visible",
            "48.2,16.37",
            "--level",
            "8",
            "--angular-radius-deg",
            "0.2",
            "--maximum-visible-tiles",
            "64",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert 0 < payload["visible_request"]["tile_count"] < 64
    assert len(payload["tiles"]) == payload["visible_request"]["tile_count"]
