from __future__ import annotations

from pathlib import Path

import numpy as np

from worldgen.planet_tiles import CUBE_FACES, TileKey
from worldgen.ultrares import UltraResolutionPlan
from worldgen.ultrares_maps import (
    KOPPEN_TO_CODE,
    _classify_koppen_geographic,
    _koppen_codes,
    _ramp,
    _sample_deepest_to_npy,
)


def test_koppen_tile_classifier_uses_geographic_latitude():
    temp = np.full((12, 4, 6), 24.0, dtype=np.float64)
    precip = np.full_like(temp, 120.0)
    lat = np.array(
        [
            [70.0] * 6,
            [20.0] * 6,
            [-20.0] * 6,
            [-70.0] * 6,
        ]
    )
    out = _classify_koppen_geographic(temp, precip, lat)
    assert out.shape == (4, 6)
    assert np.all(out == "Af")
    codes = _koppen_codes(out)
    assert np.all(codes == KOPPEN_TO_CODE["Af"])


def test_color_ramp_exact_endpoints():
    anchors = (
        (0.0, (10, 20, 30)),
        (0.5, (100, 110, 120)),
        (1.0, (240, 245, 250)),
    )
    values = np.array([[0.0, 0.5, 1.0]])
    rgb = _ramp(values, anchors)
    np.testing.assert_array_equal(rgb[0, 0], (10, 20, 30))
    np.testing.assert_array_equal(rgb[0, 1], (100, 110, 120))
    np.testing.assert_array_equal(rgb[0, 2], (240, 245, 250))


def test_deepest_cube_tiles_reproject_to_fullview_without_missing_pixels(tmp_path: Path):
    plan = UltraResolutionPlan(
        source_width=4,
        source_height=2,
        source_equatorial_m_per_sample=100.0,
        fullview_width=16,
        fullview_height=8,
        base_level=0,
        finest_level=1,
        base_m_per_sample=25.0,
        finest_m_per_sample=12.5,
        actual_base_multiplier=4.0,
        actual_subsection_multiplier=2.0,
        finest_tile_count=24,
        tile_size=4,
    )
    face_index = {face: i for i, face in enumerate(CUBE_FACES)}
    for face in CUBE_FACES:
        for y in range(2):
            for x in range(2):
                key = TileKey(face, 1, x, y)
                path = (
                    tmp_path
                    / face
                    / f"x{x}"
                    / f"y{y}.npy"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                value = float(face_index[face] * 100 + y * 10 + x)
                np.save(path, np.full((5, 5), value, dtype=np.float32))

    def resolver(key: TileKey) -> Path:
        return tmp_path / key.face / f"x{key.x}" / f"y{key.y}.npy"

    out_path = tmp_path / "fullview.npy"
    _sample_deepest_to_npy(
        plan,
        resolver,
        out_path,
        mode="nearest",
        output_dtype="float32",
        chunk_rows=3,
    )
    out = np.load(out_path, allow_pickle=False)
    assert out.shape == (8, 16)
    assert np.isfinite(out).all()
    valid = {
        float(face_index[face] * 100 + y * 10 + x)
        for face in CUBE_FACES
        for y in range(2)
        for x in range(2)
    }
    assert set(np.unique(out)).issubset(valid)
    assert len(np.unique(out)) >= 6


def test_deepest_reprojection_supports_native_resolution_and_value_scaling(tmp_path: Path):
    plan = UltraResolutionPlan(
        source_width=4,
        source_height=2,
        source_equatorial_m_per_sample=100.0,
        fullview_width=8,
        fullview_height=4,
        base_level=0,
        finest_level=0,
        base_m_per_sample=25.0,
        finest_m_per_sample=25.0,
        actual_base_multiplier=4.0,
        actual_subsection_multiplier=1.0,
        finest_tile_count=6,
        tile_size=4,
    )
    for face_i, face in enumerate(CUBE_FACES):
        path = tmp_path / face / "x0" / "y0.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.full((5, 5), 1000.0 + 100.0 * face_i, dtype=np.float64))

    def resolver(key: TileKey) -> Path:
        return tmp_path / key.face / "x0" / "y0.npy"

    out_path = tmp_path / "native.npy"
    _sample_deepest_to_npy(
        plan,
        resolver,
        out_path,
        mode="nearest",
        output_dtype="float64",
        width=16,
        height=8,
        value_scale=0.001,
        chunk_rows=2,
    )
    out = np.load(out_path, allow_pickle=False)
    assert out.shape == (8, 16)
    assert out.dtype == np.float64
    assert np.isfinite(out).all()
    assert float(out.min()) >= 1.0
    assert float(out.max()) <= 1.5
