import numpy as np

from worldgen.grid import SphereGrid
from worldgen.tectonics import _classify_subplate_boundaries, _resize


def _pair_matrices() -> tuple[np.ndarray, np.ndarray]:
    pair_type = np.zeros((2, 2), dtype=np.int8)
    pair_type[0, 1] = pair_type[1, 0] = -1
    pair_strength = np.zeros((2, 2), dtype=np.float32)
    pair_strength[0, 1] = pair_strength[1, 0] = 0.8
    return pair_type, pair_strength


def test_subplate_boundary_classification_crosses_north_pole():
    grid = SphereGrid(16, 8)
    sub = np.zeros((8, 16), dtype=np.int16)
    # These two cells touch through the north-pole reflection (+180 degrees).
    sub[0, 8] = 1
    parent = np.array([0, 1], dtype=np.int16)
    pair_type, pair_strength = _pair_matrices()

    boundary, _, convergent, _, _, conv_strength, _, _ = _classify_subplate_boundaries(
        grid, sub, parent, pair_type, pair_strength
    )

    assert boundary[0, 0]
    assert boundary[0, 8]
    assert convergent[0, 0]
    assert conv_strength[0, 0] == np.float32(0.8)


def test_subplate_boundary_classification_crosses_longitude_seam():
    grid = SphereGrid(16, 8)
    sub = np.zeros((8, 16), dtype=np.int16)
    sub[4, -1] = 1
    parent = np.array([0, 1], dtype=np.int16)
    pair_type, pair_strength = _pair_matrices()

    boundary, _, convergent, _, _, _, _, _ = _classify_subplate_boundaries(
        grid, sub, parent, pair_type, pair_strength
    )

    assert boundary[4, 0]
    assert convergent[4, 0]


def test_spherical_resize_preserves_constant_field_and_dtype_shape():
    source = np.full((7, 14), 3.25, dtype=np.float32)
    resized = _resize(source, (13, 26), order=1)
    assert resized.shape == (13, 26)
    assert np.allclose(resized, 3.25)


def test_spherical_nearest_resize_does_not_create_invalid_labels():
    source = np.zeros((5, 10), dtype=np.int16)
    source[:, 5:] = 1
    resized = _resize(source, (11, 22), order=0)
    assert resized.shape == (11, 22)
    assert set(np.unique(resized)).issubset({0, 1})
