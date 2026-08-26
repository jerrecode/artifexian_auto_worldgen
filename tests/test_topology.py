import unittest

import numpy as np

from worldgen.grid import (
    SphereGrid,
    connected_components_spherical,
    distance_to,
    spherical_shift,
)


class SphericalTopologyTests(unittest.TestCase):
    def test_distance_crosses_longitude_seam_geodesically(self):
        grid = SphereGrid(128, 64, distance_cache_bytes=8 * 1024**2)
        mask = np.zeros((64, 128), dtype=bool)
        mask[32, 0] = True
        dist = distance_to(mask, grid)
        self.assertLess(float(dist[32, -1]), 400.0)
        self.assertEqual(float(dist[32, 0]), 0.0)
        self.assertGreaterEqual(grid.geometry_cache_stats().items, 1)

    def test_north_shift_reflects_and_rotates_longitude(self):
        h, w = 8, 16
        a = np.arange(h * w).reshape(h, w)
        shifted = spherical_shift(a, -1, 0)
        expected = np.roll(a[0], w // 2)
        np.testing.assert_array_equal(shifted[0], expected)

    def test_connected_components_merge_longitude_seam(self):
        m = np.zeros((16, 32), dtype=bool)
        m[8, 0] = True
        m[8, -1] = True
        labels, n = connected_components_spherical(m)
        self.assertEqual(n, 1)
        self.assertEqual(int(labels[8, 0]), int(labels[8, -1]))

    def test_connected_components_merge_across_pole(self):
        h, w = 16, 32
        m = np.zeros((h, w), dtype=bool)
        m[0, 2] = True
        m[0, 2 + w // 2] = True
        _, n = connected_components_spherical(m)
        self.assertEqual(n, 1)


if __name__ == "__main__":
    unittest.main()
