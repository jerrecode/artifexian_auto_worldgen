import unittest

import numpy as np

from worldgen.diagnostics import array_digest, receiver_graph_is_acyclic
from worldgen.grid import SphereGrid, distance_to
from worldgen.weather import _large_land_mask


class SphericalTopologyTests(unittest.TestCase):
    def test_longitude_seam_is_adjacent(self):
        grid = SphereGrid(64, 32)
        mask = np.zeros((32, 64), dtype=bool)
        y = 16
        mask[y, 0] = True
        dist = distance_to(mask, grid)
        # The last longitude column is one cell west of column zero, not almost a
        # complete circumference away.
        self.assertLess(dist[y, -1], 1.6 * float(grid.dx_km[y, -1]))

    def test_pole_crossing_rotates_longitude(self):
        grid = SphereGrid(64, 32)
        a = np.arange(32 * 64, dtype=np.int32).reshape(32, 64)
        north = grid.ops.shift(a, -1, 0)
        # Moving north from the top row reflects onto that row at antipodal longitude.
        self.assertEqual(int(north[0, 0]), int(a[0, 32]))
        self.assertEqual(int(north[0, 7]), int(a[0, 39]))

    def test_connected_components_merge_across_seam(self):
        grid = SphereGrid(64, 32)
        mask = np.zeros((32, 64), dtype=bool)
        mask[12:15, 0] = True
        mask[12:15, -1] = True
        labels, n = grid.ops.connected_components(mask)
        self.assertEqual(n, 1)
        self.assertEqual(int(labels[13, 0]), int(labels[13, -1]))

    def test_grey_dilation_wraps_longitude_seam(self):
        grid = SphereGrid(64, 32)
        field = np.zeros((32, 64), dtype=np.float32)
        field[16, 0] = 7.0
        expanded = grid.ops.grey_dilation(field, iterations=1)
        self.assertEqual(float(expanded[16, -1]), 7.0)
        self.assertEqual(float(expanded[15, -1]), 7.0)

    def test_grey_dilation_crosses_pole_with_antipodal_rotation(self):
        grid = SphereGrid(64, 32)
        field = np.zeros((32, 64), dtype=np.float32)
        field[0, 32] = 9.0
        expanded = grid.ops.grey_dilation(field, iterations=1)
        # The north neighbor of top-row longitude 0 is the reflected top-row
        # cell at longitude +180 degrees (half a raster away).
        self.assertEqual(float(expanded[0, 0]), 9.0)

    def test_large_land_mask_merges_seam_component_and_uses_spherical_area(self):
        grid = SphereGrid(64, 32)
        land = np.zeros((32, 64), dtype=bool)
        # 32 equatorial cells split evenly across the map edges. Each planar half
        # is below the legacy 30-cell threshold, but the spherical component is not.
        land[8:24, 0] = True
        land[8:24, -1] = True
        large = _large_land_mask(grid, land)
        self.assertTrue(large[16, 0])
        self.assertTrue(large[16, -1])

    def test_metric_gradient_is_finite_at_poles(self):
        grid = SphereGrid(128, 64)
        field = np.sin(np.deg2rad(grid.lat)) + 0.2 * np.cos(np.deg2rad(grid.lon))
        gy, gx = grid.ops.metric_gradient(field)
        self.assertTrue(np.isfinite(gy).all())
        self.assertTrue(np.isfinite(gx).all())

    def test_receiver_cycle_detection(self):
        self.assertTrue(receiver_graph_is_acyclic(np.array([-1, 0, 1, 2], dtype=np.int32)))
        self.assertFalse(receiver_graph_is_acyclic(np.array([1, 2, 0], dtype=np.int32)))

    def test_array_digest_includes_shape(self):
        a = np.arange(12, dtype=np.float32)
        self.assertNotEqual(array_digest(a), array_digest(a.reshape(3, 4)))


if __name__ == "__main__":
    unittest.main()
