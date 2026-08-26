import unittest

import numpy as np

from worldgen.grid import SphereGrid, distance_to


class SpatialCacheTests(unittest.TestCase):
    def test_distance_field_cache_hits_and_is_bounded(self):
        grid = SphereGrid(128, 64, distance_cache_max_bytes=2 * 1024 * 1024)
        mask = np.zeros((64, 128), dtype=bool)
        mask[20:40, 50:70] = True
        a = distance_to(mask, grid)
        before = grid.spatial_cache_stats()
        b = distance_to(mask.copy(), grid)
        after = grid.spatial_cache_stats()
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(after.hits, before.hits + 1)
        self.assertLessEqual(after.bytes_used, after.max_bytes)

    def test_cache_can_be_disabled(self):
        grid = SphereGrid(64, 32, distance_cache_max_bytes=0)
        mask = np.zeros((32, 64), dtype=bool)
        mask[15, 10] = True
        distance_to(mask, grid)
        self.assertEqual(grid.spatial_cache_stats().items, 0)


if __name__ == "__main__":
    unittest.main()
