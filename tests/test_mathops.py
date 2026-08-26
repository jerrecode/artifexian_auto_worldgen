import unittest
import numpy as np

from worldgen.mathops import auto_chunk_shape, compensated_sum, fast_hypot, iter_tiles_2d, weighted_mean_stable


class MathOpsTests(unittest.TestCase):
    def test_chunk_shape_and_tiles_cover_grid(self):
        shape = (31, 62)
        chunk = auto_chunk_shape(shape, np.float32, target_mb=0.01, arrays_in_flight=2, minimum_rows=1)
        seen = np.zeros(shape, dtype=np.uint8)
        for tile in iter_tiles_2d(shape, chunk_shape=chunk, halo=2):
            seen[tile.core] += 1
        self.assertTrue(np.all(seen == 1))

    def test_fast_hypot_matches_numpy(self):
        a = np.arange(20, dtype=float).reshape(4, 5)
        b = a[::-1]
        np.testing.assert_allclose(fast_hypot(a, b), np.hypot(a, b))

    def test_compensated_sum_and_weighted_mean(self):
        values = np.array([1e16, 1.0, -1e16, 3.0])
        self.assertAlmostEqual(compensated_sum(values), 4.0)
        v = np.array([1.0, 2.0, 10.0])
        w = np.array([1.0, 1.0, 0.0])
        self.assertAlmostEqual(weighted_mean_stable(v, w), 1.5)


if __name__ == "__main__":
    unittest.main()
