import unittest

import numpy as np

from worldgen.drainage import DrainageGraph, topological_order


class DrainageGraphTests(unittest.TestCase):
    def test_accumulation_reuses_topology(self):
        # 0 -> 2, 1 -> 2, 2 -> 3, 3 outlet
        recv = np.array([2, 2, 3, -1], dtype=np.int32)
        graph = DrainageGraph.from_receiver(recv, (2, 2))
        acc = graph.accumulate(np.ones((2, 2), dtype=np.float64))
        self.assertEqual(float(acc.ravel()[2]), 3.0)
        self.assertEqual(float(acc.ravel()[3]), 4.0)

    def test_strahler_order(self):
        recv = np.array([2, 2, 3, -1], dtype=np.int32)
        graph = DrainageGraph.from_receiver(recv, (2, 2))
        order = graph.strahler_order(np.ones((2, 2), dtype=bool)).ravel()
        self.assertEqual(int(order[0]), 1)
        self.assertEqual(int(order[1]), 1)
        self.assertEqual(int(order[2]), 2)
        self.assertEqual(int(order[3]), 2)

    def test_basin_propagation(self):
        recv = np.array([2, 2, 3, -1], dtype=np.int32)
        graph = DrainageGraph.from_receiver(recv, (2, 2))
        terminal = np.array([[0, 0], [0, 7]], dtype=np.int32)
        roots = graph.basin_roots(terminal)
        self.assertTrue(np.all(roots == 7))

    def test_cycle_is_rejected(self):
        with self.assertRaises(ValueError):
            topological_order(np.array([1, 2, 0], dtype=np.int32))

    def test_donor_csr(self):
        recv = np.array([2, 2, 3, -1], dtype=np.int32)
        graph = DrainageGraph.from_receiver(recv, (2, 2))
        offsets, donors = graph.donor_csr()
        target_two = set(map(int, donors[offsets[2]:offsets[3]]))
        self.assertEqual(target_two, {0, 1})


if __name__ == "__main__":
    unittest.main()
