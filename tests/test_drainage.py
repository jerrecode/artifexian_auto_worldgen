import unittest

import numpy as np

from worldgen.drainage import DrainageGraph, accumulate, strahler_order


class DrainageGraphTests(unittest.TestCase):
    def test_linear_accumulation(self):
        # 0 -> 1 -> 2 -> sink
        flow = np.array([1, 2, -1], dtype=np.int32)
        z = np.array([[3.0, 2.0, 1.0]])
        source = np.array([[1.0, 1.0, 1.0]])
        out = accumulate(z, flow, source)
        np.testing.assert_allclose(out, [[1.0, 2.0, 3.0]])

    def test_branch_strahler_order(self):
        # 0 and 1 join at 2; 2 drains to 3.
        flow = np.array([2, 2, 3, -1], dtype=np.int32)
        channel = np.array([[True, True, True, True]])
        order = strahler_order(flow, channel)
        self.assertEqual(int(order.ravel()[0]), 1)
        self.assertEqual(int(order.ravel()[1]), 1)
        self.assertEqual(int(order.ravel()[2]), 2)
        self.assertEqual(int(order.ravel()[3]), 2)

    def test_graph_detects_cycles_as_invariant_failure(self):
        graph = DrainageGraph.from_receiver(np.array([1, 0], dtype=np.int32))
        self.assertEqual(graph.unresolved_count, 2)


if __name__ == "__main__":
    unittest.main()
