import unittest

import numpy as np

from worldgen.noise import _native_shape, hybrid_multifractal


class NoiseScalingTests(unittest.TestCase):
    def test_coarse_octave_uses_smaller_native_grid(self):
        self.assertLess(_native_shape((1024, 2048), 128.0)[0], 1024)
        self.assertLess(_native_shape((1024, 2048), 128.0)[1], 2048)

    def test_native_resolution_noise_is_deterministic(self):
        a = hybrid_multifractal((96, 192), np.random.default_rng(1234), octaves=6, base_scale_px=28)
        b = hybrid_multifractal((96, 192), np.random.default_rng(1234), octaves=6, base_scale_px=28)
        np.testing.assert_allclose(a, b)
        self.assertGreater(float(np.std(a)), 0.5)


if __name__ == "__main__":
    unittest.main()
