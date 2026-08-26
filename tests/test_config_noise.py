import unittest

import numpy as np

from worldgen.config import WorldConfig
from worldgen.noise import (
    _bilinear_warp_tiled,
    _natural_octave_geometry,
    hybrid_multifractal,
)


class ConfigAndNoiseTests(unittest.TestCase):
    def test_default_config_validates(self):
        self.assertIsInstance(WorldConfig().validate(), WorldConfig)

    def test_invalid_config_is_rejected(self):
        cfg = WorldConfig()
        cfg.noise.persistence = 1.4
        with self.assertRaises(ValueError):
            cfg.validate()

        cfg = WorldConfig()
        cfg.resolution.width = 777
        with self.assertRaises(ValueError):
            cfg.validate()

        cfg = WorldConfig()
        cfg.hydrology.flow_refresh_interval = 0
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_large_scale_octave_uses_coarser_synthesis_grid(self):
        shape, local_sigma, factor = _natural_octave_geometry((1024, 2048), 96.0)
        self.assertLess(shape[0], 1024)
        self.assertLess(shape[1], 2048)
        self.assertGreater(factor, 1.0)
        self.assertGreater(local_sigma, 0.0)

    def test_noise_is_deterministic_and_standardized(self):
        a = hybrid_multifractal((96, 192), np.random.default_rng(1234), octaves=6, base_scale_px=28)
        b = hybrid_multifractal((96, 192), np.random.default_rng(1234), octaves=6, base_scale_px=28)
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(a.dtype, np.float32)
        self.assertLess(abs(float(a.mean())), 1e-5)
        self.assertAlmostEqual(float(a.std()), 1.0, delta=0.03)

    def test_tiled_warp_handles_periodic_seam(self):
        h, w = 64, 128
        field = np.tile(np.arange(w, dtype=np.float32)[None, :], (h, 1))
        dy = np.zeros((h, w), np.float32)
        dx = np.full((h, w), 2.0, np.float32)
        warped = _bilinear_warp_tiled(field, dy, dx, target_scratch_mb=0.25)
        self.assertEqual(float(warped[20, w - 1]), 1.0)
        self.assertEqual(float(warped[20, 0]), 2.0)


if __name__ == "__main__":
    unittest.main()
