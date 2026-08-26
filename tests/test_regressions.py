from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np
from matplotlib import image as mpl_image

from worldgen.cache import ByteBoundLRUCache
from worldgen.checkpoint import CheckpointStore
from worldgen.config import WorldConfig
from worldgen.render import _save_rgb


class RegressionTests(unittest.TestCase):
    def test_rejected_oversized_cache_replacement_preserves_old_value(self):
        cache = ByteBoundLRUCache[str, bytes](max_bytes=8)
        self.assertTrue(cache.put("field", b"1234", size_bytes=4))
        self.assertFalse(cache.put("field", b"0123456789", size_bytes=10))
        self.assertEqual(cache.get("field"), b"1234")
        stats = cache.stats()
        self.assertEqual(stats.items, 1)
        self.assertEqual(stats.bytes_used, 4)

    def test_corrupt_checkpoint_is_invalidated_instead_of_unpickled(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CheckpointStore(tmp, max_bytes=1024 * 1024)
            info = store.put("stage", "abc123", {"ok": True})
            self.assertIsNotNone(info)
            assert info is not None
            info.path.write_bytes(b"corrupt payload")
            self.assertIsNone(store.get("abc123"))
            self.assertIsNone(store.info("abc123"))
            self.assertFalse(info.path.exists())
            store.close()

    def test_true_color_writer_preserves_simulation_pixel_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rgb.png"
            rgb = np.zeros((17, 34, 3), dtype=np.uint8)
            rgb[..., 0] = 120
            _save_rgb(path, rgb, dpi=91)
            decoded = mpl_image.imread(path)
            self.assertEqual(decoded.shape[:2], (17, 34))

    def test_world_config_rejects_invalid_algorithmic_domains(self):
        cfg = WorldConfig()
        cfg.noise.persistence = 1.0
        with self.assertRaises(ValueError):
            cfg.validate()
        cfg = WorldConfig()
        cfg.hydrology.flow_refresh_interval = 0
        with self.assertRaises(ValueError):
            cfg.validate()


if __name__ == "__main__":
    unittest.main()
