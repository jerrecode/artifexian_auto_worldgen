import tempfile
import unittest

import numpy as np

from worldgen.checkpoint import CheckpointStore, package_source_fingerprint, stage_cache_key
from worldgen.config import load_config
from worldgen.resumable import ResumableWorldPipeline


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_store_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CheckpointStore(tmp, max_bytes=8 * 1024 * 1024)
            value = {"a": np.arange(32, dtype=np.float32)}
            store.put("demo", "abc", value)
            loaded = store.get("abc")
            self.assertTrue(np.array_equal(loaded["a"], value["a"]))
            self.assertEqual(store.stats().hits, 1)
            store.close()

    def test_stage_key_changes_with_config(self):
        cfg = load_config(None)
        source = package_source_fingerprint()
        a = stage_cache_key("climate", cfg.to_dict(), source)
        cfg.climate.moisture_iterations += 1
        b = stage_cache_key("climate", cfg.to_dict(), source)
        self.assertNotEqual(a, b)

    def test_resumable_pipeline_reuses_stage(self):
        cfg = load_config(None)
        with tempfile.TemporaryDirectory() as tmp:
            first = ResumableWorldPipeline(cfg, progress=lambda _: None, checkpoint_dir=tmp)
            value = first._stage("unit-test-stage", lambda: {"x": np.arange(8, dtype=np.int16)})
            first.close()

            second = ResumableWorldPipeline(cfg, progress=lambda _: None, checkpoint_dir=tmp)
            loaded = second._stage("unit-test-stage", lambda: self.fail("checkpoint was not reused"))
            self.assertTrue(np.array_equal(value["x"], loaded["x"]))
            self.assertIn("unit-test-stage", second.checkpoint_hits)
            second.close()


if __name__ == "__main__":
    unittest.main()
