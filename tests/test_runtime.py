import io
import tempfile
import unittest

import numpy as np

from worldgen.cache import ByteBoundLRUCache
from worldgen.cli import main
from worldgen.progress import StageProgress
from worldgen.runtime import resolve_runtime_plan
from worldgen.storage import MappedArrayStore


class RuntimeInfrastructureTests(unittest.TestCase):
    def test_managed_worker_plan_is_capped(self):
        plan = resolve_runtime_plan(workers=999, worker_cap=3, reserve_cpus=0, memory_per_worker_mb=64)
        self.assertGreaterEqual(plan.workers, 1)
        self.assertLessEqual(plan.workers, 3)

    def test_byte_bound_lru_evicts_oldest(self):
        cache = ByteBoundLRUCache[str, bytes](max_bytes=6)
        self.assertTrue(cache.put("a", b"1234", size_bytes=4))
        self.assertTrue(cache.put("b", b"5678", size_bytes=4))
        self.assertIsNone(cache.get("a"))
        self.assertEqual(cache.get("b"), b"5678")
        self.assertGreaterEqual(cache.stats().evictions, 1)

    def test_mapped_array_store_random_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MappedArrayStore(tmp, max_bytes=1024 * 1024, persistent=True)
            source = np.arange(4096, dtype=np.float32).reshape(64, 64)
            store.put("height", source)
            mapped = store.open("height")
            self.assertIsInstance(mapped, np.memmap)
            self.assertEqual(float(mapped[17, 29]), float(source[17, 29]))
            self.assertEqual(store.info("height").shape, (64, 64))
            store.close()

    def test_stage_progress_snapshot(self):
        stream = io.StringIO()
        progress = StageProgress(2, stream=stream)
        progress("[one] starting")
        progress("[one] done in 1.000s")
        snap = progress.snapshot()
        self.assertEqual(snap.completed, 1)
        self.assertEqual(snap.total, 2)
        self.assertIsNotNone(snap.eta_seconds)

    def test_cli_dry_run_accepts_generic_overrides(self):
        rc = main([
            "--dry-run",
            "--no-progress",
            "--resolution", "128x64",
            "--set", "climate.moisture_iterations=7",
            "--workers", "1",
        ])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
