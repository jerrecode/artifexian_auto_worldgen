from __future__ import annotations

import tempfile
import unittest

from worldgen.checkpoint import CheckpointStore


class CheckpointTransactionTests(unittest.TestCase):
    def test_oversized_replacement_preserves_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CheckpointStore(tmp, max_bytes=512)
            old = {"value": "small"}
            first = store.put("demo", "same-key", old)
            self.assertIsNotNone(first)
            self.assertEqual(store.get("same-key"), old)

            oversized = {"value": "x" * 4096}
            self.assertIsNone(store.put("demo", "same-key", oversized))
            self.assertEqual(store.get("same-key"), old)
            self.assertEqual(store.stats().entries, 1)
            store.close()

    def test_replacement_uses_new_payload_file_and_removes_old_after_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CheckpointStore(tmp, max_bytes=1024 * 1024)
            first = store.put("demo", "key", {"value": 1})
            self.assertIsNotNone(first)
            assert first is not None
            first_path = first.path
            second = store.put("demo", "key", {"value": 2})
            self.assertIsNotNone(second)
            assert second is not None
            self.assertNotEqual(first_path, second.path)
            self.assertFalse(first_path.exists())
            self.assertTrue(second.path.exists())
            self.assertEqual(store.get("key"), {"value": 2})
            store.close()


if __name__ == "__main__":
    unittest.main()
