from __future__ import annotations

import tempfile
import unittest

import numpy as np

from worldgen.storage import MappedArrayStore, store_array_mapping


class InjectedFailure(RuntimeError):
    pass


class StorageRegressionTests(unittest.TestCase):
    def test_oversized_replacement_preserves_existing_array(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MappedArrayStore(tmp, max_bytes=256, persistent=True)
            small = np.arange(8, dtype=np.float32)
            original = store.put("field", small)
            self.assertLessEqual(original.file_bytes, 256)
            large = np.arange(1024, dtype=np.float32)
            with self.assertRaises(RuntimeError):
                store.put("field", large)
            mapped = store.open("field")
            np.testing.assert_array_equal(np.asarray(mapped), small)
            self.assertEqual(store.info("field").shape, small.shape)
            store.close()

    def test_failed_mapping_population_closes_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            arrays = {
                "small": np.arange(8, dtype=np.float32),
                "too-large": np.arange(1024, dtype=np.float32),
            }
            with self.assertRaises(RuntimeError):
                store_array_mapping(arrays, tmp, max_bytes=256)
            reopened = MappedArrayStore(tmp, max_bytes=256, persistent=True)
            self.assertIn("small", reopened.keys())
            reopened.close()

    def _replacement_failure_case(self, stage: str, *, expect_new: bool) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old = np.arange(16, dtype=np.float32)
            new = np.arange(16, dtype=np.float32) + 100.0
            store = MappedArrayStore(tmp, max_bytes=1024 * 1024, persistent=True)
            old_info = store.put("field", old)
            old_path = old_info.path
            store.close()

            def failpoint(name: str) -> None:
                if name == stage:
                    raise InjectedFailure(stage)

            store = MappedArrayStore(
                tmp,
                max_bytes=1024 * 1024,
                persistent=True,
                failure_injector=failpoint,
            )
            with self.assertRaises(InjectedFailure):
                store.put("field", new)
            store.close()

            # Constructor reconciliation represents the recovery path after a crash.
            recovered = MappedArrayStore(tmp, max_bytes=1024 * 1024, persistent=True)
            got = np.asarray(recovered.open("field"))
            np.testing.assert_array_equal(got, new if expect_new else old)
            info = recovered.info("field")
            self.assertIsNotNone(info)
            payloads = list(recovered.data_dir.glob("*.npy"))
            self.assertEqual(len(payloads), 1, payloads)
            self.assertEqual(payloads[0], info.path)
            if expect_new:
                self.assertNotEqual(info.path, old_path)
            recovered.close()

    def test_failure_before_object_write_preserves_old_value(self):
        self._replacement_failure_case("before_object_write", expect_new=False)

    def test_failure_after_object_write_preserves_old_value(self):
        self._replacement_failure_case("after_object_write", expect_new=False)

    def test_failure_after_object_publish_preserves_old_value_and_reconciles_orphan(self):
        self._replacement_failure_case("after_object_publish", expect_new=False)

    def test_failure_before_db_commit_preserves_old_value_and_reconciles_orphan(self):
        self._replacement_failure_case("before_db_commit", expect_new=False)

    def test_failure_after_db_commit_preserves_new_value(self):
        self._replacement_failure_case("after_db_commit", expect_new=True)

    def test_failure_during_old_object_delete_preserves_new_value(self):
        self._replacement_failure_case("during_old_object_delete", expect_new=True)


if __name__ == "__main__":
    unittest.main()
