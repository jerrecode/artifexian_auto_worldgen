from __future__ import annotations

import tempfile
import unittest

import numpy as np

from worldgen.storage import MappedArrayStore, store_array_mapping


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
            # A fresh connection must be able to reopen the database immediately.
            reopened = MappedArrayStore(tmp, max_bytes=256, persistent=True)
            self.assertIn("small", reopened.keys())
            reopened.close()


if __name__ == "__main__":
    unittest.main()
