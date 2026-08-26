from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from worldgen.config import WorldConfig
from worldgen.manifest import build_run_manifest, write_run_manifest


class ManifestContractTests(unittest.TestCase):
    def test_manifest_exposes_world_identity_at_top_level(self):
        cfg = WorldConfig(seed=12345)
        cfg.resolution.width = 128
        cfg.resolution.height = 64
        manifest = build_run_manifest(config=cfg)
        self.assertEqual(manifest["seed"], 12345)
        self.assertEqual(manifest["resolution"], [128, 64])
        self.assertEqual(manifest["config_sha256"], manifest["reproducibility"]["config_sha256"])
        self.assertEqual(manifest["reproducibility"]["seed"], 12345)

    def test_written_manifest_has_same_contract(self):
        cfg = WorldConfig(seed=777)
        cfg.resolution.width = 128
        cfg.resolution.height = 64
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            written = write_run_manifest(path, config=cfg)
            self.assertEqual(written["seed"], 777)
            self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
