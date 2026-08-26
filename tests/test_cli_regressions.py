from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from worldgen.cli import main


class CliRegressionTests(unittest.TestCase):
    def test_invalid_generic_override_is_rejected_after_merge(self):
        with self.assertRaises(SystemExit) as ctx:
            main([
                "--dry-run",
                "--no-progress",
                "--resolution", "128x64",
                "--set", "noise.persistence=1.4",
                "--workers", "1",
            ])
        self.assertEqual(ctx.exception.code, 2)

    def test_invalid_override_type_is_rejected_after_merge(self):
        with self.assertRaises(SystemExit) as ctx:
            main([
                "--dry-run",
                "--no-progress",
                "--resolution", "128x64",
                "--set", "climate.moisture_iterations=7.5",
                "--workers", "1",
            ])
        self.assertEqual(ctx.exception.code, 2)

    def test_malformed_yaml_config_reports_cli_error_not_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text("noise: [not, a, mapping]\n", encoding="utf-8")
            with self.assertRaises(SystemExit) as ctx:
                main(["--dry-run", "--no-progress", "--config", str(path)])
            self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
