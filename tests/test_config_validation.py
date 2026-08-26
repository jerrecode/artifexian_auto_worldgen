import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from worldgen.cli import main
from worldgen.config import ConfigValidationError, WorldConfig, config_schema


class ConfigurationValidationTests(unittest.TestCase):
    def test_default_configuration_is_valid(self):
        self.assertIsInstance(WorldConfig().validate(), WorldConfig)

    def test_invalid_physical_fraction_is_rejected(self):
        cfg = WorldConfig()
        cfg.tectonics.parent_coupling = 4.0
        with self.assertRaises(ConfigValidationError):
            cfg.validate()

    def test_invalid_cli_override_fails_during_validation(self):
        with self.assertRaises(SystemExit):
            main(["--dry-run", "--no-progress", "--set", "noise.persistence=-2"])

    def test_config_schema_has_units_and_defaults(self):
        schema = config_schema()
        self.assertIn("astronomy.rotation_hours", schema)
        self.assertEqual(schema["astronomy.rotation_hours"]["unit"], "h")
        self.assertIn("default", schema["astronomy.rotation_hours"])

    def test_config_subcommand_validate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "world.yml"
            path.write_text("seed: 42\nresolution:\n  width: 128\n  height: 64\n", encoding="utf-8")
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = main(["config", "validate", str(path)])
            self.assertEqual(rc, 0)
            payload = json.loads(buf.getvalue())
            self.assertTrue(payload["valid"])


if __name__ == "__main__":
    unittest.main()
