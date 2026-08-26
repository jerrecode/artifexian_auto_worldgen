import unittest

import numpy as np

from worldgen.field_schema import build_field_catalog


class FieldSchemaTests(unittest.TestCase):
    def test_dimensions_and_units_are_inferred(self):
        arrays = {
            "lat": np.linspace(80, -80, 4),
            "lon": np.linspace(-180, 135, 8),
            "annual_temperature_c": np.zeros((4, 8), dtype=np.float32),
            "temperature_c_monthly": np.zeros((12, 4, 8), dtype=np.float32),
            "true_color_rgb": np.zeros((4, 8, 3), dtype=np.float32),
        }
        catalog = build_field_catalog(arrays)
        self.assertEqual(catalog["annual_temperature_c"]["dimensions"], ["lat", "lon"])
        self.assertEqual(catalog["annual_temperature_c"]["units"], "degC")
        self.assertEqual(catalog["temperature_c_monthly"]["dimensions"], ["month", "lat", "lon"])
        self.assertEqual(catalog["true_color_rgb"]["dimensions"], ["lat", "lon", "channel"])


if __name__ == "__main__":
    unittest.main()
