from __future__ import annotations

import copy
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from worldgen.atmogen_coupler import (
    annual_stellar_flux_scale,
    cluster_representative_states,
    solve_representative_columns,
)
from worldgen.config import load_config
from worldgen.fingerprints import stage_source_files
from worldgen.grid import SphereGrid
from worldgen.resumable import ResumableWorldPipeline


class AtmogenCouplingTests(unittest.TestCase):
    def test_annual_stellar_flux_scale_has_unit_spherical_mean(self):
        grid = SphereGrid(64, 32, 6371.0)
        astronomy = SimpleNamespace(planet={
            "axial_tilt_deg": 23.4,
            "eccentricity": 0.03,
            "longitude_periapsis_deg": 101.0,
        })
        scale = annual_stellar_flux_scale(grid, astronomy)
        self.assertEqual(scale.shape, (32, 64))
        self.assertTrue(np.all(np.isfinite(scale)))
        self.assertTrue(np.all(scale >= 0))
        self.assertAlmostEqual(float(np.sum(scale * grid.cell_area_weights)), 1.0, places=12)

    def test_representative_state_clustering_is_deterministic_and_area_closed(self):
        grid = SphereGrid(64, 32, 6371.0)
        temperature = 18.0 - 0.55 * np.abs(grid.lat)
        forcing = 0.35 + 1.2 * np.cos(np.deg2rad(grid.lat))
        first = cluster_representative_states(
            temperature_c=temperature,
            stellar_flux_scale=forcing,
            cell_area_weights=grid.cell_area_weights,
            count=7,
        )
        second = cluster_representative_states(
            temperature_c=temperature,
            stellar_flux_scale=forcing,
            cell_area_weights=grid.cell_area_weights,
            count=7,
        )
        for a, b in zip(first, second, strict=True):
            self.assertTrue(np.array_equal(a, b))
        cluster, rep_t, rep_f, rep_w = first
        self.assertEqual(np.unique(cluster).size, 7)
        self.assertEqual(rep_t.shape, (7,))
        self.assertTrue(np.all(rep_t > 0))
        self.assertTrue(np.all(rep_f >= 0))
        self.assertAlmostEqual(float(np.sum(rep_w)), 1.0, places=14)

    def test_atmogen_setting_invalidates_atmogen_stage_not_tectonics(self):
        baseline = load_config(None)
        baseline.atmogen.enabled = True
        changed = copy.deepcopy(baseline)
        changed.atmogen.eddy_diffusivity_m2_s *= 2.0
        baseline.validate(); changed.validate()
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            first = ResumableWorldPipeline(
                baseline, progress=lambda _: None, checkpoint_dir=tmp_a, resume=False
            )
            second = ResumableWorldPipeline(
                changed, progress=lambda _: None, checkpoint_dir=tmp_b, resume=False
            )
            try:
                self.assertNotEqual(
                    first._stage_config("atmogen_columns_pass_1"),
                    second._stage_config("atmogen_columns_pass_1"),
                )
                self.assertEqual(
                    first._stage_config("tectonics"),
                    second._stage_config("tectonics"),
                )
            finally:
                first.close(); second.close()

    def test_dynamic_atmogen_stage_fingerprint_is_explicit(self):
        files = set(stage_source_files("atmogen_columns_pass_3"))
        self.assertIn("atmogen_adapter.py", files)
        self.assertIn("atmogen_coupler.py", files)
        self.assertIn("climate.py", files)

    def test_new_atmogen_config_validation(self):
        cfg = load_config(None)
        cfg.atmogen.enabled = True
        cfg.atmogen.cloud_mode = "lognormal_sedimentation"
        cfg.atmogen.vertical_transport_mode = "eddy_diffusion"
        cfg.atmogen.activity_model = "auto"
        cfg.validate()
        cfg.atmogen.cloud_particle_geometric_std = 0.99
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_representative_columns_publish_atmogen_cache_identity(self):
        cfg = load_config(None)
        cfg.atmogen.enabled = True
        cfg.atmogen.chemistry_mode = "fixed_species"
        cfg.atmogen.vertical_layers = 8
        cfg.atmogen.representative_column_count = 3
        cfg.validate()
        grid = SphereGrid(64, 32, 6371.0)
        astronomy = SimpleNamespace(
            star={"luminosity_solar": 1.0, "effective_temperature_k": 5772.0},
            planet={
                "semimajor_axis_au": 1.0,
                "radius_earth": 1.0,
                "surface_gravity_m_s2": 9.80665,
                "equilibrium_temperature_k": 255.0,
                "axial_tilt_deg": 23.4,
                "eccentricity": 0.0,
                "longitude_periapsis_deg": 103.0,
            },
            atmosphere={"surface_pressure_bar": 1.0},
            interior={"total_internal_heat_flux_w_m2_approx": 0.0},
            volatile_chemistry={},
        )
        climate = SimpleNamespace(
            annual_temperature_c=15.0 - 0.45 * np.abs(grid.lat)
        )
        result = solve_representative_columns(
            grid=grid,
            astronomy_result=astronomy,
            climate_result=climate,
            world_config=cfg,
        )
        diagnostics = result.diagnostics
        self.assertEqual(diagnostics["column_count"], 3)
        self.assertEqual(len(diagnostics["column_fingerprints"]), 3)
        self.assertEqual(
            diagnostics["unique_column_state_count"],
            len(set(diagnostics["column_fingerprints"])),
        )
        self.assertEqual(len(diagnostics["atmogen_database_sha256"]), 64)
        for index, summary in enumerate(result.summaries):
            self.assertEqual(
                summary["column_state_fingerprint"],
                diagnostics["column_fingerprints"][index],
            )


if __name__ == "__main__":
    unittest.main()
