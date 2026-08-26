import unittest
import numpy as np

from worldgen.config import WorldConfig
from worldgen.pipeline import WorldPipeline
from worldgen.grid import SphereGrid
from worldgen.tectonics import generate_tectonics
from worldgen.rng import RngPool
from worldgen.noise import hybrid_multifractal, configured_blend, TERRAIN_BLEND


def tiny(seed=1234):
    c = WorldConfig(seed=seed)
    c.resolution.width = 128
    c.resolution.height = 64
    c.resolution.history_myr = 200
    c.resolution.history_step_myr = 50
    c.climate.moisture_iterations = 8
    c.tectonics.shape_control_points_per_subplate = 1
    c.weather.hurricane_seed_count = 6
    c.weather.hurricane_max_steps = 20
    c.society.settlement_count = 16
    c.output.save_json = c.output.save_npz = c.output.save_png = c.output.save_report = False
    return c


class WorldgenSmokeTests(unittest.TestCase):
    def test_astronomy_target(self):
        w = WorldPipeline(tiny(), progress=None).generate()
        self.assertAlmostEqual(w["astronomy"].planet["mean_surface_temperature_c_approx"], 15.0, places=5)
        self.assertGreater(w["astronomy"].planet["hill_radius_km"], w["astronomy"].moon["orbit_km"])

    def test_reproducible(self):
        a = WorldPipeline(tiny(99), progress=None).generate()
        b = WorldPipeline(tiny(99), progress=None).generate()
        np.testing.assert_array_equal(a["tectonics"].plate_id, b["tectonics"].plate_id)
        np.testing.assert_allclose(a["climate"].annual_temperature_c, b["climate"].annual_temperature_c)
        self.assertEqual(a["resources"].deposits, b["resources"].deposits)

    def test_seed_changes_world(self):
        a = WorldPipeline(tiny(1), progress=None).generate()
        b = WorldPipeline(tiny(2), progress=None).generate()
        self.assertFalse(np.array_equal(a["tectonics"].plate_id, b["tectonics"].plate_id))

    def test_world_has_expected_layers(self):
        w = WorldPipeline(tiny(), progress=None).generate()
        self.assertGreater(len(w["resources"].deposits), 20)
        self.assertTrue(np.any(w["hydrology"].rivers))
        self.assertGreater(len(np.unique(w["climate"].koppen)), 3)
        self.assertEqual(len(w["society"].settlements), 16)

    def test_subplate_hierarchy_and_surface_evolution(self):
        w = WorldPipeline(tiny(314), progress=None).generate()
        t = w["tectonics"]
        self.assertGreater(t.metadata["subplate_count"], t.metadata["plate_count_final"])
        self.assertTrue(np.any(t.subplate_boundary))
        self.assertTrue(np.any(w["hydrology"].cumulative_erosion_m > 0))
        self.assertTrue(np.any(w["hydrology"].cumulative_deposition_m > 0))
        self.assertLessEqual(w["hydrology"].metadata["lake_area_fraction_of_land"], w["config"].hydrology.lake_area_soft_cap_fraction_land + 0.005)

    def test_submerged_resources_are_not_preindustrial_access(self):
        w = WorldPipeline(tiny(2718), progress=None).generate()
        for d in w["resources"].deposits:
            if d.get("submerged"):
                self.assertFalse(d.get("accessible_preindustrial"))

    def test_coupled_seasonal_circulation_deltas_and_meandering(self):
        w = WorldPipeline(tiny(731928461), progress=None).generate()
        o = w["ocean"]; c = w["climate"]; h = w["hydrology"]
        self.assertEqual(o.current_u_monthly.shape[0], 12)
        self.assertEqual(o.sst_anomaly_c_monthly.shape[0], 12)
        self.assertEqual(c.global_circulation_u.shape[0], 12)
        self.assertGreater(float(np.mean(np.abs(c.wind_u[0] - c.wind_u[6]))), 0.01)
        self.assertGreater(float(np.mean(np.abs(o.current_u_monthly[0] - o.current_u_monthly[6]))), 0.01)
        self.assertGreater(float(h.delta_deposition_m.max()), 0.0)
        self.assertGreater(float(h.tectonic_uplift_m.max()), 0.0)
        self.assertGreater(float(h.meander_potential.max()), 0.0)
        self.assertGreater(float(h.meander_migration_m.max()), 0.0)



    def test_hybrid_noise_is_deterministic_and_multiscale(self):
        c = tiny(8080)
        a = hybrid_multifractal((64, 128), np.random.default_rng(123), octaves=c.noise.octaves, persistence=c.noise.persistence, lacunarity=c.noise.lacunarity, blend=configured_blend(c.noise, TERRAIN_BLEND), domain_warp_strength=c.noise.domain_warp_strength, minimum_sigma_px=c.noise.minimum_sigma_px, wave_count=c.noise.wave_count)
        b = hybrid_multifractal((64, 128), np.random.default_rng(123), octaves=c.noise.octaves, persistence=c.noise.persistence, lacunarity=c.noise.lacunarity, blend=configured_blend(c.noise, TERRAIN_BLEND), domain_warp_strength=c.noise.domain_warp_strength, minimum_sigma_px=c.noise.minimum_sigma_px, wave_count=c.noise.wave_count)
        np.testing.assert_allclose(a, b)
        self.assertGreater(float(np.std(a)), 0.4)
        # A multifractal field must contain both broad and fine variation, not one blurred random band.
        gx = np.diff(a, axis=1)
        self.assertGreater(float(np.std(gx)), 0.01)

    def test_seasonal_memory_hierarchy_and_appearance(self):
        w = WorldPipeline(tiny(424242), progress=None).generate()
        c = w["climate"]; h = w["hydrology"]; a = w["appearance"]
        land = w["terrain"].land
        # With orbital phase + finite thermal memory, opposite calendar shoulders should not be exact mirrors.
        m0 = float(np.mean(c.temperature_c[0][land])); m11 = float(np.mean(c.temperature_c[11][land]))
        self.assertGreater(abs(m0 - m11), 1e-4)
        self.assertGreater(int(h.stream_order.max()), 1)
        self.assertGreater(float(h.river_width_proxy.max()), 0.0)
        self.assertEqual(a.true_color_rgb.shape[-1], 3)
        self.assertEqual(a.true_color_rgb.dtype, np.uint8)
        self.assertGreater(float(np.mean(a.vegetation_fraction[land])), 0.01)


    def test_shape_control_anchors_preserve_subplate_ids(self):
        c = tiny(5150)
        c.resolution.width = 96; c.resolution.height = 48
        c.tectonics.shape_control_points_per_subplate = 2
        c.tectonics.shape_control_spread_deg = 4.5
        g = SphereGrid(c.resolution.width, c.resolution.height)
        t = generate_tectonics(g, c.tectonics, c.resolution, RngPool(c.seed)("shape-control"), c.noise)
        self.assertLessEqual(int(t.subplate_id.max()), t.metadata["subplate_count"] - 1)
        self.assertEqual(t.metadata["shape_control_points_per_subplate"], 2)

    def test_plate_fusion_rule_can_fire(self):
        c = tiny(42)
        c.tectonics.plate_count = 6
        c.tectonics.mean_subplates_per_plate = 4
        c.tectonics.fuse_direction_deg = 180.0
        c.tectonics.fuse_persistence_steps = 1
        c.tectonics.split_stress_threshold = 999.0
        g = SphereGrid(c.resolution.width, c.resolution.height)
        t = generate_tectonics(g, c.tectonics, c.resolution, RngPool(c.seed)("fusion-test"))
        self.assertGreater(t.metadata["fusion_events"], 0)


    def test_noise_warp_periodic_index_safety(self):
        from worldgen.noise import _bilinear_warp
        a = np.arange(8 * 1536, dtype=np.float32).reshape(8, 1536)
        dy = np.zeros_like(a)
        dx = np.full_like(a, 1536.0)
        out = _bilinear_warp(a, dy, dx)
        self.assertEqual(out.shape, a.shape)
        self.assertTrue(np.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
