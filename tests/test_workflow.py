import tempfile
import unittest

from worldgen.workflow import StageCheckpointStore, StageRegistry, StageSpec, canonical_hash, stage_key


class WorkflowInfrastructureTests(unittest.TestCase):
    def test_registry_topological_order(self):
        r = StageRegistry()
        r.register(StageSpec("a"))
        r.register(StageSpec("b", dependencies=("a",)))
        r.register(StageSpec("c", dependencies=("b",)))
        self.assertEqual(r.topological_order(), ("a", "b", "c"))

    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = StageCheckpointStore(tmp)
            path, size = store.save("demo", "abc", {"x": [1, 2, 3]})
            self.assertTrue(path.is_file())
            self.assertGreater(size, 0)
            self.assertTrue(store.has("abc"))
            self.assertEqual(store.load("abc"), {"x": [1, 2, 3]})

    def test_stage_key_ignores_unrelated_downstream_configuration(self):
        class Cfg:
            seed = 7
            def to_dict(self):
                return {
                    "astronomy": {"star_mass_solar": 1.0},
                    "resources": {"deposit_density": 1.0},
                }
        cfg = Cfg()
        a = stage_key(stage_name="astronomy", seed=7, config=cfg)
        cfg.to_dict = lambda: {
            "astronomy": {"star_mass_solar": 1.0},
            "resources": {"deposit_density": 99.0},
        }
        b = stage_key(stage_name="astronomy", seed=7, config=cfg)
        self.assertEqual(a, b)

    def test_canonical_hash_is_order_independent_for_mappings(self):
        self.assertEqual(canonical_hash({"a": 1, "b": 2}), canonical_hash({"b": 2, "a": 1}))


if __name__ == "__main__":
    unittest.main()
