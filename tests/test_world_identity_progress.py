from __future__ import annotations

import hashlib
import numpy as np

from worldgen.dependency_graph import ProductDependencyGraph
from worldgen.identity import (
    SeedDeriver,
    build_world_identity,
    canonicalize_master_seed,
    component_fingerprint,
)
from worldgen.manifest import build_run_manifest
from worldgen.progress_tree import ProgressTree
from worldgen.rng import RngPool


def _legacy_rng(seed: int, name: str) -> np.random.Generator:
    digest = hashlib.blake2b(f"{seed}:{name}".encode("utf-8"), digest_size=16).digest()
    words = np.frombuffer(digest, dtype=np.uint32).astype(np.uint64)
    ss = np.random.SeedSequence([seed, *map(int, words)])
    return np.random.Generator(np.random.PCG64DXSM(ss))


def test_master_seed_accepts_integer_hex_and_utf8_with_stable_256_bit_identity():
    integer = canonicalize_master_seed(255)
    explicit_hex = canonicalize_master_seed("0xff")
    text_seed = canonicalize_master_seed("Cretak Δ")
    assert integer.canonical_hex == explicit_hex.canonical_hex
    assert text_seed.canonical_hex != integer.canonical_hex
    assert len(text_seed.canonical_hex) == 64


def test_semantic_and_counter_randomness_is_order_independent():
    deriver = SeedDeriver("world-A")
    first = deriver.generator("hydrology", "basin-7", "river-2").integers(0, 2**31, size=8)
    _ = deriver.generator("tectonics", "plate-3").integers(0, 2**31, size=100)
    second = deriver.generator("hydrology", "basin-7", "river-2").integers(0, 2**31, size=8)
    np.testing.assert_array_equal(first, second)

    ordered = [deriver.counter_uint64(i, "terrain", "px/12/8/4") for i in range(32)]
    permuted = {
        i: deriver.counter_uint64(i, "terrain", "px/12/8/4")
        for i in reversed(range(32))
    }
    assert ordered == [permuted[i] for i in range(32)]


def test_legacy_integer_rng_stream_remains_bit_for_bit_compatible():
    expected = _legacy_rng(20260826, "tectonics").integers(
        0, 2**63, size=16, dtype=np.int64
    )
    actual = RngPool(20260826)("tectonics").integers(
        0, 2**63, size=16, dtype=np.int64
    )
    np.testing.assert_array_equal(actual, expected)


def test_world_fingerprint_separates_physical_identity_from_output_settings():
    base = {"seed": 7, "tectonics": {"plate_count": 12}, "output": {"png": True}}
    output_changed = {
        "seed": 7,
        "tectonics": {"plate_count": 12},
        "output": {"png": False},
        "logging": {"level": "TRACE"},
    }
    physical_changed = {
        "seed": 7,
        "tectonics": {"plate_count": 13},
        "output": {"png": True},
    }
    a = build_world_identity(
        7, configuration=base, algorithm_versions={"tectonics": "v1"}
    )
    b = build_world_identity(
        7, configuration=output_changed, algorithm_versions={"tectonics": "v1"}
    )
    c = build_world_identity(
        7, configuration=physical_changed, algorithm_versions={"tectonics": "v1"}
    )
    assert a.fingerprint == b.fingerprint
    assert a.fingerprint != c.fingerprint
    assert component_fingerprint(
        a.fingerprint,
        "terrain",
        algorithm_version=3,
        address=["px", 12, 8, 4],
        lod=12,
    ) != component_fingerprint(
        a.fingerprint,
        "terrain",
        algorithm_version=4,
        address=["px", 12, 8, 4],
        lod=12,
    )


def test_dependency_invalidation_is_local_and_operational_settings_are_nonphysical():
    graph = ProductDependencyGraph()
    erosion = graph.plan(["erosion.strength"])
    assert "surface_coupling" in erosion.root_products
    assert "tectonics" not in erosion.stale_products

    plates = graph.plan(["tectonics.plate_count"])
    assert "tectonics" in plates.stale_products
    assert "terrain_refinement" in plates.stale_products

    operational = graph.plan(
        ["runtime.worker_count", "logging.level", "storage.ram_budget_mb"]
    )
    assert not operational.stale_products


def test_run_manifest_adds_canonical_world_identity_and_preserves_old_fields():
    config = {
        "seed": 12345,
        "resolution": {"width": 128, "height": 64},
        "tectonics": {"plate_count": 12},
        "atmogen": {"enabled": False},
        "output": {"png": True},
    }
    a = build_run_manifest(
        config=config, algorithm_versions={"tectonics": "v1"}
    )
    changed = {**config, "output": {"png": False}}
    b = build_run_manifest(
        config=changed, algorithm_versions={"tectonics": "v1"}
    )
    assert a["seed"] == 12345
    assert a["resolution"] == [128, 64]
    assert len(a["canonical_master_seed"]) == 64
    assert len(a["world_fingerprint"]) == 64
    assert a["world_fingerprint"] == a["reproducibility"]["world_fingerprint"]
    assert a["world_fingerprint"] == b["world_fingerprint"]
    assert a["config_sha256"] != b["config_sha256"]


class _Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def advance(self, seconds: float):
        self.t += seconds


def test_nested_progress_uses_workload_weights_and_emits_structured_events():
    clock = _Clock()
    events = []
    tree = ProgressTree(
        event_sink=events.append,
        progress_event_interval_s=0,
        clock=clock,
    )
    tree.create("planet", "Generate Planet")
    tree.create("init", "Initialization", parent="planet", total=1, weight=1)
    tree.create("erosion", "Erosion", parent="planet", total=100, weight=9)
    tree.start("init")
    tree.update("init", completed=1)
    tree.complete("init")
    tree.start("erosion")
    tree.update("erosion", completed=0)

    assert abs(tree.snapshot("planet").fraction - 0.1) < 1e-12
    assert events[0].type == "TaskCreated"
    assert any(event.type == "TaskProgress" for event in events)


def test_progress_pause_metrics_eta_and_cancellation_propagate():
    clock = _Clock()
    tree = ProgressTree(progress_event_interval_s=0, clock=clock)
    parent_token = tree.create("parent", "Parent")
    child_token = tree.create(
        "child", "Child", parent="parent", total=100, units="cells"
    )
    tree.start("child")
    clock.advance(2)
    tree.update("child", completed=20)
    clock.advance(2)
    tree.update("child", completed=40)
    tree.metric("child", "ram_bytes", 1024)

    snap = tree.snapshot("child")
    assert snap.throughput_per_second and snap.throughput_per_second > 0
    assert snap.eta_seconds and snap.eta_seconds > 0
    assert snap.resource_metrics["ram_bytes"] == 1024

    tree.pause("child")
    clock.advance(100)
    tree.resume("child")
    clock.advance(1)
    assert tree.snapshot("child").elapsed_seconds < 10

    tree.cancel("parent")
    assert parent_token.cancelled and child_token.cancelled
    assert tree.snapshot("child").state == "cancelled"
