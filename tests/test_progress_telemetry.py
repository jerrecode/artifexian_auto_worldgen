from __future__ import annotations

from worldgen.progress_telemetry import HierarchicalProgressTracker


def test_hierarchical_progress_persists_samples_and_exposes_nested_etas(tmp_path):
    tracker = HierarchicalProgressTracker(
        tmp_path,
        scope="test-shard",
        heartbeat_seconds=3600.0,
        parallelism=2,
        subsubsteps_per_substep=3,
        processing_steps_total=3,
    )
    tracker.observe("processing_step", "resume_scan", 2.0)
    tracker.set_processing_step("terrain_generation", completed_before=1)
    tracker.configure_substeps(completed=0, total=4, resumed=0)
    tracker.begin(
        "processing_step",
        "terrain_generation",
        token="processing",
        index=2,
        total=3,
    )

    tracker.begin("substep", "tile-0", token="tile-0", total=4)
    tracker.observe(
        "subsubstep",
        "initial_hydrology",
        6.0,
        parent="tile-0",
    )
    tracker.observe(
        "subsubstep",
        "physical_evolution",
        4.0,
        parent="tile-0",
    )
    tracker.end("tile-0", duration_seconds=15.0)

    tracker.begin("substep", "tile-1", token="tile-1", total=4)
    tracker.begin(
        "subsubstep",
        "initial_hydrology",
        token="tile-1:hydrology",
        parent="tile-1",
    )
    snap = tracker.snapshot()

    assert snap["means_seconds"]["all_subsubsteps"] == 5.0
    assert snap["means_seconds"]["all_substeps"] == 15.0
    assert snap["active_subsubstep"]["name"] == "initial_hydrology"
    assert snap["active_subsubstep"]["mean_same_phase_seconds"] == 6.0
    assert snap["active_substep"]["name"] == "tile-1"
    assert snap["eta_seconds"]["current_processing_step"] is not None
    assert snap["eta_seconds"]["whole_job"] is not None

    tracker.end("tile-1:hydrology", duration_seconds=7.0)
    tracker.end("tile-1", duration_seconds=16.0)
    tracker.end("processing", duration_seconds=40.0)
    tracker.observe("processing_step", "shard_finalize", 1.0)
    tracker.close()

    assert (tmp_path / "test-shard.events.jsonl").exists()
    assert (tmp_path / "test-shard.summary.json").exists()

    resumed = HierarchicalProgressTracker(
        tmp_path,
        scope="test-shard",
        heartbeat_seconds=3600.0,
        parallelism=2,
        subsubsteps_per_substep=3,
        processing_steps_total=3,
    )
    snap2 = resumed.snapshot()
    assert snap2["means_seconds"]["all_subsubsteps"] == (6.0 + 4.0 + 7.0) / 3.0
    assert snap2["means_seconds"]["all_substeps"] == 15.5
    assert snap2["means_seconds"]["all_processing_steps"] == (2.0 + 40.0 + 1.0) / 3.0
    resumed.close()
