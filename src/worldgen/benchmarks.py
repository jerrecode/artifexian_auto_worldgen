from __future__ import annotations

"""Reproducible micro/macro benchmarks for worldgen numerical kernels.

The benchmark runner intentionally records measurements instead of enforcing
fragile wall-clock thresholds. CI and developers can compare the emitted JSON
between commits while correctness tests remain deterministic and timing-agnostic.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import json
import os
import platform
import resource
import sys
import time
from typing import Any

import numpy as np

from .config import WorldConfig
from .drainage import DrainageGraph
from .grid import SphereGrid
from .pipeline import WorldPipeline
from .priority_flood import (
    numba_priority_flood_available,
    priority_flood,
    priority_flood_reference,
)
from .runtime import optional_backend_status


@dataclass(slots=True, frozen=True)
class BenchmarkProfile:
    name: str
    width: int
    height: int
    history_step_myr: int
    climate_iterations: int
    surface_iterations: int
    earth_system_passes: int


PROFILES: dict[str, BenchmarkProfile] = {
    "micro": BenchmarkProfile("micro", 128, 64, 100, 6, 1, 1),
    "quick": BenchmarkProfile("quick", 256, 128, 50, 12, 2, 2),
    "normal": BenchmarkProfile("normal", 768, 384, 25, 30, 4, 3),
    "high": BenchmarkProfile("high", 1536, 768, 10, 48, 6, 4),
}


def _peak_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB, macOS reports bytes.
    return raw if sys.platform == "darwin" else raw * 1024


def _current_rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return None


def _timed(fn):
    wall0 = time.perf_counter()
    cpu0 = time.process_time()
    value = fn()
    return value, {
        "wall_seconds": float(time.perf_counter() - wall0),
        "cpu_seconds": float(time.process_time() - cpu0),
        "peak_rss_bytes": int(_peak_rss_bytes()),
        "rss_bytes": _current_rss_bytes(),
    }


def benchmark_priority_flood(*, width: int = 192, height: int = 96, seed: int = 90210) -> dict[str, Any]:
    """Compare the reference and selected Priority-Flood backends on identical terrain."""
    grid = SphereGrid(width, height, distance_cache_max_bytes=8 * 1024**2)
    rng = np.random.default_rng(seed)
    lat = np.deg2rad(grid.lat)
    lon = np.deg2rad(grid.lon)
    elev = (
        0.55 * np.cos(2.0 * lat)
        + 0.30 * np.sin(3.0 * lon) * np.cos(lat)
        + 0.10 * rng.standard_normal((height, width))
    ).astype(np.float64)
    ocean = elev < np.quantile(elev, 0.36)

    reference, ref_stats = _timed(lambda: priority_flood_reference(elev, ocean, grid))

    # Warm the JIT before timing the accelerated path so compile latency is reported
    # separately from steady-state kernel throughput.
    warmup_seconds = 0.0
    if numba_priority_flood_available():
        t0 = time.perf_counter()
        priority_flood(elev[:16, :32], ocean[:16, :32], SphereGrid(32, 16), backend="numba")
        warmup_seconds = time.perf_counter() - t0
    accelerated, fast_stats = _timed(lambda: priority_flood(elev, ocean, grid, backend="auto"))
    max_abs = float(np.max(np.abs(reference - accelerated)))
    ref_wall = max(float(ref_stats["wall_seconds"]), 1e-12)
    fast_wall = max(float(fast_stats["wall_seconds"]), 1e-12)
    return {
        "shape": [height, width],
        "numba_available": bool(numba_priority_flood_available()),
        "jit_warmup_seconds": float(warmup_seconds),
        "reference": ref_stats,
        "selected_backend": fast_stats,
        "speedup": float(ref_wall / fast_wall),
        "max_abs_difference_km": max_abs,
    }


def benchmark_drainage_graph(*, width: int = 512, height: int = 256) -> dict[str, Any]:
    """Measure graph construction and repeated O(N) accumulation."""
    n = width * height
    # A deterministic acyclic raster-like graph: cells generally drain east, then
    # each row outlet drains to the next row. The final cell is the sink.
    receiver = np.arange(n, dtype=np.int64) + 1
    receiver[-1] = -1
    graph, build_stats = _timed(lambda: DrainageGraph.from_receiver(receiver, (height, width)))
    source = np.ones((height, width), dtype=np.float64)
    accumulated, accumulate_stats = _timed(lambda: graph.accumulate(source))
    return {
        "shape": [height, width],
        "nodes": int(n),
        "build": build_stats,
        "accumulate": accumulate_stats,
        "outlet_accumulation": float(accumulated.ravel()[-1]),
    }


def config_for_profile(profile: str, *, seed: int = 20260826) -> WorldConfig:
    try:
        p = PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"unknown benchmark profile {profile!r}; choose {sorted(PROFILES)}") from exc
    c = WorldConfig(seed=seed)
    c.resolution.width = p.width
    c.resolution.height = p.height
    c.resolution.history_step_myr = p.history_step_myr
    c.climate.moisture_iterations = p.climate_iterations
    c.climate.thermal_memory_spinup_years = min(c.climate.thermal_memory_spinup_years, 3 if profile in {"micro", "quick"} else 5)
    c.hydrology.surface_evolution_iterations = p.surface_iterations
    c.hydrology.sediment_routing_passes = min(c.hydrology.sediment_routing_passes, 6 if profile == "micro" else 10 if profile == "quick" else c.hydrology.sediment_routing_passes)
    c.hydrology.max_river_centerlines = min(c.hydrology.max_river_centerlines, 30 if profile == "micro" else 80 if profile == "quick" else c.hydrology.max_river_centerlines)
    c.simulation.earth_system_passes = p.earth_system_passes
    c.simulation.final_climate_ocean_passes = 1 if profile in {"micro", "quick"} else c.simulation.final_climate_ocean_passes
    if profile in {"micro", "quick"}:
        c.noise.octaves = min(c.noise.octaves, 4)
        c.noise.wave_count = min(c.noise.wave_count, 3)
        c.tectonics.history_grid_height = min(c.tectonics.history_grid_height, 56)
        c.weather.hurricane_seed_count = min(c.weather.hurricane_seed_count, 12)
        c.society.settlement_count = min(c.society.settlement_count, 40)
    c.output.save_png = False
    c.output.save_npz = False
    c.output.save_json = False
    c.output.save_report = False
    return c.validate()


def benchmark_world(profile: str = "micro", *, seed: int = 20260826) -> dict[str, Any]:
    cfg = config_for_profile(profile, seed=seed)
    pipeline = WorldPipeline(cfg, progress=None)
    world, stats = _timed(lambda: pipeline.generate(None))
    cache = world["grid"].spatial_cache_stats()
    return {
        "profile": profile,
        "seed": int(seed),
        "resolution": [int(cfg.resolution.width), int(cfg.resolution.height)],
        "total": stats,
        "stage_seconds": {k: float(v) for k, v in pipeline.timings.items()},
        "distance_cache": asdict(cache),
        "coupling_passes": len(world.get("coupling_history", [])),
    }


def benchmark_suite(
    profile: str = "micro",
    *,
    seed: int = 20260826,
    run_world: bool = True,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 1,
        "profile": profile,
        "seed": int(seed),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "numpy": np.__version__,
            "optional_backends": optional_backend_status(),
        },
        "priority_flood": benchmark_priority_flood(
            width=128 if profile == "micro" else 192,
            height=64 if profile == "micro" else 96,
            seed=seed,
        ),
        "drainage_graph": benchmark_drainage_graph(
            width=256 if profile == "micro" else 512,
            height=128 if profile == "micro" else 256,
        ),
    }
    if run_world:
        result["world"] = benchmark_world(profile, seed=seed)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark worldgen kernels and complete profiles")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="micro")
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--skip-world", action="store_true", help="Only benchmark numerical kernels")
    parser.add_argument("--output", type=Path, default=Path("benchmark.json"))
    args = parser.parse_args(argv)
    result = benchmark_suite(args.profile, seed=args.seed, run_world=not args.skip_world)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BenchmarkProfile",
    "PROFILES",
    "benchmark_priority_flood",
    "benchmark_drainage_graph",
    "benchmark_world",
    "benchmark_suite",
    "config_for_profile",
]
