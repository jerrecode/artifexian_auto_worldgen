from __future__ import annotations

import argparse
import cProfile
from dataclasses import replace
import json
from pathlib import Path
import pstats
import sys
from typing import Any

import yaml

from .config import config_schema, load_config
from .logging_utils import configure_logging
from .progress import StageProgress, expected_pipeline_stages
from .runtime import (
    ManagedExecutor,
    configure_numeric_threads,
    optional_backend_status,
    resolve_runtime_plan,
)
from .workflow import StageCheckpointStore


def _resolution(value: str) -> tuple[int, int]:
    text = value.lower().replace("×", "x")
    try:
        width_s, height_s = text.split("x", 1)
        width, height = int(width_s), int(height_s)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("resolution must be WIDTHxHEIGHT, e.g. 1536x768") from exc
    if width < 64 or height < 32 or width != 2 * height:
        raise argparse.ArgumentTypeError("resolution must be a 2:1 equirectangular grid and at least 64x32")
    return width, height


def _set_config_value(cfg: Any, expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"--set expects SECTION.KEY=VALUE, got {expression!r}")
    path, raw_value = expression.split("=", 1)
    parts = [p for p in path.strip().split(".") if p]
    if not parts:
        raise ValueError(f"Invalid config path: {path!r}")
    target = cfg
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise KeyError(f"Unknown configuration path: {path!r}")
        target = getattr(target, part)
    leaf = parts[-1]
    if not hasattr(target, leaf):
        raise KeyError(f"Unknown configuration path: {path!r}")
    value = yaml.safe_load(raw_value)
    setattr(target, leaf, value)


def _apply_quick(cfg) -> None:
    cfg.resolution.width = 256
    cfg.resolution.height = 128
    cfg.resolution.history_step_myr = max(cfg.resolution.history_step_myr, 50)
    cfg.tectonics.history_grid_height = min(cfg.tectonics.history_grid_height, 56)
    cfg.tectonics.boundary_detail_octaves = min(cfg.tectonics.boundary_detail_octaves, 4)
    cfg.tectonics.boundary_deformation_iterations = min(cfg.tectonics.boundary_deformation_iterations, 1)
    cfg.tectonics.strain_boundary_warp_deg = min(cfg.tectonics.strain_boundary_warp_deg, 1.8)
    cfg.noise.octaves = min(cfg.noise.octaves, 4)
    cfg.noise.domain_warp_strength = min(cfg.noise.domain_warp_strength, 0.16)
    cfg.noise.wave_count = min(cfg.noise.wave_count, 3)
    cfg.climate.moisture_iterations = min(cfg.climate.moisture_iterations, 12)
    cfg.climate.thermal_memory_spinup_years = min(cfg.climate.thermal_memory_spinup_years, 3)
    cfg.hydrology.surface_evolution_iterations = min(cfg.hydrology.surface_evolution_iterations, 3)
    cfg.hydrology.flow_refresh_interval = max(cfg.hydrology.flow_refresh_interval, 2)
    cfg.hydrology.sediment_routing_passes = min(cfg.hydrology.sediment_routing_passes, 8)
    cfg.hydrology.max_river_centerlines = min(cfg.hydrology.max_river_centerlines, 60)
    cfg.weather.hurricane_seed_count = min(cfg.weather.hurricane_seed_count, 20)
    cfg.weather.hurricane_max_steps = min(cfg.weather.hurricane_max_steps, 70)
    cfg.society.settlement_count = min(cfg.society.settlement_count, 60)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="worldgen",
        description="Automatic procedural world generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("generate", nargs="?", default="generate", help=argparse.SUPPRESS)
    p.add_argument("--version", action="version", version="%(prog)s 0.4.0")

    io = p.add_argument_group("configuration and output")
    io.add_argument("--config", type=Path, default=None, help="YAML configuration file")
    io.add_argument("--out", type=Path, default=Path("world-out"), help="Output directory")
    io.add_argument("--seed", type=int, default=None, help="Override root seed")
    io.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="Override any dataclass configuration value; may be repeated")
    io.add_argument("--write-config", type=Path, default=None, help="Write the resolved configuration as YAML")
    io.add_argument("--dry-run", action="store_true", help="Validate configuration/runtime plan without generating")
    io.add_argument("--quick", action="store_true", help="Fast-preview preset with reduced numerical workload")
    io.add_argument("--no-society", action="store_true", help="Stop after physical/geological generation")
    io.add_argument("--no-png", action="store_true", help="Skip PNG map rendering")
    io.add_argument("--no-npz", action="store_true", help="Skip NPZ array export")
    io.add_argument("--no-json", action="store_true", help="Skip JSON/GeoJSON export")
    io.add_argument("--no-report", action="store_true", help="Skip Markdown report export")
    io.add_argument("--compress-npz", action=argparse.BooleanOptionalAction, default=None,
                    help="Enable/disable DEFLATE compression for world_arrays.npz")
    io.add_argument("--checkpoint-dir", type=Path, default=None,
                    help="Content-addressed stage checkpoint directory")
    io.add_argument("--resume", action="store_true",
                    help="Restore matching completed stages from checkpoints; defaults checkpoint dir inside --out")
    io.add_argument("--clear-checkpoints", action="store_true",
                    help="Clear the selected checkpoint store before generation")

    res = p.add_argument_group("resolution and rendering")
    res.add_argument("--resolution", type=_resolution, metavar="WIDTHxHEIGHT", help="Simulation grid resolution")
    res.add_argument("--width", type=int, default=None, help="Simulation width; height is inferred when omitted")
    res.add_argument("--height", type=int, default=None, help="Simulation height; width is inferred when omitted")
    res.add_argument("--resolution-scale", type=float, default=1.0,
                     help="Scale configured simulation resolution while preserving 2:1 aspect")
    res.add_argument("--png-scale", type=float, default=1.0,
                     help="Post-render PNG upscale factor; >1 increases final image resolution")
    res.add_argument("--png-resample", choices=("nearest", "bilinear", "bicubic", "lanczos"), default="lanczos")
    res.add_argument("--max-png-megapixels", type=float, default=120.0,
                     help="Safety cap applied to each post-upscaled PNG")

    perf = p.add_argument_group("runtime and performance")
    perf.add_argument("--workers", type=int, default=0, help="Requested worker count; 0 selects automatically")
    perf.add_argument("--worker-cap", type=int, default=8, help="Hard upper bound on managed workers")
    perf.add_argument("--parallel-backend", choices=("auto", "thread", "process", "serial"), default="auto")
    perf.add_argument("--threads-per-worker", type=int, default=1,
                      help="BLAS/OpenMP/NumExpr threads allocated per worker")
    perf.add_argument("--memory-per-worker-mb", type=int, default=384,
                      help="Memory budget used when automatically capping workers")
    perf.add_argument("--runtime-info", action="store_true", help="Print resolved runtime plan and optional backends")
    perf.add_argument("--array-store", type=Path, default=None,
                      help="Also export arrays into a random-access mmap/SQLite store")
    perf.add_argument("--array-store-max-gb", type=float, default=64.0, help="Disk cap for --array-store")
    perf.add_argument("--profile", type=Path, default=None, help="Write cProfile statistics to this file")
    perf.add_argument("--timings-json", type=Path, default=None, help="Write per-stage timings/runtime metadata as JSON")

    obs = p.add_argument_group("progress and logging")
    obs.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True,
                     help="Show stage progress, elapsed time and ETA")
    obs.add_argument("-v", "--verbose", action="count", default=0, help="Increase logging verbosity; repeat for debug")
    obs.add_argument("--quiet", action="store_true", help="Only emit errors")
    obs.add_argument("--log-file", type=Path, default=None, help="Write detailed logs to a file")
    obs.add_argument("--log-json", action="store_true", help="Use JSON-lines formatting for --log-file")
    return p


def _config_command(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="worldgen config", description="Inspect and validate worldgen configuration")
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("schema", help="Print the machine-readable configuration schema")
    val = sub.add_parser("validate", help="Validate a YAML configuration")
    val.add_argument("path", type=Path)
    exp = sub.add_parser("explain", help="Explain one configuration field")
    exp.add_argument("key")
    args = p.parse_args(argv)
    if args.action == "schema":
        print(json.dumps(config_schema(), indent=2, default=str))
        return 0
    if args.action == "validate":
        cfg = load_config(args.path)
        print(json.dumps({"valid": True, "seed": cfg.seed, "resolution": [cfg.resolution.width, cfg.resolution.height]}, indent=2))
        return 0
    schema = config_schema()
    if args.key not in schema:
        p.error(f"unknown configuration field: {args.key}")
    print(json.dumps({args.key: schema[args.key]}, indent=2, default=str))
    return 0


def _cache_command(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="worldgen cache", description="Inspect or clear stage checkpoints")
    p.add_argument("action", choices=("status", "clear"))
    p.add_argument("path", type=Path)
    args = p.parse_args(argv)
    store = StageCheckpointStore(args.path)
    if args.action == "clear":
        store.clear()
        print(f"Cleared checkpoint store: {store.root}")
        return 0
    entries = store._index.get("stages", {})
    total = sum(int(v.get("size_bytes", 0)) for v in entries.values())
    print(json.dumps({"path": str(store.root), "objects": len(entries), "bytes": total}, indent=2))
    return 0


def _inspect_command(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="worldgen inspect", description="Inspect a generated world's run manifest")
    p.add_argument("world", type=Path)
    args = p.parse_args(argv)
    path = args.world if args.world.name == "run_manifest.json" else args.world / "run_manifest.json"
    if not path.is_file():
        p.error(f"run manifest not found: {path}")
    print(path.read_text(encoding="utf-8"))
    return 0


def _dispatch_special(argv: list[str]) -> int | None:
    if not argv:
        return None
    if argv[0] == "config":
        return _config_command(argv[1:])
    if argv[0] == "cache":
        return _cache_command(argv[1:])
    if argv[0] == "inspect":
        return _inspect_command(argv[1:])
    return None


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    special = _dispatch_special(argv)
    if special is not None:
        return special

    p = _parser()
    args = p.parse_args(argv)

    if args.generate not in (None, "generate"):
        p.error(f"unknown command {args.generate!r}; use generate, config, cache, or inspect")
    if args.resolution_scale <= 0:
        p.error("--resolution-scale must be > 0")
    if args.png_scale <= 0:
        p.error("--png-scale must be > 0")
    if args.max_png_megapixels <= 0:
        p.error("--max-png-megapixels must be > 0")
    if args.array_store_max_gb <= 0:
        p.error("--array-store-max-gb must be > 0")

    try:
        cfg = load_config(args.config)
        if args.quick:
            _apply_quick(cfg)
        for expression in args.overrides:
            _set_config_value(cfg, expression)
        if args.seed is not None:
            cfg.seed = args.seed
        if args.resolution is not None:
            cfg.resolution.width, cfg.resolution.height = args.resolution
        if args.resolution_scale != 1.0:
            height = max(32, int(round(cfg.resolution.height * args.resolution_scale)))
            cfg.resolution.height = height
            cfg.resolution.width = 2 * height
        if args.width is not None or args.height is not None:
            if args.width is not None and args.height is not None:
                cfg.resolution.width, cfg.resolution.height = args.width, args.height
            elif args.width is not None:
                if args.width % 2:
                    p.error("--width must be even when --height is inferred")
                cfg.resolution.width, cfg.resolution.height = args.width, args.width // 2
            else:
                cfg.resolution.height, cfg.resolution.width = args.height, 2 * args.height

        if args.no_society:
            cfg.society.enabled = False
        if args.no_png:
            cfg.output.save_png = False
        if args.no_npz:
            cfg.output.save_npz = False
        if args.no_json:
            cfg.output.save_json = False
        if args.no_report:
            cfg.output.save_report = False
        if args.compress_npz is not None:
            cfg.output.compress_npz = bool(args.compress_npz)
        cfg.validate()
    except (ValueError, KeyError, TypeError) as exc:
        p.error(str(exc))

    plan = resolve_runtime_plan(
        workers=args.workers,
        worker_cap=args.worker_cap,
        backend=args.parallel_backend,
        memory_per_worker_mb=args.memory_per_worker_mb,
        threads_per_worker=args.threads_per_worker,
    )
    configure_numeric_threads(plan.threads_per_worker)

    logger = configure_logging(
        verbose=args.verbose,
        quiet=args.quiet,
        log_file=args.log_file,
        json_file=args.log_json,
    )
    logger.info("seed=%s resolution=%sx%s", cfg.seed, cfg.resolution.width, cfg.resolution.height)
    logger.info("runtime backend=%s workers=%s/%s threads_per_worker=%s",
                plan.backend, plan.workers, plan.cpu_count, plan.threads_per_worker)

    runtime_payload = {
        "backend": plan.backend,
        "workers": plan.workers,
        "cpu_count": plan.cpu_count,
        "memory_limit_bytes": plan.memory_limit_bytes,
        "threads_per_worker": plan.threads_per_worker,
    }
    if args.runtime_info:
        print(json.dumps({"runtime_plan": runtime_payload, "optional_backends": optional_backend_status()}, indent=2))

    if args.write_config is not None:
        args.write_config.parent.mkdir(parents=True, exist_ok=True)
        args.write_config.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
        logger.info("resolved configuration written to %s", args.write_config)

    if args.dry_run:
        return 0

    from .execution import WorldPipeline

    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None and args.resume:
        checkpoint_dir = args.out / ".worldgen-checkpoints"
    if args.clear_checkpoints:
        target = checkpoint_dir or (args.out / ".worldgen-checkpoints")
        StageCheckpointStore(target).clear()
        checkpoint_dir = target

    total_stages = expected_pipeline_stages(cfg, include_output=True)
    progress = StageProgress(
        total_stages,
        enabled=bool(args.progress and not args.quiet),
        log=logger.info if args.verbose or args.log_file else None,
    )
    pipeline = WorldPipeline(
        cfg,
        progress=progress,
        checkpoint_dir=checkpoint_dir,
        resume=args.resume,
        runtime_plan=runtime_payload,
    )

    if args.profile is None:
        world = pipeline.generate(args.out)
    else:
        args.profile.parent.mkdir(parents=True, exist_ok=True)
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            world = pipeline.generate(args.out)
        finally:
            profiler.disable()
            profiler.dump_stats(args.profile)
            summary = args.profile.with_suffix(args.profile.suffix + ".txt")
            with summary.open("w", encoding="utf-8") as f:
                pstats.Stats(profiler, stream=f).sort_stats("cumtime").print_stats(80)
            logger.info("profile written to %s and %s", args.profile, summary)
    progress.finish()

    if cfg.output.save_png and args.png_scale != 1.0:
        from .imageops import upscale_png_tree
        image_plan = replace(plan, backend="thread" if plan.workers > 1 else "serial")
        with ManagedExecutor(image_plan) as executor:
            images = upscale_png_tree(
                args.out / "maps",
                scale=args.png_scale,
                resample=args.png_resample,
                max_megapixels=args.max_png_megapixels,
                executor=executor,
            )
        logger.info("post-upscaled %d PNG maps by factor %.3f", len(images), args.png_scale)

    if args.array_store is not None:
        from .storage import store_array_mapping
        arrays = pipeline._array_export(world)
        store = store_array_mapping(arrays, args.array_store, max_bytes=int(args.array_store_max_gb * 1024**3))
        logger.info("random-access array store: %s (%d arrays, %.3f GiB on disk)",
                    store.root, len(store.keys()), store.disk_usage_bytes() / 1024**3)
        store.close()

    if args.timings_json is not None:
        args.timings_json.parent.mkdir(parents=True, exist_ok=True)
        timing_payload = {
            "seed": cfg.seed,
            "resolution": [cfg.resolution.width, cfg.resolution.height],
            "runtime": runtime_payload,
            "stage_seconds": pipeline.timings,
            "total_stage_seconds": sum(pipeline.timings.values()),
            "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
        }
        args.timings_json.write_text(json.dumps(timing_payload, indent=2), encoding="utf-8")

    print(f"World written to: {args.out.resolve()}")
    return 0
