from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from worldgen.ultrares import (
    UltraResolutionSpec,
    compact_world_authority,
    finalize_ultra_resolution,
    run_ultra_resolution,
    run_ultra_resolution_shard,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Build the scale-aware Earth-like ultra-resolution terrain hierarchy "
            "from an already completed full-detail global world."
        )
    )
    p.add_argument("world", type=Path)
    p.add_argument(
        "--mode",
        choices=("all", "compact", "shard", "finalize"),
        default="all",
        help=(
            "all runs the historical end-to-end path; compact only prepares the "
            "source authority; shard generates one deterministic deepest-tile shard; "
            "finalize validates gathered shards then audits/reconstructs."
        ),
    )
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--tile-size", type=int, default=1024)
    p.add_argument("--base-multiplier", type=float, default=4.0)
    p.add_argument("--subsection-multiplier", type=float, default=3.0)
    p.add_argument("--min-samples-per-wavelength", type=float, default=4.0)
    p.add_argument("--terrain-detail-strength", type=float, default=1.15)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Regenerate selected tiles even when validated restart authority exists.",
    )
    p.add_argument("--no-compact", action="store_true")
    p.add_argument("--keep-source-maps", action="store_true")
    return p


def _spec(args: argparse.Namespace) -> UltraResolutionSpec:
    return UltraResolutionSpec(
        base_linear_multiplier=args.base_multiplier,
        subsection_linear_multiplier=args.subsection_multiplier,
        tile_size=args.tile_size,
        min_samples_per_wavelength=args.min_samples_per_wavelength,
        terrain_detail_strength=args.terrain_detail_strength,
        workers=args.workers,
    )


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = args.world.expanduser().resolve()
    compaction = None

    if args.mode == "compact":
        if args.no_compact:
            raise SystemExit("--mode compact cannot be combined with --no-compact")
        compaction = compact_world_authority(
            root,
            prune_rendered_maps=not args.keep_source_maps,
        )
        print(json.dumps({"compaction": compaction}, indent=2, sort_keys=True))
        return 0

    if args.mode in {"all", "shard"} and not args.no_compact:
        compaction = compact_world_authority(
            root,
            prune_rendered_maps=not args.keep_source_maps,
        )

    spec = _spec(args)
    if args.mode == "shard":
        report = run_ultra_resolution_shard(
            root,
            spec=spec,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            resume=not args.no_resume,
        )
        payload = {"compaction": compaction, "report": report}
    elif args.mode == "finalize":
        report = finalize_ultra_resolution(root, spec=spec)
        payload = {"compaction": None, "report": asdict(report)}
    else:
        report = run_ultra_resolution(root, spec=spec)
        payload = {"compaction": compaction, "report": asdict(report)}

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
