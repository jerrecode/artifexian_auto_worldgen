from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from worldgen.ultrares import (
    UltraResolutionSpec,
    compact_world_authority,
    run_ultra_resolution,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Build the scale-aware Earth-like ultra-resolution terrain hierarchy "
            "from an already completed full-detail global world."
        )
    )
    p.add_argument("world", type=Path)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--tile-size", type=int, default=1024)
    p.add_argument("--base-multiplier", type=float, default=4.0)
    p.add_argument("--subsection-multiplier", type=float, default=2.0)
    p.add_argument("--min-samples-per-wavelength", type=float, default=4.0)
    p.add_argument("--no-compact", action="store_true")
    p.add_argument("--keep-source-maps", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = args.world.expanduser().resolve()
    compaction = None
    if not args.no_compact:
        compaction = compact_world_authority(
            root,
            prune_rendered_maps=not args.keep_source_maps,
        )
    spec = UltraResolutionSpec(
        base_linear_multiplier=args.base_multiplier,
        subsection_linear_multiplier=args.subsection_multiplier,
        tile_size=args.tile_size,
        min_samples_per_wavelength=args.min_samples_per_wavelength,
        workers=args.workers,
    )
    report = run_ultra_resolution(root, spec=spec)
    payload = {
        "compaction": compaction,
        "report": asdict(report),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
