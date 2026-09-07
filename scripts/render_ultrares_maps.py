from __future__ import annotations

import argparse
from pathlib import Path

from worldgen.ultrares_maps import reconstruct_fullview_maps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild full-view scientific PNG maps from the deepest completed "
            "ultra-resolution cube-sphere tiles."
        )
    )
    parser.add_argument("world", type=Path)
    parser.add_argument(
        "--keep-deepest-products",
        action="store_true",
        help="Keep temporary deepest-tile climate/weather/resource products after rendering",
    )
    parser.add_argument(
        "--keep-heavy-solver-caches",
        action="store_true",
        help="Keep local hydrology/constraint and unused geomorphology caches",
    )
    args = parser.parse_args(argv)
    output = reconstruct_fullview_maps(
        args.world,
        cleanup_deepest_products=not args.keep_deepest_products,
        cleanup_heavy_solver_caches=not args.keep_heavy_solver_caches,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
