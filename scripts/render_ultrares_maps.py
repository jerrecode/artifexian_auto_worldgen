from __future__ import annotations

import argparse
import json
from pathlib import Path

from worldgen.planet_tiles import TilePyramidSpec
import worldgen.ultrares_maps as ultrares_maps


def _persisted_tile_spec(world: Path) -> TilePyramidSpec | None:
    """Return the exact spec recorded by the already-generated tile authority.

    Full-view rendering is a consumer of the persisted terrain pyramid.  It must
    not replace the generation contract with renderer-local defaults, because
    PlanetTilePyramid deliberately rejects a mismatched spec to protect cached
    scientific results from accidental cross-world/cross-configuration reuse.
    """
    world = world.expanduser().resolve()
    source_level = 0
    refinement_manifest = world / "refinement" / "manifest.json"
    if refinement_manifest.exists():
        payload = json.loads(refinement_manifest.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            source_level = int(payload.get("deepest_complete_level", 0))

    pyramid_root = world / "tiles" / "cubesphere_v1"
    if source_level > 0:
        pyramid_root = pyramid_root / f"refinement_level_{source_level:04d}"
    tileset = pyramid_root / "tileset.json"
    if not tileset.exists():
        return None

    payload = json.loads(tileset.read_text(encoding="utf-8"))
    raw_spec = payload.get("spec") if isinstance(payload, dict) else None
    if not isinstance(raw_spec, dict):
        raise RuntimeError(f"persisted tile manifest has no valid spec: {tileset}")
    spec = TilePyramidSpec(**raw_spec).validate()
    return spec


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

    # reconstruct_fullview_maps historically instantiated TilePyramidSpec with
    # renderer-local defaults.  During a resume that can differ from the spec
    # that generated the persisted terrain and triggers the manifest safety
    # guard.  Pin that constructor to the authoritative recorded spec for this
    # render invocation.  If there is no existing tile authority, retain the
    # legacy behavior so first-time/non-resume rendering still works.
    persisted_spec = _persisted_tile_spec(args.world)
    original_spec_factory = ultrares_maps.TilePyramidSpec
    if persisted_spec is not None:
        ultrares_maps.TilePyramidSpec = lambda **_ignored: persisted_spec  # type: ignore[assignment,misc]
    try:
        output = ultrares_maps.reconstruct_fullview_maps(
            args.world,
            cleanup_deepest_products=not args.keep_deepest_products,
            cleanup_heavy_solver_caches=not args.keep_heavy_solver_caches,
        )
    finally:
        ultrares_maps.TilePyramidSpec = original_spec_factory

    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
