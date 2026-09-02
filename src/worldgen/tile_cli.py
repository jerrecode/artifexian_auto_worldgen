from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Sequence

from .local_downscaling import LocalTileDownscaler
from .planet_tiles import (
    PlanetTilePyramid,
    TileKey,
    TilePyramidSpec,
    approximate_meters_per_sample,
    latlon_to_tile,
)
from .precompute import (
    PrecomputeLimitError,
    PrecomputeProducts,
    enforce_precompute_limits,
    make_precompute_plan,
    precompute_complete_prefix,
)
from .terrain_mesh import write_terrain_mesh
from .tile_pins import pin_complete_prefix
from .tile_products import TileProductExporter


def _tile_address(value: str) -> TileKey:
    text = value.strip().replace(":", "/")
    try:
        face, z, x, y = text.split("/", 3)
        return TileKey(face.lower(), int(z), int(x), int(y)).validate()
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            "tile must be FACE/Z/X/Y, for example px/8/120/90"
        ) from exc


def _latlon(value: str) -> tuple[float, float]:
    try:
        lat_s, lon_s = value.split(",", 1)
        lat, lon = float(lat_s), float(lon_s)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("coordinate must be LAT,LON") from exc
    if not -90.0 <= lat <= 90.0:
        raise argparse.ArgumentTypeError("latitude must be in [-90, 90]")
    if not -180.0 <= lon <= 180.0:
        raise argparse.ArgumentTypeError("longitude must be in [-180, 180]")
    return lat, lon


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="worldgen-tiles",
        description=(
            "Generate sparse cube-sphere planet tiles on demand or precompute a complete "
            "quadtree prefix through a selected depth"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--world", type=Path, default=Path("world-out"), help="Existing generated world directory")
    p.add_argument("--tile-size", type=int, default=256, help="Terrain cells per tile edge; output vertices are tile-size+1")
    p.add_argument("--maximum-level", type=int, default=24, help="Largest permitted quadtree zoom")
    p.add_argument("--detail-strength", type=float, default=0.20, help="Deterministic sub-grid terrain detail amplitude")
    p.add_argument("--detail-hurst", type=float, default=0.65, help="Fractal amplitude falloff exponent")
    p.add_argument("--detail-harmonics", type=int, default=6, help="Spherical waves evaluated per detail band")
    p.add_argument("--planet-radius-m", type=float, default=None, help="Override radius discovery from world.json")
    p.add_argument(
        "--source-level",
        type=int,
        default=None,
        help="Global refinement level to sample; default uses the deepest complete level (0 forces base NPZ)",
    )

    request = p.add_mutually_exclusive_group()
    request.add_argument("--tile", type=_tile_address, help="Generate one explicit FACE/Z/X/Y tile")
    request.add_argument("--at", type=_latlon, metavar="LAT,LON", help="Generate the tile containing a geographic point")
    request.add_argument("--visible", type=_latlon, metavar="LAT,LON", help="Generate all tiles intersecting a viewing cap")
    request.add_argument(
        "--precompute-depth",
        type=int,
        default=None,
        metavar="Z",
        help="Materialize every tile on all six cube faces at every level 0..Z inclusive",
    )

    p.add_argument("--level", type=int, default=None, help="Explicit zoom level for --at/--visible")
    p.add_argument("--meters-per-sample", type=float, default=None, help="Choose zoom from target ground resolution")
    p.add_argument("--angular-radius-deg", type=float, default=1.0, help="Viewing-cap radius for --visible")
    p.add_argument("--maximum-visible-tiles", type=int, default=4096, help="Safety cap for one --visible request")
    p.add_argument(
        "--field",
        dest="fields",
        action="append",
        default=[],
        help="Scientific field to materialize; repeat for multiple fields (default: elevation_m)",
    )
    p.add_argument("--mesh", action="store_true", help="Also cache a render-ready local terrain mesh with perimeter skirts")
    p.add_argument("--skirt-depth-m", type=float, default=None, help="Override automatic mixed-LOD terrain skirt depth")
    p.add_argument("--local-temperature", action="store_true", help="Cache terrain-downscaled annual temperature")
    p.add_argument("--local-temperature-monthly", action="store_true", help="Cache terrain-downscaled monthly temperature")
    p.add_argument("--height-png", action="store_true", help="Cache globally decoded 16-bit PNG height tiles")
    p.add_argument("--true-color-png", action="store_true", help="Cache bilinearly sampled global true-colour PNG tiles")
    p.add_argument("--terrain-temperature-png", action="store_true", help="Cache diagnostic terrain/local-temperature PNG tiles")

    pre = p.add_argument_group("complete-prefix precomputation")
    pre.add_argument("--precompute-workers", type=int, default=1, help="Parallel workers for complete-prefix generation")
    pre.add_argument("--precompute-max-tiles", type=int, default=100_000, help="Safety tile-count limit before an explicit override is required")
    pre.add_argument("--precompute-max-gib", type=float, default=16.0, help="Safety limit for estimated uncompressed array/mesh payload")
    pre.add_argument("--precompute-force-large", action="store_true", help="Bypass precompute tile/storage safety limits after reviewing the plan")
    pre.add_argument("--precompute-plan-only", action="store_true", help="Print the tile-count/storage plan without materializing tiles")
    pre.add_argument(
        "--precompute-no-pin",
        action="store_true",
        help="Do not protect the completed prefix from later interactive disk-LRU eviction",
    )
    pre.add_argument("--precompute-progress-every", type=int, default=128, help="Status-manifest and stderr progress interval in completed tiles")
    pre.add_argument("--precompute-orography", action="store_true", help="Precompute terrain normals/slope and terrain-aware wind/precipitation")
    pre.add_argument("--precompute-surface", action="store_true", help="Precompute soil moisture, snow, vegetation, albedo and local biome proxy")
    pre.add_argument("--precompute-hydrology", action="store_true", help="Precompute open-boundary local runoff/drainage/streams")
    pre.add_argument("--precompute-geomorphology", action="store_true", help="Precompute bounded river-constrained erosion/deposition/diffusion")
    pre.add_argument("--precompute-vectors", action="store_true", help="Precompute sparse cube-sphere vector feature tiles")
    pre.add_argument(
        "--precompute-all-derived",
        action="store_true",
        help=(
            "Enable local annual/monthly temperature, orography, surface state, hydrology, "
            "geomorphology and vectors for every precomputed tile"
        ),
    )

    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON only")
    return p


def _level(args, pyramid: PlanetTilePyramid, parser: argparse.ArgumentParser) -> int:
    if args.level is not None and args.meters_per_sample is not None:
        parser.error("use either --level or --meters-per-sample, not both")
    if args.level is not None:
        if not 0 <= args.level <= pyramid.spec.maximum_level:
            parser.error(f"--level must be in [0, {pyramid.spec.maximum_level}]")
        return int(args.level)
    if args.meters_per_sample is not None:
        if args.meters_per_sample <= 0:
            parser.error("--meters-per-sample must be > 0")
        return pyramid.level_for_resolution(args.meters_per_sample)
    return 0


def _result_payload(
    pyramid: PlanetTilePyramid,
    key: TileKey,
    fields: Sequence[str],
    *,
    mesh: bool,
    skirt_depth_m: float | None,
    downscaler: LocalTileDownscaler | None,
    products: TileProductExporter | None,
    local_temperature: bool,
    local_temperature_monthly: bool,
    height_png: bool,
    true_color_png: bool,
    terrain_temperature_png: bool,
):
    result = pyramid.generate_tile(key, fields)
    payload = {
        "key": {"face": key.face, "level": key.level, "x": key.x, "y": key.y},
        "meters_per_sample_approx": approximate_meters_per_sample(
            pyramid.planet_radius_m, key.level, pyramid.spec.tile_size
        ),
        "cache_hit": result.cache_hit,
        "metadata": str(result.metadata_path),
        "fields": {name: str(path) for name, path in result.fields.items()},
    }
    if mesh:
        payload["mesh"] = str(
            write_terrain_mesh(
                pyramid,
                key,
                skirt_depth_m=skirt_depth_m,
                overwrite=skirt_depth_m is not None,
            )
        )
    if downscaler is not None:
        derived = {}
        if local_temperature:
            downscaler.annual_temperature_c(key)
            derived["annual_temperature_c"] = str(
                downscaler._path(key, "annual_temperature_c")
            )
        if local_temperature_monthly:
            downscaler.monthly_temperature_c(key)
            derived["temperature_c_monthly"] = str(
                downscaler._path(key, "temperature_c_monthly")
            )
        if derived:
            payload["downscaled_fields"] = derived
    if products is not None:
        viewer = {}
        if height_png:
            viewer["height_png"] = str(products.height_png(key))
            viewer["height_encoding"] = str(products.encoding_path)
        if true_color_png:
            viewer["true_color_png"] = str(products.true_color_png(key))
        if terrain_temperature_png:
            viewer["terrain_temperature_png"] = str(
                products.terrain_temperature_png(key)
            )
        if viewer:
            payload["viewer_products"] = viewer
    return payload


def _precompute_products(args, fields: tuple[str, ...]) -> PrecomputeProducts:
    all_derived = bool(args.precompute_all_derived)
    return PrecomputeProducts(
        fields=fields,
        mesh=bool(args.mesh),
        skirt_depth_m=args.skirt_depth_m,
        local_temperature=bool(args.local_temperature or all_derived),
        local_temperature_monthly=bool(args.local_temperature_monthly or all_derived),
        orography=bool(args.precompute_orography or all_derived),
        surface=bool(args.precompute_surface or all_derived),
        hydrology=bool(args.precompute_hydrology or all_derived),
        geomorphology=bool(args.precompute_geomorphology or all_derived),
        vectors=bool(args.precompute_vectors or all_derived),
        height_png=bool(args.height_png),
        true_color_png=bool(args.true_color_png),
        terrain_temperature_png=bool(args.terrain_temperature_png),
    ).validate()


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.skirt_depth_m is not None and args.skirt_depth_m < 0:
        parser.error("--skirt-depth-m must be >= 0")
    if args.precompute_plan_only and args.precompute_depth is None:
        parser.error("--precompute-plan-only requires --precompute-depth")
    if args.precompute_no_pin and args.precompute_depth is None:
        parser.error("--precompute-no-pin requires --precompute-depth")
    if args.precompute_max_gib <= 0:
        parser.error("--precompute-max-gib must be > 0")
    if args.precompute_max_tiles < 1:
        parser.error("--precompute-max-tiles must be positive")
    try:
        spec = TilePyramidSpec(
            tile_size=args.tile_size,
            elevation_detail_strength=args.detail_strength,
            detail_hurst_exponent=args.detail_hurst,
            detail_harmonics=args.detail_harmonics,
            maximum_level=args.maximum_level,
        ).validate()
        pyramid = PlanetTilePyramid(
            args.world,
            spec=spec,
            planet_radius_m=args.planet_radius_m,
            source_level=args.source_level,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    fields = tuple(args.fields) if args.fields else ("elevation_m",)
    payload: dict[str, object] = {
        "tileset": str(pyramid.manifest_path),
        "planet_radius_m": pyramid.planet_radius_m,
        "tile_size": pyramid.spec.tile_size,
        "source_kind": pyramid.source_kind,
        "source_level": pyramid.source_level,
        "source_resolution": list(reversed(pyramid._source_metadata()[0])),
    }

    if args.precompute_depth is not None:
        if args.level is not None or args.meters_per_sample is not None:
            parser.error("--level/--meters-per-sample do not apply to --precompute-depth")
        cfg = _precompute_products(args, fields)
        try:
            plan = make_precompute_plan(pyramid, args.precompute_depth, cfg)
        except (OSError, KeyError, ValueError, RuntimeError) as exc:
            parser.error(str(exc))
        payload["precompute_plan"] = {
            "maximum_level": plan.maximum_level,
            "tile_count": plan.tile_count,
            "tile_size": plan.tile_size,
            "estimated_uncompressed_bytes": plan.estimated_uncompressed_bytes,
            "estimated_uncompressed_gib": plan.estimated_uncompressed_gib,
            "products": asdict(plan.products),
            "pin_on_success": not bool(args.precompute_no_pin),
            "note": (
                "estimate covers uncompressed scientific arrays/meshes; PNG/vector/filesystem "
                "overhead is data-dependent"
            ),
        }
        if not args.precompute_plan_only:
            try:
                enforce_precompute_limits(
                    plan,
                    maximum_tiles=args.precompute_max_tiles,
                    maximum_estimated_bytes=int(args.precompute_max_gib * 1024**3),
                    force_large=bool(args.precompute_force_large),
                )

                interval = max(1, int(args.precompute_progress_every))

                def show_progress(done: int, total: int, key: TileKey) -> None:
                    if done == total or done % interval == 0:
                        print(
                            f"precompute {done:,}/{total:,} ({100.0 * done / total:.2f}%) "
                            f"last={key.face}/{key.level}/{key.x}/{key.y}",
                            file=sys.stderr,
                            flush=True,
                        )

                report = precompute_complete_prefix(
                    pyramid,
                    plan.maximum_level,
                    products=cfg,
                    workers=args.precompute_workers,
                    maximum_tiles=args.precompute_max_tiles,
                    maximum_estimated_bytes=int(args.precompute_max_gib * 1024**3),
                    force_large=bool(args.precompute_force_large),
                    progress_every=interval,
                    progress=show_progress,
                )
                payload["precompute"] = asdict(report)
                if not args.precompute_no_pin:
                    payload["pinned_prefix"] = asdict(
                        pin_complete_prefix(pyramid, plan.maximum_level)
                    )
            except (OSError, KeyError, ValueError, RuntimeError, PrecomputeLimitError) as exc:
                parser.error(str(exc))
        payload["tiles"] = []
    else:
        if any(
            (
                args.precompute_orography,
                args.precompute_surface,
                args.precompute_hydrology,
                args.precompute_geomorphology,
                args.precompute_vectors,
                args.precompute_all_derived,
            )
        ):
            parser.error("--precompute-* derived flags require --precompute-depth")
        want_downscaled = bool(args.local_temperature or args.local_temperature_monthly)
        want_products = bool(
            args.height_png or args.true_color_png or args.terrain_temperature_png
        )
        downscaler = LocalTileDownscaler(pyramid) if (want_downscaled or args.terrain_temperature_png) else None
        products = TileProductExporter(pyramid, downscaler=downscaler) if want_products else None
        try:
            if args.tile is not None:
                keys = (args.tile,)
            elif args.at is not None:
                level = _level(args, pyramid, parser)
                keys = (latlon_to_tile(args.at[0], args.at[1], level),)
            elif args.visible is not None:
                level = _level(args, pyramid, parser)
                if args.angular_radius_deg < 0:
                    parser.error("--angular-radius-deg must be >= 0")
                keys = pyramid.select_visible(
                    latitude_deg=args.visible[0],
                    longitude_deg=args.visible[1],
                    angular_radius_deg=args.angular_radius_deg,
                    level=level,
                    maximum_tiles=args.maximum_visible_tiles,
                )
            else:
                keys = ()

            payload["tiles"] = [
                _result_payload(
                    pyramid,
                    key,
                    fields,
                    mesh=bool(args.mesh),
                    skirt_depth_m=args.skirt_depth_m,
                    downscaler=downscaler,
                    products=products,
                    local_temperature=bool(args.local_temperature),
                    local_temperature_monthly=bool(args.local_temperature_monthly),
                    height_png=bool(args.height_png),
                    true_color_png=bool(args.true_color_png),
                    terrain_temperature_png=bool(args.terrain_temperature_png),
                )
                for key in keys
            ]
            if args.meters_per_sample is not None:
                payload["requested_meters_per_sample"] = float(args.meters_per_sample)
            if args.visible is not None:
                payload["visible_request"] = {
                    "latitude_deg": args.visible[0],
                    "longitude_deg": args.visible[1],
                    "angular_radius_deg": args.angular_radius_deg,
                    "tile_count": len(keys),
                }
        except (OSError, KeyError, ValueError, RuntimeError) as exc:
            parser.error(str(exc))

    if args.json:
        print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
