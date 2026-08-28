from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .planet_tiles import (
    PlanetTilePyramid,
    TileKey,
    TilePyramidSpec,
    approximate_meters_per_sample,
    latlon_to_tile,
)
from .terrain_mesh import write_terrain_mesh


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
        description="Initialize, resolve and lazily generate sparse cube-sphere planet tiles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--world", type=Path, default=Path("world-out"), help="Existing generated world directory")
    p.add_argument("--tile-size", type=int, default=256, help="Terrain cells per tile edge; output vertices are tile-size+1")
    p.add_argument("--maximum-level", type=int, default=24, help="Largest permitted quadtree zoom")
    p.add_argument("--detail-strength", type=float, default=0.20, help="Deterministic sub-grid terrain detail amplitude")
    p.add_argument("--detail-hurst", type=float, default=0.65, help="Fractal amplitude falloff exponent")
    p.add_argument("--detail-harmonics", type=int, default=6, help="Spherical waves evaluated per detail band")
    p.add_argument("--planet-radius-m", type=float, default=None, help="Override radius discovery from world.json")

    request = p.add_mutually_exclusive_group()
    request.add_argument("--tile", type=_tile_address, help="Generate one explicit FACE/Z/X/Y tile")
    request.add_argument("--at", type=_latlon, metavar="LAT,LON", help="Generate the tile containing a geographic point")
    request.add_argument("--visible", type=_latlon, metavar="LAT,LON", help="Generate all tiles intersecting a viewing cap")

    p.add_argument("--level", type=int, default=None, help="Explicit zoom level for --at/--visible")
    p.add_argument("--meters-per-sample", type=float, default=None, help="Choose zoom from target ground resolution")
    p.add_argument("--angular-radius-deg", type=float, default=1.0, help="Viewing-cap radius for --visible")
    p.add_argument("--maximum-visible-tiles", type=int, default=4096, help="Safety cap for one --visible request")
    p.add_argument(
        "--field",
        dest="fields",
        action="append",
        default=[],
        help="Field to materialize; repeat for multiple fields (default: elevation_m)",
    )
    p.add_argument("--mesh", action="store_true", help="Also cache a render-ready local terrain mesh with perimeter skirts")
    p.add_argument("--skirt-depth-m", type=float, default=None, help="Override automatic mixed-LOD terrain skirt depth")
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
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.skirt_depth_m is not None and args.skirt_depth_m < 0:
        parser.error("--skirt-depth-m must be >= 0")
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
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    fields = tuple(args.fields) if args.fields else ("elevation_m",)
    payload: dict[str, object] = {
        "tileset": str(pyramid.manifest_path),
        "planet_radius_m": pyramid.planet_radius_m,
        "tile_size": pyramid.spec.tile_size,
    }

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
