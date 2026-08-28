from __future__ import annotations

"""GET-only local HTTP delivery for sparse planetary tiles.

The reference service defaults to loopback and exposes only generated world products;
it is not a generic file server.  Physical generation remains in the deterministic
modules, while :class:`PlanetTileRuntime` provides request coalescing and cache quota
enforcement for interactive access.
"""

import argparse
import gzip
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
from typing import Mapping
from urllib.parse import parse_qs, urlparse

from .geodetic_tiles import GeodeticTileKey, GeodeticTilePyramid, GeodeticTileSpec
from .lod import CameraLODRequest, required_fallback_tiles, select_camera_lod
from .planet_tiles import PlanetTilePyramid, TileKey, TilePyramidSpec
from .quantized_mesh_export import quantized_mesh_path, write_quantized_mesh_tile
from .tile_runtime import PlanetTileRuntime
from .vector_tiles import VectorTilePyramid


_FIELD_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")


def _file_etag(path: Path) -> str:
    h = hashlib.blake2b(digest_size=16)
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return '"' + h.hexdigest() + '"'


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _one(query: Mapping[str, list[str]], name: str, default: str | None = None) -> str:
    values = query.get(name)
    if not values:
        if default is None:
            raise ValueError(f"missing query parameter {name!r}")
        return default
    return values[-1]


class TileService:
    def __init__(
        self,
        pyramid: PlanetTilePyramid,
        *,
        disk_cache_max_bytes: int | None,
    ) -> None:
        self.pyramid = pyramid
        self.runtime = PlanetTileRuntime(
            pyramid, disk_cache_max_bytes=disk_cache_max_bytes
        )
        self.vectors = VectorTilePyramid(pyramid)
        self.geodetic = GeodeticTilePyramid(
            pyramid,
            spec=GeodeticTileSpec(
                tile_size=pyramid.spec.tile_size,
                maximum_level=pyramid.spec.maximum_level,
            ),
        )

    def info(self) -> dict[str, object]:
        return {
            "service": "artifexian-auto-worldgen sparse planet tiles",
            "schema_version": 1,
            "planet_radius_m": self.pyramid.planet_radius_m,
            "tile_size": self.pyramid.spec.tile_size,
            "maximum_level": self.pyramid.spec.maximum_level,
            "source_sha256": self.pyramid._source_hash(),
            "projection_internal": "cube_sphere",
            "projection_quantized_mesh": "EPSG:4326/TMS",
            "cache": self.runtime.cache_stats().__dict__,
            "routes": {
                "lod": "/api/lod?lat=...&lon=...&alt=...&width=...&height=...&fov=...&error=...",
                "field": "/tiles/{face}/{z}/{x}/{y}/fields/{field}.npy",
                "vector": "/vectors/{face}/{z}/{x}/{y}.geojson",
                "quantized_mesh_layer": "/quantized-mesh/layer.json",
                "quantized_mesh": "/quantized-mesh/{z}/{x}/{y}.terrain",
            },
        }

    def lod(self, query: Mapping[str, list[str]]) -> dict[str, object]:
        request = CameraLODRequest(
            latitude_deg=float(_one(query, "lat")),
            longitude_deg=float(_one(query, "lon")),
            altitude_m=float(_one(query, "alt")),
            viewport_width_px=int(_one(query, "width", "1280")),
            viewport_height_px=int(_one(query, "height", "720")),
            vertical_fov_deg=float(_one(query, "fov", "60")),
            maximum_screen_error_px=float(_one(query, "error", "2")),
            maximum_level=int(_one(query, "max_level", str(self.pyramid.spec.maximum_level))),
            maximum_tiles=int(_one(query, "max_tiles", "4096")),
        ).validate()
        result = select_camera_lod(
            planet_radius_m=self.pyramid.planet_radius_m,
            tile_size=self.pyramid.spec.tile_size,
            request=request,
        )
        fallbacks = required_fallback_tiles(result.keys)
        return {
            "request": {
                "latitude_deg": request.latitude_deg,
                "longitude_deg": request.longitude_deg,
                "altitude_m": request.altitude_m,
                "maximum_screen_error_px": request.maximum_screen_error_px,
            },
            "footprint_angular_radius_deg": result.footprint_angular_radius_deg,
            "estimated_resident_height_bytes": result.estimated_resident_height_bytes,
            "minimum_level": result.minimum_level,
            "maximum_level": result.maximum_level,
            "tiles": [
                {
                    "face": tile.key.face,
                    "level": tile.key.level,
                    "x": tile.key.x,
                    "y": tile.key.y,
                    "meters_per_sample_approx": tile.meters_per_sample_approx,
                    "screen_error_px": tile.screen_error_px,
                }
                for tile in result.tiles
            ],
            "fallback_parents": [
                {"face": key.face, "level": key.level, "x": key.x, "y": key.y}
                for key in fallbacks
            ],
        }

    def dynamic_quantized_layer(self) -> dict[str, object]:
        maxzoom = int(self.geodetic.spec.maximum_level)
        available = []
        for level in range(maxzoom + 1):
            width = 1 << (level + 1)
            height = 1 << level
            available.append(
                [{"startX": 0, "startY": 0, "endX": width - 1, "endY": height - 1}]
            )
        return {
            "name": "Worldgen Terrain",
            "description": "Lazily generated terrain served by artifexian-auto-worldgen",
            "version": self.pyramid._source_hash()[:16],
            "format": "quantized-mesh-1.0",
            "scheme": "tms",
            "projection": "EPSG:4326",
            "minzoom": 0,
            "maxzoom": maxzoom,
            "bounds": [-180.0, -90.0, 180.0, 90.0],
            "tiles": ["{z}/{x}/{y}.terrain?v={version}"],
            "available": available,
        }


def _handler(service: TileService):
    class Handler(BaseHTTPRequestHandler):
        server_version = "WorldgenTileServer/1.0"

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            # Standard BaseHTTPRequestHandler logging remains concise and local.
            super().log_message(format, *args)

        def _send_bytes(
            self,
            payload: bytes,
            *,
            content_type: str,
            etag: str | None = None,
            content_encoding: str | None = None,
            cache_control: str = "public, max-age=31536000, immutable",
        ) -> None:
            if etag is not None and self.headers.get("If-None-Match") == etag:
                self.send_response(HTTPStatus.NOT_MODIFIED)
                self.send_header("ETag", etag)
                self.end_headers()
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", cache_control)
            if etag is not None:
                self.send_header("ETag", etag)
            if content_encoding is not None:
                self.send_header("Content-Encoding", content_encoding)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(self, value: object, *, cache: bool = False) -> None:
            payload = _json_bytes(value)
            etag = '"' + hashlib.blake2b(payload, digest_size=16).hexdigest() + '"'
            self._send_bytes(
                payload,
                content_type="application/json; charset=utf-8",
                etag=etag,
                cache_control=(
                    "public, max-age=31536000, immutable" if cache else "no-cache"
                ),
            )

        def _send_file(self, path: Path, content_type: str) -> None:
            self._send_bytes(path.read_bytes(), content_type=content_type, etag=_file_etag(path))

        def _error(self, status: HTTPStatus, message: str) -> None:
            payload = _json_bytes({"error": message, "status": int(status)})
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query, keep_blank_values=False)
            try:
                if path in ("/", "/api/info"):
                    self._send_json(service.info())
                    return
                if path == "/healthz":
                    self._send_json({"ok": True, "source_sha256": service.pyramid._source_hash()})
                    return
                if path == "/api/lod":
                    self._send_json(service.lod(query))
                    return
                if path == "/quantized-mesh/layer.json":
                    self._send_json(service.dynamic_quantized_layer(), cache=True)
                    return

                parts = [part for part in path.split("/") if part]
                if len(parts) == 8 and parts[0] == "tiles" and parts[5] == "fields":
                    face, z, x, y = parts[1], int(parts[2]), int(parts[3]), int(parts[4])
                    filename = parts[6] if parts[7] == "" else None
                    # This branch is retained only for defensive clarity; normal
                    # split() removes empty trailing path components.
                    if filename is None:
                        raise ValueError("invalid field URL")
                if len(parts) == 7 and parts[0] == "tiles" and parts[5] == "fields":
                    face, z, x, y = parts[1], int(parts[2]), int(parts[3]), int(parts[4])
                    filename = parts[6]
                    if not filename.endswith(".npy"):
                        raise ValueError("tile field path must end in .npy")
                    field = filename[:-4]
                    if not _FIELD_RE.fullmatch(field):
                        raise ValueError("invalid field name")
                    key = TileKey(face, z, x, y).validate()
                    result = service.runtime.generate_tile(key, (field,))
                    self._send_file(result.fields[field], "application/x-npy")
                    return
                if len(parts) == 5 and parts[0] == "vectors":
                    face, z, x = parts[1], int(parts[2]), int(parts[3])
                    filename = parts[4]
                    if not filename.endswith(".geojson"):
                        raise ValueError("vector path must end in .geojson")
                    y = int(filename[:-8])
                    key = TileKey(face, z, x, y).validate()
                    with service.runtime.coalescer.hold(("vector", face, z, x, y)):
                        vector_path = service.vectors.generate_tile(key)
                        service.runtime.disk_cache.touch((vector_path,))
                        service.runtime.disk_cache.prune(protected=(vector_path,))
                    self._send_file(vector_path, "application/geo+json")
                    return
                if len(parts) == 4 and parts[0] == "quantized-mesh":
                    z, x = int(parts[1]), int(parts[2])
                    filename = parts[3]
                    if not filename.endswith(".terrain"):
                        raise ValueError("quantized-mesh path must end in .terrain")
                    y = int(filename[:-8])
                    key = GeodeticTileKey(z, x, y).validate()
                    with service.runtime.coalescer.hold(("quantized_mesh", z, x, y)):
                        meta = write_quantized_mesh_tile(service.geodetic, key)
                        terrain_path = quantized_mesh_path(service.geodetic, key)
                        service.runtime.disk_cache.touch((terrain_path,))
                        service.runtime.disk_cache.prune(protected=(terrain_path,))
                    raw = terrain_path.read_bytes()
                    compressed = gzip.compress(raw, compresslevel=5, mtime=0)
                    etag = _file_etag(terrain_path)
                    self._send_bytes(
                        compressed,
                        content_type="application/vnd.quantized-mesh",
                        content_encoding="gzip",
                        etag=etag,
                    )
                    return
                self._error(HTTPStatus.NOT_FOUND, "unknown endpoint")
            except (ValueError, KeyError, FileNotFoundError, RuntimeError) as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))

    return Handler


def serve(
    pyramid: PlanetTilePyramid,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    disk_cache_max_bytes: int | None = 8 * 1024**3,
) -> None:
    if not host:
        raise ValueError("host cannot be empty")
    if not 1 <= int(port) <= 65535:
        raise ValueError("port must be in [1,65535]")
    service = TileService(pyramid, disk_cache_max_bytes=disk_cache_max_bytes)
    server = ThreadingHTTPServer((host, int(port)), _handler(service))
    print(f"Worldgen tile service: http://{host}:{port}/")
    print("GET-only service; Ctrl+C to stop.")
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="worldgen-tile-server",
        description="Serve one generated world through a sparse loopback tile API",
    )
    p.add_argument("--world", type=Path, default=Path("world-out"))
    p.add_argument("--host", default="127.0.0.1", help="Bind address; loopback by default")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--maximum-level", type=int, default=24)
    p.add_argument("--detail-strength", type=float, default=0.20)
    p.add_argument("--disk-cache-gb", type=float, default=8.0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not math.isfinite(args.disk_cache_gb) or args.disk_cache_gb < 0:
        raise SystemExit("--disk-cache-gb must be finite and non-negative")
    pyramid = PlanetTilePyramid(
        args.world,
        spec=TilePyramidSpec(
            tile_size=args.tile_size,
            maximum_level=args.maximum_level,
            elevation_detail_strength=args.detail_strength,
        ),
    )
    serve(
        pyramid,
        host=args.host,
        port=args.port,
        disk_cache_max_bytes=int(args.disk_cache_gb * 1024**3),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
