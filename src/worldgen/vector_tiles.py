from __future__ import annotations

"""Sparse vector feature tiling on the authoritative cube-sphere quadtree.

The internal vector representation is GeoJSON-like and projection-neutral at the
feature level (longitude/latitude coordinates plus stable source IDs). It is not an
OGC/Cesium export format. Standard-specific exporters can retile/encode these
features later without changing world generation.
"""

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np

from .planet_tiles import (
    PlanetTilePyramid,
    TileKey,
    direction_to_tile,
    latlon_to_unit,
)


@dataclass(slots=True, frozen=True)
class VectorTileSpec:
    include_shoreline: bool = True
    segment_samples_per_tile_edge: int = 4

    def validate(self) -> "VectorTileSpec":
        if not 2 <= int(self.segment_samples_per_tile_edge) <= 32:
            raise ValueError("segment_samples_per_tile_edge must be in [2,32]")
        return self


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _unit_to_lonlat(direction: np.ndarray) -> tuple[float, float]:
    p = np.asarray(direction, dtype=np.float64)
    p /= max(float(np.linalg.norm(p)), 1e-300)
    lat = math.degrees(math.asin(float(np.clip(p[2], -1.0, 1.0))))
    lon = math.degrees(math.atan2(float(p[1]), float(p[0])))
    return lon, lat


def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    dot = float(np.clip(np.dot(aa, bb), -1.0, 1.0))
    omega = math.acos(dot)
    if omega < 1e-12:
        p = (1.0 - t) * aa + t * bb
    else:
        s = math.sin(omega)
        p = math.sin((1.0 - t) * omega) / s * aa + math.sin(t * omega) / s * bb
    return p / max(float(np.linalg.norm(p)), 1e-300)


def _segment_tile_pieces(
    start_lonlat: Sequence[float],
    end_lonlat: Sequence[float],
    *,
    level: int,
    samples_per_tile_edge: int,
) -> Iterator[tuple[TileKey, list[list[float]]]]:
    lon0, lat0 = map(float, start_lonlat[:2])
    lon1, lat1 = map(float, end_lonlat[:2])
    a = latlon_to_unit(lat0, lon0)
    b = latlon_to_unit(lat1, lon1)
    angle = math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0)))
    nominal_tile_angle = (math.pi / 2.0) / float(1 << int(level))
    step = nominal_tile_angle / max(2, int(samples_per_tile_edge))
    count = max(1, int(math.ceil(angle / max(step, 1e-12))))
    directions = [_slerp(a, b, i / count) for i in range(count + 1)]
    keys = [direction_to_tile(p, level) for p in directions]

    current_key = keys[0]
    current: list[list[float]] = [list(_unit_to_lonlat(directions[0]))]
    previous = directions[0]
    for direction, next_key in zip(directions[1:], keys[1:]):
        if next_key == current_key:
            current.append(list(_unit_to_lonlat(direction)))
            previous = direction
            continue
        # Locate the first quadtree-address transition along this short subsegment.
        lo = previous
        hi = direction
        for _ in range(32):
            mid = _slerp(lo, hi, 0.5)
            if direction_to_tile(mid, level) == current_key:
                lo = mid
            else:
                hi = mid
        boundary = _slerp(lo, hi, 0.5)
        boundary_ll = list(_unit_to_lonlat(boundary))
        current.append(boundary_ll)
        if len(current) >= 2:
            yield current_key, current
        current_key = next_key
        current = [boundary_ll, list(_unit_to_lonlat(direction))]
        previous = direction
    if len(current) >= 2:
        yield current_key, current


def _marching_shoreline_segments(
    elevation_km: np.ndarray, lat: np.ndarray, lon: np.ndarray
) -> Iterator[list[list[float]]]:
    """Yield coarse zero-elevation contour segments from the global raster."""
    z = np.asarray(elevation_km, dtype=np.float64)
    lat1 = np.asarray(lat, dtype=np.float64)
    lon1 = np.asarray(lon, dtype=np.float64)
    h, w = z.shape
    if lat1.shape != (h,) or lon1.shape != (w,):
        raise ValueError("lat/lon axes do not match elevation")

    def interp(p0, p1, v0, v1):
        denom = v1 - v0
        t = 0.5 if abs(denom) < 1e-15 else float(np.clip(-v0 / denom, 0.0, 1.0))
        return [p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])]

    for y in range(h - 1):
        for x in range(w):
            x1 = (x + 1) % w
            lon0 = float(lon1[x])
            lon_e = float(lon1[x1])
            if x1 == 0:
                lon_e += 360.0
            points = (
                [lon0, float(lat1[y])],
                [lon_e, float(lat1[y])],
                [lon_e, float(lat1[y + 1])],
                [lon0, float(lat1[y + 1])],
            )
            values = (
                float(z[y, x]), float(z[y, x1]), float(z[y + 1, x1]), float(z[y + 1, x])
            )
            crossings: list[list[float]] = []
            edges = ((0, 1), (1, 2), (2, 3), (3, 0))
            for i, j in edges:
                vi, vj = values[i], values[j]
                if (vi < 0.0) != (vj < 0.0) or vi == 0.0 or vj == 0.0:
                    crossings.append(interp(points[i], points[j], vi, vj))
            if len(crossings) == 2:
                yield crossings
            elif len(crossings) == 4:
                # Ambiguous saddle: deterministic pairing avoids topology changes
                # caused by feature iteration order.
                yield [crossings[0], crossings[1]]
                yield [crossings[2], crossings[3]]


def _feature_id(feature: Mapping[str, object], index: int) -> str:
    value = feature.get("id")
    if value is not None:
        return str(value)
    props = feature.get("properties")
    if isinstance(props, dict):
        for key in ("feature_id", "settlement_id", "track_id", "name"):
            if key in props:
                return f"{props.get('feature_class', 'feature')}:{props[key]}"
    return f"feature:{index}"


class VectorTilePyramid:
    def __init__(
        self,
        pyramid: PlanetTilePyramid,
        *,
        spec: VectorTileSpec | None = None,
    ) -> None:
        self.pyramid = pyramid
        self.spec = (spec or VectorTileSpec()).validate()
        self.root = pyramid.root / "vectors" / "geojson_v1"
        self._features: list[dict[str, object]] | None = None

    def _path(self, key: TileKey) -> Path:
        return (
            self.root / f"z{key.level:02d}" / key.face
            / f"x{key.x:08d}" / f"y{key.y:08d}.geojson"
        )

    def _source_features(self) -> list[dict[str, object]]:
        if self._features is not None:
            return self._features
        path = self.pyramid.world_root / "features.geojson"
        features: list[dict[str, object]] = []
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw = payload.get("features", []) if isinstance(payload, dict) else []
            if isinstance(raw, list):
                features.extend(value for value in raw if isinstance(value, dict))
        if self.spec.include_shoreline:
            with np.load(self.pyramid.source_path, allow_pickle=False) as z:
                if all(name in z for name in ("elevation_km", "lat", "lon")):
                    for i, coords in enumerate(
                        _marching_shoreline_segments(z["elevation_km"], z["lat"], z["lon"])
                    ):
                        features.append(
                            {
                                "type": "Feature",
                                "id": f"shoreline:{i}",
                                "geometry": {"type": "LineString", "coordinates": coords},
                                "properties": {
                                    "feature_class": "shoreline",
                                    "source": "global_elevation_zero_contour",
                                },
                            }
                        )
        self._features = features
        return features

    def _point_feature(self, feature: Mapping[str, object], key: TileKey) -> dict[str, object] | None:
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict) or geometry.get("type") != "Point":
            return None
        coordinates = geometry.get("coordinates")
        if not isinstance(coordinates, list) or len(coordinates) < 2:
            return None
        lon, lat = float(coordinates[0]), float(coordinates[1])
        if direction_to_tile(latlon_to_unit(lat, lon), key.level) != key:
            return None
        return dict(feature)

    def _line_features(
        self, feature: Mapping[str, object], source_id: str, key: TileKey
    ) -> list[dict[str, object]]:
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict):
            return []
        kind = geometry.get("type")
        raw = geometry.get("coordinates")
        if kind == "LineString":
            lines = [raw]
        elif kind == "MultiLineString":
            lines = raw
        else:
            return []
        if not isinstance(lines, list):
            return []
        output: list[dict[str, object]] = []
        part = 0
        properties = feature.get("properties")
        props = dict(properties) if isinstance(properties, dict) else {}
        for line in lines:
            if not isinstance(line, list) or len(line) < 2:
                continue
            for a, b in zip(line[:-1], line[1:]):
                if not isinstance(a, list) or not isinstance(b, list) or len(a) < 2 or len(b) < 2:
                    continue
                for piece_key, coords in _segment_tile_pieces(
                    a,
                    b,
                    level=key.level,
                    samples_per_tile_edge=self.spec.segment_samples_per_tile_edge,
                ):
                    if piece_key != key:
                        continue
                    output.append(
                        {
                            "type": "Feature",
                            "id": f"{source_id}:part:{part}",
                            "geometry": {"type": "LineString", "coordinates": coords},
                            "properties": {**props, "source_feature_id": source_id},
                        }
                    )
                    part += 1
        return output

    def generate_tile(self, key: TileKey) -> Path:
        key.validate()
        path = self._path(key)
        if path.exists():
            return path
        selected: list[dict[str, object]] = []
        for index, feature in enumerate(self._source_features()):
            source_id = _feature_id(feature, index)
            point = self._point_feature(feature, key)
            if point is not None:
                point = dict(point)
                point["id"] = source_id
                selected.append(point)
                continue
            selected.extend(self._line_features(feature, source_id, key))
        payload = {
            "type": "FeatureCollection",
            "worldgen_vector_tile": {
                "schema_version": 1,
                "key": asdict(key),
                "source_sha256": self.pyramid._source_hash(),
                "feature_count": len(selected),
                "semantics": "sparse internal cube-sphere vector tile; standard-specific exporters may retile/re-encode this content",
            },
            "features": selected,
        }
        _atomic_json(path, payload)
        return path


__all__ = ["VectorTilePyramid", "VectorTileSpec"]
