from __future__ import annotations

"""Open-boundary local hydrology for sparse high-resolution terrain tiles.

Continental drainage topology remains a global-world responsibility.  This module
uses a deterministic halo around one requested cube-sphere tile, inherits global
runoff and major-river constraints, rebuilds sub-grid D8 drainage with open patch
boundaries, and crops the result back to the tile core.

It intentionally does not reuse global ``flow_to`` indices: those identify cells in
the equirectangular simulation raster and are meaningless in the sparse cube-sphere
address space.
"""

from dataclasses import asdict, dataclass
import heapq
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np

from .planet_tiles import (
    PlanetTilePyramid,
    TileGeometry,
    TileKey,
    _cube_direction,
)


_D8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


@dataclass(slots=True, frozen=True)
class LocalHydrologySpec:
    halo_cells: int = 16
    priority_flood_epsilon_m: float = 0.01
    stream_quantile: float = 0.985
    fallback_runoff_base_fraction: float = 0.24

    def validate(self) -> "LocalHydrologySpec":
        if not 2 <= int(self.halo_cells) <= 256:
            raise ValueError("halo_cells must be in [2, 256]")
        if not math.isfinite(float(self.priority_flood_epsilon_m)) or float(
            self.priority_flood_epsilon_m
        ) < 0.0:
            raise ValueError("priority_flood_epsilon_m must be finite and non-negative")
        if not 0.5 <= float(self.stream_quantile) < 1.0:
            raise ValueError("stream_quantile must be in [0.5, 1)")
        if not 0.0 <= float(self.fallback_runoff_base_fraction) <= 1.0:
            raise ValueError("fallback_runoff_base_fraction must be in [0, 1]")
        return self


@dataclass(slots=True, frozen=True)
class LocalHydrologyResult:
    filled_elevation_m: np.ndarray
    flow_direction_d8: np.ndarray
    runoff_mm_year: np.ndarray
    drainage_area_km2: np.ndarray
    discharge_index: np.ndarray
    streams: np.ndarray
    inherited_major_river: np.ndarray
    metadata: Mapping[str, object]


def _atomic_save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, np.asarray(values), allow_pickle=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
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


def _patch_geometry(key: TileKey, tile_size: int, halo: int) -> TileGeometry:
    """Extend a tile's cube-face parameterization by ``halo`` sample intervals."""
    key.validate()
    n = int(tile_size)
    side = key.side
    offsets = np.arange(-halo, n + halo + 1, dtype=np.float64) / float(n)
    qx = int(key.x) + offsets
    qy = int(key.y) + offsets
    s = -1.0 + 2.0 * qx[None, :] / float(side)
    t = -1.0 + 2.0 * qy[:, None] / float(side)
    s, t = np.broadcast_arrays(s, t)
    xyz = _cube_direction(key.face, s, t)
    lat = np.rad2deg(np.arcsin(np.clip(xyz[..., 2], -1.0, 1.0)))
    lon = np.rad2deg(np.arctan2(xyz[..., 1], xyz[..., 0]))
    return TileGeometry(xyz=xyz, latitude_deg=lat, longitude_deg=lon)


def _resolved_elevation_patch(
    pyramid: PlanetTilePyramid,
    key: TileKey,
    geom: TileGeometry,
) -> np.ndarray:
    base = np.asarray(pyramid._sample_source_field("elevation_m", geom), dtype=np.float64)
    detail = pyramid._spectral_detail(geom.xyz, key.level)
    modulation = 0.65 + 0.35 * np.tanh(np.abs(base) / 1500.0)
    return base + modulation * detail


def _coastal_land(ocean: np.ndarray) -> np.ndarray:
    ocean = np.asarray(ocean, dtype=bool)
    land = ~ocean
    result = np.zeros_like(ocean)
    h, w = ocean.shape
    for dy, dx in _D8:
        sy0 = max(0, -dy)
        sy1 = min(h, h - dy)
        sx0 = max(0, -dx)
        sx1 = min(w, w - dx)
        ty0, ty1 = sy0 + dy, sy1 + dy
        tx0, tx1 = sx0 + dx, sx1 + dx
        result[sy0:sy1, sx0:sx1] |= ocean[ty0:ty1, tx0:tx1]
    return result & land


def _priority_flood_open(
    elevation_m: np.ndarray,
    ocean: np.ndarray,
    *,
    epsilon_m: float,
) -> np.ndarray:
    """Priority-Flood with ocean and patch perimeter as explicit open boundaries."""
    z = np.asarray(elevation_m, dtype=np.float64).copy()
    oc = np.asarray(ocean, dtype=bool)
    if z.ndim != 2 or oc.shape != z.shape:
        raise ValueError("elevation and ocean must be equal-shaped 2-D arrays")
    h, w = z.shape
    if h < 3 or w < 3:
        raise ValueError("open local priority flood requires at least 3x3 samples")
    visited = oc.copy()
    seed = _coastal_land(oc)
    seed[0, :] |= ~oc[0, :]
    seed[-1, :] |= ~oc[-1, :]
    seed[:, 0] |= ~oc[:, 0]
    seed[:, -1] |= ~oc[:, -1]
    heap: list[tuple[float, int, int]] = []
    ys, xs = np.where(seed & ~visited)
    for y, x in zip(ys.tolist(), xs.tolist()):
        visited[y, x] = True
        heapq.heappush(heap, (float(z[y, x]), y, x))
    if not heap and not np.all(oc):
        raise RuntimeError("local priority flood could not establish an open boundary")
    eps = max(float(epsilon_m), 0.0)
    while heap:
        cur, y, x = heapq.heappop(heap)
        for dy, dx in _D8:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w or visited[ny, nx]:
                continue
            visited[ny, nx] = True
            nz = float(z[ny, nx])
            if nz <= cur:
                nz = cur + eps
                z[ny, nx] = nz
            heapq.heappush(heap, (nz, ny, nx))
    return z


def _great_circle_distance_m(a: np.ndarray, b: np.ndarray, radius_m: float) -> np.ndarray:
    dot = np.sum(a * b, axis=-1)
    return float(radius_m) * np.arccos(np.clip(dot, -1.0, 1.0))


def _flow_d8_open(
    filled_elevation_m: np.ndarray,
    ocean: np.ndarray,
    xyz: np.ndarray,
    radius_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Steepest-descent D8 flow with no wrapping and perimeter cells as outlets."""
    z = np.asarray(filled_elevation_m, dtype=np.float64)
    oc = np.asarray(ocean, dtype=bool)
    h, w = z.shape
    best = np.zeros((h, w), dtype=np.float64)
    receiver = np.full((h, w), -1, dtype=np.int64)
    code = np.full((h, w), -1, dtype=np.int8)
    for direction, (dy, dx) in enumerate(_D8):
        sy0 = max(0, -dy)
        sy1 = min(h, h - dy)
        sx0 = max(0, -dx)
        sx1 = min(w, w - dx)
        ty0, ty1 = sy0 + dy, sy1 + dy
        tx0, tx1 = sx0 + dx, sx1 + dx
        source = z[sy0:sy1, sx0:sx1]
        target = z[ty0:ty1, tx0:tx1]
        distance = _great_circle_distance_m(
            xyz[sy0:sy1, sx0:sx1],
            xyz[ty0:ty1, tx0:tx1],
            radius_m,
        )
        slope = (source - target) / np.maximum(distance, 1.0e-6)
        view_best = best[sy0:sy1, sx0:sx1]
        better = slope > view_best
        if np.any(better):
            target_y = np.arange(ty0, ty1, dtype=np.int64)[:, None]
            target_x = np.arange(tx0, tx1, dtype=np.int64)[None, :]
            target_flat = target_y * w + target_x
            receiver_view = receiver[sy0:sy1, sx0:sx1]
            code_view = code[sy0:sy1, sx0:sx1]
            receiver_view[better] = np.broadcast_to(target_flat, better.shape)[better]
            code_view[better] = direction
            view_best[better] = slope[better]
    receiver[oc] = -1
    code[oc] = -1
    receiver[0, :] = -1
    receiver[-1, :] = -1
    receiver[:, 0] = -1
    receiver[:, -1] = -1
    code[0, :] = -1
    code[-1, :] = -1
    code[:, 0] = -1
    code[:, -1] = -1
    return receiver.ravel(), code


def _sample_area_km2(xyz: np.ndarray, radius_m: float) -> np.ndarray:
    """Approximate vertex support area from local great-circle neighbour spacing."""
    h, w, _ = xyz.shape
    dx = np.empty((h, w), dtype=np.float64)
    dy = np.empty((h, w), dtype=np.float64)
    dx[:, 1:-1] = 0.5 * _great_circle_distance_m(
        xyz[:, :-2], xyz[:, 2:], radius_m
    )
    dx[:, 0] = _great_circle_distance_m(xyz[:, 0], xyz[:, 1], radius_m)
    dx[:, -1] = _great_circle_distance_m(xyz[:, -2], xyz[:, -1], radius_m)
    dy[1:-1, :] = 0.5 * _great_circle_distance_m(
        xyz[:-2, :], xyz[2:, :], radius_m
    )
    dy[0, :] = _great_circle_distance_m(xyz[0, :], xyz[1, :], radius_m)
    dy[-1, :] = _great_circle_distance_m(xyz[-2, :], xyz[-1, :], radius_m)
    return np.maximum(dx * dy / 1.0e6, 1.0e-12)


def _accumulate_open(
    filled_elevation_m: np.ndarray,
    receiver_flat: np.ndarray,
    runoff_mm_year: np.ndarray,
    area_km2: np.ndarray,
    ocean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(filled_elevation_m, dtype=np.float64).ravel()
    receiver = np.asarray(receiver_flat, dtype=np.int64)
    area = np.asarray(area_km2, dtype=np.float64).ravel()
    runoff = np.asarray(runoff_mm_year, dtype=np.float64).ravel()
    land = ~np.asarray(ocean, dtype=bool).ravel()
    drainage = area * land
    discharge = np.maximum(runoff, 0.0) * area * land
    # Priority-Flood epsilon makes interior receiver heights strictly lower.  A
    # descending elevation pass is therefore a deterministic topological order.
    for node in np.argsort(z, kind="stable")[::-1]:
        target = int(receiver[node])
        if target >= 0:
            drainage[target] += drainage[node]
            discharge[target] += discharge[node]
    return drainage.reshape(filled_elevation_m.shape), discharge.reshape(
        filled_elevation_m.shape
    )


def _normalize_log(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    active = np.asarray(mask, dtype=bool) & np.isfinite(values) & (values > 0)
    if not np.any(active):
        return out
    logged = np.log1p(np.asarray(values, dtype=np.float64))
    lo = float(np.min(logged[active]))
    hi = float(np.max(logged[active]))
    if hi <= lo:
        out[active] = 1.0
    else:
        out[active] = (logged[active] - lo) / (hi - lo)
    return out


class LocalHydrologySolver:
    """Compute/cache local high-resolution drainage for individual terrain tiles."""

    def __init__(
        self,
        pyramid: PlanetTilePyramid,
        *,
        spec: LocalHydrologySpec | None = None,
    ) -> None:
        self.pyramid = pyramid
        self.spec = (spec or LocalHydrologySpec()).validate()
        self.root = pyramid.root / "derived" / "local_hydrology_v1"

    def _path(self, key: TileKey, field: str) -> Path:
        key.validate()
        return (
            self.root
            / field
            / f"z{key.level:02d}"
            / key.face
            / f"x{key.x:08d}"
            / f"y{key.y:08d}.npy"
        )

    def _metadata_path(self, key: TileKey) -> Path:
        return (
            self.root
            / "metadata"
            / f"z{key.level:02d}"
            / key.face
            / f"x{key.x:08d}"
            / f"y{key.y:08d}.json"
        )

    def _load_cached(self, key: TileKey) -> LocalHydrologyResult | None:
        fields = {
            "filled_elevation_m": np.float32,
            "flow_direction_d8": np.int8,
            "runoff_mm_year": np.float32,
            "drainage_area_km2": np.float32,
            "discharge_index": np.float32,
            "streams": np.bool_,
            "inherited_major_river": np.bool_,
        }
        paths = {name: self._path(key, name) for name in fields}
        meta_path = self._metadata_path(key)
        if not meta_path.exists() or not all(path.exists() for path in paths.values()):
            return None
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        arrays = {
            name: np.load(path, mmap_mode="r", allow_pickle=False)
            for name, path in paths.items()
        }
        return LocalHydrologyResult(metadata=metadata, **arrays)

    def _runoff_patch(
        self, geom: TileGeometry, elevation_m: np.ndarray
    ) -> tuple[np.ndarray, str]:
        _shape, fields = self.pyramid._source_metadata()
        if "runoff_mm_year" in fields:
            runoff = np.asarray(
                self.pyramid._sample_source_field("runoff_mm_year", geom),
                dtype=np.float64,
            )
            return np.maximum(runoff, 0.0), "inherited global runoff_mm_year"
        if "annual_precipitation_mm" not in fields:
            return np.zeros(elevation_m.shape, dtype=np.float64), (
                "no global runoff_mm_year or annual_precipitation_mm; local runoff set to zero"
            )
        precipitation = np.maximum(
            np.asarray(
                self.pyramid._sample_source_field("annual_precipitation_mm", geom),
                dtype=np.float64,
            ),
            0.0,
        )
        frac = float(self.spec.fallback_runoff_base_fraction) + 0.46 * (
            1.0 - np.exp(-precipitation / 1050.0)
        )
        if "annual_temperature_c" in fields:
            temperature = np.asarray(
                self.pyramid._sample_source_field("annual_temperature_c", geom),
                dtype=np.float64,
            )
            # Apply the same resolved-relief lapse correction used by local climate.
            inherited_elevation = np.asarray(
                self.pyramid._sample_source_field("elevation_m", geom),
                dtype=np.float64,
            )
            temperature = temperature - 6.5 * (
                elevation_m - inherited_elevation
            ) / 1000.0
            frac -= np.clip((temperature - 8.0) / 38.0, 0.0, 0.28)
        return precipitation * np.clip(frac, 0.05, 0.92), (
            "fallback runoff derived from inherited precipitation and temperature; "
            "lithology/snow terms unavailable at this local boundary"
        )

    def _major_river_patch(self, geom: TileGeometry) -> np.ndarray:
        _shape, fields = self.pyramid._source_metadata()
        if "rivers" not in fields:
            return np.zeros(geom.latitude_deg.shape, dtype=bool)
        inherited = np.asarray(
            self.pyramid._sample_source_field("rivers", geom), dtype=np.float64
        )
        return inherited >= 0.5

    def solve(self, key: TileKey) -> LocalHydrologyResult:
        key.validate()
        cached = self._load_cached(key)
        if cached is not None:
            return cached
        n = int(self.pyramid.spec.tile_size)
        halo = int(self.spec.halo_cells)
        geom = _patch_geometry(key, n, halo)
        elevation = _resolved_elevation_patch(self.pyramid, key, geom)
        ocean = elevation < 0.0
        filled = _priority_flood_open(
            elevation,
            ocean,
            epsilon_m=float(self.spec.priority_flood_epsilon_m),
        )
        receiver, direction = _flow_d8_open(
            filled,
            ocean,
            geom.xyz,
            self.pyramid.planet_radius_m,
        )
        runoff, runoff_semantics = self._runoff_patch(geom, elevation)
        runoff[ocean] = 0.0
        area = _sample_area_km2(geom.xyz, self.pyramid.planet_radius_m)
        drainage, discharge = _accumulate_open(
            filled, receiver, runoff, area, ocean
        )
        local_discharge = _normalize_log(discharge, ~ocean)
        inherited_river = self._major_river_patch(geom) & ~ocean
        land_values = local_discharge[~ocean]
        if land_values.size:
            threshold = float(np.quantile(land_values, self.spec.stream_quantile))
            streams = (~ocean) & (local_discharge >= max(threshold, 1.0e-12))
        else:
            threshold = 1.0
            streams = np.zeros_like(ocean)
        streams |= inherited_river

        core = (slice(halo, halo + n + 1), slice(halo, halo + n + 1))
        arrays = {
            "filled_elevation_m": np.asarray(filled[core], dtype=np.float32),
            "flow_direction_d8": np.asarray(direction[core], dtype=np.int8),
            "runoff_mm_year": np.asarray(runoff[core], dtype=np.float32),
            "drainage_area_km2": np.asarray(drainage[core], dtype=np.float32),
            "discharge_index": np.asarray(local_discharge[core], dtype=np.float32),
            "streams": np.asarray(streams[core], dtype=np.bool_),
            "inherited_major_river": np.asarray(inherited_river[core], dtype=np.bool_),
        }
        metadata = {
            "schema_version": 1,
            "key": asdict(key),
            "spec": asdict(self.spec),
            "source_sha256": self.pyramid._source_hash(),
            "patch_shape": [int(elevation.shape[0]), int(elevation.shape[1])],
            "core_shape": [n + 1, n + 1],
            "runoff_semantics": runoff_semantics,
            "stream_threshold_discharge_index": threshold,
            "inherited_major_river_cells": int(np.count_nonzero(arrays["inherited_major_river"])),
            "local_stream_cells": int(np.count_nonzero(arrays["streams"])),
            "boundary_semantics": (
                "halo patch has open non-periodic perimeter; continental/global basin topology "
                "is inherited rather than inferred from the tile"
            ),
            "flow_direction_semantics": {
                "type": "D8 direction code",
                "codes": {
                    str(i): [dy, dx] for i, (dy, dx) in enumerate(_D8)
                },
                "outlet": -1,
                "not_global_flow_to": True,
            },
            "limitations": [
                "local stream accumulation can terminate at halo perimeter and is not a substitute for continental drainage area",
                "cube-face-edge halos use normalized extension of the tile face parameterization and are boundary context only",
                "terrain erosion is not applied by this diagnostic drainage solve",
            ],
        }
        for name, values in arrays.items():
            _atomic_save_npy(self._path(key, name), values)
        _atomic_json(self._metadata_path(key), metadata)
        return LocalHydrologyResult(metadata=metadata, **arrays)


__all__ = [
    "LocalHydrologyResult",
    "LocalHydrologySolver",
    "LocalHydrologySpec",
]
