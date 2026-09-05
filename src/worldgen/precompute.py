from __future__ import annotations

"""Complete-prefix precomputation for the sparse planetary tile hierarchy.

The sparse/on-demand architecture remains the default.  This module adds the
complementary offline mode requested by users who want a fully materialized prefix
of the quadtree: every cube-sphere tile from z0 through a chosen depth is generated,
while deeper levels remain sparse and available on demand.

Generation is deterministic, resumable through the existing per-product caches, and
bounded in memory even when the requested prefix contains many tiles.  A bounded
submission window prevents the thread pool from queuing the complete hierarchy at
once.
"""

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from threading import local
from typing import Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .local_downscaling import LocalTileDownscaler
from .local_geomorphology import LocalGeomorphologySolver
from .local_hydrology import LocalHydrologySolver
from .local_orography import LocalOrographicDownscaler
from .local_surface import LocalSurfaceGenerator
from .planet_tiles import CUBE_FACES, PlanetTilePyramid, TileKey
from .terrain_mesh import write_terrain_mesh
from .tile_products import TileProductExporter
from .vector_tiles import VectorTilePyramid


@dataclass(slots=True, frozen=True)
class PrecomputeProducts:
    """Products to materialize for every tile in a complete hierarchy prefix."""

    fields: tuple[str, ...] = ("elevation_m",)
    mesh: bool = False
    skirt_depth_m: float | None = None
    local_temperature: bool = False
    local_temperature_monthly: bool = False
    orography: bool = False
    surface: bool = False
    hydrology: bool = False
    geomorphology: bool = False
    vectors: bool = False
    height_png: bool = False
    true_color_png: bool = False
    terrain_temperature_png: bool = False

    def validate(self) -> "PrecomputeProducts":
        fields = tuple(dict.fromkeys(str(name) for name in self.fields if str(name)))
        if not fields:
            raise ValueError("precomputation requires at least one scientific field")
        if self.skirt_depth_m is not None and (
            not math.isfinite(float(self.skirt_depth_m)) or float(self.skirt_depth_m) < 0
        ):
            raise ValueError("skirt_depth_m must be finite and non-negative")
        if fields != self.fields:
            object.__setattr__(self, "fields", fields)
        return self


@dataclass(slots=True, frozen=True)
class PrecomputePlan:
    maximum_level: int
    tile_count: int
    tile_size: int
    estimated_uncompressed_bytes: int
    estimated_uncompressed_gib: float
    products: PrecomputeProducts


@dataclass(slots=True, frozen=True)
class PrecomputeReport:
    maximum_level: int
    total_tiles: int
    completed_tiles: int
    base_cache_hits: int
    base_generated_tiles: int
    workers: int
    status_path: str
    estimated_uncompressed_bytes: int
    started_at: str
    finished_at: str


@dataclass(slots=True, frozen=True)
class _TileOutcome:
    key: TileKey
    base_cache_hit: bool


class PrecomputeLimitError(RuntimeError):
    pass


def complete_pyramid_tile_count(maximum_level: int) -> int:
    """Number of cube-sphere tiles in levels ``0..maximum_level`` inclusive."""
    level = int(maximum_level)
    if level < 0:
        raise ValueError("maximum_level must be >= 0")
    # 6 * (1 + 4 + ... + 4**level) = 2 * (4**(level+1) - 1)
    return 2 * (4 ** (level + 1) - 1)


def iter_complete_pyramid(maximum_level: int) -> Iterator[TileKey]:
    """Yield a deterministic breadth-first complete cube-sphere hierarchy prefix."""
    level = int(maximum_level)
    if level < 0:
        raise ValueError("maximum_level must be >= 0")
    for z in range(level + 1):
        side = 1 << z
        for face in CUBE_FACES:
            for y in range(side):
                for x in range(side):
                    yield TileKey(face, z, x, y)


def _field_bytes_per_tile(pyramid: PlanetTilePyramid, field: str) -> int:
    n = int(pyramid.spec.tile_size) + 1
    source_name = "elevation_km" if field == "elevation_m" else str(field)
    if field in {"latitude_deg", "longitude_deg"}:
        return n * n * 8
    if field == "ocean_depth_m":
        return n * n * 4
    try:
        (h, w), names = pyramid._source_metadata()
        if source_name not in names:
            # The generator will raise a useful error later.  Keep planning
            # conservative rather than forcing a full source load here.
            return n * n * 8
        a = np.asarray(pyramid._load_source_array(source_name))
        if a.ndim >= 2 and tuple(a.shape[-2:]) == (h, w):
            multiplier = int(np.prod(a.shape[:-2], dtype=np.int64)) if a.ndim > 2 else 1
        elif a.ndim >= 2 and tuple(a.shape[:2]) == (h, w):
            multiplier = int(np.prod(a.shape[2:], dtype=np.int64)) if a.ndim > 2 else 1
        else:
            multiplier = 1
        itemsize = 4 if field == "elevation_m" else max(1, int(a.dtype.itemsize))
        return n * n * multiplier * itemsize
    except (KeyError, OSError, ValueError):
        return n * n * 8


def estimate_precompute_bytes(
    pyramid: PlanetTilePyramid, maximum_level: int, products: PrecomputeProducts
) -> int:
    """Estimate uncompressed payload bytes before filesystem/container overhead."""
    cfg = products.validate()
    n = int(pyramid.spec.tile_size) + 1
    per_tile = sum(_field_bytes_per_tile(pyramid, field) for field in cfg.fields)
    if cfg.mesh:
        # Conservative local float32 positions + uint32 triangle indices including
        # a perimeter skirt.  NPZ overhead is intentionally ignored.
        vertices = n * n + 4 * (n - 1)
        triangles = 2 * (n - 1) * (n - 1) + 8 * (n - 1)
        per_tile += vertices * 3 * 4 + triangles * 3 * 4
    if cfg.local_temperature:
        per_tile += n * n * 4
    if cfg.local_temperature_monthly:
        per_tile += 12 * n * n * 4
    if cfg.orography:
        per_tile += n * n * (3 * 4 + 4 + 12 * 4 * 3 + 4)
    if cfg.surface:
        per_tile += n * n * (4 * 4 + 1)
    if cfg.hydrology:
        per_tile += n * n * (4 + 1 + 4 + 4 + 4 + 1 + 1)
    if cfg.geomorphology:
        per_tile += n * n * (6 * 4 + 1)
    # PNG and vector output sizes are data-dependent.  They are deliberately not
    # included so the estimate is labelled uncompressed-array/mesh payload rather
    # than a false precise disk forecast.
    return complete_pyramid_tile_count(maximum_level) * per_tile


def make_precompute_plan(
    pyramid: PlanetTilePyramid,
    maximum_level: int,
    products: PrecomputeProducts | None = None,
) -> PrecomputePlan:
    level = int(maximum_level)
    if level < 0 or level > int(pyramid.spec.maximum_level):
        raise ValueError(
            f"precompute depth must be in [0, {pyramid.spec.maximum_level}]"
        )
    cfg = (products or PrecomputeProducts()).validate()
    count = complete_pyramid_tile_count(level)
    estimate = estimate_precompute_bytes(pyramid, level, cfg)
    return PrecomputePlan(
        maximum_level=level,
        tile_count=count,
        tile_size=int(pyramid.spec.tile_size),
        estimated_uncompressed_bytes=estimate,
        estimated_uncompressed_gib=estimate / float(1024**3),
        products=cfg,
    )


def enforce_precompute_limits(
    plan: PrecomputePlan,
    *,
    maximum_tiles: int = 100_000,
    maximum_estimated_bytes: int = 16 * 1024**3,
    force_large: bool = False,
) -> None:
    if force_large:
        return
    reasons: list[str] = []
    if plan.tile_count > int(maximum_tiles):
        reasons.append(
            f"{plan.tile_count:,} tiles exceeds safety limit {int(maximum_tiles):,}"
        )
    if plan.estimated_uncompressed_bytes > int(maximum_estimated_bytes):
        reasons.append(
            f"estimated uncompressed payload {plan.estimated_uncompressed_gib:.2f} GiB exceeds "
            f"safety limit {int(maximum_estimated_bytes) / 1024**3:.2f} GiB"
        )
    if reasons:
        raise PrecomputeLimitError(
            "; ".join(reasons)
            + ". Increase the explicit limits or use force_large=True only after checking storage."
        )


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _status_path(pyramid: PlanetTilePyramid, plan: PrecomputePlan) -> Path:
    payload = json.dumps(
        {"level": plan.maximum_level, "products": asdict(plan.products)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:12]
    return pyramid.root / "precompute" / f"prefix_z{plan.maximum_level:02d}_{digest}.json"


class _WorkerProducts:
    def __init__(self, pyramid: PlanetTilePyramid, cfg: PrecomputeProducts) -> None:
        self.pyramid = pyramid
        self.cfg = cfg
        want_downscale = (
            cfg.local_temperature
            or cfg.local_temperature_monthly
            or cfg.terrain_temperature_png
            or cfg.surface
        )
        self.downscaler = LocalTileDownscaler(pyramid) if want_downscale else None
        self.product_exporter = (
            TileProductExporter(pyramid, downscaler=self.downscaler)
            if (cfg.height_png or cfg.true_color_png or cfg.terrain_temperature_png)
            else None
        )
        self.orography = LocalOrographicDownscaler(pyramid) if cfg.orography else None
        self.surface = LocalSurfaceGenerator(pyramid) if cfg.surface else None
        self.hydrology = LocalHydrologySolver(pyramid) if cfg.hydrology else None
        self.geomorphology = LocalGeomorphologySolver(pyramid) if cfg.geomorphology else None
        self.vectors = VectorTilePyramid(pyramid) if cfg.vectors else None

    def generate(self, key: TileKey) -> _TileOutcome:
        cfg = self.cfg
        base = self.pyramid.generate_tile(key, cfg.fields)
        if cfg.mesh:
            write_terrain_mesh(
                self.pyramid,
                key,
                skirt_depth_m=cfg.skirt_depth_m,
                overwrite=cfg.skirt_depth_m is not None,
            )
        if self.downscaler is not None:
            if cfg.local_temperature:
                self.downscaler.annual_temperature_c(key)
            if cfg.local_temperature_monthly:
                self.downscaler.monthly_temperature_c(key)
        if self.orography is not None:
            self.orography.generate(key)
        if self.surface is not None:
            self.surface.generate(key)
        if self.hydrology is not None:
            self.hydrology.solve(key)
        if self.geomorphology is not None:
            self.geomorphology.solve(key)
        if self.vectors is not None:
            self.vectors.generate_tile(key)
        if self.product_exporter is not None:
            if cfg.height_png:
                self.product_exporter.height_png(key)
            if cfg.true_color_png:
                self.product_exporter.true_color_png(key)
            if cfg.terrain_temperature_png:
                self.product_exporter.terrain_temperature_png(key)
        return _TileOutcome(key=key, base_cache_hit=bool(base.cache_hit))


def precompute_complete_prefix(
    pyramid: PlanetTilePyramid,
    maximum_level: int,
    *,
    products: PrecomputeProducts | None = None,
    workers: int = 1,
    maximum_tiles: int = 100_000,
    maximum_estimated_bytes: int = 16 * 1024**3,
    force_large: bool = False,
    progress_every: int = 128,
    progress: Callable[[int, int, TileKey], None] | None = None,
) -> PrecomputeReport:
    """Materialize every tile from z0 through ``maximum_level`` inclusively.

    Existing products are reused, so an interrupted run can simply be invoked again.
    The thread-pool queue is bounded to roughly ``2 * workers`` in-flight tiles.
    """
    plan = make_precompute_plan(pyramid, maximum_level, products)
    enforce_precompute_limits(
        plan,
        maximum_tiles=maximum_tiles,
        maximum_estimated_bytes=maximum_estimated_bytes,
        force_large=force_large,
    )
    worker_count = int(workers)
    if not 1 <= worker_count <= 64:
        raise ValueError("workers must be in [1, 64]")
    interval = max(1, int(progress_every))
    status_path = _status_path(pyramid, plan)
    started = _utc_now()
    completed = 0
    cache_hits = 0
    thread_state = local()

    def context() -> _WorkerProducts:
        value = getattr(thread_state, "products", None)
        if value is None:
            value = _WorkerProducts(pyramid, plan.products)
            thread_state.products = value
        return value

    def run_one(key: TileKey) -> _TileOutcome:
        return context().generate(key)

    def write_status(last_key: TileKey | None, state: str, error: str | None = None) -> None:
        payload: dict[str, object] = {
            "schema_version": 1,
            "state": state,
            "plan": {
                "maximum_level": plan.maximum_level,
                "tile_count": plan.tile_count,
                "tile_size": plan.tile_size,
                "estimated_uncompressed_bytes": plan.estimated_uncompressed_bytes,
                "estimated_uncompressed_gib": plan.estimated_uncompressed_gib,
                "products": asdict(plan.products),
            },
            "progress": {
                "completed_tiles": completed,
                "base_cache_hits": cache_hits,
                "base_generated_tiles": completed - cache_hits,
                "last_key": asdict(last_key) if last_key is not None else None,
            },
            "workers": worker_count,
            "source_sha256": pyramid._source_hash(),
            "started_at": started,
            "updated_at": _utc_now(),
        }
        if error is not None:
            payload["error"] = error
        _atomic_json(status_path, payload)

    write_status(None, "running")
    last_key: TileKey | None = None
    try:
        if worker_count == 1:
            for key in iter_complete_pyramid(plan.maximum_level):
                outcome = run_one(key)
                completed += 1
                cache_hits += int(outcome.base_cache_hit)
                last_key = outcome.key
                if progress is not None:
                    progress(completed, plan.tile_count, outcome.key)
                if completed % interval == 0:
                    write_status(last_key, "running")
        else:
            iterator = iter(iter_complete_pyramid(plan.maximum_level))
            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="worldgen-precompute") as executor:
                pending: dict[Future[_TileOutcome], TileKey] = {}

                def fill() -> None:
                    while len(pending) < 2 * worker_count:
                        try:
                            key = next(iterator)
                        except StopIteration:
                            return
                        pending[executor.submit(run_one, key)] = key

                fill()
                while pending:
                    done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                    for future in done:
                        key = pending.pop(future)
                        outcome = future.result()
                        completed += 1
                        cache_hits += int(outcome.base_cache_hit)
                        last_key = outcome.key
                        if progress is not None:
                            progress(completed, plan.tile_count, outcome.key)
                        if completed % interval == 0:
                            write_status(last_key, "running")
                    fill()
    except Exception as exc:
        write_status(last_key, "failed", f"{type(exc).__name__}: {exc}")
        raise

    finished = _utc_now()
    write_status(last_key, "complete")
    return PrecomputeReport(
        maximum_level=plan.maximum_level,
        total_tiles=plan.tile_count,
        completed_tiles=completed,
        base_cache_hits=cache_hits,
        base_generated_tiles=completed - cache_hits,
        workers=worker_count,
        status_path=str(status_path),
        estimated_uncompressed_bytes=plan.estimated_uncompressed_bytes,
        started_at=started,
        finished_at=finished,
    )


__all__ = [
    "PrecomputeLimitError",
    "PrecomputePlan",
    "PrecomputeProducts",
    "PrecomputeReport",
    "complete_pyramid_tile_count",
    "enforce_precompute_limits",
    "estimate_precompute_bytes",
    "iter_complete_pyramid",
    "make_precompute_plan",
    "precompute_complete_prefix",
]
