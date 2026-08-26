from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import numpy as np
from scipy import ndimage

from .config import HydrologyConfig, NoiseConfig
from .grid import SphereGrid, normalize01, smooth_periodic
from .terrain import TerrainResult
from .ocean import OceanResult
from .climate import ClimateResult
from .geology import GeologyResult
from .tectonics import TectonicResult
from .noise import hybrid_multifractal, hybrid_noise01, noise_kwargs, HYDRO_BLEND, NoiseBlend, StaticNoiseFields
from .drainage import DrainageGraph


@dataclass(slots=True)
class SurfaceEvolutionResult:
    elevation_km: np.ndarray
    cumulative_erosion_m: np.ndarray
    cumulative_deposition_m: np.ndarray
    sediment_flux_index: np.ndarray
    delta_deposition_m: np.ndarray
    tectonic_uplift_m: np.ndarray
    meander_migration_m: np.ndarray
    meander_potential: np.ndarray
    metadata: dict


@dataclass(slots=True)
class HydrologyResult:
    filled_elevation_km: np.ndarray
    flow_to: np.ndarray
    accumulation: np.ndarray
    drainage_area_km2: np.ndarray
    discharge_index: np.ndarray
    rivers: np.ndarray
    stream_order: np.ndarray
    river_width_proxy: np.ndarray
    lakes: np.ndarray
    basin_id: np.ndarray
    runoff: np.ndarray
    cumulative_erosion_m: np.ndarray
    cumulative_deposition_m: np.ndarray
    sediment_flux_index: np.ndarray
    delta_deposition_m: np.ndarray
    tectonic_uplift_m: np.ndarray
    meander_migration_m: np.ndarray
    meander_potential: np.ndarray
    sinuosity_proxy: np.ndarray
    river_centerlines: list[dict]
    metadata: dict


_FLOOD_NEIGHBORS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
_FLOW_NEIGHBORS = _FLOOD_NEIGHBORS + [
    (-2,-1),(-2,1),(2,-1),(2,1),(-1,-2),(-1,2),(1,-2),(1,2),
    (-2,0),(2,0),(0,-2),(0,2),
]
_LITH_ERODIBILITY = np.array([1.75, 1.20, 0.82, 0.46, 0.36, 0.55, 0.52, 0.58, 0.40], dtype=float)
_LITH_RUNOFF = np.array([0.90, 0.88, 0.68, 1.08, 1.12, 0.96, 1.00, 0.96, 1.05], dtype=float)


def _cell_area_km2(grid: SphereGrid) -> np.ndarray:
    return grid.cell_area_weights * (4.0 * math.pi * grid.radius_km ** 2)


def _priority_flood(elev: np.ndarray, ocean: np.ndarray, grid: SphereGrid) -> np.ndarray:
    """Priority-Flood across land with canonical spherical seam/pole topology."""
    h, w = elev.shape
    z = elev.astype(np.float64).copy()
    visited = ocean.astype(bool).copy()
    heap: list[tuple[float, int, int]] = []
    land = ~ocean
    coastal = land & grid.ops.binary_dilation(ocean, iterations=1)
    ys, xs = np.where(coastal)
    for y, x in zip(ys.tolist(), xs.tolist()):
        visited[y, x] = True
        heapq.heappush(heap, (float(z[y, x]), y, x))
    if not heap:
        for x in range(w):
            for y in (0, h - 1):
                if land[y, x] and not visited[y, x]:
                    visited[y, x] = True
                    heapq.heappush(heap, (float(z[y, x]), y, x))
    eps = 1e-7
    while heap:
        cur, y, x = heapq.heappop(heap)
        for dy, dx in _FLOOD_NEIGHBORS:
            ny = y + dy
            nx = x + dx
            if ny < 0:
                ny = -ny - 1
                nx += w // 2
            elif ny >= h:
                ny = 2 * h - ny - 1
                nx += w // 2
            nx %= w
            if visited[ny, nx]:
                continue
            visited[ny, nx] = True
            nz = float(z[ny, nx])
            if nz <= cur:
                nz = cur + eps
                z[ny, nx] = nz
            heapq.heappush(heap, (nz, ny, nx))
    return z


def _flow_directions(z: np.ndarray, ocean: np.ndarray, grid: SphereGrid) -> np.ndarray:
    """Extended-direction steepest descent with one canonical spherical neighbor rule."""
    h, w = z.shape
    best = np.zeros((h, w), dtype=np.float64)
    receiver = np.full((h, w), -1, dtype=np.int32)
    for dy, dx in _FLOW_NEIGHBORS:
        ny, nx = grid.ops.neighbor_indices(dy, dx)
        nb = z[ny, nx]
        if dx and dy:
            dist = np.hypot(abs(dy) * grid.dy_km, abs(dx) * grid.dx_km)
        elif dx:
            dist = abs(dx) * grid.dx_km
        else:
            dist = np.full_like(grid.dx_km, abs(dy) * grid.dy_km)
        slope = (z - nb) / np.maximum(dist, 1e-6)
        better = slope > best
        if np.any(better):
            target = (ny * w + nx).astype(np.int32, copy=False)
            receiver[better] = target[better]
            best[better] = slope[better]
    receiver[ocean] = -1
    return receiver.ravel()


def _receiver_slope(z: np.ndarray, flow: np.ndarray, grid: SphereGrid) -> np.ndarray:
    h, w = z.shape
    flat = z.ravel()
    idx = np.arange(flat.size)
    j = np.asarray(flow, dtype=np.int64)
    good = j >= 0
    out = np.zeros(flat.size, dtype=np.float64)
    yi, xi = np.divmod(idx[good], w)
    yj, xj = np.divmod(j[good], w)
    dlat = np.abs(yj - yi)
    rawdx = np.abs(xj - xi)
    dlon = np.minimum(rawdx, w - rawdx)
    pole = (dlat == 0) & (dlon > w // 4) & ((yi == 0) | (yi == h - 1))
    dlon = np.where(pole, 0, dlon)
    dist = np.hypot(dlat * grid.dy_km, dlon * grid.dx_km[yi, xi])
    out[good] = np.maximum(flat[good] - flat[j[good]], 0) / np.maximum(dist, 1e-6)
    return out.reshape(h, w)


def _runoff_mm(climate: ClimateResult, land: np.ndarray, geology: GeologyResult | None, cfg: HydrologyConfig) -> np.ndarray:
    p = climate.annual_precipitation_mm.astype(float)
    frac = cfg.runoff_base_fraction + 0.46 * (1.0 - np.exp(-p / 1050.0))
    if geology is not None:
        frac *= _LITH_RUNOFF[np.clip(geology.rock_code, 0, len(_LITH_RUNOFF) - 1)]
    evap_penalty = np.clip((climate.annual_temperature_c - 8.0) / 38.0, 0, 0.28)
    frac = np.clip(frac - evap_penalty + 0.16 * climate.snow_fraction, 0.05, 0.92)
    return (p * frac * land).astype(np.float64)


def _transport_sediment(
    routing_z: np.ndarray, flow: np.ndarray, erosion_m: np.ndarray,
    discharge_norm: np.ndarray, slope: np.ndarray, cell_area: np.ndarray,
    land: np.ndarray, cfg: HydrologyConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Route sediment, deposit on low-gradient reaches, and retain export at river mouths."""
    n = routing_z.size
    area = cell_area.ravel()
    lf = land.ravel()
    recv = flow.astype(np.int64, copy=False)
    q = discharge_norm.ravel()
    s = slope.ravel()
    flatness = np.exp(-s / 0.0028)
    depfrac = np.clip(cfg.deposition_strength * flatness * (0.70 + 0.30 * (1.0 - q)), 0.0, 0.80)
    depfrac *= lf
    load = (erosion_m * cell_area).ravel().astype(float)
    deposited = np.zeros(n, float)
    transit = np.zeros(n, float)
    exported = np.zeros(n, float)
    valid_source = (recv >= 0) & lf
    safe_recv = np.where(recv >= 0, recv, 0)
    target_land = np.zeros(n, bool)
    target_land[valid_source] = lf[safe_recv[valid_source]]
    inland = valid_source & target_land
    outlet = valid_source & ~target_land
    delta_outlet = outlet & (q >= float(np.clip(cfg.delta_min_outlet_discharge_norm, 0.0, 1.0)))
    passes = max(1, int(cfg.sediment_routing_passes))
    for _ in range(passes):
        if float(load.sum()) < 1e-9:
            break
        transit += load
        dvol = load * depfrac
        deposited += dvol
        remain = load - dvol
        if np.any(delta_outlet):
            exported += np.bincount(
                safe_recv[delta_outlet],
                weights=remain[delta_outlet] * np.clip(q[delta_outlet], 0.0, 1.0),
                minlength=n,
            )
        if np.any(inland):
            load = np.bincount(safe_recv[inland], weights=remain[inland], minlength=n).astype(float, copy=False)
        else:
            load = np.zeros(n, float)
    dep_depth = deposited / np.maximum(area, 1e-9)
    dep_depth = np.minimum(dep_depth, 16.0) * lf
    return (
        dep_depth.reshape(routing_z.shape),
        normalize01(np.log1p(transit.reshape(routing_z.shape))),
        exported.reshape(routing_z.shape),
    )


def _delta_deposition(
    grid: SphereGrid, z: np.ndarray, land: np.ndarray, exported_volume: np.ndarray,
    cell_area: np.ndarray, cfg: HydrologyConfig, marine_energy: np.ndarray | None = None,
    distributary_texture: np.ndarray | None = None,
) -> np.ndarray:
    total = float(np.sum(exported_volume)) * float(np.clip(cfg.delta_retention_fraction, 0.0, 1.0))
    if total <= 1e-12:
        return np.zeros_like(z)
    ocean = ~land
    shallow = ocean & (z < 0.0) & (z >= -cfg.delta_max_depth_m / 1000.0)
    if not np.any(shallow):
        return np.zeros_like(z)
    sigma = max(0.55, float(cfg.delta_spread_cells))
    spread = smooth_periodic(exported_volume.astype(float), (sigma, sigma * 1.45))
    if cfg.delta_tide_reworking_strength > 0:
        along = smooth_periodic(exported_volume.astype(float), (max(0.5, sigma * 0.65), max(0.8, sigma * 2.2)))
        spread = (1.0 - float(cfg.delta_tide_reworking_strength)) * spread + float(cfg.delta_tide_reworking_strength) * along
    shallow_pref = np.exp(np.clip(z, -cfg.delta_max_depth_m / 1000.0, 0.0) / (max(cfg.delta_max_depth_m, 1.0) / 1000.0))
    coast_pref = 0.45 + 0.55 * grid.ops.binary_dilation(land, iterations=max(1, int(round(sigma)))).astype(float)
    gy, gx = grid.ops.metric_gradient(z)
    shelf_slope = np.hypot(gx, gy)
    slope_pref = np.exp(-shelf_slope / max(float(cfg.delta_shelf_slope_scale), 1e-5))
    if marine_energy is None:
        marine_pref = 1.0
    else:
        me = np.clip(np.asarray(marine_energy, float), 0.0, 1.5)
        marine_pref = np.clip(1.0 - float(cfg.delta_wave_reworking_strength) * me, 0.18, 1.0)
    texture_pref = 1.0
    if distributary_texture is not None:
        tex = np.clip(np.asarray(distributary_texture, float), 0.0, 1.0)
        st = float(np.clip(cfg.delta_distributary_texture_strength, 0.0, 0.75))
        texture_pref = (1.0 - st) + 2.0 * st * tex
    weights = spread * shallow * shallow_pref * coast_pref * slope_pref * marine_pref * texture_pref
    ws = float(weights.sum())
    if ws <= 1e-12:
        return np.zeros_like(z)
    volume = weights * (total / ws)
    depth = volume / np.maximum(cell_area, 1e-9)
    return np.clip(depth, 0.0, float(cfg.delta_max_aggradation_m_per_iteration)) * shallow


def _meander_field(
    geology: GeologyResult, lith: np.ndarray, slope: np.ndarray, qn: np.ndarray,
    land: np.ndarray, cfg: HydrologyConfig,
) -> np.ndarray:
    soft = np.clip((lith - 0.32) / 1.45, 0.0, 1.0)
    alluvial = np.isin(geology.rock_code, [0, 1, 2]).astype(float)
    lowgrad = np.exp(-slope / max(float(cfg.meander_slope_scale), 1e-5))
    q = np.clip(qn / 1.8, 0.0, 1.0)
    m = (q ** 0.62) * lowgrad * (0.35 + 0.65 * soft) * (0.72 + 0.28 * alluvial) * land
    return np.clip(cfg.river_meander_strength * m, 0.0, 1.0)


def evolve_surface(
    grid: SphereGrid,
    terrain: TerrainResult,
    ocean: OceanResult,
    climate: ClimateResult,
    geology: GeologyResult,
    cfg: HydrologyConfig,
    tectonics: TectonicResult | None = None,
    rng: np.random.Generator | None = None,
    noise_cfg: NoiseConfig | None = None,
    static_noise: StaticNoiseFields | None = None,
) -> SurfaceEvolutionResult:
    z = ocean.elevation_km.astype(np.float64).copy()
    erosion_total = np.zeros_like(z)
    dep_total = np.zeros_like(z)
    flux = np.zeros_like(z)
    delta_total = np.zeros_like(z)
    uplift_total = np.zeros_like(z)
    migration_total = np.zeros_like(z)
    meander = np.zeros_like(z)
    cell_area = _cell_area_km2(grid)
    lith = _LITH_ERODIBILITY[np.clip(geology.rock_code, 0, len(_LITH_ERODIBILITY) - 1)]
    if rng is None:
        wiggle = np.sin(np.deg2rad(grid.lon * 3.7 + grid.lat * 1.9)) + 0.45 * np.sin(np.deg2rad(grid.lon * 11.3 - grid.lat * 4.1))
        delta_texture = 0.5 + 0.5 * np.sin(np.deg2rad(grid.lon * 7.1 - grid.lat * 3.3))
    elif static_noise is not None:
        wiggle = static_noise.hydro_wiggle
        delta_texture = static_noise.delta_texture
    else:
        wiggle = hybrid_multifractal(
            z.shape, rng, base_scale_px=max(grid.height / 36.0, 2.5),
            **noise_kwargs(noise_cfg, profile=HYDRO_BLEND, octaves=max(5, min(8, getattr(noise_cfg, "octaves", 7)))),
        )
        delta_texture = hybrid_noise01(
            z.shape, rng, base_scale_px=max(grid.height / 50.0, 2.0),
            **noise_kwargs(noise_cfg, profile=NoiseBlend(0.36, 0.25, 0.12, 0.27), octaves=max(4, min(7, getattr(noise_cfg, "octaves", 6)))),
        )
    wiggle = (wiggle - np.mean(wiggle)) / max(np.std(wiggle), 1e-8)
    meander_prior = np.clip(lith / np.max(_LITH_ERODIBILITY), 0, 1) * terrain.lowland_strength
    last_flow = None
    last_route = None
    last_graph: DrainageGraph | None = None

    for it in range(max(0, cfg.surface_evolution_iterations)):
        land = z > 0
        oc = ~land
        if it % max(1, cfg.flow_refresh_interval) == 0 or last_flow is None or last_graph is None:
            micro = (cfg.meander_microrelief_m / 1000.0) * wiggle * np.maximum(meander_prior, meander)
            route = _priority_flood(z + micro * land, oc, grid)
            flow = _flow_directions(route, oc, grid)
            graph = DrainageGraph.from_receiver(flow, z.shape)
            last_route, last_flow, last_graph = route, flow, graph
        else:
            route, flow, graph = last_route, last_flow, last_graph
        runoff = _runoff_mm(climate, land, geology, cfg)
        water_source = (runoff / 1000.0) * cell_area
        discharge = graph.accumulate(water_source)
        slope = _receiver_slope(route, flow, grid)
        vals = discharge[land & (discharge > 0)]
        qref = np.quantile(vals, 0.985) if len(vals) else 1.0
        qn = np.clip(discharge / max(qref, 1e-12), 0, 2.5)
        meander = _meander_field(geology, lith, slope, qn, land, cfg)

        recv = flow.astype(np.int64, copy=False)
        lf = land.ravel()
        valid_recv = recv >= 0
        safe = np.where(valid_recv, recv, 0)
        receiver_land = np.zeros(recv.size, bool)
        receiver_land[valid_recv] = lf[safe[valid_recv]]
        receiver_land = receiver_land.reshape(land.shape)
        major_mouth = land & (~receiver_land) & (qn > 0.34)
        fluvial_domain = land & (receiver_land | major_mouth)
        sref = 0.006
        stream_e = cfg.max_fluvial_erosion_m_per_iteration * lith * (qn ** cfg.stream_power_m) * ((slope / sref) ** cfg.stream_power_n)
        stream_e = np.clip(stream_e, 0, cfg.max_fluvial_erosion_m_per_iteration) * fluvial_domain
        weathering = 0.20 * lith * np.clip(climate.annual_precipitation_mm / 1200.0, 0, 1.5) * land
        e = np.clip(stream_e + weathering, 0, cfg.max_fluvial_erosion_m_per_iteration)

        channel = (meander > 0.06) & (qn > 0.08) & land
        banks = grid.ops.binary_dilation(channel, iterations=1) & land & ~channel
        lateral_source = ndimage.maximum_filter(e * meander, size=3)
        asym = 0.35 + 0.65 * normalize01(wiggle, robust=False)
        lateral = np.clip(
            cfg.lateral_erosion_fraction * lateral_source * asym * banks,
            0,
            cfg.max_fluvial_erosion_m_per_iteration * 0.75,
        )
        e_total = np.clip(e + lateral, 0, cfg.max_fluvial_erosion_m_per_iteration * 1.35)
        migration_total += lateral

        dep, load, exported = _transport_sediment(route, flow, e_total, np.clip(qn / 2.5, 0, 1), slope, cell_area, land, cfg)
        delta = _delta_deposition(
            grid, z, land, exported, cell_area, cfg,
            marine_energy=getattr(ocean, "current_speed", None), distributary_texture=delta_texture,
        )

        uplift = np.zeros_like(z)
        subsidence = np.zeros_like(z)
        if tectonics is not None:
            active = 0.78 * tectonics.convergence_strength + 0.22 * tectonics.stress_field
            uplift = cfg.tectonic_uplift_m_per_iteration * (active ** 1.15) * land
            subsidence = cfg.rift_subsidence_m_per_iteration * tectonics.divergence_strength * land
            uplift_total += uplift

        sm = smooth_periodic(z, (0.65, 0.75))
        diffusion = (sm - z) * cfg.hillslope_diffusion_strength * np.clip(lith, 0.35, 1.8) * land
        diffusion = np.clip(diffusion, -0.010, 0.010)
        z += diffusion
        z += (uplift - subsidence) / 1000.0
        z -= e_total / 1000.0
        z += dep / 1000.0
        z += delta / 1000.0
        erosion_total += e_total
        dep_total += dep + delta
        delta_total += delta
        flux = np.maximum(flux, load)
        meander_prior = meander

    meta = {
        "iterations": int(cfg.surface_evolution_iterations),
        "max_cumulative_erosion_m": float(erosion_total.max()),
        "max_cumulative_deposition_m": float(dep_total.max()),
        "max_delta_aggradation_m": float(delta_total.max()),
        "max_tectonic_uplift_m": float(uplift_total.max()),
        "max_meander_bank_migration_m": float(migration_total.max()),
        "mean_land_erosion_m": float(np.average(erosion_total[terrain.land], weights=grid.cell_area_weights[terrain.land])) if np.any(terrain.land) else 0.0,
        "model": "rainfall/snowmelt runoff + lithology-dependent stream power + lateral meander migration + hierarchical sediment routing/shelf-controlled deltas + active tectonic uplift + hillslope diffusion",
        "drainage_graph": "reusable topological order; accumulation is O(N) and Numba-capable",
        "noise_model": "shared hybrid multi-type multifractal channel microrelief and distributary texture",
    }
    return SurfaceEvolutionResult(
        z.astype(np.float32), erosion_total.astype(np.float32), dep_total.astype(np.float32),
        normalize01(flux).astype(np.float32), delta_total.astype(np.float32),
        uplift_total.astype(np.float32), migration_total.astype(np.float32),
        meander.astype(np.float32), meta,
    )


def _lake_mask(
    grid: SphereGrid, z: np.ndarray, filled: np.ndarray, land: np.ndarray,
    drainage_area: np.ndarray, runoff_acc: np.ndarray, climate: ClimateResult, cfg: HydrologyConfig,
) -> np.ndarray:
    depth_m = (filled - z) * 1000.0
    pet = np.maximum(0.0, 26.0 * (climate.annual_temperature_c + 5.0))
    moisture = climate.annual_precipitation_mm / np.maximum(pet, 250.0)
    rv = runoff_acc[land & (runoff_acc > 0)]
    rref = np.quantile(rv, 0.60) if len(rv) else 1.0
    candidate = (
        land & (depth_m >= cfg.lake_min_depth_m) &
        (drainage_area >= cfg.lake_min_catchment_km2) &
        ((runoff_acc >= rref) | (moisture > 0.85))
    )
    labs, n = grid.ops.connected_components(candidate)
    if n == 0:
        return np.zeros_like(candidate)
    ids = np.arange(1, n + 1, dtype=np.int32)
    cell_area = _cell_area_km2(grid)
    areas = np.asarray(ndimage.sum(cell_area, labels=labs, index=ids), dtype=float)
    max_depth = np.asarray(ndimage.maximum(depth_m, labels=labs, index=ids), dtype=float)
    max_drain = np.asarray(ndimage.maximum(drainage_area, labels=labs, index=ids), dtype=float)
    mean_moist = np.asarray(ndimage.mean(moisture, labels=labs, index=ids), dtype=float)
    min_area = max(20.0, 0.08 * cfg.lake_min_catchment_km2)
    valid = (areas >= min_area) & ~((areas > 850000) & (mean_moist < 1.15))
    scores = max_depth * np.log1p(np.maximum(max_drain, 0.0)) * np.clip(mean_moist, 0.15, 3.0)
    order = np.argsort(scores)[::-1]
    max_area = float(cell_area[land].sum()) * float(np.clip(cfg.lake_area_soft_cap_fraction_land, 0.001, 0.20))
    keep = np.zeros(n + 1, dtype=bool)
    used = 0.0
    for j in order:
        if not valid[j]:
            continue
        area = float(areas[j])
        if used + area > max_area and used > 0:
            continue
        keep[j + 1] = True
        used += area
    return keep[labs]


def _build_river_centerlines(
    grid: SphereGrid, flow: np.ndarray, rivers: np.ndarray, accumulation: np.ndarray,
    meander: np.ndarray, geology: GeologyResult | None, max_paths: int = 120,
) -> list[dict]:
    h, w = rivers.shape
    n = h * w
    rf = rivers.ravel()
    recv = flow.astype(np.int64, copy=False)
    up = np.zeros(n, np.int16)
    src = np.flatnonzero(rf & (recv >= 0))
    if src.size:
        tgt = recv[src]
        good = rf[tgt]
        np.add.at(up, tgt[good], 1)
    heads = np.flatnonzero(rf & (up == 0))
    if heads.size == 0:
        return []
    af = accumulation.ravel()
    mf = meander.ravel()
    heads = heads[np.argsort(af[heads])[::-1]]
    heads = heads[:min(len(heads), 900)]
    visited = np.zeros(n, bool)
    raw = []
    for head in heads:
        path = []
        cur = int(head)
        seen = set()
        while cur >= 0 and cur not in seen and len(path) < 2400:
            seen.add(cur)
            path.append(cur)
            nxt = int(recv[cur])
            if nxt < 0:
                break
            cur = nxt
            if not rf[cur] and len(path) > 2:
                path.append(cur)
                break
        if len(path) < 6:
            continue
        overlap = float(np.mean(visited[np.asarray(path[:-1], dtype=int)]))
        if overlap > 0.68:
            continue
        score = len(path) * (0.55 + 0.45 * float(np.mean(mf[np.asarray(path[:-1], dtype=int)])))
        raw.append((score, path))
        visited[np.asarray(path[:-1], dtype=int)] = True
        if len(raw) >= max_paths * 2:
            break
    raw.sort(key=lambda x: x[0], reverse=True)
    out = []
    for rid, (_, path) in enumerate(raw[:max_paths]):
        idx = np.asarray(path, dtype=int)
        yy, xx = np.divmod(idx, w)
        lat = grid.lat_1d[yy].astype(float)
        lon = np.rad2deg(np.unwrap(np.deg2rad(grid.lon_1d[xx].astype(float))))
        mvals = mf[idx]
        k = max(2, int(round(4.0 * 512 / max(w, 1))))
        k = max(2, min(k, 4))
        t = np.arange(len(path), dtype=float)
        td = np.linspace(0, len(path) - 1, (len(path) - 1) * k + 1)
        lati = np.interp(td, t, lat)
        loni = np.interp(td, t, lon)
        mi = np.interp(td, t, mvals)
        if len(td) >= 3:
            dlat = np.gradient(lati)
            dlon = np.gradient(loni) * np.cos(np.deg2rad(lati))
            norm = np.hypot(dlat, dlon) + 1e-9
            east = dlon / norm
            north = dlat / norm
            pe = -north
            pn = east
            step_km = grid.dy_km / max(k, 1)
            ss = np.arange(len(td)) * step_km
            mean_m = float(np.mean(mi))
            wavelength = max(grid.dy_km * 3.0, grid.dy_km * (5.4 - 1.6 * mean_m))
            phase = (int(head) * 0.61803398875) % (2 * np.pi)
            amp = (0.020 + 0.19 * mi) * grid.dy_km
            wave = (
                0.58 * np.sin(2 * np.pi * ss / max(wavelength, 1.0) + phase)
                + 0.27 * np.sin(2 * np.pi * ss / max(wavelength * 1.73, 1.0) + 1.91 * phase + 0.7)
                + 0.15 * np.sin(2 * np.pi * ss / max(wavelength * 0.63, 1.0) + 2.47 * phase + 1.8)
            )
            taper = np.minimum(
                1.0,
                np.minimum(
                    np.arange(len(td)) / max(3, k * 2),
                    (len(td) - 1 - np.arange(len(td))) / max(3, k * 2),
                ),
            )
            offset = amp * wave * np.clip(taper, 0, 1)
            lati = lati + pn * offset / 111.2
            loni = loni + pe * offset / (111.2 * np.maximum(np.cos(np.deg2rad(lati)), 0.18))
        loni = ((loni + 180.0) % 360.0) - 180.0
        rock_code = None
        if geology is not None:
            vals = geology.rock_code.ravel()[idx[:-1] if len(idx) > 1 else idx]
            if vals.size:
                rock_code = int(np.bincount(vals.astype(int)).argmax())
        points = [[float(a), float(b)] for a, b in zip(lati, loni)]
        out.append({
            "river_id": rid,
            "points_lat_lon": points,
            "source_cell": int(head),
            "mean_meander_potential": float(np.mean(mvals)),
            "sinuosity_proxy": float(1.0 + 2.35 * np.mean(mvals)),
            "dominant_rock_code": rock_code,
        })
    return out


def build_hydrology(
    grid: SphereGrid,
    terrain: TerrainResult,
    ocean: OceanResult,
    climate: ClimateResult,
    cfg: HydrologyConfig,
    geology: GeologyResult | None = None,
    surface: SurfaceEvolutionResult | None = None,
) -> HydrologyResult:
    z0 = ocean.elevation_km.astype(float)
    filled = _priority_flood(z0, terrain.ocean, grid)
    flow = _flow_directions(filled, terrain.ocean, grid)
    graph = DrainageGraph.from_receiver(flow, z0.shape)
    runoff = _runoff_mm(climate, terrain.land, geology, cfg)
    cell_area = _cell_area_km2(grid)
    water_source = (runoff / 1000.0) * cell_area
    acc = graph.accumulate(water_source)
    drainage = graph.accumulate(cell_area * terrain.land)
    slope = _receiver_slope(filled, flow, grid)

    source_valid = terrain.land & (climate.annual_precipitation_mm >= cfg.min_river_precip_mm_year) & (drainage >= cfg.min_drainage_area_km2)
    vals = acc[source_valid]
    qthr = np.quantile(vals, cfg.river_accumulation_quantile) if len(vals) else np.inf
    valid = terrain.land & (drainage >= cfg.min_drainage_area_km2)
    tributary_thr = float(np.clip(cfg.tributary_discharge_fraction, 0.05, 0.95)) * qthr
    candidate = valid & (acc >= tributary_thr)
    stream_order = graph.strahler_order(candidate)
    rivers = candidate & ((stream_order >= 2) | (acc >= 0.52 * qthr))
    if np.any(rivers):
        near = grid.ops.binary_dilation(rivers & (stream_order >= 2), iterations=1)
        rivers |= candidate & near
    stream_order = np.where(rivers, stream_order, 0).astype(np.uint8)
    logq = normalize01(np.log1p(acc)).astype(np.float32)
    order_norm = stream_order.astype(float) / max(float(stream_order.max()), 1.0)
    river_width = (logq * (0.62 + 0.38 * order_norm) * rivers).astype(np.float32)

    lakes = _lake_mask(grid, z0, filled, terrain.land, drainage, acc, climate, cfg)
    ocean_labels, _ = grid.ops.connected_components(terrain.ocean)
    terminal = np.where(terrain.ocean, ocean_labels, 0).astype(np.int32)
    basins = graph.basin_roots(terminal)
    discharge_norm = normalize01(np.log1p(acc)).astype(np.float32)
    lith = _LITH_ERODIBILITY[np.clip(geology.rock_code, 0, len(_LITH_ERODIBILITY) - 1)] if geology is not None else np.ones_like(z0)
    if geology is not None:
        positive = acc[terrain.land & (acc > 0)]
        qref = float(np.quantile(positive, 0.985)) if positive.size else 1.0
        meander = _meander_field(geology, lith, slope, np.clip(acc / max(qref, 1e-9), 0, 2.5), terrain.land, cfg)
    else:
        meander = np.zeros_like(z0)
    meander *= rivers
    sinuosity = (1.0 + 2.35 * meander).astype(np.float32)
    if surface is None:
        zero = np.zeros_like(z0, dtype=np.float32)
        er = dep = sf = dd = up = mm = zero
    else:
        er = surface.cumulative_erosion_m
        dep = surface.cumulative_deposition_m
        sf = surface.sediment_flux_index
        dd = surface.delta_deposition_m
        up = surface.tectonic_uplift_m
        mm = surface.meander_migration_m
    centerlines = _build_river_centerlines(grid, flow, rivers, acc, meander, geology, max_paths=max(20, int(cfg.max_river_centerlines)))
    meta = {
        "river_threshold_discharge_index": float(qthr) if np.isfinite(qthr) else None,
        "river_area_fraction_of_land": grid.weighted_fraction(rivers) / max(grid.weighted_fraction(terrain.land), 1e-12),
        "lake_area_fraction_of_land": grid.weighted_fraction(lakes) / max(grid.weighted_fraction(terrain.land), 1e-12),
        "mean_runoff_mm_year_land": float(np.average(runoff[terrain.land], weights=grid.cell_area_weights[terrain.land])) if np.any(terrain.land) else 0.0,
        "routing": "extended-direction steepest descent with canonical spherical wrap/pole topology",
        "drainage_graph": "single reusable O(N) topological order for discharge, drainage area, basins and Strahler order; Numba-capable kernels",
        "mean_major_river_sinuosity_proxy": float(np.mean(sinuosity[rivers])) if np.any(rivers) else 1.0,
        "delta_area_fraction": grid.weighted_fraction(dd > 0.15),
        "river_centerline_count": len(centerlines),
        "max_strahler_order": int(stream_order.max()) if np.any(rivers) else 0,
        "tributary_discharge_fraction": float(cfg.tributary_discharge_fraction),
    }
    return HydrologyResult(
        filled.astype(np.float32), flow, acc.astype(np.float32), drainage.astype(np.float32), discharge_norm,
        rivers, stream_order, river_width, lakes, basins, runoff.astype(np.float32), er, dep, sf, dd, up, mm,
        meander.astype(np.float32), sinuosity, centerlines, meta,
    )
