from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .config import OceanConfig, TerrainConfig, NoiseConfig
from .grid import SphereGrid, distance_to, normalize01, smooth_periodic
from .tectonics import TectonicResult
from .terrain import TerrainResult
from .noise import hybrid_multifractal, noise_kwargs, OCEAN_BLEND, StaticNoiseFields
from .topology import prepare_spherical_bilinear_sampler, apply_bilinear_sampler


@dataclass(slots=True)
class OceanResult:
    elevation_km: np.ndarray
    depth_m: np.ndarray
    current_u: np.ndarray
    current_v: np.ndarray
    current_speed: np.ndarray
    sst_anomaly_c: np.ndarray
    upwelling: np.ndarray
    current_u_monthly: np.ndarray
    current_v_monthly: np.ndarray
    sst_anomaly_c_monthly: np.ndarray
    heat_transport_index: np.ndarray
    metadata: dict


def _analytic_surface_winds(grid: SphereGrid, months: int = 12) -> tuple[np.ndarray, np.ndarray]:
    lat = grid.lat
    out_u = np.zeros((months, *lat.shape), np.float32)
    out_v = np.zeros_like(out_u)
    for m in range(months):
        itcz = 11.0 * np.sin(2.0 * np.pi * (m + 0.5) / 12.0 - np.pi / 2.0)
        rel = lat - itcz
        a = np.abs(rel)
        trade = np.exp(-((a - 15.0) / 13.0) ** 4)
        west = np.exp(-((a - 45.0) / 13.0) ** 4)
        polar = np.exp(-((a - 72.0) / 13.0) ** 4)
        out_u[m] = -0.95 * trade + 0.72 * west - 0.42 * polar
        out_v[m] = -0.34 * np.tanh(rel / 9.0) * np.exp(-(a / 31.0) ** 4)
    return out_u, out_v


def _normalize_monthly_vectors(u: np.ndarray, v: np.ndarray, ocean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = np.asarray(u, float).copy(); v = np.asarray(v, float).copy()
    for m in range(u.shape[0]):
        sp = np.hypot(u[m], v[m]); vals = sp[ocean]
        ref = float(np.percentile(vals, 95)) if vals.size else 1.0
        u[m] /= max(ref, 1e-9); v[m] /= max(ref, 1e-9)
    return u, v


def _prepare_bilinear_sampler(src_y: np.ndarray, src_x: np.ndarray, shape: tuple[int, int]):
    return prepare_spherical_bilinear_sampler(src_y, src_x, shape)


def _bilinear_sample(a: np.ndarray, sampler) -> np.ndarray:
    return apply_bilinear_sampler(a, sampler)


def _advect_ocean_heat(grid: SphereGrid, base_sst: np.ndarray, u: np.ndarray, v: np.ndarray,
                       ocean: np.ndarray, cfg: OceanConfig) -> np.ndarray:
    h, w = ocean.shape
    yy, xx = np.indices((h, w), dtype=float)
    step_km = 170.0 * max(float(cfg.heat_advection_strength), 0.02)
    src_y = yy - v * (step_km / max(grid.dy_km, 1e-6))
    src_x = xx - u * (step_km / np.maximum(grid.dx_km, 1e-6))
    sampler = _prepare_bilinear_sampler(src_y, src_x, (h, w))
    heat = base_sst.astype(float).copy()
    for _ in range(max(1, int(cfg.heat_transport_iterations))):
        adv = _bilinear_sample(heat, sampler)
        heat = np.where(ocean, 0.76 * heat + 0.24 * adv, base_sst)
        heat = smooth_periodic(heat, max(0.25, float(cfg.heat_diffusion_sigma)))
        heat = np.where(ocean, 0.94 * heat + 0.06 * base_sst, base_sst)
    return (heat - base_sst) * ocean


def build_ocean(grid: SphereGrid, tect: TectonicResult, terrain: TerrainResult, ocfg: OceanConfig,
                tcfg: TerrainConfig, rng: np.random.Generator,
                atmospheric_wind_u: np.ndarray | None = None,
                atmospheric_wind_v: np.ndarray | None = None,
                noise_cfg: NoiseConfig | None = None,
                static_noise: StaticNoiseFields | None = None) -> OceanResult:
    ocean = terrain.ocean; age = tect.crust_age_myr
    thermal_depth = ocfg.young_crust_depth_m + ocfg.subsidence_sqrt_m_per_sqrt_myr * np.sqrt(np.clip(age, 0, 220))
    thermal_depth = np.minimum(thermal_depth, ocfg.max_abyss_depth_m)
    dist_land = distance_to(terrain.land, grid)
    shelf_w = np.where(distance_to(tect.convergent, grid) < 180, tcfg.shelf_width_km_active, tcfg.shelf_width_km_passive)
    shelf_frac = np.clip(dist_land / np.maximum(shelf_w, 1.0), 0, 1)
    shelf_depth = 8.0 + max(tcfg.shelf_depth_m - 8.0, 1.0) * np.power(shelf_frac, 0.72)
    depth = np.where(terrain.shelf, shelf_depth, thermal_depth)
    conv_dist = distance_to(tect.convergent, grid)
    depth += 3000.0 * np.exp(-conv_dist / 72.0) * (0.35 + 0.65 * tect.convergence_strength) * ocean
    depth -= (1400.0 * tect.hotspot_strength + 800.0 * tect.lip_strength) * ocean
    fine = static_noise.ocean_fine if static_noise is not None else hybrid_multifractal(
        ocean.shape, rng, base_scale_px=max(grid.height / 32.0, 2.5),
        **noise_kwargs(noise_cfg, profile=OCEAN_BLEND, octaves=max(5, min(8, getattr(noise_cfg, "octaves", 7)))))
    fracture = (float(ocfg.abyssal_relief_noise_m) * fine + 260.0 * tect.transform_strength + 180.0 * tect.strain_field)
    fracture *= np.exp(-np.clip(age, 0, 220) / 180.0) * ocean
    depth += fracture
    depth = np.where(ocean, np.clip(depth, 0, 11500), 0.0)
    if terrain.metadata.get("surface_evolved", False):
        evolved_depth = np.maximum(-terrain.elevation_km.astype(float) * 1000.0, 0.0)
        preserve = ocean & (evolved_depth > 0.0) & (evolved_depth < depth)
        depth[preserve] = evolved_depth[preserve]
    elev = terrain.elevation_km.astype(float).copy(); elev[ocean] = -depth[ocean] / 1000.0

    coast_dist = distance_to(terrain.land, grid)
    dscale = np.clip(coast_dist / 2500.0, 0, 1); latr = np.deg2rad(grid.lat)
    psi = dscale * (np.sin(2.0 * latr) - 0.28 * np.sin(4.0 * latr))
    psi = smooth_periodic(psi, sigma=(2.0, 3.0)) * ocean
    dpsi_dy, dpsi_dx = grid.ops.metric_gradient(psi)
    cell_scale_km = float(grid.dy_km)
    base_u = ocfg.gyre_strength * dpsi_dy * cell_scale_km
    base_v = -ocfg.gyre_strength * dpsi_dx * cell_scale_km
    base_u += -0.10 * np.exp(-(grid.lat / 11.0) ** 2)
    base_u += 0.05 * np.exp(-(grid.lat / 4.0) ** 2)
    base_u += 0.07 * (np.abs(grid.lat) > 55) * (np.abs(grid.lat) < 72)
    base_u *= ocean; base_v *= ocean
    base_sp = np.hypot(base_u, base_v)
    if np.any(ocean):
        bref = float(np.percentile(base_sp[ocean], 95)); base_u /= max(bref, 1e-9); base_v /= max(bref, 1e-9)

    cgy, cgx = grid.ops.metric_gradient(coast_dist.astype(float)); cgn = np.hypot(cgx, cgy) + 1e-8
    east_from_coast = cgx / cgn
    near_coast = np.exp(-coast_dist / max(float(ocfg.boundary_current_width_km), 50.0)) * ocean
    western = np.clip(east_from_coast, 0.0, 1.0) * near_coast; eastern = np.clip(-east_from_coast, 0.0, 1.0) * near_coast
    hemi = np.sign(np.sin(latr))
    base_v += -hemi * float(ocfg.western_boundary_strength) * western
    base_v += hemi * float(ocfg.eastern_boundary_strength) * eastern

    sm_depth = smooth_periodic(depth.astype(float), (2.0, 2.8)); dzy, dzx = grid.ops.metric_gradient(sm_depth)
    dzn = np.hypot(dzx, dzy) + 1e-8; tx, ty = -dzy / dzn, dzx / dzn
    bsp = np.hypot(base_u, base_v); align = np.sign(base_u * tx + base_v * ty + 1e-9)
    relief = normalize01(dzn, robust=True) * ocean
    mix = float(np.clip(ocfg.bathymetric_steering_strength, 0.0, 0.65)) * relief
    base_u = (1.0 - mix) * base_u + mix * align * tx * np.maximum(bsp, 0.15)
    base_v = (1.0 - mix) * base_v + mix * align * ty * np.maximum(bsp, 0.15)
    base_u *= ocean; base_v *= ocean

    if atmospheric_wind_u is None or atmospheric_wind_v is None:
        awu, awv = _analytic_surface_winds(grid, 12)
    else:
        awu = np.asarray(atmospheric_wind_u, float); awv = np.asarray(atmospheric_wind_v, float)
        if awu.shape[0] != 12 or awu.shape[1:] != ocean.shape:
            raise ValueError("atmospheric wind arrays must have shape [12,height,width]")
    awu, awv = _normalize_monthly_vectors(awu, awv, ocean)
    fsign = np.sign(np.sin(latr)); fsign[np.abs(grid.lat) < 3.0] = 0.0
    cu = np.empty_like(awu, dtype=np.float32); cv = np.empty_like(awv, dtype=np.float32); sst_month = np.empty_like(awu, dtype=np.float32)
    base_sst = np.clip(28.5 - 0.32 * np.abs(grid.lat) - 0.0017 * grid.lat ** 2, -2.0, 30.5)
    smooth_reps = max(1, int(ocfg.current_iterations) // 18)
    for m in range(12):
        wu, wv = awu[m], awv[m]; ek_u = -wv * fsign; ek_v = wu * fsign
        wind_u = 0.55 * wu + ocfg.ekman_strength * ek_u; wind_v = 0.55 * wv + ocfg.ekman_strength * ek_v
        seasonal = float(np.clip(ocfg.seasonal_current_strength, 0.0, 1.0))
        u = ((1.0 - 0.55 * seasonal) * base_u + seasonal * ocfg.wind_coupling * wind_u) * ocean
        v = ((1.0 - 0.55 * seasonal) * base_v + seasonal * ocfg.wind_coupling * wind_v) * ocean
        for _ in range(smooth_reps):
            u = smooth_periodic(u, (1.0, 1.45)) * ocean; v = smooth_periodic(v, (1.0, 1.45)) * ocean
        sp = np.hypot(u, v); ref = float(np.percentile(sp[ocean], 95)) if np.any(ocean) else 1.0
        u /= max(ref, 1e-9); v /= max(ref, 1e-9)
        cu[m] = u.astype(np.float32); cv[m] = v.astype(np.float32)
        sst_month[m] = np.clip(_advect_ocean_heat(grid, base_sst, u, v, ocean, ocfg) * float(ocfg.sst_transport_gain), -7.0, 7.0).astype(np.float32)

    u_ann = cu.mean(0); v_ann = cv.mean(0); speed = np.hypot(u_ann, v_ann); sst_anom = sst_month.mean(0)
    heat_transport = normalize01(np.mean(np.hypot(cu, cv) * (np.abs(sst_month) + 0.35), axis=0)) * ocean
    coast_factor = np.exp(-dist_land / 220.0) * ocean; cold = np.clip(-sst_anom / 5.0, 0, 1)
    divergence = normalize01(np.maximum(0.0, grid.ops.divergence(u_ann.astype(float), v_ann.astype(float))))
    up = coast_factor * (0.55 * cold + 0.45 * divergence)
    up += 0.62 * np.exp(-(grid.lat / 5.0) ** 2) * ocean
    up = normalize01(smooth_periodic(up, (1.1, 1.4))).astype(np.float32)
    meta = {
        "mean_ocean_depth_m": float(np.average(depth[ocean], weights=grid.cell_area_weights[ocean])) if np.any(ocean) else 0.0,
        "max_ocean_depth_m": float(depth.max()),
        "bathymetry_relation": "depth = young_crust_depth + coefficient*sqrt(age), modified by shelves/trenches/plumes",
        "circulation": "12-month wind-coupled gyres + Ekman proxy + western/eastern boundary currents + bathymetric steering + iterative mixed-layer heat advection/diffusion",
        "noise_model": "shared hybrid multi-type multifractal abyssal/fracture relief",
        "january_current_mean_speed": float(np.average(np.hypot(cu[0], cv[0])[ocean], weights=grid.cell_area_weights[ocean])) if np.any(ocean) else 0.0,
        "july_current_mean_speed": float(np.average(np.hypot(cu[6], cv[6])[ocean], weights=grid.cell_area_weights[ocean])) if np.any(ocean) else 0.0,
    }
    return OceanResult(elev.astype(np.float32), depth.astype(np.float32), u_ann.astype(np.float32), v_ann.astype(np.float32),
                       speed.astype(np.float32), sst_anom.astype(np.float32), up, cu, cv, sst_month,
                       heat_transport.astype(np.float32), meta)
