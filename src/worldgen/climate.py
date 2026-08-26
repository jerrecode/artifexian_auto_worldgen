from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import ndimage

from .astronomy import AstronomyResult
from .config import ClimateConfig, TerrainConfig, NoiseConfig
from .grid import SphereGrid, distance_to, smooth_periodic, normalize01
from .ocean import OceanResult
from .terrain import TerrainResult
from .noise import hybrid_multifractal, hybrid_noise01, noise_kwargs, CLIMATE_BLEND, NoiseBlend, StaticNoiseFields


@dataclass(slots=True)
class ClimateResult:
    temperature_c: np.ndarray          # [month, y, x]
    precipitation_mm: np.ndarray       # [month, y, x]
    pressure_anomaly: np.ndarray       # [month, y, x]
    wind_u: np.ndarray                 # [month, y, x], normalized pseudo-m/s
    wind_v: np.ndarray
    global_circulation_u: np.ndarray   # explicit trades/westerlies/polar easterlies
    global_circulation_v: np.ndarray
    humidity_proxy: np.ndarray         # [month,y,x], advected atmospheric moisture reservoir
    humidity_transport_u: np.ndarray   # moisture-flux proxy = wind * humidity
    humidity_transport_v: np.ndarray
    annual_temperature_c: np.ndarray
    annual_precipitation_mm: np.ndarray
    koppen: np.ndarray                 # [y, x] strings
    continentality_index_c: np.ndarray # hottest-minus-coldest monthly mean
    continentality_class: np.ndarray
    snow_fraction: np.ndarray
    metadata: dict


def _daily_insolation_factor(lat_deg: np.ndarray, decl_deg: float) -> np.ndarray:
    phi = np.deg2rad(lat_deg)
    delta = np.deg2rad(decl_deg)
    x = -np.tan(phi) * np.tan(delta)
    h0 = np.arccos(np.clip(x, -1, 1))
    h0 = np.where(x <= -1, np.pi, h0)
    h0 = np.where(x >= 1, 0.0, h0)
    q = (h0 * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(h0)) / np.pi
    return np.maximum(q, 0.0)


def _pressure_and_winds(
    grid: SphereGrid, temp: np.ndarray, land: np.ndarray, elevation_km: np.ndarray, cfg: ClimateConfig, itcz_lat_deg: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lat = grid.lat
    # Pressure belts migrate with the thermal equator/ITCZ rather than remaining locked to 0° all year.
    rel = lat - float(itcz_lat_deg)
    al = np.abs(rel)
    p = (
        -7.2 * np.exp(-(al / 10.5) ** 2)
        + 6.0 * np.exp(-((al - 30.0) / 10.0) ** 2)
        - 5.0 * np.exp(-((al - 60.0) / 9.0) ** 2)
        + 4.0 * np.exp(-((al - 86.0) / 9.0) ** 2)
    )
    zonal_t = np.mean(temp, axis=1, keepdims=True)
    thermal_wave = smooth_periodic(temp - zonal_t, (2.4, 5.2))
    p += -cfg.pressure_land_seasonality * np.tanh((temp - zonal_t) / 12.0) * land
    # Longitudinal land/ocean thermal contrasts force stationary planetary waves rather than leaving
    # every pressure belt purely zonal.
    p += -float(cfg.stationary_wave_strength) * thermal_wave
    p = smooth_periodic(p, (1.8, 2.7))

    # Explicit global circulation: tropical easterly trades converging on the seasonal ITCZ,
    # mid-latitude westerlies, and polar easterlies. v follows raster convention: + is southward.
    trade = np.exp(-((al - 15.0) / 13.0) ** 4)
    west = np.exp(-((al - 45.0) / 13.0) ** 4)
    polar = np.exp(-((al - 72.0) / 13.0) ** 4)
    gu = (-cfg.trade_wind_strength * trade
          + cfg.westerly_strength * west
          - cfg.polar_easterly_strength * polar)
    gv = -float(cfg.hadley_meridional_strength) * cfg.trade_wind_strength * np.tanh(rel / 8.0) * np.exp(-(al / 31.0) ** 4)
    # Weak return branches of the Ferrel/polar cells add meridional exchange outside the tropics.
    gv += float(cfg.ferrel_meridional_strength) * np.tanh(rel / 10.0) * west
    gv -= 0.08 * np.tanh(rel / 8.0) * polar

    # Add terrain/thermal pressure anomalies as a weather-scale/geostrophic perturbation.
    gy, gx = np.gradient(p)
    f = np.sin(np.deg2rad(lat))
    turn = np.clip(np.abs(f) * 1.15, 0, 1)
    direct_u, direct_v = -gx, -gy
    geo_u = -gy * np.sign(f + 1e-12)
    geo_v = gx * np.sign(f + 1e-12)
    pu = (1 - turn) * direct_u + turn * geo_u
    pv = (1 - turn) * direct_v + turn * geo_v
    ps = np.hypot(pu, pv)
    pref = np.percentile(ps, 95)
    pu /= max(pref, 1e-8); pv /= max(pref, 1e-8)

    u = gu + 0.46 * pu
    v = gv + 0.46 * pv
    speed = np.hypot(u, v)
    norm = np.percentile(speed, 95)
    u = u / max(norm, 1e-8)
    v = v / max(norm, 1e-8)

    # Partial terrain steering: resolve the uphill component and divert a configurable fraction along
    # the local contour. Air is not forced to avoid mountains completely, so orographic lifting remains.
    topo = smooth_periodic(np.maximum(elevation_km, 0.0), (1.5, 2.2))
    tgy, tgx = np.gradient(topo)
    tgn = np.hypot(tgx, tgy) + 1e-8
    nx, ny = tgx / tgn, tgy / tgn
    uphill = np.maximum(u * nx + v * ny, 0.0)
    relief_factor = normalize01(tgn, robust=True)
    steer = float(np.clip(cfg.topographic_wind_steering, 0.0, 0.65)) * relief_factor
    u -= steer * uphill * nx
    v -= steer * uphill * ny
    # Coriolis-signed along-contour deflection creates stationary lee/windward asymmetry.
    hemi = np.sign(np.sin(np.deg2rad(lat)))
    u += 0.35 * steer * uphill * (-ny) * hemi
    v += 0.35 * steer * uphill * nx * hemi
    return p, u, v, gu, gv




def _prepare_bilinear_sampler(src_y: np.ndarray, src_x: np.ndarray, shape: tuple[int, int]):
    """Precompute flat indices/weights for repeated semi-Lagrangian bilinear sampling.

    This replaces scipy.ndimage.map_coordinates in the inner humidity loop. The wind field and
    therefore the backtraced coordinates are fixed during a monthly solve, so recomputing generic
    interpolation bookkeeping for every iteration is unnecessary and can become pathological on
    large rasters. Longitude is periodic; latitude is clamped at the polar row.
    """
    h, w = shape
    sy = np.clip(np.asarray(src_y, dtype=np.float64), 0.0, h - 1.0000001)
    sx = np.mod(np.asarray(src_x, dtype=np.float64), w)
    y0 = np.floor(sy).astype(np.int32)
    x0 = np.floor(sx).astype(np.int32)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = (x0 + 1) % w
    fy = (sy - y0).astype(np.float32)
    fx = (sx - x0).astype(np.float32)
    i00 = (y0 * w + x0).ravel()
    i01 = (y0 * w + x1).ravel()
    i10 = (y1 * w + x0).ravel()
    i11 = (y1 * w + x1).ravel()
    w00 = ((1.0 - fy) * (1.0 - fx)).ravel()
    w01 = ((1.0 - fy) * fx).ravel()
    w10 = (fy * (1.0 - fx)).ravel()
    w11 = (fy * fx).ravel()
    return i00, i01, i10, i11, w00, w01, w10, w11, shape


def _bilinear_sample(a: np.ndarray, sampler) -> np.ndarray:
    i00, i01, i10, i11, w00, w01, w10, w11, shape = sampler
    f = np.asarray(a).ravel()
    out = f[i00] * w00 + f[i01] * w01 + f[i10] * w10 + f[i11] * w11
    return out.reshape(shape)

def _advective_precip(
    grid: SphereGrid,
    temp: np.ndarray,
    p: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    elevation_km: np.ndarray,
    ocean: np.ndarray,
    cfg: ClimateConfig,
    sst_anomaly_c: np.ndarray | None = None,
    convective_texture: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = temp.shape
    yy, xx = np.indices((h, w), dtype=float)
    speed = np.hypot(u, v)
    un = u / np.maximum(speed, 0.15)
    vn = v / np.maximum(speed, 0.15)

    if sst_anomaly_c is None:
        sst_anomaly_c = np.zeros_like(temp)
    ocean_t = temp + np.asarray(sst_anomaly_c, float)
    # Clausius-Clapeyron-inspired exponential temperature sensitivity, clipped for stability.
    humidity_capacity = np.exp(cfg.humidity_temperature_sensitivity * np.clip(ocean_t - 15.0, -35.0, 25.0))
    evap = ocean * np.clip(humidity_capacity, 0.12, 3.2)
    moisture = 0.48 * evap + 0.012
    rain = np.zeros_like(temp, dtype=float)

    lowp = normalize01(-p)
    warm_conv = np.clip((temp - 4.0) / 26.0, 0, 1)

    step_km = max(float(cfg.moisture_step_km), 25.0)
    src_y = np.clip(yy - vn * (step_km / max(grid.dy_km, 1e-6)), 0, h - 1)
    src_x = (xx - un * (step_km / np.maximum(grid.dx_km, 1e-6))) % w
    sampler = _prepare_bilinear_sampler(src_y, src_x, (h, w))
    src_elev = _bilinear_sample(elevation_km.astype(float), sampler)
    # Orographic lift is the actual elevation gain over one transport step, not a globally normalized gradient.
    lift_km = np.maximum(elevation_km - src_elev, 0.0)
    descend_km = np.maximum(src_elev - elevation_km, 0.0)
    lift_response = 1.0 - np.exp(-lift_km / max(float(cfg.orographic_lift_scale_km), 0.03))
    # Strong windward enhancement; the finite moisture reservoir naturally creates a lee-side rain shadow.
    oro_cond = np.clip(0.34 * cfg.orographic_strength * lift_response, 0.0, 0.88)
    conv_cond = 0.010 + 0.055 * lowp * warm_conv
    if convective_texture is not None:
        tex = np.clip(np.asarray(convective_texture, float), 0.0, 1.0)
        conv_cond *= 1.0 + float(cfg.convective_texture_strength) * (2.0 * tex - 1.0)
    cond = np.clip(conv_cond + oro_cond, 0.006, 0.92)

    for _ in range(max(1, int(cfg.moisture_iterations))):
        moisture = _bilinear_sample(moisture, sampler)
        # Continuous ocean evaporation and weak terrestrial moisture recycling.
        moisture += evap * (0.11 / max(cfg.moisture_iterations / 20.0, 1.0))
        drop = moisture * cond
        rain += drop
        moisture -= drop
        # Descending air warms/dries in relative terms; keep absolute humidity but reduce local condensation
        # indirectly by not adding any descent precipitation. A tiny mixing loss prevents endless transport.
        moisture *= (0.998 - 0.004 * np.clip(descend_km, 0.0, 2.0))
    psig = max(0.55, float(cfg.precipitation_mesoscale_sigma_px))
    rain = smooth_periodic(rain, (psig, psig * 1.45))
    return rain, np.maximum(moisture, 0.0)


def _cyclic_thermal_memory(forcing: np.ndarray, land: np.ndarray, cfg: ClimateConfig) -> np.ndarray:
    """First-order land/ocean thermal inertia solved to a periodic annual steady state."""
    forcing = np.asarray(forcing, dtype=np.float32)
    land_tau = max(float(cfg.land_thermal_lag_months), 0.05)
    ocean_tau = max(float(cfg.ocean_thermal_lag_months), 0.10)
    tau = np.where(land, land_tau, ocean_tau).astype(np.float32)
    alpha = (1.0 - np.exp(-1.0 / tau)).astype(np.float32)
    state = np.zeros(forcing.shape[1:], dtype=np.float32)
    for _ in range(max(2, int(cfg.thermal_memory_spinup_years))):
        for m in range(12):
            state += alpha * (forcing[m] - state)
    out = np.empty_like(forcing, dtype=np.float32)
    for m in range(12):
        state += alpha * (forcing[m] - state)
        out[m] = state
    return out


def _orbital_flux_factors(eccentricity: float, periapsis_deg: float) -> np.ndarray:
    """Approximate monthly inverse-square orbital flux factors normalized to annual mean 1."""
    e = float(np.clip(eccentricity, 0.0, 0.35))
    phase = 2.0 * np.pi * (np.arange(12, dtype=float) + 0.5) / 12.0
    peri = np.deg2rad(float(periapsis_deg))
    # First-order true-anomaly correction is sufficient for worldbuilding eccentricities.
    mean_anom = phase - peri
    nu = mean_anom + 2.0 * e * np.sin(mean_anom) + 1.25 * e * e * np.sin(2.0 * mean_anom)
    r_over_a = (1.0 - e * e) / np.maximum(1.0 + e * np.cos(nu), 1e-6)
    f = 1.0 / np.maximum(r_over_a * r_over_a, 1e-6)
    return (f / np.mean(f)).astype(np.float32)


def classify_koppen(temp: np.ndarray, precip: np.ndarray) -> np.ndarray:
    """Vectorized Köppen-Geiger-like classification from 12 monthly T/P arrays.

    Uses the common 0°C coldest-month C/D boundary and the standard aridity threshold.
    This is deliberately more quantitative than the coarse hand-colored transcript workflow.
    """
    if temp.shape[0] != 12 or precip.shape[0] != 12:
        raise ValueError("Köppen classification requires 12 monthly values")
    h, w = temp.shape[1:]
    out = np.full((h, w), "", dtype="<U3")
    t_ann = temp.mean(0)
    p_ann = precip.sum(0)
    t_min = temp.min(0)
    t_max = temp.max(0)
    months_gt10 = (temp > 10).sum(0)

    rows = np.arange(h)[:, None]
    north = rows < h // 2
    summer_n = np.array([3, 4, 5, 6, 7, 8])
    winter_n = np.array([9, 10, 11, 0, 1, 2])
    p_sum_n = precip[summer_n].sum(0)
    p_win_n = precip[winter_n].sum(0)
    p_summer = np.where(north, p_sum_n, p_win_n)
    p_winter = np.where(north, p_win_n, p_sum_n)
    frac_s = p_summer / np.maximum(p_ann, 1e-6)
    arid_threshold = 20 * t_ann + np.where(frac_s >= 0.70, 280.0, np.where(frac_s >= 0.30, 140.0, 0.0))
    arid_threshold = np.maximum(arid_threshold, 0.0)
    desert = p_ann < 0.5 * arid_threshold
    steppe = (~desert) & (p_ann < arid_threshold)
    hot = t_ann >= 18.0
    out[desert & hot] = "BWh"
    out[desert & ~hot] = "BWk"
    out[steppe & hot] = "BSh"
    out[steppe & ~hot] = "BSk"
    bmask = desert | steppe

    # Polar.
    ef = (~bmask) & (t_max < 0.0)
    et = (~bmask) & (t_max >= 0.0) & (t_max < 10.0)
    out[ef] = "EF"
    out[et] = "ET"

    # Tropical.
    A = (~bmask) & (t_min >= 18.0)
    pmin = precip.min(0)
    af = A & (pmin >= 60.0)
    am = A & ~af & (pmin >= 100.0 - p_ann / 25.0)
    out[af] = "Af"
    out[am] = "Am"
    # s/w: dry season lies in summer/winter half.
    pmin_s_n = precip[summer_n].min(0); pmin_w_n = precip[winter_n].min(0)
    pmin_s = np.where(north, pmin_s_n, pmin_w_n)
    pmin_w = np.where(north, pmin_w_n, pmin_s_n)
    a_other = A & ~(af | am)
    out[a_other & (pmin_s < pmin_w)] = "As"
    out[a_other & ~(pmin_s < pmin_w)] = "Aw"

    # Temperate and continental; second letter from precipitation seasonality.
    base = (~bmask) & ~A & ~(ef | et)
    C = base & (t_min > 0.0) & (t_min < 18.0) & (t_max > 10.0)
    D = base & (t_min <= 0.0) & (t_max > 10.0)

    pmax_s_n = precip[summer_n].max(0); pmax_w_n = precip[winter_n].max(0)
    pmax_s = np.where(north, pmax_s_n, pmax_w_n)
    pmax_w = np.where(north, pmax_w_n, pmax_s_n)
    dry_s = (pmin_s < 40.0) & (pmin_s < pmax_w / 3.0)
    dry_w = pmin_w < pmax_s / 10.0
    second = np.where(dry_s, "s", np.where(dry_w, "w", "f"))

    third_c = np.where((t_max >= 22) & (months_gt10 >= 4), "a",
                       np.where(months_gt10 >= 4, "b", "c"))
    third_d = np.where((t_max >= 22) & (months_gt10 >= 4), "a",
                       np.where(months_gt10 >= 4, "b",
                                np.where(t_min <= -38, "d", "c")))
    for sec in ("s", "w", "f"):
        for third in ("a", "b", "c"):
            m = C & (second == sec) & (third_c == third)
            out[m] = "C" + sec + third
        for third in ("a", "b", "c", "d"):
            m = D & (second == sec) & (third_d == third)
            out[m] = "D" + sec + third
    out[out == ""] = "UNK"
    return out


def build_climate(
    grid: SphereGrid,
    astronomy: AstronomyResult,
    terrain: TerrainResult,
    ocean: OceanResult,
    cfg: ClimateConfig,
    tcfg: TerrainConfig,
    rng: np.random.Generator,
    noise_cfg: NoiseConfig | None = None,
    static_noise: StaticNoiseFields | None = None,
) -> ClimateResult:
    if cfg.months != 12:
        raise ValueError("Current climate implementation requires 12 months")
    h, w = terrain.land.shape
    temp = np.empty((12, h, w), dtype=np.float32)
    precip_raw = np.empty_like(temp)
    pressure = np.empty_like(temp)
    wu = np.empty_like(temp)
    wv = np.empty_like(temp)
    gwu = np.empty_like(temp)
    gwv = np.empty_like(temp)
    humidity = np.empty_like(temp)
    hfu = np.empty_like(temp)
    hfv = np.empty_like(temp)

    mean_target = astronomy.planet["mean_surface_temperature_c_approx"]
    tilt = astronomy.planet["axial_tilt_deg"]
    dist_ocean = distance_to(terrain.ocean, grid)
    # Inland thermal inertia decays with distance to ocean. The old 2500-km linear ramp made even
    # supercontinent interiors too oceanic; a nonlinear response produces Earth-like continental ranges.
    continentality = np.clip(dist_ocean / max(cfg.inland_thermal_length_km, 1.0), 0, 1) ** 0.78 * terrain.land
    # Ocean current thermal effects spread several hundred km inland.
    ocean_anom = ocean.sst_anomaly_c.astype(float)
    ocean_anom_fill = ndimage.grey_dilation(ocean_anom, size=(3, 3))
    coastal_current = smooth_periodic(ocean_anom_fill, (5, 8)) * np.exp(-dist_ocean / 650.0)

    # Annual latitudinal gradient calibrated around the configured target mean.
    lat_base = mean_target - 0.48 * np.abs(grid.lat) - 0.0028 * grid.lat ** 2
    # Shift area-weighted planet mean back to target approximately.
    lat_base += mean_target - np.sum(lat_base * grid.cell_area_weights)

    # Insolation includes orbital eccentricity/periapsis phase, then passes through separate land/ocean
    # thermal-memory filters.  This intentionally breaks the old exact solstice mirror symmetry.
    declinations = [tilt * np.sin(2 * np.pi * (m + 0.5) / 12.0 - np.pi / 2) for m in range(12)]
    eccentricity = float(astronomy.planet.get("eccentricity", 0.0))
    periapsis = float(astronomy.planet.get("longitude_periapsis_deg", 103.0))
    orbital_flux = _orbital_flux_factors(eccentricity, periapsis)
    insolations = np.stack([_daily_insolation_factor(grid.lat, d) * orbital_flux[m] for m, d in enumerate(declinations)], axis=0).astype(np.float32)
    insol_annual = insolations.mean(axis=0)
    raw_forcing = (insolations - insol_annual[None, ...]) / np.maximum(insol_annual[None, ...] + 0.25, 0.25)
    seasonal_forcing = _cyclic_thermal_memory(raw_forcing, terrain.land, cfg)

    if static_noise is not None:
        texture = static_noise.climate_texture * float(cfg.climate_texture_c)
        conv_texture = static_noise.convective_texture
    else:
        texture = hybrid_multifractal(
            (h, w), rng, base_scale_px=max(h / 16.0, 4.0),
            **noise_kwargs(noise_cfg, profile=CLIMATE_BLEND, octaves=max(4, min(7, getattr(noise_cfg, "octaves", 7)))),
        ) * float(cfg.climate_texture_c)
        conv_texture = hybrid_noise01(
            (h, w), rng, base_scale_px=max(h / 20.0, 3.0),
            **noise_kwargs(noise_cfg, profile=NoiseBlend(0.48,0.07,0.27,0.18), octaves=max(4, min(6, getattr(noise_cfg, "octaves", 6)))),
        )

    for m in range(12):
        insol_delta = seasonal_forcing[m]
        # Oceans have high thermal inertia; continental interiors respond much more strongly. The
        # latitude term increases seasonality poleward where insolation contrast is intrinsically larger.
        land_response = 10.0 + cfg.continentality_k * continentality + 0.10 * np.abs(grid.lat) * terrain.land
        response = np.where(terrain.ocean, cfg.ocean_seasonal_response_c, land_response)
        tm = lat_base + response * insol_delta
        tm += coastal_current
        tm -= tcfg.lapse_rate_k_per_km * np.maximum(ocean.elevation_km, 0.0)
        # Stationary multi-octave climate texture is generated once per world, not rerolled monthly.
        # Monthly ocean heat transport modifies SST and coastal temperatures.
        if hasattr(ocean, "sst_anomaly_c_monthly"):
            sst_m = np.asarray(ocean.sst_anomaly_c_monthly[m], float)
        else:
            sst_m = np.asarray(ocean.sst_anomaly_c, float)
        tm += sst_m * terrain.ocean
        tm += smooth_periodic(sst_m, (3.0, 5.0)) * np.exp(-dist_ocean / 520.0) * terrain.land
        tm += texture
        itcz_lat = cfg.seasonal_itcz_shift_fraction * declinations[m]
        p, u, v, gu, gv = _pressure_and_winds(grid, tm, terrain.land, ocean.elevation_km, cfg, itcz_lat)
        pr, hm = _advective_precip(grid, tm, p, u, v, ocean.elevation_km, terrain.ocean, cfg, sst_m, conv_texture)
        temp[m] = tm
        pressure[m] = p
        wu[m] = u
        wv[m] = v
        gwu[m] = gu
        gwv[m] = gv
        humidity[m] = hm
        hfu[m] = hm * u
        hfv[m] = hm * v
        precip_raw[m] = pr

    # Scale by spherical land area, not equirectangular pixel count. Extreme orographic cells are
    # compressed smoothly rather than clipped to an artificial 800 mm/month ceiling.
    raw_ann = precip_raw.sum(0)
    lm = terrain.land & (raw_ann > 0)
    if np.any(lm):
        raw_ref = float(np.average(raw_ann[lm], weights=grid.cell_area_weights[lm]))
    else:
        pos = raw_ann > 0
        raw_ref = float(np.average(raw_ann[pos], weights=grid.cell_area_weights[pos])) if np.any(pos) else 1.0
    scale = cfg.precip_scale_mm_year / max(raw_ref, 1e-8)
    scaled = np.maximum(precip_raw * scale, 0.0)
    soft = max(cfg.precipitation_softscale_mm_month, 10.0)
    tail = float(np.clip(cfg.precipitation_tail_exponent, 0.45, 1.0))
    shaped = soft * np.power(np.maximum(scaled, 0.0) / soft, tail)

    # Bound extreme windward precipitation with a smooth asymptote rather than a hard clip. Directly
    # renormalizing after compression can re-inflate the extreme tail, so solve one monotonic scalar
    # alpha such that cap*tanh(alpha*shaped/cap) retains the requested spherical land-area mean.
    cap = max(float(cfg.precipitation_extreme_softcap_mm_month), 50.0)
    weights_land = grid.cell_area_weights[terrain.land] if np.any(terrain.land) else None
    target = float(cfg.precip_scale_mm_year)

    def _mean_for_alpha(alpha: float) -> float:
        trial = cap * np.tanh(alpha * shaped / cap)
        annual = trial.sum(0)
        if np.any(terrain.land):
            return float(np.average(annual[terrain.land], weights=weights_land))
        pos = annual > 0
        return float(np.average(annual[pos], weights=grid.cell_area_weights[pos])) if np.any(pos) else 0.0

    lo, hi = 0.0, 1.0
    while _mean_for_alpha(hi) < target and hi < 64.0:
        hi *= 2.0
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        if _mean_for_alpha(mid) < target:
            lo = mid
        else:
            hi = mid
    precip_alpha = 0.5 * (lo + hi)
    precip = (cap * np.tanh(precip_alpha * shaped / cap)).astype(np.float32)
    annp = precip.sum(0)
    annt = temp.mean(0)
    koppen = classify_koppen(temp, precip)
    cont_index = temp.max(0) - temp.min(0)
    cont_class = np.full((h, w), "hyperoceanic", dtype="<U16")
    cont_class[cont_index >= 11.0] = "oceanic"
    cont_class[cont_index >= 21.0] = "subcontinental"
    cont_class[cont_index >= 28.0] = "continental"
    cont_class[cont_index >= 46.0] = "hypercontinental"
    snow = np.mean((temp <= 0.0) & (precip > 1.0), axis=0).astype(np.float32)

    meta = {
        "global_mean_temperature_c": float(np.sum(annt * grid.cell_area_weights)),
        "land_mean_annual_precip_mm": float(np.average(annp[terrain.land], weights=grid.cell_area_weights[terrain.land])) if np.any(terrain.land) else 0.0,
        "precipitation_scaling": "spherical area-weighted target + power-tail shaping + tanh extreme soft-cap; no hard clipping",
        "precipitation_extreme_softcap_mm_month": float(cap),
        "precipitation_softcap_alpha": float(precip_alpha),
        "classification": "Köppen-Geiger-like, monthly quantitative extension of transcript map workflow",
    }
    meta["circulation"] = "seasonally migrating ITCZ + explicit trade winds/westerlies/polar easterlies + pressure anomalies"
    meta["orographic_precipitation"] = "iterative humidity advection with physical upwind elevation gain and lee-side moisture depletion"
    meta["humidity_transport"] = "monthly semi-Lagrangian moisture reservoir and wind-carried flux proxies are persisted"
    meta["seasonality"] = "eccentric-orbit inverse-square forcing + cyclic land/ocean thermal inertia + stationary planetary waves"
    meta["orbital_flux_factors"] = [float(x) for x in orbital_flux]
    meta["noise_model"] = "shared hybrid multi-type multifractal climate/convective texture"
    return ClimateResult(temp, precip, pressure, wu, wv, gwu, gwv, humidity, hfu, hfv,
                         annt.astype(np.float32), annp.astype(np.float32), koppen,
                         cont_index.astype(np.float32), cont_class, snow, meta)
