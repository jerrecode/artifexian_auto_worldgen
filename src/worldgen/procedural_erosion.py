from __future__ import annotations

"""Deterministic phase-cell procedural erosion morphology.

The implementation is independently written from the published phase-cell erosion
description. It preserves the useful locality/recursive-steering ideas while using
a seamless 3-D planet-centred cell lattice instead of a 2-D texture domain.
"""

from dataclasses import dataclass

import numpy as np

from .erosion_forcing import ErosionForcing


@dataclass(slots=True)
class ProceduralErosionResult:
    delta_height_m: np.ndarray
    phase_coherence: np.ndarray
    ridge_map: np.ndarray
    crease_map: np.ndarray
    effective_strength: np.ndarray
    effective_scale_km: np.ndarray
    metadata: dict


def _hash01(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, seed: int, salt: int) -> np.ndarray:
    x = np.asarray(ix, dtype=np.int64).astype(np.uint64, copy=False)
    y = np.asarray(iy, dtype=np.int64).astype(np.uint64, copy=False)
    z = np.asarray(iz, dtype=np.int64).astype(np.uint64, copy=False)
    with np.errstate(over="ignore"):
        h = (
            x * np.uint64(0x9E3779B185EBCA87)
            ^ y * np.uint64(0xC2B2AE3D27D4EB4F)
            ^ z * np.uint64(0x165667B19E3779F9)
            ^ np.uint64(int(seed) & 0xFFFFFFFFFFFFFFFF)
            ^ np.uint64(int(salt) & 0xFFFFFFFFFFFFFFFF)
        )
        h ^= h >> np.uint64(30)
        h *= np.uint64(0xBF58476D1CE4E5B9)
        h ^= h >> np.uint64(27)
        h *= np.uint64(0x94D049BB133111EB)
        h ^= h >> np.uint64(31)
    mantissa = h >> np.uint64(11)
    return mantissa.astype(np.float64) * (1.0 / 9007199254740992.0)


def _unit_positions(grid) -> np.ndarray:
    lat = np.deg2rad(np.asarray(grid.lat, dtype=np.float64))
    lon = np.deg2rad(np.asarray(grid.lon, dtype=np.float64))
    clat = np.cos(lat)
    return np.stack(
        (clat * np.cos(lon), clat * np.sin(lon), np.sin(lat)),
        axis=-1,
    )


def _tangent_perpendicular(grid, south: np.ndarray, east_component: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(np.asarray(grid.lat, dtype=np.float64))
    lon = np.deg2rad(np.asarray(grid.lon, dtype=np.float64))
    east = np.stack((-np.sin(lon), np.cos(lon), np.zeros_like(lon)), axis=-1)
    north = np.stack(
        (-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)),
        axis=-1,
    )
    south_basis = -north
    direction = south[..., None] * south_basis + east_component[..., None] * east
    dn = np.linalg.norm(direction, axis=-1, keepdims=True)
    direction = np.divide(direction, np.maximum(dn, 1.0e-15))
    normal = _unit_positions(grid)
    perp = np.cross(normal, direction)
    pn = np.linalg.norm(perp, axis=-1, keepdims=True)
    return np.divide(perp, np.maximum(pn, 1.0e-15))


def phase_cell_octave_xyz(
    unit_xyz: np.ndarray,
    radius_km: float,
    wavelength_km: np.ndarray,
    perpendicular_xyz: np.ndarray,
    *,
    cell_scale: float,
    seed: int,
    octave: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate one seamless 3-D phase-cell octave on arbitrary planet positions."""
    normal = np.asarray(unit_xyz, dtype=np.float64)
    perp = np.asarray(perpendicular_xyz, dtype=np.float64)
    if normal.shape != perp.shape or normal.shape[-1] != 3:
        raise ValueError("unit_xyz and perpendicular_xyz must have identical (...,3) shape")
    pn = np.linalg.norm(perp, axis=-1, keepdims=True)
    perp = np.divide(perp, np.maximum(pn, 1.0e-15))
    xyz = normal * float(radius_km)
    scale = np.maximum(
        np.asarray(wavelength_km, dtype=np.float64) * float(cell_scale),
        1.0e-6,
    )
    q = xyz / scale[..., None]
    base = np.floor(q).astype(np.int64)

    csum = np.zeros(q.shape[:-1], dtype=np.float64)
    ssum = np.zeros_like(csum)
    wsum = np.zeros_like(csum)
    support_r2 = 4.25
    phase_scale = 2.0 * np.pi * float(cell_scale)
    salt0 = 0x51ED270B + 0x9E37 * int(octave)

    for oz in (-1, 0, 1):
        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                ix = base[..., 0] + ox
                iy = base[..., 1] + oy
                iz = base[..., 2] + oz
                jx = 0.15 + 0.70 * _hash01(ix, iy, iz, seed, salt0 + 1)
                jy = 0.15 + 0.70 * _hash01(ix, iy, iz, seed, salt0 + 2)
                jz = 0.15 + 0.70 * _hash01(ix, iy, iz, seed, salt0 + 3)
                anchor = np.stack(
                    (
                        ix.astype(np.float64) + jx,
                        iy.astype(np.float64) + jy,
                        iz.astype(np.float64) + jz,
                    ),
                    axis=-1,
                )
                delta = q - anchor
                d2 = np.sum(delta * delta, axis=-1)
                w = np.maximum(1.0 - d2 / support_r2, 0.0) ** 3
                phase0 = 2.0 * np.pi * _hash01(ix, iy, iz, seed, salt0 + 4)
                phase = phase_scale * np.sum(perp * delta, axis=-1) + phase0
                csum += w * np.cos(phase)
                ssum += w * np.sin(phase)
                wsum += w

    magnitude = np.hypot(csum, ssum)
    cosine = np.divide(csum, np.maximum(magnitude, 1.0e-12))
    sine = np.divide(ssum, np.maximum(magnitude, 1.0e-12))
    coherence = np.divide(magnitude, np.maximum(wsum, 1.0e-12))
    return cosine, sine, np.clip(coherence, 0.0, 1.0)


def _phase_cell_octave(
    grid,
    wavelength_km: np.ndarray,
    direction_south: np.ndarray,
    direction_east: np.ndarray,
    *,
    cell_scale: float,
    seed: int,
    octave: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return phase_cell_octave_xyz(
        _unit_positions(grid),
        float(grid.radius_km),
        wavelength_km,
        _tangent_perpendicular(grid, direction_south, direction_east),
        cell_scale=cell_scale,
        seed=seed,
        octave=octave,
    )

def _rounded_profile(c: np.ndarray, ridge_rounding: np.ndarray, crease_rounding: np.ndarray) -> np.ndarray:
    a = np.abs(np.asarray(c, dtype=np.float64))
    rounding = np.where(c >= 0.0, ridge_rounding, crease_rounding)
    exponent = 1.0 + 2.5 * np.clip(rounding, 0.0, 1.0)
    shaped = 1.0 - np.power(np.maximum(1.0 - a, 0.0), exponent)
    return np.sign(c) * shaped



def _finalize_displacement(
    delta: np.ndarray,
    active: np.ndarray,
    weights: np.ndarray,
    *,
    zero_mean: bool,
    limit_m: float,
) -> tuple[np.ndarray, dict[str, float | str]]:
    """Apply the displacement cap without violating the configured mean invariant.

    A cellwise clip after mean removal can introduce a new non-zero weighted mean
    whenever positive and negative extrema saturate asymmetrically. When zero-mean
    displacement is requested we therefore remove the weighted mean first and,
    if needed, apply one uniform scale factor to the whole active field. Uniform
    scaling preserves both the morphology and a zero weighted mean in float64
    arithmetic while enforcing the same absolute cap.
    """
    out = np.asarray(delta, dtype=np.float64).copy()
    mask = np.asarray(active, dtype=bool)
    area = np.asarray(weights, dtype=np.float64)
    if out.shape != mask.shape or out.shape != area.shape:
        raise ValueError("delta, active mask and cell-area weights must have identical shape")
    if not np.isfinite(out).all():
        raise ValueError("procedural displacement contains non-finite values before constraints")
    if not np.isfinite(area).all() or np.any(area < 0.0):
        raise ValueError("cell-area weights must be finite and non-negative")
    if not np.isfinite(limit_m) or limit_m < 0.0:
        raise ValueError("displacement limit must be finite and non-negative")

    out[~mask] = 0.0
    scale_factor = 1.0
    limiter = "cellwise_clip"
    prelimit_max = float(np.max(np.abs(out[mask]))) if np.any(mask) else 0.0

    if np.any(mask) and zero_mean:
        active_weights = area[mask]
        weight_sum = float(np.sum(active_weights))
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            raise ValueError("active cell-area weights must have a positive finite sum")
        weighted_mean = float(np.sum(out[mask] * active_weights) / weight_sum)
        out[mask] -= weighted_mean

        centered_max = float(np.max(np.abs(out[mask])))
        if centered_max > limit_m and centered_max > 0.0:
            scale_factor = float(limit_m / centered_max)
            out[mask] *= scale_factor
            limiter = "uniform_rescale"
        else:
            limiter = "none"
    else:
        np.clip(out, -limit_m, limit_m, out=out)

    out[~mask] = 0.0
    post_mean = 0.0
    if np.any(mask):
        active_weights = area[mask]
        post_mean = float(np.sum(out[mask] * active_weights) / np.sum(active_weights))

    return out, {
        "displacement_limiter": limiter,
        "displacement_scale_factor": scale_factor,
        "preconstraint_max_absolute_displacement_m": prelimit_max,
        "area_weighted_mean_displacement_m": post_mean,
    }


def apply_procedural_erosion(grid, terrain, forcing: ErosionForcing, cfg, *, seed: int) -> ProceduralErosionResult:
    shape = terrain.elevation_km.shape
    direction_s = np.asarray(forcing.orientation_south, dtype=np.float64).copy()
    direction_e = np.asarray(forcing.orientation_east, dtype=np.float64).copy()
    strength = np.asarray(forcing.strength, dtype=np.float64)
    detail = np.asarray(forcing.detail, dtype=np.float64)
    target = np.asarray(forcing.ridge_valley_target, dtype=np.float64)
    ridge_round = np.asarray(forcing.ridge_rounding, dtype=np.float64)
    crease_round = np.asarray(forcing.crease_rounding, dtype=np.float64)

    delta = np.zeros(shape, dtype=np.float64)
    coherence_max = np.zeros(shape, dtype=np.float64)
    ridge_map = np.zeros(shape, dtype=np.float64)
    crease_map = np.zeros(shape, dtype=np.float64)
    mask = np.zeros(shape, dtype=np.float64)

    octaves = int(cfg.octaves)
    cos_lat = np.maximum(np.cos(np.deg2rad(np.asarray(grid.lat, dtype=np.float64))), 1.0e-6)
    dx_km = float(grid.radius_km) * float(grid.dlon_rad) * cos_lat
    sample_km = np.maximum(float(grid.dy_km), dx_km)
    executed_octaves = 0
    for octave in range(octaves):
        wavelength = np.asarray(forcing.preferred_scale_km, dtype=np.float64) / (float(cfg.lacunarity) ** octave)
        resolved = wavelength >= float(cfg.min_samples_per_wavelength) * sample_km
        if not np.any(resolved & (strength > 1.0e-6)):
            break
        c, s, coherence = _phase_cell_octave(
            grid,
            wavelength,
            direction_s,
            direction_e,
            cell_scale=float(cfg.cell_scale),
            seed=int(seed),
            octave=octave,
        )
        coherence = coherence * resolved
        profile = _rounded_profile(c, ridge_round, crease_round)
        mask = 1.0 - (1.0 - mask) * (1.0 - np.clip(coherence * detail, 0.0, 1.0))
        visible = mask * profile + (1.0 - mask) * float(cfg.fade_target_strength) * target

        spectral_detail = 0.70 + 0.30 * detail * (octave / max(octaves - 1, 1))
        amplitude = (
            float(cfg.base_amplitude_m)
            * (float(cfg.gain) ** octave)
            * strength
            * spectral_detail
        )
        delta += amplitude * visible
        coherence_max = np.maximum(coherence_max, coherence)
        ridge_map = np.maximum(ridge_map, np.clip(profile, 0.0, 1.0) * coherence)
        crease_map = np.maximum(crease_map, np.clip(-profile, 0.0, 1.0) * coherence)

        # Recursive line-field steering: orientation is axial, so a signed phase
        # perturbation rotates the local tangent direction without introducing a
        # downhill/uphill sign discontinuity.
        turn = (
            float(cfg.steering_strength)
            * np.sign(s)
            * coherence
            * detail
            * (float(cfg.gain) ** octave)
        )
        ct = np.cos(turn)
        st = np.sin(turn)
        new_s = direction_s * ct - direction_e * st
        new_e = direction_s * st + direction_e * ct
        norm = np.hypot(new_s, new_e)
        direction_s = np.divide(new_s, np.maximum(norm, 1.0e-12))
        direction_e = np.divide(new_e, np.maximum(norm, 1.0e-12))
        executed_octaves += 1

    active = strength > 1.0e-6
    limit = float(cfg.max_displacement_m)
    delta, constraint_meta = _finalize_displacement(
        delta,
        active,
        np.asarray(grid.cell_area_weights, dtype=np.float64),
        zero_mean=bool(cfg.zero_mean_displacement),
        limit_m=limit,
    )

    metadata = {
        "model": "seamless 3-D phase-cell procedural erosion with environment forcing and recursive tangent-line steering",
        "octaves_requested": octaves,
        "octaves_executed": executed_octaves,
        "min_samples_per_wavelength": float(cfg.min_samples_per_wavelength),
        "lacunarity": float(cfg.lacunarity),
        "gain": float(cfg.gain),
        "cell_scale": float(cfg.cell_scale),
        "base_amplitude_m": float(cfg.base_amplitude_m),
        "max_displacement_m": limit,
        "max_absolute_displacement_m": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "mean_absolute_displacement_m": float(np.mean(np.abs(delta))) if delta.size else 0.0,
        "zero_mean_displacement": bool(cfg.zero_mean_displacement),
        **constraint_meta,
    }
    return ProceduralErosionResult(
        delta_height_m=np.asarray(delta, np.float32),
        phase_coherence=np.asarray(coherence_max, np.float32),
        ridge_map=np.asarray(ridge_map, np.float32),
        crease_map=np.asarray(crease_map, np.float32),
        effective_strength=np.asarray(strength, np.float32),
        effective_scale_km=np.asarray(forcing.preferred_scale_km, np.float32),
        metadata=metadata,
    )


__all__ = ["ProceduralErosionResult", "apply_procedural_erosion", "phase_cell_octave_xyz"]
