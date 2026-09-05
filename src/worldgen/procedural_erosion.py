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



def _validated_field(name: str, values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64)
    if out.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {out.shape}")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must contain only finite values")
    return out


def _pow_inv(values: np.ndarray, power: np.ndarray | float) -> np.ndarray:
    """Complement-power-complement mapping used by stacked erosion masks."""
    x = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    p = np.maximum(np.asarray(power, dtype=np.float64), 0.0)
    return 1.0 - np.power(1.0 - x, p)


def _ease_out(values: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    inv = 1.0 - x
    return 1.0 - inv * inv


def _smooth_start(values: np.ndarray, smoothing: np.ndarray) -> np.ndarray:
    """Quadratic onset transitioning to a linear ramp without a derivative jump."""
    t = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    s = np.maximum(np.asarray(smoothing, dtype=np.float64), 0.0)
    safe = np.maximum(s, 1.0e-12)
    curved = 0.5 * t * t / safe
    linear = t - 0.5 * s
    smoothed = np.where(t >= s, linear, curved)
    return np.where(s <= 1.0e-12, t, smoothed)


def _slope_mask(slope_ratio: np.ndarray, onset: float, rounding: np.ndarray) -> np.ndarray:
    o = max(float(onset), 0.0)
    scaled = np.maximum(np.asarray(slope_ratio, dtype=np.float64), 0.0) * o
    smoothing = np.clip(np.asarray(rounding, dtype=np.float64), 0.0, 1.0) * o
    return _ease_out(_smooth_start(scaled, smoothing))


def _unit_positions(grid) -> np.ndarray:
    lat = np.deg2rad(np.asarray(grid.lat, dtype=np.float64))
    lon = np.deg2rad(np.asarray(grid.lon, dtype=np.float64))
    clat = np.cos(lat)
    return np.stack(
        (clat * np.cos(lon), clat * np.sin(lon), np.sin(lat)),
        axis=-1,
    )


def _tangent_bases(grid) -> tuple[np.ndarray, np.ndarray]:
    """Return immutable south/east tangent bases for the spherical raster."""
    lat = np.deg2rad(np.asarray(grid.lat, dtype=np.float64))
    lon = np.deg2rad(np.asarray(grid.lon, dtype=np.float64))
    east = np.stack((-np.sin(lon), np.cos(lon), np.zeros_like(lon)), axis=-1)
    north = np.stack(
        (-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)),
        axis=-1,
    )
    return -north, east


def _tangent_perpendicular_precomputed(
    normal: np.ndarray,
    south_basis: np.ndarray,
    east_basis: np.ndarray,
    south: np.ndarray,
    east_component: np.ndarray,
) -> np.ndarray:
    """Build the phase-line perpendicular from immutable spherical geometry."""
    direction = (
        south[..., None] * south_basis
        + east_component[..., None] * east_basis
    )
    dn = np.linalg.norm(direction, axis=-1, keepdims=True)
    direction = np.divide(direction, np.maximum(dn, 1.0e-15))
    perp = np.cross(normal, direction)
    pn = np.linalg.norm(perp, axis=-1, keepdims=True)
    return np.divide(perp, np.maximum(pn, 1.0e-15))


def _tangent_perpendicular(grid, south: np.ndarray, east_component: np.ndarray) -> np.ndarray:
    """Reference convenience path that derives spherical geometry on demand."""
    normal = _unit_positions(grid)
    south_basis, east_basis = _tangent_bases(grid)
    return _tangent_perpendicular_precomputed(
        normal, south_basis, east_basis, south, east_component
    )


def phase_cell_octave_xyz(
    unit_xyz: np.ndarray,
    radius_km: float,
    wavelength_km: np.ndarray,
    perpendicular_xyz: np.ndarray,
    *,
    cell_scale: float,
    seed: int,
    octave: int,
    normalization: float = 1.0,
    chunk_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate one seamless 3-D phase-cell octave on arbitrary planet positions.

    normalization is in [0, 1]. One preserves the former full phase-vector
    normalization; lower values retain amplitude information in low-coherence
    regions. The high-level advanced filter uses 0.5 by default.
    """
    normal = np.asarray(unit_xyz, dtype=np.float64)
    perp = np.asarray(perpendicular_xyz, dtype=np.float64)
    if normal.shape != perp.shape or normal.ndim < 1 or normal.shape[-1] != 3:
        raise ValueError("unit_xyz and perpendicular_xyz must have identical (...,3) shape")
    if not np.isfinite(normal).all() or not np.isfinite(perp).all():
        raise ValueError("unit_xyz and perpendicular_xyz must contain only finite values")
    radius = float(radius_km)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius_km must be finite and positive")
    cell = float(cell_scale)
    if not np.isfinite(cell) or cell <= 0.0:
        raise ValueError("cell_scale must be finite and positive")
    wavelength = np.asarray(wavelength_km, dtype=np.float64)
    if wavelength.shape != normal.shape[:-1]:
        raise ValueError(
            f"wavelength_km must have shape {normal.shape[:-1]}, got {wavelength.shape}"
        )
    if not np.isfinite(wavelength).all() or np.any(wavelength <= 0.0):
        raise ValueError("wavelength_km must contain only finite positive values")
    norm_parameter = float(normalization)
    if not np.isfinite(norm_parameter) or not 0.0 <= norm_parameter <= 1.0:
        raise ValueError("normalization must be finite and in [0,1]")

    rows = 0
    if chunk_rows is not None:
        if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, (int, np.integer)):
            raise TypeError("chunk_rows must be an integer, zero, or None")
        rows = int(chunk_rows)
        if rows < 0:
            raise ValueError("chunk_rows must be non-negative")
    if rows > 0 and normal.ndim >= 2 and normal.shape[0] > rows:
        output_shape = normal.shape[:-1]
        cosine = np.empty(output_shape, dtype=np.float64)
        sine = np.empty(output_shape, dtype=np.float64)
        coherence = np.empty(output_shape, dtype=np.float64)
        for start in range(0, normal.shape[0], rows):
            stop = min(start + rows, normal.shape[0])
            part = slice(start, stop)
            c_part, s_part, q_part = phase_cell_octave_xyz(
                normal[part],
                radius,
                wavelength[part],
                perp[part],
                cell_scale=cell,
                seed=seed,
                octave=octave,
                normalization=norm_parameter,
                chunk_rows=None,
            )
            cosine[part] = c_part
            sine[part] = s_part
            coherence[part] = q_part
        return cosine, sine, coherence

    pn = np.linalg.norm(perp, axis=-1, keepdims=True)
    perp = np.divide(perp, np.maximum(pn, 1.0e-15))
    xyz = normal * radius
    scale = np.maximum(wavelength * cell, 1.0e-6)
    q = xyz / scale[..., None]
    base = np.floor(q).astype(np.int64)

    csum = np.zeros(q.shape[:-1], dtype=np.float64)
    ssum = np.zeros_like(csum)
    wsum = np.zeros_like(csum)
    support_r2 = 4.25
    phase_scale = 2.0 * np.pi * cell
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
                # Keep displacement components scalar-field shaped. Building an
                # (..., 3) anchor and another (..., 3) delta for every one of the
                # 27 neighbours roughly doubles the dominant temporary working set.
                dx = q[..., 0] - (ix.astype(np.float64) + jx)
                dy = q[..., 1] - (iy.astype(np.float64) + jy)
                dz = q[..., 2] - (iz.astype(np.float64) + jz)
                d2 = dx * dx + dy * dy + dz * dz
                w = np.maximum(1.0 - d2 / support_r2, 0.0) ** 3
                phase0 = 2.0 * np.pi * _hash01(ix, iy, iz, seed, salt0 + 4)
                projected = (
                    perp[..., 0] * dx
                    + perp[..., 1] * dy
                    + perp[..., 2] * dz
                )
                phase = phase_scale * projected + phase0
                csum += w * np.cos(phase)
                ssum += w * np.sin(phase)
                wsum += w

    magnitude = np.hypot(csum, ssum)
    denominator = np.maximum(
        magnitude,
        (1.0 - norm_parameter) * np.maximum(wsum, 1.0e-12),
    )
    denominator = np.maximum(denominator, 1.0e-12)
    cosine = csum / denominator
    sine = ssum / denominator
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
    normalization: float,
    chunk_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return phase_cell_octave_xyz(
        _unit_positions(grid),
        float(grid.radius_km),
        wavelength_km,
        _tangent_perpendicular(grid, direction_south, direction_east),
        cell_scale=cell_scale,
        seed=seed,
        octave=octave,
        normalization=normalization,
        chunk_rows=chunk_rows,
    )

def _finalize_displacement(
    delta: np.ndarray,
    active: np.ndarray,
    weights: np.ndarray,
    *,
    zero_mean: bool,
    limit_m: float,
) -> tuple[np.ndarray, dict[str, float | str]]:
    """Enforce displacement constraints without corrupting the mean invariant.

    With zero-mean morphology enabled, remove the spherical area-weighted mean
    first and enforce the absolute cap using one uniform scale factor.  Uniform
    rescaling preserves the relative morphology and keeps an exactly-zero input
    exactly zero.  Cellwise clipping remains available when zero-mean behavior is
    explicitly disabled.
    """
    out = np.asarray(delta, dtype=np.float64).copy()
    mask = np.asarray(active, dtype=bool)
    area = np.asarray(weights, dtype=np.float64)
    if out.shape != mask.shape or out.shape != area.shape:
        raise ValueError(
            "delta, active mask and cell-area weights must have identical shape"
        )
    if not np.isfinite(out).all():
        raise ValueError(
            "procedural displacement contains non-finite values before constraints"
        )
    if not np.isfinite(area).all() or np.any(area < 0.0):
        raise ValueError("cell-area weights must be finite and non-negative")
    limit = float(limit_m)
    if not np.isfinite(limit) or limit < 0.0:
        raise ValueError("displacement limit must be finite and non-negative")

    out[~mask] = 0.0
    prelimit_max = float(np.max(np.abs(out[mask]))) if np.any(mask) else 0.0
    scale_factor = 1.0
    limiter = "cellwise_clip"

    if np.any(mask) and bool(zero_mean):
        active_weights = area[mask]
        weight_sum = float(np.sum(active_weights))
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            raise ValueError(
                "active cell-area weights must have a positive finite sum"
            )
        weighted_mean = float(
            np.sum(out[mask] * active_weights) / weight_sum
        )
        if weighted_mean != 0.0:
            out[mask] -= weighted_mean

        centered_max = float(np.max(np.abs(out[mask])))
        if centered_max > limit and centered_max > 0.0:
            scale_factor = float(limit / centered_max)
            out[mask] *= scale_factor
            limiter = "uniform_rescale"
        else:
            limiter = "none"
    else:
        np.clip(out, -limit, limit, out=out)

    out[~mask] = 0.0
    post_mean = 0.0
    if np.any(mask):
        active_weights = area[mask]
        post_mean = float(
            np.sum(out[mask] * active_weights) / np.sum(active_weights)
        )

    return out, {
        "displacement_limiter": limiter,
        "displacement_scale_factor": scale_factor,
        "preconstraint_max_absolute_displacement_m": prelimit_max,
        "area_weighted_mean_displacement_m": post_mean,
    }


def apply_procedural_erosion(
    grid,
    terrain,
    forcing: ErosionForcing,
    cfg,
    *,
    seed: int,
) -> ProceduralErosionResult:
    """Apply the environment-conditioned spherical advanced phase-cell filter."""
    elevation = np.asarray(terrain.elevation_km, dtype=np.float64)
    shape = elevation.shape
    if shape != tuple(grid.shape):
        raise ValueError(f"terrain elevation must have grid shape {grid.shape}, got {shape}")
    if not np.isfinite(elevation).all():
        raise ValueError("terrain elevation must contain only finite values")

    strength = np.clip(_validated_field("forcing.strength", forcing.strength, shape), 0.0, None)
    preferred_scale = _validated_field(
        "forcing.preferred_scale_km", forcing.preferred_scale_km, shape
    )
    if np.any(preferred_scale <= 0.0):
        raise ValueError("forcing.preferred_scale_km must be positive")
    environmental_detail = np.clip(
        _validated_field("forcing.detail", forcing.detail, shape), 0.0, 1.0
    )
    target = np.clip(
        _validated_field("forcing.ridge_valley_target", forcing.ridge_valley_target, shape),
        -1.0,
        1.0,
    )
    fallback_s = _validated_field(
        "forcing.orientation_south", forcing.orientation_south, shape
    )
    fallback_e = _validated_field(
        "forcing.orientation_east", forcing.orientation_east, shape
    )
    fallback_norm = np.hypot(fallback_s, fallback_e)
    fallback_s = np.divide(
        fallback_s, np.maximum(fallback_norm, 1.0e-12), out=np.zeros_like(fallback_s)
    )
    fallback_e = np.divide(
        fallback_e, np.maximum(fallback_norm, 1.0e-12), out=np.ones_like(fallback_e)
    )
    ridge_round = np.clip(
        _validated_field("forcing.ridge_rounding", forcing.ridge_rounding, shape), 0.0, 1.0
    )
    crease_round = np.clip(
        _validated_field("forcing.crease_rounding", forcing.crease_rounding, shape), 0.0, 1.0
    )

    delta = np.zeros(shape, dtype=np.float64)
    coherence_max = np.zeros(shape, dtype=np.float64)
    ridge_map = np.zeros(shape, dtype=np.float64)
    crease_map = np.zeros(shape, dtype=np.float64)

    # Planet geometry is invariant across erosion octaves. The previous reference
    # path rebuilt these trigonometric fields inside every phase-cell octave.
    unit_xyz = _unit_positions(grid)
    south_basis, east_basis = _tangent_bases(grid)

    slope_s, slope_e = grid.ops.metric_gradient(elevation)
    slope_length = np.hypot(slope_s, slope_e)
    slope_reference = max(float(getattr(cfg, "slope_reference", 0.08)), 1.0e-12)
    slope_ratio = slope_length / slope_reference

    actual_s = slope_s / slope_reference
    actual_e = slope_e / slope_reference
    actual_norm = np.hypot(actual_s, actual_e)
    unit_s = np.divide(
        actual_s,
        np.maximum(actual_norm, 1.0e-12),
        out=-fallback_s.copy(),
        where=actual_norm > 1.0e-12,
    )
    unit_e = np.divide(
        actual_e,
        np.maximum(actual_norm, 1.0e-12),
        out=-fallback_e.copy(),
        where=actual_norm > 1.0e-12,
    )
    assumed_slope = max(float(getattr(cfg, "assumed_slope", 0.70)), 0.0)
    assumed_blend = float(np.clip(getattr(cfg, "assumed_slope_blend", 1.0), 0.0, 1.0))
    gully_s = (1.0 - assumed_blend) * actual_s + assumed_blend * assumed_slope * unit_s
    gully_e = (1.0 - assumed_blend) * actual_e + assumed_blend * assumed_slope * unit_e

    initial_rounding = np.where(target >= 0.0, ridge_round, crease_round)
    initial_rounding *= max(float(getattr(cfg, "rounding_initial_multiplier", 0.10)), 0.0)
    combi_mask = _slope_mask(
        slope_ratio,
        float(getattr(cfg, "initial_onset", 1.25)),
        initial_rounding,
    )
    ridge_map_combi_mask = _ease_out(
        slope_ratio * max(float(getattr(cfg, "ridge_map_initial_onset", 2.8)), 0.0)
    )
    fade_target = target.copy()
    ridge_map_fade_target = target.copy()
    rounding_multiplier = 1.0

    octaves = int(cfg.octaves)
    cos_lat = np.maximum(
        np.cos(np.deg2rad(np.asarray(grid.lat, dtype=np.float64))), 1.0e-6
    )
    dx_km = float(grid.radius_km) * float(grid.dlon_rad) * cos_lat
    sample_km = np.maximum(float(grid.dy_km), dx_km)
    executed_octaves = 0

    phase_normalization = float(getattr(cfg, "phase_normalization", 0.5))
    gully_weight = float(getattr(cfg, "gully_weight", 0.5))
    if not 0.0 < gully_weight <= 1.0:
        raise ValueError("cfg.gully_weight must be in (0,1]")
    detail_power = max(float(getattr(cfg, "detail_power", 1.5)), 0.0)
    fade_target_strength = float(cfg.fade_target_strength)
    steering_strength = float(cfg.steering_strength)
    rounding_octave_multiplier = max(
        float(getattr(cfg, "rounding_octave_multiplier", 2.0)), 0.0
    )

    for octave in range(octaves):
        wavelength = preferred_scale / (float(cfg.lacunarity) ** octave)
        resolved = wavelength >= float(cfg.min_samples_per_wavelength) * sample_km
        if not np.any(resolved & (strength > 1.0e-6)):
            break

        gully_norm = np.hypot(gully_s, gully_e)
        direction_s = np.divide(
            gully_s,
            np.maximum(gully_norm, 1.0e-12),
            out=-fallback_s.copy(),
            where=gully_norm > 1.0e-12,
        )
        direction_e = np.divide(
            gully_e,
            np.maximum(gully_norm, 1.0e-12),
            out=-fallback_e.copy(),
            where=gully_norm > 1.0e-12,
        )

        perpendicular_xyz = _tangent_perpendicular_precomputed(
            unit_xyz,
            south_basis,
            east_basis,
            direction_s,
            direction_e,
        )
        c, s, coherence = phase_cell_octave_xyz(
            unit_xyz,
            float(grid.radius_km),
            wavelength,
            perpendicular_xyz,
            cell_scale=float(cfg.cell_scale),
            seed=int(seed),
            octave=octave,
            normalization=phase_normalization,
            chunk_rows=int(getattr(cfg, "phase_chunk_rows", 128)),
        )
        coherence = coherence * resolved
        c = c * resolved
        s = s * resolved

        profile = c
        faded_profile = (
            combi_mask * (gully_weight * profile)
            + (1.0 - combi_mask) * fade_target_strength * fade_target
        )
        spectral_detail = 0.70 + 0.30 * environmental_detail * (
            octave / max(octaves - 1, 1)
        )
        amplitude = (
            float(cfg.base_amplitude_m)
            * (float(cfg.gain) ** octave)
            * strength
            * spectral_detail
            / gully_weight
        )
        delta += amplitude * faded_profile

        coherence_max = np.maximum(coherence_max, coherence)
        ridge_map = np.maximum(
            ridge_map, np.clip(profile, 0.0, 1.0) * coherence * combi_mask
        )
        crease_map = np.maximum(
            crease_map, np.clip(-profile, 0.0, 1.0) * coherence * combi_mask
        )

        # Straight-gully steering follows the sign of the phase derivative rather
        # than its sinusoidal magnitude; the additive lateral component is the
        # spherical analogue of the advanced filter's recursive internal slope.
        side_s = -direction_e
        side_e = direction_s
        steer = (
            steering_strength
            * np.sign(s)
            * coherence
            * environmental_detail
            * (float(cfg.gain) ** octave)
            * gully_weight
        )
        gully_s += steer * side_s
        gully_e += steer * side_e

        sloping = np.abs(s)
        rounding_for_octave = np.where(profile >= 0.0, ridge_round, crease_round)
        rounding_for_octave *= rounding_multiplier
        new_mask = _slope_mask(
            sloping,
            float(getattr(cfg, "octave_onset", 1.25)),
            rounding_for_octave,
        )
        local_detail_power = detail_power * (0.45 + 0.55 * environmental_detail)
        combi_mask = _pow_inv(combi_mask, local_detail_power) * new_mask

        ridge_map_fade_target = (
            (1.0 - ridge_map_combi_mask) * ridge_map_fade_target
            + ridge_map_combi_mask * profile
        )
        new_ridge_mask = _ease_out(
            sloping * max(float(getattr(cfg, "ridge_map_octave_onset", 1.5)), 0.0)
        )
        ridge_map_combi_mask *= new_ridge_mask
        fade_target = faded_profile
        rounding_multiplier *= rounding_octave_multiplier
        executed_octaves += 1

    ridge_signal = ridge_map_fade_target * (1.0 - ridge_map_combi_mask)
    ridge_map = np.maximum(ridge_map, np.clip(ridge_signal, 0.0, 1.0))
    crease_map = np.maximum(crease_map, np.clip(-ridge_signal, 0.0, 1.0))

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
        "model": (
            "seamless 3-D phase-cell procedural erosion with partial phase "
            "normalization, stacked slope masks and environment forcing"
        ),
        "algorithm_lineage": (
            "spherical NumPy adaptation of the advanced phase-cell/Phacelle "
            "erosion-filter principles described by Runevision"
        ),
        "octaves_requested": octaves,
        "octaves_executed": executed_octaves,
        "min_samples_per_wavelength": float(cfg.min_samples_per_wavelength),
        "lacunarity": float(cfg.lacunarity),
        "gain": float(cfg.gain),
        "cell_scale": float(cfg.cell_scale),
        "phase_normalization": phase_normalization,
        "phase_chunk_rows": int(getattr(cfg, "phase_chunk_rows", 128)),
        "spherical_geometry_precomputed": True,
        "gully_weight": gully_weight,
        "detail_power": detail_power,
        "slope_reference": slope_reference,
        "assumed_slope": assumed_slope,
        "assumed_slope_blend": assumed_blend,
        "base_amplitude_m": float(cfg.base_amplitude_m),
        "max_displacement_m": limit,
        "max_absolute_displacement_m": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "mean_absolute_displacement_m": float(np.mean(np.abs(delta))) if delta.size else 0.0,
        "zero_mean_displacement": bool(cfg.zero_mean_displacement),
        **constraint_meta,
        "mass_semantics": (
            "area-zero-mean geometric displacement when enabled; this is a volume "
            "closure constraint for the procedural layer, not grain-resolved sediment "
            "mass conservation"
        ),
    }
    return ProceduralErosionResult(
        delta_height_m=np.asarray(delta, np.float32),
        phase_coherence=np.asarray(coherence_max, np.float32),
        ridge_map=np.asarray(ridge_map, np.float32),
        crease_map=np.asarray(crease_map, np.float32),
        effective_strength=np.asarray(strength, np.float32),
        effective_scale_km=np.asarray(preferred_scale, np.float32),
        metadata=metadata,
    )


__all__ = ["ProceduralErosionResult", "apply_procedural_erosion", "phase_cell_octave_xyz"]
