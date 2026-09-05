from __future__ import annotations

"""Remove isotropic blob/ring artifacts from global procedural fields.

The original generator used exact great-circle Gaussian splats for hotspot/LIP
provinces and exact geodesic-radius thresholds for several geological proximity
rules. Those are mathematically convenient but leave conspicuous circles in
rendered geology/resources. Climate also treated every water pixel as a full
marine thermal reservoir, allowing tiny isolated lakes to imprint planet-scale
circular continentality halos.

This compatibility layer keeps spherical topology and deterministic RNG behavior
while replacing those shortcuts with anisotropic/lobate volcanic provinces,
heterogeneous geological proximity envelopes, and component-aware marine thermal
influence with wind/topographic anisotropy.
"""

import hashlib
import math

import numpy as np

from .grid import SphereGrid, distance_to as _distance_to, normalize01, smooth_periodic
from .topology import spherical_resize


def _tangent_basis(center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(center, dtype=float)
    c /= max(float(np.linalg.norm(c)), 1e-12)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(c, ref))) > 0.90:
        ref = np.array([0.0, 1.0, 0.0])
    a = np.cross(ref, c)
    a /= max(float(np.linalg.norm(a)), 1e-12)
    b = np.cross(c, a)
    b /= max(float(np.linalg.norm(b)), 1e-12)
    return a, b


def _hash01(values: np.ndarray, salt: float) -> float:
    v = np.asarray(values, dtype=float).ravel()
    coeff = np.array([12.9898, 78.233, 37.719, 19.913, 53.117], dtype=float)
    n = min(len(v), len(coeff))
    phase = float(np.dot(v[:n], coeff[:n])) + float(salt) * 17.171
    return float((math.sin(phase) * 43758.5453123) % 1.0)


def irregular_blob_field(
    grid: SphereGrid,
    centers_xyz: np.ndarray,
    sigmas_deg: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Build anisotropic lobate spherical provinces with non-circular margins.

    Integral support remains centered on the requested hotspot/LIP locations and
    retains approximately the requested angular scale. Shape anisotropy and lobes
    are derived deterministically from each center, so this consumes no extra RNG
    state and does not perturb unrelated seeded stages.
    """
    centers = np.asarray(centers_xyz, dtype=float)
    sigmas = np.asarray(sigmas_deg, dtype=float)
    wts = np.asarray(weights, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers_xyz must have shape (n, 3)")
    if sigmas.shape != (len(centers),) or wts.shape != (len(centers),):
        raise ValueError("sigmas_deg and weights must contain one value per center")

    pts = np.asarray(grid.xyz, dtype=np.float64).reshape(-1, 3)
    out = np.zeros(len(pts), dtype=np.float64)
    chunk = 65536

    for k, raw_center in enumerate(centers):
        c = raw_center / max(float(np.linalg.norm(raw_center)), 1e-12)
        sigma = max(math.radians(float(sigmas[k])), 1e-6)
        base_a, base_b = _tangent_basis(c)

        h0 = _hash01(c, 1.0 + k)
        h1 = _hash01(c, 11.0 + 3.0 * k)
        h2 = _hash01(c, 37.0 + 7.0 * k)
        bearing = 2.0 * math.pi * h0
        axis_a = math.cos(bearing) * base_a + math.sin(bearing) * base_b
        axis_b = np.cross(c, axis_a)
        axis_b /= max(float(np.linalg.norm(axis_b)), 1e-12)

        # Preserve geometric-mean scale while turning a disc into a province.
        anisotropy = 1.35 + 1.05 * h1
        major = sigma * math.sqrt(anisotropy)
        minor = sigma / math.sqrt(anisotropy)
        lobe_sign = -1.0 if h2 < 0.5 else 1.0
        lobe_shift = sigma * (0.48 + 0.50 * h2) * lobe_sign
        cross_shift = sigma * (h0 - 0.5) * 0.30
        phase1 = 2.0 * math.pi * h1
        phase2 = 2.0 * math.pi * h2
        phase3 = 2.0 * math.pi * _hash01(c, 71.0 + k)

        basis = np.column_stack((c, axis_a, axis_b))
        for i in range(0, len(pts), chunk):
            p = pts[i:i + chunk]
            proj = p @ basis
            dotc = np.clip(proj[:, 0], -1.0, 1.0)
            ang = np.arccos(dotc)
            sinang = np.sqrt(np.maximum(1.0 - dotc * dotc, 0.0))
            scale = np.divide(ang, sinang, out=np.ones_like(ang), where=sinang > 1e-8)
            u = proj[:, 1] * scale
            v = proj[:, 2] * scale

            un = u / major
            vn = v / minor
            q = np.hypot(un, vn)
            az = np.arctan2(vn, un)

            # Multi-frequency crenulation avoids perfectly smooth circumferences
            # without introducing pixel-scale speckle.
            rough = (
                1.0
                + 0.16 * np.sin(3.0 * az + phase1)
                + 0.09 * np.sin(5.0 * az + phase2)
                + 0.055 * np.sin(9.0 * az + phase3)
                + 0.035 * np.sin(4.0 * q + 2.0 * az + phase1 - phase3)
            )
            q_primary = q / np.clip(rough, 0.62, 1.42)
            primary = np.exp(-0.5 * q_primary * q_primary)

            # Leading/trailing lobes mimic plume tracks and irregular LIP margins.
            q_lobe = np.hypot(
                (u - lobe_shift) / max(0.82 * major, 1e-9),
                (v - cross_shift) / max(0.72 * minor, 1e-9),
            )
            q_counter = np.hypot(
                (u + 0.42 * lobe_shift) / max(0.68 * major, 1e-9),
                (v + 0.55 * cross_shift) / max(0.62 * minor, 1e-9),
            )
            envelope = (
                primary
                + (0.18 + 0.10 * h0) * np.exp(-0.5 * q_lobe * q_lobe)
                + (0.07 + 0.06 * h2) * np.exp(-0.5 * q_counter * q_counter)
            )
            out[i:i + chunk] += float(wts[k]) * envelope

    return out.reshape(grid.height, grid.width)


def _mask_digest(mask: np.ndarray, extra: float = 0.0) -> int:
    m = np.asarray(mask, dtype=bool)
    packed = np.packbits(m.ravel(order="C"))
    h = hashlib.blake2b(digest_size=8)
    h.update(np.asarray(m.shape, dtype=np.int64).tobytes())
    h.update(packed)
    h.update(np.asarray([float(extra)], dtype=np.float64).tobytes())
    return int.from_bytes(h.digest(), "little", signed=False)


def _coherent_mask_texture(mask: np.ndarray, grid: SphereGrid, scale_km: float) -> np.ndarray:
    """Cheap deterministic low/mid-frequency roughness keyed to a source mask."""
    h, w = mask.shape
    target_px = max(6.0, min(26.0, float(scale_km) / max(grid.dy_km, 1e-6) * 0.65))
    factor = max(3.0, target_px / 1.8)
    ch = max(14, int(round(h / factor)))
    cw = max(28, int(round(w / factor)))
    seed = _mask_digest(mask, scale_km)
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((ch, cw), dtype=np.float32)
    coarse = smooth_periodic(raw, (1.05, 1.55)).astype(np.float32)
    coarse -= float(np.mean(coarse))
    sd = float(np.std(coarse))
    if sd > 1e-8:
        coarse /= sd
    fine = spherical_resize(coarse, (h, w), order=1).astype(np.float32)
    return np.tanh(0.72 * fine).astype(np.float32)


def irregular_near(mask: np.ndarray, grid: SphereGrid, km: float) -> np.ndarray:
    """Geological proximity with heterogeneous continuity instead of exact discs."""
    source = np.asarray(mask, dtype=bool)
    if not np.any(source):
        return np.zeros_like(source)
    radius = max(float(km), 1e-6)
    d = _distance_to(source, grid)
    rough = _coherent_mask_texture(source, grid, radius)
    local_radius = radius * np.clip(1.0 + 0.24 * rough, 0.72, 1.28)
    return d <= local_radius


def major_water_mask(
    grid: SphereGrid,
    water: np.ndarray,
    *,
    min_global_fraction: float = 1.0e-3,
    min_component_fraction_of_water: float = 2.0e-3,
) -> np.ndarray:
    """Keep ocean/large-sea thermal reservoirs and reject tiny lake point sources."""
    w = np.asarray(water, dtype=bool)
    total = grid.weighted_fraction(w)
    if total < float(min_global_fraction) or not np.any(w):
        return np.zeros_like(w)
    labels, n = grid.ops.connected_components(w)
    if n <= 0:
        return np.zeros_like(w)
    areas = np.bincount(
        labels.ravel(),
        weights=np.asarray(grid.cell_area_weights, float).ravel(),
        minlength=n + 1,
    )
    threshold = max(float(min_global_fraction), float(min_component_fraction_of_water) * total)
    good = areas >= threshold
    good[0] = False
    if not np.any(good[1:]):
        largest = 1 + int(np.argmax(areas[1:]))
        good[largest] = True
    return good[labels]


def marine_thermal_distance(
    grid: SphereGrid,
    water: np.ndarray,
    elevation_km: np.ndarray,
    *,
    inland_scale_km: float,
    climate_texture: np.ndarray | None = None,
) -> np.ndarray:
    """Effective marine thermal distance including flow and relief anisotropy."""
    water = np.asarray(water, dtype=bool)
    elev = np.asarray(elevation_km, dtype=float)
    scale_km = max(float(inland_scale_km), 1.0)
    marine = major_water_mask(grid, water)

    if np.any(marine):
        d = _distance_to(marine, grid).astype(np.float64)
    else:
        # Tiny lakes remain available to local volatile/hydrology processes but do
        # not each create a thousands-of-km circular climate halo.
        d = np.full(water.shape, 3.25 * scale_km, dtype=np.float64)
        d[water] = 0.0

    # Approximate annual-mean circulation.  grad(distance) indicates whether the
    # prevailing flow carries marine air inland or offshore.
    lat = np.asarray(grid.lat, dtype=float)
    al = np.abs(lat)
    trade = np.exp(-((al - 15.0) / 14.0) ** 4)
    west = np.exp(-((al - 45.0) / 14.0) ** 4)
    polar = np.exp(-((al - 72.0) / 13.0) ** 4)
    u = -1.0 * trade + 0.78 * west - 0.46 * polar
    v = -0.32 * np.tanh(lat / 11.0) * np.exp(-(al / 34.0) ** 4)
    gy, gx = grid.ops.metric_gradient(d)
    gn = np.hypot(gx, gy)
    windn = np.hypot(u, v)
    flow_into_interior = np.divide(
        u * gx + v * gy,
        np.maximum(gn * windn, 1e-8),
        out=np.zeros_like(d),
        where=(gn > 1e-8) & (windn > 1e-8),
    )
    d *= np.clip(1.0 - 0.19 * flow_into_interior, 0.76, 1.28)

    # Relief and steep topography impede marine thermal penetration and break
    # otherwise perfectly parallel/symmetric distance contours.
    positive = np.maximum(elev, 0.0)
    smoothed = smooth_periodic(positive, (1.25, 2.0))
    egy, egx = grid.ops.metric_gradient(smoothed)
    relief = normalize01(smoothed, robust=True)
    slope = normalize01(np.hypot(egx, egy), robust=True)
    barrier = np.clip(0.58 * relief + 0.42 * slope, 0.0, 1.0)
    d += scale_km * 0.28 * barrier * (~water)

    if climate_texture is not None:
        tex = np.asarray(climate_texture, dtype=float)
        if tex.shape == d.shape and np.all(np.isfinite(tex)):
            d *= np.clip(1.0 + 0.11 * np.tanh(tex), 0.84, 1.18)

    d[marine] = 0.0
    return np.maximum(d, 0.0).astype(np.float32)



__all__ = [
    "irregular_blob_field",
    "irregular_near",
    "major_water_mask",
    "marine_thermal_distance",
]
