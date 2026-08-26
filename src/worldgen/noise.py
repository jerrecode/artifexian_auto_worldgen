from __future__ import annotations

"""Fast deterministic hybrid multi-octave procedural noise.

The generator deliberately avoids one single noise family.  Every octave mixes several
statistically different fields (smooth/value-like, ridged, billow and oriented wave fields),
then combines octaves with geometrically decreasing amplitude.  A low-frequency periodic
coordinate warp breaks obvious octave alignment.

All fields wrap in longitude and clamp at the polar rows, matching the world generator's
2:1 equirectangular spherical raster convention.
"""

from dataclasses import dataclass
import math
import numpy as np
from scipy import ndimage

from .grid import smooth_periodic


@dataclass(frozen=True, slots=True)
class NoiseBlend:
    value: float = 0.44
    ridge: float = 0.25
    billow: float = 0.16
    wave: float = 0.15

    def normalized(self) -> tuple[float, float, float, float]:
        vals = np.asarray([self.value, self.ridge, self.billow, self.wave], dtype=float)
        vals = np.maximum(vals, 0.0)
        s = float(vals.sum())
        if s <= 1e-12:
            return (1.0, 0.0, 0.0, 0.0)
        return tuple((vals / s).tolist())


TERRAIN_BLEND = NoiseBlend(0.40, 0.31, 0.14, 0.15)
TECTONIC_BLEND = NoiseBlend(0.34, 0.24, 0.10, 0.32)
CLIMATE_BLEND = NoiseBlend(0.54, 0.08, 0.22, 0.16)
OCEAN_BLEND = NoiseBlend(0.38, 0.32, 0.12, 0.18)
HYDRO_BLEND = NoiseBlend(0.42, 0.22, 0.12, 0.24)
GEOLOGY_BLEND = NoiseBlend(0.46, 0.20, 0.20, 0.14)


def _standardize(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    m = float(np.mean(a, dtype=np.float64))
    s = float(np.std(a, dtype=np.float64))
    if not math.isfinite(s) or s < 1e-8:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - m) / s).astype(np.float32, copy=False)


def _wave_field(
    shape: tuple[int, int],
    rng: np.random.Generator,
    sigma_px: float,
    wave_count: int = 5,
) -> np.ndarray:
    """Band-limited oriented wave field.

    Integer longitudinal wavenumbers guarantee seam continuity. Several oblique waves with
    incommensurate frequencies avoid the visual regularity of a single sinusoid.
    """
    h, w = shape
    x = np.linspace(-math.pi, math.pi, w, endpoint=False, dtype=np.float32)[None, :]
    y = np.linspace(math.pi / 2, -math.pi / 2, h, endpoint=True, dtype=np.float32)[:, None]
    # Approximate cycles corresponding to the Gaussian correlation scale.
    cycles = max(1.0, w / max(7.5 * sigma_px, 4.0))
    out = np.zeros(shape, dtype=np.float32)
    for _ in range(max(2, int(wave_count))):
        kx = max(1, int(round(cycles * float(rng.uniform(0.55, 1.55)))))
        ky = float(rng.uniform(-0.85, 0.85) * kx * 0.55)
        phase = float(rng.uniform(0.0, 2 * math.pi))
        amp = float(rng.uniform(0.65, 1.0))
        out += amp * np.sin(kx * x + ky * y + phase).astype(np.float32)
    return _standardize(out)


def _bilinear_warp(a: np.ndarray, dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """Warp a 2-D field with latitude clamp and longitude wrap using vectorized bilinear sampling."""
    h, w = a.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    sy = np.clip(yy + dy, 0.0, h - 1.00001)
    sx = np.mod(xx + dx, w)
    y0 = np.floor(sy).astype(np.int32)
    x0 = np.floor(sx).astype(np.int32)
    # Float32 remainder can round a wrapped coordinate exactly to ``w`` on large rasters.
    # Enforce integer periodicity after flooring so seam samples always remain in [0, w-1].
    x0 %= w
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = (x0 + 1) % w
    fy = sy - y0
    fx = sx - x0
    return (
        a[y0, x0] * (1 - fy) * (1 - fx)
        + a[y0, x1] * (1 - fy) * fx
        + a[y1, x0] * fy * (1 - fx)
        + a[y1, x1] * fy * fx
    ).astype(np.float32)


def hybrid_multifractal(
    shape: tuple[int, int],
    rng: np.random.Generator,
    *,
    octaves: int = 7,
    base_scale_px: float | None = None,
    persistence: float = 0.56,
    lacunarity: float = 2.0,
    blend: NoiseBlend = TERRAIN_BLEND,
    domain_warp_strength: float = 0.32,
    minimum_sigma_px: float = 0.52,
    wave_count: int = 5,
    robust_clip_sigma: float = 3.6,
) -> np.ndarray:
    """Return a zero-mean, unit-variance hybrid multi-octave field.

    Each octave is a weighted mixture of four noise families.  Octave amplitude is
    ``persistence**octave`` and characteristic frequency grows as ``lacunarity**octave``.
    This implements the requested "higher frequency -> lower intensity" behavior explicitly.
    """
    h, w = map(int, shape)
    if h < 2 or w < 2:
        return np.zeros((h, w), dtype=np.float32)
    octaves = max(1, int(octaves))
    persistence = float(np.clip(persistence, 0.05, 0.95))
    lacunarity = float(max(lacunarity, 1.15))
    base = float(base_scale_px if base_scale_px is not None else max(h / 7.5, 3.0))
    bw = blend.normalized()

    acc = np.zeros((h, w), dtype=np.float32)
    weight_sum = 0.0
    for o in range(octaves):
        sigma = max(float(minimum_sigma_px), base / (lacunarity ** o))
        amp = persistence ** o
        # One white field supplies two decorrelated smooth components by periodic translation;
        # this keeps memory traffic low compared with allocating several full random maps per octave.
        raw = rng.standard_normal((h, w), dtype=np.float32)
        value = smooth_periodic(raw, (sigma, sigma * 1.30)).astype(np.float32, copy=False)
        value = _standardize(value)

        shift_y = max(1, int(round(0.73 * sigma)))
        shift_x = max(1, int(round(1.17 * sigma)))
        aux_raw = np.roll(raw, shift=(shift_y, shift_x), axis=(0, 1))
        aux = smooth_periodic(aux_raw, (max(minimum_sigma_px, sigma * 0.76),
                                        max(minimum_sigma_px, sigma * 1.04))).astype(np.float32, copy=False)
        aux = _standardize(aux)
        # Distinct distribution families derived from correlated underlying structure.
        ridge = _standardize(1.0 - np.abs(value))
        billow = _standardize(np.abs(aux))
        wave = _wave_field((h, w), rng, sigma, wave_count=wave_count)
        octave = (bw[0] * value + bw[1] * ridge + bw[2] * billow + bw[3] * wave).astype(np.float32)
        octave = _standardize(octave)
        acc += float(amp) * octave
        weight_sum += float(amp)
        # Let raw be reclaimed before the next octave on high-resolution runs.
        del raw, aux_raw, value, aux, ridge, billow, wave, octave

    acc /= max(weight_sum, 1e-8)

    # Low-frequency domain warp after octave synthesis.  This bends ridges/isocontours without
    # multiplying the expensive per-octave interpolation cost.
    warp_strength = max(0.0, float(domain_warp_strength))
    if warp_strength > 1e-5:
        wsigma = max(2.0, base * 1.35)
        wy = rng.standard_normal((h, w), dtype=np.float32)
        wx = np.roll(wy, shift=(max(1, int(wsigma * 0.37)), max(1, int(wsigma * 0.61))), axis=(0, 1))
        wy = _standardize(smooth_periodic(wy, (wsigma, wsigma * 1.25)))
        wx = _standardize(smooth_periodic(wx, (wsigma * 0.86, wsigma * 1.12)))
        # Warp amount is relative to the largest structural wavelength, capped for topology safety.
        warp_px = min(base * warp_strength, 0.09 * min(h, w))
        # East-west degrees collapse near poles; suppress x warp there.
        lat = np.linspace(90 - 90 / h, -90 + 90 / h, h, dtype=np.float32)[:, None]
        taper = np.clip(np.cos(np.deg2rad(lat)), 0.18, 1.0)
        acc = _bilinear_warp(acc, wy * warp_px, wx * warp_px * taper)

    acc = _standardize(acc)
    if robust_clip_sigma > 0:
        acc = np.clip(acc, -robust_clip_sigma, robust_clip_sigma)
        acc = _standardize(acc)
    return acc.astype(np.float32, copy=False)


def hybrid_noise01(*args, **kwargs) -> np.ndarray:
    """Convenience 0..1 version using a fixed robust logistic-like remap."""
    z = hybrid_multifractal(*args, **kwargs)
    # Tanh avoids percentile sorting of million-cell arrays and is stable across resolutions.
    return (0.5 + 0.5 * np.tanh(0.72 * z)).astype(np.float32)


def configured_blend(cfg, profile: NoiseBlend = TERRAIN_BLEND) -> NoiseBlend:
    """Combine global user weights with a stage profile and renormalize lazily in NoiseBlend."""
    if cfg is None:
        return profile
    return NoiseBlend(
        float(getattr(cfg, "value_weight", 0.44)) * profile.value,
        float(getattr(cfg, "ridge_weight", 0.25)) * profile.ridge,
        float(getattr(cfg, "billow_weight", 0.16)) * profile.billow,
        float(getattr(cfg, "wave_weight", 0.15)) * profile.wave,
    )


def noise_kwargs(cfg, *, profile: NoiseBlend = TERRAIN_BLEND, octaves: int | None = None) -> dict:
    if cfg is None:
        return {"blend": profile, **({} if octaves is None else {"octaves": int(octaves)})}
    return {
        "octaves": int(getattr(cfg, "octaves", 7) if octaves is None else octaves),
        "persistence": float(getattr(cfg, "persistence", 0.56)),
        "lacunarity": float(getattr(cfg, "lacunarity", 2.0)),
        "blend": configured_blend(cfg, profile),
        "domain_warp_strength": float(getattr(cfg, "domain_warp_strength", 0.32)),
        "minimum_sigma_px": float(getattr(cfg, "minimum_sigma_px", 0.52)),
        "wave_count": int(getattr(cfg, "wave_count", 5)),
    }

@dataclass(slots=True)
class StaticNoiseFields:
    ocean_fine: np.ndarray
    climate_texture: np.ndarray
    convective_texture: np.ndarray
    geology_lith: np.ndarray
    geology_igneous: np.ndarray
    hydro_wiggle: np.ndarray
    delta_texture: np.ndarray


def build_static_noise_fields(shape: tuple[int,int], cfg, rng_factory) -> StaticNoiseFields:
    """Precompute stationary stochastic fields reused across coupled Earth-system passes.

    Reusing these fields is both physically sensible (sub-grid geology does not reroll every model
    iteration) and materially faster at high resolution.
    """
    h,w=shape
    octv=int(getattr(cfg,"octaves",7))
    ocean_fine=hybrid_multifractal(shape,rng_factory("noise:ocean"),base_scale_px=max(h/32.0,2.5),
        **noise_kwargs(cfg,profile=OCEAN_BLEND,octaves=max(5,min(8,octv))))
    climate_texture=hybrid_multifractal(shape,rng_factory("noise:climate"),base_scale_px=max(h/16.0,4.0),
        **noise_kwargs(cfg,profile=CLIMATE_BLEND,octaves=max(4,min(7,octv))))
    convective_texture=hybrid_noise01(shape,rng_factory("noise:convection"),base_scale_px=max(h/20.0,3.0),
        **noise_kwargs(cfg,profile=NoiseBlend(0.48,0.07,0.27,0.18),octaves=max(4,min(6,octv))))
    geology_lith=hybrid_noise01(shape,rng_factory("noise:geology-lith"),base_scale_px=max(h/22.0,3.0),
        **noise_kwargs(cfg,profile=GEOLOGY_BLEND,octaves=max(5,min(8,octv))))
    geology_igneous=hybrid_noise01(shape,rng_factory("noise:geology-igneous"),base_scale_px=max(h/30.0,2.5),
        **noise_kwargs(cfg,profile=NoiseBlend(0.34,0.28,0.12,0.26),octaves=max(4,min(7,octv))))
    hydro_wiggle=hybrid_multifractal(shape,rng_factory("noise:hydro"),base_scale_px=max(h/36.0,2.5),
        **noise_kwargs(cfg,profile=HYDRO_BLEND,octaves=max(5,min(8,octv))))
    delta_texture=hybrid_noise01(shape,rng_factory("noise:delta"),base_scale_px=max(h/50.0,2.0),
        **noise_kwargs(cfg,profile=NoiseBlend(0.36,0.25,0.12,0.27),octaves=max(4,min(7,octv))))
    return StaticNoiseFields(ocean_fine,climate_texture,convective_texture,geology_lith,geology_igneous,hydro_wiggle,delta_texture)
