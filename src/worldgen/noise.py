from __future__ import annotations

"""Fast deterministic hybrid multi-octave procedural noise.

Coarse octaves are synthesized at their natural spatial resolution and expanded
with periodic-longitude bilinear interpolation. This avoids spending full-raster
random generation and Gaussian filtering on structures that contain only a few
tens of independent degrees of freedom. Fine octaves remain native-resolution.
Domain warping is processed in bounded row tiles to cap temporary memory.
"""

from dataclasses import dataclass
import math
import numpy as np

from .grid import smooth_periodic
from .mathops import auto_chunk_shape


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
    h, w = shape
    x = np.linspace(-math.pi, math.pi, w, endpoint=False, dtype=np.float32)[None, :]
    y = np.linspace(math.pi / 2, -math.pi / 2, h, endpoint=True, dtype=np.float32)[:, None]
    cycles = max(1.0, w / max(7.5 * sigma_px, 4.0))
    out = np.zeros(shape, dtype=np.float32)
    for _ in range(max(2, int(wave_count))):
        kx = max(1, int(round(cycles * float(rng.uniform(0.55, 1.55)))))
        ky = float(rng.uniform(-0.85, 0.85) * kx * 0.55)
        phase = float(rng.uniform(0.0, 2 * math.pi))
        amp = float(rng.uniform(0.65, 1.0))
        out += amp * np.sin(kx * x + ky * y + phase).astype(np.float32)
    return _standardize(out)


def _resize_periodic_bilinear(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a 2-D field with longitude wrap and latitude clamp."""
    src = np.asarray(a, dtype=np.float32)
    h, w = map(int, shape)
    sh, sw = src.shape
    if (sh, sw) == (h, w):
        return src.copy()
    # Pixel-center mapping. X is periodic; Y remains clamped at the polar rows.
    yp = (np.arange(h, dtype=np.float32) + 0.5) * (sh / h) - 0.5
    xp = (np.arange(w, dtype=np.float32) + 0.5) * (sw / w) - 0.5
    y0 = np.floor(yp).astype(np.int32)
    x0 = np.floor(xp).astype(np.int32)
    fy = yp - y0
    fx = xp - x0
    y0 = np.clip(y0, 0, sh - 1)
    y1 = np.clip(y0 + 1, 0, sh - 1)
    x0 %= sw
    x1 = (x0 + 1) % sw
    fy = fy[:, None]
    fx = fx[None, :]
    out = (
        src[y0[:, None], x0[None, :]] * (1.0 - fy) * (1.0 - fx)
        + src[y0[:, None], x1[None, :]] * (1.0 - fy) * fx
        + src[y1[:, None], x0[None, :]] * fy * (1.0 - fx)
        + src[y1[:, None], x1[None, :]] * fy * fx
    )
    return out.astype(np.float32, copy=False)


def _native_shape(shape: tuple[int, int], sigma_px: float, *, target_sigma: float = 1.55) -> tuple[int, int]:
    """Choose the smallest useful raster for a Gaussian-correlated octave."""
    h, w = shape
    down = max(1, int(math.floor(max(sigma_px, 1.0) / max(target_sigma, 0.5))))
    # Preserve enough samples for anisotropy/waves and the 2:1 global structure.
    nh = min(h, max(8, int(math.ceil(h / down))))
    nw = min(w, max(16, int(math.ceil(w / down))))
    return nh, nw


def _octave_field(
    shape: tuple[int, int],
    rng: np.random.Generator,
    sigma_px: float,
    minimum_sigma_px: float,
    blend_weights: tuple[float, float, float, float],
    wave_count: int,
) -> np.ndarray:
    h, w = shape
    nh, nw = _native_shape(shape, sigma_px)
    scale_y = h / nh
    scale_x = w / nw
    sigma_y = max(0.42, sigma_px / scale_y)
    sigma_x = max(0.42, sigma_px * 1.30 / scale_x)

    raw = rng.standard_normal((nh, nw), dtype=np.float32)
    value = _standardize(smooth_periodic(raw, (sigma_y, sigma_x)).astype(np.float32, copy=False))
    shift_y = max(1, int(round(0.73 * sigma_y)))
    shift_x = max(1, int(round(1.17 * sigma_x)))
    aux_raw = np.roll(raw, shift=(shift_y, shift_x), axis=(0, 1))
    aux = _standardize(smooth_periodic(
        aux_raw,
        (max(0.42, sigma_y * 0.76), max(0.42, sigma_x * 0.80)),
    ).astype(np.float32, copy=False))
    ridge = _standardize(1.0 - np.abs(value))
    billow = _standardize(np.abs(aux))
    # Wave scale is expressed in the native raster so its target wavelength is preserved.
    wave = _wave_field((nh, nw), rng, max(0.5, sigma_y), wave_count=wave_count)
    bw = blend_weights
    native = _standardize(bw[0] * value + bw[1] * ridge + bw[2] * billow + bw[3] * wave)
    return _standardize(_resize_periodic_bilinear(native, shape))


def _bilinear_warp(a: np.ndarray, dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """Warp a field in bounded row tiles with latitude clamp and longitude wrap."""
    src = np.asarray(a, dtype=np.float32)
    dy = np.asarray(dy, dtype=np.float32)
    dx = np.asarray(dx, dtype=np.float32)
    if src.shape != dy.shape or src.shape != dx.shape:
        raise ValueError("warp source and displacement fields must have equal shapes")
    h, w = src.shape
    out = np.empty_like(src)
    chunk_rows, _ = auto_chunk_shape(src.shape, np.float32, target_mb=24.0, arrays_in_flight=10, minimum_rows=4)
    xx = np.arange(w, dtype=np.float32)[None, :]
    for y0 in range(0, h, chunk_rows):
        y1 = min(h, y0 + chunk_rows)
        yy = np.arange(y0, y1, dtype=np.float32)[:, None]
        sy = np.clip(yy + dy[y0:y1], 0.0, h - 1.00001)
        sx = np.mod(xx + dx[y0:y1], w)
        iy0 = np.floor(sy).astype(np.int32)
        ix0 = np.floor(sx).astype(np.int32) % w
        iy1 = np.minimum(iy0 + 1, h - 1)
        ix1 = (ix0 + 1) % w
        fy = sy - iy0
        fx = sx - ix0
        out[y0:y1] = (
            src[iy0, ix0] * (1 - fy) * (1 - fx)
            + src[iy0, ix1] * (1 - fy) * fx
            + src[iy1, ix0] * fy * (1 - fx)
            + src[iy1, ix1] * fy * fx
        )
    return out


def _coarse_smoothed_random(
    shape: tuple[int, int], rng: np.random.Generator, sigma_px: float, *, anisotropy: float = 1.2
) -> np.ndarray:
    nh, nw = _native_shape(shape, sigma_px, target_sigma=1.8)
    sy = shape[0] / nh
    sx = shape[1] / nw
    raw = rng.standard_normal((nh, nw), dtype=np.float32)
    sm = smooth_periodic(raw, (max(0.5, sigma_px / sy), max(0.5, sigma_px * anisotropy / sx)))
    return _standardize(_resize_periodic_bilinear(_standardize(sm), shape))


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
    """Return a deterministic zero-mean, unit-variance hybrid multifractal field."""
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
        octave = _octave_field((h, w), rng, sigma, minimum_sigma_px, bw, wave_count)
        acc += float(amp) * octave
        weight_sum += float(amp)
    acc /= max(weight_sum, 1e-8)

    warp_strength = max(0.0, float(domain_warp_strength))
    if warp_strength > 1e-5:
        wsigma = max(2.0, base * 1.35)
        wy = _coarse_smoothed_random((h, w), rng, wsigma, anisotropy=1.25)
        wx = _coarse_smoothed_random((h, w), rng, wsigma * 0.86, anisotropy=1.12)
        warp_px = min(base * warp_strength, 0.09 * min(h, w))
        lat = np.linspace(90 - 90 / h, -90 + 90 / h, h, dtype=np.float32)[:, None]
        taper = np.clip(np.cos(np.deg2rad(lat)), 0.18, 1.0)
        acc = _bilinear_warp(acc, wy * warp_px, wx * warp_px * taper)

    acc = _standardize(acc)
    if robust_clip_sigma > 0:
        acc = np.clip(acc, -robust_clip_sigma, robust_clip_sigma)
        acc = _standardize(acc)
    return acc.astype(np.float32, copy=False)


def hybrid_noise01(*args, **kwargs) -> np.ndarray:
    z = hybrid_multifractal(*args, **kwargs)
    return (0.5 + 0.5 * np.tanh(0.72 * z)).astype(np.float32)


def configured_blend(cfg, profile: NoiseBlend = TERRAIN_BLEND) -> NoiseBlend:
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


def build_static_noise_fields(shape: tuple[int, int], cfg, rng_factory) -> StaticNoiseFields:
    h, _ = shape
    octv = int(getattr(cfg, "octaves", 7))
    ocean_fine = hybrid_multifractal(
        shape, rng_factory("noise:ocean"), base_scale_px=max(h / 32.0, 2.5),
        **noise_kwargs(cfg, profile=OCEAN_BLEND, octaves=max(5, min(8, octv))),
    )
    climate_texture = hybrid_multifractal(
        shape, rng_factory("noise:climate"), base_scale_px=max(h / 16.0, 4.0),
        **noise_kwargs(cfg, profile=CLIMATE_BLEND, octaves=max(4, min(7, octv))),
    )
    convective_texture = hybrid_noise01(
        shape, rng_factory("noise:convection"), base_scale_px=max(h / 20.0, 3.0),
        **noise_kwargs(cfg, profile=NoiseBlend(0.48, 0.07, 0.27, 0.18), octaves=max(4, min(6, octv))),
    )
    geology_lith = hybrid_noise01(
        shape, rng_factory("noise:geology-lith"), base_scale_px=max(h / 22.0, 3.0),
        **noise_kwargs(cfg, profile=GEOLOGY_BLEND, octaves=max(5, min(8, octv))),
    )
    geology_igneous = hybrid_noise01(
        shape, rng_factory("noise:geology-igneous"), base_scale_px=max(h / 30.0, 2.5),
        **noise_kwargs(cfg, profile=NoiseBlend(0.34, 0.28, 0.12, 0.26), octaves=max(4, min(7, octv))),
    )
    hydro_wiggle = hybrid_multifractal(
        shape, rng_factory("noise:hydro"), base_scale_px=max(h / 36.0, 2.5),
        **noise_kwargs(cfg, profile=HYDRO_BLEND, octaves=max(5, min(8, octv))),
    )
    delta_texture = hybrid_noise01(
        shape, rng_factory("noise:delta"), base_scale_px=max(h / 50.0, 2.0),
        **noise_kwargs(cfg, profile=NoiseBlend(0.36, 0.25, 0.12, 0.27), octaves=max(4, min(7, octv))),
    )
    return StaticNoiseFields(
        ocean_fine, climate_texture, convective_texture, geology_lith, geology_igneous, hydro_wiggle, delta_texture
    )
