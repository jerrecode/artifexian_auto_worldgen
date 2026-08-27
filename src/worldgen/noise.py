from __future__ import annotations

"""Deterministic hybrid multi-octave procedural noise.

Every octave mixes smooth/value-like, ridged, billow and oriented-wave structure,
then octaves are combined with geometrically decreasing amplitude. Coarse octaves
are synthesized at their natural spatial resolution and resampled upward instead
of allocating full-resolution white-noise rasters for wavelengths spanning tens or
hundreds of pixels. This preserves multi-scale structure while sharply reducing
random-generation/filter memory traffic at high world resolutions.
"""

from dataclasses import dataclass
import math
import numpy as np

from .grid import smooth_periodic
from .mathops import auto_chunk_shape, iter_tiles_2d
from .topology import (
    apply_bilinear_sampler,
    prepare_spherical_bilinear_sampler,
    spherical_resize,
    spherical_shift,
)


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


def _resample_to_shape(a: np.ndarray, shape: tuple[int, int], *, order: int = 3) -> np.ndarray:
    """Pole-aware global resize used by natural-resolution noise octaves.

    Cubic Cartesian interpolation previously clamped latitude at both poles. The
    spherical bilinear sampler avoids that topological discontinuity; ``order`` is
    retained as an internal compatibility argument but all continuous noise fields
    now use the canonical bilinear method.
    """
    a = np.asarray(a, dtype=np.float32)
    return np.asarray(spherical_resize(a, shape, order=1), dtype=np.float32)


def _natural_octave_geometry(shape: tuple[int, int], sigma_px: float) -> tuple[tuple[int, int], float, float]:
    """Choose a cheaper synthesis grid for a target correlation scale.

    The local Gaussian width is kept near 1.25--2 pixels. Fine octaves remain at
    native output resolution while large-wavelength octaves can be tens of times
    smaller in area.
    """
    h, w = shape
    if sigma_px <= 2.25 or min(h, w) < 32:
        return (h, w), float(sigma_px), 1.0
    factor = min(32.0, max(1.0, float(sigma_px) / 1.55))
    ch = max(12, int(math.ceil(h / factor)))
    cw = max(24, int(math.ceil(w / factor)))
    # Maintain approximately the same aspect ratio even for non-2:1 unit tests.
    cw = max(4, int(round(ch * (w / h))))
    factor_y = h / ch
    factor_x = w / cw
    effective = math.sqrt(factor_y * factor_x)
    sigma_local = max(0.65, float(sigma_px) / max(effective, 1.0))
    return (ch, cw), sigma_local, effective


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


def _native_smooth_random(
    shape: tuple[int, int], rng: np.random.Generator, sigma_px: float,
    *, anisotropy: float = 1.30,
) -> np.ndarray:
    synth_shape, sigma_local, _ = _natural_octave_geometry(shape, sigma_px)
    raw = rng.standard_normal(synth_shape, dtype=np.float32)
    smooth = smooth_periodic(raw, (sigma_local, sigma_local * anisotropy)).astype(np.float32, copy=False)
    smooth = _standardize(smooth)
    if synth_shape != shape:
        smooth = _resample_to_shape(smooth, shape, order=3)
    return _standardize(smooth)


def _octave_field(
    shape: tuple[int, int], rng: np.random.Generator, sigma_px: float,
    blend: tuple[float, float, float, float], minimum_sigma_px: float, wave_count: int,
) -> np.ndarray:
    synth_shape, sigma_local, _ = _natural_octave_geometry(shape, sigma_px)
    raw = rng.standard_normal(synth_shape, dtype=np.float32)
    value = smooth_periodic(raw, (sigma_local, sigma_local * 1.30)).astype(np.float32, copy=False)
    value = _standardize(value)

    shift_y = max(1, int(round(0.73 * sigma_local)))
    shift_x = max(1, int(round(1.17 * sigma_local)))
    aux_raw = spherical_shift(raw, shift_y, shift_x)
    aux_sigma = max(minimum_sigma_px, sigma_local * 0.76)
    aux = smooth_periodic(aux_raw, (aux_sigma, max(minimum_sigma_px, sigma_local * 1.04))).astype(np.float32, copy=False)
    aux = _standardize(aux)
    ridge = _standardize(1.0 - np.abs(value))
    billow = _standardize(np.abs(aux))
    wave = _wave_field(synth_shape, rng, sigma_local, wave_count=wave_count)
    octave = _standardize(blend[0] * value + blend[1] * ridge + blend[2] * billow + blend[3] * wave)
    if synth_shape != shape:
        octave = _resample_to_shape(octave, shape, order=3)
    return _standardize(octave)


def _bilinear_warp_tiled(
    a: np.ndarray,
    dy: np.ndarray,
    dx: np.ndarray,
    *,
    target_scratch_mb: float = 32.0,
) -> np.ndarray:
    """Warp with bounded scratch memory rather than full-raster coordinate arrays."""
    a = np.asarray(a, dtype=np.float32)
    dy = np.asarray(dy, dtype=np.float32)
    dx = np.asarray(dx, dtype=np.float32)
    if a.shape != dy.shape or a.shape != dx.shape:
        raise ValueError("warp field shapes must match")
    h, w = a.shape
    out = np.empty_like(a)
    chunk = auto_chunk_shape(a.shape, np.float32, target_mb=target_scratch_mb, arrays_in_flight=14)
    for tile in iter_tiles_2d(a.shape, chunk_shape=chunk):
        ys, xs = tile.core
        y0, y1 = ys.start or 0, ys.stop or h
        x0, x1 = xs.start or 0, xs.stop or w
        yy = np.arange(y0, y1, dtype=np.float32)[:, None]
        xx = np.arange(x0, x1, dtype=np.float32)[None, :]
        sy = yy + dy[ys, xs]
        sx = xx + dx[ys, xs]
        sampler = prepare_spherical_bilinear_sampler(sy, sx, (h, w))
        out[ys, xs] = apply_bilinear_sampler(a, sampler)
    return out


def _bilinear_warp(a: np.ndarray, dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """Compatibility wrapper for the bounded-memory tiled implementation."""
    return _bilinear_warp_tiled(a, dy, dx)


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
    """Return a zero-mean, unit-variance hybrid multifractal field.

    Octave amplitude is ``persistence**o`` while characteristic frequency grows as
    ``lacunarity**o``. Low-frequency octaves are generated on coarse natural grids,
    reducing work without reducing their represented spatial bandwidth.
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
        octave = _octave_field((h, w), rng, sigma, bw, minimum_sigma_px, wave_count)
        acc += float(amp) * octave
        weight_sum += float(amp)
    acc /= max(weight_sum, 1e-8)

    warp_strength = max(0.0, float(domain_warp_strength))
    if warp_strength > 1e-5:
        wsigma = max(2.0, base * 1.35)
        wy = _native_smooth_random((h, w), rng, wsigma, anisotropy=1.25)
        wx = _native_smooth_random((h, w), rng, wsigma * 0.86, anisotropy=1.30)
        warp_px = min(base * warp_strength, 0.09 * min(h, w))
        lat = np.linspace(90 - 90 / h, -90 + 90 / h, h, dtype=np.float32)[:, None]
        taper = np.clip(np.cos(np.deg2rad(lat)), 0.18, 1.0)
        acc = _bilinear_warp_tiled(acc, wy * warp_px, wx * warp_px * taper)

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
    ocean_fine = hybrid_multifractal(shape, rng_factory("noise:ocean"), base_scale_px=max(h / 32.0, 2.5),
        **noise_kwargs(cfg, profile=OCEAN_BLEND, octaves=max(5, min(8, octv))))
    climate_texture = hybrid_multifractal(shape, rng_factory("noise:climate"), base_scale_px=max(h / 16.0, 4.0),
        **noise_kwargs(cfg, profile=CLIMATE_BLEND, octaves=max(4, min(7, octv))))
    convective_texture = hybrid_noise01(shape, rng_factory("noise:convection"), base_scale_px=max(h / 20.0, 3.0),
        **noise_kwargs(cfg, profile=NoiseBlend(0.48, 0.07, 0.27, 0.18), octaves=max(4, min(6, octv))))
    geology_lith = hybrid_noise01(shape, rng_factory("noise:geology-lith"), base_scale_px=max(h / 22.0, 3.0),
        **noise_kwargs(cfg, profile=GEOLOGY_BLEND, octaves=max(5, min(8, octv))))
    geology_igneous = hybrid_noise01(shape, rng_factory("noise:geology-igneous"), base_scale_px=max(h / 30.0, 2.5),
        **noise_kwargs(cfg, profile=NoiseBlend(0.34, 0.28, 0.12, 0.26), octaves=max(4, min(7, octv))))
    hydro_wiggle = hybrid_multifractal(shape, rng_factory("noise:hydro"), base_scale_px=max(h / 36.0, 2.5),
        **noise_kwargs(cfg, profile=HYDRO_BLEND, octaves=max(5, min(8, octv))))
    delta_texture = hybrid_noise01(shape, rng_factory("noise:delta"), base_scale_px=max(h / 50.0, 2.0),
        **noise_kwargs(cfg, profile=NoiseBlend(0.36, 0.25, 0.12, 0.27), octaves=max(4, min(7, octv))))
    return StaticNoiseFields(
        ocean_fine, climate_texture, convective_texture, geology_lith,
        geology_igneous, hydro_wiggle, delta_texture,
    )
