from __future__ import annotations

"""Post-processing corrections for physically plausible visible-ocean appearance."""

import numpy as np

from .appearance import _composite_clouds


def attenuate_deep_bathymetry(appearance, terrain, ocean, weather, cfg):
    """Remove impossible deep-seafloor hillshade leakage from true-color products.

    The canonical appearance model intentionally uses bathymetric relief for visual
    readability, but kilometres-deep seafloor cannot contribute visible reflectance at
    the sea surface.  Here water RGB is reconstructed from depth/turbidity/sea ice and
    only a rapidly attenuated shallow-bottom modulation is retained.  Land pixels are
    untouched.
    """
    water = np.asarray(terrain.ocean, dtype=bool)
    if not np.any(water):
        return appearance
    depth = np.maximum(np.asarray(ocean.depth_m, dtype=float), 0.0)
    shallow = np.exp(-depth / 850.0)
    deep = np.clip(depth / 8000.0, 0.0, 1.0)
    rgb = np.zeros((*water.shape, 3), dtype=np.float32)
    rgb[..., 0] = 0.025 + 0.055 * shallow
    rgb[..., 1] = 0.15 + 0.43 * shallow
    rgb[..., 2] = 0.31 + 0.58 * shallow - 0.08 * deep

    turbidity = np.clip(np.asarray(appearance.water_turbidity, float) * float(cfg.turbidity_strength), 0.0, 0.85)
    turbid_rgb = np.asarray([0.30, 0.49, 0.39], np.float32)
    tt = turbidity[..., None]
    rgb = rgb * (1.0 - tt) + turbid_rgb * tt

    coral = np.asarray(getattr(weather, "coral_reef", np.zeros_like(water)), bool) & water
    if np.any(coral):
        rgb[coral] = 0.76 * rgb[coral] + 0.24 * np.asarray([0.08, 0.63, 0.60], np.float32)
    ice = np.asarray(getattr(weather, "sea_ice_max", np.zeros_like(water)), bool) & water
    if np.any(ice):
        rgb[ice] = 0.20 * rgb[ice] + 0.80 * np.asarray([0.88, 0.94, 0.97], np.float32)

    # Optional bottom visibility falls exponentially and is negligible beyond the
    # photic/shallow-water scale.  It changes brightness only, never imprints a deep
    # ridge/trench network through kilometres of water.
    visibility_scale_m = max(float(getattr(cfg, "ocean_bottom_visibility_scale_m", 55.0)), 5.0)
    bottom_visibility = np.exp(-depth / visibility_scale_m)
    shallow_brightness = 1.0 + 0.05 * bottom_visibility
    rgb *= shallow_brightness[..., None]
    rgb = np.clip(rgb, 0.0, 1.0)

    for name in ("true_color_rgb", "true_color_january_rgb", "true_color_july_rgb"):
        arr = np.asarray(getattr(appearance, name), dtype=np.float32).copy()
        arr[water] = rgb[water]
        setattr(appearance, name, arr)

    appearance.true_color_with_clouds_rgb = _composite_clouds(
        appearance.true_color_rgb, appearance.cloud_fraction_annual, cfg
    )
    appearance.true_color_january_with_clouds_rgb = _composite_clouds(
        appearance.true_color_january_rgb, appearance.cloud_fraction_monthly[0], cfg
    )
    appearance.true_color_july_with_clouds_rgb = _composite_clouds(
        appearance.true_color_july_rgb, appearance.cloud_fraction_monthly[6], cfg
    )
    appearance.metadata = {
        **appearance.metadata,
        "ocean_optical_bottom_attenuation": "exponential shallow-bottom visibility; deep bathymetric hillshade suppressed",
        "ocean_bottom_visibility_scale_m": visibility_scale_m,
    }
    return appearance


__all__ = ["attenuate_deep_bathymetry"]
