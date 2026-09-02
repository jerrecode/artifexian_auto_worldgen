from __future__ import annotations

"""Additional physical-system maps for advanced hydrology and planetary layers."""

from pathlib import Path
import numpy as np

from .render import _save_field, _save_power_field, _save_rgb


def _categorical(labels: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(labels, dtype=np.int64)
    # Deterministic multiplicative hash makes adjacent integer basin IDs visually
    # distinct without allocating a color table for tens of thousands of catchments.
    hashed = ((x.astype(np.uint64) * np.uint64(2654435761)) & np.uint64(0xFFFFFF)).astype(np.float64)
    hashed /= float(0xFFFFFF)
    out = hashed
    if mask is not None:
        out = np.where(mask, out, np.nan)
    return out


def render_advanced_physical_maps(out: Path, world: dict, *, dpi: int = 120) -> None:
    maps = Path(out) / "maps"
    maps.mkdir(parents=True, exist_ok=True)
    terrain = world["terrain"]
    hydro = world["hydrology"]
    land = np.asarray(terrain.land, bool)
    exotic_hydrology = world.get("condensate_hydrology") is not None

    if hasattr(hydro, "basin_id"):
        _save_field(maps / "40_watershed_outlet_basins.png", _categorical(hydro.basin_id, land),
                    "Terminal Drainage Basins", "turbo", dpi=dpi)
    for i, name in enumerate(("subbasin_level_1", "subbasin_level_2", "subbasin_level_3"), 1):
        if hasattr(hydro, name):
            _save_field(maps / f"4{i}_watershed_subbasins.png", _categorical(getattr(hydro, name), land),
                        f"Nested Watershed Level {i}", "turbo", dpi=dpi)
    if hasattr(hydro, "channel_class"):
        _save_field(maps / "44_channel_classes.png", np.asarray(hydro.channel_class, float),
                    "Resolved Channel Hierarchy (0 hillslope → 5 major river)", "viridis", vmin=0, vmax=5, dpi=dpi)
    if hasattr(hydro, "subgrid_drainage_density_km_per_km2"):
        _save_power_field(maps / "45_subgrid_drainage_density.png", hydro.subgrid_drainage_density_km_per_km2,
                          "Expected Sub-grid Drainage Density (km/km²)", "viridis", gamma=0.75, dpi=dpi)
    if hasattr(hydro, "bankfull_discharge_index"):
        _save_power_field(maps / "46_bankfull_discharge.png", hydro.bankfull_discharge_index,
                          "Bankfull / Channel-forming Discharge Index", "Blues", gamma=0.45, dpi=dpi)
    if hasattr(hydro, "baseflow_mm_year"):
        base_title = "Subsurface Liquid Baseflow (liquid-equivalent mm/year)" if exotic_hydrology else "Groundwater Baseflow (mm/year)"
        recharge_title = "Subsurface Liquid Recharge (liquid-equivalent mm/year)" if exotic_hydrology else "Groundwater Recharge (mm/year)"
        _save_power_field(maps / "47_groundwater_baseflow.png", hydro.baseflow_mm_year,
                          base_title, "Blues", gamma=0.6, dpi=dpi)
        _save_power_field(maps / "48_groundwater_recharge.png", hydro.groundwater_recharge_mm_year,
                          recharge_title, "YlGnBu", gamma=0.6, dpi=dpi)
    if hasattr(hydro, "topographic_wetness_index"):
        _save_field(maps / "49_topographic_wetness.png", np.where(land, hydro.topographic_wetness_index, np.nan),
                    "Topographic Wetness Index", "YlGnBu", dpi=dpi)
        _save_power_field(maps / "50_height_above_drainage.png", np.where(land, hydro.height_above_nearest_drainage_m, 0.0),
                          "Height Above Nearest Drainage (m)", "terrain", gamma=0.55, dpi=dpi)

    depressions = world.get("depressions")
    if depressions is not None:
        _save_power_field(maps / "51_depression_depth.png", depressions.depression_depth_m,
                          "Potential Depression Fill Depth (m)", "Blues", gamma=0.45, dpi=dpi)
        _save_field(maps / "52_endorheic_basins.png", depressions.endorheic_depression.astype(float),
                    "Climatically Endorheic Depression Basins", "magma", vmin=0, vmax=1, dpi=dpi)

    tides = world.get("tides")
    if tides is not None:
        _save_power_field(maps / "53_equilibrium_tidal_range.png", tides.tidal_range_m,
                          "Screened Equilibrium Tidal Range (m)", "viridis", gamma=0.55, dpi=dpi)
        _save_power_field(maps / "54_tidal_current_index.png", tides.tidal_current_index,
                          "Tidal Current / Reworking Index", "viridis", gamma=0.55, dpi=dpi)
        _save_power_field(maps / "55_intertidal_potential.png", tides.intertidal_potential,
                          "Intertidal Wetting Potential", "YlGnBu", gamma=0.55, dpi=dpi)

    planetary = world.get("planetary_appearance")
    if planetary is not None:
        liquids = world.get("surface_liquids")
        if liquids is not None:
            wet = np.asarray(liquids.liquid_mask, dtype=bool)
            liquid_rgb = np.asarray(planetary.surface_liquid_rgb, dtype=np.float32).copy()
            # Neutral land background makes the composition-derived lake/sea color
            # readable without falsely coloring dry cells as liquid.
            liquid_rgb[~wet] = np.asarray([0.16, 0.16, 0.16], dtype=np.float32)
            _save_rgb(maps / "55b_surface_liquid_optical_color.png", liquid_rgb, dpi=dpi)
        _save_field(
            maps / "56_ground_liquid_humidity.png",
            np.where(land, planetary.ground_liquid_humidity_index, np.nan),
            "Ground / Pore-liquid Humidity Index (active condensate)",
            "YlGnBu", vmin=0, vmax=1, dpi=dpi,
        )
        _save_power_field(
            maps / "57_atmospheric_haze_optical_depth.png",
            planetary.atmospheric_haze_optical_depth,
            "Visible Atmospheric Haze Optical-depth Proxy",
            "magma", gamma=0.7, dpi=dpi,
        )
        _save_field(
            maps / "58_solid_condensate_persistence.png",
            np.where(land, planetary.solid_condensate_persistence, np.nan),
            "Persistent Solid Condensate (species-generic)",
            "Blues", vmin=0, vmax=1, dpi=dpi,
        )


__all__ = ["render_advanced_physical_maps"]
