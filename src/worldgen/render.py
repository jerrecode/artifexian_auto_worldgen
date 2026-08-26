from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import image as mpl_image

from .geology import ROCK_NAMES


def _save_field(
    path: Path, field: np.ndarray, title: str, cmap: str = "viridis",
    vmin=None, vmax=None, *, dpi: int = 120,
) -> None:
    fig = plt.figure(figsize=(14, 7), dpi=dpi)
    ax = fig.add_axes([0.04, 0.06, 0.88, 0.88])
    im = ax.imshow(field, origin="upper", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, extent=(-180, 180, -90, 90))
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    # Axes are explicitly positioned, so bbox_inches='tight' only triggers an
    # expensive second layout/render pass without providing useful information.
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _save_power_field(
    path: Path, field: np.ndarray, title: str, cmap: str = "viridis",
    gamma: float = 0.5, percentile: float = 99.5, *, dpi: int = 120,
) -> None:
    arr = np.asarray(field, float)
    finite = arr[np.isfinite(arr)]
    vmax = float(np.percentile(finite, percentile)) if finite.size else 1.0
    vmax = max(vmax, 1e-6)
    fig = plt.figure(figsize=(14, 7), dpi=dpi)
    ax = fig.add_axes([0.04, 0.06, 0.88, 0.88])
    im = ax.imshow(
        arr, origin="upper", aspect="auto", cmap=cmap,
        norm=colors.PowerNorm(gamma=gamma, vmin=0, vmax=vmax, clip=True),
        extent=(-180, 180, -90, 90),
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _save_rgb(path: Path, rgb: np.ndarray, title: str | None = None, *, dpi: int = 135) -> None:
    """Write true-color rasters directly at simulation resolution.

    A Matplotlib Figure previously expanded every RGB raster to a large annotated
    canvas and performed a layout pass. True-color products are remote-sensing
    rasters rather than charts, so direct encoding is both faster and preserves
    exact pixel-to-cell correspondence. ``dpi`` is retained as PNG metadata.
    """
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    mpl_image.imsave(path, arr, format="png", origin="upper", dpi=dpi)


def _save_vector(
    path: Path, u: np.ndarray, v: np.ndarray, title: str,
    background: np.ndarray | None = None, *, dpi: int = 120,
) -> None:
    fig = plt.figure(figsize=(14, 7), dpi=dpi)
    ax = fig.add_axes([0.04, 0.06, 0.88, 0.88])
    speed = np.hypot(u, v)
    bg = speed if background is None else background
    im = ax.imshow(bg, origin="upper", aspect="auto", cmap="viridis", extent=(-180, 180, -90, 90))
    h, w = u.shape
    sy = max(1, h // 24)
    sx = max(1, w // 48)
    yy = np.linspace(90 - 90 / h, -90 + 90 / h, h)[::sy]
    xx = np.linspace(-180, 180, w, endpoint=False)[::sx]
    U = u[::sy, ::sx]
    V = -v[::sy, ::sx]
    ax.quiver(xx, yy, U, V, angles="xy", scale_units="xy", scale=0.13, width=0.0015)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _save_river_centerlines(
    path: Path, elevation: np.ndarray, centerlines: list[dict], title: str, *, dpi: int = 120,
) -> None:
    fig = plt.figure(figsize=(14, 7), dpi=dpi)
    ax = fig.add_axes([0.04, 0.06, 0.92, 0.88])
    ax.imshow(elevation, origin="upper", aspect="auto", cmap="terrain", extent=(-180, 180, -90, 90))
    for r in centerlines:
        pts = np.asarray(r.get("points_lat_lon", []), float)
        if pts.ndim != 2 or len(pts) < 2:
            continue
        lat = pts[:, 0]
        lon = pts[:, 1]
        jumps = np.where(np.abs(np.diff(lon)) > 150)[0]
        starts = np.r_[0, jumps + 1]
        ends = np.r_[jumps + 1, len(lon)]
        lw = 0.45 + 0.75 * float(r.get("mean_meander_potential", 0.0))
        for a, b in zip(starts, ends):
            if b - a >= 2:
                ax.plot(lon[a:b], lat[a:b], linewidth=lw, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _categorical_codes(strings: np.ndarray) -> tuple[np.ndarray, list[str]]:
    cats = sorted(set(map(str, strings.ravel().tolist())))
    lookup = {c: i for i, c in enumerate(cats)}
    code = np.vectorize(lookup.get, otypes=[np.int16])(strings)
    return code, cats


def render_all(out: Path, world: dict) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    maps = out / "maps"
    maps.mkdir(exist_ok=True)
    paths: list[str] = []
    output_cfg = world["config"].output
    map_dpi = int(output_cfg.map_dpi)
    rgb_dpi = int(output_cfg.rgb_dpi)

    def add(name: str, field: np.ndarray, title: str, cmap: str = "viridis", **kw):
        p = maps / f"{name}.png"
        _save_field(p, field, title, cmap, dpi=map_dpi, **kw)
        paths.append(str(p))

    def addpower(name: str, field: np.ndarray, title: str, cmap: str = "viridis", gamma: float = 0.5, percentile: float = 99.5):
        p = maps / f"{name}.png"
        _save_power_field(p, field, title, cmap, gamma, percentile, dpi=map_dpi)
        paths.append(str(p))

    def addrgb(name: str, rgb: np.ndarray, title: str):
        p = maps / f"{name}.png"
        _save_rgb(p, rgb, title, dpi=rgb_dpi)
        paths.append(str(p))

    def addvec(name: str, u: np.ndarray, v: np.ndarray, title: str, background: np.ndarray | None = None):
        p = maps / f"{name}.png"
        _save_vector(p, u, v, title, background, dpi=map_dpi)
        paths.append(str(p))

    def addrivers(name: str, title: str):
        p = maps / f"{name}.png"
        _save_river_centerlines(p, ocean.elevation_km, hydro.river_centerlines, title, dpi=map_dpi)
        paths.append(str(p))

    terrain = world["terrain"]
    ocean = world["ocean"]
    tect = world["tectonics"]
    climate = world["climate"]
    hydro = world["hydrology"]
    weather = world["weather"]
    geo = world["geology"]
    appearance = world["appearance"]
    resources = world["resources"]
    society = world["society"]

    add("01_plate_ids", tect.plate_id, "Hierarchical tectonic parent plates", "tab20")
    add("01b_subplate_ids", tect.subplate_id, "Tectonic subplates / rigid blocks", "nipy_spectral")
    add("01c_tectonic_stress", tect.stress_field, "Accumulated tectonic stress", "inferno", vmin=0, vmax=1)
    add("01d_tectonic_strain", tect.strain_field, "Long-term crustal strain", "magma", vmin=0, vmax=1)
    add("02_elevation", ocean.elevation_km, "Elevation / bathymetry after fluvial evolution (km)", "terrain")
    add("03_ocean_crust_age", np.where(terrain.ocean, tect.crust_age_myr, np.nan), "Oceanic crust age (Myr)", "magma")
    addvec("03b_ocean_currents_january", ocean.current_u_monthly[0], ocean.current_v_monthly[0], "January ocean currents / heat transport", ocean.heat_transport_index)
    addvec("03c_ocean_currents_july", ocean.current_u_monthly[6], ocean.current_v_monthly[6], "July ocean currents / heat transport", ocean.heat_transport_index)
    add("03d_ocean_sst_transport", ocean.sst_anomaly_c, "Annual ocean-current SST anomaly (°C)", "coolwarm")
    add("04_temperature_annual", climate.annual_temperature_c, "Annual mean temperature (°C)", "coolwarm")
    addvec("04b_winds_january", climate.wind_u[0], climate.wind_v[0], "January global winds: trades, westerlies, polar easterlies", np.hypot(climate.wind_u[0], climate.wind_v[0]))
    addvec("04c_winds_july", climate.wind_u[6], climate.wind_v[6], "July global winds: trades, westerlies, polar easterlies", np.hypot(climate.wind_u[6], climate.wind_v[6]))
    addvec("04d_humidity_transport_january", climate.humidity_transport_u[0], climate.humidity_transport_v[0], "January atmospheric humidity transport", climate.humidity_proxy[0])
    addvec("04e_humidity_transport_july", climate.humidity_transport_u[6], climate.humidity_transport_v[6], "July atmospheric humidity transport", climate.humidity_proxy[6])
    addpower("05_precipitation_annual", climate.annual_precipitation_mm, "Annual precipitation (mm; power-scaled through P99.5)", "Blues", 0.48, 99.5)
    addpower("05b_precipitation_january", climate.precipitation_mm[0], "January precipitation incl. orographic enhancement (mm/month)", "Blues", 0.50, 99.5)
    addpower("05c_precipitation_july", climate.precipitation_mm[6], "July precipitation incl. orographic enhancement (mm/month)", "Blues", 0.50, 99.5)
    kcode, kcats = _categorical_codes(climate.koppen)
    add("06_koppen", kcode, "Köppen-Geiger climate classes (integer-coded; legend in metadata)", "tab20")
    add("07_continentality", climate.continentality_index_c, "Annual temperature range / continentality index (°C)", "magma")
    river_hierarchy = hydro.stream_order.astype(float) + 0.65 * hydro.river_width_proxy + 0.40 * hydro.lakes.astype(float)
    add("08_rivers", river_hierarchy, "Rainfall-fed river hierarchy: Strahler order + width/lakes", "Blues")
    add("08a_stream_order", hydro.stream_order, "Strahler stream order", "viridis", vmin=0)
    add("08a2_river_width", hydro.river_width_proxy, "Relative river-width/discharge proxy", "Blues", vmin=0, vmax=1)
    add("08b_runoff", hydro.runoff, "Annual runoff (mm/year equivalent)", "Blues")
    add("08c_erosion", hydro.cumulative_erosion_m, "Cumulative modeled fluvial erosion (m)", "inferno")
    add("08d_deposition", hydro.cumulative_deposition_m, "Cumulative modeled sediment deposition (m)", "copper")
    add("08e_sediment_flux", hydro.sediment_flux_index, "Relative downstream sediment flux", "magma", vmin=0, vmax=1)
    add("08f_meander_potential", hydro.meander_potential, "Rock/slope/discharge-controlled river meander potential", "viridis", vmin=0, vmax=1)
    add("08g_delta_deposition", hydro.delta_deposition_m, "Cumulative river-delta sediment aggradation (m)", "copper")
    add("08h_tectonic_uplift", hydro.tectonic_uplift_m, "Cumulative active tectonic uplift during landscape passes (m)", "inferno")
    add("08i_meander_migration", hydro.meander_migration_m, "Cumulative lateral bank erosion / channel migration proxy (m)", "magma")
    addrivers("08j_meandering_river_centerlines", "Sub-cell major-river centerlines; sinuosity controlled by lithology, slope and discharge")
    add("09_geology", geo.rock_code, "Surface rock classes incl. modeled alluvium (integer-coded; legend in metadata)", "tab10")
    addrgb("15_true_color", appearance.true_color_rgb, "Simulated cloud-free true color / annual surface state")
    addrgb("15b_true_color_january", appearance.true_color_january_rgb, "Simulated cloud-free true color / January")
    addrgb("15c_true_color_july", appearance.true_color_july_rgb, "Simulated cloud-free true color / July")
    addrgb("15c2_true_color_clouds", appearance.true_color_with_clouds_rgb, "Simulated true color with annual mean cloud field")
    addrgb("15c3_true_color_january_clouds", appearance.true_color_january_with_clouds_rgb, "Simulated true color with January cloud field")
    addrgb("15c4_true_color_july_clouds", appearance.true_color_july_with_clouds_rgb, "Simulated true color with July cloud field")
    add("15c5_cloud_fraction", appearance.cloud_fraction_annual, "Annual mean cloud-fraction proxy", "Greys", vmin=0, vmax=1)
    add("15c6_cloud_fraction_january", appearance.cloud_fraction_monthly[0], "January cloud-fraction proxy", "Greys", vmin=0, vmax=1)
    add("15c7_cloud_fraction_july", appearance.cloud_fraction_monthly[6], "July cloud-fraction proxy", "Greys", vmin=0, vmax=1)
    add("15d_vegetation", appearance.vegetation_fraction, "Annual vegetation fraction proxy", "YlGn", vmin=0, vmax=1)
    add("15e_forest", appearance.forest_fraction, "Forest-cover fraction proxy", "Greens", vmin=0, vmax=1)
    add("15f_soil_moisture", appearance.soil_moisture_index, "Soil-moisture index", "YlGnBu", vmin=0, vmax=1)
    add("15g_snow_persistence", appearance.snow_persistence, "Annual snow-persistence fraction", "Blues", vmin=0, vmax=1)
    add("15h_surface_albedo", appearance.surface_albedo, "Surface albedo proxy", "Greys", vmin=0, vmax=0.8)
    add("15i_water_turbidity", appearance.water_turbidity, "Coastal water turbidity / sediment plume proxy", "YlGnBu", vmin=0, vmax=1)
    ore_total = np.zeros_like(terrain.elevation_km, dtype=float)
    for value in resources.suitability.values():
        ore_total += np.asarray(value, float)
    add("10_resource_intensity", ore_total, "Combined resource suitability intensity", "inferno")
    add("11_thunderstorms", weather.thunderstorm_level, "Thunderstorm severity level", "plasma", vmin=0, vmax=4)
    add("12_hurricane_genesis", weather.hurricane_genesis, "Tropical-cyclone genesis potential", "magma", vmin=0, vmax=1)
    add("13_sea_ice_coral", weather.sea_ice_max.astype(float) + 2.0 * weather.coral_reef.astype(float),
        "Seasonal sea ice (1) and coral-reef potential (2)", "coolwarm", vmin=0, vmax=2)
    if society.portal is not None:
        add("14_settlement_suitability", society.suitability, "Human settlement suitability", "YlGn", vmin=0, vmax=1)

    legend_path = maps / "legends.json"
    with legend_path.open("w", encoding="utf-8") as f:
        json.dump({"koppen": {i: c for i, c in enumerate(kcats)}, "rock": ROCK_NAMES}, f, indent=2)
    paths.append(str(legend_path))

    manifest_path = maps / "render_manifest.json"
    manifest_path.write_text(json.dumps({
        "map_dpi": map_dpi,
        "rgb_dpi": rgb_dpi,
        "true_color_pixel_resolution": [int(appearance.true_color_rgb.shape[1]), int(appearance.true_color_rgb.shape[0])],
        "products": [Path(p).name for p in paths],
    }, indent=2), encoding="utf-8")
    paths.append(str(manifest_path))
    return paths
