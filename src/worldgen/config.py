from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import copy
import yaml


@dataclass(slots=True)
class ResolutionConfig:
    width: int = 768
    height: int = 384
    history_step_myr: int = 25
    history_myr: int = 850




@dataclass(slots=True)
class NoiseConfig:
    """Shared hybrid multi-octave noise controls used by every map-like stochastic field."""
    octaves: int = 7
    persistence: float = 0.56
    lacunarity: float = 2.0
    domain_warp_strength: float = 0.32
    minimum_sigma_px: float = 0.52
    value_weight: float = 0.44
    ridge_weight: float = 0.25
    billow_weight: float = 0.16
    wave_weight: float = 0.15
    wave_count: int = 5


@dataclass(slots=True)
class AstronomyConfig:
    star_mass_solar: float = 0.92
    planet_mass_earth: float = 1.0
    planet_density_g_cm3: float = 5.45
    semimajor_axis_au: float | None = None
    eccentricity: float = 0.025
    longitude_periapsis_deg: float = 103.0
    axial_tilt_deg: float = 23.0
    rotation_hours: float = 24.8
    albedo: float = 0.30
    greenhouse_k: float = 33.0
    target_mean_surface_c: float = 15.0
    moon_mass_earth: float = 0.0123
    moon_orbit_km: float = 385000.0
    atmosphere_pressure_bar: float = 1.0
    system_planet_count: int = 8
    stellar_neighborhood_radius_ly: float = 20.0
    stellar_density_per_ly3: float = 0.004
    atmosphere: dict[str, float] = field(default_factory=lambda: {
        "N2": 0.7800, "O2": 0.2090, "Ar": 0.0093, "CO2": 0.0006
    })


@dataclass(slots=True)
class TectonicsConfig:
    plate_count: int = 14
    continental_fraction_target: float = 0.29
    continental_plate_fraction: float = 0.47
    max_plate_speed_cm_yr: float = 8.0
    hotspot_count: int = 22
    lip_interval_myr: float = 16.0
    mountain_uplift_km: float = 5.0
    ridge_uplift_km: float = 2.3
    trench_depth_km: float = 4.0
    terrain_noise_km: float = 0.55
    # Hierarchical microplate/subplate model. Individual subplates have their own
    # preferred Euler motion while parent plates transmit coupling and stress.
    mean_subplates_per_plate: float = 5.0
    min_subplates_per_plate: int = 3
    max_subplates_per_plate: int = 9
    subplate_motion_dispersion: float = 0.32
    parent_coupling: float = 0.68
    collision_nudge: float = 0.10
    split_stress_threshold: float = 1.35
    fuse_direction_deg: float = 15.0
    fuse_persistence_steps: int = 3
    boundary_warp_deg: float = 4.8
    boundary_detail_octaves: int = 6
    boundary_deformation_iterations: int = 2
    strain_boundary_warp_deg: float = 2.6
    # Shape-control anchors make individual rigid blocks non-convex instead of pure spherical Voronoi cells.
    shape_control_points_per_subplate: int = 2
    shape_control_spread_deg: float = 5.0
    history_grid_height: int = 96


@dataclass(slots=True)
class TerrainConfig:
    sea_level_mode: str = "target_land_fraction"
    erosion_m_per_myr: float = 4.6
    lapse_rate_k_per_km: float = 6.5
    shelf_depth_m: float = 200.0
    shelf_width_km_passive: float = 140.0
    shelf_width_km_active: float = 55.0
    fractal_octaves: int = 7
    relief_detail_strength: float = 0.42
    fault_block_relief_km: float = 0.75
    rift_shoulder_uplift_km: float = 0.85
    coastal_reworking_strength: float = 0.20
    coastal_reworking_sigma_px: float = 0.85
    coastal_reworking_band_km: float = 0.22
    min_island_area_km2: float = 900.0


@dataclass(slots=True)
class OceanConfig:
    young_crust_depth_m: float = 2600.0
    subsidence_sqrt_m_per_sqrt_myr: float = 350.0
    max_abyss_depth_m: float = 6200.0
    current_iterations: int = 36
    wind_coupling: float = 0.55
    gyre_strength: float = 2.35
    ekman_strength: float = 0.28
    seasonal_current_strength: float = 0.72
    heat_transport_iterations: int = 8
    heat_advection_strength: float = 0.42
    heat_diffusion_sigma: float = 0.75
    sst_transport_gain: float = 8.0
    western_boundary_strength: float = 0.58
    eastern_boundary_strength: float = 0.18
    bathymetric_steering_strength: float = 0.28
    boundary_current_width_km: float = 420.0
    abyssal_relief_noise_m: float = 150.0


@dataclass(slots=True)
class ClimateConfig:
    months: int = 12
    continentality_k: float = 28.0
    moisture_iterations: int = 30
    precip_scale_mm_year: float = 1100.0
    orographic_strength: float = 2.35
    orographic_lift_scale_km: float = 0.42
    pressure_land_seasonality: float = 5.0
    precipitation_softscale_mm_month: float = 520.0
    precipitation_tail_exponent: float = 0.64
    precipitation_extreme_softcap_mm_month: float = 1350.0
    inland_thermal_length_km: float = 1800.0
    ocean_seasonal_response_c: float = 5.5
    moisture_step_km: float = 190.0
    trade_wind_strength: float = 1.05
    westerly_strength: float = 0.78
    polar_easterly_strength: float = 0.48
    seasonal_itcz_shift_fraction: float = 0.62
    humidity_temperature_sensitivity: float = 0.055
    land_thermal_lag_months: float = 0.85
    ocean_thermal_lag_months: float = 3.2
    thermal_memory_spinup_years: int = 5
    hadley_meridional_strength: float = 0.54
    ferrel_meridional_strength: float = 0.16
    stationary_wave_strength: float = 0.34
    topographic_wind_steering: float = 0.18
    climate_texture_c: float = 0.85
    convective_texture_strength: float = 0.18
    precipitation_mesoscale_sigma_px: float = 1.20


@dataclass(slots=True)
class HydrologyConfig:
    river_accumulation_quantile: float = 0.985
    min_river_precip_mm_year: float = 120.0
    min_drainage_area_km2: float = 520.0
    runoff_base_fraction: float = 0.24
    surface_evolution_iterations: int = 4
    flow_refresh_interval: int = 1
    stream_power_m: float = 0.50
    stream_power_n: float = 1.00
    max_fluvial_erosion_m_per_iteration: float = 15.0
    hillslope_diffusion_strength: float = 0.028
    deposition_strength: float = 0.54
    sediment_routing_passes: int = 16
    river_meander_strength: float = 0.95
    meander_slope_scale: float = 0.0019
    lateral_erosion_fraction: float = 0.34
    meander_microrelief_m: float = 12.0
    delta_retention_fraction: float = 0.52
    delta_max_depth_m: float = 180.0
    delta_spread_cells: float = 2.2
    delta_max_aggradation_m_per_iteration: float = 18.0
    delta_min_outlet_discharge_norm: float = 0.28
    tectonic_uplift_m_per_iteration: float = 4.5
    rift_subsidence_m_per_iteration: float = 1.8
    lake_min_depth_m: float = 5.0
    lake_min_catchment_km2: float = 350.0
    lake_area_soft_cap_fraction_land: float = 0.022
    tributary_discharge_fraction: float = 0.28
    max_river_centerlines: int = 180
    delta_wave_reworking_strength: float = 0.32
    delta_tide_reworking_strength: float = 0.16
    delta_shelf_slope_scale: float = 0.0016
    delta_distributary_texture_strength: float = 0.26


@dataclass(slots=True)
class SimulationConfig:
    # Macro passes couple ocean circulation -> atmosphere -> runoff -> erosion/uplift/deltas -> terrain.
    earth_system_passes: int = 3
    final_climate_ocean_passes: int = 1
    # Intermediate passes are predictors: they retain the same physics but use fewer expensive
    # atmospheric/ocean iterations. The last macro pass and final coupling pass run full fidelity.
    intermediate_climate_fraction: float = 0.28
    intermediate_ocean_fraction: float = 0.52
    preserve_initial_sea_level: bool = True

@dataclass(slots=True)
class WeatherConfig:
    hurricane_sst_c: float = 26.5
    hurricane_lat_min_deg: float = 5.0
    hurricane_lat_max_deg: float = 35.0
    hurricane_seed_count: int = 40
    hurricane_max_steps: int = 100


@dataclass(slots=True)
class ResourcesConfig:
    deposit_density: float = 1.0
    meteorite_expected_count: int = 3


@dataclass(slots=True)
class SocietyConfig:
    enabled: bool = True
    settlement_count: int = 80
    history_years: int = 2200
    history_step_years: int = 25
    portal_prefer_mountains: bool = True
    portal_latitude_max_deg: float = 55.0
    agricultural_productivity_weight: float = 0.22
    river_navigation_bonus: float = 0.10
    coastal_trade_bonus: float = 0.06




@dataclass(slots=True)
class AppearanceConfig:
    vegetation_temp_mid_c: float = 7.0
    vegetation_temp_scale_c: float = 6.5
    forest_precip_mid_mm_year: float = 850.0
    forest_precip_scale_mm_year: float = 420.0
    alpine_bare_start_km: float = 2.6
    alpine_bare_full_km: float = 5.2
    snow_albedo: float = 0.78
    vegetation_albedo: float = 0.16
    desert_albedo: float = 0.31
    ocean_albedo: float = 0.065
    turbidity_spread_sigma_px: float = 3.2
    turbidity_strength: float = 0.78
    hillshade_strength: float = 0.30
    cloud_smoothing_sigma_px: float = 0.85
    cloud_max_optical_opacity: float = 0.72
    cloud_humidity_mid: float = 0.13
    cloud_precip_mid_mm_month: float = 45.0


@dataclass(slots=True)
class OutputConfig:
    save_npz: bool = True
    save_png: bool = True
    save_json: bool = True
    save_report: bool = True
    # Compression is optional because deflate can dominate runtime on high-resolution monthly fields.
    compress_npz: bool = False
    map_dpi: int = 120
    rgb_dpi: int = 135


@dataclass(slots=True)
class WorldConfig:
    seed: int = 20260826
    resolution: ResolutionConfig = field(default_factory=ResolutionConfig)
    astronomy: AstronomyConfig = field(default_factory=AstronomyConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    tectonics: TectonicsConfig = field(default_factory=TectonicsConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    ocean: OceanConfig = field(default_factory=OceanConfig)
    climate: ClimateConfig = field(default_factory=ClimateConfig)
    hydrology: HydrologyConfig = field(default_factory=HydrologyConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    resources: ResourcesConfig = field(default_factory=ResourcesConfig)
    society: SocietyConfig = field(default_factory=SocietyConfig)
    appearance: AppearanceConfig = field(default_factory=AppearanceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SECTION_TYPES = {
    "resolution": ResolutionConfig,
    "astronomy": AstronomyConfig,
    "noise": NoiseConfig,
    "tectonics": TectonicsConfig,
    "terrain": TerrainConfig,
    "ocean": OceanConfig,
    "climate": ClimateConfig,
    "hydrology": HydrologyConfig,
    "simulation": SimulationConfig,
    "weather": WeatherConfig,
    "resources": ResourcesConfig,
    "society": SocietyConfig,
    "appearance": AppearanceConfig,
    "output": OutputConfig,
}


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(instance, key):
            raise KeyError(f"Unknown configuration key: {key}")
        setattr(instance, key, copy.deepcopy(value))
    return instance


def load_config(path: str | Path | None = None) -> WorldConfig:
    cfg = WorldConfig()
    if path is None:
        return cfg
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    for key, value in raw.items():
        if key == "seed":
            cfg.seed = int(value)
        elif key in _SECTION_TYPES:
            if not isinstance(value, dict):
                raise TypeError(f"Configuration section {key!r} must be a mapping")
            _merge_dataclass(getattr(cfg, key), value)
        else:
            raise KeyError(f"Unknown top-level configuration key: {key}")
    if cfg.resolution.width < 64 or cfg.resolution.height < 32:
        raise ValueError("Resolution must be at least 64x32")
    if cfg.resolution.width != 2 * cfg.resolution.height:
        raise ValueError("Use a 2:1 equirectangular grid: width must equal 2*height")
    return cfg
