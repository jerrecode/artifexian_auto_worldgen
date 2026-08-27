from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import copy
import math
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

    # Legacy fixed greenhouse offset remains the default for exact compatibility.
    greenhouse_model: str = "legacy"  # legacy | composition
    greenhouse_k: float = 33.0
    target_mean_surface_c: float = 15.0
    thermodynamics_backend: str = "auto"  # auto | builtin | coolprop

    # Atmosphere is a mole-fraction mapping.  Pressure plus composition and gravity
    # determine scale height/column mass; thickness may optionally override the
    # diagnostic top-of-atmosphere thickness for speculative worlds.
    atmosphere_pressure_bar: float = 1.0
    atmosphere_top_pressure_bar: float = 1.0e-6
    atmosphere_thickness_km: float | None = None
    atmosphere: dict[str, float] = field(
        default_factory=lambda: {"N2": 0.7800, "O2": 0.2090, "Ar": 0.0093, "CO2": 0.0006}
    )
    surface_volatiles: dict[str, float] = field(default_factory=lambda: {"H2O": 1.0})
    surface_condensible: str = "auto"

    # Body hierarchy.  role=moon means the generated world orbits a parent planet
    # which itself follows semimajor_axis_au around the star.
    body_role: str = "planet"  # planet | moon
    parent_body_mass_earth: float | None = None
    parent_body_radius_earth: float | None = None
    parent_orbit_km: float | None = None
    parent_orbit_eccentricity: float = 0.0
    tidal_love_number_k2: float = 0.30
    tidal_quality_factor_q: float = 100.0
    radiogenic_heat_flux_w_m2: float = 0.087

    # General moon list.  An empty list preserves the historical single-moon fields.
    moon_mass_earth: float = 0.0123
    moon_orbit_km: float = 385000.0
    moons: list[dict[str, Any]] = field(default_factory=list)

    system_planet_count: int = 8
    stellar_neighborhood_radius_ly: float = 20.0
    stellar_density_per_ly3: float = 0.004


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
    shape_control_points_per_subplate: int = 2
    shape_control_spread_deg: float = 5.0
    history_grid_height: int = 96

    # Geological regime controls.  auto is resolved from radiogenic + tidal heat.
    geological_activity_mode: str = "auto"  # auto | active | stagnant_lid | inactive | tidal
    activity_strength: float = 1.0
    ice_geology_mode: str = "auto"  # auto | active | inactive
    ice_shell_thickness_km: float = 0.0


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
    backend: str = "fast"
    fluid_species: str = "auto"
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
    condensible_species: str = "auto"
    phase_coupled_evaporation: bool = True


@dataclass(slots=True)
class HydrologyConfig:
    river_accumulation_quantile: float = 0.985
    min_river_precip_mm_year: float = 120.0
    min_drainage_area_km2: float = 520.0
    runoff_base_fraction: float = 0.24
    surface_evolution_iterations: int = 4
    flow_refresh_mode: str = "interval"
    flow_refresh_interval: int = 1
    flow_refresh_max_interval: int = 3
    flow_refresh_elevation_threshold_m: float = 4.0
    flow_refresh_land_change_fraction: float = 2.0e-4
    flow_refresh_delta_threshold_m: float = 2.0
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
    earth_system_passes: int = 3
    final_climate_ocean_passes: int = 1
    intermediate_climate_fraction: float = 0.28
    intermediate_ocean_fraction: float = 0.52
    preserve_initial_sea_level: bool = True
    adaptive_convergence: bool = False
    min_earth_system_passes: int = 2
    max_earth_system_passes: int = 6
    convergence_temperature_c: float = 0.15
    convergence_precip_mm_year: float = 15.0
    convergence_elevation_m: float = 2.0
    required_consecutive_converged_passes: int = 2
    adaptive_final_coupling: bool = False
    min_final_climate_ocean_passes: int = 1
    max_final_climate_ocean_passes: int = 4
    final_convergence_temperature_c: float = 0.08
    final_convergence_precip_mm_year: float = 8.0
    required_consecutive_final_converged_passes: int = 1


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
    compress_npz: bool = False
    map_dpi: int = 120
    rgb_dpi: int = 135


def _number(name: str, value: Any, *, minimum: float | None = None, maximum: float | None = None,
            min_inclusive: bool = True, max_inclusive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise TypeError(f"{name} must be a finite number, got {value!r}")
    x = float(value)
    if minimum is not None and (x < minimum if min_inclusive else x <= minimum):
        raise ValueError(f"{name} must be {'>=' if min_inclusive else '>'} {minimum}, got {value!r}")
    if maximum is not None and (x > maximum if max_inclusive else x >= maximum):
        raise ValueError(f"{name} must be {'<=' if max_inclusive else '<'} {maximum}, got {value!r}")
    return x


def _integer(name: str, value: Any, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {value!r}")
    return value


def _fraction(name: str, value: Any) -> float:
    return _number(name, value, minimum=0.0, maximum=1.0)


def _boolean(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


def _mapping_numbers(name: str, value: Any, *, nonempty: bool = True) -> None:
    if not isinstance(value, dict) or (nonempty and not value):
        raise ValueError(f"{name} must be {'a non-empty' if nonempty else 'a'} mapping")
    for key, number in value.items():
        if not isinstance(key, str) or not key:
            raise TypeError(f"{name} keys must be non-empty strings")
        _number(f"{name}.{key}", number, minimum=0.0)


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

    def validate(self) -> "WorldConfig":
        _integer("seed", self.seed)
        r = self.resolution
        _integer("resolution.width", r.width, minimum=64); _integer("resolution.height", r.height, minimum=32)
        if r.width != 2 * r.height:
            raise ValueError("resolution must be a 2:1 equirectangular grid: width == 2*height")
        _integer("resolution.history_step_myr", r.history_step_myr, minimum=1)
        _integer("resolution.history_myr", r.history_myr, minimum=1)
        if r.history_step_myr > r.history_myr:
            raise ValueError("resolution.history_step_myr cannot exceed resolution.history_myr")

        n = self.noise
        _integer("noise.octaves", n.octaves, minimum=1, maximum=16)
        _number("noise.persistence", n.persistence, minimum=0.0, maximum=1.0, min_inclusive=False, max_inclusive=False)
        _number("noise.lacunarity", n.lacunarity, minimum=1.0, min_inclusive=False)
        _number("noise.domain_warp_strength", n.domain_warp_strength, minimum=0.0, maximum=2.5)
        _number("noise.minimum_sigma_px", n.minimum_sigma_px, minimum=0.05)
        _integer("noise.wave_count", n.wave_count, minimum=2, maximum=64)
        weights = [n.value_weight, n.ridge_weight, n.billow_weight, n.wave_weight]
        for i, value in enumerate(weights): _number(f"noise.weight[{i}]", value, minimum=0.0)
        if sum(map(float, weights)) <= 0: raise ValueError("at least one noise blend weight must be positive")

        a = self.astronomy
        for name in ("star_mass_solar", "planet_mass_earth", "planet_density_g_cm3", "rotation_hours", "atmosphere_pressure_bar", "moon_orbit_km"):
            _number(f"astronomy.{name}", getattr(a, name), minimum=0.0, min_inclusive=False)
        if a.semimajor_axis_au is not None: _number("astronomy.semimajor_axis_au", a.semimajor_axis_au, minimum=0.0, min_inclusive=False)
        _number("astronomy.eccentricity", a.eccentricity, minimum=0.0, maximum=0.95, max_inclusive=False)
        _number("astronomy.axial_tilt_deg", a.axial_tilt_deg, minimum=0.0, maximum=180.0)
        _fraction("astronomy.albedo", a.albedo)
        _number("astronomy.moon_mass_earth", a.moon_mass_earth, minimum=0.0)
        _number("astronomy.atmosphere_top_pressure_bar", a.atmosphere_top_pressure_bar, minimum=0.0, min_inclusive=False)
        if a.atmosphere_top_pressure_bar >= a.atmosphere_pressure_bar:
            raise ValueError("astronomy.atmosphere_top_pressure_bar must be below surface pressure")
        if a.atmosphere_thickness_km is not None: _number("astronomy.atmosphere_thickness_km", a.atmosphere_thickness_km, minimum=0.0, min_inclusive=False)
        if a.greenhouse_model not in {"legacy", "composition"}: raise ValueError("astronomy.greenhouse_model must be legacy or composition")
        if a.thermodynamics_backend not in {"auto", "builtin", "coolprop"}: raise ValueError("astronomy.thermodynamics_backend must be auto, builtin, or coolprop")
        if a.body_role not in {"planet", "moon"}: raise ValueError("astronomy.body_role must be planet or moon")
        _number("astronomy.parent_orbit_eccentricity", a.parent_orbit_eccentricity, minimum=0.0, maximum=0.95, max_inclusive=False)
        _number("astronomy.tidal_love_number_k2", a.tidal_love_number_k2, minimum=0.0, maximum=1.5)
        _number("astronomy.tidal_quality_factor_q", a.tidal_quality_factor_q, minimum=0.0, min_inclusive=False)
        _number("astronomy.radiogenic_heat_flux_w_m2", a.radiogenic_heat_flux_w_m2, minimum=0.0)
        if a.body_role == "moon":
            for key in ("parent_body_mass_earth", "parent_body_radius_earth", "parent_orbit_km"):
                val = getattr(a, key)
                if val is None: raise ValueError(f"astronomy.{key} is required when body_role=moon")
                _number(f"astronomy.{key}", val, minimum=0.0, min_inclusive=False)
        _mapping_numbers("astronomy.atmosphere", a.atmosphere)
        _mapping_numbers("astronomy.surface_volatiles", a.surface_volatiles, nonempty=False)
        if sum(float(x) for x in a.atmosphere.values()) <= 0: raise ValueError("astronomy.atmosphere fractions must sum to a positive value")
        if not isinstance(a.surface_condensible, str) or not a.surface_condensible: raise TypeError("astronomy.surface_condensible must be a non-empty string")
        if not isinstance(a.moons, list): raise TypeError("astronomy.moons must be a list")
        for i, moon in enumerate(a.moons):
            if not isinstance(moon, dict): raise TypeError(f"astronomy.moons[{i}] must be a mapping")
            for key in ("mass_earth", "orbit_km"):
                if key not in moon: raise ValueError(f"astronomy.moons[{i}].{key} is required")
                _number(f"astronomy.moons[{i}].{key}", moon[key], minimum=0.0, min_inclusive=False)
            if "eccentricity" in moon: _number(f"astronomy.moons[{i}].eccentricity", moon["eccentricity"], minimum=0.0, maximum=0.95, max_inclusive=False)
            if "love_number_k2" in moon: _number(f"astronomy.moons[{i}].love_number_k2", moon["love_number_k2"], minimum=0.0, maximum=1.5)
            if "quality_factor_q" in moon: _number(f"astronomy.moons[{i}].quality_factor_q", moon["quality_factor_q"], minimum=0.0, min_inclusive=False)
        _integer("astronomy.system_planet_count", a.system_planet_count, minimum=1, maximum=128)
        _number("astronomy.stellar_neighborhood_radius_ly", a.stellar_neighborhood_radius_ly, minimum=0.0)
        _number("astronomy.stellar_density_per_ly3", a.stellar_density_per_ly3, minimum=0.0)

        t = self.tectonics
        _integer("tectonics.plate_count", t.plate_count, minimum=2, maximum=32766)
        _fraction("tectonics.continental_fraction_target", t.continental_fraction_target); _fraction("tectonics.continental_plate_fraction", t.continental_plate_fraction)
        _number("tectonics.max_plate_speed_cm_yr", t.max_plate_speed_cm_yr, minimum=0.0, min_inclusive=False)
        _integer("tectonics.hotspot_count", t.hotspot_count, minimum=0)
        _number("tectonics.mean_subplates_per_plate", t.mean_subplates_per_plate, minimum=1.0)
        _integer("tectonics.min_subplates_per_plate", t.min_subplates_per_plate, minimum=1); _integer("tectonics.max_subplates_per_plate", t.max_subplates_per_plate, minimum=t.min_subplates_per_plate)
        _fraction("tectonics.parent_coupling", t.parent_coupling); _fraction("tectonics.collision_nudge", t.collision_nudge)
        _integer("tectonics.fuse_persistence_steps", t.fuse_persistence_steps, minimum=1); _integer("tectonics.boundary_detail_octaves", t.boundary_detail_octaves, minimum=1, maximum=16)
        _integer("tectonics.boundary_deformation_iterations", t.boundary_deformation_iterations, minimum=0, maximum=64)
        _integer("tectonics.shape_control_points_per_subplate", t.shape_control_points_per_subplate, minimum=1, maximum=32); _integer("tectonics.history_grid_height", t.history_grid_height, minimum=16)
        if t.geological_activity_mode not in {"auto", "active", "stagnant_lid", "inactive", "tidal"}: raise ValueError("tectonics.geological_activity_mode is invalid")
        if t.ice_geology_mode not in {"auto", "active", "inactive"}: raise ValueError("tectonics.ice_geology_mode is invalid")
        _number("tectonics.activity_strength", t.activity_strength, minimum=0.0, maximum=4.0); _number("tectonics.ice_shell_thickness_km", t.ice_shell_thickness_km, minimum=0.0)

        tr = self.terrain
        if tr.sea_level_mode not in {"target_land_fraction"}: raise ValueError(f"unsupported terrain.sea_level_mode: {tr.sea_level_mode!r}")
        _integer("terrain.fractal_octaves", tr.fractal_octaves, minimum=1, maximum=16)
        for name in ("shelf_depth_m", "shelf_width_km_passive", "shelf_width_km_active", "coastal_reworking_sigma_px", "coastal_reworking_band_km", "min_island_area_km2"):
            _number(f"terrain.{name}", getattr(tr, name), minimum=0.0)
        _fraction("terrain.coastal_reworking_strength", tr.coastal_reworking_strength)

        o = self.ocean
        if o.backend not in {"fast", "barotropic"}: raise ValueError("ocean.backend must be fast or barotropic")
        if not isinstance(o.fluid_species, str) or not o.fluid_species: raise TypeError("ocean.fluid_species must be a non-empty string")
        _integer("ocean.current_iterations", o.current_iterations, minimum=1); _integer("ocean.heat_transport_iterations", o.heat_transport_iterations, minimum=1)
        for name in ("young_crust_depth_m", "subsidence_sqrt_m_per_sqrt_myr", "max_abyss_depth_m", "boundary_current_width_km"):
            _number(f"ocean.{name}", getattr(o, name), minimum=0.0, min_inclusive=False)
        for name in ("wind_coupling", "ekman_strength", "seasonal_current_strength", "heat_advection_strength", "western_boundary_strength", "eastern_boundary_strength", "bathymetric_steering_strength"):
            _number(f"ocean.{name}", getattr(o, name), minimum=0.0, maximum=2.0)

        c = self.climate
        if c.months != 12: raise ValueError("climate.months must currently equal 12")
        _integer("climate.moisture_iterations", c.moisture_iterations, minimum=1); _integer("climate.thermal_memory_spinup_years", c.thermal_memory_spinup_years, minimum=1)
        for name in ("precip_scale_mm_year", "orographic_lift_scale_km", "precipitation_softscale_mm_month", "precipitation_extreme_softcap_mm_month", "inland_thermal_length_km", "moisture_step_km", "land_thermal_lag_months", "ocean_thermal_lag_months", "precipitation_mesoscale_sigma_px"):
            _number(f"climate.{name}", getattr(c, name), minimum=0.0, min_inclusive=False)
        _number("climate.precipitation_tail_exponent", c.precipitation_tail_exponent, minimum=0.1, maximum=1.5); _number("climate.topographic_wind_steering", c.topographic_wind_steering, minimum=0.0, maximum=1.0)
        if not isinstance(c.condensible_species, str) or not c.condensible_species: raise TypeError("climate.condensible_species must be a non-empty string")
        _boolean("climate.phase_coupled_evaporation", c.phase_coupled_evaporation)

        h = self.hydrology
        _fraction("hydrology.river_accumulation_quantile", h.river_accumulation_quantile); _fraction("hydrology.runoff_base_fraction", h.runoff_base_fraction)
        _integer("hydrology.surface_evolution_iterations", h.surface_evolution_iterations, minimum=0)
        if h.flow_refresh_mode not in {"interval", "adaptive", "hybrid"}: raise ValueError("hydrology.flow_refresh_mode must be interval, adaptive, or hybrid")
        _integer("hydrology.flow_refresh_interval", h.flow_refresh_interval, minimum=1); _integer("hydrology.flow_refresh_max_interval", h.flow_refresh_max_interval, minimum=1)
        _number("hydrology.flow_refresh_elevation_threshold_m", h.flow_refresh_elevation_threshold_m, minimum=0.0); _fraction("hydrology.flow_refresh_land_change_fraction", h.flow_refresh_land_change_fraction); _number("hydrology.flow_refresh_delta_threshold_m", h.flow_refresh_delta_threshold_m, minimum=0.0)
        _integer("hydrology.sediment_routing_passes", h.sediment_routing_passes, minimum=1); _integer("hydrology.max_river_centerlines", h.max_river_centerlines, minimum=0)
        for name in ("stream_power_m", "stream_power_n", "min_drainage_area_km2", "max_fluvial_erosion_m_per_iteration", "meander_slope_scale", "delta_max_depth_m", "delta_spread_cells", "lake_min_depth_m", "lake_min_catchment_km2"):
            _number(f"hydrology.{name}", getattr(h, name), minimum=0.0)
        for name in ("deposition_strength", "river_meander_strength", "lateral_erosion_fraction", "delta_retention_fraction", "delta_min_outlet_discharge_norm", "lake_area_soft_cap_fraction_land", "tributary_discharge_fraction", "delta_wave_reworking_strength", "delta_tide_reworking_strength", "delta_distributary_texture_strength"):
            _fraction(f"hydrology.{name}", getattr(h, name))

        sim = self.simulation
        _integer("simulation.earth_system_passes", sim.earth_system_passes, minimum=1); _integer("simulation.final_climate_ocean_passes", sim.final_climate_ocean_passes, minimum=1)
        _fraction("simulation.intermediate_climate_fraction", sim.intermediate_climate_fraction); _fraction("simulation.intermediate_ocean_fraction", sim.intermediate_ocean_fraction); _boolean("simulation.preserve_initial_sea_level", sim.preserve_initial_sea_level)
        _boolean("simulation.adaptive_convergence", sim.adaptive_convergence); _integer("simulation.min_earth_system_passes", sim.min_earth_system_passes, minimum=1); _integer("simulation.max_earth_system_passes", sim.max_earth_system_passes, minimum=sim.min_earth_system_passes)
        _number("simulation.convergence_temperature_c", sim.convergence_temperature_c, minimum=0.0); _number("simulation.convergence_precip_mm_year", sim.convergence_precip_mm_year, minimum=0.0); _number("simulation.convergence_elevation_m", sim.convergence_elevation_m, minimum=0.0); _integer("simulation.required_consecutive_converged_passes", sim.required_consecutive_converged_passes, minimum=1)
        _boolean("simulation.adaptive_final_coupling", sim.adaptive_final_coupling); _integer("simulation.min_final_climate_ocean_passes", sim.min_final_climate_ocean_passes, minimum=1); _integer("simulation.max_final_climate_ocean_passes", sim.max_final_climate_ocean_passes, minimum=sim.min_final_climate_ocean_passes)
        _number("simulation.final_convergence_temperature_c", sim.final_convergence_temperature_c, minimum=0.0); _number("simulation.final_convergence_precip_mm_year", sim.final_convergence_precip_mm_year, minimum=0.0); _integer("simulation.required_consecutive_final_converged_passes", sim.required_consecutive_final_converged_passes, minimum=1)

        w = self.weather
        _number("weather.hurricane_lat_min_deg", w.hurricane_lat_min_deg, minimum=0.0, maximum=90.0); _number("weather.hurricane_lat_max_deg", w.hurricane_lat_max_deg, minimum=w.hurricane_lat_min_deg, maximum=90.0); _integer("weather.hurricane_seed_count", w.hurricane_seed_count, minimum=0); _integer("weather.hurricane_max_steps", w.hurricane_max_steps, minimum=1)
        rs = self.resources; _number("resources.deposit_density", rs.deposit_density, minimum=0.0); _integer("resources.meteorite_expected_count", rs.meteorite_expected_count, minimum=0)
        s = self.society; _boolean("society.enabled", s.enabled); _integer("society.settlement_count", s.settlement_count, minimum=0); _integer("society.history_years", s.history_years, minimum=1); _integer("society.history_step_years", s.history_step_years, minimum=1); _number("society.portal_latitude_max_deg", s.portal_latitude_max_deg, minimum=0.0, maximum=90.0); _fraction("society.agricultural_productivity_weight", s.agricultural_productivity_weight); _fraction("society.river_navigation_bonus", s.river_navigation_bonus); _fraction("society.coastal_trade_bonus", s.coastal_trade_bonus)
        ap = self.appearance
        for name in ("snow_albedo", "vegetation_albedo", "desert_albedo", "ocean_albedo", "turbidity_strength", "hillshade_strength", "cloud_max_optical_opacity"): _fraction(f"appearance.{name}", getattr(ap, name))
        if ap.alpine_bare_full_km <= ap.alpine_bare_start_km: raise ValueError("appearance.alpine_bare_full_km must exceed alpine_bare_start_km")
        out = self.output; _integer("output.map_dpi", out.map_dpi, minimum=40, maximum=1200); _integer("output.rgb_dpi", out.rgb_dpi, minimum=40, maximum=1200)
        return self


_SECTION_TYPES = {
    "resolution": ResolutionConfig, "astronomy": AstronomyConfig, "noise": NoiseConfig,
    "tectonics": TectonicsConfig, "terrain": TerrainConfig, "ocean": OceanConfig,
    "climate": ClimateConfig, "hydrology": HydrologyConfig, "simulation": SimulationConfig,
    "weather": WeatherConfig, "resources": ResourcesConfig, "society": SocietyConfig,
    "appearance": AppearanceConfig, "output": OutputConfig,
}


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(instance, key): raise KeyError(f"Unknown configuration key: {key}")
        setattr(instance, key, copy.deepcopy(value))
    return instance


def load_config(path: str | Path | None = None) -> WorldConfig:
    cfg = WorldConfig()
    if path is None: return cfg.validate()
    with Path(path).open("r", encoding="utf-8") as f: raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict): raise TypeError("top-level configuration document must be a mapping")
    for key, value in raw.items():
        if key == "seed":
            if isinstance(value, bool) or not isinstance(value, int): raise TypeError("seed must be an integer")
            cfg.seed = value
        elif key in _SECTION_TYPES:
            if not isinstance(value, dict): raise TypeError(f"Configuration section {key!r} must be a mapping")
            _merge_dataclass(getattr(cfg, key), value)
        else: raise KeyError(f"Unknown top-level configuration key: {key}")
    return cfg.validate()
