from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any
import copy
import yaml


class ConfigValidationError(ValueError):
    """Raised when a configuration is structurally valid but physically/numerically unsafe."""


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
    earth_system_passes: int = 3
    final_climate_ocean_passes: int = 1
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

    def validate(self) -> "WorldConfig":
        r = self.resolution
        _require(r.width >= 64 and r.height >= 32, "resolution must be at least 64x32")
        _require(r.width == 2 * r.height, "resolution.width must equal 2*resolution.height")
        _require(r.history_step_myr > 0 and r.history_myr > 0, "geological history durations must be positive")
        _require(r.history_step_myr <= r.history_myr, "history_step_myr cannot exceed history_myr")

        n = self.noise
        _require(n.octaves >= 1, "noise.octaves must be >= 1")
        _require(0 < n.persistence <= 1, "noise.persistence must be in (0, 1]")
        _require(n.lacunarity > 1, "noise.lacunarity must be > 1")
        _require(n.domain_warp_strength >= 0 and n.minimum_sigma_px > 0, "noise warp must be nonnegative and sigma positive")
        weights = (n.value_weight, n.ridge_weight, n.billow_weight, n.wave_weight)
        _require(all(x >= 0 for x in weights) and sum(weights) > 0, "noise mixture weights must be nonnegative and not all zero")
        _require(n.wave_count >= 0, "noise.wave_count must be >= 0")

        a = self.astronomy
        _require(a.star_mass_solar > 0 and a.planet_mass_earth > 0 and a.planet_density_g_cm3 > 0, "stellar/planetary mass and density must be positive")
        _require(0 <= a.eccentricity < 1, "astronomy.eccentricity must be in [0,1)")
        _require(0 <= a.albedo < 1, "astronomy.albedo must be in [0,1)")
        _require(0 <= a.axial_tilt_deg <= 180, "astronomy.axial_tilt_deg must be in [0,180]")
        _require(a.rotation_hours > 0 and a.atmosphere_pressure_bar > 0, "rotation period and surface pressure must be positive")
        _require(a.system_planet_count >= 1, "astronomy.system_planet_count must be >= 1")
        _require(a.moon_mass_earth >= 0 and a.moon_orbit_km > 0, "moon mass must be nonnegative and orbit positive")
        _require(bool(a.atmosphere) and all(float(v) >= 0 for v in a.atmosphere.values()) and sum(map(float, a.atmosphere.values())) > 0,
                 "astronomy.atmosphere must contain nonnegative fractions with positive total")

        t = self.tectonics
        _require(t.plate_count >= 2, "tectonics.plate_count must be >= 2")
        _fraction(t.continental_fraction_target, "tectonics.continental_fraction_target")
        _fraction(t.continental_plate_fraction, "tectonics.continental_plate_fraction")
        _fraction(t.parent_coupling, "tectonics.parent_coupling")
        _fraction(t.collision_nudge, "tectonics.collision_nudge")
        _require(t.max_plate_speed_cm_yr > 0 and t.lip_interval_myr > 0, "plate speed and LIP interval must be positive")
        _require(t.hotspot_count >= 0, "tectonics.hotspot_count must be >= 0")
        _require(1 <= t.min_subplates_per_plate <= t.max_subplates_per_plate, "subplate min/max are inconsistent")
        _require(t.mean_subplates_per_plate >= t.min_subplates_per_plate, "mean_subplates_per_plate must be >= minimum")
        _require(t.boundary_detail_octaves >= 1 and t.history_grid_height >= 16, "tectonic detail/history resolution is too small")
        _require(t.boundary_deformation_iterations >= 0 and t.fuse_persistence_steps >= 1, "tectonic iteration counts are invalid")

        tr = self.terrain
        _require(bool(tr.sea_level_mode), "terrain.sea_level_mode cannot be empty")
        _require(tr.lapse_rate_k_per_km > 0 and tr.shelf_depth_m > 0, "lapse rate and shelf depth must be positive")
        _require(tr.shelf_width_km_passive > 0 and tr.shelf_width_km_active > 0, "shelf widths must be positive")
        _require(tr.fractal_octaves >= 1 and tr.min_island_area_km2 >= 0, "terrain octave/island settings are invalid")
        _require(0 <= tr.coastal_reworking_strength <= 1, "terrain.coastal_reworking_strength must be in [0,1]")

        o = self.ocean
        _require(o.young_crust_depth_m > 0 and o.max_abyss_depth_m > o.young_crust_depth_m, "ocean depth parameters are inconsistent")
        _require(o.current_iterations >= 1 and o.heat_transport_iterations >= 1, "ocean iteration counts must be >= 1")
        for name in ("wind_coupling", "ekman_strength", "seasonal_current_strength", "heat_advection_strength", "bathymetric_steering_strength"):
            _require(getattr(o, name) >= 0, f"ocean.{name} must be nonnegative")

        cl = self.climate
        _require(cl.months == 12, "climate.months must currently equal 12")
        _require(cl.moisture_iterations >= 1 and cl.thermal_memory_spinup_years >= 1, "climate iteration counts must be >= 1")
        _require(cl.precip_scale_mm_year > 0 and cl.moisture_step_km > 0 and cl.inland_thermal_length_km > 0, "climate scales must be positive")
        _require(0 < cl.precipitation_tail_exponent <= 1, "climate.precipitation_tail_exponent must be in (0,1]")
        _require(cl.precipitation_extreme_softcap_mm_month > 0, "climate precipitation soft cap must be positive")
        _require(cl.land_thermal_lag_months > 0 and cl.ocean_thermal_lag_months > 0, "thermal lag constants must be positive")

        h = self.hydrology
        _fraction(h.river_accumulation_quantile, "hydrology.river_accumulation_quantile", inclusive_upper=False)
        _fraction(h.runoff_base_fraction, "hydrology.runoff_base_fraction")
        _fraction(h.delta_retention_fraction, "hydrology.delta_retention_fraction")
        _fraction(h.lake_area_soft_cap_fraction_land, "hydrology.lake_area_soft_cap_fraction_land")
        _fraction(h.tributary_discharge_fraction, "hydrology.tributary_discharge_fraction")
        _require(h.surface_evolution_iterations >= 0 and h.flow_refresh_interval >= 1 and h.sediment_routing_passes >= 1, "hydrology iteration counts are invalid")
        _require(h.stream_power_m >= 0 and h.stream_power_n >= 0 and h.max_fluvial_erosion_m_per_iteration >= 0, "stream-power parameters must be nonnegative")
        _require(h.min_drainage_area_km2 >= 0 and h.lake_min_catchment_km2 >= 0 and h.lake_min_depth_m >= 0, "hydrology area/depth thresholds must be nonnegative")

        sim = self.simulation
        _require(sim.earth_system_passes >= 1 and sim.final_climate_ocean_passes >= 1, "simulation pass counts must be >= 1")
        _fraction(sim.intermediate_climate_fraction, "simulation.intermediate_climate_fraction")
        _fraction(sim.intermediate_ocean_fraction, "simulation.intermediate_ocean_fraction")

        w = self.weather
        _require(0 <= w.hurricane_lat_min_deg < w.hurricane_lat_max_deg <= 90, "hurricane latitude bounds are invalid")
        _require(w.hurricane_seed_count >= 0 and w.hurricane_max_steps >= 1, "hurricane count/step settings are invalid")

        rs = self.resources
        _require(rs.deposit_density >= 0 and rs.meteorite_expected_count >= 0, "resource density/count must be nonnegative")

        s = self.society
        _require(s.settlement_count >= 0 and s.history_years > 0 and s.history_step_years > 0, "society count/history settings are invalid")
        _require(0 <= s.portal_latitude_max_deg <= 90, "society.portal_latitude_max_deg must be in [0,90]")
        for name in ("agricultural_productivity_weight", "river_navigation_bonus", "coastal_trade_bonus"):
            _fraction(getattr(s, name), f"society.{name}")

        ap = self.appearance
        _require(ap.vegetation_temp_scale_c > 0 and ap.forest_precip_scale_mm_year > 0, "appearance response scales must be positive")
        _require(ap.alpine_bare_full_km > ap.alpine_bare_start_km, "appearance alpine thresholds are inconsistent")
        for name in ("snow_albedo", "vegetation_albedo", "desert_albedo", "ocean_albedo", "turbidity_strength", "hillshade_strength", "cloud_max_optical_opacity"):
            _fraction(getattr(ap, name), f"appearance.{name}")

        _require(self.output.map_dpi > 0 and self.output.rgb_dpi > 0, "output DPI values must be positive")
        return self


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


CONFIG_UNITS: dict[str, str] = {
    "resolution.history_step_myr": "Myr",
    "resolution.history_myr": "Myr",
    "astronomy.star_mass_solar": "solar_mass",
    "astronomy.planet_mass_earth": "earth_mass",
    "astronomy.planet_density_g_cm3": "g/cm3",
    "astronomy.semimajor_axis_au": "AU",
    "astronomy.rotation_hours": "h",
    "astronomy.atmosphere_pressure_bar": "bar",
    "tectonics.max_plate_speed_cm_yr": "cm/yr",
    "terrain.shelf_depth_m": "m",
    "ocean.young_crust_depth_m": "m",
    "ocean.max_abyss_depth_m": "m",
    "climate.precip_scale_mm_year": "mm/yr",
    "climate.moisture_step_km": "km",
    "hydrology.min_drainage_area_km2": "km2",
    "hydrology.lake_min_depth_m": "m",
    "weather.hurricane_sst_c": "degC",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigValidationError(message)


def _fraction(value: float, name: str, *, inclusive_upper: bool = True) -> None:
    ok = 0 <= float(value) <= 1 if inclusive_upper else 0 <= float(value) < 1
    if not ok:
        bound = "[0,1]" if inclusive_upper else "[0,1)"
        raise ConfigValidationError(f"{name} must be in {bound}")


def config_schema() -> dict[str, dict[str, Any]]:
    """Return a machine-readable flat schema for CLI/UI configuration tooling."""
    cfg = WorldConfig()
    out: dict[str, dict[str, Any]] = {"seed": {"type": "int", "default": cfg.seed, "unit": None}}
    for section in _SECTION_TYPES:
        obj = getattr(cfg, section)
        for f in fields(obj):
            key = f"{section}.{f.name}"
            value = getattr(obj, f.name)
            out[key] = {
                "type": str(f.type),
                "default": copy.deepcopy(value),
                "unit": CONFIG_UNITS.get(key),
            }
    return out


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(instance, key):
            raise KeyError(f"Unknown configuration key: {key}")
        setattr(instance, key, copy.deepcopy(value))
    return instance


def load_config(path: str | Path | None = None) -> WorldConfig:
    cfg = WorldConfig()
    if path is None:
        return cfg.validate()
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise TypeError("top-level configuration must be a mapping")
    for key, value in raw.items():
        if key == "seed":
            cfg.seed = int(value)
        elif key in _SECTION_TYPES:
            if not isinstance(value, dict):
                raise TypeError(f"Configuration section {key!r} must be a mapping")
            _merge_dataclass(getattr(cfg, key), value)
        else:
            raise KeyError(f"Unknown top-level configuration key: {key}")
    return cfg.validate()
