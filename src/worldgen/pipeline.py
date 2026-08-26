from __future__ import annotations
from dataclasses import dataclass, asdict, is_dataclass
from pathlib import Path
import json
import time
import copy
import gc
from typing import Callable, Any
import numpy as np

from .config import WorldConfig
from .rng import RngPool
from .grid import SphereGrid
from .astronomy import build_astronomy
from .tectonics import generate_tectonics
from .terrain import build_terrain, rebuild_terrain_from_elevation
from .ocean import build_ocean
from .climate import build_climate
from .hydrology import build_hydrology, evolve_surface, SurfaceEvolutionResult
from .weather import build_weather
from .geology import build_geology
from .resources import build_resources
from .society import build_society
from .appearance import build_surface_appearance
from .noise import build_static_noise_fields
from .render import render_all


class WorldPipeline:
    """Dependency-ordered automatic world generator.

    Each stage receives its own deterministic RNG stream. Changing a downstream
    parameter does not silently randomize the star, tectonics, etc.
    """

    def __init__(self, config: WorldConfig, progress: Callable[[str], None] | None = print):
        self.cfg = config
        self.progress = progress or (lambda _: None)
        self.rng = RngPool(config.seed)
        self.timings: dict[str, float] = {}

    def _stage(self, name: str, fn: Callable[[], Any]) -> Any:
        self.progress(f"[{name}] starting")
        t0 = time.perf_counter()
        value = fn()
        dt = time.perf_counter() - t0
        self.timings[name] = dt
        self.progress(f"[{name}] done in {dt:.3f}s")
        return value

    def generate(self, out_dir: str | Path | None = None) -> dict[str, Any]:
        c = self.cfg
        astro = self._stage("astronomy", lambda: build_astronomy(c.astronomy, self.rng("astronomy")))
        radius_km = 6371.0 * astro.planet["radius_earth"]
        grid = SphereGrid(c.resolution.width, c.resolution.height, radius_km)
        tect = self._stage("tectonics", lambda: generate_tectonics(grid, c.tectonics, c.resolution, self.rng("tectonics"), c.noise))
        terrain = self._stage("terrain_initial", lambda: build_terrain(grid, tect, c.tectonics, c.terrain, self.rng("terrain"), c.noise))

        shape = terrain.elevation_km.shape
        static_noise = self._stage("noise_cache", lambda: build_static_noise_fields(shape, c.noise, self.rng))
        cum_er = np.zeros(shape, np.float32)
        cum_dep = np.zeros(shape, np.float32)
        cum_delta = np.zeros(shape, np.float32)
        cum_up = np.zeros(shape, np.float32)
        cum_mig = np.zeros(shape, np.float32)
        flux = np.zeros(shape, np.float32)
        meander = np.zeros(shape, np.float32)
        climate_prev = None
        ocean = climate = geology = None
        coupling_history: list[dict[str, Any]] = []
        macro = max(1, int(c.simulation.earth_system_passes))

        # Multirate coupled Earth-system passes. Intermediate atmosphere/ocean solves are predictors:
        # they retain the same mechanisms but use fewer numerical iterations. The last macro pass and
        # the final circulation pass always run at full fidelity. This greatly improves high-resolution
        # scaling without breaking the feedback chain.
        for ipass in range(macro):
            terrain_before = terrain.elevation_km.astype(float)
            prev_temp = None if climate_prev is None else climate_prev.annual_temperature_c
            prev_precip = None if climate_prev is None else climate_prev.annual_precipitation_mm
            if macro <= 1 or ipass == macro - 1:
                progress = 1.0
            else:
                progress = ipass / max(macro - 1, 1)

            cmin = float(np.clip(c.simulation.intermediate_climate_fraction, 0.10, 1.0))
            omin = float(np.clip(c.simulation.intermediate_ocean_fraction, 0.15, 1.0))
            climate_fidelity = 1.0 if ipass == macro - 1 else cmin + (1.0 - cmin) * progress ** 1.6
            ocean_fidelity = 1.0 if ipass == macro - 1 else omin + (1.0 - omin) * progress ** 1.35

            wind_u = None if climate_prev is None else climate_prev.wind_u
            wind_v = None if climate_prev is None else climate_prev.wind_v
            # The previous ocean is no longer needed. Release it before constructing another set of
            # 12 monthly current/SST fields; this avoids a high-resolution peak-memory cliff.
            if ipass > 0:
                ocean = None
                gc.collect()
            ocfg = copy.deepcopy(c.ocean)
            ocfg.current_iterations = max(10, int(round(c.ocean.current_iterations * ocean_fidelity)))
            ocfg.heat_transport_iterations = max(3, int(round(c.ocean.heat_transport_iterations * ocean_fidelity)))
            ocean = self._stage(
                f"ocean_pass_{ipass+1}",
                lambda wu=wind_u, wv=wind_v, ip=ipass, oc=ocfg: build_ocean(
                    grid, tect, terrain, oc, c.terrain, self.rng("ocean-stationary"), wu, wv, c.noise, static_noise
                ),
            )

            # build_ocean has consumed the wind arrays; retain only the small annual convergence fields.
            wind_u = wind_v = None
            climate_prev = None
            climate = None
            gc.collect()
            ccfg = copy.deepcopy(c.climate)
            ccfg.moisture_iterations = max(6, int(round(c.climate.moisture_iterations * climate_fidelity)))
            climate = self._stage(
                f"climate_pass_{ipass+1}",
                lambda ip=ipass, cc=ccfg: build_climate(
                    grid, astro, terrain, ocean, cc, c.terrain, self.rng("climate-stationary"), c.noise, static_noise
                ),
            )
            geology = self._stage(
                f"geology_pass_{ipass+1}",
                lambda ip=ipass: build_geology(grid, tect, terrain, ocean, climate, self.rng("geology-stationary"), c.noise, static_noise),
            )
            surf = self._stage(
                f"surface_pass_{ipass+1}",
                lambda ip=ipass: evolve_surface(
                    grid, terrain, ocean, climate, geology, c.hydrology, tect, self.rng("surface-stationary"), c.noise, static_noise
                ),
            )
            cum_er += surf.cumulative_erosion_m
            cum_dep += surf.cumulative_deposition_m
            cum_delta += surf.delta_deposition_m
            cum_up += surf.tectonic_uplift_m
            cum_mig += surf.meander_migration_m
            flux = np.maximum(flux, surf.sediment_flux_index)
            meander = surf.meander_potential

            elev = surf.elevation_km.astype(float)
            sea_shift = 0.0
            if not c.simulation.preserve_initial_sea_level and c.terrain.sea_level_mode == "target_land_fraction":
                sea_shift = grid.weighted_quantile(elev, 1.0 - c.tectonics.continental_fraction_target)
                elev -= sea_shift
            terrain = self._stage(
                f"terrain_pass_{ipass+1}",
                lambda e=elev, ss=sea_shift, ip=ipass: rebuild_terrain_from_elevation(
                    grid, tect, c.terrain, e, float(ss),
                    {**terrain.metadata, "surface_evolved": True, "earth_system_pass": ip + 1},
                ),
            )

            stat = {
                "stage": f"earth_system_pass_{ipass+1}",
                "climate_fidelity": float(climate_fidelity),
                "ocean_fidelity": float(ocean_fidelity),
                "moisture_iterations": int(ccfg.moisture_iterations),
                "mean_abs_elevation_change_m": float(
                    np.average(
                        np.abs(terrain.elevation_km[(terrain.elevation_km > 0) | (terrain_before > 0)] - terrain_before[(terrain.elevation_km > 0) | (terrain_before > 0)]),
                        weights=grid.cell_area_weights[(terrain.elevation_km > 0) | (terrain_before > 0)],
                    ) * 1000.0
                ) if np.any((terrain.elevation_km > 0) | (terrain_before > 0)) else 0.0,
                "max_pass_erosion_m": float(surf.cumulative_erosion_m.max()),
                "max_pass_deposition_m": float(surf.cumulative_deposition_m.max()),
                "max_pass_delta_aggradation_m": float(surf.delta_deposition_m.max()),
                "max_pass_uplift_m": float(surf.tectonic_uplift_m.max()),
            }
            if prev_temp is not None:
                stat["mean_abs_temperature_change_c"] = float(
                    np.sum(np.abs(climate.annual_temperature_c - prev_temp) * grid.cell_area_weights)
                )
                stat["mean_abs_precipitation_change_mm_year"] = float(
                    np.sum(np.abs(climate.annual_precipitation_mm - prev_precip) * grid.cell_area_weights)
                )
            coupling_history.append(stat)
            climate_prev = climate
            # Per-pass geology/surface objects are fully accumulated above and can be released.
            surf = None
            geology = None
            gc.collect()

        # Final circulation-only convergence pass(es): no further geomorphic time step, so atmosphere
        # and ocean can equilibrate against the evolved mountain ranges, deltas and coastlines.
        final_coupling = max(1, int(c.simulation.final_climate_ocean_passes))
        for k in range(final_coupling):
            prev_temp = climate_prev.annual_temperature_c
            prev_precip = climate_prev.annual_precipitation_mm
            wind_u = climate_prev.wind_u
            wind_v = climate_prev.wind_v
            ocean = None
            gc.collect()
            ocean = self._stage(
                f"ocean_final_couple_{k+1}",
                lambda kk=k: build_ocean(
                    grid, tect, terrain, c.ocean, c.terrain, self.rng("ocean-stationary"),
                    wind_u, wind_v, c.noise, static_noise,
                ),
            )
            wind_u = wind_v = None
            climate_prev = None
            climate = None
            gc.collect()
            climate = self._stage(
                f"climate_final_couple_{k+1}",
                lambda kk=k: build_climate(
                    grid, astro, terrain, ocean, c.climate, c.terrain, self.rng("climate-stationary"), c.noise, static_noise
                ),
            )
            coupling_history.append({
                "stage": f"final_climate_ocean_pass_{k+1}",
                "climate_fidelity": 1.0,
                "ocean_fidelity": 1.0,
                "moisture_iterations": int(c.climate.moisture_iterations),
                "mean_abs_temperature_change_c": float(
                    np.sum(np.abs(climate.annual_temperature_c - prev_temp) * grid.cell_area_weights)
                ),
                "mean_abs_precipitation_change_mm_year": float(
                    np.sum(np.abs(climate.annual_precipitation_mm - prev_precip) * grid.cell_area_weights)
                ),
            })
            climate_prev = climate

        geology=self._stage("geology_final", lambda: build_geology(grid,tect,terrain,ocean,climate,self.rng("geology-stationary"),c.noise,static_noise))
        geology.rock_code[(cum_dep>4.0)&terrain.land]=0
        surface=SurfaceEvolutionResult(
            terrain.elevation_km.astype(np.float32),cum_er,cum_dep,flux,cum_delta,cum_up,cum_mig,meander,
            {
                "earth_system_passes":macro,
                "geomorphic_iterations_total":macro*int(c.hydrology.surface_evolution_iterations),
                "max_cumulative_erosion_m":float(cum_er.max()),
                "max_cumulative_deposition_m":float(cum_dep.max()),
                "max_delta_aggradation_m":float(cum_delta.max()),
                "max_tectonic_uplift_m":float(cum_up.max()),
                "max_meander_bank_migration_m":float(cum_mig.max()),
                "model":"multi-pass atmosphere-ocean-tectonics-fluvial surface evolution",
            })
        hydro=self._stage("hydrology_final", lambda: build_hydrology(grid,terrain,ocean,climate,c.hydrology,geology,surface))
        weather=self._stage("weather", lambda: build_weather(grid,terrain,ocean,climate,hydro,c.weather,self.rng("weather")))
        appearance=self._stage("surface_appearance", lambda: build_surface_appearance(grid,terrain,ocean,climate,hydro,geology,weather,c.appearance))
        resources=self._stage("resources", lambda: build_resources(grid,tect,terrain,ocean,climate,hydro,geology,c.resources,self.rng("resources")))
        society=self._stage("society", lambda: build_society(grid,terrain,climate,hydro,resources,weather,c.society,self.rng("society"),appearance))
        world={
            "config":c,"grid":grid,"astronomy":astro,"tectonics":tect,"terrain":terrain,"ocean":ocean,
            "climate":climate,"surface_evolution":surface,"hydrology":hydro,"weather":weather,
            "geology":geology,"appearance":appearance,"resources":resources,"society":society,"coupling_history":coupling_history,
        }
        if out_dir is not None:
            self._stage("output",lambda:self.save(world,Path(out_dir)))
        return world

    def save(self, world: dict[str, Any], out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        c = self.cfg
        if c.output.save_png:
            render_all(out, world)
        if c.output.save_npz:
            writer = np.savez_compressed if c.output.compress_npz else np.savez
            writer(out / "world_arrays.npz", **self._array_export(world))
        if c.output.save_json:
            payload = self._json_export(world)
            with (out / "world.json").open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            with (out / "features.geojson").open("w", encoding="utf-8") as f:
                json.dump(self._geojson_export(world), f, ensure_ascii=False)
        if c.output.save_report:
            (out / "world_report.md").write_text(self._report(world), encoding="utf-8")

    def _array_export(self, w: dict[str, Any]) -> dict[str, np.ndarray]:
        return {
            "lat": w["grid"].lat_1d, "lon": w["grid"].lon_1d,
            "plate_id": w["tectonics"].plate_id, "subplate_id": w["tectonics"].subplate_id,
            "subplate_parent": w["tectonics"].subplate_parent,
            "subplate_centers_xyz": w["tectonics"].subplate_centers_xyz,
            "subplate_omega_xyz": w["tectonics"].subplate_omega_xyz,
            "plate_centers_xyz": w["tectonics"].plate_centers_xyz,
            "plate_omega_xyz": w["tectonics"].plate_omega_xyz,
            "subplate_stress": w["tectonics"].subplate_stress,
            "continental_crust": w["tectonics"].continental_crust,
            "plate_boundary": w["tectonics"].boundary, "subplate_boundary": w["tectonics"].subplate_boundary,
            "intraplate_fault": w["tectonics"].intraplate_fault,
            "convergent": w["tectonics"].convergent, "divergent": w["tectonics"].divergent, "transform": w["tectonics"].transform,
            "convergence_strength": w["tectonics"].convergence_strength, "divergence_strength": w["tectonics"].divergence_strength,
            "tectonic_stress": w["tectonics"].stress_field, "tectonic_strain": w["tectonics"].strain_field,
            "crust_age_myr": w["tectonics"].crust_age_myr,
            "elevation_km": w["ocean"].elevation_km, "ocean_depth_m": w["ocean"].depth_m,
            "ocean_current_u": w["ocean"].current_u, "ocean_current_v": w["ocean"].current_v,
            "ocean_current_u_monthly": w["ocean"].current_u_monthly, "ocean_current_v_monthly": w["ocean"].current_v_monthly,
            "ocean_sst_anomaly_c_monthly": w["ocean"].sst_anomaly_c_monthly,
            "ocean_heat_transport_index": w["ocean"].heat_transport_index,
            "temperature_c_monthly": w["climate"].temperature_c,
            "precipitation_mm_monthly": w["climate"].precipitation_mm,
            "pressure_monthly": w["climate"].pressure_anomaly,
            "wind_u_monthly": w["climate"].wind_u, "wind_v_monthly": w["climate"].wind_v,
            "global_circulation_u_monthly": w["climate"].global_circulation_u,
            "global_circulation_v_monthly": w["climate"].global_circulation_v,
            "humidity_proxy_monthly": w["climate"].humidity_proxy,
            "humidity_transport_u_monthly": w["climate"].humidity_transport_u,
            "humidity_transport_v_monthly": w["climate"].humidity_transport_v,
            "annual_temperature_c": w["climate"].annual_temperature_c,
            "annual_precipitation_mm": w["climate"].annual_precipitation_mm,
            "koppen": w["climate"].koppen,
            "continentality_index_c": w["climate"].continentality_index_c,
            "continentality_class": w["climate"].continentality_class,
            "flow_to": w["hydrology"].flow_to, "flow_accumulation": w["hydrology"].accumulation,
            "drainage_area_km2": w["hydrology"].drainage_area_km2, "discharge_index": w["hydrology"].discharge_index,
            "rivers": w["hydrology"].rivers,
            "stream_order": w["hydrology"].stream_order,
            "river_width_proxy": w["hydrology"].river_width_proxy,
            "lakes": w["hydrology"].lakes,
            "runoff_mm_year": w["hydrology"].runoff,
            "cumulative_erosion_m": w["hydrology"].cumulative_erosion_m,
            "cumulative_deposition_m": w["hydrology"].cumulative_deposition_m,
            "sediment_flux_index": w["hydrology"].sediment_flux_index,
            "delta_deposition_m": w["hydrology"].delta_deposition_m,
            "tectonic_uplift_m": w["hydrology"].tectonic_uplift_m,
            "meander_migration_m": w["hydrology"].meander_migration_m,
            "meander_potential": w["hydrology"].meander_potential,
            "river_sinuosity_proxy": w["hydrology"].sinuosity_proxy,
            "rock_code": w["geology"].rock_code, "paleoshallow_sea": w["geology"].paleoshallow_sea,
            "fog": w["weather"].fog, "thunderstorm_level": w["weather"].thunderstorm_level,
            "lightning_flashes_km2_year": w["weather"].lightning_flashes_km2_year,
            "tornado_potential": w["weather"].tornado_potential, "blizzard": w["weather"].blizzard,
            "sandstorm": w["weather"].sandstorm, "duststorm": w["weather"].duststorm,
            "hurricane_genesis": w["weather"].hurricane_genesis, "aurora": w["weather"].aurora,
            "sea_ice_max": w["weather"].sea_ice_max, "sea_ice_min": w["weather"].sea_ice_min,
            "coral_reef": w["weather"].coral_reef,
            "vegetation_fraction": w["appearance"].vegetation_fraction,
            "forest_fraction": w["appearance"].forest_fraction,
            "grass_fraction": w["appearance"].grass_fraction,
            "bare_ground_fraction": w["appearance"].bare_ground_fraction,
            "soil_moisture_index": w["appearance"].soil_moisture_index,
            "snow_persistence": w["appearance"].snow_persistence,
            "surface_albedo": w["appearance"].surface_albedo,
            "water_turbidity": w["appearance"].water_turbidity,
            "cloud_fraction_monthly": w["appearance"].cloud_fraction_monthly,
            "cloud_fraction_annual": w["appearance"].cloud_fraction_annual,
            "true_color_rgb": w["appearance"].true_color_rgb,
            "true_color_january_rgb": w["appearance"].true_color_january_rgb,
            "true_color_july_rgb": w["appearance"].true_color_july_rgb,
            "true_color_with_clouds_rgb": w["appearance"].true_color_with_clouds_rgb,
            "true_color_january_with_clouds_rgb": w["appearance"].true_color_january_with_clouds_rgb,
            "true_color_july_with_clouds_rgb": w["appearance"].true_color_july_with_clouds_rgb,
            "settlement_suitability": w["society"].suitability,
            **{f"resource_{k}": v for k, v in w["resources"].suitability.items()},
        }

    def _geojson_export(self, w: dict[str, Any]) -> dict[str, Any]:
        features = []
        for d in w["resources"].deposits:
            props = {k: v for k, v in d.items() if k not in {"latitude", "longitude"}}
            features.append({"type": "Feature", "geometry": {"type": "Point",
                "coordinates": [d["longitude"], d["latitude"]]}, "properties": {"feature_class": "resource_deposit", **props}})
        if w["society"].portal is not None:
            p = w["society"].portal
            features.append({"type": "Feature", "geometry": {"type": "Point",
                "coordinates": [p["longitude"], p["latitude"]]}, "properties": {"feature_class": "portal"}})
        for st in w["society"].settlements:
            props = {k: v for k, v in st.items() if k not in {"latitude", "longitude"}}
            features.append({"type": "Feature", "geometry": {"type": "Point",
                "coordinates": [st["longitude"], st["latitude"]]}, "properties": {"feature_class": "settlement", **props}})
        for rv in w["hydrology"].river_centerlines:
            coords = [[lon, lat] for lat, lon in rv["points_lat_lon"]]
            props = {k:v for k,v in rv.items() if k!="points_lat_lon"}
            features.append({"type":"Feature","geometry":{"type":"LineString","coordinates":coords},
                             "properties":{"feature_class":"river_centerline",**props}})
        for i, tr in enumerate(w["weather"].hurricane_tracks):
            coords = [[lon, lat] for lat, lon in tr["points_lat_lon"]]
            features.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords},
                             "properties": {"feature_class": "hurricane_track", "track_id": i, "month": tr["month"], "steps": tr["steps"]}})
        return {"type": "FeatureCollection", "features": features}

    def _json_export(self, w: dict[str, Any]) -> dict[str, Any]:
        return {
            "seed": self.cfg.seed,
            "config": self.cfg.to_dict(),
            "astronomy": w["astronomy"].to_dict(),
            "timings_seconds": self.timings,
            "metadata": {
                "tectonics": w["tectonics"].metadata, "terrain": w["terrain"].metadata,
                "ocean": w["ocean"].metadata, "climate": w["climate"].metadata,
                "surface_evolution": w["surface_evolution"].metadata,
                "hydrology": w["hydrology"].metadata, "weather": w["weather"].metadata,
                "geology": w["geology"].metadata, "appearance": w["appearance"].metadata,
                "resources": w["resources"].metadata, "society": w["society"].metadata,
            },
            "resource_deposits": w["resources"].deposits,
            "hurricane_tracks": w["weather"].hurricane_tracks,
            "river_centerlines": w["hydrology"].river_centerlines,
            "coupling_history": w.get("coupling_history", []),
            "portal": w["society"].portal,
            "settlements": w["society"].settlements,
            "cultures": w["society"].cultures,
            "society_links": w["society"].links,
            "history_events": w["society"].history_events,
        }

    def _report(self, w: dict[str, Any]) -> str:
        a = w["astronomy"]
        t = w["terrain"].metadata
        c = w["climate"].metadata
        r = w["resources"].metadata
        s = w["society"].metadata
        ap = w["appearance"].metadata
        lines = [
            "# Automatically Generated World Report", "",
            f"Seed: `{self.cfg.seed}`", "",
            "## Astronomy", "",
            f"- Star: {a.star['mass_solar']:.3f} M☉, {a.star['luminosity_solar']:.3f} L☉, {a.star['effective_temperature_k']:.0f} K.",
            f"- Planet: {a.planet['mass_earth']:.3f} M⊕, {a.planet['radius_earth']:.3f} R⊕, {a.planet['surface_gravity_g']:.3f} g.",
            f"- Orbit: {a.planet['semimajor_axis_au']:.3f} AU; local year {a.calendar['local_year_days']:.1f} days.",
            f"- Moon synodic month: {a.calendar['synodic_month_days']:.2f} local days.", "",
            "## Planet", "",
            f"- Land fraction: {100*t['actual_land_fraction']:.1f}%.",
            f"- Mean temperature: {c['global_mean_temperature_c']:.1f} °C.",
            f"- Mean land precipitation: {c['land_mean_annual_precip_mm']:.0f} mm/year.",
            f"- Resource deposit records: {r['deposit_count']}.",
            f"- Settlements: {s.get('settlement_count', 0)}; cultures: {s.get('culture_count', 0)}.",
            f"- Mean land vegetation fraction: {100*ap.get('mean_land_vegetation_fraction',0):.1f}%.", "",
            "## Reproducibility", "",
            "Every stage uses a deterministic RNG stream derived from the root seed. Change the seed for a new world; change a module parameter to tune one mechanism without intentionally rerolling unrelated stages.", "",
            "## Scientific/creative boundary", "",
            "This is a procedural equivalent of the transcript workflow, not a full mantle-convection, GCM, or sociohistorical first-principles simulator. Explicit transcript heuristics are encoded where available; manual/vibes-based or omitted procedures are replaced by seeded constraints or quantitative approximations and are documented in SOURCE_COVERAGE.md.",
        ]
        return "\n".join(lines) + "\n"
