from __future__ import annotations

"""Canonical advanced planetary pipeline extension.

Composition-aware worlds add multicomponent volatile cycling, exotic-ocean state,
automatic geodynamic regime selection, cryogeology, fluid-property-dependent actual
landscape evolution, depression storage, multi-moon/stellar tides and expanded
hydrology outputs.  The expensive global climate/ocean solve is still reduced-order,
but the advanced layers now feed back into terrain rather than remaining diagnostic.
"""

from pathlib import Path
from typing import Any
import json

import numpy as np

from .pipeline_liquids import WorldPipeline as _LiquidWorldPipeline
from .volatile_cycle import build_volatile_cycle
from .exotic_ocean import build_exotic_ocean
from .geodynamics import build_geodynamic_regime
from .cryogeology import build_cryogeology
from .geomorphic_fluids import (
    build_geomorphic_fluid_parameters,
    scaled_hydrology_config,
    build_exotic_geomorphology,
)
from .planetary_chemistry import chemistry_metadata
from .depressions import build_depressions
from .tides import build_tides
from .appearance_advanced import attenuate_deep_bathymetry
from .advanced_render import render_advanced_physical_maps


class WorldPipeline(_LiquidWorldPipeline):
    """Surface-liquid pipeline plus bidirectionally coupled advanced planet physics."""

    @staticmethod
    def _advanced_chemistry_enabled(cfg) -> bool:
        if bool(getattr(getattr(cfg, "atmogen", None), "enabled", False)):
            # atmogen is authoritative on the new path; do not run the older proxy
            # chemistry/optics engine in competition with it.
            return False
        return str(getattr(cfg.astronomy, "greenhouse_model", "legacy")) == "composition"

    @staticmethod
    def _inject_tidal_reworking(ocean, tides):
        """Expose spatial tidal energy to the existing shelf/delta sediment kernel."""
        if tides is None:
            return ocean
        current = np.asarray(getattr(ocean, "current_speed", 0.0), dtype=float)
        tide = np.asarray(tides.tidal_current_index, dtype=float)
        ocean.current_speed = np.maximum(current, 0.85 * tide).astype(np.float32)
        ocean.metadata = {
            **ocean.metadata,
            "tidal_sediment_reworking_coupled": True,
            "tidal_current_index_max": float(np.max(tide)) if tide.size else 0.0,
        }
        return ocean

    def _build_exotic_layers(self, world: dict[str, Any], *, suffix: str = "") -> dict[str, Any]:
        c = self.cfg
        tag = f"_{suffix}" if suffix else ""
        grid = world["grid"]
        astro = world["astronomy"]
        climate = world["climate"]
        terrain = world["terrain"]
        tect = world["tectonics"]
        liquids = world.get("surface_liquids")

        volatile = self._stage(
            f"volatile_cycle_multicomponent{tag}",
            lambda: build_volatile_cycle(
                grid,
                astro,
                climate,
                surface_volatiles=c.astronomy.surface_volatiles,
                surface_liquids=liquids,
            ),
        )
        exotic_ocean = None
        if liquids is not None:
            exotic_ocean = self._stage(
                f"exotic_ocean_state{tag}",
                lambda: build_exotic_ocean(
                    grid, astro, climate, world["ocean"], liquids, volatile
                ),
            )

        geodynamics = self._stage(
            f"automatic_geodynamic_regime{tag}",
            lambda: build_geodynamic_regime(
                astro, c.tectonics, climate=climate, exotic_ocean=exotic_ocean
            ),
        )
        cryogeology = self._stage(
            f"cryogeology{tag}",
            lambda: build_cryogeology(
                grid,
                astro,
                terrain,
                climate,
                tect,
                c.tectonics,
                geodynamics,
                exotic_ocean=exotic_ocean,
                volatile_cycle=volatile,
            ),
        )
        geomorphic_params = self._stage(
            f"geomorphic_fluid_parameters{tag}",
            lambda: build_geomorphic_fluid_parameters(
                astro, exotic_ocean, volatile, cryogeology
            ),
        )
        tides = self._stage(
            f"tides{tag}",
            lambda: build_tides(grid, astro, terrain, world["ocean"]),
        )
        depressions = self._stage(
            f"depression_storage{tag}",
            lambda: build_depressions(grid, terrain, climate, world["hydrology"]),
        )
        exotic_geomorphology = self._stage(
            f"exotic_geomorphology{tag}",
            lambda: build_exotic_geomorphology(
                grid,
                terrain,
                climate,
                world["hydrology"],
                geomorphic_params,
                volatile,
                cryogeology,
            ),
        )

        world.update(
            {
                "volatile_cycle": volatile,
                "exotic_ocean": exotic_ocean,
                "geodynamics": geodynamics,
                "cryogeology": cryogeology,
                "geomorphic_fluid_parameters": geomorphic_params,
                "exotic_geomorphology": exotic_geomorphology,
                "depressions": depressions,
                "tides": tides,
            }
        )
        world.setdefault("coupling_summary", {})["advanced_planetary_layers"] = {
            "enabled": True,
            "precipitating_species": list(volatile.metadata.get("active_precipitating_species", [])),
            "photochemical_products": sorted(volatile.photochemical_products),
            "geodynamic_regime": geodynamics.regime,
            "silicate_regime": geodynamics.silicate_regime,
            "cryogenic_regime": geodynamics.cryogenic_regime,
            "ocean_class": None if exotic_ocean is None else exotic_ocean.ocean_class,
            "active_geomorphic_fluid": geomorphic_params.active_fluid,
            "terminal_watershed_count": int(world["hydrology"].metadata.get("watersheds", {}).get("outlet_basin_count", 0)),
            "endorheic_depression_count": int(depressions.metadata.get("endorheic_depression_count", 0)),
            "tidal_constituent_count": int(tides.constituent_count),
        }
        return world

    def _fluid_aware_landscape_recouple(self, world: dict[str, Any]) -> dict[str, Any]:
        """Feed exotic-fluid properties and tides into real erosion/deposition."""
        if world.get("surface_liquids") is None:
            return world
        params = world["geomorphic_fluid_parameters"]
        # One final surface pass is intentionally enough at default resolution; the
        # dynamic liquid pipeline itself internally re-solves bed → sea level → ocean
        # → climate after that pass. Maximal configs may increase surface iterations.
        iterations = max(1, int(getattr(self.cfg.hydrology, "fluid_aware_surface_iterations", max(1, self.cfg.hydrology.surface_evolution_iterations // 2))))
        scaled = scaled_hydrology_config(self.cfg.hydrology, params, iterations=iterations)
        self._inject_tidal_reworking(world["ocean"], world.get("tides"))
        before = np.asarray(world["terrain"].elevation_km, dtype=float).copy()
        world = self._equilibrate_surface_liquids(world, hydrology_cfg=scaled)
        dz = (np.asarray(world["terrain"].elevation_km, dtype=float) - before) * 1000.0
        world.setdefault("coupling_summary", {})["fluid_aware_landscape"] = {
            "enabled": True,
            "active_fluid": params.active_fluid,
            "surface_iterations": iterations,
            "stream_power_multiplier": float(params.stream_power_multiplier),
            "deposition_multiplier": float(params.deposition_multiplier),
            "max_absolute_surface_change_m": float(np.max(np.abs(dz))) if dz.size else 0.0,
            "mean_absolute_surface_change_m": float(np.mean(np.abs(dz))) if dz.size else 0.0,
            "tidal_reworking_coupled": world.get("tides") is not None,
        }
        return world

    def generate(self, out_dir: str | Path | None = None) -> dict[str, Any]:
        world = super().generate(out_dir=None)
        if self._advanced_chemistry_enabled(self.cfg):
            world = self._build_exotic_layers(world, suffix="pre_feedback")
            world = self._fluid_aware_landscape_recouple(world)
            # Recompute chemistry/ocean/geodynamics/cryo/tides/depressions from the
            # physically changed final surface rather than retaining pre-feedback maps.
            world = self._build_exotic_layers(world, suffix="final")

        # Deep seabed relief must never leak through kilometres of water in the
        # visible-light true-color raster. Apply this correction to all worlds.
        world["appearance"] = attenuate_deep_bathymetry(
            world["appearance"], world["terrain"], world["ocean"], world["weather"], self.cfg.appearance
        )
        if out_dir is not None:
            self._stage("output", lambda: self.save(world, Path(out_dir)))
        return world

    def _array_export(self, world: dict[str, Any]) -> dict[str, np.ndarray]:
        arrays = super()._array_export(world)
        h = world["hydrology"]
        optional = {
            "basin_id": "basin_id",
            "subbasin_level_1": "subbasin_level_1",
            "subbasin_level_2": "subbasin_level_2",
            "subbasin_level_3": "subbasin_level_3",
            "channel_class": "channel_class",
            "exorheic": "exorheic",
            "distance_to_outlet_km": "distance_to_outlet_km",
            "topographic_wetness_index": "topographic_wetness_index",
            "height_above_nearest_drainage_m": "height_above_nearest_drainage_m",
            "surface_runoff_mm_year": "surface_runoff_mm_year",
            "baseflow_mm_year": "baseflow_mm_year",
            "groundwater_recharge_mm_year": "groundwater_recharge_mm_year",
            "actual_evapotranspiration_mm_year": "actual_evapotranspiration_mm_year",
            "soil_water_storage_mm": "soil_water_storage_mm",
            "groundwater_storage_mm": "groundwater_storage_mm",
            "snowpack_mm": "snowpack_mm",
            "storminess_index": "storminess_index",
            "bankfull_discharge_index": "bankfull_discharge_index",
            "subgrid_drainage_density_km_per_km2": "subgrid_drainage_density_km_per_km2",
        }
        for key, attr in optional.items():
            if hasattr(h, attr):
                arrays[key] = np.asarray(getattr(h, attr))
        dep = world.get("depressions")
        if dep is not None:
            arrays.update({
                "depression_id": np.asarray(dep.depression_id),
                "depression_depth_m": np.asarray(dep.depression_depth_m),
                "depression_storage_capacity_m3": np.asarray(dep.depression_storage_capacity_m3),
                "endorheic_depression": np.asarray(dep.endorheic_depression),
                "seasonally_inundated_depression": np.asarray(dep.seasonally_inundated),
            })
        tides = world.get("tides")
        if tides is not None:
            arrays.update({
                "equilibrium_tide_amplitude_m": np.asarray(tides.equilibrium_tide_amplitude_m),
                "tidal_range_m": np.asarray(tides.tidal_range_m),
                "tidal_current_index": np.asarray(tides.tidal_current_index),
                "intertidal_potential": np.asarray(tides.intertidal_potential),
            })
        return arrays

    def _json_export(self, world: dict[str, Any]) -> dict[str, Any]:
        payload = super()._json_export(world)
        volatile = world.get("volatile_cycle")
        if volatile is not None:
            payload["volatile_cycle"] = volatile.to_dict()
        exotic = world.get("exotic_ocean")
        if exotic is not None:
            payload["exotic_ocean"] = exotic.to_dict()
        geo = world.get("geodynamics")
        if geo is not None:
            payload["geodynamics"] = geo.to_dict()
        cryo = world.get("cryogeology")
        if cryo is not None:
            payload["cryogeology"] = cryo.to_dict()
        params = world.get("geomorphic_fluid_parameters")
        if params is not None:
            payload["geomorphic_fluid_parameters"] = params.to_dict()
        exogeo = world.get("exotic_geomorphology")
        if exogeo is not None:
            payload["exotic_geomorphology"] = exogeo.to_dict()
        dep = world.get("depressions")
        if dep is not None:
            payload["depressions"] = dep.to_dict()
        tides = world.get("tides")
        if tides is not None:
            payload["tides"] = tides.to_dict()
        payload["planetary_chemistry_database"] = chemistry_metadata()
        return payload

    def save(self, world: dict[str, Any], out: Path) -> None:
        super().save(world, out)
        out.mkdir(parents=True, exist_ok=True)
        if self.cfg.output.save_png:
            render_advanced_physical_maps(out, world, dpi=int(self.cfg.output.map_dpi))

        volatile = world.get("volatile_cycle")
        if volatile is None:
            return
        summary = {
            "volatile_cycle": volatile.to_dict(),
            "exotic_ocean": None if world.get("exotic_ocean") is None else world["exotic_ocean"].to_dict(),
            "geodynamics": world["geodynamics"].to_dict(),
            "cryogeology": world["cryogeology"].to_dict(),
            "geomorphic_fluid_parameters": world["geomorphic_fluid_parameters"].to_dict(),
            "exotic_geomorphology": world["exotic_geomorphology"].to_dict(),
            "depressions": world["depressions"].to_dict(),
            "tides": world["tides"].to_dict(),
            "hydrology_advanced": world["hydrology"].metadata,
        }
        (out / "advanced_planetary_physics.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        arrays: dict[str, np.ndarray] = {
            "total_condensate_precipitation_mm": np.asarray(volatile.total_condensate_precipitation_mm, np.float32),
            "aerosol_optical_depth_proxy": np.asarray(volatile.aerosol_optical_depth_proxy, np.float32),
            "aerosol_deposition_index": np.asarray(volatile.aerosol_deposition_index, np.float32),
            "cryovolcanism_index": np.asarray(world["cryogeology"].cryovolcanism_index, np.float32),
            "ice_shell_thickness_km": np.asarray(world["cryogeology"].ice_shell_thickness_km, np.float32),
            "plume_venting_index": np.asarray(world["cryogeology"].plume_venting_index, np.float32),
            "fluid_erosion_potential": np.asarray(world["exotic_geomorphology"].fluid_erosion_potential, np.float32),
            "fluid_deposition_potential": np.asarray(world["exotic_geomorphology"].fluid_deposition_potential, np.float32),
            "evaporite_deposition_index": np.asarray(world["exotic_geomorphology"].evaporite_deposition_index, np.float32),
            "organic_sediment_deposition_index": np.asarray(world["exotic_geomorphology"].organic_sediment_deposition_index, np.float32),
            "depression_depth_m": np.asarray(world["depressions"].depression_depth_m, np.float32),
            "endorheic_depression": np.asarray(world["depressions"].endorheic_depression, np.uint8),
            "tidal_range_m": np.asarray(world["tides"].tidal_range_m, np.float32),
            "tidal_current_index": np.asarray(world["tides"].tidal_current_index, np.float32),
            "intertidal_potential": np.asarray(world["tides"].intertidal_potential, np.float32),
        }
        if world.get("exotic_ocean") is not None:
            eo = world["exotic_ocean"]
            arrays.update({
                "mixed_layer_depth_m": np.asarray(eo.mixed_layer_depth_m, np.float32),
                "ocean_stratification_index": np.asarray(eo.stratification_index, np.float32),
                "sea_ice_fraction": np.asarray(eo.sea_ice_fraction, np.float32),
                "clathrate_stability_index": np.asarray(eo.clathrate_stability_index, np.float32),
            })
        for key, cyc in volatile.species.items():
            arrays[f"precip_{key}_mm"] = np.asarray(cyc.annual_precipitation_mm_equivalent, np.float32)
            arrays[f"frost_{key}"] = np.asarray(cyc.frost_deposition_index, np.float32)
        for key, arr in volatile.photochemical_deposition_by_species.items():
            arrays[f"photodeposition_{key}"] = np.asarray(arr, np.float32)
        saver = np.savez_compressed if self.cfg.output.compress_npz else np.savez
        saver(out / "advanced_planetary_fields.npz", **arrays)

    def _report(self, world: dict[str, Any]) -> str:
        report = super()._report(world).rstrip()
        volatile = world.get("volatile_cycle")
        if volatile is None:
            return report + "\n"
        geo = world["geodynamics"]
        cryo = world["cryogeology"]
        params = world["geomorphic_fluid_parameters"]
        exotic = world.get("exotic_ocean")
        hydro = world["hydrology"]
        tides = world["tides"]
        depressions = world["depressions"]
        lines = [
            report, "", "## Advanced planetary chemistry, hydrology and geophysics", "",
            f"- Simultaneous precipitating condensates: {', '.join(volatile.metadata.get('active_precipitating_species', [])) or 'none'}.",
            f"- Irradiation-driven products: {', '.join(sorted(volatile.photochemical_products)) or 'none'}.",
            f"- Automatic geodynamic regime: {geo.regime} (silicate={geo.silicate_regime}, cryogenic={geo.cryogenic_regime}).",
            f"- Internal heat flux: {geo.internal_heat_flux_w_m2:.5g} W/m²; tidal-heating fraction: {geo.tidal_fraction:.3f}.",
            f"- Cryogeology: mean shell {float(np.mean(cryo.ice_shell_thickness_km)):.3f} km; max cryovolcanism index {float(np.max(cryo.cryovolcanism_index)):.3f}.",
            f"- Active geomorphic fluid: {params.active_fluid}; stream-power multiplier {params.stream_power_multiplier:.3f}; deposition multiplier {params.deposition_multiplier:.3f}.",
            f"- Terminal drainage basins: {int(hydro.metadata.get('watersheds', {}).get('outlet_basin_count', 0))}; max resolved Strahler order: {int(hydro.metadata.get('max_strahler_order_all_resolved_channels', hydro.metadata.get('max_strahler_order', 0)))}.",
            f"- Mean sub-grid drainage density: {float(hydro.metadata.get('mean_subgrid_drainage_density_km_per_km2_land', 0.0)):.3f} km/km².",
            f"- Depression storage: {depressions.metadata.get('depression_count', 0)} depressions, {depressions.metadata.get('endorheic_depression_count', 0)} climatically endorheic.",
            f"- Tide system: {tides.constituent_count} constituents; screened maximum tidal range {float(np.max(tides.tidal_range_m)):.3f} m.",
        ]
        if exotic is not None:
            lines.append(
                f"- Exotic ocean: {exotic.ocean_class}; density {exotic.bulk_density_kg_m3:.2f} kg/m³; viscosity {exotic.dynamic_viscosity_mpa_s:.3f} mPa·s; effective freezing {exotic.effective_freezing_temperature_k:.2f} K."
            )
        lines.append(
            "- Global advanced models remain reduced-order: local Richards groundwater, dynamic lake spill routing, primitive-equation tides/oceans, grain-resolved sediment transport and full radiative-transfer chemistry are higher-fidelity refinement backends rather than silently implied precision."
        )
        return "\n".join(lines) + "\n"


__all__ = ["WorldPipeline"]
