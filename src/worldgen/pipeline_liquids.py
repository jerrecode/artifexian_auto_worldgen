from __future__ import annotations

"""Composition-aware post-coupling surface-reservoir equilibration.

Composition-aware worlds partition configured volatile inventories between vapor,
solid and mobile liquid and solve the global liquid equipotential from conserved
volume.  The advanced path also closes the previously missing feedback loop between
that shoreline, climate/runoff and landscape evolution: erosion/deposition is rerun
on the final liquid geometry, then the changed solid bed is re-filled by the same
conserved volatile inventory.

``surface_volatiles`` values are interpreted as multiples of one modern-Earth ocean
mass in this advanced path.  Legacy-greenhouse configurations retain their previous
unitless-selection semantics and target-land-fraction sea level exactly.
"""

from pathlib import Path
from typing import Any
import json

import numpy as np

from . import pipeline_base as _base
from .pipeline import WorldPipeline as _AdaptiveWorldPipeline
from .surface_liquids import SurfaceLiquidResult, solve_surface_liquids

EARTH_OCEAN_MASS_KG = 1.3321e21


class WorldPipeline(_AdaptiveWorldPipeline):
    """Adaptive pipeline plus inventory-controlled volatile/landscape closure."""

    @staticmethod
    def _surface_liquids_enabled(cfg) -> bool:
        return bool(
            getattr(cfg.astronomy, "greenhouse_model", "legacy") == "composition"
            and getattr(cfg.astronomy, "surface_volatiles", None)
        )

    @staticmethod
    def _volatile_inventory_kg(cfg) -> dict[str, float]:
        return {
            str(name): float(amount) * EARTH_OCEAN_MASS_KG
            for name, amount in cfg.astronomy.surface_volatiles.items()
            if float(amount) > 0.0
        }

    @staticmethod
    def _replace_ocean_geometry(ocean, terrain, liquids: SurfaceLiquidResult):
        ocean.depth_m = np.asarray(liquids.liquid_depth_m, dtype=np.float32)
        ocean.elevation_km = np.asarray(terrain.elevation_km, dtype=np.float32).copy()
        ocean.metadata = {
            **ocean.metadata,
            "surface_liquid_geometry": "inventory-controlled spherical equipotential fill",
            "surface_liquid_level_km_datum": float(liquids.liquid_level_km),
            "surface_liquid_volume_m3": float(liquids.total_liquid_volume_m3),
            "surface_liquid_mass_kg": float(liquids.total_liquid_mass_kg),
            "surface_liquid_volume_residual_m3": float(liquids.volume_residual_m3),
        }
        return ocean

    @staticmethod
    def _surface_with_updated_elevation(surface, terrain):
        return _base.SurfaceEvolutionResult(
            np.asarray(terrain.elevation_km, dtype=np.float32),
            surface.cumulative_erosion_m,
            surface.cumulative_deposition_m,
            surface.sediment_flux_index,
            surface.delta_deposition_m,
            surface.tectonic_uplift_m,
            surface.meander_migration_m,
            surface.meander_potential,
            {
                **surface.metadata,
                "surface_liquid_level_reapplied": True,
            },
        )

    @staticmethod
    def _combine_surface_evolution(previous, current, elevation_km):
        """Accumulate material history while retaining the newest physical bed."""
        return _base.SurfaceEvolutionResult(
            np.asarray(elevation_km, dtype=np.float32),
            np.asarray(previous.cumulative_erosion_m, np.float32) + np.asarray(current.cumulative_erosion_m, np.float32),
            np.asarray(previous.cumulative_deposition_m, np.float32) + np.asarray(current.cumulative_deposition_m, np.float32),
            np.maximum(np.asarray(previous.sediment_flux_index, np.float32), np.asarray(current.sediment_flux_index, np.float32)),
            np.asarray(previous.delta_deposition_m, np.float32) + np.asarray(current.delta_deposition_m, np.float32),
            np.asarray(previous.tectonic_uplift_m, np.float32) + np.asarray(current.tectonic_uplift_m, np.float32),
            np.asarray(previous.meander_migration_m, np.float32) + np.asarray(current.meander_migration_m, np.float32),
            np.asarray(current.meander_potential, np.float32),
            {
                **previous.metadata,
                "final_coupled_surface_model": current.metadata,
                "surface_liquid_landscape_recoupled": True,
            },
        )

    @staticmethod
    def _reconcile_geometry_metadata(grid, terrain, ocean, liquids: SurfaceLiquidResult):
        """Recompute geometry-derived metadata from canonical final rasters."""
        land_fraction = float(grid.weighted_fraction(terrain.land))
        ocean_fraction = float(grid.weighted_fraction(terrain.ocean))
        terrain.metadata = {
            **terrain.metadata,
            "actual_land_fraction": land_fraction,
            "actual_ocean_fraction": ocean_fraction,
            "geometry_metadata_reconciled": True,
        }
        depth = np.asarray(liquids.liquid_depth_m, dtype=float)
        wet = depth > 0.0
        if np.any(wet):
            weights = grid.cell_area_weights[wet]
            mean_depth = float(np.average(depth[wet], weights=weights))
            max_depth = float(np.max(depth[wet]))
        else:
            mean_depth = max_depth = 0.0
        ocean.metadata = {
            **ocean.metadata,
            "actual_ocean_fraction": ocean_fraction,
            "actual_land_fraction": land_fraction,
            "mean_surface_liquid_depth_m": mean_depth,
            "max_surface_liquid_depth_m": max_depth,
            "geometry_metadata_reconciled": True,
        }
        liquids.metadata["actual_land_fraction"] = land_fraction
        liquids.metadata["actual_ocean_fraction"] = ocean_fraction

    def _solve_liquid_state(self, grid, bed_datum_km, climate, inventories, pressure_bar, gravity, backend, stage_name):
        return self._stage(
            stage_name,
            lambda: solve_surface_liquids(
                grid,
                bed_datum_km,
                climate.annual_temperature_c,
                inventories,
                surface_pressure_bar=pressure_bar,
                gravity_m_s2=gravity,
                relative_humidity=0.65,
                ice_fixation_efficiency=0.25,
                thermodynamics_backend=backend,
            ),
        )

    def _equilibrate_surface_liquids(self, world: dict[str, Any], hydrology_cfg=None) -> dict[str, Any]:
        c = self.cfg
        hcfg = c.hydrology if hydrology_cfg is None else hydrology_cfg
        grid = world["grid"]
        astro = world["astronomy"]
        tect = world["tectonics"]
        climate = world["climate"]
        terrain0 = world["terrain"]
        static_noise = None

        inventories = self._volatile_inventory_kg(c)
        if not inventories:
            return world

        bed_datum_km = np.asarray(terrain0.elevation_km, dtype=np.float64).copy()
        pressure_bar = float(astro.atmosphere["surface_pressure_bar"])
        gravity = float(astro.planet["surface_gravity_m_s2"])
        backend = str(getattr(c.astronomy, "thermodynamics_backend", "auto"))

        liquids: SurfaceLiquidResult | None = None
        terrain = terrain0
        ocean = world["ocean"]
        previous_level: float | None = None
        history: list[dict[str, float | int | str]] = []
        max_iterations = 4
        tolerance_m = 2.0

        # Thermal/phase/sea-level fixed point on the current solid bed.
        for iteration in range(1, max_iterations + 1):
            liquids = self._solve_liquid_state(
                grid, bed_datum_km, climate, inventories, pressure_bar, gravity, backend,
                f"surface_liquid_equilibrium_{iteration}",
            )
            level_change_m = float("inf") if previous_level is None else abs(float(liquids.liquid_level_km) - previous_level) * 1000.0
            history.append({
                "kind": "thermal_liquid",
                "iteration": iteration,
                "liquid_level_km": float(liquids.liquid_level_km),
                "liquid_mass_kg": float(liquids.total_liquid_mass_kg),
                "liquid_volume_m3": float(liquids.total_liquid_volume_m3),
                "level_change_m": level_change_m,
            })

            terrain = self._stage(
                f"terrain_surface_liquid_{iteration}",
                lambda liq=liquids: _base.rebuild_terrain_from_elevation(
                    grid, tect, c.terrain, liq.relative_surface_elevation_km,
                    float(liq.liquid_level_km),
                    {
                        **terrain0.metadata,
                        "surface_liquid_inventory_controlled": True,
                        "surface_liquid_level_km_datum": float(liq.liquid_level_km),
                        "surface_liquid_mass_kg": float(liq.total_liquid_mass_kg),
                        "surface_liquid_volume_m3": float(liq.total_liquid_volume_m3),
                        "surface_liquid_volume_residual_m3": float(liq.volume_residual_m3),
                    },
                ),
            )
            ocean = self._stage(
                f"ocean_surface_liquid_{iteration}",
                lambda tr=terrain, liq=liquids, cl=climate: self._replace_ocean_geometry(
                    _base.build_ocean(
                        grid, tect, tr, c.ocean, c.terrain, self.rng("ocean-stationary"),
                        cl.wind_u, cl.wind_v, c.noise, static_noise,
                    ), tr, liq,
                ),
            )
            climate = self._stage(
                f"climate_surface_liquid_{iteration}",
                lambda tr=terrain, oc=ocean: _base.build_climate(
                    grid, astro, tr, oc, c.climate, c.terrain,
                    self.rng("climate-stationary"), c.noise, static_noise,
                ),
            )
            if previous_level is not None and level_change_m <= tolerance_m:
                break
            previous_level = float(liquids.liquid_level_km)

        assert liquids is not None

        # Critical closure omitted by the old implementation: the new shoreline and
        # rainfall now alter the solid bed, after which the conserved liquid inventory
        # is solved again.  One pass is the default for composition-aware worlds;
        # maximal presets can request more through the advanced hydrology config.
        surface = self._surface_with_updated_elevation(world["surface_evolution"], terrain)
        landscape_passes = max(0, int(getattr(hcfg, "surface_liquid_landscape_coupling_passes", 1)))
        landscape_history: list[dict[str, float | int]] = []
        for iteration in range(1, landscape_passes + 1):
            geology_for_surface = self._stage(
                f"geology_surface_liquid_landscape_{iteration}",
                lambda tr=terrain, oc=ocean, cl=climate: _base.build_geology(
                    grid, tect, tr, oc, cl, self.rng("geology-stationary"), c.noise, static_noise,
                ),
            )
            evolved = self._stage(
                f"surface_evolution_surface_liquid_{iteration}",
                lambda tr=terrain, oc=ocean, cl=climate, ge=geology_for_surface: _base.evolve_surface(
                    grid, tr, oc, cl, ge, hcfg, tect,
                    self.rng("surface-liquid-landscape"), c.noise, static_noise,
                ),
            )
            surface = self._combine_surface_evolution(surface, evolved, evolved.elevation_km)
            bed_datum_km = np.asarray(evolved.elevation_km, dtype=np.float64).copy()

            liquids = self._solve_liquid_state(
                grid, bed_datum_km, climate, inventories, pressure_bar, gravity, backend,
                f"surface_liquid_post_landscape_{iteration}",
            )
            terrain = self._stage(
                f"terrain_post_landscape_{iteration}",
                lambda liq=liquids, ev=evolved: _base.rebuild_terrain_from_elevation(
                    grid, tect, c.terrain, liq.relative_surface_elevation_km,
                    float(liq.liquid_level_km),
                    {
                        **terrain.metadata,
                        "post_liquid_landscape_evolution": True,
                        "surface_liquid_level_km_datum": float(liq.liquid_level_km),
                        "last_landscape_max_erosion_m": float(np.max(ev.cumulative_erosion_m)),
                    },
                ),
            )
            surface = self._surface_with_updated_elevation(surface, terrain)
            ocean = self._stage(
                f"ocean_post_landscape_{iteration}",
                lambda tr=terrain, liq=liquids, cl=climate: self._replace_ocean_geometry(
                    _base.build_ocean(
                        grid, tect, tr, c.ocean, c.terrain, self.rng("ocean-stationary"),
                        cl.wind_u, cl.wind_v, c.noise, static_noise,
                    ), tr, liq,
                ),
            )
            climate = self._stage(
                f"climate_post_landscape_{iteration}",
                lambda tr=terrain, oc=ocean: _base.build_climate(
                    grid, astro, tr, oc, c.climate, c.terrain,
                    self.rng("climate-stationary"), c.noise, static_noise,
                ),
            )
            landscape_history.append({
                "iteration": iteration,
                "liquid_level_km": float(liquids.liquid_level_km),
                "max_incremental_erosion_m": float(np.max(evolved.cumulative_erosion_m)),
                "mean_incremental_erosion_m": float(np.mean(evolved.cumulative_erosion_m[terrain.land])) if np.any(terrain.land) else 0.0,
            })

        liquids.metadata["coupling_iterations"] = len(history)
        liquids.metadata["coupling_tolerance_m"] = tolerance_m
        liquids.metadata["coupling_history"] = history
        liquids.metadata["landscape_coupling_history"] = landscape_history
        liquids.metadata["inventory_unit"] = "multiples of EARTH_OCEAN_MASS_KG"
        liquids.metadata["earth_ocean_mass_kg"] = EARTH_OCEAN_MASS_KG
        self._reconcile_geometry_metadata(grid, terrain, ocean, liquids)

        geology = self._stage(
            "geology_surface_liquid_final",
            lambda: _base.build_geology(
                grid, tect, terrain, ocean, climate,
                self.rng("geology-stationary"), c.noise, static_noise,
            ),
        )
        hydro = self._stage(
            "hydrology_surface_liquid_final",
            lambda: _base.build_hydrology(grid, terrain, ocean, climate, hcfg, geology, surface),
        )
        weather = self._stage(
            "weather_surface_liquid_final",
            lambda: _base.build_weather(
                grid, terrain, ocean, climate, hydro, c.weather, self.rng("weather"),
            ),
        )
        appearance = self._stage(
            "appearance_surface_liquid_final",
            lambda: _base.build_surface_appearance(
                grid, terrain, ocean, climate, hydro, geology, weather, c.appearance,
            ),
        )
        resources = self._stage(
            "resources_surface_liquid_final",
            lambda: _base.build_resources(
                grid, tect, terrain, ocean, climate, hydro, geology, c.resources, self.rng("resources"),
            ),
        )
        society = self._stage(
            "society_surface_liquid_final",
            lambda: _base.build_society(
                grid, terrain, climate, hydro, resources, weather, c.society,
                self.rng("society"), appearance,
            ),
        )

        world.update({
            "terrain": terrain,
            "ocean": ocean,
            "climate": climate,
            "surface_evolution": surface,
            "hydrology": hydro,
            "weather": weather,
            "geology": geology,
            "appearance": appearance,
            "resources": resources,
            "society": society,
            "surface_liquids": liquids,
        })
        world.setdefault("coupling_summary", {})["surface_liquid_equilibrium"] = {
            "enabled": True,
            "thermal_iterations": len(history),
            "landscape_coupling_passes": landscape_passes,
            "final_level_km_datum": float(liquids.liquid_level_km),
            "liquid_mass_kg": float(liquids.total_liquid_mass_kg),
            "liquid_volume_m3": float(liquids.total_liquid_volume_m3),
            "volume_residual_m3": float(liquids.volume_residual_m3),
            "actual_land_fraction": float(grid.weighted_fraction(terrain.land)),
        }
        return world

    def generate(self, out_dir: str | Path | None = None) -> dict[str, Any]:
        world = super().generate(out_dir=None)
        if self._surface_liquids_enabled(self.cfg):
            world = self._equilibrate_surface_liquids(world)
        if out_dir is not None:
            self._stage("output", lambda: self.save(world, Path(out_dir)))
        return world

    def _json_export(self, world: dict[str, Any]) -> dict[str, Any]:
        payload = super()._json_export(world)
        liquids = world.get("surface_liquids")
        if liquids is not None:
            payload["surface_liquids"] = liquids.to_dict()
        return payload

    def save(self, world: dict[str, Any], out: Path) -> None:
        super().save(world, out)
        liquids = world.get("surface_liquids")
        if liquids is None:
            return
        out.mkdir(parents=True, exist_ok=True)
        saver = np.savez_compressed if self.cfg.output.compress_npz else np.savez
        saver(
            out / "surface_liquids.npz",
            liquid_depth_m=np.asarray(liquids.liquid_depth_m, dtype=np.float32),
            liquid_mask=np.asarray(liquids.liquid_mask, dtype=np.uint8),
            relative_surface_elevation_km=np.asarray(liquids.relative_surface_elevation_km, dtype=np.float32),
        )
        (out / "surface_liquids.json").write_text(
            json.dumps(liquids.to_dict(), indent=2, sort_keys=True), encoding="utf-8",
        )

    def _report(self, world: dict[str, Any]) -> str:
        report = super()._report(world).rstrip()
        liquids = world.get("surface_liquids")
        if liquids is None:
            return report + "\n"
        lines = [
            report, "", "## Surface volatile reservoirs", "",
            f"- Dynamic liquid level: {liquids.liquid_level_km:.6f} km relative to the current solid-bed datum.",
            f"- Mobile liquid mass: {liquids.total_liquid_mass_kg:.6e} kg; liquid volume: {liquids.total_liquid_volume_m3:.6e} m³.",
            f"- Raster-wedge volume closure residual: {liquids.volume_residual_m3:.6e} m³.",
            f"- Final emergent land fraction after conserved-volume fill: {100.0 * float(world['grid'].weighted_fraction(world['terrain'].land)):.2f}%.",
            f"- Shoreline/climate/landscape closure passes: {len(liquids.metadata.get('landscape_coupling_history', []))}.",
            "- Volatile inventories are partitioned into saturation-limited vapor, thermally fixed solid, and mobile liquid before the global equipotential fill is solved.",
            "- The final shoreline is fed back through climate, runoff, erosion/deposition and then re-solved against the conserved liquid inventory.",
            "- Multicomponent liquids are currently volume-additive pure-species reservoirs; non-ideal mixture thermodynamics and local basin isolation remain higher-fidelity backends.",
        ]
        for key, part in liquids.partitions.items():
            lines.append(
                f"- {key}: total={part.total_mass_kg:.6e} kg, vapor={part.vapor_mass_kg:.6e} kg, solid={part.solid_mass_kg:.6e} kg, liquid={part.liquid_mass_kg:.6e} kg, density={part.liquid_density_kg_m3:.3f} kg/m³."
            )
        return "\n".join(lines) + "\n"


__all__ = ["WorldPipeline", "EARTH_OCEAN_MASS_KG"]
