from __future__ import annotations

"""Composition-aware post-coupling surface-reservoir equilibration.

This module subclasses the adaptive pipeline without disturbing its legacy execution
path.  Worlds using the composition greenhouse model and a non-empty
``astronomy.surface_volatiles`` mapping receive an additional fixed-point correction
that partitions volatile inventory between vapor/solid/liquid and solves the global
liquid equipotential from volume.

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
from .surface_liquids import SurfaceLiquidResult, place_partitioned_liquids, solve_surface_liquids

# Approximate mass of Earth's oceans.  The reference is deliberately a named unit so
# configs can express 0.1 ocean masses, 3 ocean masses, etc. without unwieldy values.
EARTH_OCEAN_MASS_KG = 1.3321e21


class WorldPipeline(_AdaptiveWorldPipeline):
    """Adaptive pipeline plus optional inventory-controlled volatile sea level."""

    @staticmethod
    def _surface_liquids_enabled(cfg) -> bool:
        return bool(
            (bool(getattr(getattr(cfg, "atmogen", None), "enabled", False))
             or getattr(cfg.astronomy, "greenhouse_model", "legacy") == "composition")
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
        """Keep solved currents/SST but make ocean geometry obey conserved liquid volume."""
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

    def _equilibrate_surface_liquids(self, world: dict[str, Any]) -> dict[str, Any]:
        c = self.cfg
        grid = world["grid"]
        astro = world["astronomy"]
        tect = world["tectonics"]
        climate = world["climate"]
        terrain0 = world["terrain"]
        static_noise = None
        # The public world intentionally does not retain the internal static-noise
        # cache.  Rebuilt advanced stages therefore use their deterministic named RNG
        # streams and the same config rather than holding a large extra cache alive.

        inventories = self._volatile_inventory_kg(c)
        if not inventories:
            return world

        # The final terrain from the ordinary coupled solve is treated as the solid
        # bed datum.  Subsequent liquid-level shifts never destroy this datum, so
        # repeated thermal iterations cannot compound a uniform sea-level offset.
        bed_datum_km = np.asarray(terrain0.elevation_km, dtype=np.float64).copy()
        pressure_bar = float(astro.atmosphere["surface_pressure_bar"])
        gravity = float(astro.planet["surface_gravity_m_s2"])
        backend = str(getattr(c.astronomy, "thermodynamics_backend", "auto"))

        liquids: SurfaceLiquidResult | None = None
        terrain = terrain0
        ocean = world["ocean"]
        previous_level: float | None = None
        history: list[dict[str, float | int]] = []
        max_iterations = 4
        tolerance_m = 2.0

        for iteration in range(1, max_iterations + 1):
            atmogen_result = world.get("atmogen")
            liquids = self._stage(
                f"surface_liquid_equilibrium_{iteration}",
                lambda cl=climate, ar=atmogen_result: place_partitioned_liquids(
                    grid, bed_datum_km, inventories, ar.surface,
                    temperature_k=float(ar.atmosphere.temperature_k[0]),
                ) if ar is not None else solve_surface_liquids(
                    grid,
                    bed_datum_km,
                    cl.annual_temperature_c,
                    inventories,
                    surface_pressure_bar=pressure_bar,
                    gravity_m_s2=gravity,
                    relative_humidity=0.65,
                    ice_fixation_efficiency=0.25,
                    thermodynamics_backend=backend,
                ),
            )
            level_change_m = (
                float("inf")
                if previous_level is None
                else abs(float(liquids.liquid_level_km) - previous_level) * 1000.0
            )
            history.append(
                {
                    "iteration": iteration,
                    "liquid_level_km": float(liquids.liquid_level_km),
                    "liquid_mass_kg": float(liquids.total_liquid_mass_kg),
                    "liquid_volume_m3": float(liquids.total_liquid_volume_m3),
                    "level_change_m": level_change_m,
                }
            )

            terrain = self._stage(
                f"terrain_surface_liquid_{iteration}",
                lambda liq=liquids: _base.rebuild_terrain_from_elevation(
                    grid,
                    tect,
                    c.terrain,
                    liq.relative_surface_elevation_km,
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
                        grid,
                        tect,
                        tr,
                        c.ocean,
                        c.terrain,
                        self.rng("ocean-stationary"),
                        cl.wind_u,
                        cl.wind_v,
                        c.noise,
                        static_noise,
                    ),
                    tr,
                    liq,
                ),
            )
            climate = self._stage(
                f"climate_surface_liquid_{iteration}",
                lambda tr=terrain, oc=ocean: _base.build_climate(
                    grid,
                    astro,
                    tr,
                    oc,
                    c.climate,
                    c.terrain,
                    self.rng("climate-stationary"),
                    c.noise,
                    static_noise,
                ),
            )

            if previous_level is not None and level_change_m <= tolerance_m:
                break
            previous_level = float(liquids.liquid_level_km)

        assert liquids is not None
        liquids.metadata["coupling_iterations"] = len(history)
        liquids.metadata["coupling_tolerance_m"] = tolerance_m
        liquids.metadata["coupling_history"] = history
        liquids.metadata["inventory_unit"] = "multiples of EARTH_OCEAN_MASS_KG"
        liquids.metadata["earth_ocean_mass_kg"] = EARTH_OCEAN_MASS_KG

        # Coastline-sensitive downstream products are recomputed once from the final
        # inventory-controlled terrain/climate state.  Tectonic history and completed
        # geomorphic erosion are retained; only their final reference elevation shifts.
        geology = self._stage(
            "geology_surface_liquid_final",
            lambda: _base.build_geology(
                grid,
                tect,
                terrain,
                ocean,
                climate,
                self.rng("geology-stationary"),
                c.noise,
                static_noise,
            ),
        )
        surface = self._surface_with_updated_elevation(world["surface_evolution"], terrain)
        hydro = self._stage(
            "hydrology_surface_liquid_final",
            lambda: _base.build_hydrology(
                grid, terrain, ocean, climate, c.hydrology, geology, surface
            ),
        )
        weather = self._stage(
            "weather_surface_liquid_final",
            lambda: _base.build_weather(
                grid,
                terrain,
                ocean,
                climate,
                hydro,
                c.weather,
                self.rng("weather"),
            ),
        )
        appearance = self._stage(
            "appearance_surface_liquid_final",
            lambda: _base.build_surface_appearance(
                grid,
                terrain,
                ocean,
                climate,
                hydro,
                geology,
                weather,
                c.appearance,
            ),
        )
        resources = self._stage(
            "resources_surface_liquid_final",
            lambda: _base.build_resources(
                grid,
                tect,
                terrain,
                ocean,
                climate,
                hydro,
                geology,
                c.resources,
                self.rng("resources"),
            ),
        )
        society = self._stage(
            "society_surface_liquid_final",
            lambda: _base.build_society(
                grid,
                terrain,
                climate,
                hydro,
                resources,
                weather,
                c.society,
                self.rng("society"),
                appearance,
            ),
        )

        world.update(
            {
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
            }
        )
        world.setdefault("coupling_summary", {})["surface_liquid_equilibrium"] = {
            "enabled": True,
            "iterations": len(history),
            "final_level_km_datum": float(liquids.liquid_level_km),
            "liquid_mass_kg": float(liquids.total_liquid_mass_kg),
            "liquid_volume_m3": float(liquids.total_liquid_volume_m3),
            "volume_residual_m3": float(liquids.volume_residual_m3),
        }
        return world

    def generate(self, out_dir: str | Path | None = None) -> dict[str, Any]:
        # Defer output until the optional volatile correction is complete, otherwise
        # maps/NPZ would describe the pre-correction shoreline.
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
            relative_surface_elevation_km=np.asarray(
                liquids.relative_surface_elevation_km, dtype=np.float32
            ),
        )
        (out / "surface_liquids.json").write_text(
            json.dumps(liquids.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _report(self, world: dict[str, Any]) -> str:
        report = super()._report(world).rstrip()
        liquids = world.get("surface_liquids")
        if liquids is None:
            return report + "\n"
        lines = [
            report,
            "",
            "## Surface volatile reservoirs",
            "",
            f"- Dynamic liquid level: {liquids.liquid_level_km:.6f} km relative to the pre-correction solid-bed datum.",
            f"- Mobile liquid mass: {liquids.total_liquid_mass_kg:.6e} kg; liquid volume: {liquids.total_liquid_volume_m3:.6e} m³.",
            f"- Raster-wedge volume closure residual: {liquids.volume_residual_m3:.6e} m³.",
            "- Volatile inventories are partitioned into saturation-limited vapor, thermally fixed solid, and mobile liquid before the global equipotential fill is solved.",
            "- Multicomponent liquids are currently volume-additive pure-species reservoirs; non-ideal mixture thermodynamics and local basin isolation are not yet solved.",
        ]
        for key, part in liquids.partitions.items():
            lines.append(
                f"- {key}: total={part.total_mass_kg:.6e} kg, vapor={part.vapor_mass_kg:.6e} kg, solid={part.solid_mass_kg:.6e} kg, liquid={part.liquid_mass_kg:.6e} kg, density={part.liquid_density_kg_m3:.3f} kg/m³."
            )
        return "\n".join(lines) + "\n"


__all__ = ["WorldPipeline", "EARTH_OCEAN_MASS_KG"]
