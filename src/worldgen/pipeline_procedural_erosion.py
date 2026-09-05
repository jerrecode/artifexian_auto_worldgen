from __future__ import annotations

"""Final environment-conditioned procedural erosion and recoupling layer."""

from pathlib import Path
from typing import Any

import numpy as np

from . import pipeline_base as _base
from .appearance_advanced import attenuate_deep_bathymetry
from .appearance_planetary import apply_composition_aware_appearance
from .atmogen_coupler import solve_representative_columns
from .condensate_pipeline import install_multicondensate_hydrology
from .erosion_forcing import build_erosion_forcing
from .pipeline_geomorphology import WorldPipeline as _GeomorphologyWorldPipeline
from .procedural_erosion import apply_procedural_erosion
from .render import _save_field, _save_power_field


class WorldPipeline(_GeomorphologyWorldPipeline):
    """Add deterministic sub-grid morphology to the accepted physical surface."""

    def _recouple_without_dynamic_liquids(self, world: dict[str, Any]) -> dict[str, Any]:
        c = self.cfg
        grid = world["grid"]
        tect = world["tectonics"]
        terrain = world["terrain"]
        previous_climate = world["climate"]

        ocean = self._stage(
            "ocean_post_procedural_erosion",
            lambda: _base.build_ocean(
                grid,
                tect,
                terrain,
                c.ocean,
                c.terrain,
                self.rng("ocean-stationary"),
                previous_climate.wind_u,
                previous_climate.wind_v,
                c.noise,
                None,
            ),
        )
        climate = self._stage(
            "climate_post_procedural_erosion",
            lambda: _base.build_climate(
                grid,
                world["astronomy"],
                terrain,
                ocean,
                c.climate,
                c.terrain,
                self.rng("climate-stationary"),
                c.noise,
                None,
            ),
        )
        if bool(c.atmogen.enabled) and bool(c.atmogen.representative_columns_enabled):
            columns = self._stage(
                "atmogen_columns_post_procedural_erosion",
                lambda: solve_representative_columns(
                    grid=grid,
                    astronomy_result=world["astronomy"],
                    climate_result=climate,
                    world_config=c,
                    terrain_result=terrain,
                ),
            )
            climate = self._stage(
                "climate_post_procedural_erosion_atmogen_corrected",
                lambda: _base.build_climate(
                    grid,
                    world["astronomy"],
                    terrain,
                    ocean,
                    c.climate,
                    c.terrain,
                    self.rng("climate-stationary"),
                    c.noise,
                    None,
                    temperature_correction=columns.temperature_correction_c,
                ),
            )
            world["atmogen_representative_columns"] = {
                "diagnostics": dict(columns.diagnostics),
                "summaries": columns.summaries,
            }

        geology = self._stage(
            "geology_post_procedural_erosion",
            lambda: _base.build_geology(
                grid,
                tect,
                terrain,
                ocean,
                climate,
                self.rng("geology-stationary"),
                c.noise,
                None,
            ),
        )
        hydrology = self._stage(
            "hydrology_post_procedural_erosion",
            lambda: _base.build_hydrology(
                grid,
                terrain,
                ocean,
                climate,
                c.hydrology,
                geology,
                world["surface_evolution"],
            ),
        )
        weather = self._stage(
            "weather_post_procedural_erosion",
            lambda: _base.build_weather(
                grid,
                terrain,
                ocean,
                climate,
                hydrology,
                c.weather,
                self.rng("weather"),
            ),
        )
        appearance = self._stage(
            "appearance_post_procedural_erosion",
            lambda: _base.build_surface_appearance(
                grid,
                terrain,
                ocean,
                climate,
                hydrology,
                geology,
                weather,
                c.appearance,
            ),
        )
        resources = self._stage(
            "resources_post_procedural_erosion",
            lambda: _base.build_resources(
                grid,
                tect,
                terrain,
                ocean,
                climate,
                hydrology,
                geology,
                c.resources,
                self.rng("resources"),
            ),
        )
        society = self._stage(
            "society_post_procedural_erosion",
            lambda: _base.build_society(
                grid,
                terrain,
                climate,
                hydrology,
                resources,
                weather,
                c.society,
                self.rng("society"),
                appearance,
            ),
        )
        world.update(
            {
                "ocean": ocean,
                "climate": climate,
                "geology": geology,
                "hydrology": hydrology,
                "weather": weather,
                "appearance": appearance,
                "resources": resources,
                "society": society,
            }
        )
        return world

    def _apply_procedural_erosion(self, world: dict[str, Any]) -> dict[str, Any]:
        c = self.cfg
        cfg = c.procedural_erosion
        grid = world["grid"]
        terrain_before = world["terrain"]

        # The atmogen path deliberately disables the legacy chemistry engine, but it
        # still needs the mass-conservative spatial condensate bridge for runoff,
        # freeze/thaw phase state, and local liquid transport properties. Install or
        # refresh that bridge before erosion forcing is sampled.
        if world.get("condensate_hydrology") is None:
            world = install_multicondensate_hydrology(
                self,
                world,
                suffix="pre_procedural_erosion",
                rebuild_dependents=True,
            )
            terrain_before = world["terrain"]

        forcing = self._stage(
            "procedural_erosion_forcing",
            lambda: build_erosion_forcing(
                grid,
                terrain_before,
                world["ocean"],
                world["climate"],
                world["hydrology"],
                world["geology"],
                world["astronomy"],
                cfg,
                condensate_hydrology=world.get("condensate_hydrology"),
                cryogeology=world.get("cryogeology"),
                surface_liquids=world.get("surface_liquids"),
            ),
        )
        result = self._stage(
            "procedural_erosion",
            lambda: apply_procedural_erosion(
                grid,
                terrain_before,
                forcing,
                cfg,
                seed=int(c.seed) ^ 0x6E624EB7,
            ),
        )
        delta_m = np.asarray(result.delta_height_m, dtype=np.float64)
        elevation = np.asarray(terrain_before.elevation_km, dtype=np.float64) + delta_m / 1000.0
        terrain = self._stage(
            "terrain_post_procedural_erosion",
            lambda: _base.rebuild_terrain_from_elevation(
                grid,
                world["tectonics"],
                c.terrain,
                elevation,
                float(terrain_before.sea_level_offset_km),
                {
                    **terrain_before.metadata,
                    "procedural_erosion_active": True,
                    "procedural_erosion": result.metadata,
                    "procedural_erosion_forcing": forcing.metadata,
                },
            ),
        )

        previous_surface = world["surface_evolution"]
        zero = np.zeros(grid.shape, dtype=np.float32)
        erosion = np.maximum(-delta_m, 0.0).astype(np.float32)
        deposition = np.maximum(delta_m, 0.0).astype(np.float32)
        incremental = _base.SurfaceEvolutionResult(
            np.asarray(terrain.elevation_km, np.float32),
            erosion,
            deposition,
            np.asarray(result.phase_coherence, np.float32),
            zero,
            zero,
            zero,
            np.asarray(getattr(previous_surface, "meander_potential", zero), np.float32),
            {
                **result.metadata,
                "mass_semantics": (
                    "zero-mean procedural microrelief; positive displacement is recorded "
                    "as morphological deposition and negative displacement as incision, "
                    "not as a grain-resolved sediment-transport solve"
                ),
            },
        )
        world["terrain"] = terrain
        world["surface_evolution"] = self._combine_surface_evolution(
            previous_surface, incremental, terrain.elevation_km
        )
        world["procedural_erosion"] = result
        world["procedural_erosion_forcing"] = forcing

        changed = bool(np.any(np.abs(delta_m) > 1.0e-7))
        if changed and bool(cfg.recouple_after_canonical_pass):
            if world.get("surface_liquids") is not None:
                world = self._equilibrate_surface_liquids(world)
            else:
                world = self._recouple_without_dynamic_liquids(world)

            # Recoupling rebuilds climate/hydrology and can invalidate the phase and
            # precipitation fields sampled above. Reinstall unconditionally; the
            # installer is a no-op for worlds without an eligible volatile inventory.
            world = install_multicondensate_hydrology(
                self,
                world,
                suffix="post_procedural_erosion",
                rebuild_dependents=True,
            )
            if world.get("geomorphic_fluid_parameters") is not None:
                world = self._build_exotic_layers(world, suffix="post_procedural_erosion")

        world["procedural_erosion"] = result
        world["procedural_erosion_forcing"] = forcing
        world["appearance"] = attenuate_deep_bathymetry(
            world["appearance"], world["terrain"], world["ocean"], world["weather"], c.appearance
        )
        world = apply_composition_aware_appearance(world, c)
        world.setdefault("coupling_summary", {})["procedural_erosion"] = {
            "enabled": True,
            "recoupled": bool(changed and cfg.recouple_after_canonical_pass),
            **result.metadata,
            **forcing.metadata,
        }
        return world

    def generate(self, out_dir: str | Path | None = None) -> dict[str, Any]:
        world = super().generate(out_dir=None)
        if bool(self.cfg.procedural_erosion.enabled):
            world = self._apply_procedural_erosion(world)
        if out_dir is not None:
            self._stage("output", lambda: self.save(world, Path(out_dir)))
        return world

    def _array_export(self, world: dict[str, Any]) -> dict[str, np.ndarray]:
        arrays = super()._array_export(world)
        result = world.get("procedural_erosion")
        forcing = world.get("procedural_erosion_forcing")
        if result is not None:
            arrays.update(
                {
                    "procedural_erosion_delta_m": np.asarray(result.delta_height_m, np.float32),
                    "procedural_erosion_phase_coherence": np.asarray(result.phase_coherence, np.float32),
                    "procedural_erosion_ridge_map": np.asarray(result.ridge_map, np.float32),
                    "procedural_erosion_crease_map": np.asarray(result.crease_map, np.float32),
                    "procedural_erosion_strength": np.asarray(result.effective_strength, np.float32),
                    "procedural_erosion_scale_km": np.asarray(result.effective_scale_km, np.float32),
                }
            )
        if forcing is not None:
            arrays.update(
                {
                    "erosion_fluvial_activity": np.asarray(forcing.fluvial_activity, np.float32),
                    "erosion_pluvial_activity": np.asarray(forcing.pluvial_activity, np.float32),
                    "erosion_glacial_activity": np.asarray(forcing.glacial_activity, np.float32),
                    "erosion_marine_activity": np.asarray(forcing.marine_activity, np.float32),
                    "erosion_chemical_weathering": np.asarray(forcing.chemical_weathering, np.float32),
                    "erosion_freeze_thaw_activity": np.asarray(forcing.freeze_thaw_activity, np.float32),
                    "erosion_soil_saturation": np.asarray(forcing.soil_saturation, np.float32),
                    "erosion_fluid_mechanical_factor": np.asarray(forcing.fluid_mechanical_factor, np.float32),
                }
            )
        return arrays

    def _json_export(self, world: dict[str, Any]) -> dict[str, Any]:
        payload = super()._json_export(world)
        result = world.get("procedural_erosion")
        forcing = world.get("procedural_erosion_forcing")
        if result is not None:
            payload["procedural_erosion"] = dict(result.metadata)
        if forcing is not None:
            payload["procedural_erosion_forcing"] = dict(forcing.metadata)
        return payload

    def save(self, world: dict[str, Any], out: Path) -> None:
        super().save(world, out)
        result = world.get("procedural_erosion")
        forcing = world.get("procedural_erosion_forcing")
        if result is None or not self.cfg.output.save_png:
            return
        maps = Path(out) / "maps"
        dpi = int(self.cfg.output.map_dpi)
        _save_power_field(
            maps / "75_procedural_erosion_delta.png",
            np.abs(result.delta_height_m),
            "Procedural Erosion Absolute Displacement (m)",
            "inferno",
            gamma=0.55,
            dpi=dpi,
        )
        _save_field(maps / "76_procedural_erosion_ridges.png", result.ridge_map, "Procedural Ridge Morphology", "viridis", vmin=0, vmax=1, dpi=dpi)
        _save_field(maps / "77_procedural_erosion_creases.png", result.crease_map, "Procedural Crease Morphology", "viridis", vmin=0, vmax=1, dpi=dpi)
        _save_field(maps / "78_procedural_erosion_coherence.png", result.phase_coherence, "Procedural Phase Coherence", "viridis", vmin=0, vmax=1, dpi=dpi)
        if forcing is not None:
            _save_field(maps / "79_procedural_erosion_strength.png", forcing.strength, "Environment-conditioned Procedural Erosion Strength", "magma", dpi=dpi)

    def _report(self, world: dict[str, Any]) -> str:
        report = super()._report(world).rstrip()
        result = world.get("procedural_erosion")
        if result is None:
            return report + "\n"
        m = result.metadata
        f = world["procedural_erosion_forcing"].metadata
        return report + "\n\n## Environment-conditioned procedural erosion\n\n" + (
            f"- Seamless 3-D phase-cell octaves executed: {m['octaves_executed']} of {m['octaves_requested']}; maximum absolute displacement {m['max_absolute_displacement_m']:.3f} m.\n"
            f"- Dominant precipitating liquid for runoff/pluvial scaling: {f['dominant_condensate']}; spherical-mean precipitation/runoff multiplier {f['fluid_mechanical_factor']:.3f} (range {f['fluid_mechanical_factor_min']:.3f}–{f['fluid_mechanical_factor_max']:.3f}).\n"
            f"- Standing surface-liquid marine scaling: {f['marine_fluid_dominant_species']} via {f['marine_fluid_property_source']}; marine fluid multiplier {f['marine_fluid_mechanical_factor']:.3f}.\n"
            f"- Canonical dependency recoupling: {bool(self.cfg.procedural_erosion.recouple_after_canonical_pass)}.\n"
            "- The procedural layer supplies deterministic unresolved morphology; physical runoff, sediment routing, glacial/coastal processes and volatile mass conservation remain owned by their dedicated solvers.\n"
        )


__all__ = ["WorldPipeline"]
