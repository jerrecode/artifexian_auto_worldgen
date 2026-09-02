from __future__ import annotations

"""Final geologic-time landscape closure for composition-aware worlds."""

from pathlib import Path
from typing import Any
import numpy as np

from . import pipeline_base as _base
from .pipeline_exotic import WorldPipeline as _ExoticWorldPipeline
from .geomorphic_fluids import scaled_hydrology_config
from .landscape_longterm import evolve_landscape_longterm
from .appearance_advanced import attenuate_deep_bathymetry
from .render import _save_power_field
from .condensate_pipeline import install_multicondensate_hydrology


class WorldPipeline(_ExoticWorldPipeline):
    """Advanced planet pipeline plus implicit geologic-timescale fluvial maturation."""

    def _longterm_landscape_enabled(self, world: dict[str, Any]) -> bool:
        return bool(
            self._advanced_chemistry_enabled(self.cfg)
            and world.get("surface_liquids") is not None
            and world.get("geomorphic_fluid_parameters") is not None
        )

    def _apply_longterm_landscape(self, world: dict[str, Any]) -> dict[str, Any]:
        c = self.cfg
        grid = world["grid"]
        # The geologic solver must see the same multicomponent condensate forcing as
        # the final exported drainage graph.  This makes methane/ethane/ammonia/etc.
        # runoff affect actual incision instead of being a post-hoc diagnostic.
        world = install_multicondensate_hydrology(
            self, world, suffix="pre_geologic_time", rebuild_dependents=False
        )
        terrain = world["terrain"]
        previous_surface = world["surface_evolution"]
        params = world["geomorphic_fluid_parameters"]
        scaled_cfg = scaled_hydrology_config(
            c.hydrology,
            params,
            iterations=max(1, int(c.hydrology.surface_evolution_iterations // 2)),
        )
        elapsed_myr = float(max(c.resolution.history_myr, c.resolution.history_step_myr))
        result = self._stage(
            "landscape_geologic_time_implicit",
            lambda: evolve_landscape_longterm(
                grid,
                terrain,
                world["ocean"],
                world["hydrology"],
                world["geology"],
                world["tectonics"],
                scaled_cfg,
                params,
                elapsed_myr=elapsed_myr,
            ),
        )
        evolved_terrain = self._stage(
            "terrain_geologic_time_implicit",
            lambda: _base.rebuild_terrain_from_elevation(
                grid,
                world["tectonics"],
                c.terrain,
                result.elevation_km,
                float(terrain.sea_level_offset_km),
                {
                    **terrain.metadata,
                    "geologic_time_landscape_evolution": True,
                    "geologic_time_landscape": result.metadata,
                },
            ),
        )
        zero = np.zeros(grid.shape, dtype=np.float32)
        incremental_surface = _base.SurfaceEvolutionResult(
            result.elevation_km,
            (np.asarray(result.channel_incision_m, np.float32) + 0.34 * np.asarray(result.valley_widening_m, np.float32)),
            (np.asarray(result.routed_deposition_m, np.float32) + np.asarray(result.delta_deposition_m, np.float32)),
            np.asarray(np.clip(result.channel_incision_m / max(float(np.max(result.channel_incision_m)), 1.0), 0.0, 1.0), np.float32),
            np.asarray(result.delta_deposition_m, np.float32),
            zero,
            zero,
            np.asarray(getattr(previous_surface, "meander_potential", zero), np.float32),
            result.metadata,
        )
        world["terrain"] = evolved_terrain
        world["surface_evolution"] = self._combine_surface_evolution(
            previous_surface, incremental_surface, evolved_terrain.elevation_km
        )
        world["longterm_landscape"] = result

        # Any large geologic incision changes basin capacity and coastline. Re-close
        # volatile volume, ocean and climate on the changed bed, then reinstall the
        # species-aware hydrology before dependent outputs are accepted as canonical.
        world = self._equilibrate_surface_liquids(world, hydrology_cfg=scaled_cfg)
        world = self._build_exotic_layers(world, suffix="post_geologic_time")
        world = install_multicondensate_hydrology(
            self,
            world,
            hydrology_cfg=scaled_cfg,
            suffix="post_geologic_time",
            rebuild_dependents=True,
        )
        world["appearance"] = attenuate_deep_bathymetry(
            world["appearance"], world["terrain"], world["ocean"], world["weather"], c.appearance
        )
        world.setdefault("coupling_summary", {})["geologic_time_landscape"] = {
            "enabled": True,
            **result.metadata,
        }
        return world

    def generate(self, out_dir: str | Path | None = None) -> dict[str, Any]:
        world = super().generate(out_dir=None)
        if self._longterm_landscape_enabled(world):
            world = self._apply_longterm_landscape(world)
        if out_dir is not None:
            self._stage("output", lambda: self.save(world, Path(out_dir)))
        return world

    def _array_export(self, world: dict[str, Any]) -> dict[str, np.ndarray]:
        arrays = super()._array_export(world)
        lt = world.get("longterm_landscape")
        if lt is not None:
            arrays.update({
                "longterm_channel_incision_m": np.asarray(lt.channel_incision_m, np.float32),
                "longterm_routed_deposition_m": np.asarray(lt.routed_deposition_m, np.float32),
                "longterm_delta_deposition_m": np.asarray(lt.delta_deposition_m, np.float32),
                "longterm_valley_widening_m": np.asarray(lt.valley_widening_m, np.float32),
            })
        forcing = world.get("condensate_hydrology")
        if forcing is not None:
            arrays.update({
                "condensate_precipitation_depth_mm_monthly": np.asarray(forcing.monthly_total_precipitation_depth_mm, np.float32),
                "condensate_liquid_input_mm_monthly": np.asarray(forcing.monthly_liquid_input_mm, np.float32),
                "condensate_solid_input_mm_monthly": np.asarray(forcing.monthly_solid_input_mm, np.float32),
                "condensate_thaw_fraction_monthly": np.asarray(forcing.monthly_thaw_fraction, np.float32),
            })
            for key, value in forcing.species_monthly_mass_kg_m2.items():
                arrays[f"condensate_mass_{key}_kg_m2_monthly"] = np.asarray(value, np.float32)
        return arrays

    def _json_export(self, world: dict[str, Any]) -> dict[str, Any]:
        payload = super()._json_export(world)
        lt = world.get("longterm_landscape")
        if lt is not None:
            payload["longterm_landscape"] = dict(lt.metadata)
        forcing = world.get("condensate_hydrology")
        if forcing is not None:
            payload["condensate_hydrology"] = forcing.to_dict()
        return payload

    def save(self, world: dict[str, Any], out: Path) -> None:
        super().save(world, out)
        lt = world.get("longterm_landscape")
        if lt is not None and self.cfg.output.save_png:
            maps = Path(out) / "maps"
            _save_power_field(maps / "56_longterm_channel_incision.png", lt.channel_incision_m,
                              "Geologic-time Channel Incision (m)", "inferno", gamma=0.45, dpi=int(self.cfg.output.map_dpi))
            _save_power_field(maps / "57_longterm_valley_widening.png", lt.valley_widening_m,
                              "Geologic-time Valley Widening / Relief Relaxation (m)", "inferno", gamma=0.45, dpi=int(self.cfg.output.map_dpi))
            _save_power_field(maps / "58_longterm_sediment_deposition.png", lt.routed_deposition_m + lt.delta_deposition_m,
                              "Geologic-time Routed Sediment Deposition (m)", "YlOrBr", gamma=0.45, dpi=int(self.cfg.output.map_dpi))

    def _report(self, world: dict[str, Any]) -> str:
        report = super()._report(world).rstrip()
        lt = world.get("longterm_landscape")
        if lt is None:
            return report + "\n"
        m = lt.metadata
        forcing = world.get("condensate_hydrology")
        species = ""
        if forcing is not None:
            species = ", ".join(forcing.metadata.get("active_hydrologic_species", []))
        return report + "\n\n## Geologic-time landscape evolution\n\n" + (
            f"- Implicit fluvial profile relaxation elapsed time: {m['elapsed_myr']:.1f} Myr.\n"
            f"- Mean land channel incision: {m['mean_land_channel_incision_m']:.2f} m; maximum {m['max_channel_incision_m']:.2f} m.\n"
            f"- Hydrologically active condensates driving the mature drainage network: {species or 'reference condensable only'}.\n"
            f"- The long-term solve routes generated sediment conservatively across complete outlet paths, then the volatile-volume/ocean/climate/hydrology system is re-equilibrated on the changed bed.\n"
            f"- This is a stable reduced-order geologic backend; transient knickpoint PDEs, flexural isostasy and grain-resolved abrasion remain separate higher-fidelity extensions.\n"
        )


__all__ = ["WorldPipeline"]
