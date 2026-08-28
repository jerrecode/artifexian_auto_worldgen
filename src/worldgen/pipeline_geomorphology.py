from __future__ import annotations

"""Final secondary-geomorphology feedback and drainage rerouting layer."""

from pathlib import Path
from typing import Any
import numpy as np

from . import pipeline_base as _base
from .pipeline_landscape import WorldPipeline as _LandscapeWorldPipeline
from .geomorphic_fluids import scaled_hydrology_config
from .secondary_geomorphology import evolve_secondary_geomorphology
from .appearance_advanced import attenuate_deep_bathymetry
from .appearance_planetary import apply_composition_aware_appearance
from .render import _save_field, _save_power_field
from .condensate_pipeline import install_multicondensate_hydrology


class WorldPipeline(_LandscapeWorldPipeline):
    """Landscape pipeline plus active secondary surface-process feedback."""

    def _secondary_enabled(self, world: dict[str, Any]) -> bool:
        return bool(
            self._advanced_chemistry_enabled(self.cfg)
            and world.get("surface_liquids") is not None
            and world.get("tides") is not None
        )

    def _apply_secondary_feedback(self, world: dict[str, Any]) -> dict[str, Any]:
        c = self.cfg
        grid = world["grid"]
        if world.get("condensate_hydrology") is None:
            world = install_multicondensate_hydrology(
                self, world, suffix="pre_secondary_geomorphology", rebuild_dependents=True
            )
        result = self._stage(
            "secondary_geomorphology",
            lambda: evolve_secondary_geomorphology(
                grid,
                world["terrain"],
                world["ocean"],
                world["climate"],
                world["hydrology"],
                world["geology"],
                world["weather"],
                world["appearance"],
                world["tectonics"],
                c.hydrology,
                tides=world.get("tides"),
            ),
        )
        previous_surface = world["surface_evolution"]
        terrain_before = world["terrain"]
        terrain = self._stage(
            "terrain_secondary_geomorphology",
            lambda: _base.rebuild_terrain_from_elevation(
                grid,
                world["tectonics"],
                c.terrain,
                result.elevation_km,
                float(terrain_before.sea_level_offset_km),
                {
                    **terrain_before.metadata,
                    "secondary_geomorphology": result.metadata,
                    "secondary_geomorphology_active": True,
                },
            ),
        )
        erosion = (
            np.asarray(result.landslide_erosion_m, np.float32)
            + np.asarray(result.glacial_erosion_m, np.float32)
            + np.asarray(result.spring_erosion_m, np.float32)
            + np.asarray(result.karst_erosion_m, np.float32)
            + np.asarray(result.coastal_erosion_m, np.float32)
        )
        deposition = (
            np.asarray(result.glacial_deposition_m, np.float32)
            + np.asarray(result.floodplain_deposition_m, np.float32)
            + np.asarray(result.alluvial_fan_deposition_m, np.float32)
        )
        zero = np.zeros(grid.shape, np.float32)
        secondary_surface = _base.SurfaceEvolutionResult(
            result.elevation_km,
            erosion,
            deposition,
            np.asarray(np.clip(
                (erosion + deposition) / max(float(np.max(erosion + deposition)), 1.0), 0.0, 1.0
            ), np.float32),
            zero,
            zero,
            zero,
            np.asarray(getattr(previous_surface, "meander_potential", zero), np.float32),
            result.metadata,
        )
        world["terrain"] = terrain
        world["surface_evolution"] = self._combine_surface_evolution(
            previous_surface, secondary_surface, terrain.elevation_km
        )
        world["secondary_geomorphology"] = result

        # Rebuild the full liquid/climate/drainage graph. Capture breaches, fans,
        # coastal erosion and avulsion aggradation therefore influence actual receiver
        # topology rather than staying as display-only indices.  The final accepted
        # drainage state is then reinstalled from the multicomponent condensate mass
        # forcing so exotic precipitation drives the exported network as well.
        scaled = scaled_hydrology_config(
            c.hydrology,
            world["geomorphic_fluid_parameters"],
            iterations=max(1, int(c.hydrology.surface_evolution_iterations // 2)),
        )
        world = self._equilibrate_surface_liquids(world, hydrology_cfg=scaled)
        world = self._build_exotic_layers(world, suffix="post_secondary_geomorphology")
        world = install_multicondensate_hydrology(
            self,
            world,
            hydrology_cfg=scaled,
            suffix="post_secondary_geomorphology",
            rebuild_dependents=True,
        )
        world["secondary_geomorphology"] = result
        world["appearance"] = attenuate_deep_bathymetry(
            world["appearance"], world["terrain"], world["ocean"], world["weather"], c.appearance
        )
        world.setdefault("coupling_summary", {})["secondary_geomorphology"] = {
            "enabled": True,
            **result.metadata,
        }
        return world

    def generate(self, out_dir: str | Path | None = None) -> dict[str, Any]:
        world = super().generate(out_dir=None)
        if self._secondary_enabled(world):
            world = self._apply_secondary_feedback(world)
        # This must be the final visual pass: surface-liquid composition, final
        # precipitation-driven pore-liquid storage, exotic sea ice, photochemical
        # deposits, clouds and atmospheric transfer all depend on the accepted final
        # shoreline/hydrology/volatile state.
        world = apply_composition_aware_appearance(world, self.cfg)
        if out_dir is not None:
            self._stage("output", lambda: self.save(world, Path(out_dir)))
        return world

    def _array_export(self, world: dict[str, Any]) -> dict[str, np.ndarray]:
        arrays = super()._array_export(world)
        g = world.get("secondary_geomorphology")
        if g is not None:
            for name in (
                "regolith_thickness_m", "landslide_erosion_m", "glacial_erosion_m",
                "glacial_deposition_m", "spring_erosion_m", "karst_erosion_m",
                "floodplain_deposition_m", "alluvial_fan_deposition_m", "wetland_index",
                "braided_channel_index", "avulsion_potential", "estuary_index",
                "submarine_canyon_incision_m", "coastal_erosion_m",
                "river_capture_susceptibility", "isostatic_adjustment_m",
            ):
                arrays[name] = np.asarray(getattr(g, name), np.float32)
        pa = world.get("planetary_appearance")
        if pa is not None:
            arrays.update({
                "surface_liquid_true_color_rgb": np.rint(
                    np.clip(np.asarray(pa.surface_liquid_rgb, np.float32), 0.0, 1.0) * 255.0
                ).astype(np.uint8),
                "atmospheric_haze_optical_depth": np.asarray(pa.atmospheric_haze_optical_depth, np.float32),
                "ground_liquid_humidity_index": np.asarray(pa.ground_liquid_humidity_index, np.float32),
                "solid_condensate_persistence": np.asarray(pa.solid_condensate_persistence, np.float32),
            })
        return arrays

    def _json_export(self, world: dict[str, Any]) -> dict[str, Any]:
        payload = super()._json_export(world)
        g = world.get("secondary_geomorphology")
        if g is not None:
            payload["secondary_geomorphology"] = dict(g.metadata)
        pa = world.get("planetary_appearance")
        if pa is not None:
            payload["planetary_appearance"] = pa.to_dict()
        return payload

    def save(self, world: dict[str, Any], out: Path) -> None:
        super().save(world, out)
        g = world.get("secondary_geomorphology")
        if g is None or not self.cfg.output.save_png:
            return
        maps = Path(out) / "maps"
        dpi = int(self.cfg.output.map_dpi)
        products = [
            ("59_regolith_thickness.png", g.regolith_thickness_m, "Regolith Thickness (m)", "YlOrBr"),
            ("60_landslide_erosion.png", g.landslide_erosion_m, "Mass-wasting / Landslide Erosion (m)", "inferno"),
            ("61_glacial_erosion.png", g.glacial_erosion_m, "Glacial Erosion (m)", "Blues"),
            ("62_glacial_deposition.png", g.glacial_deposition_m, "Glacial Deposition (m)", "PuBuGn"),
            ("63_groundwater_spring_erosion.png", g.spring_erosion_m, "Spring / Groundwater Erosion (m)", "Blues"),
            ("64_karst_erosion.png", g.karst_erosion_m, "Karst Dissolution / Erosion (m)", "cividis"),
            ("65_floodplain_deposition.png", g.floodplain_deposition_m, "Floodplain Deposition (m)", "YlOrBr"),
            ("66_alluvial_fans.png", g.alluvial_fan_deposition_m, "Alluvial Fan Deposition (m)", "YlOrBr"),
            ("67_submarine_canyons.png", g.submarine_canyon_incision_m, "Submarine Canyon Incision (m)", "Blues"),
            ("68_coastal_erosion.png", g.coastal_erosion_m, "Coastal Erosion (m)", "inferno"),
        ]
        for filename, field, title, cmap in products:
            _save_power_field(maps / filename, field, title, cmap, gamma=0.5, dpi=dpi)
        _save_field(maps / "69_wetlands.png", g.wetland_index, "Wetland Potential", "YlGnBu", vmin=0, vmax=1, dpi=dpi)
        _save_field(maps / "70_braided_channels.png", g.braided_channel_index, "Braided-channel Regime Index", "magma", vmin=0, vmax=1, dpi=dpi)
        _save_field(maps / "71_avulsion_potential.png", g.avulsion_potential, "River Avulsion Potential", "magma", vmin=0, vmax=1, dpi=dpi)
        _save_field(maps / "72_estuaries.png", g.estuary_index, "Estuarine Tidal Influence", "viridis", vmin=0, vmax=1, dpi=dpi)
        _save_field(maps / "73_river_capture.png", g.river_capture_susceptibility, "River-capture Susceptibility", "magma", vmin=0, vmax=1, dpi=dpi)
        _save_field(maps / "74_isostatic_adjustment.png", g.isostatic_adjustment_m, "Isostatic Rebound / Subsidence (m)", "coolwarm", dpi=dpi)

    def _report(self, world: dict[str, Any]) -> str:
        report = super()._report(world).rstrip()
        g = world.get("secondary_geomorphology")
        if g is None:
            return report + "\n"
        m = g.metadata
        return report + "\n\n## Secondary geomorphic processes\n\n" + (
            f"- Coupled active processes: regolith/weathering, mass wasting, glacial erosion/deposition, groundwater/spring erosion, karst, floodplains, alluvial fans, wetlands, braided channels, avulsion, estuaries, submarine canyons, coastal erosion, river capture and isostatic response.\n"
            f"- Maximum landslide erosion {m['max_landslide_erosion_m']:.2f} m; glacial erosion {m['max_glacial_erosion_m']:.2f} m; coastal erosion {m['max_coastal_erosion_m']:.2f} m.\n"
            f"- Capture-breach candidate cells: {m['river_capture_breach_cells']}; final receiver topology is recomputed after the terrain modifications.\n"
            f"- Wetland fraction of land (index > 0.45): {100.0*m['wetland_area_fraction_land']:.2f}%.\n"
        )


__all__ = ["WorldPipeline"]
