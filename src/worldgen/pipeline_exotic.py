from __future__ import annotations

"""Canonical advanced planetary pipeline extension.

This subclass keeps the existing fast/adaptive/surface-liquid hierarchy intact and
adds chemistry, multicomponent volatile cycling, exotic-ocean state, automatic
geodynamic regime selection, cryogeology and fluid-aware geomorphic diagnostics only
for composition-aware worlds. Legacy generation remains byte-for-byte on the older
path except for output ordering through this subclass.
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
    build_exotic_geomorphology,
)
from .planetary_chemistry import chemistry_metadata


class WorldPipeline(_LiquidWorldPipeline):
    """Surface-liquid pipeline plus advanced exotic planetary process layers."""

    @staticmethod
    def _advanced_chemistry_enabled(cfg) -> bool:
        return str(getattr(cfg.astronomy, "greenhouse_model", "legacy")) == "composition"

    def _build_exotic_layers(self, world: dict[str, Any]) -> dict[str, Any]:
        c = self.cfg
        grid = world["grid"]
        astro = world["astronomy"]
        climate = world["climate"]
        terrain = world["terrain"]
        tect = world["tectonics"]
        liquids = world.get("surface_liquids")

        volatile = self._stage(
            "volatile_cycle_multicomponent",
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
                "exotic_ocean_state",
                lambda: build_exotic_ocean(
                    grid, astro, climate, world["ocean"], liquids, volatile
                ),
            )

        geodynamics = self._stage(
            "automatic_geodynamic_regime",
            lambda: build_geodynamic_regime(
                astro, c.tectonics, climate=climate, exotic_ocean=exotic_ocean
            ),
        )
        cryogeology = self._stage(
            "cryogeology",
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
            "geomorphic_fluid_parameters",
            lambda: build_geomorphic_fluid_parameters(
                astro, exotic_ocean, volatile, cryogeology
            ),
        )
        exotic_geomorphology = self._stage(
            "exotic_geomorphology",
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
        }
        return world

    def generate(self, out_dir: str | Path | None = None) -> dict[str, Any]:
        world = super().generate(out_dir=None)
        if self._advanced_chemistry_enabled(self.cfg):
            world = self._build_exotic_layers(world)
        if out_dir is not None:
            self._stage("output", lambda: self.save(world, Path(out_dir)))
        return world

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
        payload["planetary_chemistry_database"] = chemistry_metadata()
        return payload

    def save(self, world: dict[str, Any], out: Path) -> None:
        super().save(world, out)
        volatile = world.get("volatile_cycle")
        if volatile is None:
            return
        out.mkdir(parents=True, exist_ok=True)
        summary = {
            "volatile_cycle": volatile.to_dict(),
            "exotic_ocean": None if world.get("exotic_ocean") is None else world["exotic_ocean"].to_dict(),
            "geodynamics": world["geodynamics"].to_dict(),
            "cryogeology": world["cryogeology"].to_dict(),
            "geomorphic_fluid_parameters": world["geomorphic_fluid_parameters"].to_dict(),
            "exotic_geomorphology": world["exotic_geomorphology"].to_dict(),
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
            "evaporite_deposition_index": np.asarray(world["exotic_geomorphology"].evaporite_deposition_index, np.float32),
            "organic_sediment_deposition_index": np.asarray(world["exotic_geomorphology"].organic_sediment_deposition_index, np.float32),
        }
        if world.get("exotic_ocean") is not None:
            eo = world["exotic_ocean"]
            arrays.update(
                {
                    "mixed_layer_depth_m": np.asarray(eo.mixed_layer_depth_m, np.float32),
                    "ocean_stratification_index": np.asarray(eo.stratification_index, np.float32),
                    "sea_ice_fraction": np.asarray(eo.sea_ice_fraction, np.float32),
                    "clathrate_stability_index": np.asarray(eo.clathrate_stability_index, np.float32),
                }
            )
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
        lines = [
            report,
            "",
            "## Advanced planetary chemistry and geophysics",
            "",
            f"- Simultaneous precipitating condensates: {', '.join(volatile.metadata.get('active_precipitating_species', [])) or 'none'}.",
            f"- Irradiation-driven products: {', '.join(sorted(volatile.photochemical_products)) or 'none'}.",
            f"- Automatic geodynamic regime: {geo.regime} (silicate={geo.silicate_regime}, cryogenic={geo.cryogenic_regime}).",
            f"- Internal heat flux: {geo.internal_heat_flux_w_m2:.5g} W/m²; tidal fraction: {geo.tidal_fraction:.3f}.",
            f"- Cryogeology: mean shell {float(np.mean(cryo.ice_shell_thickness_km)):.3f} km; max cryovolcanism index {float(np.max(cryo.cryovolcanism_index)):.3f}.",
            f"- Active geomorphic fluid: {params.active_fluid}; stream-power multiplier {params.stream_power_multiplier:.3f}; deposition multiplier {params.deposition_multiplier:.3f}.",
        ]
        if exotic is not None:
            lines.append(
                f"- Exotic ocean: {exotic.ocean_class}; density {exotic.bulk_density_kg_m3:.2f} kg/m³; viscosity {exotic.dynamic_viscosity_mpa_s:.3f} mPa·s; effective freezing {exotic.effective_freezing_temperature_k:.2f} K."
            )
        lines.append(
            "- These are reduced-order screening/coupling layers; detailed chemical kinetics, cloud microphysics, mixture fugacity, primitive-equation oceans and viscoelastic ice-shell mechanics remain higher-fidelity future backends."
        )
        return "\n".join(lines) + "\n"


__all__ = ["WorldPipeline"]
