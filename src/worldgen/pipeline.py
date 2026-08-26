from __future__ import annotations

"""Public world pipeline with optional adaptive coupled-system convergence.

The historically fixed-pass implementation is retained in :mod:`pipeline_base` and
still supplies output/export/report behavior.  This orchestration subclass changes
only coupled-pass scheduling and remains fixed-pass by default for reproducibility.
"""

from pathlib import Path
from typing import Any
import copy
import gc

import numpy as np

from . import pipeline_base as _base
from .convergence import ConvergenceThresholds, ConvergenceTracker


class WorldPipeline(_base.WorldPipeline):
    """Dependency-ordered generator with optional physical convergence stopping."""

    def _macro_schedule(self) -> tuple[bool, int, int]:
        sim = self.cfg.simulation
        adaptive = bool(sim.adaptive_convergence)
        if adaptive:
            return True, int(sim.min_earth_system_passes), int(sim.max_earth_system_passes)
        fixed = max(1, int(sim.earth_system_passes))
        return False, fixed, fixed

    def _final_schedule(self) -> tuple[bool, int, int]:
        sim = self.cfg.simulation
        adaptive = bool(sim.adaptive_final_coupling)
        if adaptive:
            return (
                True,
                int(sim.min_final_climate_ocean_passes),
                int(sim.max_final_climate_ocean_passes),
            )
        fixed = max(1, int(sim.final_climate_ocean_passes))
        return False, fixed, fixed

    @staticmethod
    def _decision_payload(decision) -> dict[str, Any]:
        return {
            "eligible": bool(decision.eligible),
            "metrics_available": bool(decision.metrics_available),
            "converged": bool(decision.converged),
            "stop": bool(decision.stop),
            "consecutive_converged": int(decision.consecutive_converged),
            "normalized_residual": float(decision.normalized_residual),
            "temperature_ratio": decision.temperature_ratio,
            "precipitation_ratio": decision.precipitation_ratio,
            "elevation_ratio": decision.elevation_ratio,
            "reason": decision.reason,
        }

    def generate(self, out_dir: str | Path | None = None) -> dict[str, Any]:
        c = self.cfg
        astro = self._stage(
            "astronomy",
            lambda: _base.build_astronomy(c.astronomy, self.rng("astronomy")),
        )
        radius_km = 6371.0 * astro.planet["radius_earth"]
        grid = _base.SphereGrid(c.resolution.width, c.resolution.height, radius_km)
        tect = self._stage(
            "tectonics",
            lambda: _base.generate_tectonics(
                grid,
                c.tectonics,
                c.resolution,
                self.rng("tectonics"),
                c.noise,
            ),
        )
        terrain = self._stage(
            "terrain_initial",
            lambda: _base.build_terrain(
                grid,
                tect,
                c.tectonics,
                c.terrain,
                self.rng("terrain"),
                c.noise,
            ),
        )

        shape = terrain.elevation_km.shape
        static_noise = self._stage(
            "noise_cache",
            lambda: _base.build_static_noise_fields(shape, c.noise, self.rng),
        )
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

        adaptive_macro, macro_min, macro_max = self._macro_schedule()
        macro_tracker = (
            ConvergenceTracker(
                ConvergenceThresholds(
                    temperature_c=float(c.simulation.convergence_temperature_c),
                    precipitation_mm_year=float(c.simulation.convergence_precip_mm_year),
                    elevation_m=float(c.simulation.convergence_elevation_m),
                    required_consecutive=int(
                        c.simulation.required_consecutive_converged_passes
                    ),
                    minimum_passes=macro_min,
                )
            )
            if adaptive_macro
            else None
        )
        executed_macro = 0
        macro_stop_reason = "fixed_pass_count" if not adaptive_macro else "max_passes"

        # In adaptive mode predictor fidelity reaches 1.0 at the configured minimum
        # pass, so convergence is never declared on a deliberately cheap predictor.
        for ipass in range(macro_max):
            pass_number = ipass + 1
            executed_macro = pass_number
            terrain_before = terrain.elevation_km.astype(float)
            prev_temp = None if climate_prev is None else climate_prev.annual_temperature_c
            prev_precip = (
                None if climate_prev is None else climate_prev.annual_precipitation_mm
            )

            if adaptive_macro:
                if pass_number >= macro_min:
                    progress = 1.0
                    full_fidelity = True
                else:
                    progress = ipass / max(macro_min - 1, 1)
                    full_fidelity = False
            else:
                full_fidelity = macro_max <= 1 or ipass == macro_max - 1
                progress = 1.0 if full_fidelity else ipass / max(macro_max - 1, 1)

            cmin = float(
                np.clip(c.simulation.intermediate_climate_fraction, 0.10, 1.0)
            )
            omin = float(
                np.clip(c.simulation.intermediate_ocean_fraction, 0.15, 1.0)
            )
            climate_fidelity = (
                1.0
                if full_fidelity
                else cmin + (1.0 - cmin) * progress**1.6
            )
            ocean_fidelity = (
                1.0
                if full_fidelity
                else omin + (1.0 - omin) * progress**1.35
            )

            wind_u = None if climate_prev is None else climate_prev.wind_u
            wind_v = None if climate_prev is None else climate_prev.wind_v
            if ipass > 0:
                ocean = None
                gc.collect()

            ocfg = copy.deepcopy(c.ocean)
            ocfg.current_iterations = max(
                10, int(round(c.ocean.current_iterations * ocean_fidelity))
            )
            ocfg.heat_transport_iterations = max(
                3, int(round(c.ocean.heat_transport_iterations * ocean_fidelity))
            )
            ocean = self._stage(
                f"ocean_pass_{pass_number}",
                lambda wu=wind_u, wv=wind_v, oc=ocfg: _base.build_ocean(
                    grid,
                    tect,
                    terrain,
                    oc,
                    c.terrain,
                    self.rng("ocean-stationary"),
                    wu,
                    wv,
                    c.noise,
                    static_noise,
                ),
            )

            wind_u = wind_v = None
            climate_prev = None
            climate = None
            gc.collect()
            ccfg = copy.deepcopy(c.climate)
            ccfg.moisture_iterations = max(
                6, int(round(c.climate.moisture_iterations * climate_fidelity))
            )
            climate = self._stage(
                f"climate_pass_{pass_number}",
                lambda cc=ccfg: _base.build_climate(
                    grid,
                    astro,
                    terrain,
                    ocean,
                    cc,
                    c.terrain,
                    self.rng("climate-stationary"),
                    c.noise,
                    static_noise,
                ),
            )
            geology = self._stage(
                f"geology_pass_{pass_number}",
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
            surf = self._stage(
                f"surface_pass_{pass_number}",
                lambda: _base.evolve_surface(
                    grid,
                    terrain,
                    ocean,
                    climate,
                    geology,
                    c.hydrology,
                    tect,
                    self.rng("surface-stationary"),
                    c.noise,
                    static_noise,
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
            if (
                not c.simulation.preserve_initial_sea_level
                and c.terrain.sea_level_mode == "target_land_fraction"
            ):
                sea_shift = grid.weighted_quantile(
                    elev, 1.0 - c.tectonics.continental_fraction_target
                )
                elev -= sea_shift
            terrain = self._stage(
                f"terrain_pass_{pass_number}",
                lambda e=elev, ss=sea_shift, ip=ipass: _base.rebuild_terrain_from_elevation(
                    grid,
                    tect,
                    c.terrain,
                    e,
                    float(ss),
                    {
                        **terrain.metadata,
                        "surface_evolved": True,
                        "earth_system_pass": ip + 1,
                    },
                ),
            )

            active = (terrain.elevation_km > 0) | (terrain_before > 0)
            elevation_change_m = (
                float(
                    np.average(
                        np.abs(
                            terrain.elevation_km[active] - terrain_before[active]
                        ),
                        weights=grid.cell_area_weights[active],
                    )
                    * 1000.0
                )
                if np.any(active)
                else 0.0
            )
            stat: dict[str, Any] = {
                "stage": f"earth_system_pass_{pass_number}",
                "pass_number": pass_number,
                "climate_fidelity": float(climate_fidelity),
                "ocean_fidelity": float(ocean_fidelity),
                "full_fidelity": bool(full_fidelity),
                "moisture_iterations": int(ccfg.moisture_iterations),
                "mean_abs_elevation_change_m": elevation_change_m,
                "max_pass_erosion_m": float(surf.cumulative_erosion_m.max()),
                "max_pass_deposition_m": float(surf.cumulative_deposition_m.max()),
                "max_pass_delta_aggradation_m": float(surf.delta_deposition_m.max()),
                "max_pass_uplift_m": float(surf.tectonic_uplift_m.max()),
                "flow_refresh_count": int(
                    surf.metadata.get("flow_refresh_count", cfg.hydrology.surface_evolution_iterations)
                ),
            }
            temperature_change = None
            precipitation_change = None
            if prev_temp is not None:
                temperature_change = float(
                    np.sum(
                        np.abs(climate.annual_temperature_c - prev_temp)
                        * grid.cell_area_weights
                    )
                )
                precipitation_change = float(
                    np.sum(
                        np.abs(climate.annual_precipitation_mm - prev_precip)
                        * grid.cell_area_weights
                    )
                )
                stat["mean_abs_temperature_change_c"] = temperature_change
                stat[
                    "mean_abs_precipitation_change_mm_year"
                ] = precipitation_change

            convergence_decision = None
            if macro_tracker is not None:
                convergence_decision = macro_tracker.evaluate(
                    pass_number,
                    temperature_change_c=temperature_change,
                    precipitation_change_mm_year=precipitation_change,
                    elevation_change_m=elevation_change_m,
                    full_fidelity=full_fidelity,
                )
                stat["convergence"] = self._decision_payload(convergence_decision)
            coupling_history.append(stat)
            climate_prev = climate
            surf = None
            geology = None
            gc.collect()

            if convergence_decision is not None and convergence_decision.stop:
                macro_stop_reason = "converged"
                break

        adaptive_final, final_min, final_max = self._final_schedule()
        final_tracker = (
            ConvergenceTracker(
                ConvergenceThresholds(
                    temperature_c=float(
                        c.simulation.final_convergence_temperature_c
                    ),
                    precipitation_mm_year=float(
                        c.simulation.final_convergence_precip_mm_year
                    ),
                    elevation_m=None,
                    required_consecutive=int(
                        c.simulation.required_consecutive_final_converged_passes
                    ),
                    minimum_passes=final_min,
                )
            )
            if adaptive_final
            else None
        )
        executed_final = 0
        final_stop_reason = "fixed_pass_count" if not adaptive_final else "max_passes"

        for k in range(final_max):
            pass_number = k + 1
            executed_final = pass_number
            prev_temp = climate_prev.annual_temperature_c
            prev_precip = climate_prev.annual_precipitation_mm
            wind_u = climate_prev.wind_u
            wind_v = climate_prev.wind_v
            ocean = None
            gc.collect()
            ocean = self._stage(
                f"ocean_final_couple_{pass_number}",
                lambda: _base.build_ocean(
                    grid,
                    tect,
                    terrain,
                    c.ocean,
                    c.terrain,
                    self.rng("ocean-stationary"),
                    wind_u,
                    wind_v,
                    c.noise,
                    static_noise,
                ),
            )
            wind_u = wind_v = None
            climate_prev = None
            climate = None
            gc.collect()
            climate = self._stage(
                f"climate_final_couple_{pass_number}",
                lambda: _base.build_climate(
                    grid,
                    astro,
                    terrain,
                    ocean,
                    c.climate,
                    c.terrain,
                    self.rng("climate-stationary"),
                    c.noise,
                    static_noise,
                ),
            )
            temperature_change = float(
                np.sum(
                    np.abs(climate.annual_temperature_c - prev_temp)
                    * grid.cell_area_weights
                )
            )
            precipitation_change = float(
                np.sum(
                    np.abs(climate.annual_precipitation_mm - prev_precip)
                    * grid.cell_area_weights
                )
            )
            stat = {
                "stage": f"final_climate_ocean_pass_{pass_number}",
                "pass_number": pass_number,
                "climate_fidelity": 1.0,
                "ocean_fidelity": 1.0,
                "full_fidelity": True,
                "moisture_iterations": int(c.climate.moisture_iterations),
                "mean_abs_temperature_change_c": temperature_change,
                "mean_abs_precipitation_change_mm_year": precipitation_change,
            }
            final_decision = None
            if final_tracker is not None:
                final_decision = final_tracker.evaluate(
                    pass_number,
                    temperature_change_c=temperature_change,
                    precipitation_change_mm_year=precipitation_change,
                    full_fidelity=True,
                )
                stat["convergence"] = self._decision_payload(final_decision)
            coupling_history.append(stat)
            climate_prev = climate
            if final_decision is not None and final_decision.stop:
                final_stop_reason = "converged"
                break

        coupling_summary = {
            "earth_system_mode": "adaptive" if adaptive_macro else "fixed",
            "earth_system_passes_executed": int(executed_macro),
            "earth_system_stop_reason": macro_stop_reason,
            "final_coupling_mode": "adaptive" if adaptive_final else "fixed",
            "final_coupling_passes_executed": int(executed_final),
            "final_coupling_stop_reason": final_stop_reason,
        }

        geology = self._stage(
            "geology_final",
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
        geology.rock_code[(cum_dep > 4.0) & terrain.land] = 0
        surface = _base.SurfaceEvolutionResult(
            terrain.elevation_km.astype(np.float32),
            cum_er,
            cum_dep,
            flux,
            cum_delta,
            cum_up,
            cum_mig,
            meander,
            {
                "earth_system_passes": int(executed_macro),
                "earth_system_stop_reason": macro_stop_reason,
                "geomorphic_iterations_total": int(executed_macro)
                * int(c.hydrology.surface_evolution_iterations),
                "max_cumulative_erosion_m": float(cum_er.max()),
                "max_cumulative_deposition_m": float(cum_dep.max()),
                "max_delta_aggradation_m": float(cum_delta.max()),
                "max_tectonic_uplift_m": float(cum_up.max()),
                "max_meander_bank_migration_m": float(cum_mig.max()),
                "model": "multi-pass atmosphere-ocean-tectonics-fluvial surface evolution",
            },
        )
        hydro = self._stage(
            "hydrology_final",
            lambda: _base.build_hydrology(
                grid, terrain, ocean, climate, c.hydrology, geology, surface
            ),
        )
        weather = self._stage(
            "weather",
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
            "surface_appearance",
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
            "resources",
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
            "society",
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
        world = {
            "config": c,
            "grid": grid,
            "astronomy": astro,
            "tectonics": tect,
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
            "coupling_history": coupling_history,
            "coupling_summary": coupling_summary,
        }
        if out_dir is not None:
            self._stage("output", lambda: self.save(world, Path(out_dir)))
        return world

    def _json_export(self, world: dict[str, Any]) -> dict[str, Any]:
        payload = super()._json_export(world)
        payload["coupling_summary"] = world.get("coupling_summary", {})
        return payload

    def _report(self, world: dict[str, Any]) -> str:
        report = super()._report(world).rstrip()
        summary = world.get("coupling_summary", {})
        if not summary:
            return report + "\n"
        lines = [
            report,
            "",
            "## Coupled-system convergence",
            "",
            f"- Earth-system mode: {summary.get('earth_system_mode', 'fixed')}; passes executed: {summary.get('earth_system_passes_executed', 0)}; stop reason: {summary.get('earth_system_stop_reason', 'unknown')}.",
            f"- Final atmosphere-ocean mode: {summary.get('final_coupling_mode', 'fixed')}; passes executed: {summary.get('final_coupling_passes_executed', 0)}; stop reason: {summary.get('final_coupling_stop_reason', 'unknown')}.",
            "- Adaptive convergence is a numerical stopping criterion for this reduced-order model; it is not evidence that the model is a complete physical Earth-system solution.",
        ]
        return "\n".join(lines) + "\n"


__all__ = ["WorldPipeline"]
