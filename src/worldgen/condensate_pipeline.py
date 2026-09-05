from __future__ import annotations

"""Pipeline utility for installing multicomponent condensate hydrology consistently."""

from typing import Any

from . import pipeline_base as _base
from .condensate_hydrology import climate_for_hydrology
from .depressions import build_depressions
from .geomorphic_fluids import build_exotic_geomorphology


def install_multicondensate_hydrology(
    pipeline: Any,
    world: dict[str, Any],
    *,
    hydrology_cfg: Any | None = None,
    suffix: str = "",
    rebuild_dependents: bool = True,
) -> dict[str, Any]:
    """Replace single-reference hydrology with conservative species-aware forcing.

    This is intentionally a pipeline-level operation instead of a hidden global
    monkeypatch: the final world records the forcing object and every dependent state
    can be rebuilt from the same climate/shoreline before outputs are written.
    """
    c = pipeline.cfg
    atmogen_enabled = bool(getattr(getattr(c, "atmogen", None), "enabled", False))
    composition_enabled = str(getattr(c.astronomy, "greenhouse_model", "legacy")) == "composition"
    surface_volatiles = getattr(c.astronomy, "surface_volatiles", None)
    if not surface_volatiles or not (atmogen_enabled or composition_enabled):
        return world
    tag = f"_{suffix}" if suffix else ""
    hcfg = c.hydrology if hydrology_cfg is None else hydrology_cfg
    climate_view, forcing = climate_for_hydrology(
        world["astronomy"],
        world["climate"],
        surface_volatiles=c.astronomy.surface_volatiles,
    )
    hydro = pipeline._stage(
        f"hydrology_multicondensate{tag}",
        lambda: _base.build_hydrology(
            world["grid"],
            world["terrain"],
            world["ocean"],
            climate_view,
            hcfg,
            world.get("geology"),
            world.get("surface_evolution"),
        ),
    )
    world["hydrology"] = hydro
    world["condensate_hydrology"] = forcing
    world.setdefault("coupling_summary", {})["multicondensate_hydrology"] = {
        "enabled": True,
        "reference_species": forcing.reference_species,
        "active_hydrologic_species": list(forcing.metadata.get("active_hydrologic_species", [])),
        "mass_conservation_relative_l1_residual": float(
            forcing.metadata.get("mass_conservation_relative_l1_residual", 0.0)
        ),
        "precipitation_depth_semantics": "species liquid-equivalent depths after exact mass partition",
    }

    if not rebuild_dependents:
        return world

    if world.get("depressions") is not None:
        world["depressions"] = pipeline._stage(
            f"depression_storage_multicondensate{tag}",
            lambda: build_depressions(
                world["grid"], world["terrain"], climate_view, world["hydrology"]
            ),
        )
    if (
        world.get("geomorphic_fluid_parameters") is not None
        and world.get("volatile_cycle") is not None
        and world.get("cryogeology") is not None
    ):
        world["exotic_geomorphology"] = pipeline._stage(
            f"exotic_geomorphology_multicondensate{tag}",
            lambda: build_exotic_geomorphology(
                world["grid"],
                world["terrain"],
                climate_view,
                world["hydrology"],
                world["geomorphic_fluid_parameters"],
                world["volatile_cycle"],
                world["cryogeology"],
            ),
        )

    world["weather"] = pipeline._stage(
        f"weather_multicondensate{tag}",
        lambda: _base.build_weather(
            world["grid"],
            world["terrain"],
            world["ocean"],
            world["climate"],
            world["hydrology"],
            c.weather,
            pipeline.rng("weather"),
        ),
    )
    world["appearance"] = pipeline._stage(
        f"appearance_multicondensate{tag}",
        lambda: _base.build_surface_appearance(
            world["grid"],
            world["terrain"],
            world["ocean"],
            world["climate"],
            world["hydrology"],
            world["geology"],
            world["weather"],
            c.appearance,
        ),
    )
    world["resources"] = pipeline._stage(
        f"resources_multicondensate{tag}",
        lambda: _base.build_resources(
            world["grid"],
            world["tectonics"],
            world["terrain"],
            world["ocean"],
            world["climate"],
            world["hydrology"],
            world["geology"],
            c.resources,
            pipeline.rng("resources"),
        ),
    )
    world["society"] = pipeline._stage(
        f"society_multicondensate{tag}",
        lambda: _base.build_society(
            world["grid"],
            world["terrain"],
            world["climate"],
            world["hydrology"],
            world["resources"],
            world["weather"],
            c.society,
            pipeline.rng("society"),
            world["appearance"],
        ),
    )
    return world


__all__ = ["install_multicondensate_hydrology"]
