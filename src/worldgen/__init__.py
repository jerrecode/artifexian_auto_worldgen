"""Automatic, procedural worldbuilding pipeline."""
from __future__ import annotations

from .config import WorldConfig, load_config

__all__ = ["WorldConfig", "load_config", "WorldPipeline"]
__version__ = "0.5.0"


# Install the low-speed tectonic initializer before pipeline modules bind their
# runtime dependencies. This keeps stagnant-lid/icy worlds within their configured
# plate-speed regime instead of relying on a workflow-time source patch.
from .tectonics_low_speed import install_low_speed_tectonics_fix as _install_low_speed_tectonics_fix  # noqa: E402
_install_low_speed_tectonics_fix()


# Correct the historical Bond-albedo normalization before pipeline_base binds
# build_astronomy. This keeps target-temperature orbit solving and Atmogen stellar
# forcing on one physically consistent radiative-energy convention.
from .astronomy_radiative_fix import install_astronomy_radiative_fix as _install_astronomy_radiative_fix  # noqa: E402
_install_astronomy_radiative_fix()


# Replace exact circular hotspot/LIP splats, exact-radius geological proximity
# masks, and tiny-lake continentality halos before the pipeline captures stage
# functions. The replacement fields remain deterministic and spherical but are
# anisotropic, lobate, structurally rough, and component-aware.
from .spatial_naturalism import install_spatial_naturalism_fix as _install_spatial_naturalism_fix  # noqa: E402
_install_spatial_naturalism_fix()


# Install the complete advanced pipeline hierarchy onto the canonical pipeline module.
# Import order is deliberate: each layer subclasses the preceding public behavior,
# ending with drainage-rerouted secondary geomorphology. Existing imports from
# ``worldgen.pipeline`` therefore gain advanced behavior without a second CLI.
from . import pipeline as _pipeline  # noqa: E402
from .pipeline_liquids import WorldPipeline as _SurfaceLiquidWorldPipeline  # noqa: E402,F401
from .pipeline_exotic import WorldPipeline as _ExoticWorldPipeline  # noqa: E402,F401
from .pipeline_landscape import WorldPipeline as _LandscapeWorldPipeline  # noqa: E402,F401
from .pipeline_geomorphology import WorldPipeline as _GeomorphologyWorldPipeline  # noqa: E402
_pipeline.WorldPipeline = _GeomorphologyWorldPipeline


def __getattr__(name: str):
    if name == "WorldPipeline":
        return _pipeline.WorldPipeline
    raise AttributeError(name)
