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


# Install the advanced pipeline hierarchy onto the canonical pipeline module.
# Import order is deliberate: pipeline_liquids subclasses the original adaptive
# pipeline; pipeline_exotic subclasses the liquid-aware pipeline. Existing imports
# from ``worldgen.pipeline`` therefore gain advanced behavior without a second CLI.
from . import pipeline as _pipeline  # noqa: E402
from .pipeline_liquids import WorldPipeline as _SurfaceLiquidWorldPipeline  # noqa: E402,F401
from .pipeline_exotic import WorldPipeline as _ExoticWorldPipeline  # noqa: E402
_pipeline.WorldPipeline = _ExoticWorldPipeline


def __getattr__(name: str):
    if name == "WorldPipeline":
        return _pipeline.WorldPipeline
    raise AttributeError(name)
