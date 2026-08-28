"""Automatic, procedural worldbuilding pipeline."""
from __future__ import annotations

from .config import WorldConfig, load_config

__all__ = ["WorldConfig", "load_config", "WorldPipeline"]
__version__ = "0.5.0"


# Install the complete advanced pipeline hierarchy onto the canonical pipeline module.
# Import order is deliberate: each layer subclasses the preceding public behavior,
# ending with geologic-time landscape closure. Existing imports from
# ``worldgen.pipeline`` therefore gain advanced behavior without a second CLI.
from . import pipeline as _pipeline  # noqa: E402
from .pipeline_liquids import WorldPipeline as _SurfaceLiquidWorldPipeline  # noqa: E402,F401
from .pipeline_exotic import WorldPipeline as _ExoticWorldPipeline  # noqa: E402,F401
from .pipeline_landscape import WorldPipeline as _LandscapeWorldPipeline  # noqa: E402
_pipeline.WorldPipeline = _LandscapeWorldPipeline


def __getattr__(name: str):
    if name == "WorldPipeline":
        return _pipeline.WorldPipeline
    raise AttributeError(name)
