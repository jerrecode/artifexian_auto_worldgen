"""Automatic, procedural worldbuilding pipeline."""
from __future__ import annotations

from .config import WorldConfig, load_config

__all__ = ["WorldConfig", "load_config", "WorldPipeline"]
__version__ = "0.4.0"


# Install the advanced surface-reservoir subclass onto the canonical pipeline module.
# Import order is deliberate: pipeline_liquids captures the original adaptive class,
# subclasses it, and only then replaces the public module attribute.  Consequently
# existing ``from worldgen.pipeline import WorldPipeline`` imports continue to work
# without forcing callers onto a new command or API path.
from . import pipeline as _pipeline  # noqa: E402
from .pipeline_liquids import WorldPipeline as _SurfaceLiquidWorldPipeline  # noqa: E402
_pipeline.WorldPipeline = _SurfaceLiquidWorldPipeline


def __getattr__(name: str):
    if name == "WorldPipeline":
        return _pipeline.WorldPipeline
    raise AttributeError(name)
