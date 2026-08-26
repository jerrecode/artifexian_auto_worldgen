"""Automatic, procedural worldbuilding pipeline."""
from __future__ import annotations

from .config import WorldConfig, load_config

__all__ = ["WorldConfig", "load_config", "WorldPipeline"]
__version__ = "0.4.0"


def __getattr__(name: str):
    if name == "WorldPipeline":
        from .pipeline import WorldPipeline
        return WorldPipeline
    raise AttributeError(name)
