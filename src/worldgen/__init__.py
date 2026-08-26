"""Automatic, procedural worldbuilding pipeline."""
from .config import WorldConfig, load_config
from .pipeline import WorldPipeline

__all__ = ["WorldConfig", "load_config", "WorldPipeline"]
__version__ = "0.1.0"
