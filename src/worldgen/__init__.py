"""Automatic, procedural worldbuilding pipeline."""
from __future__ import annotations

from .config import WorldConfig, load_config

__all__ = ["WorldConfig", "load_config", "WorldPipeline"]
__version__ = "0.5.0"


# Install numerical compatibility patches before the pipeline imports modules that
# call the canonical kernels.  Active terrestrial worlds retain their historical RNG
# path exactly; only stagnant/inactive worlds below the legacy 0.8 cm/yr floor use the
# regime-safe low-speed motion sampler.
from .tectonics_regimes import install_regime_safe_tectonic_initializer as _install_tectonic_regimes  # noqa: E402
_install_tectonic_regimes()


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
