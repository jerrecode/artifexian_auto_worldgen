from __future__ import annotations

"""Public hydrology facade with optimized and physically richer execution layers.

The long-standing physical kernels remain in ``hydrology_base``. This facade patches
performance/conservation-sensitive primitives at import time, then exposes advanced
water balance, channel hierarchy and watershed results without duplicating the stable
public result types.
"""

from . import hydrology_base as _base
from . import hydrology_advanced as _advanced
from .priority_flood import (
    priority_flood,
    priority_flood_reference,
    numba_priority_flood_available,
)
from .hydrology_advanced import (
    AdvancedHydrologyResult,
    WaterBalanceResult,
    build_hydrology_advanced,
    evolve_surface_advanced,
    runoff_mm_advanced,
    transport_sediment_topological,
)
from .hydrology_reliability import (
    channel_hierarchy_discharge_guarded,
    enforce_hydrology_guardrails,
    flow_directions_multidirection,
    lake_mask_volume_guarded,
    priority_flood_closed_aware,
)
from .multicondensate_water_balance import build_multicondensate_water_balance


# Keep the legacy/ordinary conservative bucket as the default, but let a hydrologic
# climate view activate the mass-conservative multicomponent bucket. The cached
# water-balance function in hydrology_advanced resolves ``build_water_balance`` from
# its module globals at call time, so this dispatch automatically reaches runoff,
# surface evolution and final hydrology without duplicating those algorithms.
_original_build_water_balance = _advanced.build_water_balance


def _build_water_balance_dispatch(climate, land, geology, cfg):
    if getattr(climate, "hydrologic_forcing", None) is not None:
        return build_multicondensate_water_balance(climate, land, geology, cfg)
    return _original_build_water_balance(climate, land, geology, cfg)


_advanced.build_water_balance = _build_water_balance_dispatch


# Compatibility terminology: these storage fields predate exotic-fluid support and
# retain their serialized names, but their numerical semantics in a composition-aware
# world are active-condensate liquid/solid equivalent depths, not intrinsically H2O.
def _soil_liquid_storage(self):
    return self.soil_water_storage_mm


def _subsurface_liquid_storage(self):
    return self.groundwater_storage_mm


def _solid_condensate_storage(self):
    return self.snowpack_mm


for _result_cls in (WaterBalanceResult, AdvancedHydrologyResult):
    _result_cls.soil_liquid_storage_mm = property(_soil_liquid_storage)
    _result_cls.subsurface_liquid_storage_mm = property(_subsurface_liquid_storage)
    _result_cls.solid_condensate_storage_mm = property(_solid_condensate_storage)


# Install deterministic optimized/conservative backends into the shared equations.
# The closed-aware flood leaves genuine endorheic minima intact when there is no
# ocean, and the dense angular receiver stencil removes the strong raster-direction
# bias seen in maximum-complexity runs.  Lake and channel classifiers then enforce
# strict occupied-area and accumulated-discharge semantics.
_base._priority_flood = priority_flood_closed_aware
_base._flow_directions = flow_directions_multidirection
_base._lake_mask = lake_mask_volume_guarded
_base._runoff_mm = runoff_mm_advanced
_base._transport_sediment = transport_sediment_topological
_advanced._channel_hierarchy = channel_hierarchy_discharge_guarded

from .hydrology_base import *  # noqa: F401,F403,E402


def build_hydrology(
    grid,
    terrain,
    ocean,
    climate,
    cfg,
    geology=None,
    surface=None,
):
    """Build advanced hydrology and reject globally pathological classifications."""
    result = build_hydrology_advanced(
        grid, terrain, ocean, climate, cfg, geology, surface
    )
    enforce_hydrology_guardrails(result, terrain, cfg)
    result.metadata["routing_reliability_model"] = (
        "closed-basin-aware Priority-Flood + dense primitive-direction steepest "
        "descent + discharge-gated channels + strict lake-area budget"
    )
    return result


# Public advanced surface wrapper. Existing callers continue using the same name.
evolve_surface = evolve_surface_advanced

# Preserve historically importable private name for tests/tools. The public
# ``priority_flood`` symbol remains the optimized open-boundary implementation;
# world generation uses the closed-aware wrapper installed above.
_priority_flood = priority_flood_closed_aware

__all__ = [name for name in dir(_base) if not name.startswith("_")]
__all__ += [
    "build_hydrology",
    "evolve_surface",
    "priority_flood",
    "priority_flood_reference",
    "numba_priority_flood_available",
]
