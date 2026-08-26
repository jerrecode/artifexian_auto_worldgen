from __future__ import annotations

"""Public hydrology facade with pluggable optimized kernels.

The physical model remains in ``hydrology_base`` while performance-sensitive
kernels can evolve independently. Functions defined in the base module resolve
``_priority_flood`` through that module's globals at call time, so replacing the
binding here accelerates both surface evolution and final hydrology without
forking the physical implementation.
"""

from . import hydrology_base as _base
from .priority_flood import (
    priority_flood,
    priority_flood_reference,
    numba_priority_flood_available,
)

# Install the optimized deterministic backend into the existing physical model.
_base._priority_flood = priority_flood

from .hydrology_base import *  # noqa: F401,F403,E402

# Preserve the historically importable private name for tests/tools.
_priority_flood = priority_flood

__all__ = [name for name in dir(_base) if not name.startswith("_")]
__all__ += [
    "priority_flood",
    "priority_flood_reference",
    "numba_priority_flood_available",
]
