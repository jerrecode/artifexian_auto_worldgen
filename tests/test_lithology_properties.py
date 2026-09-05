from __future__ import annotations

import numpy as np

from worldgen import hydrology_base
from worldgen.hydrology_advanced import _INFILTRATION, _SOIL_CAPACITY
from worldgen.lithology_properties import (
    INFILTRATION_FRACTION,
    MECHANICAL_ERODIBILITY,
    RUNOFF_MULTIPLIER,
    SOIL_CAPACITY_MM,
    properties_for_codes,
)


def test_central_lithology_tables_preserve_legacy_hydrology_coefficients():
    assert np.array_equal(hydrology_base._LITH_ERODIBILITY, MECHANICAL_ERODIBILITY)
    assert np.array_equal(hydrology_base._LITH_RUNOFF, RUNOFF_MULTIPLIER)
    assert np.array_equal(_INFILTRATION, INFILTRATION_FRACTION)
    assert np.array_equal(_SOIL_CAPACITY, SOIL_CAPACITY_MM)


def test_lithology_property_lookup_is_vectorized_and_clamped():
    fields = properties_for_codes(np.asarray([[-3, 0, 8, 99]]))
    assert fields["mechanical_erodibility"].shape == (1, 4)
    assert fields["mechanical_erodibility"][0, 0] == MECHANICAL_ERODIBILITY[0]
    assert fields["mechanical_erodibility"][0, -1] == MECHANICAL_ERODIBILITY[-1]
