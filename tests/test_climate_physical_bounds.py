from __future__ import annotations

import numpy as np
import pytest

from worldgen.climate import (
    MIN_MODEL_TEMPERATURE_C,
    MIN_MODEL_TEMPERATURE_K,
    _enforce_physical_temperature_floor,
)


def test_reduced_order_climate_floor_preserves_valid_values_and_clamps_impossible_tail():
    values = np.asarray(
        [[15.0, -100.0, -273.15, -400.0]],
        dtype=np.float64,
    )
    bounded, count = _enforce_physical_temperature_floor(values)

    assert count == 2
    assert bounded[0, 0] == 15.0
    assert bounded[0, 1] == -100.0
    assert bounded[0, 2] == MIN_MODEL_TEMPERATURE_C
    assert bounded[0, 3] == MIN_MODEL_TEMPERATURE_C
    assert np.all(bounded + 273.15 >= MIN_MODEL_TEMPERATURE_K)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_reduced_order_climate_floor_rejects_nonfinite_state(bad):
    with pytest.raises(ValueError, match="temperature field must remain finite"):
        _enforce_physical_temperature_floor(np.asarray([[bad]], dtype=np.float64))
