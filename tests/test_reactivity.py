from __future__ import annotations

import numpy as np
import pytest

from tokamak_source_model.utils.reactivity import dt_reactivity_m3_per_s

def test_dt_reactivity_is_positive_over_valid_range():
    Ti_keV = np.array([1.0, 5.0, 10.0, 15.0, 20.0, 50.0, 100.0])
    reactivity = dt_reactivity_m3_per_s(Ti_keV)

    assert np.all(reactivity > 0)

def test_dt_reactivity_rejects_nonpositive_temperature():
    with pytest.raises(ValueError):
        dt_reactivity_m3_per_s(np.array([0.0, 10.0]))

def test_dt_reactivity_rejects_temperature_above_valid_range():
    with pytest.raises(ValueError):
        dt_reactivity_m3_per_s(np.array([10.0, 101.0]))