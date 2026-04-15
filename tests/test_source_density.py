from __future__ import annotations

import numpy as np

from tokamak_source_model.parameters import (
    FuelParameters,
    GeometryParameters,
    ProfileParameters,
    SourceModelParameters,
)
from tokamak_source_model.source_density import (
    evaluate_profiles,
    source_density_profile_n_per_m3_per_s,
)

def make_l_mode_model(
    deuterium_fraction: float = 0.5,
    tritium_fraction: float = 0.5,
) -> SourceModelParameters:
    geometry = GeometryParameters(
        major_radius_m=2.0,
        minor_radius_m=0.5,
        elongation=1.7,
        triangularity=0.33,
        shafranov_shift_m=0.1,
    )
    profile = ProfileParameters(
        mode="l_mode",
        ion_density_center_m3=2.0e20,
        ion_temp_center_keV=15.0,
        alpha_n=0.5,
        alpha_T=1.0,
    )
    fuel = FuelParameters(
        deuterium_fraction=deuterium_fraction,
        tritium_fraction=tritium_fraction,
    )
    return SourceModelParameters(
        geometry=geometry,
        profile=profile,
        fuel=fuel,
    )

def test_source_density_is_nonnegative():
    model = make_l_mode_model()
    a_m = np.linspace(0.0, model.geometry.minor_radius_m, 200)

    source_density = source_density_profile_n_per_m3_per_s(a_m, model)

    assert np.all(source_density >= 0.0)

def test_source_density_is_zero_when_no_deuterium():
    model = make_l_mode_model(deuterium_fraction=0.0, tritium_fraction=1.0)
    a_m = np.linspace(0.0, model.geometry.minor_radius_m, 200)

    source_density = source_density_profile_n_per_m3_per_s(a_m, model)

    assert np.allclose(source_density, 0.0)

def test_source_density_is_zero_when_no_tritium():
    model = make_l_mode_model(deuterium_fraction=1.0, tritium_fraction=0.0)
    a_m = np.linspace(0.0, model.geometry.minor_radius_m, 200)

    source_density = source_density_profile_n_per_m3_per_s(a_m, model)

    assert np.allclose(source_density, 0.0)

def test_source_density_peaks_at_center_for_l_mode_class():
    model = make_l_mode_model()
    a_m = np.linspace(0.0, model.geometry.minor_radius_m, 200)

    evaluation = evaluate_profiles(a_m, model)

    center_value = evaluation.source_density_n_per_m3_per_s[0]
    max_value = np.max(evaluation.source_density_n_per_m3_per_s)

    assert np.isclose(center_value, max_value)