from __future__ import annotations

import numpy as np

from tokamak_source_model.case_builder import build_default_fuel, build_l_mode_model

from tokamak_source_model.source_density import (
    evaluate_profiles,
    source_density_profile_n_per_m3_per_s,
)

def test_source_density_is_nonnegative():
    model = build_l_mode_model()
    a_m = np.linspace(0.0, model.geometry.minor_radius_m, 200)

    source_density = source_density_profile_n_per_m3_per_s(a_m, model)

    assert np.all(source_density >= 0.0)

def test_source_density_is_zero_when_no_deuterium():
    fuel = build_default_fuel(deuterium_fraction=0.0, tritium_fraction=1.0)
    model = build_l_mode_model(fuel=fuel)
    a_m = np.linspace(0.0, model.geometry.minor_radius_m, 200)

    source_density = source_density_profile_n_per_m3_per_s(a_m, model)

    assert np.allclose(source_density, 0.0)

def test_source_density_is_zero_when_no_tritium():
    fuel = build_default_fuel(deuterium_fraction=1.0, tritium_fraction=0.0)
    model = build_l_mode_model(fuel=fuel)
    a_m = np.linspace(0.0, model.geometry.minor_radius_m, 200)

    source_density = source_density_profile_n_per_m3_per_s(a_m, model)

    assert np.allclose(source_density, 0.0)

def test_source_density_peaks_at_center_for_l_mode_class():
    model = build_l_mode_model()
    a_m = np.linspace(0.0, model.geometry.minor_radius_m, 200)

    evaluation = evaluate_profiles(a_m, model)

    center_value = evaluation.source_density_n_per_m3_per_s[0]
    max_value = np.max(evaluation.source_density_n_per_m3_per_s)

    assert np.isclose(center_value, max_value)

def test_evaluate_profiles_non_5050_fuel_split():
    fuel = build_default_fuel(deuterium_fraction=0.3, tritium_fraction=0.7)
    model = build_l_mode_model(fuel=fuel)
    a_m = np.linspace(0.0, model.geometry.minor_radius_m, 200)

    evaluation = evaluate_profiles(a_m, model)

    assert np.allclose(
        evaluation.deuterium_density_m3,
        0.3 * evaluation.ion_density_m3,
    )

    assert np.allclose(
        evaluation.tritium_density_m3,
        0.7 * evaluation.ion_density_m3,
    )

    assert not np.allclose(
        evaluation.deuterium_density_m3,
        evaluation.tritium_density_m3,
    )