from __future__ import annotations

import pytest

from tokamak_source_model.case_builder import build_default_fuel, build_default_geometry, build_l_mode_profile
from tokamak_source_model.parameters import ProfileParameters, SourceModelParameters
from tokamak_source_model.validation import validate_source_model_parameters

def test_validation_rejects_fuel_fraction_sum_not_equal():
    geometry = build_default_geometry()
    profile = build_l_mode_profile()
    bad_fuel = build_default_fuel(deuterium_fraction=0.6, tritium_fraction=0.5)

    model = SourceModelParameters(
        geometry=geometry,
        profile=profile,
        fuel=bad_fuel,
    )

    with pytest.raises(ValueError):
        validate_source_model_parameters(model)

def test_validation_rejects_negative_center_density():
    geometry = build_default_geometry()
    fuel = build_default_fuel()

    bad_profile = ProfileParameters(
        mode="l_mode",
        ion_density_center_m3=-1.0,
        ion_temp_center_keV=15.0,
        alpha_n=0.5,
        alpha_T=1.0,
    )

    model = SourceModelParameters(
        geometry=geometry,
        profile=bad_profile,
        fuel=fuel,
    )

    with pytest.raises(ValueError):
        validate_source_model_parameters(model)

def test_validation_rejects_negative_center_temperature():
    geometry = build_default_geometry()
    fuel = build_default_fuel()

    bad_profile = ProfileParameters(
        mode="l_mode",
        ion_density_center_m3=2e20,
        ion_temp_center_keV=-1.0,
        alpha_n=0.5,
        alpha_T=1.0,
    )

    model = SourceModelParameters(
        geometry=geometry,
        profile=bad_profile,
        fuel=fuel,
    )

    with pytest.raises(ValueError):
        validate_source_model_parameters(model)

def test_validation_rejects_pedestal_radius_larger_than_minor_radius():
    geometry = build_default_geometry()
    fuel = build_default_fuel()

    bad_profile = ProfileParameters(
        mode="pedestal",
        ion_density_center_m3=2.0e20,
        ion_temp_center_keV=20.0,
        alpha_n=1.0,
        alpha_T=4.0,
        pedestal_radius_m=1.1 * geometry.minor_radius_m,
        ion_density_pedestal_m3=1.8e20,
        ion_density_separatrix_m3=3.0e19,
        ion_temp_pedestal_keV=4.0,
        ion_temp_separatrix_keV=0.1,
        beta_T=6.0,
    )

    model = SourceModelParameters(
        geometry=geometry,
        profile=bad_profile,
        fuel=fuel,
    )

    with pytest.raises(ValueError):
        validate_source_model_parameters(model)

def test_validation_rejects_missing_pedestal_fields():
    geometry = build_default_geometry()
    fuel = build_default_fuel()

    bad_profile = ProfileParameters(
        mode="pedestal",
        ion_density_center_m3=2.0e20,
        ion_temp_center_keV=20.0,
        alpha_n=1.0,
        alpha_T=4.0,
        pedestal_radius_m=None,
        ion_density_pedestal_m3=None,
        ion_density_separatrix_m3=3.0e19,
        ion_temp_pedestal_keV=4.0,
        ion_temp_separatrix_keV=0.1,
        beta_T=6.0,
    )

    model = SourceModelParameters(
        geometry=geometry,
        profile=bad_profile,
        fuel=fuel,
    )

    with pytest.raises(ValueError):
        validate_source_model_parameters(model)