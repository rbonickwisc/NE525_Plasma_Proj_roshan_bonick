from __future__ import annotations

import numpy as np

from tokamak_source_model.parameters import GeometryParameters, ProfileParameters
from tokamak_source_model.profiles import ion_density_profile_m3, ion_temperature_profile_keV

def test_l_mode_profiles_match_center_values():
    geometry = GeometryParameters(
        major_radius_m=2.0,
        minor_radius_m=0.5,
        elongation=1.7,
        triangularity=0.33,
        shafranov_shift_m=1.0
    )

    profile = ProfileParameters(
        mode="l_mode",
        ion_density_center_m3=2.0e20,
        ion_temp_center_keV=15.0,
        alpha_n=0.5,
        alpha_T=1.0,
    )

    a_m = np.array([0.0])

    ni = ion_density_profile_m3(a_m, geometry, profile)
    Ti = ion_temperature_profile_keV(a_m, geometry, profile)

    assert np.isclose(ni[0], profile.ion_density_center_m3)
    assert np.isclose(Ti[0], profile.ion_temp_center_keV)

def test_l_mode_profiles_go_to_zero_at_edge():
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

    a_m = np.array([geometry.minor_radius_m])

    ni = ion_density_profile_m3(a_m, geometry, profile)
    Ti = ion_temperature_profile_keV(a_m, geometry, profile)

    assert np.isclose(ni[0], 0.0)
    assert np.isclose(Ti[0], 0.0)