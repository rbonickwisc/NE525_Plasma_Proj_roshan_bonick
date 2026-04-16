from __future__ import annotations

import numpy as np

from tokamak_source_model.parameters import GeometryParameters, ProfileParameters
from tokamak_source_model.profiles import ion_density_profile_m3, ion_temperature_profile_keV


def make_geometry() -> GeometryParameters:
    return GeometryParameters(
        major_radius_m=2.0,
        minor_radius_m=0.5,
        elongation=1.7,
        triangularity=0.33,
        shafranov_shift_m=0.1,
    )

def make_pedestal_profile(geometry: GeometryParameters) -> ProfileParameters:
    return ProfileParameters(
        mode="pedestal",
        ion_density_center_m3=2.0e20,
        ion_temp_center_keV=20.0,
        alpha_n=1.0,
        alpha_T=4.0,
        pedestal_radius_m=0.8 * geometry.minor_radius_m,
        ion_density_pedestal_m3=1.8e20,
        ion_density_separatrix_m3=3.0e19,
        ion_temp_pedestal_keV=4.0,
        ion_temp_separatrix_keV=0.1,
        beta_T=6.0,
    )

def test_pedestal_profile_center_matches_center_values():
    geometry = make_geometry()
    profile = make_pedestal_profile(geometry)

    a_m = np.array([0.0])

    ni = ion_density_profile_m3(a_m, geometry, profile)
    Ti = ion_temperature_profile_keV(a_m, geometry, profile)

    assert np.isclose(ni[0], profile.ion_density_center_m3)
    assert np.isclose(Ti[0], profile.ion_temp_center_keV)

def test_pedestal_edge_matches_separatrix_values():
    geometry = make_geometry()
    profile = make_pedestal_profile(geometry)

    a_m = np.array([geometry.minor_radius_m])

    ni = ion_density_profile_m3(a_m, geometry, profile)
    Ti = ion_temperature_profile_keV(a_m, geometry, profile)

    assert np.isclose(ni[0], profile.ion_density_separatrix_m3)
    assert np.isclose(Ti[0], profile.ion_temp_separatrix_keV)

def test_pedestal_profile_is_continuous_at_pedestal_radius():
    geometry = make_geometry()
    profile = make_pedestal_profile(geometry)

    aped = profile.pedestal_radius_m
    assert aped is not None

    eps = 1.0e-8
    a_vals = np.array([aped - eps, aped + eps])

    ni = ion_density_profile_m3(a_vals, geometry, profile)
    Ti = ion_temperature_profile_keV(a_vals, geometry, profile)

    assert np.isclose(ni[0], ni[1], rtol=1e-6, atol=0.0)
    assert np.isclose(Ti[0], Ti[1], rtol=0.0, atol=1e-4)

def test_pedestal_profile_matches_pedestal_values_at_aped():
    geometry = make_geometry()
    profile = make_pedestal_profile(geometry)

    aped = profile.pedestal_radius_m
    assert aped is not None
    assert profile.ion_density_center_m3 is not None
    assert profile.ion_temp_pedestal_keV is not None

    a_m = np.array([aped])
    
    ni = ion_density_profile_m3(a_m, geometry, profile)
    Ti = ion_temperature_profile_keV(a_m, geometry, profile)

    assert np.isclose(ni[0], profile.ion_density_pedestal_m3, rtol=1e-12, atol=0)
    assert np.isclose(Ti[0], profile.ion_temp_pedestal_keV, rtol=1e-12, atol=0)