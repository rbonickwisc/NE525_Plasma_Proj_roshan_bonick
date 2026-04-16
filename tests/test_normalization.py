from __future__ import annotations

import numpy as np

from tokamak_source_model.normalization import build_source_probability_map, estimate_total_neutron_rate_n_per_s, estimate_total_plasma_volume_m3
from tokamak_source_model.parameters import FuelParameters, GeometryParameters, MeshParameters, ProfileParameters, SourceModelParameters

def make_l_mode_model() -> SourceModelParameters:
    geometry = GeometryParameters(
        major_radius_m=2.0,
        minor_radius_m=0.5,
        elongation=1.7,
        triangularity=0.33,
        shafranov_shift_m=0.1
    )
    profile = ProfileParameters(
        mode="l_mode",
        ion_density_center_m3=2.0e20,
        ion_temp_center_keV=15.0,
        alpha_n=0.5,
        alpha_T=1.0,
    )
    fuel = FuelParameters(
        deuterium_fraction=0.5,
        tritium_fraction=0.5,
    )
    
    return SourceModelParameters(
        geometry=geometry,
        profile=profile,
        fuel=fuel,
    )

def test_total_plasma_volume_is_positive():
    model = make_l_mode_model()
    mesh = MeshParameters(num_a=120, num_alpha=180)

    volume_m3 = estimate_total_plasma_volume_m3(model, mesh)

    assert volume_m3 > 0.0

def test_total_neutron_rate_is_positive():
    model = make_l_mode_model()
    mesh = MeshParameters(num_a=120, num_alpha=180)

    total_rate = estimate_total_neutron_rate_n_per_s(model, mesh)

    assert total_rate > 0.0

def test_probability_map_is_normalized():
    model = make_l_mode_model()
    mesh = MeshParameters(num_a=120, num_alpha=180)

    _, _, _, _, probability_map = build_source_probability_map(model, mesh)

    assert np.isclose(np.sum(probability_map), 1.0)
    assert np.all(probability_map >= 0.0)