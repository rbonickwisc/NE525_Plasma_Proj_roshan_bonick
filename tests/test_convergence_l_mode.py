from __future__ import annotations

import numpy as np

from tokamak_source_model.utils.normalization import build_source_probability_map, estimate_total_neutron_rate_n_per_s, estimate_total_plasma_volume_m3
from tokamak_source_model.utils.parameters import FuelParameters, GeometryParameters, MeshParameters, ProfileParameters, SourceModelParameters

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

def test_multiple_meshes_produce_positive_integrals():
    model = make_l_mode_model()
    
    mesh_list = [
        MeshParameters(num_a=40, num_alpha=60),
        MeshParameters(num_a=80, num_alpha=120),
        MeshParameters(num_a=120, num_alpha=180),
    ]

    for mesh in mesh_list:
        volume_m3 = estimate_total_plasma_volume_m3(model, mesh)
        total_rate_n_per_s = estimate_total_neutron_rate_n_per_s(model, mesh)
        _, _, _, _, probability_map = build_source_probability_map(model, mesh)

        assert volume_m3 > 0
        assert total_rate_n_per_s > 0
        assert np.isclose(np.sum(probability_map), 1.0)