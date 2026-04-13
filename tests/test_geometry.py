from __future__ import annotations

import numpy as np

from tokamak_source_model.geometry import surface_to_rz
from tokamak_source_model.parameters import GeometryParameters

def test_surface_to_rz_center_point():
    geometry = GeometryParameters(
        major_radius_m=2.0,
        minor_radius_m=0.5,
        elongation=1.7,
        triangularity=0.33,
        shafranov_shift_m=0.1
    )

    R_m, Z_m = surface_to_rz(0.0, 0.0, geometry)

    assert np.isclose(R_m, geometry.major_radius_m + geometry.shafranov_shift_m)
    assert np.isclose(Z_m, 0.0)

def test_surface_to_rz_top_point_has_positive_z():
    geometry = GeometryParameters(
        major_radius_m=2.0,
        minor_radius_m=0.5,
        elongation=1.7,
        triangularity=0.33,
        shafranov_shift_m=0.1,
    )

    a_m = 0.5
    alpha_rad = np.pi / 2.0

    R_m, Z_m = surface_to_rz(a_m, alpha_rad, geometry)

    assert Z_m > 0.0