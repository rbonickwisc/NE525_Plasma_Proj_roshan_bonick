from __future__ import annotations

import numpy as np

from tokamak_source_model.normalization import build_source_probability_map, estimate_total_neutron_rate_n_per_s, estimate_total_plasma_volume_m3, build_source_cell_probability_map
from tokamak_source_model.case_builder import build_default_mesh, build_l_mode_model

def test_total_plasma_volume_is_positive():
    model = build_l_mode_model()
    mesh = build_default_mesh(num_a=120, num_alpha=180)

    volume_m3 = estimate_total_plasma_volume_m3(model, mesh)

    assert volume_m3 > 0.0

def test_total_neutron_rate_is_positive():
    model = build_l_mode_model()
    mesh = build_default_mesh(num_a=120, num_alpha=180)

    total_rate = estimate_total_neutron_rate_n_per_s(model, mesh)

    assert total_rate > 0.0

def test_probability_map_is_normalized():
    model = build_l_mode_model()
    mesh = build_default_mesh(num_a=120, num_alpha=180)

    _, _, _, _, probability_map = build_source_probability_map(model, mesh)

    assert np.isclose(np.sum(probability_map), 1.0)
    assert np.all(probability_map >= 0.0)

def test_source_cell_probability_map_is_normalized():
    model = build_l_mode_model()
    mesh = build_default_mesh(num_a=120, num_alpha=180)

    a_edges_m, alpha_edges_rad, cell_probability_map = build_source_cell_probability_map(model, mesh)

    assert len(a_edges_m) == mesh.num_a + 1
    assert len(alpha_edges_rad) == mesh.num_alpha + 1
    assert cell_probability_map.shape == (mesh.num_a, mesh.num_alpha)
    assert np.isclose(np.sum(cell_probability_map), 1.0)
    assert np.all(cell_probability_map >= 0.0)