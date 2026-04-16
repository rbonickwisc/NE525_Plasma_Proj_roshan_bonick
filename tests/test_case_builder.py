from __future__ import annotations

from tokamak_source_model.case_builder import build_default_geometry, build_default_mesh, build_generic_pedestal_model, build_l_mode_model, build_a_mode_paper_model

def test_default_geometry_is_physically_positive():
    geometry = build_default_geometry()

    assert geometry.major_radius_m > 0.0
    assert geometry.minor_radius_m > 0.0
    assert geometry.elongation > 0.0

def test_model_builders_return_expected_modes():
    l_model = build_l_mode_model()
    pedestal_model = build_generic_pedestal_model()
    a_model = build_a_mode_paper_model()

    assert l_model.profile.mode == "l_mode"
    assert pedestal_model.profile.mode == "pedestal"
    assert a_model.profile.mode == "pedestal"