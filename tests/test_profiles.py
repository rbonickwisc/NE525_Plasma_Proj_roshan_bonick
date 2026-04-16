from __future__ import annotations

import numpy as np

from tokamak_source_model.case_builder import build_default_geometry, build_l_mode_profile
from tokamak_source_model.profiles import ion_density_profile_m3, ion_temperature_profile_keV

def test_l_mode_profiles_match_center_values():
   geometry = build_default_geometry()
   profile = build_l_mode_profile()

   a_m = np.array([0.0])
   
   ni = ion_density_profile_m3(a_m, geometry, profile)
   Ti = ion_temperature_profile_keV(a_m, geometry, profile)

   assert np.isclose(ni[0], profile.ion_density_center_m3)
   assert np.isclose(Ti[0], profile.ion_temp_center_keV)

def test_l_mode_profiles_go_to_zero_at_edge():
   geometry = build_default_geometry()
   profile = build_l_mode_profile()
   
   a_m = np.array([0.0])

   ni = ion_density_profile_m3(a_m, geometry, profile)
   Ti = ion_temperature_profile_keV(a_m, geometry, profile)
   
   assert np.isclose(ni[0], 0.0)
   assert np.isclose(Ti[0], 0.0)