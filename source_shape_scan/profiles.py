from __future__ import annotations

from tokamak_source_model.utils.parameters import GeometryParameters, ProfileParameters, SourceModelParameters
from tokamak_source_model.utils.case_builder import build_default_energy_spectrum, build_default_fuel

def build_scan_geometry() -> GeometryParameters:
    return GeometryParameters(
        major_radius_m=2.0,
        minor_radius_m=0.5,
        elongation=1.7,
        triangularity=0.33,
        shafranov_shift_m=0.1,
    )

def build_l_mode_reference() -> SourceModelParameters:
    geometry = build_scan_geometry()

    profile = ProfileParameters(
        mode="l_mode",
        ion_density_center_m3=8.0e19,
        ion_temp_center_keV=12.0,
        alpha_n=0.5,
        alpha_T=1.5,
    )

    return SourceModelParameters(
        geometry=geometry,
        profile=profile,
        fuel=build_default_fuel(),
        energy_spectrum=build_default_energy_spectrum(),
    )

def build_a_mode_from_parameters(
    pedestal_fraction: float = 0.8,
    alpha_n: float = 1.0,
    alpha_T: float = 8.06,
    beta_T: float = 6.0,
    ion_density_center_m3: float = 8.0e19,
    ion_temp_center_keV: float = 12.0,
    ion_density_pedestal_m3: float | None = None,
    ion_density_separatrix_m3: float = 3.0e19,
    ion_temp_pedestal_keV: float = 6.09,
    ion_temp_separatrix_keV: float = 0.1,
) -> SourceModelParameters:
    geometry = build_scan_geometry()

    if ion_density_pedestal_m3 is None:
        ion_density_pedestal_m3 = ion_density_center_m3

    profile = ProfileParameters(
        mode="pedestal",
        ion_density_center_m3=ion_density_center_m3,
        ion_temp_center_keV=ion_temp_center_keV,
        alpha_n=alpha_n,
        alpha_T=alpha_T,
        pedestal_radius_m=pedestal_fraction * geometry.minor_radius_m,
        ion_density_pedestal_m3=ion_density_pedestal_m3,
        ion_density_separatrix_m3=ion_density_separatrix_m3,
        ion_temp_pedestal_keV=ion_temp_pedestal_keV,
        ion_temp_separatrix_keV=ion_temp_separatrix_keV,
        beta_T=beta_T,
    )

    return SourceModelParameters(
        geometry=geometry,
        profile=profile,
        fuel=build_default_fuel(),
        energy_spectrum=build_default_energy_spectrum(),
    )