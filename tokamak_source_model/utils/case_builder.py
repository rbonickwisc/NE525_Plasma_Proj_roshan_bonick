from __future__ import annotations

from .parameters import FuelParameters, GeometryParameters, MeshParameters, ProfileParameters, SourceModelParameters
from .energy_spectra import EnergySpectrumParameters

def build_default_geometry() -> GeometryParameters:
    return GeometryParameters(
        major_radius_m=2.0,
        minor_radius_m=0.5,
        elongation=1.7,
        triangularity=0.33,
        shafranov_shift_m=0.1,
    )

def build_default_fuel(
    deuterium_fraction: float = 0.5,
    tritium_fraction: float = 0.5,
) -> FuelParameters:
    return FuelParameters(
        deuterium_fraction=deuterium_fraction,
        tritium_fraction=tritium_fraction,
    )

def build_default_mesh(
    num_a: int = 200,
    num_alpha: int = 360,
    num_R: int= 300,
    num_Z:int = 300,
    a_grid_min_m: float = 0.0,
) -> MeshParameters:
    return MeshParameters(
        num_a=num_a,
        num_alpha=num_alpha,
        num_R=num_R,
        num_Z=num_Z,
        a_grid_min_m=a_grid_min_m,
    )

def build_default_energy_spectrum() -> EnergySpectrumParameters:
    return EnergySpectrumParameters(model="muir_velocity_gaussian_dt")

def build_l_mode_profile() -> ProfileParameters:
    return ProfileParameters(
        mode="l_mode",
        ion_density_center_m3=8.0e19,
        ion_temp_center_keV=12.0,
        alpha_n=0.5,
        alpha_T=1.5,
    )

def build_generic_pedestal_profile(
    geometry: GeometryParameters | None = None,
) -> ProfileParameters:
    if geometry is None:
        geometry = build_default_geometry()

    return ProfileParameters(
        mode="pedestal",
        ion_density_center_m3=9.5e19,
        ion_temp_center_keV=22.0,
        alpha_n=1.0,
        alpha_T=4.5,
        pedestal_radius_m=0.8 * geometry.minor_radius_m,
        ion_density_pedestal_m3=8.5e19,
        ion_density_separatrix_m3=3.0e19,
        ion_temp_pedestal_keV=3.5,
        ion_temp_separatrix_keV=0.1,
        beta_T=5.0,
    )

def build_a_mode_paper_profile(
    geometry: GeometryParameters | None = None,
) -> ProfileParameters:
    if geometry is None:
        geometry = build_default_geometry()

    return ProfileParameters(
        mode="pedestal", 
        ion_density_center_m3=1.09e20, 
        ion_temp_center_keV=45.9,
        alpha_n=1.0,
        alpha_T=8.06,
        pedestal_radius_m=0.8 * geometry.minor_radius_m,
        ion_density_pedestal_m3=1.09e20,
        ion_density_separatrix_m3=3.0e19,
        ion_temp_pedestal_keV=6.09,
        ion_temp_separatrix_keV=0.1,
        beta_T=6.0,
    )

def build_l_mode_model(
    geometry: GeometryParameters | None = None,
    fuel: FuelParameters | None = None,
    energy_spectrum: EnergySpectrumParameters | None = None,
) -> SourceModelParameters:
    if geometry is None:
        geometry = build_default_geometry()
    if fuel is None:
        fuel = build_default_fuel()
    if energy_spectrum is None:
        energy_spectrum = build_default_energy_spectrum()

    return SourceModelParameters(
        geometry=geometry,
        profile=build_l_mode_profile(),
        fuel=fuel,
        energy_spectrum=energy_spectrum,
    )

def build_generic_pedestal_model(
    geometry: GeometryParameters | None = None,
    fuel: FuelParameters | None = None,
    energy_spectrum: EnergySpectrumParameters | None = None,
) -> SourceModelParameters:
    if geometry is None:
        geometry = build_default_geometry()
    if fuel is None:
        fuel = build_default_fuel()
    if energy_spectrum is None:
        energy_spectrum = build_default_energy_spectrum()

    return SourceModelParameters(
        geometry=geometry,
        profile=build_generic_pedestal_profile(geometry),
        fuel=fuel,
        energy_spectrum=energy_spectrum,
    )

def build_a_mode_paper_model(
    geometry: GeometryParameters | None = None,
    fuel: FuelParameters | None = None,
    energy_spectrum: EnergySpectrumParameters | None = None,
) -> SourceModelParameters:
    if geometry is None:
        geometry = build_default_geometry()
    if fuel is None:
        fuel = build_default_fuel()
    if energy_spectrum is None:
        energy_spectrum = build_default_energy_spectrum()

    return SourceModelParameters(
        geometry=geometry,
        profile=build_a_mode_paper_profile(geometry),
        fuel=fuel,
        energy_spectrum=energy_spectrum,
    )