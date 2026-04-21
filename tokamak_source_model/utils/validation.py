from __future__ import annotations

import math

from .parameters import FuelParameters, GeometryParameters, ProfileParameters, SourceModelParameters, EnergySpectrumParameters

def validate_geometry_parameters(geometry: GeometryParameters) -> None:
    if geometry.major_radius_m <= 0.0:
        raise ValueError("major_radius_m must be > 0")
    if geometry.minor_radius_m <= 0.0:
        raise ValueError("minor_radius_m must be > 0")
    if geometry.elongation <= 0.0:
        raise ValueError("elongation must be > 0")
    if not (0.0 <= geometry.triangularity < 1.0):
        raise ValueError("triangularity must satisfy 0 <= triangularity < 1")
    if geometry.shafranov_shift_m < 0.0:
        raise ValueError("shafranov_shift_m must be >= 0")


def validate_profile_parameters(
    profile: ProfileParameters,
    geometry: GeometryParameters,
) -> None:
    if profile.mode not in ("l_mode", "pedestal"):
        raise ValueError("profile.mode must be 'l_mode' or 'pedestal'")

    if profile.ion_density_center_m3 <= 0.0:
        raise ValueError("ion_density_center_m3 must be > 0")
    if profile.ion_temp_center_keV <= 0.0:
        raise ValueError("ion_temp_center_keV must be > 0")
    if profile.alpha_n < 0.0:
        raise ValueError("alpha_n must be >= 0")
    if profile.alpha_T < 0.0:
        raise ValueError("alpha_T must be >= 0")

    if profile.mode == "pedestal":
        required = {
            "pedestal_radius_m": profile.pedestal_radius_m,
            "ion_density_pedestal_m3": profile.ion_density_pedestal_m3,
            "ion_density_separatrix_m3": profile.ion_density_separatrix_m3,
            "ion_temp_pedestal_keV": profile.ion_temp_pedestal_keV,
            "ion_temp_separatrix_keV": profile.ion_temp_separatrix_keV,
            "beta_T": profile.beta_T,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                f"Pedestal mode requires these non-None fields: {', '.join(missing)}"
            )

        assert profile.pedestal_radius_m is not None
        assert profile.ion_density_pedestal_m3 is not None
        assert profile.ion_density_separatrix_m3 is not None
        assert profile.ion_temp_pedestal_keV is not None
        assert profile.ion_temp_separatrix_keV is not None
        assert profile.beta_T is not None

        if not (0.0 < profile.pedestal_radius_m <= geometry.minor_radius_m):
            raise ValueError("pedestal_radius_m must satisfy 0 < aped <= minor_radius_m")
        if profile.ion_density_pedestal_m3 < 0.0:
            raise ValueError("ion_density_pedestal_m3 must be >= 0")
        if profile.ion_density_separatrix_m3 < 0.0:
            raise ValueError("ion_density_separatrix_m3 must be >= 0")
        if profile.ion_temp_pedestal_keV <= 0.0:
            raise ValueError("ion_temp_pedestal_keV must be > 0")
        if profile.ion_temp_separatrix_keV <= 0.0:
            raise ValueError("ion_temp_separatrix_keV must be > 0")
        if profile.beta_T <= 0.0:
            raise ValueError("beta_T must be > 0")

        if profile.ion_density_center_m3 < profile.ion_density_pedestal_m3:
            raise ValueError(
                "Expected ion_density_center_m3 >= ion_density_pedestal_m3"
            )
        if profile.ion_density_pedestal_m3 < profile.ion_density_separatrix_m3:
            raise ValueError(
                "Expected ion_density_pedestal_m3 >= ion_density_separatrix_m3"
            )
        if profile.ion_temp_center_keV < profile.ion_temp_pedestal_keV:
            raise ValueError(
                "Expected ion_temp_center_keV >= ion_temp_pedestal_keV"
            )
        if profile.ion_temp_pedestal_keV < profile.ion_temp_separatrix_keV:
            raise ValueError(
                "Expected ion_temp_pedestal_keV >= ion_temp_separatrix_keV"
            )


def validate_fuel_parameters(fuel: FuelParameters) -> None:
    if not (0.0 <= fuel.deuterium_fraction <= 1.0):
        raise ValueError("deuterium_fraction must satisfy 0 <= f_D <= 1")
    if not (0.0 <= fuel.tritium_fraction <= 1.0):
        raise ValueError("tritium_fraction must satisfy 0 <= f_T <= 1")

    if not math.isclose(
        fuel.deuterium_fraction + fuel.tritium_fraction,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("deuterium_fraction + tritium_fraction must equal 1")

def validate_energy_spectrum_parameters(spectrum: EnergySpectrumParameters) -> None:
    if spectrum.model not in {"muir_velocity_gaussian_dt", "monoenergetic_dt"}:
        raise ValueError("Must use either muir_velocity or monoenergetic")
    if spectrum.clip_min_ev <= 0.0:
        raise ValueError("clip_min_ev must be greater than 0.0")

def validate_source_model_parameters(model: SourceModelParameters) -> None:
    validate_geometry_parameters(model.geometry)
    validate_profile_parameters(model.profile, model.geometry)
    validate_fuel_parameters(model.fuel)
    validate_energy_spectrum_parameters(model.energy_spectrum)