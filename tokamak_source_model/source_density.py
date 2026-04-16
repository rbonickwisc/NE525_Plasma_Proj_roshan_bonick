from __future__ import annotations

import numpy as np

from .parameters import FuelParameters, ProfileEvaluation, SourceModelParameters
from .profiles import ion_density_profile_m3, ion_temperature_profile_keV
from .reactivity import dt_reactivity_m3_per_s

def deuterium_density_profile_m3(
        ion_density_m3: np.ndarray,
        fuel: FuelParameters,
) -> np.ndarray:
    """
    Compute deuterium density from total ion density and D fraction
    """
    return fuel.deuterium_fraction * ion_density_m3

def tritium_density_profile_m3(
        ion_density_m3: np.ndarray,
        fuel: FuelParameters,
) -> np.ndarray:
    """
    Compute tritium density from total ion density and T fraction
    """
    return fuel.tritium_fraction * ion_density_m3

def source_density_profile_n_per_m3_per_s(
        a_m: np.ndarray,
        model: SourceModelParameters,
) -> np.ndarray:
    """
    Compute the local DT neutron source density as a function of a:

        S(a) = n_D(a) * n_T(a) * <sigma v>_Dt(T_i(a))

    Returns
    ---------------
    np.ndarray
        Source density in neutrons / (m^3 s)
    """
    ion_density_m3 = ion_density_profile_m3(
        a_m=a_m,
        geometry=model.geometry,
        profile=model.profile,
    )
    ion_temp_keV = ion_temperature_profile_keV(
        a_m=a_m,
        geometry=model.geometry,
        profile=model.profile,
    )

    deuterium_density_m3 = deuterium_density_profile_m3(
        ion_density_m3=ion_density_m3,
        fuel=model.fuel,
    )
    tritium_density_m3 = tritium_density_profile_m3(
        ion_density_m3=ion_density_m3,
        fuel=model.fuel,
    )

    reactivity_m3_per_s = np.zeros_like(ion_temp_keV, dtype=float)
    positive_temp_mask = ion_temp_keV > 0.0

    if np.any(positive_temp_mask):
        reactivity_m3_per_s[positive_temp_mask] = dt_reactivity_m3_per_s(
            ion_temp_keV[positive_temp_mask]
        )

    source_density = ( deuterium_density_m3 * tritium_density_m3 * reactivity_m3_per_s)

    return source_density

def evaluate_profiles(
        a_m: np.ndarray,
        model: SourceModelParameters,
) -> ProfileEvaluation:
    """
    Evaluate all major 1D profile quantities on the given a-grid

    Returns
    -------------------
    ProfileEvaluation
        Container with n_i, n_D, n_T, T_i, reactivity, and source density
    """
    ion_density_m3 = ion_density_profile_m3(
        a_m=a_m,
        geometry=model.geometry,

        profile=model.profile,
    )
    ion_temp_keV = ion_temperature_profile_keV(
        a_m=a_m,
        geometry=model.geometry,
        profile=model.profile,
    )
    deuterium_density_m3 = deuterium_density_profile_m3(
        ion_density_m3=ion_density_m3,
        fuel=model.fuel,
    )
    tritium_density_m3 = tritium_density_profile_m3(
        ion_density_m3=ion_density_m3,
        fuel=model.fuel,
    )

    reactivity_m3_per_s = np.zeros_like(ion_temp_keV, float)
    positive_temp_mask = ion_temp_keV > 0.0

    if np.any(positive_temp_mask):
        reactivity_m3_per_s[positive_temp_mask] = dt_reactivity_m3_per_s(
            ion_temp_keV[positive_temp_mask]
        )

    source_density_n_per_m3_per_s = (deuterium_density_m3 * tritium_density_m3 * reactivity_m3_per_s)

    return ProfileEvaluation(
        a_m=np.asarray(a_m, dtype=float),
        ion_density_m3=ion_density_m3,
        deuterium_density_m3=deuterium_density_m3,
        tritium_density_m3=tritium_density_m3,
        ion_temp_keV=ion_temp_keV,
        reactivity_m3_per_s=reactivity_m3_per_s,
        source_density_n_per_m3_per_s=source_density_n_per_m3_per_s,
    )