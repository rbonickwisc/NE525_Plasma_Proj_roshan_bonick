from __future__ import annotations

import numpy as np

from .parameters import FuelParameters, ProfileEvaluation, SourceModelParameters
from .profiles import ion_density_profile_m3, ion_temperature_profile_keV
from .reactivity import dt_reactivity_m3_per_s

def deuterium_density_profile_m3(
        ion_density_m3: np.ndarray,
        fuel,
) -> np.ndarray:
    """
    Convert Fausser paper convention ion density n_i into deuterium density n_D

    Convention used:
    - ion_density_m3 stores Fausser paper's n_i profile
    - for a 50/50 DT mix, n_D = n_T = n_i
    - for a general mix, n_D = 2 * f_D * n_i and n_T = 2 * f_T * n_i

    This guarantees that for f_D = f_T = 0.5:
        S = n_D * n_T * <sigma v> = n_i^2 * <sigma v>
    which matches Eq. (1) in Fausser paper
    """
    return 2.0 * fuel.deuterium_fraction * ion_density_m3

def tritium_density_profile_m3(
        ion_density_m3: np.ndarray,
        fuel: FuelParameters,
) -> np.ndarray:
    """
    Convert Fausser paper convention ion density n_i into tritium density n_T
    """
    return 2.0 * fuel.tritium_fraction * ion_density_m3

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
    # Important density convention:
        # ion_density_m3 in this project follows the Fausser paper convention for n_i
        # For a 50/50 D-T plasma, the paper writes S = n_i^2 * <sigma v>
        # To reproduce this while still allowing arbitrary D/T fractions
        # Interpret:
            # n_D = 2 * f_D * n_i
            # n_T = 2 * f_T * n_i
        # So that:
            # S = n_D*n_T*<sigma v> = 4*f_D*f_T*n_i^2*<sigma v>
        # and for f_D = f_T = 0.5 this reduces to:
            # S = n_i^2 * <sigma v>

            
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