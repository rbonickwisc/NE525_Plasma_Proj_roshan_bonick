from __future__ import annotations

import numpy as np

from .parameters import GeometryParameters, ProfileParameters

def ion_density_profile_m3(
        a_m: np.ndarray,
        geometry: GeometryParameters,
        profile: ProfileParameters,
) -> np.ndarray:
    """
    Dispatch to correct ion density profile
    """
    if profile.mode == "l_mode":
        return l_mode_density_profile_m3(a_m, geometry, profile)
    if profile.mode == "pedestal":
        return pedestal_density_profile_m3(a_m, geometry, profile)
    raise ValueError(f"Unsupported profile mode: {profile.mode}")

def ion_temperature_profile_keV(
        a_m: np.ndarray,
        geometry: GeometryParameters,
        profile: ProfileParameters,
) -> np.ndarray:
    """
    Dispatch to correct ion temperature profile
    """
    if profile.mode == "l_mode":
        return l_mode_temperature_profile_keV(a_m, geometry, profile)
    if profile.mode == "pedestal":
        return pedestal_temperature_profile_keV(a_m, geometry, profile)
    raise ValueError(f"Unsupported profile mode: {profile.mode}")

def l_mode_density_profile_m3(
        a_m: np.ndarray,
        geometry: GeometryParameters,
        profile: ProfileParameters,
) -> np.ndarray:
    """
    L-mode density profile

        n_i(a) = n_i0 * (1 - (a/A)^2)^(alpha_n)
    """

    a_m = np.asarray(a_m, dtype=float)
    A = geometry.minor_radius_m
    ni0 = profile.ion_density_center_m3
    alpha_n = profile.alpha_n

    shape = 1.0 - (a_m / A) ** 2
    shape = np.clip(shape, 0.0, None)

    return ni0 * shape**alpha_n

def l_mode_temperature_profile_keV(
        a_m: np.ndarray,
        geometry: GeometryParameters,
        profile: ProfileParameters,
) -> np.ndarray:
    """
    L-mode temperature profile

        T_i(a) = T_i0 * (1 - (a/A)^2)^(alpha_T)
    """

    a_m = np.asarray(a_m, dtype=float)
    A = geometry.minor_radius_m
    Ti0 = profile.ion_temp_center_keV
    alpha_T = profile.alpha_T

    shape = 1.0 - (a_m / A) ** 2
    shape = np.clip(shape, 0.0, None)

    return Ti0 * shape**alpha_T

def pedestal_density_profile_m3(
        a_m: np.ndarray,
        geometry: GeometryParameters,
        profile: ProfileParameters,
) -> np.ndarray:
    """
    Pedestal density profile

    Core region:
        n_i = n_i,ped + (n_i0 - n_i,ped) * (1 - (a/a_ped)^2)^(alpha_n)

    Edge region:
        n_i = n_i,sep + (n_i,ped - n_i,sep) * (A -a)/(A - a_ped)
    """
    
    a_m = np.asarray(a_m, dtype=float)
    A = geometry.minor_radius_m

    aped = profile.pedestal_radius_m
    ni0 = profile.ion_density_center_m3
    niped = profile.ion_density_pedestal_m3
    nisep = profile.ion_density_separatrix_m3
    alpha_n = profile.alpha_n

    if aped is None or niped is None or nisep is None:
        raise ValueError("Pedestal density parameters must not be None")
    
    result = np.empty_like(a_m)

    core_mask = a_m <= aped
    edge_mask = ~core_mask

    core_shape = 1.0 - (a_m[core_mask] / aped) ** 2
    core_shape = np.clip(core_shape, 0.0, None)

    result[core_mask] = niped + (ni0 - niped) * core_shape**alpha_n
    result[edge_mask] = nisep + (niped - nisep) * (A - a_m[edge_mask]) / (A - aped)

    return result

def pedestal_temperature_profile_keV(
        a_m: np.ndarray,
        geometry: GeometryParameters,
        profile: ProfileParameters,
) -> np.ndarray:
    """
    Pedestal temperature profile

    Core region:
        T_i = T_i,ped + (Ti0 - T_i,ped) * (1 - (a/a_ped)^(beta_T))^(alpha_T)

    Edge region:
        T_i = T_i,sep + (T_i,ped - T_i,sep) * (A - a)/(A - a_ped)
    """
    
    a_m = np.asarray(a_m, dtype=float)
    A = geometry.minor_radius_m

    aped = profile.pedestal_radius_m
    Ti0 = profile.ion_temp_center_keV
    Tiped = profile.ion_temp_pedestal_keV
    Tisep = profile.ion_temp_separatrix_keV
    alpha_T = profile.alpha_T
    beta_T = profile.beta_T

    if aped is None or Tiped is None or Tisep is None or beta_T is None:
        raise ValueError("Pedestal temperature parameters must not be None")
    
    result = np.empty_like(a_m)

    core_mask = a_m <= aped
    edge_mask = ~core_mask

    core_shape = 1.0 - (a_m[core_mask] / aped) ** beta_T
    core_shape = np.clip(core_shape, 0.0, None)

    result[core_mask] = Tiped + (Ti0 - Tiped) * core_shape**alpha_T
    result[edge_mask] = Tisep + (Tiped - Tisep) * (A - a_m[edge_mask]) / (A - aped)

    return result