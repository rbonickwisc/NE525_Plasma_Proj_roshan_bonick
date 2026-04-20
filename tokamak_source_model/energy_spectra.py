from __future__ import annotations

import numpy as np
from dataclasses import dataclass

# Neutron spectra energy calculations using Ballabio(2003)
# DT neutron energy at sampled ion temperature
# Ballabio gives E0 = m_alpha / (m_n + m_alpha) * Q
    # For DT this is approx 14.1 MeV
DT_E0_EV = 14.1e6

# Ballabio gives the DT FWHM as :
    # DeltaE_keV = 177 * sqrt(T_i_keV)
DT_FWHM_COEFF_KEV = 177.0

@dataclass(frozen=True)
class EnergySpectrumParameters:
    """
    Define parameters controlling neutron birth energy sampling
    """
    model: str = "muir_velocity_gaussian_dt"
    clip_min_ev: float = 1.0e3

def _validate_temperature_keV(T_i_keV: np.ndarray) -> np.ndarray:
    T_i_keV = np.asarray(T_i_keV, dtype=float)
    if np.any(T_i_keV < 0):
        raise ValueError("Ion temp must be non-negative")
    return T_i_keV

def dt_ballabio_fwhm_eV(T_i_keV: np.ndarray) -> np.ndarray:
    """
    DT neutron FWHM in eV from Ballabio thermal-spectrum approximation
    """
    T_i_keV = _validate_temperature_keV(T_i_keV)
    return 1.0e3 * DT_FWHM_COEFF_KEV * np.sqrt(T_i_keV)

def fwhm_to_sigma_eV(fwhm_eV: np.ndarray) -> np.ndarray:
    """
    Convert gausian FWHM to standard deviation sigma
    """
    fwhm_eV = np.asarray(fwhm_eV, dtype=float)
    return fwhm_eV / (2.0 * np.sqrt(2.0 * np.log(2.0)))

def dt_ballabio_mean_energy_eV(T_i_keV: np.ndarray) -> np.ndarray:
    """
    Mean DT neutron energy in eV
    """
    T_i_keV = _validate_temperature_keV(T_i_keV)
    return np.full_like(T_i_keV, DT_E0_EV, dtype=float)

def sample_muir_velocity_gaussian_dt_energies_eV(
    T_i_keV: np.ndarray,
    rng: np.random.Generator,
    clip_min_ev: float = 1.0e3,
) -> np.ndarray:
    """
    Sample DT neutron energies from the Ballabio thermal model
    """
    T_i_keV = _validate_temperature_keV(T_i_keV)

    mu_eV = dt_ballabio_mean_energy_eV(T_i_keV)
    sigma_eV = fwhm_to_sigma_eV(dt_ballabio_fwhm_eV(T_i_keV))

    energies_eV = rng.normal(loc=mu_eV, scale=sigma_eV)

    return np.maximum(energies_eV, clip_min_ev)


def sample_birth_energies_from_model_eV(
    T_i_keV: np.ndarray,
    spectrum: EnergySpectrumParameters,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Dispatch neutron birth energy sampling by model name
    """
    T_i_keV = _validate_temperature_keV(T_i_keV)

    if spectrum.model == "monoenergetic_dt":
        return np.full_like(T_i_keV, DT_E0_EV, dtype=float)
    
    if spectrum.model == "muir_velocity_gaussian_dt":
        return sample_muir_velocity_gaussian_dt_energies_eV(
            T_i_keV=T_i_keV,
            rng=rng,
            clip_min_ev=spectrum.clip_min_ev,
        )
