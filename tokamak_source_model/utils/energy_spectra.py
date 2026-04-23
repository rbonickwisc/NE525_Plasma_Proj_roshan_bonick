from __future__ import annotations

import numpy as np
from dataclasses import dataclass

# DT neutron zero-temp from Ballabio 1998
DT_E0_EV = 14.021e6

# DT width coefficient from Ballabio 1998
DT_FWHM_COEFF_KEV_SQRT_KEV = 177.259

# Bosch-Hale 1992 DT reactivity-fit coefficients
DT_BG = 34.3827
DT_C1 = 1.17302e-9
DT_C2 = 1.51361e-2
DT_C3 = 7.51886e-2
DT_C4 = 4.60643e-3
DT_C5 = 1.35000e-2
DT_C6 = -1.06750e-4
DT_C7 = 1.36600e-5

# Rest mass energy equivalents in keV
M_D_KEV = 1875.61294500e3
M_T_KEV = 2808.92113668e3
M_N_KEV = 939.56542194e3
M_A_KEV = 3727.3794118e3

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


def dt_bosch_hale_theta_keV(T_i_keV: np.ndarray) -> np.ndarray:
    """
    Bosch-Hale effective temp parameter theta(T) for DT

        theta = T / (1 - T*(C2 + T*(C4 + T*C6)) / (1 + T*(C3 + T*(C5 + T*C7))))
    T and theta in keV
    """
    T_i_keV = _validate_temperature_keV(T_i_keV)

    numerator = T_i_keV * (DT_C2 + T_i_keV * (DT_C4 + T_i_keV * DT_C6))
    denominator = 1.0 + T_i_keV * (DT_C3 + T_i_keV * (DT_C5 + T_i_keV * DT_C7))

    return T_i_keV / (1.0 - numerator / denominator)

def dt_bosch_hale_dtheta_dT(T_i_keV: np.ndarray) -> np.ndarray:
    """
    Derivative dtheta/dT for DT Bosch-Hale theta(T)
    """
    T_i_keV = _validate_temperature_keV(T_i_keV)

    # a(T) = T * (C2 + T*(C4 + T*C6))
    a = T_i_keV * (DT_C2 + T_i_keV * (DT_C4 + T_i_keV * DT_C6))
    da = DT_C2 + 2.0 * DT_C4 * T_i_keV + 3.0 * DT_C6 * T_i_keV**2

    #b(T) = 1 + T * (C3 + T*(C5 + T*C7))
    b = 1.0 + T_i_keV * (DT_C3 + T_i_keV * (DT_C5 + T_i_keV * DT_C7))
    db = DT_C3 + 2.0 * DT_C5 * T_i_keV + 3.0 * DT_C7 * T_i_keV**2

    g = 1.0 - a / b
    dg = -(da * b - a * db) / (b**2)
    
    return (g - T_i_keV * dg) / (g**2)

def dt_bosch_hale_xi(T_i_keV: np.ndarray) -> np.ndarray:
    """
    Bosch-Hale xi(T), used in Ballabio for exact <K> expression
    """
    theta_keV = dt_bosch_hale_theta_keV(T_i_keV)
    return (DT_BG**2 / (4.0 * theta_keV)) ** (1.0 / 3.0)

def dt_ballabio_mean_K_keV(T_i_keV: np.ndarray) -> np.ndarray:
    """
    Cross section weighted <K> from Ballabio 1998
        <K> = (T_i^2 / theta) * (5/6 + xi) * dtheta/dT_i
    """
    T_i_keV = _validate_temperature_keV(T_i_keV)

    mean_K_keV = np.zeros_like(T_i_keV, dtype=float)

    positive = T_i_keV > 0.0
    if not np.any(positive):
        return mean_K_keV

    T_pos = T_i_keV[positive]
    theta_keV = dt_bosch_hale_theta_keV(T_pos)
    xi = dt_bosch_hale_xi(T_pos)
    dtheta_dT = dt_bosch_hale_dtheta_dT(T_pos)

    mean_K_keV[positive] = (T_pos**2 / theta_keV) * ((5.0 / 6.0) + xi) * dtheta_dT
    return mean_K_keV

def dt_ballabio_mean_vcm2(T_i_keV: np.ndarray) -> np.ndarray:
    """
    <v_CM^2> from Ballabio 1998

        <v_VM^2> = 3 T_i / (m_d + m_t)
    """
    T_i_keV = _validate_temperature_keV(T_i_keV)
    return 3.0 * T_i_keV / (M_D_KEV + M_T_KEV)

def dt_ballabio_mean_energy_eV(T_i_keV: np.ndarray) -> np.ndarray:
    """
    DT neutron mean energy from Bellabio relativistic first moment

    <E_n> = E0 + DeltaE_th
    with:
        E0 = ((m_d+m_t)^2 + m_n^2 - m_a^2) / (2 (m_d+m_t)) - m_n

        DeltaE_th =
            [((m_d+m_t)^2 - m_n^2 + m_a^2) / (2 (m_d+m_t)^2)] <K>
          + [((m_d+m_t)^2 + m_n^2 - m_a^2) / (4 (m_d+m_t))] <v_CM^2>

    Outputs in [eV]
    """
    T_i_keV = _validate_temperature_keV(T_i_keV)

    msum = M_D_KEV + M_T_KEV

    coeff_K = (msum**2 - M_N_KEV**2 + M_A_KEV**2) / (2.0 * msum**2)
    coeff_vcm2 = (msum**2 + M_N_KEV**2 - M_A_KEV**2) / (4.0 * msum)

    mean_K_keV = dt_ballabio_mean_K_keV(T_i_keV)
    mean_vcm2 = dt_ballabio_mean_vcm2(T_i_keV)

    mean_energy_keV = (DT_E0_EV / 1.0e3) + coeff_K * mean_K_keV + coeff_vcm2 * mean_vcm2

    return 1.0e3 * mean_energy_keV

def dt_ballabio_fwhm_eV(T_i_keV: np.ndarray) -> np.ndarray:
    """
    DT neutron FWHM from Ballabio 1998

        W_1/2 = omega0 * sqrt(T_i)
    """
    T_i_keV = _validate_temperature_keV(T_i_keV)
    return 1.0e3 * DT_FWHM_COEFF_KEV_SQRT_KEV * np.sqrt(T_i_keV)

def fwhm_to_sigma_eV(fwhm_eV: np.ndarray) -> np.ndarray:
    fwhm_eV = np.asarray(fwhm_eV, dtype=float)
    return fwhm_eV / (2.0 * np.sqrt(2.0 * np.log(2.0)))

def sample_muir_velocity_gaussian_dt_energies_eV(
    T_i_keV: np.ndarray,
    rng: np.random.Generator,
    clip_min_ev: float = 1.0e3,
) -> np.ndarray:
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
    T_i_keV = _validate_temperature_keV(T_i_keV)

    if spectrum.model == "monoenergetic_dt":
        return np.full_like(T_i_keV, DT_E0_EV, dtype=float)
    if spectrum.model == "muir_velocity_gaussian_dt":
        return sample_muir_velocity_gaussian_dt_energies_eV(
            T_i_keV=T_i_keV,
            rng=rng,
            clip_min_ev=spectrum.clip_min_ev,
        )
    
    raise ValueError(f"Unsupported energy spectrum model: {spectrum.model}")