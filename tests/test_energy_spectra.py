from __future__ import annotations
import numpy as np

from tokamak_source_model.energy_spectra import DT_E0_EV, EnergySpectrumParameters, dt_ballabio_fwhm_eV, fwhm_to_sigma_eV, sample_birth_energies_from_model_eV, dt_ballabio_mean_energy_eV, dt_bosch_hale_theta_keV

def test_dt_ballabio_fwhm_zero_at_zero_temperatures():
    T_i_keV = np.array([0.0])
    fwhm_eV = dt_ballabio_fwhm_eV(T_i_keV)
    assert np.isclose(fwhm_eV[0], 0.0)

def test_dt_ballabio_fwhm_matches_tablle_value():
    T_i_keV = np.array([1.0, 4.0, 9.0])
    fwhm_eV = dt_ballabio_fwhm_eV(T_i_keV)

    expected_eV = 1.0e3 * 177.259 * np.sqrt(T_i_keV)
    assert np.allclose(fwhm_eV, expected_eV)

def test_fwhm_to_sigma_is_correct():
    fwhm_eV = np.array([1000.0])
    sigma_eV = fwhm_to_sigma_eV(fwhm_eV)

    expected = 1000.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    assert np.isclose(sigma_eV[0], expected)

def test_mg_dt_sampling_has_reasonable_mean_nonzero_spread():
    rng = np.random.default_rng(123)
    T_i_keV = np.full(50000, 10.0)

    spectrum = EnergySpectrumParameters(model="muir_velocity_gaussian_dt")
    energies_eV = sample_birth_energies_from_model_eV(T_i_keV, spectrum, rng)

    assert np.mean(energies_eV) > 1.39e7
    assert np.mean(energies_eV) < 1.43e7
    assert np.std(energies_eV) > 1.0e5

def test_doppler_broadening_exists():
    rng = np.random.default_rng(123)

    spectrum = EnergySpectrumParameters(model="muir_velocity_gaussian_dt")

    low_T = np.full(30000, 5.0)
    high_T = np.full(30000, 20.0)

    low_E = sample_birth_energies_from_model_eV(low_T, spectrum, rng)
    high_E = sample_birth_energies_from_model_eV(high_T, spectrum, rng)

    assert np.std(high_E) > np.std(low_E)

def test_dt_ballabio_mean_energy_matches_E0_at_zero_temperature():
    T_i_keV = np.array([0.0])
    mean_eV = dt_ballabio_mean_energy_eV(T_i_keV)
    assert np.isclose(mean_eV[0], 14.021e6)


def test_dt_ballabio_mean_energy_increases_with_temperature():
    T_i_keV = np.array([0.0, 10.0, 20.0])
    mean_eV = dt_ballabio_mean_energy_eV(T_i_keV)

    assert mean_eV[1] > mean_eV[0]
    assert mean_eV[2] > mean_eV[1]


def test_dt_ballabio_fwhm_matches_table_value():
    T_i_keV = np.array([1.0])
    fwhm_eV = dt_ballabio_fwhm_eV(T_i_keV)
    assert np.isclose(fwhm_eV[0], 177.259e3)


def test_dt_bosch_hale_theta_is_positive():
    T_i_keV = np.array([0.2, 1.0, 10.0, 20.0, 100.0])
    theta_keV = dt_bosch_hale_theta_keV(T_i_keV)
    assert np.all(theta_keV > 0.0)