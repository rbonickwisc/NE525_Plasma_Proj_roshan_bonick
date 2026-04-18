from tokamak_source_model.energy_spectra import dt_ballabio_fwhm_eV, fwhm_to_sigma_eV

T_i_keV = 10


print("T_i_keV[0] =", T_i_keV)
print("FWHM_eV[0] =", dt_ballabio_fwhm_eV(T_i_keV))
print("sigma_eV[0] =", fwhm_to_sigma_eV(dt_ballabio_fwhm_eV(T_i_keV)))