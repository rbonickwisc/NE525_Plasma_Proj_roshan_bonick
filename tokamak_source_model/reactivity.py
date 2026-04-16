from __future__ import annotations

import numpy as np

# Sadler-Van Belle coefficients
C1 = 2.5663271e-18
C2 = 19.983026
C3 = 2.5077133e-2
C4 = 2.5773408e-3
C5 = 6.1880463e-5
C6 = 6.6024089e-2
C7 = 8.1215505e-3

def dt_reactivity_m3_per_s(
        ion_temp_keV: np.ndarray | float,
) -> np.ndarray:
    """
    Compute DT Maxwellian reactivity <sigma v> using Sadler-Van Belle fit

    Validity range:
        0 < Ti <= 100 keV

    Parameters
    -----------------------
    ion_temp_keV
        Ion temperature in keV

    Returns
    -----------------------
    np.ndarray
        DT reactivity in m^3 / s
    """
    Ti = np.asarray(ion_temp_keV, dtype=float)

    if np.any(Ti <= 0.0):
        raise ValueError("Ion temperature must be > 0 keV for DT reactivity")
    if np.any(Ti > 100.0):
        raise ValueError("Sadler-Van Belle DT reactivity fit only valid for Ti <= 100 keV")
    
    numerator = C3 + Ti * (C4 - C5 * Ti)
    denominator = 1.0 + Ti * (C6 + C7 * Ti)

    U = 1.0 - Ti * numerator / denominator

    if np.any(U <= 0.0):
        raise ValueError("Computed U became non-positive, recheck input temperature")
    
    reactivity = (
        C1 / (U ** (5.0/6.0) * Ti ** (2.0/3.0)) * np.exp(-C2 * (U/Ti) ** (1.0/3.0))
    )

    return reactivity