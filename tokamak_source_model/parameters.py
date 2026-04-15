from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np

ProfileMode = Literal["l_mode", "pedestal"]

@dataclass(frozen=True)
class GeometryParameters:
    """
    Tokamak shaping parameters based on magnetic-surfaces.

    Parameters:
    ------------------------------------------------
    major_radius_m
        Major radius R0 [m]
    minor_radius_m
        Minor radius A of the last closed magnetic surface [m]
    elongation
        Elongation factor [-]
    triangularity
        Triangularity factor(delta) [-]
    shafranov_shift_m
        Shafranov shift factor [m]
    """

    major_radius_m: float
    minor_radius_m: float
    elongation: float
    triangularity: float
    shafranov_shift_m: float

@dataclass(frozen=True)
class ProfileParameters:
    """
    Plasma profile parameters

    Supports both l_mode and pedestal(H/A mode)

    For l_mode only center values and exponents are required
    For pedestal mode both the pedestal and separatrix fields are required as well
    """

    mode: ProfileMode

    ion_density_center_m3: float
    ion_temp_center_keV: float

    alpha_n: float
    alpha_T: float

    pedestal_radius_m: float | None = None
    ion_density_pedestal_m3: float | None = None
    ion_density_separatrix_m3: float | None = None
    ion_temp_pedestal_keV: float | None = None
    ion_temp_separatrix_keV: float | None = None
    beta_T: float | None = None

@dataclass(frozen=True)
class FuelParameters:
    """
    Fuel composition parameters
    """

    deuterium_fraction: float = 0.5
    tritium_fraction: float = 0.5

@dataclass(frozen=True)
class MeshParameters:
    """
    Numerical mesh settings

    num_a
        Number of radial-like a samples
    num_alpha
        Number of poloidal angle samples
    num_R, num_Z
        Reserved for later R-Z source maps
    a_grid_min_m
        Minimum a value for sampling [m]
    """

    num_a: int = 200
    num_alpha: int = 360

    num_R: int = 300
    num_Z: int = 300

    a_grid_min_m: float = 0.0

@dataclass(frozen=True)
class SourceModelParameters:
    """
    Wrapper for source model inputs
    """
    geometry: GeometryParameters
    profile: ProfileParameters
    fuel: FuelParameters

@dataclass
class ProfileEvaluation:
    """
    Container for evaluated 1D source-profile quantities on the a-grid
    """

    a_m: np.ndarray

    ion_density_m3: np.ndarray
    deuterium_density_m3: np.ndarray
    tritium_density_m3: np.ndarray

    ion_temp_keV: np.ndarray
    reactivity_m3_per_s: np.ndarray
    source_density_n_per_m3_per_s: np.ndarray