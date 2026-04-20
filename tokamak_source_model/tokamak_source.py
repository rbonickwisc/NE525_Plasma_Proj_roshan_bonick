from __future__ import annotations

from typing import Any
from dataclasses import fields, replace
import numpy as np

from .utils.case_builder import build_default_mesh
from .utils.energy_spectra import EnergySpectrumParameters
from .utils.openmc_adapter import build_openmc_independent_sources
from .utils.parameters import FuelParameters, GeometryParameters, ProfileParameters, SourceModelParameters

def tokamak_source(
    *,
    mode: str,
    major_radius_m: float,
    minor_radius_m: float,
    elongation: float,
    triangularity: float = 0.0,
    shafranov_shift_m: float = 0.0,
    ion_density_center_m3: float,
    ion_temp_center_keV: float,
    alpha_n: float = 1.0,
    alpha_T: float = 1.0,
    pedestal_radius_m: float | None = None,
    ion_density_pedestal_m3: float| None = None,
    ion_density_separatrix_m3: float| None = None,
    ion_temp_pedestal_keV: float| None = None,
    ion_temp_separatrix_keV: float| None = None,
    beta_T: float| None = None,
    deuterium_fraction: float = 0.5,
    tritium_fraction: float = 0.5,
    n_samples: int = 1000,
    num_a: int = 200,
    num_alpha: int = 360,
    num_R: int = 300,
    num_Z: int = 300,
    a_grid_min_m: float = 0.0,
    rng: np.random.Generator | None = None,
    energy_model: str = "muir_velocity_gaussian_dt",
) -> list:
    """
    Build OpenMC Independent Source objects directly from tokamak source parameters

    Parameters
    ------------
        Mode: Plasma confinement mode
        major_radius_m: Plasma major radius
        minor_radius_m: Plasma minor radius
        elongation: Plasma elongation
        triangularity: Plasma triangularity
        shafranov_shift_m: Outward radial shift of magnetic surfaces
        ion_density_center_m3: Ion density at plasma center
        ion_temp_center_keV: Ion temperature at the plasma center
        alpha_n: Ion density profile exponent
            In L-mode this controls how sharply the density falls from the center to the plasma edge
            In H/A mode this controls the core-side density shape inside the pedestal radius
        alpha_T: Ion tempereature profile exponent
            Functions the same as alpha_n just for temperature instead of density
        pedestal_radius_m: Pedestal radius, minor-radius location where the pedestal region begins
        ion_density_pedestal_m3: Ion density at the pedestal
        ion_density_separatrix_m3: Ion density at the separatrix
        ion_temp_pedestal_keV: Ion temperature at the pedestal
        ion_temp_separatrix_keV: Ion temperature at the separatrix
        beta_T: Temperature pedestal exponent used in H and A modes
        deuterium_fraction: Atomic fraction of deuterium in the D-T fuel mixture
        tritium_fraction: Atomic fraction of tritium in the D-T fuel mixture
        n_samples: Number of sampled neutron birth sources to generate
        num_a: Number of bins in the magnetic surface radius direction used to build source probability map
            defaults to 200
        num_alpha: Number of bins in the poloidal angle direction used to build source probability map
            defaults to 360
        num_R: Number of bins in the R dir used for mesh building
            defaults to 300
        num_Z: Number of bins in Z dir used for mesh building
            defaults to 300
        a_grid_min_m: Minimum magnetic surface radius value used in source mesh
            defaults to 0
        rng: np.random.Generator: NumPy random number gen used for source sampling
            if set to None a default genarator is created
        energy_model: Neutron birth energy model
            default is "muir_velocity_gaussian_dt" which is the Ballabio-based thermal DT spectrum model
    
    Returns a list of OpenMC 'IndependentSource' objects
    """
    mode = mode.upper()
    if mode not in {"L", "H", "A"}:
        raise  ValueError("mode must be one of {'L', 'H', 'A'}")
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    if num_a <= 1:
        raise ValueError("num_a must be > 1")
    if num_alpha <= 1:
        raise  ValueError("num_alpha must be > 1")
    if major_radius_m <= 0.0:
        raise ValueError("major radius must be > 0")
    if minor_radius_m <= 0.0:
        raise ValueError("minor radius must be > 0")
    if elongation <= 0.0:
        raise ValueError("elongation must be > 0")
    if ion_density_center_m3 <= 0.0:
        raise ValueError("ion density as center must be > 0")
    if ion_temp_center_keV < 0.0:
        raise ValueError("ion temp as center must be > 0")
    if deuterium_fraction < 0.0 or tritium_fraction < 0.0:
        raise ValueError("Fuel fractions must be non-negative")
    if not np.isclose(deuterium_fraction + tritium_fraction, 1.0):
        raise ValueError("Fuel fractions must equal 1")
    
    if rng is None:
        rng = np.random.default_rng()

    geometry = GeometryParameters(
        major_radius_m=major_radius_m,
        minor_radius_m=minor_radius_m,
        elongation=elongation,
        triangularity=triangularity,
        shafranov_shift_m=shafranov_shift_m,
    )

    if mode == "L":
        profile = ProfileParameters(
            mode="l_mode",
            ion_density_center_m3=ion_density_center_m3,
            ion_temp_center_keV=ion_temp_center_keV,
            alpha_n=alpha_n,
            alpha_T=alpha_T,
        )
    else:
        if pedestal_radius_m is None:
            raise ValueError("pedestal radius required for H/A mode")
        if ion_density_pedestal_m3 is None:
            raise ValueError("pedestal ion density required for H/A mode")
        if ion_density_separatrix_m3 is None:
            raise ValueError("separatrix ion density required for H/A mode")
        if ion_temp_pedestal_keV is None:
            raise ValueError("pedestal temperature required for H/A mode")
        if ion_temp_separatrix_keV is None:
            raise ValueError("separatrix temperature required for H/A mode")
        if beta_T is None:
            raise ValueError("beta_T required for H/A mode")
        
        if pedestal_radius_m <= 0.0:
            raise ValueError("pedestal radius must be > 0")
        if pedestal_radius_m > minor_radius_m:
            raise ValueError("pedestal radius must be <= minor radius")
        if ion_density_pedestal_m3 <= 0.0:
            raise ValueError("pedestal ion density must be > 0")
        if ion_density_separatrix_m3 <=0.0:
            raise ValueError("separatrix ion density must be > 0")
        if ion_temp_pedestal_keV <=0.0:
            raise ValueError("pedestal temperature must be > 0")
        if ion_temp_separatrix_keV <=0.0:
            raise ValueError("separatrix temperature must be > 0")
        if beta_T <=0.0:
            raise ValueError("beta_T required must be > 0")
        
        profile = ProfileParameters(
            mode="pedestal",
            ion_density_center_m3=ion_density_center_m3,
            ion_temp_center_keV=ion_temp_center_keV,
            alpha_n=alpha_n,
            alpha_T=alpha_T,
            pedestal_radius_m=pedestal_radius_m,
            ion_density_pedestal_m3=ion_density_pedestal_m3,
            ion_density_separatrix_m3=ion_density_separatrix_m3,
            ion_temp_pedestal_keV=ion_temp_pedestal_keV,
            ion_temp_separatrix_keV=ion_temp_separatrix_keV,
            beta_T=beta_T,
        )
    
    fuel = FuelParameters(
        deuterium_fraction=deuterium_fraction,
        tritium_fraction=tritium_fraction,
    )

    energy_spectrum = EnergySpectrumParameters(model=energy_model)

    model = SourceModelParameters(
        geometry=geometry,
        profile=profile,
        fuel=fuel,
        energy_spectrum=energy_spectrum,
    )

    mesh = build_default_mesh(
        num_a=num_a,
        num_alpha=num_alpha,
        num_R=num_R,
        num_Z=num_Z,
        a_grid_min_m=a_grid_min_m,
    )

    return build_openmc_independent_sources(
        n_samples=n_samples,
        model=model,
        mesh=mesh,
        rng=rng,
    )