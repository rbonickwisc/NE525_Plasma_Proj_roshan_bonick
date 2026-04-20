from __future__ import annotations

from typing import Any
from dataclasses import fields, replace
import numpy as np

from .case_builder import build_default_mesh
from .energy_spectra import EnergySpectrumParameters
from .openmc_adapter import build_openmc_independent_sources
from .parameters import FuelParameters, GeometryParameters, ProfileParameters, SourceModelParameters

def tokamak_source(
    *,
    mode: str,
    major_radius_m: float,
    minor_radius_m: float,
    elongation: float,
    triangularity: float,
    shafranov_shift_m: float,
    ion_density_center_m3: float,
    ion_temp_center_keV: float,
    alpha_n: float,
    alpha_T: float,
    pedestal_radius_m: float,
    ion_density_pedestal_m3: float,
    ion_density_separatrix_m3: float,
    ion_temp_pedestal_keV: float,
    ion_temp_separatrix_keV: float,
    beta_T: float,
    deuterium_fraction: float,
    tritium_fraction: float,
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
    Build OpenMC Independant Source objects directly from tokamak source parameters

    Args:

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
    if ion_temp_center_keV <= 0.0:
        raise ValueError("ion temp as center must be > 0")
    if deuterium_fraction <= 0.0 or tritium_fraction < 0.0:
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