from __future__ import annotations

from pathlib import Path

import numpy as np

from tokamak_source_model.geometry import make_a_alpha_grids
from tokamak_source_model.parameters import (
    FuelParameters, 
    GeometryParameters,
    MeshParameters,
    ProfileParameters,
    SourceModelParameters,
)

from tokamak_source_model.plotting import plot_magnetic_surfaces, plot_profiles_vs_a
from tokamak_source_model.profiles import ion_density_profile_m3, ion_temperature_profile_keV


def main() -> None:
    output_dir = Path("studies/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    geometry = GeometryParameters(
        major_radius_m=2.0,
        minor_radius_m=0.5,
        elongation=1.7,
        triangularity=0.33,
        shafranov_shift_m=0.1,
    )

    profile = ProfileParameters(
        mode="l_mode",
        ion_density_center_m3=2.0e20,
        ion_temp_center_keV=15.0,
        alpha_n=0.5,
        alpha_T=1.0,
    )

    fuel = FuelParameters(
        deuterium_fraction=0.5,
        tritium_fraction=0.5,
    )

    model = SourceModelParameters(
        geometry=geometry,
        profile=profile,
        fuel=fuel,
    )

    mesh = MeshParameters(
        num_a=200,
        num_alpha=360,
    )

    a_grid_m, alpha_grid_rad = make_a_alpha_grids(model.geometry, mesh)

    ni_m3 = ion_density_profile_m3(a_grid_m, model.geometry, model.profile)
    Ti_keV = ion_temperature_profile_keV(a_grid_m, model.geometry, model.profile)

    print("L-mode demo case")
    print("----------------")
    print(f"n_i(center) = {ni_m3[0]:.6e} m^-3")
    print(f"T_i(center) = {Ti_keV[0]:.6e} keV")
    print(f"n_i(edge) = {ni_m3[-1]:.6e} m^-3")
    print(f"T_i(edge) = {Ti_keV[-1]:.6e} keV")

    surface_radii_m = np.linspace(0.1 * geometry.minor_radius_m, geometry.minor_radius_m, 8)
    
    plot_magnetic_surfaces(
        geometry=geometry,
        surface_radii_m=surface_radii_m,
        alpha_rad=alpha_grid_rad,
        output_path=output_dir / "magnetic_surfaces.png",
    )

    plot_profiles_vs_a(
        a_m=a_grid_m,
        geometry=geometry,
        profile=profile,
        output_path=output_dir / "l_mode_profiles.png"
    )

if __name__ == "__main__":
    main()