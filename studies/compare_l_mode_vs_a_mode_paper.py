from __future__ import annotations

from pathlib import Path

import numpy as np

from tokamak_source_model.geometry import make_a_alpha_grids
from tokamak_source_model.normalization import build_source_probability_map, estimate_total_neutron_rate_n_per_s, estimate_total_plasma_volume_m3
from tokamak_source_model.parameters import FuelParameters, GeometryParameters, MeshParameters, ProfileParameters, SourceModelParameters
from tokamak_source_model.plotting import plot_magnetic_surfaces, plot_probability_map_rz, plot_profiles_vs_a, plot_sampled_birth_points, plot_source_quantities_vs_a, plot_profile_comparison_custom_labels
from tokamak_source_model.sampling import sample_source_particles
from tokamak_source_model.source_density import evaluate_profiles
from tokamak_source_model.validation import validate_source_model_parameters

def main() -> None:
    output_dir = Path("studies/output/comparisons")
    output_dir.mkdir(parents=True, exist_ok=True)

    geometry = GeometryParameters(
        major_radius_m=2.0,
        minor_radius_m=0.5,
        elongation=1.7,
        triangularity=0.33,
        shafranov_shift_m=0.1
    )

    fuel = FuelParameters(
        deuterium_fraction=0.5,
        tritium_fraction=0.5,
    )

    a_mode_profile = ProfileParameters(
        mode="pedestal", 
        ion_density_center_m3=1.09e20, 
        ion_temp_center_keV=45.9,
        alpha_n=1.0,
        alpha_T=8.06,
        pedestal_radius_m=0.8 * geometry.minor_radius_m,
        ion_density_pedestal_m3=1.09e20,
        ion_density_separatrix_m3=3.0e19,
        ion_temp_pedestal_keV=6.09,
        ion_temp_separatrix_keV=0.1,
        beta_T=6.0,
    )

    l_mode_profile = ProfileParameters(
        mode="l_mode",
        ion_density_center_m3=2.0e20,
        ion_temp_center_keV=15.0,
        alpha_n=0.5,
        alpha_T=1.0,
    )

    a_model = SourceModelParameters(
        geometry=geometry,
        profile=a_mode_profile,
        fuel=fuel,
    )

    l_model = SourceModelParameters(
        geometry=geometry,
        profile=l_mode_profile,
        fuel=fuel,
    )

    validate_source_model_parameters(a_model)
    validate_source_model_parameters(l_model)

    mesh = MeshParameters(
        num_a=200,
        num_alpha=360,
    )

    a_grid_m = np.linspace(0.0, geometry.minor_radius_m, mesh.num_a)

    l_eval = evaluate_profiles(a_grid_m, l_model)
    a_eval = evaluate_profiles(a_grid_m, a_model)

    l_rate = estimate_total_neutron_rate_n_per_s(l_model, mesh)
    a_rate = estimate_total_neutron_rate_n_per_s(a_model, mesh)

    print("L-mode vs A-mode comparison")
    print("-" * 25)
    print(f"L-mode total neutron rate        = {l_rate:.6e} n/s")
    print(f"A-mode total neutron rate        = {a_rate:.6e} n/s")
    print(f"Relative difference [%]          = {100.0 * (a_rate - l_rate) / l_rate:.6f}")

    plot_profile_comparison_custom_labels(
        a_m=a_grid_m,
        first_evaluation=l_eval,
        second_evaluation=a_eval,
        first_label="L-mode",
        second_label="A-mode",
        output_path=output_dir / "l_mode_vs_a_mode_comparison.png"
    )

if __name__ == "__main__":
    main()