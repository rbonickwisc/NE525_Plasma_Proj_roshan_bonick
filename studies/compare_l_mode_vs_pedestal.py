from __future__ import annotations

from pathlib import Path

import numpy as np

from tokamak_source_model.normalization import estimate_total_neutron_rate_n_per_s
from tokamak_source_model.parameters import FuelParameters, GeometryParameters, MeshParameters, ProfileParameters, SourceModelParameters
from tokamak_source_model.plotting import plot_profile_comparison_vs_a
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
        shafranov_shift_m=0.1,
    )

    fuel= FuelParameters(
        deuterium_fraction=0.5,
        tritium_fraction=0.5,
    )

    l_mode_profile = ProfileParameters(
        mode="l_mode",
        ion_density_center_m3=2.0e20,
        ion_temp_center_keV=15.0,
        alpha_n=0.5,
        alpha_T=1.0,
    )

    pedestal_profile = ProfileParameters(
        mode="pedestal",
        ion_density_center_m3=2.0e20,
        ion_temp_center_keV=20.0,
        alpha_n=1.0,
        alpha_T=4.0,
        pedestal_radius_m=0.8 * geometry.minor_radius_m,
        ion_density_pedestal_m3=1.8e20,
        ion_density_separatrix_m3=3.0e19,
        ion_temp_pedestal_keV=4.0,
        ion_temp_separatrix_keV=0.1,
        beta_T=6.0,
    )

    l_model = SourceModelParameters(
        geometry=geometry,
        profile=l_mode_profile,
        fuel=fuel,
    )

    p_model = SourceModelParameters(
        geometry=geometry,
        profile=pedestal_profile,
        fuel=fuel,
    )

    validate_source_model_parameters(l_model)
    validate_source_model_parameters(p_model)

    mesh = MeshParameters(
        num_a=200,
        num_alpha=360,
    )

    a_grid_m = np.linspace(0.0, geometry.minor_radius_m, mesh.num_a)

    l_eval = evaluate_profiles(a_grid_m, l_model)
    p_eval = evaluate_profiles(a_grid_m, p_model)

    l_rate = estimate_total_neutron_rate_n_per_s(l_model, mesh)
    p_rate = estimate_total_neutron_rate_n_per_s(p_model, mesh)

    print("L-mode vs pedestal-mode comparison")
    print("-" * 25)
    print(f"L-mode total neutron rate         ={l_rate:.6e} n/s")
    print(f"Pedestal-mode total neutron rate  ={p_rate:.6e} n/s")
    print(f"Relative difference [%]           ={100.0 * (p_rate / l_rate):.6f}")

    plot_profile_comparison_vs_a(
        a_m= a_grid_m,
        l_mode_evaluation=l_eval,
        pedestal_evaluation=p_eval,
        output_path=output_dir / "l_mode_vs_pedestal_comparison.png"
    )

if __name__ == "__main__":
    main()