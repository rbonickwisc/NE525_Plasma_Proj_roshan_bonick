from __future__ import annotations

from pathlib import Path

import numpy as np

from tokamak_source_model.case_builder import build_default_mesh, build_generic_pedestal_model, build_l_mode_model
from tokamak_source_model.normalization import estimate_total_neutron_rate_n_per_s
from tokamak_source_model.plotting import plot_profile_comparison_vs_a
from tokamak_source_model.source_density import evaluate_profiles
from tokamak_source_model.validation import validate_source_model_parameters

def main() -> None:
    output_dir = Path("studies/output/comparisons")
    output_dir.mkdir(parents=True, exist_ok=True)

    l_model = build_l_mode_model()
    p_model = build_generic_pedestal_model()
    mesh = build_default_mesh()

    geometry = l_model.geometry

    validate_source_model_parameters(l_model)
    validate_source_model_parameters(p_model)

    a_grid_m = np.linspace(0.0, geometry.minor_radius_m, mesh.num_a)

    l_eval = evaluate_profiles(a_grid_m, l_model)
    p_eval = evaluate_profiles(a_grid_m, p_model)

    l_rate = estimate_total_neutron_rate_n_per_s(l_model, mesh)
    p_rate = estimate_total_neutron_rate_n_per_s(p_model, mesh)

    print("L-mode vs pedestal-mode comparison")
    print("-" * 25)
    print(f"L-mode total neutron rate         ={l_rate:.6e} n/s")
    print(f"Pedestal-mode total neutron rate  ={p_rate:.6e} n/s")
    print(f"Relative difference [%]           ={100.0 * (p_rate - l_rate) / l_rate:.6f}")

    plot_profile_comparison_vs_a(
        a_m= a_grid_m,
        l_mode_evaluation=l_eval,
        pedestal_evaluation=p_eval,
        output_path=output_dir / "l_mode_vs_pedestal_comparison.png"
    )

if __name__ == "__main__":
    main()