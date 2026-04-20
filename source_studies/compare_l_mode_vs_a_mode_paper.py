from __future__ import annotations

from pathlib import Path

import numpy as np

from tokamak_source_model.utils.case_builder import build_default_mesh, build_l_mode_model, build_a_mode_paper_model
from tokamak_source_model.utils.normalization import estimate_total_neutron_rate_n_per_s
from tokamak_source_model.utils.plotting import plot_profile_comparison_custom_labels
from tokamak_source_model.utils.source_density import evaluate_profiles
from tokamak_source_model.utils.validation import validate_source_model_parameters

def main() -> None:
    output_dir = Path("source_studies/output/comparisons")
    output_dir.mkdir(parents=True, exist_ok=True)

    l_model = build_l_mode_model()
    a_model = build_a_mode_paper_model()
    mesh = build_default_mesh()

    geometry = l_model.geometry

    validate_source_model_parameters(a_model)
    validate_source_model_parameters(l_model)

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