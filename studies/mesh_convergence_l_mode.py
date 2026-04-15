from __future__ import annotations

from pathlib import Path
import numpy as np

from tokamak_source_model.normalization import build_source_cell_probability_map, estimate_total_neutron_rate_n_per_s, estimate_total_plasma_volume_m3
from tokamak_source_model.parameters import FuelParameters, GeometryParameters, MeshParameters, ProfileParameters, SourceModelParameters
from tokamak_source_model.plotting import plot_mesh_convergence
from tokamak_source_model.validation import validate_source_model_parameters


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

    validate_source_model_parameters(model)

    mesh_list = [
        MeshParameters(num_a=40, num_alpha=60),
        MeshParameters(num_a=80, num_alpha=120),
        MeshParameters(num_a=120, num_alpha=180),
        MeshParameters(num_a=200, num_alpha=360),
        MeshParameters(num_a=300, num_alpha=540),
    ]

    mesh_labels: list[str] = []
    volumes_m3: list[float] = []
    total_rates_n_per_s: list[float] = []
    probability_sums: list[float] = []

    for mesh in mesh_list:
        label = f"({mesh.num_a}, {mesh.num_alpha})"
        mesh_labels.append(label)

        volumes_m3 = estimate_total_plasma_volume_m3(model, mesh)
        total_rates_n_per_s = estimate_total_neutron_rate_n_per_s(model, mesh)

        _, _, _, _, probability_map = build_source_cell_probability_map(model, mesh)
        probability_sum = float(np.sum(probability_map))

        volumes_m3.append(volumes_m3)
        total_rates_n_per_s.append(total_rates_n_per_s)
        probability_sums.append(probability_sum)

        volumes_m3 = np.asarray(volumes_m3, dtype=float)
        total_rates_n_per_s = np.asarray(total_rates_n_per_s, dtype=float)
        probability_sums = np.asarray(probability_sums, dtype=float)

        finest_volume = volumes_m3[-1]
        finest_total_rate = total_rates_n_per_s[-1]

        volume_rel_diff_percent = 100.0 * (volumes_m3 - finest_volume) / finest_volume
        total_rate_rel_diff_percent = 100.0 * (total_rates_n_per_s - finest_total_rate) / finest_total_rate

        print("L-mode mesh convergence study")
        print("-------------------------")
        print(
            f"{'mesh':>16} | {'volume [m^3]':>16} | {'Δvol [%]':>12} | "
            f"{'total rate [n/s]':>18} | {'Δrate [%]':>12} | {'prob sum':>12}"
        )
        print("-" * 25)

        for i, label in enumerate(mesh_labels):
            print(
                f"{label:>16} | "
                f"{volumes_m3[i]:>16.6e} | "
                f"{volume_rel_diff_percent[i]:>12.6f} | "
                f"{total_rates_n_per_s[i]:>18.6e} | "
                f"{total_rate_rel_diff_percent[i]:>12.6f} | "
                f"{probability_sums[i]:>12.12f}"
            )

        plot_mesh_convergence(
            mesh_sizes=mesh_labels,
            volumes_m3=volumes_m3,
            total_rates_n_per_s=total_rates_n_per_s,
            output_path = output_dir / "l_mode_mesh_convergence.png"
        )

    if __name__ == "__main__":
        main()