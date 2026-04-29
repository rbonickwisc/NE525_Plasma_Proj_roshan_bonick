from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from tokamak_source_model.utils.case_builder import build_default_mesh
from tokamak_source_model.utils.normalization import estimate_total_neutron_rate_n_per_s
from tokamak_source_model.utils.source_density import evaluate_profiles

from metrics import source_weighted_mean_radius
from profiles import build_a_mode_from_parameters, build_l_mode_reference


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh = build_default_mesh()

    pedestal_fracs = np.linspace(0.55, 0.95, 9)

    rows = []

    l_model = build_l_mode_reference()
    l_rate = estimate_total_neutron_rate_n_per_s(l_model, mesh)
    l_metric = source_weighted_mean_radius(l_model, mesh)

    rows.append(
        {
            "case": "L-mode reference",
            "pedestal_fraction": np.nan,
            "source_weighted_mean_radius": l_metric,
            "total_neutron_rate_n_per_s": l_rate,
        }
    )

    for frac in pedestal_fracs:
        model = build_a_mode_from_parameters(pedestal_fraction=float(frac))
        rate = estimate_total_neutron_rate_n_per_s(model, mesh)
        metric = source_weighted_mean_radius(model, mesh)

        rows.append(
            {
                "case": "A-like pedestal scan",
                "pedestal_fraction": float(frac),
                "source_weighted_mean_radius": metric,
                "total_neutron_rate_n_per_s": rate,
            }
        )

    df = pd.DataFrame(rows)
    csv_path = output_dir / "pedestal_radius_scan.csv"
    df.to_csv(csv_path, index=False)

    scan_df = df[df["case"] == "A-like pedestal scan"].copy()

    #plot 1 - source shape metric vs pedestal fraction
    plt.figure(figsize=(8, 5))
    plt.plot(
        scan_df["pedestal_fraction"],
        scan_df["source_weighted_mean_radius"],
        marker="o",
    )
    plt.axhline(l_metric, linestyle="--", label="L-mode reference")
    plt.xlabel(r"Pedestal radius fraction $a_{\mathrm{ped}}/A$")
    plt.ylabel(r"Source-weighted mean radius $\langle \rho \rangle_S$")
    plt.title("Source-weighted mean radius vs pedestal radius")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pedestal_fraction_vs_mean_radius.png", dpi=300)
    plt.close()

    #plot 2-  neutron rate vs metric
    plt.figure(figsize=(8, 5))
    plt.plot(
        scan_df["source_weighted_mean_radius"],
        scan_df["total_neutron_rate_n_per_s"],
        marker="o",
    )
    plt.scatter([l_metric], [l_rate], marker="x", s=80, label="L-mode reference")
    plt.xlabel(r"Source-weighted mean radius $\langle \rho \rangle_S$")
    plt.ylabel(r"Total neutron rate $\dot N$ [n/s]")
    plt.title("Total neutron rate vs source-weighted mean radius")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "neutron_rate_vs_mean_radius.png", dpi=300)
    plt.close()

    print(df.to_string(index=False))
    print(f"\nSaved CSV to: {csv_path}")
    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()