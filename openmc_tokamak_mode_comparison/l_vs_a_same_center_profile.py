from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tokamak_source_model.utils.parameters import (
    GeometryParameters,
    ProfileParameters,
    SourceModelParameters,
)
from tokamak_source_model.utils.case_builder import (
    build_default_energy_spectrum,
    build_default_fuel,
    build_default_mesh,
)
from tokamak_source_model.utils.normalization import estimate_total_neutron_rate_n_per_s
from tokamak_source_model.utils.source_density import evaluate_profiles


def build_same_center_l_model() -> SourceModelParameters:
    geometry = GeometryParameters(
        major_radius_m=2.0,
        minor_radius_m=0.5,
        elongation=1.7,
        triangularity=0.33,
        shafranov_shift_m=0.1,
    )

    profile = ProfileParameters(
        mode="l_mode",
        ion_density_center_m3=8.0e19,
        ion_temp_center_keV=12.0,
        alpha_n=0.5,
        alpha_T=1.5,
    )

    return SourceModelParameters(
        geometry=geometry,
        profile=profile,
        fuel=build_default_fuel(),
        energy_spectrum=build_default_energy_spectrum(),
    )


def build_same_center_a_model() -> SourceModelParameters:
    geometry = GeometryParameters(
        major_radius_m=2.0,
        minor_radius_m=0.5,
        elongation=1.7,
        triangularity=0.33,
        shafranov_shift_m=0.1,
    )

    profile = ProfileParameters(
        mode="pedestal",
        ion_density_center_m3=8.0e19,          # same as L-mode
        ion_temp_center_keV=12.0,              # same as L-mode
        alpha_n=1.0,
        alpha_T=8.06,
        pedestal_radius_m=0.8 * geometry.minor_radius_m,
        ion_density_pedestal_m3=8.0e19,        # flat-topped density core
        ion_density_separatrix_m3=3.0e19,
        ion_temp_pedestal_keV=2.09,
        ion_temp_separatrix_keV=0.1,
        beta_T=6.0,
    )

    return SourceModelParameters(
        geometry=geometry,
        profile=profile,
        fuel=build_default_fuel(),
        energy_spectrum=build_default_energy_spectrum(),
    )


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "output" / "same_center_l_vs_a"
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh = build_default_mesh()
    l_model = build_same_center_l_model()
    a_model = build_same_center_a_model()

    a_grid_m = np.linspace(0.0, l_model.geometry.minor_radius_m, mesh.num_a)
    rho = a_grid_m / l_model.geometry.minor_radius_m
    a_cm = a_grid_m * 100.0

    l_eval = evaluate_profiles(a_grid_m, l_model)
    a_eval = evaluate_profiles(a_grid_m, a_model)

    l_rate = estimate_total_neutron_rate_n_per_s(l_model, mesh)
    a_rate = estimate_total_neutron_rate_n_per_s(a_model, mesh)

    # ---- print summary ----
    print("Same-center L vs A comparison")
    print("-----------------------------")
    print(f"L-mode center n_i = {l_eval.ion_density_m3[0]:.6e} m^-3")
    print(f"A-mode center n_i = {a_eval.ion_density_m3[0]:.6e} m^-3")
    print(f"L-mode center T_i = {l_eval.ion_temp_keV[0]:.6e} keV")
    print(f"A-mode center T_i = {a_eval.ion_temp_keV[0]:.6e} keV")
    print()
    print(f"L-mode max S      = {np.max(l_eval.source_density_n_per_m3_per_s):.6e} n/(m^3 s)")
    print(f"A-mode max S      = {np.max(a_eval.source_density_n_per_m3_per_s):.6e} n/(m^3 s)")
    print(f"L-mode total rate = {l_rate:.6e} n/s")
    print(f"A-mode total rate = {a_rate:.6e} n/s")
    print(f"A/L total rate    = {a_rate / l_rate:.6f}")

    # ---- 4-panel comparison ----
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))

    axs[0, 0].plot(rho, l_eval.ion_density_m3, label="L-mode", linewidth=2)
    axs[0, 0].plot(rho, a_eval.ion_density_m3, label="A-mode (same center)", linewidth=2)
    axs[0, 0].set_title("Ion density")
    axs[0, 0].set_xlabel(r"$\rho = a/A$")
    axs[0, 0].set_ylabel(r"$n_i$ [m$^{-3}$]")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()

    axs[0, 1].plot(rho, l_eval.ion_temp_keV, label="L-mode", linewidth=2)
    axs[0, 1].plot(rho, a_eval.ion_temp_keV, label="A-mode (same center)", linewidth=2)
    axs[0, 1].set_title("Ion temperature")
    axs[0, 1].set_xlabel(r"$\rho = a/A$")
    axs[0, 1].set_ylabel(r"$T_i$ [keV]")
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend()

    axs[1, 0].plot(rho, l_eval.reactivity_m3_per_s, label="L-mode", linewidth=2)
    axs[1, 0].plot(rho, a_eval.reactivity_m3_per_s, label="A-mode (same center)", linewidth=2)
    axs[1, 0].set_title("DT reactivity")
    axs[1, 0].set_xlabel(r"$\rho = a/A$")
    axs[1, 0].set_ylabel(r"$\langle \sigma v \rangle$ [m$^3$/s]")
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend()

    axs[1, 1].plot(rho, l_eval.source_density_n_per_m3_per_s, label="L-mode", linewidth=2)
    axs[1, 1].plot(rho, a_eval.source_density_n_per_m3_per_s, label="A-mode (same center)", linewidth=2)
    axs[1, 1].set_title("Source density")
    axs[1, 1].set_xlabel(r"$\rho = a/A$")
    axs[1, 1].set_ylabel(r"$S$ [n/(m$^3$ s)]")
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend()

    fig.suptitle("L-mode vs A-mode with matched central density and temperature", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / "same_center_l_vs_a_profiles.png", dpi=300)
    plt.close(fig)

    # ---- Fausser-style source-shape comparison ----
    l_source_avg = np.mean(l_eval.source_density_n_per_m3_per_s)
    a_source_avg = np.mean(a_eval.source_density_n_per_m3_per_s)

    plt.figure(figsize=(8, 5))
    plt.plot(a_cm, a_eval.source_density_n_per_m3_per_s / a_source_avg, label="A-mode", linewidth=2)
    plt.plot(a_cm, l_eval.source_density_n_per_m3_per_s / l_source_avg, label="L-mode", linewidth=2)
    plt.xlabel("a [cm]")
    plt.ylabel("Normalized neutron source density")
    plt.title("L- and A-mode source distribution normalized to average value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "same_center_l_vs_a_source_avg_normalized.png", dpi=300)
    plt.close()

    # ---- raw source-density-only comparison ----
    plt.figure(figsize=(8, 5))
    plt.plot(a_cm, l_eval.source_density_n_per_m3_per_s, label="L-mode", linewidth=2)
    plt.plot(a_cm, a_eval.source_density_n_per_m3_per_s, label="A-mode (same center)", linewidth=2)
    plt.xlabel("a [cm]")
    plt.ylabel(r"Source density $S(a)$ [n/(m$^3$ s)]")
    plt.title("Radial source-density comparison with matched center parameters")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "same_center_l_vs_a_source_density.png", dpi=300)
    plt.close()

    print(f"\nSaved plots to: {output_dir}")


if __name__ == "__main__":
    main()