from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tokamak_source_model.utils.case_builder import build_default_mesh, build_generic_pedestal_model, build_l_mode_model
from tokamak_source_model.utils.normalization import estimate_total_neutron_rate_n_per_s, estimate_total_plasma_volume_m3
from tokamak_source_model.utils.sampling import sample_source_particles
from tokamak_source_model.utils.source_density import evaluate_profiles


def summarize_model(
    label: str,
    model,
    *,
    n_profile_points: int = 400,
    n_energy_samples: int = 20000,
    seed: int = 42,
) -> dict:
    geometry = model.geometry
    mesh = build_default_mesh()

    a_m = np.linspace(0.0, geometry.minor_radius_m, n_profile_points)
    rho = a_m / geometry.minor_radius_m

    evaluation = evaluate_profiles(a_m, model)

    rng = np.random.default_rng(seed)
    samples = sample_source_particles(
        n_samples=n_energy_samples,
        model=model,
        mesh=mesh,
        rng=rng,
    )

    total_rate = estimate_total_neutron_rate_n_per_s(model, mesh)
    total_volume = estimate_total_plasma_volume_m3(model, mesh)

    return {
        "label": label,
        "rho": rho,
        "a_m": a_m,
        "ion_density_m3": evaluation.ion_density_m3,
        "ion_temp_keV": evaluation.ion_temp_keV,
        "reactivity_m3_per_s": evaluation.reactivity_m3_per_s,
        "source_density_n_per_m3_per_s": evaluation.source_density_n_per_m3_per_s,
        "total_rate_n_per_s": total_rate,
        "total_volume_m3": total_volume,
        "mean_energy_eV": float(np.mean(samples.energy_eV)),
        "std_energy_eV": float(np.std(samples.energy_eV)),
        "min_energy_eV": float(np.min(samples.energy_eV)),
        "max_energy_eV": float(np.max(samples.energy_eV)),
    }


def main() -> None:
    output_dir = Path("source_studies/output/comparisons")
    output_dir.mkdir(parents=True, exist_ok=True)

    l_mode = summarize_model(
        "L-mode",
        build_l_mode_model(),
        seed=42,
    )
    h_mode = summarize_model(
        "H-mode",
        build_generic_pedestal_model(),
        seed=43,
    )

    relative_rate_percent = 100.0 * (
        h_mode["total_rate_n_per_s"] - l_mode["total_rate_n_per_s"]
    ) / l_mode["total_rate_n_per_s"]

    print("L-mode vs H-mode comparison")
    print("-" * 40)
    print(f"L-mode plasma volume        = {l_mode['total_volume_m3']:.6e} m^3")
    print(f"H-mode plasma volume        = {h_mode['total_volume_m3']:.6e} m^3")
    print(f"L-mode total neutron rate   = {l_mode['total_rate_n_per_s']:.6e} n/s")
    print(f"H-mode total neutron rate   = {h_mode['total_rate_n_per_s']:.6e} n/s")
    print(f"Relative difference [%]     = {relative_rate_percent:.6f}")
    print("-" * 25)
    print(f"L-mode mean energy [MeV]    = {l_mode['mean_energy_eV'] / 1.0e6:.6f}")
    print(f"H-mode mean energy [MeV]    = {h_mode['mean_energy_eV'] / 1.0e6:.6f}")
    print(f"L-mode std energy  [MeV]    = {l_mode['std_energy_eV'] / 1.0e6:.6f}")
    print(f"H-mode std energy  [MeV]    = {h_mode['std_energy_eV'] / 1.0e6:.6f}")
    print(f"L-mode min/max [MeV]        = "
          f"{l_mode['min_energy_eV'] / 1.0e6:.6f} / {l_mode['max_energy_eV'] / 1.0e6:.6f}")
    print(f"H-mode min/max [MeV]        = "
          f"{h_mode['min_energy_eV'] / 1.0e6:.6f} / {h_mode['max_energy_eV'] / 1.0e6:.6f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].plot(l_mode["rho"], l_mode["ion_density_m3"], label="L-mode")
    axes[0, 0].plot(h_mode["rho"], h_mode["ion_density_m3"], label="H-mode")
    axes[0, 0].set_xlabel(r"$\rho = a / a_{\mathrm{minor}}$ (normalized minor radius)")
    axes[0, 0].set_ylabel(r"$n_i$ [m$^{-3}$]")
    axes[0, 0].set_title("Ion density")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(l_mode["rho"], l_mode["ion_temp_keV"], label="L-mode")
    axes[0, 1].plot(h_mode["rho"], h_mode["ion_temp_keV"], label="H-mode")
    axes[0, 1].set_xlabel(r"$\rho = a / a_{\mathrm{minor}}$ (normalized minor radius)")
    axes[0, 1].set_ylabel(r"$T_i$ [keV]")
    axes[0, 1].set_title("Ion temperature")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(l_mode["rho"], l_mode["reactivity_m3_per_s"], label="L-mode")
    axes[1, 0].plot(h_mode["rho"], h_mode["reactivity_m3_per_s"], label="H-mode")
    axes[1, 0].set_xlabel(r"$\rho = a / a_{\mathrm{minor}}$ (normalized minor radius)")
    axes[1, 0].set_ylabel(r"$\langle \sigma v \rangle$ [m$^3$/s]")
    axes[1, 0].set_title("DT reactivity")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(
        l_mode["rho"],
        l_mode["source_density_n_per_m3_per_s"],
        label="L-mode",
    )
    axes[1, 1].plot(
        h_mode["rho"],
        h_mode["source_density_n_per_m3_per_s"],
        label="H-mode",
    )
    axes[1, 1].set_xlabel(r"$\rho = a / a_{\mathrm{minor}}$ (normalized minor radius)")
    axes[1, 1].set_ylabel(r"$S$ [n/(m$^3$ s)]")
    axes[1, 1].set_title("Source density")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.suptitle("L-mode vs H-mode source comparison (Moving from plasma center to plasma edge)")
    fig.tight_layout()
    fig.savefig(output_dir / "l_mode_vs_h_mode_comparison.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()