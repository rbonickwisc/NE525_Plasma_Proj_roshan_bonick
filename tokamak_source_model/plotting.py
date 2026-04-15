from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .geometry import make_magnetic_surface_curve
from .profiles import ion_density_profile_m3, ion_temperature_profile_keV
from .parameters import GeometryParameters, ProfileParameters, ProfileEvaluation

def plot_magnetic_surfaces(
    geometry: GeometryParameters,
    surface_radii_m: np.ndarray,
    alpha_rad: np.ndarray,
    output_path: str | Path | None = None,
) -> None:
    """
    Plot magnetic surfaces in R-Z plane
    """
    fig, ax = plt.subplots(figsize=(7,6))

    for a_m in surface_radii_m:
        R_m, Z_m = make_magnetic_surface_curve(a_m, alpha_rad, geometry)
        ax.plot(R_m, Z_m)

        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_title("Tokamak Magnetic Surfaces")
        ax.set_aspect("equal")
        ax.grid(True)

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=200, bbox_inches="tight")

        plt.close(fig)

def plot_profiles_vs_a(
        a_m: np.ndarray,
        geometry: GeometryParameters,
        profile: ProfileParameters,
        output_path: str | Path | None = None,
) -> None:
    """
    Plot ion density and ion temperature vs a
    """
    ni_m3 = ion_density_profile_m3(a_m, geometry, profile)
    Ti_keV = ion_temperature_profile_keV(a_m, geometry, profile)

    fig, ax1 = plt.subplots(figsize=(7,5))

    ax1.plot(a_m, ni_m3, label="Ion density")
    ax1.set_xlabel( "a [m]")
    ax1.set_ylabel(r"$n_i$ [m$^{-3}]")

    ax2 = ax1.twinx()
    ax2.plot(a_m, Ti_keV, linestyle="--", label = "Ion temperature")
    ax2.set_ylabel(r"$T_i$ [keV]")

    ax1.set_title(f"Profiles vs a ({profile.mode})")
    ax1.grid(True)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc = "upper right")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi = 200, bbox_inches = "tight")

    plt.close(fig)

def plot_source_quantities_vs_a(
    evaluation: ProfileEvaluation,
    output_path: str | Path | None=None,
) -> None:
    fig, ax1 = plt.subplots(figsize=(7,5))

    ax1.plot(
        evaluation.a_m,
        evaluation.reactivity_m3_per_s,
        label = "DT Reactivity",
    )
    ax1.set_xlabel("a [m]")
    ax1.set_ylabel(r"$\langle \sigma v \rangle$ [m$^3$/s]")

    ax2 = ax1.twinx()
    ax2.plot(
        evaluation.a_m,
        evaluation.source_density_n_per_m3_per_s,
        linestyle="--",
        label="Source Density"
    )
    ax2.set_ylabel(r"$S$ [n / (m$^3 s)]")

    ax1.set_title("DT Reactivity and Source Density vs a")
    ax1.grid(True)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

        plt.close(fig)

def plot_probability_map_rz(
    R_m: np.ndarray,
    Z_m: np.ndarray,
    probability_map: np.ndarray,
    output_path: str | Path | None = None,
) -> None:
    """
    Plot a source probability map in the R-Z plane using a scatter plot
    """
    fig, ax = plt.subplots(figsize=(7,6))

    sc = ax.scatter(
        R_m.ravel(),
        Z_m.ravel(),
        c=probability_map.ravel(),
        s=6,
    )

    fig.colorbar(sc, ax=ax, label="Probability weight")

    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title("Tokamak Source Probability Map")
    ax.set_aspect("equal")
    ax.grid(True)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    plt.close(fig)

def plot_sampled_birth_points(
    x_m: np.ndarray,
    z_m: np.ndarray,
    output_path: str | Path | None = None,
) -> None:
    """
    Plot sampled birth points in an X-Z projection
    (not a full toroidal visualization, useful quick check)
    """
    
    fig, ax = plt.subplots(figsize=(7,6))

    ax.scatter(x_m, z_m, s=4)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title("Sampled Neutron Birth Points (X-Z projection)")
    ax.set_aspect("equal")
    ax.grid(True)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    plt.close(fig)

    def plot_mesh_convergence(
            mesh_sizes: list[str],
            volumes_m3: np.ndarray,
            total_rates_n_per_s: np.ndarray,
            output_path: str | Path | None = None,
    ) -> None:
        """
        Plot mesh-convergence trends for plasma volume and total neutron rate
        """
        fig, ax1 = plt.subplots(figsize=(8,5))

        x = np.arange(len(mesh_sizes))

        ax1.plot(x, volumes_m3, maker="o", label="Plasma volume")
        ax1.set_xlabel("Mesh resolution (num_a, num_alpha)")
        ax1.set_ylabel("Plasma volume [m^3]")
        ax1.set_xticks(x)
        ax1.set_xticklabels(mesh_sizes, rotation=30)

        ax2 = ax1.twinx()
        ax2.plot(x, total_rates_n_per_s, maker="s", linestyle="--", label = "Total neutron rate")
        ax2.set_ylabel("Total neutron rate [n/s]")

        ax1.set_title("Mesh Convergence")
        ax1.grid(True)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=200, bbox_inches="tight")

        plt.close(fig)