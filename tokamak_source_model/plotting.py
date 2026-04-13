from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .geometry import make_magnetic_surface_curve
from .profiles import ion_density_profile_m3, ion_temperature_profile_keV
from .parameters import GeometryParameters, ProfileParameters

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