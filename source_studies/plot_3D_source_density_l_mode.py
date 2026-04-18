from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from tokamak_source_model.case_builder import build_default_mesh, build_l_mode_model
from tokamak_source_model.sampling import sample_birth_positions
from tokamak_source_model.source_density import evaluate_profiles

def set_axes_equal(ax) -> None:
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max(x_range, y_range, z_range)

    ax.set_xlim3d([x_middle-plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle-plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle-plot_radius, z_middle + plot_radius])

def main() -> None:
    output_dir = Path("source_studies/output/energy_spectrum")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_l_mode_model()
    mesh = build_default_mesh()
    rng = np.random.default_rng(42)

    n_samples = 120000
    plot_every = 8

    a_m, alpha_rad, x_m, y_m, z_m = sample_birth_positions(
        n_samples=n_samples,
        model=model,
        mesh=mesh,
        rng=rng,
    )

    evaluation = evaluate_profiles(a_m, model)
    source_density = evaluation.source_density_n_per_m3_per_s

    idx = np.arange(0, n_samples, plot_every)

    x_plot = x_m[idx]
    y_plot = y_m[idx]
    z_plot = z_m[idx]
    s_plot = source_density[idx]

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        x_plot,
        y_plot,
        z_plot,
        c=s_plot,
        s=1.5,
        alpha=0.18,
        linewidths=0,
    )

    ax.set_title("Sampled DT neutron birth source cloud (L-mode)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    set_axes_equal(ax)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.08)
    cbar.set_label("Local source density [n m$^{-3}$ s$^{-1}$]")

    fig.tight_layout()
    fig.savefig(output_dir / "l_mode_source_density_3d_linear.png", dpi=220)
    plt.close(fig)

    #log-color plot
    positive = s_plot > 0.0
    s_positive = s_plot[positive]
    x_positive = x_plot[positive]
    y_positive = y_plot[positive]
    z_positive = z_plot[positive]

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        x_positive,
        y_positive,
        z_positive,
        c=s_positive,
        s=1.5,
        alpha=0.18,
        linewidths=0,
        norm=LogNorm(vmin=np.min(s_positive), vmax=np.max(s_positive))
    )
    
    ax.set_title("Sampled DT neutron birth source cloud (L-mode, log color)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    set_axes_equal(ax)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.08)
    cbar.set_label("Local source density [n m$^{-3}$ s$^{-1}$]")

    fig.tight_layout()
    fig.savefig(output_dir / "log_l_mode_source_density_3d.png", dpi=220)
    plt.close(fig)

if __name__ == "__main__":
    main()