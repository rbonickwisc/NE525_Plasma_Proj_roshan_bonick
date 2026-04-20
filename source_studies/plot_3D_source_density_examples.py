from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize

from tokamak_source_model.utils.case_builder import build_default_mesh, build_l_mode_model, build_a_mode_paper_model, build_generic_pedestal_model
from tokamak_source_model.utils.sampling import sample_birth_positions
from tokamak_source_model.utils.source_density import evaluate_profiles

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

def prepare_plot_data(
    model,
    label:str,
    n_samples: int = 120000,
    plot_every: int = 8,
    seed: int = 42,
) -> dict:
    mesh = build_default_mesh()
    rng = np.random.default_rng(seed)

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

    return {
        "label": label,
        "safe_label": label.lower().replace(" ", "_").replace("-", "_"),
        "x_plot": x_plot,
        "y_plot": y_plot,
        "z_plot": z_plot,
        "s_plot": s_plot,
    }

def save_linear_plot(
    data: dict,
    output_dir: Path,
    vmin: float,
    vmax: float,
) -> None:
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        data["x_plot"],
        data["y_plot"],
        data["z_plot"],
        c=data["s_plot"],
        s=1.5,
        alpha=0.18,
        linewidths=0,
        norm=Normalize(vmin=vmin, vmax=vmax),
    )

    ax.set_title(f"Sampled DT neutron birth source cloud ({data['label']})")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    set_axes_equal(ax)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.08)
    cbar.set_label("Local source density [n m$^{-3}$ s$^{-1}$]")

    fig.tight_layout()
    fig.savefig(output_dir / f"{data['safe_label']}_source_density_3d_linear.png", dpi=220)
    plt.close(fig)

def save_log_plot(
    data:dict,
    output_dir: Path,
    vmin: float,
    vmax: float,
) -> None:
    positive = data["s_plot"] > 0.0

    s_positive = data["s_plot"][positive]
    x_positive = data["x_plot"][positive]
    y_positive = data["y_plot"][positive]
    z_positive = data["z_plot"][positive]

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
        norm=LogNorm(vmin=vmin, vmax=vmax)
    )
    
    ax.set_title(f"Sampled DT neutron birth source cloud ({data['label']}, log color)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    set_axes_equal(ax)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.08)
    cbar.set_label("Local source density [n m$^{-3}$ s$^{-1}$]")

    fig.tight_layout()
    fig.savefig(output_dir / f"LOG_{data['label']}_source_density_3d.png", dpi=220)
    plt.close(fig)

def main() -> None:
    output_dir = Path("source_studies/output/energy_spectrum/source_density_3d")
    output_dir.mkdir(parents=True, exist_ok=True)

    l_mode = build_l_mode_model()
    pedestal_mode = build_generic_pedestal_model()
    a_mode = build_a_mode_paper_model()

    datasets = [
        prepare_plot_data(l_mode, "L-mode", seed=42),
        prepare_plot_data(pedestal_mode, "Pedestal mode", seed=43),
        prepare_plot_data(a_mode, "A-mode", seed=44)
    ]

    all_s = np.concatenate([d["s_plot"] for d in datasets])
    linear_vmin = float(np.min(all_s))
    linear_vmax = float(np.max(all_s))
    all_positive_s = np.concatenate([d["s_plot"][d["s_plot"] > 0.0] for d in datasets])
    log_vmin = float(np.min(all_positive_s))
    log_vmax = float(np.max(all_positive_s))

    for data in datasets:
        save_linear_plot(
            data=data,
            output_dir=output_dir,
            vmin=linear_vmin,
            vmax=linear_vmax,
        )
        save_log_plot(
            data=data,
            output_dir=output_dir,
            vmin=log_vmin,
            vmax=log_vmax,
        )

if __name__ == "__main__":
    main()