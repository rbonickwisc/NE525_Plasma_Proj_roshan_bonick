from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import openmc

MODE_SOURCE_RATE_N_PER_S = {
    "L": 3.150941e18,
    "H": 1.895748e19,
    "A": 5.326079e19,
}
R0 = 200.0
plasma_r = 20.0
first_wall_r = 50.0
flibe_r = 100.0
Y_HALF = 5
N_BINS = 80
N_A = 6.02214076e23
M_T_G_PER_MOL = 3.016049
SEC_PER_YR = 365.25 * 24.0 * 3600.0

RUN_DIRS = {
    "L": Path("openmc_tokamak_mode_comparison/output/torus_mode_l"),
    "H": Path("openmc_tokamak_mode_comparison/output/torus_mode_h"),
    "A": Path("openmc_tokamak_mode_comparison/output/torus_mode_a"),
}

PLOT_CONFIG = {
    "fast_flux": {
        "tally_name": "Fast flux x-y-z map",
        "ylabel": r"Fast flux  $[n/(cm^2 \cdot s)]$",
        "title": "Radial fast-flux comparison",
        "outfile": "radial_compare_fast_flux.png",
        "yscale": "log",
    },
    "tritium": {
        "tally_name": "Tritium production x-y-z map (FLiBe only)",
        "ylabel": r"Tritium breeding rate  $\left[\frac{g}{m^3 \, yr}\right]$",
        "title": "Radial tritium-breeding comparison",
        "outfile": "radial_compare_tritium.png",
        "yscale": "log",
    },
}

def reshape_mesh(mean_1d: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    return mean_1d.reshape((nx, ny, nz), order="F")


def find_latest_statepoint(run_dir: Path) -> Path:
    statepoints = sorted(run_dir.glob("statepoint.*.h5"))
    if not statepoints:
        raise FileNotFoundError(f"No statepoint .h5 file found in {run_dir}")
    return statepoints[-1]


def load_mesh_tally(mode: str, tally_name: str):
    run_dir = RUN_DIRS[mode]
    statepoint_path = find_latest_statepoint(run_dir)

    with openmc.StatePoint(statepoint_path) as sp:
        tally = sp.get_tally(name=tally_name)

        mesh_filter = next(
            (f for f in tally.filters if isinstance(f, openmc.MeshFilter)),
            None,
        )
        if mesh_filter is None:
            raise RuntimeError(f"No MeshFilter found for tally '{tally_name}'")

        mesh = mesh_filter.mesh
        nx, ny, nz = mesh.dimension
        ll = np.array(mesh.lower_left, dtype=float)
        ur = np.array(mesh.upper_right, dtype=float)

        x_edges = np.linspace(ll[0], ur[0], nx + 1)
        y_edges = np.linspace(ll[1], ur[1], ny + 1)
        z_edges = np.linspace(ll[2], ur[2], nz + 1)

        dx, dy, dz = (ur - ll) / np.array([nx, ny, nz], dtype=float)

        mean_1d = np.squeeze(tally.mean).ravel()
        mean_3d = reshape_mesh(mean_1d, nx, ny, nz)

    return mean_3d, x_edges, y_edges, z_edges, dx, dy, dz


def build_physical_field(plot_type: str, mean_3d: np.ndarray, dx: float, dy: float, dz: float, source_n_per_s: float):
    voxel_vol_cm3 = dx * dy * dz
    
    if plot_type == "fast_flux":
        # openmc flux tally gives a tracklegnth quantity per souce particle
        # need to divide by voxel volume and multiply by physical source strenght
        field = (mean_3d / voxel_vol_cm3) * source_n_per_s
        return field

    if plot_type == "tritium":
        # tally gives tritium producing reactions per source particle per voxel
        # convert to tritons/s then g/yr and divide by voxel volume
        voxel_vol_m3 = voxel_vol_cm3 * 1.0e-6
        tritons_per_s = mean_3d * source_n_per_s
        grams_per_yr = (tritons_per_s * SEC_PER_YR / N_A) * M_T_G_PER_MOL
        field = grams_per_yr / voxel_vol_m3
        return field

    raise ValueError(f"Unknown plot_type: {plot_type}")


def build_midplane_radial_profile(plot_type: str, mode: str, n_bins: int = N_BINS):
    mean_3d, x_edges, y_edges, z_edges, dx, dy, dz = load_mesh_tally(
        mode=mode,
        tally_name=PLOT_CONFIG[plot_type]["tally_name"],
    )
    source_n_per_s = MODE_SOURCE_RATE_N_PER_S[mode]
    field_3d = build_physical_field(plot_type, mean_3d, dx, dy, dz, source_n_per_s)

    ny = field_3d.shape[1]
    j0 = ny // 2
    jlo = max(0, j0 - Y_HALF)
    jhi = min(ny, j0 + Y_HALF + 1)

    field_xz = np.nanmean(field_3d[:, jlo:jhi, :], axis=1)

    x_cent = 0.5 * (x_edges[:-1] + x_edges[1:])
    z_cent = 0.5 * (z_edges[:-1] + z_edges[1:])
    Xc, Zc = np.meshgrid(x_cent, z_cent, indexing="ij")

    r_minor = np.sqrt((np.abs(Xc) - R0) ** 2 + Zc**2)

    if plot_type == "fast_flux":
        region_mask = r_minor <= flibe_r
    elif plot_type == "tritium":
        region_mask = (r_minor >= first_wall_r) & (r_minor <= flibe_r)
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")

    valid = region_mask & np.isfinite(field_xz) & (field_xz > 0.0)

    bin_edges = np.linspace(0.0, flibe_r, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    profile_mean = np.full(n_bins, np.nan)
    profile_std = np.full(n_bins, np.nan)

    for i in range(n_bins):
        in_bin = (
            (r_minor >= bin_edges[i]) &
            (r_minor < bin_edges[i + 1]) &
            valid
        )

        vals = field_xz[in_bin]
        if vals.size > 0:
            profile_mean[i] = np.nanmean(vals)
            profile_std[i] = np.nanstd(vals)

    return bin_centers, profile_mean, profile_std


def add_region_lines(ax):
    ax.axvline(plasma_r, linestyle="--", linewidth=1.5, label="Plasma edge")
    ax.axvline(first_wall_r, linestyle="--", linewidth=1.5, label="First wall outer edge")
    ax.axvline(flibe_r, linestyle="--", linewidth=1.5, label="FLiBe outer edge")


def main():
    parser = argparse.ArgumentParser(
        description="plot radial L/H/A mode comparisons from OpenMC mesh tallies"
    )
    parser.add_argument(
        "--plot-type",
        type=str,
        required=True,
        choices=["fast_flux", "tritium"],
        help="which radial comparison to make",
    )
    args = parser.parse_args()

    cfg = PLOT_CONFIG[args.plot_type]
    output_dir = Path("openmc_tokamak_mode_comparison/plotting/output/2D_combined")
    output_dir.mkdir(parents=True, exist_ok=True)
    outpath = output_dir / cfg["outfile"]

    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    for mode in ["L", "H", "A"]:
        r, mean_prof, std_prof = build_midplane_radial_profile(args.plot_type, mode)

        ax.plot(r, mean_prof, linewidth=2.0, label=f"{mode}-mode")
    
    add_region_lines(ax)

    ax.set_xlim(0.0, flibe_r)
    ax.set_xlabel("Minor-radius distance from plasma center [cm]")
    ax.set_ylabel(cfg["ylabel"])
    ax.set_title(cfg["title"])

    if cfg["yscale"] == "log":
        ax.set_yscale("log")

    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    # Optional region labels
    y0, y1 = ax.get_ylim()
    y_text = 10 ** (0.15 * np.log10(y1) + 0.85 * np.log10(y0)) if ax.get_yscale() == "log" else (y0 + 0.85 * (y1 - y0))

    ax.text(0.5 * plasma_r, y_text, "Plasma", ha="center", va="center")
    ax.text(0.5 * (plasma_r + first_wall_r), y_text, "First wall", ha="center", va="center")
    ax.text(0.5 * (first_wall_r + flibe_r), y_text, "FLiBe", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

    print(f"Wrote {outpath}")


if __name__ == "__main__":
    main()