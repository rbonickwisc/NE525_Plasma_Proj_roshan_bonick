from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import openmc
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext, LogLocator


MODE_SOURCE_RATE_N_PER_S = {
    "L": 3.150941e18,
    "H": 1.895748e19,
    "A": 5.326079e19,
}

# torus geometry [cm]
R0 = 200.0
plasma_r = 20.0
first_wall_r = 50.0
flibe_r = 100.0

MARGIN_CM = 30.0
XTICK_STEP_CM = 50.0
ZTICK_STEP_CM = 50.0

INPAINT_ITERS = 40
Y_HALF = 5

N_A = 6.02214076e23
M_T_G_PER_MOL = 3.016049
SEC_PER_YR = 365.25 * 24 * 3600.0


PLOT_CONFIG = {
    "topdown_flux": {
        "tally_name": "Fast flux x-y-z map",
        "title": "Fast neutron flux (top-down view)",
        "colorbar_label": r"Fast flux  $[n/(cm^2\cdot s)]$",
        "outfile": "compare_topdown_fast_flux_shared.png",
        "plow": 2.0,
        "phigh": 99.5,
    },
    "poloidal_flux": {
        "tally_name": "Fast flux x-y-z map",
        "title": "Fast neutron flux (poloidal x-z slice)",
        "colorbar_label": r"Fast flux  $[n/(cm^2\cdot s)]$",
        "outfile": "compare_poloidal_fast_flux_shared.png",
        "plow": 2.0,
        "phigh": 99.5,
    },
    "tritium": {
        "tally_name": "Tritium production x-y-z map (FLiBe only)",
        "title": "Tritium breeding rate (poloidal x-z slice)",
        "colorbar_label": r"Tritium breeding rate  $\left[\frac{g}{m^3\,yr}\right]$",
        "outfile": "compare_tritium_breeding_shared.png",
        "plow": 5.0,
        "phigh": 99.0,
    },
}


def reshape_mesh(mean_1d, nx, ny, nz):
    return mean_1d.reshape((nx, ny, nz), order="F")


def ring_outline_xy(ax, R0, a, **kwargs):
    th = np.linspace(0, 2 * np.pi, 800)
    r_in = max(R0 - a, 0.0)
    r_out = R0 + a
    ax.plot(r_in * np.cos(th), r_in * np.sin(th), **kwargs)
    ax.plot(r_out * np.cos(th), r_out * np.sin(th), **kwargs)


def circle_xz(ax, xc, zc, r, **kw):
    th = np.linspace(0, 2 * np.pi, 800)
    ax.plot(xc + r * np.cos(th), zc + r * np.sin(th), **kw)


def fill_nans_neighbor_mean(A, fill_region=None, n_iter=30):
    B = A.copy()
    if fill_region is None:
        fill_region = np.ones_like(B, dtype=bool)

    for _ in range(n_iter):
        nanmask = (~np.isfinite(B)) & fill_region
        if not nanmask.any():
            break

        P = np.pad(B, 1, mode="constant", constant_values=np.nan)
        up, down = P[:-2, 1:-1], P[2:, 1:-1]
        left, right = P[1:-1, :-2], P[1:-1, 2:]

        s = np.zeros_like(B, dtype=float)
        c = np.zeros_like(B, dtype=float)

        for nb in (up, down, left, right):
            ok = np.isfinite(nb)
            s += np.where(ok, nb, 0.0)
            c += ok.astype(float)

        update = nanmask & (c > 0)
        if not update.any():
            break

        B[update] = s[update] / c[update]

    return B


def find_latest_statepoint(run_dir: Path) -> Path:
    statepoints = sorted(run_dir.glob("statepoint.*.h5"))
    if not statepoints:
        raise FileNotFoundError(f"No statepoint h5 file found in {run_dir}")
    return statepoints[-1]


def load_mesh_data(mode: str, tally_name: str):
    run_dir = Path(f"openmc_tokamak_mode_comparison/output/torus_mode_{mode}")
    statepoint_path = find_latest_statepoint(run_dir)

    with openmc.StatePoint(statepoint_path) as sp:
        tally = sp.get_tally(name=tally_name)

        mesh_filter = next(
            (f for f in tally.filters if isinstance(f, openmc.MeshFilter)),
            None,
        )
        if mesh_filter is None:
            raise RuntimeError(f"No MeshFilter found on tally '{tally_name}'")

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


def build_topdown_flux(mode: str):
    mean_3d, x_edges, y_edges, z_edges, dx, dy, dz = load_mesh_data(
        mode=mode,
        tally_name="Fast flux x-y-z map",
    )

    voxel_vol_cm3 = dx * dy * dz
    source_n_per_s = MODE_SOURCE_RATE_N_PER_S[mode.upper()]
    flux_cm2_s = (mean_3d / voxel_vol_cm3) * source_n_per_s
    nz = mean_3d.shape[2]
    k = nz // 2
    A = flux_cm2_s[:, :, k]
    A = np.where(np.isfinite(A) & (A > 0.0), A, np.nan)

    x_cent = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_cent = 0.5 * (y_edges[:-1] + y_edges[1:])
    Xc, Yc = np.meshgrid(x_cent, y_cent, indexing="ij")
    Rc = np.sqrt(Xc**2 + Yc**2)

    r_in = max(R0 - flibe_r, 0.0)
    r_out = R0 + flibe_r
    inside_device = (Rc >= r_in) & (Rc <= r_out)

    A = np.where(inside_device, A, np.nan)
    A = fill_nans_neighbor_mean(A, fill_region=inside_device, n_iter=INPAINT_ITERS)

    return {
        "A": A,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "mask": inside_device,
    }


def build_poloidal_flux(mode: str):
    mean_3d, x_edges, y_edges, z_edges, dx, dy, dz = load_mesh_data(
        mode=mode,
        tally_name="Fast flux x-y-z map",
    )

    voxel_vol_cm3 = dx * dy * dz
    source_n_per_s = MODE_SOURCE_RATE_N_PER_S[mode.upper()]
    flux_cm2_s = (mean_3d / voxel_vol_cm3) * source_n_per_s
    ny = mean_3d.shape[1]
    j0 = ny // 2
    jlo = max(0, j0 - Y_HALF)
    jhi = min(ny, j0 + Y_HALF + 1)

    A = np.nanmean(flux_cm2_s[:, jlo:jhi, :], axis=1)

    x_cent = 0.5 * (x_edges[:-1] + x_edges[1:])
    z_cent = 0.5 * (z_edges[:-1] + z_edges[1:])
    Xc, Zc = np.meshgrid(x_cent, z_cent, indexing="ij")
    inside_flibe = ((np.abs(Xc) - R0) ** 2 + Zc**2) <= flibe_r**2

    A = np.where(inside_flibe, A, np.nan)
    A = fill_nans_neighbor_mean(A, fill_region=inside_flibe, n_iter=INPAINT_ITERS)

    pos = A[np.isfinite(A) & (A > 0.0) & inside_flibe]
    floor = float(np.nanpercentile(pos, 1.0))
    A = np.where(np.isfinite(A) & (A > 0.0), A, floor)
    A = np.where(inside_flibe & np.isfinite(A) & (A > 0.0), A, np.nan)
    A = np.where(inside_flibe & (~np.isfinite(A) | (A <= 0.0)), floor, A)

    return {
        "A": A,
        "x_edges": x_edges,
        "z_edges": z_edges,
        "mask": inside_flibe,
    }


def build_tritium(mode: str):
    mean_3d, x_edges, y_edges, z_edges, dx, dy, dz = load_mesh_data(
        mode=mode,
        tally_name="Tritium production x-y-z map (FLiBe only)",
    )

    voxel_vol_cm3 = dx * dy * dz
    voxel_vol_m3 = voxel_vol_cm3 * 1e-6

    source_n_per_s = MODE_SOURCE_RATE_N_PER_S[mode.upper()]
    tritons_per_s = mean_3d * source_n_per_s    
    grams_per_yr = (tritons_per_s * SEC_PER_YR / N_A) * M_T_G_PER_MOL
    rate_xyz = grams_per_yr / voxel_vol_m3

    ny = mean_3d.shape[1]
    j0 = ny // 2
    jlo = max(0, j0 - Y_HALF)
    jhi = min(ny, j0 + Y_HALF + 1)

    A = np.nanmean(rate_xyz[:, jlo:jhi, :], axis=1)
    A = np.where(np.isfinite(A) & (A > 0.0), A, np.nan)

    x_cent = 0.5 * (x_edges[:-1] + x_edges[1:])
    z_cent = 0.5 * (z_edges[:-1] + z_edges[1:])
    Xc, Zc = np.meshgrid(x_cent, z_cent, indexing="ij")
    r2 = (np.abs(Xc) - R0) ** 2 + Zc**2
    inside_flibe = (r2 <= flibe_r**2) & (r2 >= first_wall_r**2)

    A = np.where(inside_flibe, A, np.nan)
    A = fill_nans_neighbor_mean(A, fill_region=inside_flibe, n_iter=INPAINT_ITERS)

    return {
        "A": A,
        "x_edges": x_edges,
        "z_edges": z_edges,
        "mask": inside_flibe,
    }


def get_plot_data(plot_type: str, mode: str):
    if plot_type == "topdown_flux":
        return build_topdown_flux(mode)
    if plot_type == "poloidal_flux":
        return build_poloidal_flux(mode)
    if plot_type == "tritium":
        return build_tritium(mode)
    raise ValueError(f"Unknown plot_type: {plot_type}")


def compute_shared_limits(data_list, plow: float, phigh: float):
    all_vals = []
    for d in data_list:
        A = d["A"]
        mask = d["mask"]
        vals = A[np.isfinite(A) & (A > 0.0) & mask]
        all_vals.append(vals)

    all_vals = np.concatenate(all_vals)
    vmin = float(np.nanpercentile(all_vals, plow))
    vmax = float(np.nanpercentile(all_vals, phigh))
    vmin = max(vmin, 1e-30)
    if vmax <= vmin:
        vmax = 10.0 * vmin
    return vmin, vmax


def plot_topdown_panel(ax, d, mode_label: str, norm):
    A = d["A"]
    x_edges = d["x_edges"]
    y_edges = d["y_edges"]

    pcm = ax.pcolormesh(
        x_edges,
        y_edges,
        A.T,
        shading="auto",
        norm=norm,
    )

    ring_outline_xy(ax, R0, plasma_r, color="white", lw=2.0, ls="-", zorder=5)
    ring_outline_xy(ax, R0, first_wall_r, color="white", lw=2.0, ls="--", zorder=5)
    ring_outline_xy(ax, R0, flibe_r, color="white", lw=2.5, ls="-", zorder=5)

    ax.set_aspect("equal", adjustable="box")
    lim = R0 + flibe_r + 30.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    ax.set_title(mode_label)

    return pcm


def plot_poloidal_panel(ax, d, mode_label: str, norm):
    A = d["A"]
    x_edges = d["x_edges"]
    z_edges = d["z_edges"]

    pcm = ax.pcolormesh(
        x_edges,
        z_edges,
        A.T,
        shading="auto",
        norm=norm,
    )

    for r, ls, lw in [
        (plasma_r, "-", 2.0),
        (first_wall_r, "--", 2.0),
        (flibe_r, "-", 2.5),
    ]:
        circle_xz(ax, +R0, 0.0, r, color="white", ls=ls, lw=lw, zorder=5)
        circle_xz(ax, -R0, 0.0, r, color="white", ls=ls, lw=lw, zorder=5)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("z [cm]")
    ax.set_title(mode_label)

    xlim = R0 + flibe_r + MARGIN_CM
    zlim = flibe_r + MARGIN_CM
    ax.set_xlim(-xlim, +xlim)
    ax.set_ylim(-zlim, +zlim)

    max_xtick = XTICK_STEP_CM * np.ceil(xlim / XTICK_STEP_CM)
    xticks = np.arange(-max_xtick, max_xtick + 0.1, XTICK_STEP_CM)
    ax.set_xticks(xticks)

    max_ztick = ZTICK_STEP_CM * np.ceil(zlim / ZTICK_STEP_CM)
    zticks = np.arange(-max_ztick, max_ztick + 0.1, ZTICK_STEP_CM)
    ax.set_yticks(zticks)

    return pcm


def plot_tritium_panel(ax, d, mode_label: str, norm):
    A = d["A"]
    x_edges = d["x_edges"]
    z_edges = d["z_edges"]

    pcm = ax.pcolormesh(
        x_edges,
        z_edges,
        A.T,
        shading="auto",
        norm=norm,
    )

    for r, ls, lw in [
        (plasma_r, "-", 1.6),
        (first_wall_r, "--", 1.6),
        (flibe_r, "-", 2.2),
    ]:
        circle_xz(ax, +R0, 0.0, r, color="white", ls=ls, lw=lw, zorder=5)
        circle_xz(ax, -R0, 0.0, r, color="white", ls=ls, lw=lw, zorder=5)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("z [cm]")
    ax.set_title(mode_label)

    xlim = R0 + flibe_r + MARGIN_CM
    zlim = flibe_r + MARGIN_CM
    ax.set_xlim(-xlim, +xlim)
    ax.set_ylim(-zlim, +zlim)

    tick_step = 50.0
    max_tick = tick_step * np.ceil(xlim / tick_step)
    ticks = np.arange(-max_tick, max_tick + 0.1, tick_step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.0f}" for t in ticks])

    return pcm


def main():
    parser = argparse.ArgumentParser(
        description="Create L/H/A comparison plots with a shared colorbar"
    )
    parser.add_argument(
        "--plot-type",
        type=str,
        required=True,
        choices=["topdown_flux", "poloidal_flux", "tritium"],
        help="Which comparison plot to make",
    )
    args = parser.parse_args()

    cfg = PLOT_CONFIG[args.plot_type]
    modes = ["L", "H", "A"]
    mode_labels = ["L-mode", "H-mode", "A-mode"]
    data_list = [get_plot_data(args.plot_type, mode) for mode in modes]
    vmin, vmax = compute_shared_limits(data_list, cfg["plow"], cfg["phigh"])
    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    pcm = None
    for ax, d, label in zip(axes, data_list, mode_labels):
        if args.plot_type == "topdown_flux":
            pcm = plot_topdown_panel(ax, d, label, norm)
        elif args.plot_type == "poloidal_flux":
            pcm = plot_poloidal_panel(ax, d, label, norm)
        elif args.plot_type == "tritium":
            pcm = plot_tritium_panel(ax, d, label, norm)

    fig.suptitle(cfg["title"], fontsize=16)

    cbar = fig.colorbar(
        pcm,
        ax=axes,
        orientation="horizontal",
        fraction=0.06,
        pad=0.08,
    )
    cbar.set_label(cfg["colorbar_label"])

    if args.plot_type == "tritium":
        cbar.ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
        cbar.ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        cbar.ax.xaxis.set_minor_locator(
            LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1)
        )
        cbar.update_ticks()

    output_dir = Path("openmc_tokamak_mode_comparison/plotting/output/combined_shared_colorbar")
    output_dir.mkdir(parents=True, exist_ok=True)
    outpath = output_dir / cfg["outfile"]

    plt.savefig(outpath, dpi=300)
    plt.close()

    print(f"Wrote {outpath}")
    print(f"Shared color scale: vmin={vmin:.6e}, vmax={vmax:.6e}")


if __name__ == "__main__":
    main()