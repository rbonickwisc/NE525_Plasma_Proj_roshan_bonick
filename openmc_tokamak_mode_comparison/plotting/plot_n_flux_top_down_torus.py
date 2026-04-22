from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import openmc

TALLY_NAME = "Fast flux x-y-z map"

MODE_SOURCE_RATE_N_PER_S = {
    "L": 3.150941e18,
    "H": 1.895748e19,
    "A": 5.326079e19,
}

#torus geometry [cm]
R0 = 200.0
plasma_r = 20.0
first_wall_r = 50.0
flibe_r = 100.0

MASK_OUTSIDE_DEVICE = True
INPAINT_ITERS = 40

AUTOSCALE = True
PLOW = 2.0
PHIGH = 99.5

#reshape 1D array into 3D array
def reshape_mesh(mean_1d, nx, ny, nz):
    return mean_1d.reshape((nx, ny, nz), order="F")

#create two circles in x-y plane corresponding to torus cross section
def ring_outline_xy(ax, R0, a, **kwargs):
    th = np.linspace(0, 2 * np.pi, 800)
    r_in = max(R0 - a, 0.0)
    r_out = R0 + a
    ax.plot(r_in * np.cos(th), r_in * np.sin(th), **kwargs)
    ax.plot(r_out * np.cos(th), r_out * np.sin(th), **kwargs)

#define NaNs(inside torus) by replacing NaN cells with mean of their four neighbors
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot top-down fast neutron flux map for L / H / A torus mode output"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["L", "H", "A"],
        help="Tokamak confinement mode output to plot",
    )
    args = parser.parse_args()

    mode = args.mode.upper()
    source_n_per_s = MODE_SOURCE_RATE_N_PER_S[mode]
    run_dir = Path(f"openmc_tokamak_mode_comparison/output/torus_mode_{mode.lower()}")
    plot_dir = Path(f"openmc_tokamak_mode_comparison/plotting/output/torus_mode_{mode.lower()}")
    plot_dir.mkdir(parents=True, exist_ok=True)

    statepoint_path = find_latest_statepoint(run_dir)
    out_png = plot_dir / f"{mode.lower()}_fast_flux_topdown_xy.png"

    with openmc.StatePoint(statepoint_path) as sp:
        tally = sp.get_tally(name=TALLY_NAME)

        mesh_filter = next(
            (f for f in tally.filters if isinstance(f, openmc.MeshFilter)),
            None,
        )
        if mesh_filter is None:
            raise RuntimeError("No MeshFilter found on the tally")

        mesh = mesh_filter.mesh
        nx, ny, nz = mesh.dimension
        ll = np.array(mesh.lower_left, dtype=float)
        ur = np.array(mesh.upper_right, dtype=float)

        x_edges = np.linspace(ll[0], ur[0], nx + 1)
        y_edges = np.linspace(ll[1], ur[1], ny + 1)
        z_edges = np.linspace(ll[2], ur[2], nz + 1)

        dx, dy, dz = (ur - ll) / np.array([nx, ny, nz], dtype=float)
        voxel_vol_cm3 = dx * dy * dz

        mean_1d = np.squeeze(tally.mean).ravel()
        mean_3d = reshape_mesh(mean_1d, nx, ny, nz)

    flux_cm2_s = (mean_3d / voxel_vol_cm3) * source_n_per_s

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

    fill_region = inside_device if MASK_OUTSIDE_DEVICE else np.ones_like(A, dtype=bool)

    if MASK_OUTSIDE_DEVICE:
        A = np.where(inside_device, A, np.nan)

    A = fill_nans_neighbor_mean(A, fill_region=fill_region, n_iter=INPAINT_ITERS)

    vals = A[np.isfinite(A) & (A > 0.0) & fill_region]
    vmin = float(np.nanpercentile(vals, PLOW))
    vmax = float(np.nanpercentile(vals, PHIGH))
    vmin = max(vmin, 1e-30)
    if vmax <= vmin:
        vmax = 10.0 * vmin

    fig = plt.figure(figsize=(7.5, 7.5))
    ax = plt.gca()

    pcm = ax.pcolormesh(
        x_edges,
        y_edges,
        A.T,
        shading="auto",
        norm=LogNorm(vmin=vmin, vmax=vmax),
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
    ax.set_title(f"Fast neutron flux (E > 0.1 MeV) (top-down view) ({mode}-mode)")

    cbar = plt.colorbar(pcm, orientation="horizontal", pad=0.12)
    cbar.set_label(r"Fast flux  $[n/(cm^2\cdot s)]$")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()