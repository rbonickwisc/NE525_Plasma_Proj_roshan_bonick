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

PLOW = 2.0
PHIGH = 99.5

MARGIN_CM = 30.0
XTICK_STEP_CM = 50.0
ZTICK_STEP_CM = 50.0

MASK_OUTSIDE_FLIBE = True
INPAINT_ITERS = 40

def reshape_mesh(mean_1d, nx, ny, nz):
    return mean_1d.reshape((nx, ny, nz), order="F")


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

        P = np.pad(B, pad_width=1, mode="constant", constant_values=np.nan)
        up = P[:-2, 1:-1]
        down = P[2:, 1:-1]
        left = P[1:-1, :-2]
        right = P[1:-1, 2:]

        neighbors = [up, down, left, right]

        s = np.zeros_like(B, dtype=float)
        c = np.zeros_like(B, dtype=float)

        for nb in neighbors:
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
        description="Plot poloidal fast neutron flux map for L / H / A torus mode output"
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
    out_png = plot_dir / f"{mode.lower()}_fast_flux_poloidal_xz_y0.png"

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

    j0 = ny // 2
    half = 5
    jlo = max(0, j0 - half)
    jhi = min(ny, j0 + half + 1)

    A = np.nanmean(flux_cm2_s[:, jlo:jhi, :], axis=1)

    x_cent = 0.5 * (x_edges[:-1] + x_edges[1:])
    z_cent = 0.5 * (z_edges[:-1] + z_edges[1:])
    Xc, Zc = np.meshgrid(x_cent, z_cent, indexing="ij")
    inside_flibe = ((np.abs(Xc) - R0) ** 2 + Zc**2) <= flibe_r**2

    fill_region = inside_flibe if MASK_OUTSIDE_FLIBE else np.ones_like(A, dtype=bool)

    if MASK_OUTSIDE_FLIBE:
        A = np.where(inside_flibe, A, np.nan)

    A = fill_nans_neighbor_mean(A, fill_region=fill_region, n_iter=INPAINT_ITERS)

    pos = A[np.isfinite(A) & (A > 0.0) & fill_region]
    floor = float(np.nanpercentile(pos, 1.0))
    A = np.where(np.isfinite(A) & (A > 0.0), A, floor)
    A = np.where(fill_region & np.isfinite(A) & (A > 0.0), A, np.nan)
    A = np.where(fill_region & (~np.isfinite(A) | (A <= 0.0)), floor, A)

    vals = A[np.isfinite(A) & (A > 0) & fill_region]
    vmin = float(np.nanpercentile(vals, PLOW))
    vmax = float(np.nanpercentile(vals, PHIGH))
    vmin = max(vmin, 1e-30)
    if vmax <= vmin:
        vmax = 10.0 * vmin

    fig = plt.figure(figsize=(9, 6))
    ax = plt.gca()

    pcm = ax.pcolormesh(
        x_edges,
        z_edges,
        A.T,
        shading="auto",
        norm=LogNorm(vmin=vmin, vmax=vmax),
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
    ax.set_title(f"Fast neutron flux (E > 0.1 MeV) (poloidal x-z slice) ({mode}-mode)")

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

    cbar = plt.colorbar(pcm, orientation="horizontal", pad=0.12)
    cbar.set_label(r"Fast flux  $[n/(cm^2\cdot s)]$")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()