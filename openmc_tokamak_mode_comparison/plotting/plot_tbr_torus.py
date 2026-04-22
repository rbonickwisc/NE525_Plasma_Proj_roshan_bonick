from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext, LogLocator
import openmc

TALLY_NAME = "Tritium production x-y-z map (FLiBe only)"

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

PLOW = 5.0
PHIGH = 99.0
MARGIN_CM = 30.0

Y_HALF = 5
INPAINT_ITERS = 40

N_A = 6.02214076e23
M_T_G_PER_MOL = 3.016049
SEC_PER_YR = 365.25 * 24 * 3600.0


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
        raise FileNotFoundError(f"No statepoint .h5 file found in {run_dir}")
    return statepoints[-1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot tritium breeding map for L / H / A torus mode output"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="L",
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
    out_png = plot_dir / f"{mode.lower()}_tritium_breeding_poloidal_xz_y0.png"

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
        voxel_vol_m3 = voxel_vol_cm3 * 1e-6

        mean_1d = np.squeeze(tally.mean).ravel()
        mean_3d = reshape_mesh(mean_1d, nx, ny, nz)

    tritons_per_s = mean_3d * source_n_per_s
    grams_per_yr = (tritons_per_s * SEC_PER_YR / N_A) * M_T_G_PER_MOL
    rate_xyz = grams_per_yr / voxel_vol_m3

    j0 = ny // 2
    jlo = max(0, j0 - Y_HALF)
    jhi = min(ny, j0 + Y_HALF + 1)
    rate_xz = np.nanmean(rate_xyz[:, jlo:jhi, :], axis=1)

    A = rate_xz.copy()
    A = np.where(np.isfinite(A) & (A > 0.0), A, np.nan)

    x_cent = 0.5 * (x_edges[:-1] + x_edges[1:])
    z_cent = 0.5 * (z_edges[:-1] + z_edges[1:])
    Xc, Zc = np.meshgrid(x_cent, z_cent, indexing="ij")
    r2 = (np.abs(Xc) - R0) ** 2 + Zc**2
    inside_flibe = (r2 <= flibe_r**2) & (r2 >= first_wall_r**2)

    A = np.where(inside_flibe, A, np.nan)
    A = fill_nans_neighbor_mean(A, fill_region=inside_flibe, n_iter=INPAINT_ITERS)

    vals = A[np.isfinite(A) & (A > 0.0) & inside_flibe]
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
        (plasma_r, "-", 1.6),
        (first_wall_r, "--", 1.6),
        (flibe_r, "-", 2.2),
    ]:
        circle_xz(ax, +R0, 0.0, r, color="white", ls=ls, lw=lw, zorder=5)
        circle_xz(ax, -R0, 0.0, r, color="white", ls=ls, lw=lw, zorder=5)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("z [cm]")
    ax.set_title(
        f"Tritium breeding rate (poloidal x-z slice) ({mode}-mode), "
        f"S = {source_n_per_s:.1e} n/s"
    )

    xlim = R0 + flibe_r + MARGIN_CM
    zlim = flibe_r + MARGIN_CM
    ax.set_xlim(-xlim, +xlim)
    ax.set_ylim(-zlim, +zlim)

    tick_step = 50.0
    max_tick = tick_step * np.ceil(xlim / tick_step)
    ticks = np.arange(-max_tick, max_tick + 0.1, tick_step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.0f}" for t in ticks])

    cbar = plt.colorbar(pcm, orientation="horizontal", pad=0.12)
    cbar.set_label(r"Tritium breeding rate  $\left[\frac{g}{m^3\,yr}\right]$")

    y_off = -0.45
    cbar.ax.text(-0.04, y_off, f"{vmin:.2e}", transform=cbar.ax.transAxes,
                 ha="left", va="top")
    cbar.ax.text(1.04, y_off, f"{vmax:.2e}", transform=cbar.ax.transAxes,
                 ha="right", va="top")

    cbar.ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    cbar.ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    cbar.ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    cbar.update_ticks()

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()