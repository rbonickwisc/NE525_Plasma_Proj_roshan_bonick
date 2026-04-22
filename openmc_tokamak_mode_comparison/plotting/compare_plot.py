from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


PLOT_CONFIG = {
    "topdown_flux": {
        "filename": "{mode}_fast_flux_topdown_xy.png",
        "title": "Fast neutron flux (top-down view)",
        "outname": "compare_topdown_fast_flux.png",
    },
    "poloidal_flux": {
        "filename": "{mode}_fast_flux_poloidal_xz_y0.png",
        "title": "Fast neutron flux (poloidal x-z slice)",
        "outname": "compare_poloidal_fast_flux.png",
    },
    "tritium": {
        "filename": "{mode}_tritium_breeding_poloidal_xz_y0.png",
        "title": "Tritium breeding rate (poloidal x-z slice)",
        "outname": "compare_tritium_breeding.png",
    },
}

def load_plot(mode: str, filename_pattern: str):
    path = Path(
        f"openmc_tokamak_mode_comparison/plotting/output/torus_mode_{mode}/"
        f"{filename_pattern.format(mode=mode)}"
    )
    if not path.exists():
        raise FileNotFoundError(f"Could not find plot: {path}")
    return mpimg.imread(path), path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="combine L/H/A torus PNG plots into one comparison figure"
    )
    parser.add_argument(
        "--plot-type",
        type=str,
        required=True,
        choices=list(PLOT_CONFIG.keys()),
        help="which type of plot to combine",
    )
    args = parser.parse_args()

    cfg = PLOT_CONFIG[args.plot_type]

    l_img, l_path = load_plot("l", cfg["filename"])
    h_img, h_path = load_plot("h", cfg["filename"])
    a_img, a_path = load_plot("a", cfg["filename"])

    output_dir = Path("openmc_tokamak_mode_comparison/plotting/output/combined_plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    outpath = output_dir / cfg["outname"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, img, label in zip(
        axes,
        [l_img, h_img, a_img],
        ["L-mode", "H-mode", "A-mode"],
    ):
        ax.imshow(img)
        ax.set_title(label)
        ax.axis("off")

    fig.suptitle(cfg["title"], fontsize=16)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {outpath}")
    print("Used:")
    print(f"  {l_path}")
    print(f"  {h_path}")
    print(f"  {a_path}")


if __name__ == "__main__":
    main()