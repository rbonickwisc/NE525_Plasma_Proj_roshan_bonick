from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all torus plotting scripts for a chosen confinemnet mode"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="L",
        choices=["L", "H", "A"],
        help="Tokamak confinement mode to plot"
    )
    args = parser.parse_args()
    mode=args.mode.upper()
    base_dir = Path(__file__).resolve().parent

    plot_scripts = [
        "plot_n_flux_top_down_torus.py",
        "plot_poloidal_flux_torus.py",
        "plot_tbr_torus.py",
    ]

    print(f"Running all torus plots for mode={mode}")

    for script_name in plot_scripts:
        script_path = base_dir / script_name
        
        subprocess.run(
            [sys.executable, str(script_path), "--mode", mode],
            check=True,
        )

    print(f"finished torus plots for {mode} mode")

if __name__ == "__main__":
    main()