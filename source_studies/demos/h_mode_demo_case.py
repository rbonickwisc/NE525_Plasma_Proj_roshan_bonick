from __future__ import annotations
from pathlib import Path

from tokamak_source_model.utils.case_builder import build_generic_pedestal_model
from utils.demo_case_util import run_demo_case

def main() -> None:
    run_demo_case(
        model=build_generic_pedestal_model(),
        output_dir=Path("source_studies/output/demos/h_mode"),
        mode_label="H-mode",
        file_prefix="h_mode",
        seed=43,
    )

if __name__ == "__main__":
    main()