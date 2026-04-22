from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tokamak_source_model.utils.case_builder import build_l_mode_model
from source_studies.demo_case_util import run_demo_case

def main() -> None:
    run_demo_case(
        model=build_l_mode_model(),
        output_dir=Path("source_studies/output/demos/l_mode"),
        mode_label="L-mode",
        file_prefix="l_mode",
        seed=42,
    )

if __name__ == "__main__":
    main()