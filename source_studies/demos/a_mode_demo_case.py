from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tokamak_source_model.utils.case_builder import build_a_mode_paper_model
from source_studies.demo_case_util import run_demo_case

def main() -> None:
    run_demo_case(
        model=build_a_mode_paper_model(),
        output_dir=Path("source_studies/output/demos/a_mode"),
        mode_label="A-mode",
        file_prefix="a_mode",
        seed=44,
    )

if __name__ == "__main__":
    main()