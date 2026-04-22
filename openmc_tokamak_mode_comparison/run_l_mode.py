from __future__ import annotations

import subprocess
import sys
from pathlib import Path

def main() -> None:
    repo_root = Path(__file__).resolve().parent
    build_script = repo_root / "torus_build.py"
    output_dir = repo_root / "output" / "torus_mode_l"

    output_dir.mkdir(parents=True, exist_ok=True)

    #export .xml files for L-mode
    subprocess.run(
        [sys.executable, str(build_script), "--mode", "L"],
        check=True,
    )

    #run openmc inside folder containing those .xml files
    log_path = output_dir / "run.log"

    with open(log_path, "w") as log_file:
        process = subprocess.Popen(
            ["openmc"],
            cwd=output_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)

        return_code = process.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, ["openmc"])

    print(f"L-mode openmc run finished, saved outputs in {output_dir}")

if __name__ == "__main__":
    main()