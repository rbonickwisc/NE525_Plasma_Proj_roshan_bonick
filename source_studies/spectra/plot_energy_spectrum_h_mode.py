from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tokamak_source_model.utils.case_builder import build_default_mesh, build_generic_pedestal_model
from tokamak_source_model.utils.sampling import sample_source_particles

def main() -> None:
    output_dir = Path("source_studies/output/energy_spectrum")
    output_dir.mkdir(parents=True, exist_ok=True)

    model= build_generic_pedestal_model()
    mesh = build_default_mesh()
    rng = np.random.default_rng(42)
    
    samples = sample_source_particles(
        n_samples=50000,
        model=model,
        mesh=mesh,
        rng=rng,
    )

    energies_MeV = samples.energy_eV / 1.0e6

    print("H-mode Ballabio DT spectrum")
    print("-" * 25)
    print(f"Mean energy [MeV] = {np.mean(energies_MeV):.6f}")
    print(f"std energy [MeV] = {np.std(energies_MeV):.6f}")
    print(f"Min energy [MeV] = {np.min(energies_MeV):.6f}")
    print(f"Max energy [MeV] = {np.max(energies_MeV):.6f}")

    fig, ax = plt.subplots(figsize = (8,5))
    ax.hist(energies_MeV, bins=120)
    ax.set_xlabel("neutron energy [MeV]")
    ax.set_ylabel("Counts")
    ax.set_title("Sampled DT neutron birth spectrum (Ballabio thermal model)(H-mode)")
    fig.tight_layout()
    fig.savefig(output_dir / "h_mode_ballabio_spectrum.png", dpi=200)

if __name__ == "__main__":
    main()