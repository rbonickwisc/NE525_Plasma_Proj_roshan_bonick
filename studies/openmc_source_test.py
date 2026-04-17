from __future__ import annotations

import numpy as np

from tokamak_source_model.case_builder import build_default_mesh, build_l_mode_model
from tokamak_source_model.openmc_adapter import build_openmc_independent_sources, sample_openmc_source_particles, summarize_openmc_source_particles

def main() -> None:
    model = build_l_mode_model()
    mesh = build_default_mesh()

    rng = np.random.default_rng(42)

    particles = sample_openmc_source_particles(
        n_samples=1000,
        model=model,
        mesh=mesh,
        rng=rng,
    )

    summary = summarize_openmc_source_particles(particles)

    print("OpenMC source test")
    print("-" * 25)
    print(f"n_particles          = {int(summary['n_particles'])}")
    print(f"x range [cm]         = [{summary['x_cm_min']:.6e}, {summary['x_cm_max']:.6e}]")
    print(f"z range [cm]         = [{summary['z_cm_min']:.6e}, {summary['z_cm_max']:.6e}]")
    print(f"mean energy [eV]     = {summary['mean_energy_eV']:.6e}")
    print(f"mean direction norm  = {summary['mean_direction_norm']:.12f}")
    print(f"weight sum           = {summary['weight_sum']:.12f}")

    sources = build_openmc_independent_sources(
        n_samples=100,
        model=model,
        mesh=mesh,
        rng=rng,
    )

    print(f"n_openmc_sources     = {len(sources)}")
    print(f"first source type    = {type(sources[0]).__name__}")

if __name__ == "__main__":
    main()