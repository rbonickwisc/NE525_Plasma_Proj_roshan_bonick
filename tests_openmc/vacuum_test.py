from __future__ import annotations

from pathlib import Path
import numpy as np
import openmc

from tokamak_source_model.utils.case_builder import build_default_mesh, build_l_mode_model
from tokamak_source_model.utils.openmc_adapter import build_openmc_independent_sources

def main() -> None:
    output_dir = Path("openmc_tests/output/openmc_vacuum")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_l_mode_model()
    mesh = build_default_mesh()
    rng = np.random.default_rng(42)

    sources = build_openmc_independent_sources(
        n_samples=100,
        model=model,
        mesh=mesh,
        rng=rng,
    )

    outer_sphere = openmc.Sphere(r=400.0, boundary_type="vacuum")

    cell = openmc.Cell(region = -outer_sphere)
    universe = openmc.Universe(cells=[cell])
    geometry = openmc.Geometry(universe)

    materials = openmc.Materials([])

    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.batches = 10
    settings.particles = 1000
    settings.source = sources

    surface_filter = openmc.SurfaceFilter(outer_sphere)
    particle_filter = openmc.ParticleFilter(["neutron"])

    current_tally = openmc.Tally(name="surface_current")
    current_tally.filters = [surface_filter, particle_filter]
    current_tally.scores = ["current"]

    tallies = openmc.Tallies([current_tally])

    omodel = openmc.Model(
        geometry=geometry,
        materials=materials,
        settings=settings,
        tallies=tallies,
    )

    omodel.export_to_xml()
    statepoint_path = omodel.run(cwd=output_dir)

    print("vacuum sphere OpenMC fixed source run complete")
    print("-" * 25)
    print(f"Statepoint file = {statepoint_path}")

    sp = openmc.StatePoint(statepoint_path)
    tally = sp.get_tally(name="surface_current")

    mean = tally.mean.ravel()
    std_dev = tally.std_dev.ravel()

    print(f"Surface current mean = {mean[0]:.6e}")
    print(f"Surface current std dev = {std_dev[0]:.6e}")

if __name__ == "__main__":
    main()