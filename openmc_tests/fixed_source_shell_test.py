from __future__ import annotations

from pathlib import Path
import numpy as np
import openmc

from tokamak_source_model.utils.case_builder import build_default_mesh, build_l_mode_model
from tokamak_source_model.utils.openmc_adapter import build_openmc_independent_sources

# Basic spherical material shell around source w/ outer vacuum boundary

def main() -> None:
    output_dir = Path("openmc_tests/output/openmc_shell")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_l_mode_model()
    mesh = build_default_mesh()
    rng = np.random.default_rng(42)

    # ---------------------------------
    # Materials
    # ---------------------------------
    iron = openmc.Material(name="iron")
    iron.set_density("g/cm3", 7.87)
    iron.add_element("Fe", 1.0)

    materials = openmc.Materials([iron])

    # ---------------------------------
    # Geometry
    # ---------------------------------
    inner_sphere = openmc.Sphere(r=260.0)
    outer_sphere = openmc.Sphere(r=400.0, boundary_type="vacuum")

    inner_vacuum_cell = openmc.Cell(
        name = "inner_vacuum",
        region = -inner_sphere,
    )

    shell_cell = openmc.Cell(
        name="iron shell",
        fill=iron,
        region=+inner_sphere & -outer_sphere,
    )

    universe = openmc.Universe(cells=[inner_vacuum_cell, shell_cell])
    geometry = openmc.Geometry(universe)

    # ---------------------------------
    # Settings
    # ---------------------------------
    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.batches = 10
    settings.particles = 1000

    sources = build_openmc_independent_sources(
        n_samples=1000,
        model=model,
        mesh=mesh,
        rng=np.random.default_rng(42)
    )
    settings.source = sources

    # ---------------------------------
    # Tallies
    # ---------------------------------

    shell_filter = openmc.CellFilter(shell_cell)
    outer_surface_filter = openmc.SurfaceFilter(outer_sphere)
    particle_filter = openmc.ParticleFilter(["neutron"])

    flux_tally = openmc.Tally(name="shell_flux")
    flux_tally.filters = [shell_filter, particle_filter]
    flux_tally.scores = ["flux"]

    heating_tally = openmc.Tally(name = "shell_heating")
    heating_tally.filters = [shell_filter, particle_filter]
    heating_tally.scores = ["heating"]

    leakage_tally = openmc.Tally(name="outer_surface_current")
    leakage_tally.filters = [outer_surface_filter, particle_filter]
    leakage_tally.scores = ["current"]

    tallies = openmc.Tallies([flux_tally, heating_tally, leakage_tally])

    # ---------------------------------
    # Model and run
    # ---------------------------------
    omodel = openmc.Model(
        geometry=geometry,
        materials=materials,
        settings=settings,
        tallies=tallies,
    )

    omodel.export_to_xml()
    statepoint_path = omodel.run(cwd=output_dir, threads = 1)

    print("Basic spherical iron shell OpenMC fixed source run complete")
    print("-" * 25)
    print(f"Statepoint file = {statepoint_path}")

    sp = openmc.StatePoint(statepoint_path)

    flux = sp.get_tally(name="shell_flux")
    heating = sp.get_tally(name="shell_heating")
    leakage = sp.get_tally(name="outer_surface_current")

    flux_mean = flux.mean.ravel()[0]
    flux_std_dev = flux.std_dev.ravel()[0]

    heating_mean = heating.mean.ravel()[0]
    heating_std_dev = heating.std_dev.ravel()[0]

    leakage_mean = leakage.mean.ravel()[0]
    leakage_std_dev = leakage.std_dev.ravel()[0]

    print(f"Shell flux mean = {flux_mean:.6e}")
    print(f"Shell flux std dev = {flux_std_dev:.6e}")
    print(f"Shell heating mean = {heating_mean:.6e}")
    print(f"Shell heating std dev = {heating_std_dev:.6e}")
    print(f"Outer surface current = {leakage_mean:.6e}")
    print(f"Outer surface current std = {leakage_std_dev:.6e}")


if __name__ == "__main__":
    main()