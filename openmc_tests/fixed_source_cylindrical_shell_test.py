from __future__ import annotations

from pathlib import Path
import numpy as np
import openmc

from tokamak_source_model.case_builder import build_default_mesh, build_l_mode_model
from tokamak_source_model.openmc_adapter import build_openmc_independent_sources, sample_openmc_source_particles

def main() -> None:
    output_dir = Path("openmc_tests/output/openmc_cylindrical_shell")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_l_mode_model()
    mesh = build_default_mesh()
    rng = np.random.default_rng(42)

    particles = sample_openmc_source_particles(
        n_samples=5000,
        model=model,
        mesh=mesh,
        rng=rng,
    )

    r_xy_cm = np.sqrt(particles.x_cm**2 + particles.y_cm**2)
    z_abs_cm = np.abs(particles.z_cm)

    max_r_xy_cm = float(np.max(r_xy_cm))
    max_z_cm = float(np.max(z_abs_cm))

    inner_radius_cm = max_r_xy_cm +20.0
    outer_radius_cm = inner_radius_cm + 100.0

    inner_half_height_cm = max_z_cm +20.0
    outer_half_height_cm = inner_half_height_cm +100.0

    print("OpenMC cylindrical shell test")
    print("-" * 25)
    
    # ---------------------------------
    # Material
    # ---------------------------------
    carbon = openmc.Material(name="carbon")
    carbon.set_density("g/cm3", 1.8)
    carbon.add_element("C", 1.0)

    materials = openmc.Materials([carbon])

    # ---------------------------------
    # Geometry
    # ---------------------------------
    inner_cyl = openmc.ZCylinder(r=inner_radius_cm)
    outer_cyl = openmc.ZCylinder(r=outer_radius_cm, boundary_type="vacuum")

    z_top_inner = openmc.ZPlane(z0=inner_half_height_cm)
    z_bot_inner = openmc.ZPlane(z0=-inner_half_height_cm)

    z_top_out = openmc.ZPlane(z0=outer_half_height_cm, boundary_type="vacuum")
    z_bot_out = openmc.ZPlane(z0=-outer_half_height_cm, boundary_type="vacuum")

    inner_region = -inner_cyl & -z_top_inner & +z_bot_inner
    outer_region = -outer_cyl & -z_top_out & +z_bot_out
    shell_region = outer_region & ~inner_region

    inner_vacuum_cell = openmc.Cell(
        name="inner vacuum cylinder",
        region=inner_region,
    )
    shell_cell = openmc.Cell(
        name="carbon cylindrical shell",
        fill=carbon,
        region=shell_region,
    )

    universe = openmc.Universe(cells = [inner_vacuum_cell, shell_cell])
    geometry = openmc.Geometry(universe)

    # ---------------------------------
    # Settings 
    # ---------------------------------
    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.batches = 15
    settings.particles = 1000

    sources = build_openmc_independent_sources(
        n_samples=100,
        model=model,
        mesh=mesh,
        rng=np.random.default_rng(42),
    )
    settings.source = sources

    # ---------------------------------
    # Tallies
    # ---------------------------------
    shell_filter = openmc.CellFilter(shell_cell)
    particle_filter = openmc.ParticleFilter(["neutron"])

    outer_cyl_filter = openmc.SurfaceFilter(outer_cyl)

    flux_tally = openmc.Tally(name="shell_flux")
    flux_tally.filters = [shell_filter, particle_filter]
    flux_tally.scores = ["flux"]

    leakage_tally = openmc.Tally(name="outer_cyl_current")
    leakage_tally.filters = [outer_cyl_filter, particle_filter]
    leakage_tally.scores = ["current"]

    tallies = openmc.Tallies([flux_tally, leakage_tally])

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

    print("Basic cylindrical shell run complete")
    print("-" * 25)
    print(f"Statepoint file = {statepoint_path}")

    sp = openmc.StatePoint(statepoint_path)

    flux = sp.get_tally(name="shell_flux")
    leakage = sp.get_tally(name="outer_cyl_current")

    flux_mean = flux.mean.ravel()[0]
    flux_std_dev = flux.std_dev.ravel()[0]

    leakage_mean = leakage.mean.ravel()[0]
    leakage_std_dev = leakage.std_dev.ravel()[0]

    print(f"Shell flux mean = {flux_mean:.6e}")
    print(f"Shell flux std dev = {flux_std_dev:.6e}")
    print(f"Outer cyl current = {leakage_mean:.6e}")
    print(f"Outer cyl current std = {leakage_std_dev:.6e}")


if __name__ == "__main__":
    main()