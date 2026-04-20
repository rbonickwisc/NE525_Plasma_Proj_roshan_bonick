from __future__ import annotations

from pathlib import Path
import numpy as np
import openmc

from tokamak_source_model.case_builder import build_default_mesh, build_l_mode_model
from tokamak_source_model.openmc_adapter import build_openmc_independent_sources, sample_openmc_source_particles

def main() -> None:
    output_dir = Path("openmc_tests/output/openmc_toroidal_shell")
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

    rho_cm = np.sqrt(particles.x_cm**2 + particles.y_cm**2)

    major_radius_cm = model.geometry.major_radius_m * 100.0

    radial_offset_cm = np.abs(rho_cm - major_radius_cm)
    vertical_offset_cm = np.abs(particles.z_cm)

    max_radial_offset_cm = float(np.max(radial_offset_cm))
    max_vertical_offset_cm = float(np.max(vertical_offset_cm))

    inner_minor_radial = max_radial_offset_cm + 20.0
    inner_minor_vertical = max_vertical_offset_cm + 20.0

    outer_minor_radial = inner_minor_radial + 40.0
    outer_minor_vertical = inner_minor_vertical + 40.0

    print("OpenMC toroidal shell test")
    print("-" * 25)
    #Check containment relative to inner torus
    inside_inner = (
        (particles.z_cm / inner_minor_vertical) ** 2 + ((rho_cm - major_radius_cm) / inner_minor_radial) ** 2
        <= 1.0
    )
    print(f"Number of sampled particles outside inner torus = {np.count_nonzero(~inside_inner)}")

    # ---------------------------------
    # Material
    # ---------------------------------
    iron = openmc.Material(name="iron")
    iron.set_density("g/cm3", 7.87)
    iron.add_element("Fe", 1.0)

    materials = openmc.Materials([iron])

    # ---------------------------------
    # Geometry
    # ---------------------------------
    inner_torus = openmc.ZTorus(
        x0=0.0,
        y0=0.0,
        z0=0.0,
        a=major_radius_cm,
        b=inner_minor_vertical,
        c=inner_minor_radial,
    )
    outer_torus = openmc.ZTorus(
        x0=0.0,
        y0=0.0,
        z0=0.0,
        a=major_radius_cm,
        b=outer_minor_vertical,
        c=outer_minor_radial,
        boundary_type="vacuum",
    )

    inner_vacuum_cell = openmc.Cell(
        name="inner vacuum torus",
        region=-inner_torus,
    )
    shell_cell = openmc.Cell(
        name="iron toroidal shell",
        fill=iron,
        region=+inner_torus & -outer_torus,
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
    outer_torus_filter = openmc.SurfaceFilter(outer_torus)

    flux_tally = openmc.Tally(name="shell_flux")
    flux_tally.filters = [shell_filter, particle_filter]
    flux_tally.scores = ["flux"]

    leakage_tally = openmc.Tally(name="outer_torus_current")
    leakage_tally.filters = [outer_torus_filter, particle_filter]
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

    print("\nBasic toroidal shell run complete")
    print("-" * 25)
    print(f"Statepoint file = {statepoint_path}")

    sp = openmc.StatePoint(statepoint_path)

    flux = sp.get_tally(name="shell_flux")
    leakage = sp.get_tally(name="outer_torus_current")

    flux_mean = flux.mean.ravel()[0]
    flux_std_dev = flux.std_dev.ravel()[0]

    leakage_mean = leakage.mean.ravel()[0]
    leakage_std_dev = leakage.std_dev.ravel()[0]

    print(f"Shell flux mean = {flux_mean:.6e}")
    print(f"Shell flux std dev = {flux_std_dev:.6e}")
    print(f"Outer torus current = {leakage_mean:.6e}")
    print(f"Outer torus current std = {leakage_std_dev:.6e}")


if __name__ == "__main__":
    main()