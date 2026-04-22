from __future__ import annotations
import argparse
from pathlib import Path
import openmc
import numpy as np
from tokamak_source_model import tokamak_source


#Materials
#---------------------------#
def build_materials() -> openmc.Materials:
    mats = openmc.Materials()

    steel = openmc.Material(name="steel")
    steel.set_density("g/cm3", 7.8)
    steel.add_element("Fe", 0.88)
    steel.add_element("Cr", 0.12)
    mats.append(steel)

    flibe = openmc.Material(name="FLiBe")
    flibe.set_density("g/cm3", 1.95)
    flibe.add_nuclide("Li6", 0.075)
    flibe.add_nuclide("Li7", 0.925)
    flibe.add_element("Be", 1.0)
    flibe.add_element("F", 2.0)
    mats.append(flibe)

    return mats

#Geometry
#---------------------------------#

#Geometry values for torus
R0_CM = 200.0
PLASMA_R_CM = 20.0
FIRST_WALL_R_CM = 50.0
FLIBE_R_CM = 100.0
OUTER_SPHERE_R_CM = R0_CM + FLIBE_R_CM + 100.0

#Build geometry

def build_geometry() -> tuple[
    openmc.Geometry,
    openmc.Cell,
    openmc.Cell,
    openmc.Cell,
    openmc.Cell,
]:
    ###Surfaces###
    plasma_torus = openmc.ZTorus(x0=0.0, y0=0.0, z0=0.0, a=R0_CM, b=PLASMA_R_CM, c=PLASMA_R_CM)
    first_wall_torus = openmc.ZTorus(x0=0.0, y0=0.0, z0=0.0, a=R0_CM, b=FIRST_WALL_R_CM, c=FIRST_WALL_R_CM)
    flibe_torus = openmc.ZTorus(x0=0.0, y0=0.0, z0=0.0, a=R0_CM, b=FLIBE_R_CM, c=FLIBE_R_CM)

    outer_sphere = openmc.Sphere(r=OUTER_SPHERE_R_CM, boundary_type="vacuum")

    ###Regions###
    plasma_region = -plasma_torus
    first_wall_region = +plasma_torus & -first_wall_torus
    flibe_region = +first_wall_torus & -flibe_torus
    outside_region = +flibe_torus & -outer_sphere

    ###Cells###
    plasma_cell=openmc.Cell(name="Plasma", region=plasma_region, fill=None)
    first_wall_cell=openmc.Cell(name="First Wall", region=first_wall_region)
    flibe_cell = openmc.Cell(name="FLiBe Blanket", region=flibe_region)
    outside_cell = openmc.Cell(name="Outside Void", region=outside_region, fill=None)

    root_universe = openmc.Universe(cells=[plasma_cell, first_wall_cell, flibe_cell, outside_cell])
    geometry = openmc.Geometry(root_universe)

    return geometry, plasma_cell, first_wall_cell, flibe_cell, outside_cell

def assign_materials(
    mats: openmc.Materials,
    first_wall_cell: openmc.Cell,
    flibe_cell: openmc.Cell,
) -> None:
    steel = next(mat for mat in mats if mat.name == "steel")
    flibe = next(mat for mat in mats if mat.name == "FLiBe")

    first_wall_cell.fill = steel
    flibe_cell.fill = flibe

#Settings
#--------------------------------#

def build_tokamak_sources(
    mode: str,
    n_samples: int,
    seed: int,
) -> list[openmc.IndependentSource]:
    """
    Build L / H / A tokamak sources using a common torus geometry

    Hold geometry variables constant over each confinement mode:
        major radius = 2.0 m
        minor radius = 0.18 m
        elongation = 1.0
        triangularity = 0.0
        shafranov shift = 0.0
    
    Doing this isolates the confinment mode comparison to profile differences
    rather than changing the source geometry
    """
    mode = mode.upper()

    common_kwargs = dict(
        mode=mode,
        major_radius_m = 2.0,
        minor_radius_m=0.18,
        elongation=1.0,
        triangularity=0.0,
        shafranov_shift_m=0.0,
        deuterium_fraction=0.5,
        tritium_fraction=0.5,
        n_samples=n_samples,
        num_a=120,
        num_alpha=180,
        num_R=160,
        num_Z=160,
        rng=np.random.default_rng(seed),
    )

    if mode == "L":
        return tokamak_source(
            ion_density_center_m3=2.0e20,
            ion_temp_center_keV=15.0,
            alpha_n=0.5,
            alpha_T=1.0,
            **common_kwargs,
        )

    if mode == "H":
        return tokamak_source(
            ion_density_center_m3=2.0e20,
            ion_temp_center_keV=20.0,
            alpha_n=1.0,
            alpha_T=4.0,
            pedestal_radius_m=0.8 * 0.18,
            ion_density_pedestal_m3=1.8e20,
            ion_density_separatrix_m3=3.0e19,
            ion_temp_pedestal_keV=4.0,
            ion_temp_separatrix_keV=0.1,
            beta_T=6.0,
            **common_kwargs,
        )

    if mode == "A":
        return tokamak_source(
            ion_density_center_m3=1.09e20,
            ion_temp_center_keV=45.9,
            alpha_n=1.0,
            alpha_T=8.06,
            pedestal_radius_m=0.8 * 0.18,
            ion_density_pedestal_m3=1.09e20,
            ion_density_separatrix_m3=3.0e19,
            ion_temp_pedestal_keV=6.09,
            ion_temp_separatrix_keV=0.1,
            beta_T=6.0,
            **common_kwargs,
        )

#Tallies
#----------------------------#

def build_tallies(
    first_wall_cell: openmc.Cell,
    flibe_cell: openmc.Cell,
) -> openmc.Tallies:
    tallies = openmc.Tallies()

    fw_filter = openmc.CellFilter(first_wall_cell)
    flibe_filter = openmc.CellFilter(flibe_cell)
    energy_filter = openmc.EnergyFilter([0.0, 0.625, 1e5, 20e6])

    #tritium prod in flibe blanket
    tbr = openmc.Tally(name="Tritium production in FLiBe")
    tbr.filters = [flibe_filter]
    tbr.scores = ["H3-production"]
    tallies.append(tbr)

    #nuetron flux in flibe
    flibe_flux = openmc.Tally(name="Flux in FLiBe (E-binned)")
    flibe_flux.filters = [flibe_filter, energy_filter]
    flibe_flux.scores = ["flux"]
    tallies.append(flibe_flux)

    #neutron flux in steel first wall
    fw_flux = openmc.Tally(name="Flux in first wall (E-binned)")
    fw_flux.filters = [fw_filter, energy_filter]
    fw_flux.scores = ["flux"]
    tallies.append(fw_flux)

    ###Mesh tallies###
    nx = 300
    ny = 300
    nz = 160

    #mesh bounds
    xmin = -(R0_CM + FLIBE_R_CM + 100)
    xmax = +(R0_CM + FLIBE_R_CM + 100)
    ymin = -(R0_CM + FLIBE_R_CM + 100)
    ymax = +(R0_CM + FLIBE_R_CM + 100)
    zmin = -FLIBE_R_CM
    zmax = +FLIBE_R_CM

    reg_mesh = openmc.RegularMesh()
    reg_mesh.dimension = [nx, ny, nz]
    reg_mesh.lower_left = [xmin, ymin, zmin]
    reg_mesh.upper_right = [xmax, ymax, zmax]

    mesh_filter = openmc.MeshFilter(reg_mesh)

    #flux map
    flux_map = openmc.Tally(name="Flux x-y-z map")
    flux_map.filters = [mesh_filter]
    flux_map.scores = ["flux"]
    tallies.append(flux_map)

    #tritium prod map(in flibe)
    t_map = openmc.Tally(name="Tritium production x-y-z map (FLiBe only)")
    t_map.filters = [mesh_filter, flibe_filter]
    t_map.scores = ["H3-production"]
    tallies.append(t_map)

    #fast neutron flux mesh tally
    fast_energy_filter = openmc.EnergyFilter([1e5, 2e7])

    fast_flux_map = openmc.Tally(name="Fast flux x-y-z map")
    fast_flux_map.filters = [mesh_filter, fast_energy_filter]
    fast_flux_map.scores = ["flux"]
    tallies.append(fast_flux_map)

    return tallies

#Build and export
#--------------------#

def build_model(
    mode: str,
    n_samples: int,
    particles: int,
    batches: int,
    seed: int,
) -> openmc.Model:
    mats = build_materials()
    geometry, plasma_cell, first_wall_cell, flibe_cell, outside_cell = build_geometry()
    assign_materials(mats, first_wall_cell, flibe_cell)

    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.batches = batches
    settings.inactive = 0
    settings.particles = particles
    settings.max_lost_particles = 1000
    settings.rel_max_lost_particles = 0.01

    settings.source = build_tokamak_sources(
        mode=mode,
        n_samples=n_samples,
        seed=seed,
    )

    tallies = build_tallies(first_wall_cell, flibe_cell)

    return openmc.Model(
        geometry=geometry,
        materials=mats,
        settings=settings,
        tallies=tallies,
    )

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create torus openmc xml files for L / H / A tokamak confinement mode"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="L",
        choices=["L", "H", "A"],
        help="Tokamak confinement mode to use",
    )

    args = parser.parse_args()
    mode = args.mode.upper()
    seed_map = {"L":42, "H":43, "A": 44}
    seed = seed_map[mode]
    output_dir = Path(f"openmc_tokamak_mode_comparison/output/torus_mode_{mode.lower()}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(
        mode=mode,
        n_samples=2000,
        particles=100000,
        batches=25,
        seed=seed,
    )

    model.export_to_xml(directory=output_dir)

    print(f"expotred XML files fro mode={mode} to {output_dir}")

if __name__ == "__main__":
    main()