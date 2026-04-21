from __future__ import annotations
from pathlib import Path
import numpy as np
import openmc

from tokamak_source_model import tokamak_source


sources = tokamak_source(
    mode="H",
    major_radius_m=9.06,
    minor_radius_m=2.92258,
    elongation=1.557,
    triangularity=0.270,
    shafranov_shift_m=0.44789,
    pedestal_radius_m=0.8 * 2.92258,
    ion_density_center_m3=1.09e20,
    ion_density_pedestal_m3=1.09e20,
    ion_density_separatrix_m3=3.0e19,
    ion_temp_center_keV=45.9,
    ion_temp_pedestal_keV=6.09,
    ion_temp_separatrix_keV=0.1,
    alpha_n=1.0,
    alpha_T=8.06,
    beta_T=6.0,
    deuterium_fraction=0.5,
    tritium_fraction=0.5,
    n_samples=1000,
    num_a=120,
    num_alpha=180,
    num_R=120,
    num_Z=120,
    rng=np.random.default_rng(43),
)

sphere_radius_cm = 1300.0
boundary = openmc.Sphere(r=sphere_radius_cm, boundary_type="vacuum")
cell = openmc.Cell(region=-boundary)
geometry = openmc.Geometry([cell])

materials = openmc.Materials([])

cell_filter = openmc.CellFilter(cell)
particle_filter = openmc.ParticleFilter(["neutron"])
flux_tally = openmc.Tally(name="shell_flux")
flux_tally.filters = [cell_filter, particle_filter]
flux_tally.scores = ["flux"]
tallies = openmc.Tallies([flux_tally])

settings = openmc.Settings()
settings.run_mode = "fixed source"
settings.batches = 10
settings.particles = 1000
settings.source = sources

model = openmc.Model(
    geometry=geometry,
    materials=materials,
    settings=settings,
    tallies=tallies,
)

model.export_to_xml()