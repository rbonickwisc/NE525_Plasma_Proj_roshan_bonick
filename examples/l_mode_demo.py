from __future__ import annotations
from pathlib import Path
import numpy as np
import openmc

from tokamak_source_model import tokamak_source

sources = tokamak_source(
    mode="L",
    major_radius_m=2.0,
    minor_radius_m=0.5,
    elongation=1.7,
    triangularity=0.33,
    shafranov_shift_m=0.1,
    ion_density_center_m3=2.0e20,
    ion_temp_center_keV=15.0,
    alpha_n=0.5,
    alpha_T=1.0,
    deuterium_fraction=0.5,
    tritium_fraction=0.5,
    n_samples=1000,
    num_a=120,
    num_alpha=180,
    num_R=120,
    num_Z=120,
    rng=np.random.default_rng(42),
)

sphere_radius_cm = 300.0
boundary = openmc.Sphere(r=sphere_radius_cm, boundary_type="vacuum")
cell = openmc.Cell(region=-boundary)
geometry = openmc.Geometry([cell])

materials = openmc.Materials([])

settings = openmc.Settings()
settings.run_mode = "fixed source"
settings.batches = 10
settings.particles = 1000
settings.source = sources

model = openmc.Model(
    geometry=geometry,
    materials=materials,
    settings=settings,
)

model.export_to_xml()