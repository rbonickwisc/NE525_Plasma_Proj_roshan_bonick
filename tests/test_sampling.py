from __future__ import annotations

import numpy as np

from tokamak_source_model.parameters import FuelParameters, GeometryParameters, MeshParameters, ProfileParameters, SourceModelParameters
from tokamak_source_model.sampling import sample_source_particles, sample_birth_energies_eV, sample_birth_positions, sample_isotropic_directions

def make_l_mode_model() -> SourceModelParameters:
    geometry = GeometryParameters(
        major_radius_m=2.0,
        minor_radius_m=0.5,
        elongation=1.7,
        triangularity=0.33,
        shafranov_shift_m=0.1
    )
    profile = ProfileParameters(
        mode="l_mode",
        ion_density_center_m3=2.0e20,
        ion_temp_center_keV=15.0,
        alpha_n=0.5,
        alpha_T=1.0,
    )
    fuel = FuelParameters(
        deuterium_fraction=0.5,
        tritium_fraction=0.5,
    )
    
    return SourceModelParameters(
        geometry=geometry,
        profile=profile,
        fuel=fuel,
    )

def test_isotropic_direction_vectors_are_unit_norm():
    rng = np.random.default_rng(123)
    u_x, u_y, u_z = sample_isotropic_directions(5000, rng=rng)

    norms = np.sqrt(u_x**2 + u_y**2 + u_z**2)
    assert np.allclose(norms, 1.0)

def test_birth_energies_are_monoenergetic_in_first_pass():
    energies = sample_birth_energies_eV(1000)
    assert np.allclose(energies, 14.1e6)

def test_sample_source_particles_has_consistent_lengths():
    model = make_l_mode_model()
    mesh = MeshParameters(num_a=120, num_alpha=180)
    rng = np.random.default_rng(123)

    samples = sample_source_particles(
        n_samples=2000,
        model=model,
        mesh=mesh,
        rng=rng,
    )

    n = 2000
    assert len(samples.x_m) == n
    assert len(samples.y_m) == n
    assert len(samples.z_m) == n
    assert len(samples.u_x) == n
    assert len(samples.u_y) == n
    assert len(samples.u_z) == n
    assert len(samples.energy_eV) == n
    assert len(samples.weight) == n

def test_sample_weights_sum_to_one():
    model = make_l_mode_model()
    mesh = MeshParameters(num_a=120, num_alpha=180)
    rng = np.random.default_rng(123)

    samples = sample_source_particles(
        n_samples=2000,
        model=model,
        mesh=mesh,
        rng=rng,
    )

    assert np.isclose(np.sum(samples.weight), 1.0)