from __future__ import annotations

import numpy as np

from tokamak_source_model.case_builder import build_default_mesh, build_l_mode_model
from tokamak_source_model.sampling import sample_source_particles, sample_birth_energies_eV, sample_birth_positions, sample_isotropic_directions

def test_isotropic_direction_vectors_are_unit_norm():
    rng = np.random.default_rng(123)
    u_x, u_y, u_z = sample_isotropic_directions(5000, rng=rng)

    norms = np.sqrt(u_x**2 + u_y**2 + u_z**2)
    assert np.allclose(norms, 1.0)

def test_isotropic_direction_vectors_are_unit_norm_with_default_rng():
    u_x, u_y, u_z = sample_isotropic_directions(5000)

    norms = np.sqrt(u_x**2 + u_y**2 + u_z**2)
    assert np.allclose(norms, 1.0)

def test_birth_energies_are_monoenergetic_in_first_pass():
    energies = sample_birth_energies_eV(1000)
    assert np.allclose(energies, 14.1e6)

def test_sample_birth_positions_returns_correct_length():
    model = build_l_mode_model()
    mesh = build_default_mesh(num_a=120, num_alpha=180)
    rng = np.random.default_rng(123)

    x_m, y_m, z_m = sample_birth_positions(
        n_samples=2000,
        model=model,
        mesh=mesh,
        rng=rng
    )

    assert len(x_m) == 2000
    assert len(y_m) == 2000
    assert len(z_m) == 2000

def test_sample_birth_positions_are_finite():
    model = build_l_mode_model()
    mesh = build_default_mesh(num_a=120, num_alpha=180)
    rng = np.random.default_rng(123)

    x_m, y_m, z_m = sample_birth_positions(
        n_samples=2000,
        model=model,
        mesh=mesh,
        rng=rng
    )

    assert np.all(np.isfinite(x_m))
    assert np.all(np.isfinite(y_m))
    assert np.all(np.isfinite(z_m))

def test_sample_source_particles_has_consistent_lengths():
    model = build_l_mode_model()
    mesh = build_default_mesh(num_a=120, num_alpha=180)
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
    model = build_l_mode_model()
    mesh = build_default_mesh(num_a=120, num_alpha=180)
    rng = np.random.default_rng(123)

    samples = sample_source_particles(
        n_samples=2000,
        model=model,
        mesh=mesh,
        rng=rng,
    )

    assert np.isclose(np.sum(samples.weight), 1.0)