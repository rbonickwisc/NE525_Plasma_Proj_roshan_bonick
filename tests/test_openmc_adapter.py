from __future__ import annotations

import numpy as np
import openmc

from tokamak_source_model.case_builder import build_default_mesh, build_l_mode_model
from tokamak_source_model.openmc_adapter import build_openmc_independent_sources, summarize_openmc_source_particles, sample_openmc_source_particles


def test_sample_openmc_source_particles_has_expected_lengths():
    model = build_l_mode_model()
    mesh = build_default_mesh(num_a=120, num_alpha=180)
    rng = np.random.default_rng(123)

    particles = sample_openmc_source_particles(
        n_samples=500,
        model=model,
        mesh=mesh,
        rng=rng,
    )

    assert len(particles.x_cm) == 500
    assert len(particles.y_cm) == 500
    assert len(particles.z_cm) == 500
    assert len(particles.u_x) == 500
    assert len(particles.energy_eV) == 500
    assert len(particles.weight) == 500

def test_openmc_particle_summary_is_sane():
    model = build_l_mode_model()
    mesh = build_default_mesh(num_a=120, num_alpha=180)

    particles = sample_openmc_source_particles(
        n_samples=500,
        model=model,
        mesh=mesh,
    )

    summary = summarize_openmc_source_particles(particles)

    assert summary["n_particles"] == 500
    assert 1.39e7 < summary["mean_energy_eV"] < 1.43e7
    assert np.isclose(summary["mean_direction_norm"], 1.0)
    assert np.isclose(summary["weight_sum"], 1.0)

def test_build_openmc_independent_sources_returns_sources():
    model = build_l_mode_model()
    mesh = build_default_mesh(num_a=80, num_alpha=120)
    rng = np.random.default_rng(123)

    sources = build_openmc_independent_sources(
        n_samples=50,
        model=model,
        mesh=mesh,
        rng=rng,
    )

    assert len(sources) == 50
    assert all(isinstance(src, openmc.IndependentSource) for src in sources)