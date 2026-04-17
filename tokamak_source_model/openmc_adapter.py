from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import openmc

from .parameters import MeshParameters, SourceModelParameters
from .sampling import SourceSample, sample_source_particles

@dataclass
class OpenMCSourceParticles:
    """
    Container holding sampled source particles in OpenMC ready units

    Notes
    ------------------
    OpenMC uses centimeters for spatial coordinates
    Energies are in eV
    """
    x_cm: np.ndarray
    y_cm: np.ndarray
    z_cm: np.ndarray

    u_x: np.ndarray
    u_y: np.ndarray
    u_z: np.ndarray

    energy_eV: np.ndarray
    weight: np.ndarray

def sample_openmc_source_particles(
    n_samples: int,
    model: SourceModelParameters,
    mesh: MeshParameters,
    rng: np.random.Generator | None = None,
) -> OpenMCSourceParticles:
    """
    Sample source particles from the tokamak source model and convert them into OpenMC units

    Parameters
    ------------------
    n_samples
        Number of source particles to make
    model
        Source model parameters
    mesh
        mesh used by the source model
    
    Returns
    -----------
    OpenMCSourceParticles
        sampled particles with coordinates in cm
    """
    samples: SourceSample = sample_source_particles(
        n_samples=n_samples,
        model=model,
        mesh=mesh,
        rng=rng,
    )

    return OpenMCSourceParticles(
        x_cm=100.0 * samples.x_m,
        y_cm=100.0 * samples.y_m,
        z_cm=100.0 * samples.z_m,
        u_x=samples.u_x,
        u_y=samples.u_y,
        u_z=samples.u_z,
        energy_eV=samples.energy_eV,
        weight=samples.weight,
    )

def build_openmc_independent_sources(
    n_samples: int,
    model: SourceModelParameters,
    mesh: MeshParameters,
    rng: np.random.Generator | None = None,
) -> list[openmc.IndependentSource]:
    """
    Build a list of OpenMC independent source objects from sampled particles
    """
    particles = sample_openmc_source_particles(
        n_samples=n_samples,
        model=model,
        mesh=mesh,
        rng=rng,
    )

    sources: list[openmc.IndependentSource] = []

    for i in range(n_samples):
        source = openmc.IndependentSource()

        source.space = openmc.stats.Point(
            (
                float(particles.x_cm[i]),
                float(particles.y_cm[i]),
                float(particles.z_cm[i]),
            )
        )

        source.angle = openmc.stats.Monodirectional(
            (
                float(particles.u_x[i]),
                float(particles.u_y[i]),
                float(particles.u_z[i]),
            )
        )

        source.energy = openmc.stats.Discrete(
            [float(particles.energy_eV[i])],
            [1.0],
        )

        source.strength = float(particles.weight[i])
        sources.append(source)

    return sources

def summarize_openmc_source_particles(
    particles: OpenMCSourceParticles,
) -> dict[str, float]:
    """
    Return simple diagnostics for OpenMC ready source particles
    """
    direction_norms = np.sqrt(
        particles.u_x**2 + particles.u_y**2 + particles.u_z**2
    )

    return {
        "n_particles": float(len(particles.x_cm)),
        "x_cm_min": float(np.min(particles.x_cm)),
        "x_cm_max": float(np.max(particles.x_cm)),
        "z_cm_min": float(np.min(particles.z_cm)),
        "z_cm_max": float(np.max(particles.z_cm)),
        "mean_energy_eV": float(np.mean(particles.energy_eV)),
        "mean_direction_norm": float(np.mean(direction_norms)),
        "weight_sum": float(np.sum(particles.weight))
    }