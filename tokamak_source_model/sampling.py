from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .normalization import build_source_probability_map
from .parameters import MeshParameters, SourceModelParameters

@dataclass
class SourceSample:
    """
    Container for sampled neutron birth data
    """

    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray

    u_x: np.ndarray
    u_y: np.ndarray
    u_z: np.ndarray

    energy_eV: np.ndarray
    weight: np.ndarray

def sample_birth_positions(
    n_samples: int,
    model: SourceModelParameters,
    mesh: MeshParameters,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample neutron birth positions in 3D cartesian coordinates
    
    Method:
    1. Build normalized source probability map on the (a, alpha) mesh
    2. Sample one poloidal source point (R, Z) from that map
    3. Sample toroidal angle phi uniformly in [0, 2*pi]
    4. Convert cylindrical (R, phi, Z) to cartestian (x, y, z)
    """
    
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    
    if rng is None:
        rng = np.random.default_rng()

    _, _, R_m, Z_m, probability_map = build_source_probability_map(model, mesh)

    flat_prob = probability_map.ravel()
    flat_prob = flat_prob / np.sum(flat_prob)

    flat_indices = rng.choice(
        flat_prob.size,
        size=n_samples,
        replace=True,
        p=flat_prob,
    )

    sampled_R_m = R_m.ravel()[flat_indices]
    sampled_Z_m = Z_m.ravel()[flat_indices]

    phi_rad = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)

    x_m = sampled_R_m * np.cos(phi_rad)
    y_m = sampled_R_m * np.sin(phi_rad)
    z_m = sampled_Z_m.copy()

    return x_m, y_m, z_m

def sample_isotropic_directions(
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample isotropic 3D unit vectors
    """
    
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    
    if rng is None:
        rng = np.random.default_rng

    mu = rng.uniform(-1.0, 1.0, size=n_samples)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)

    sin_theta = np.sqrt(1.0 - mu**2)

    u_x = sin_theta * np.cos(phi)
    u_y = sin_theta * np.sin(phi)
    u_z = mu

    return u_x, u_y, u_z

def sample_birth_energies_eV(
        n_samples: int,
        mean_energy_eV: float = 14.1e6,
        rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Sample neutron birth energies

    First-pass: monoenergetic 14.1 MeV DT neutrons
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    
    return np.full(n_samples, mean_energy_eV, dtype=float)

def sample_source_particles(
        n_samples: int,
        model: SourceModelParameters,
        mesh: MeshParameters,
        rng: np.random.Generator | None = None,
) -> SourceSample:
    """
    Wrapper returning sampled source particles
    """
    if rng is None:
        rng = np.random.default_rng()

    x_m, y_m, z_m = sample_birth_positions(
        n_samples=n_samples,
        model=model,
        mesh=mesh,
        rng=rng,
    )

    u_x, u_y, u_z = sample_isotropic_directions(
        n_samples=n_samples,
        rng=rng,
    )

    energy_eV = sample_birth_energies_eV(
        n_samples=n_samples,
        rng=rng,
    )

    weight = np.full(n_samples, 1.0 / n_samples, dtype=float)

    return SourceSample(
        x_m=x_m,
        y_m=y_m,
        z_m=z_m,
        u_x=u_x,
        u_y=u_y,
        u_z=u_z,
        energy_eV=energy_eV,
        weight=weight
    )