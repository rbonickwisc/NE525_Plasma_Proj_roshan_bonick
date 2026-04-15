from __future__ import annotations

import numpy as np

from .geometry import surface_to_rz
from .parameters import MeshParameters, SourceModelParameters
from .source_density import evaluate_profiles

def build_a_alpha_mesh(
    model: SourceModelParameters,
    mesh: MeshParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build 2D a-alpha mesh grids

    Returns
    -----------------
    a_mesh_m: np.ndarray
        2D array of a values [m]
    alpha_mesh_rad: np.ndarray
        2D array of alpha values [rad]
    """
    a_grid_m = np.linspace(
        mesh.a_grid_min_m,
        model.geometry.minor_radius_m,
        mesh.num_a,
    )
    alpha_grid_rad = np.linspace(
        0.0,
        2.0*np.pi,
        mesh.num_alpha,
        endpoint=False
    )

    a_mesh_m, alpha_mesh_rad = np.meshgrid(
        a_grid_m,
        alpha_grid_rad,
        indexing="ij"
    )

    return a_mesh_m, alpha_mesh_rad

def build_a_alpha_cell_centers_and_edges(
        model: SourceModelParameters,
        mesh: MeshParameters,
) ->tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build 1D arrays of a/alpha cell centers and edges

    Returns
    ------------------
    a_centers_m
    alpha_centers_rad
    a_edges_m
    alpha_edges_rad
    """
    a_centers_m = np.linspace(
        mesh.a_grid_min_m,
        model.geometry.minor_radius_m,
        mesh.num_a,
    )
    alpha_centers_rad = np.linspace(
        0.0,
        2.0 * np.pi,
        mesh.num_alpha,
        endpoint=False,
    )

    if len(a_centers_m) < 2 or len(alpha_centers_rad) < 2:
        raise ValueError("Mesh must have at least 2 opints in both a and alpha")
    da = a_centers_m[1] - a_centers_m[0]
    dalpha = alpha_centers_rad[1] - alpha_centers_rad[0]

    a_edges_m = np.empty(len(a_centers_m) + 1, dtype=float)
    a_edges_m[1:-1] = 0.5 * (a_centers_m[:-1] + a_centers_m[1:])
    a_edges_m[0] = a_centers_m[0] - 0.5 * da
    a_edges_m[-1] = a_centers_m[-1] + 0.5 * da

    a_edges_m[0] = max(a_edges_m[0], mesh.a_grid_min_m)
    a_edges_m[-1] = min(a_edges_m[-1], model.geometry.minor_radius_m)

    alpha_edges_rad = np.empty(len(alpha_centers_rad) + 1, dtype=float)
    alpha_edges_rad[:-1] = alpha_centers_rad - 0.5 * dalpha
    alpha_edges_rad[-1] = alpha_centers_rad[-1] + 0.5 * dalpha

    alpha_edges_rad[0] = 0.0
    alpha_edges_rad[-1] = 2.0 * np.pi

    return a_centers_m, alpha_centers_rad, a_edges_m, alpha_edges_rad

def poloidal_area_element_m2(
        model: SourceModelParameters,
        mesh: MeshParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute differential poloidal area element on the (a, alpha) mesh

    Area element is computed numerically from the Jacobian:
        dA_pol = |∂(R,Z) / ∂(a,alpha)| da dalpha

    Returns
    ----------------
    a_mesh_m
    alpha_mesh_rad
    R_m
    Z_m
    dA_pol_m2
    """
    a_mesh_m, alpha_mesh_rad = build_a_alpha_mesh(model, mesh)
    R_m, Z_m = surface_to_rz(a_mesh_m, alpha_mesh_rad, model.geometry)

    a_grid_m = a_mesh_m[:, 0]
    alpha_grid_rad = alpha_mesh_rad[0, :]

    dR_da, dR_dalpha = np.gradient(R_m, a_grid_m, alpha_grid_rad, edge_order=2)
    dZ_da, dZ_dalpha = np.gradient(Z_m, a_grid_m, alpha_grid_rad, edge_order=2)

    jacobian = np.abs(dR_da * dZ_dalpha - dR_dalpha * dZ_da)
    dalpha = alpha_grid_rad[1] - alpha_grid_rad[0]
    da = a_grid_m[1] - a_grid_m[0]
    dA_pol_m2 = jacobian * da * dalpha

    return a_mesh_m, alpha_mesh_rad, R_m, Z_m, dA_pol_m2

def toroidal_volume_element_m3(
    model: SourceModelParameters,
    mesh: MeshParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the differential toroidal volume element:

        dV = 2*pi*R * dA_pol

    Returns
    -----------
    a_mesh_m
    alpha_mesh_rad
    R_m
    Z_m
    dA_pol_m2
    dV_m3
    """

    a_mesh_m, alpah_mesh_rad, R_m, Z_m, dA_pol_m2 = poloidal_area_element_m2(model, mesh)

    dV_m3 = 2.0 * np.pi * R_m * dA_pol_m2

    return a_mesh_m, alpah_mesh_rad, R_m, Z_m, dA_pol_m2, dV_m3

def estimate_total_neutron_rate_n_per_s(
    model: SourceModelParameters,
    mesh: MeshParameters,
) -> float:
    """
    Estimate total neutron production rate by integrating the local
    source density over the tokamak volume

    Returns
    -----------
    float
        Total neutron production rate [n/s]
    """
    a_mesh_m, _, _, _, _, dV_m3 = toroidal_volume_element_m3(model, mesh)

    a_grid_m = a_mesh_m[:, 0]
    evaluation = evaluate_profiles(a_grid_m, model)
    source_1d = evaluation.source_density_n_per_m3_per_s

    source_2d = np.repeat(source_1d[:, np.newaxis], dV_m3.shape[1], axis=1)

    total_rate_n_per_s = np.sum(source_2d * dV_m3)

    return float(total_rate_n_per_s)

def estimate_total_plasma_volume_m3(
    model: SourceModelParameters,
    mesh: MeshParameters,
) -> float:
    """
    Estimate plasma volume by integrating the toroidal volume element

    Returns
    -----------------
    float
        Plasma volume [m^3]
    """
    _, _, _, _, _, dV_m3 = toroidal_volume_element_m3(model, mesh)
    return float(np.sum(dV_m3))

def build_source_probability_map(
    model: SourceModelParameters,
    mesh: MeshParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build normalized source probability map over the (a, alpha) mesh

    Probability weights are proportional to:
        S(a) * dV

    Returns
    ------------
    a_mesh_m
    alpha_mesh_rad
    R_m
    Z_m
    probability_map
    """
    a_mesh_m, alpha_mesh_rad, R_m, Z_m, _, dV_m3 = toroidal_volume_element_m3(model, mesh)

    a_grid_m = a_mesh_m[:, 0]
    evaluation = evaluate_profiles(a_grid_m, model)
    source_1d = evaluation.source_density_n_per_m3_per_s
    source_2d = np.repeat(source_1d[:, np.newaxis], dV_m3.shape[1], axis=1)

    unnormalized_weights = source_2d * dV_m3
    total_weight = np.sum(unnormalized_weights)

    if total_weight <= 0.0:
        raise ValueError("Total source weight is non-positive, can't normalize")
    
    probability_map = unnormalized_weights / total_weight

    return a_mesh_m, alpha_mesh_rad, R_m, Z_m, probability_map

def build_source_cell_probability_map(
        model: SourceModelParameters,
        mesh: MeshParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a normalized probability map over source cells, along
    with the 1D cell edge arrays for a and alpha

    Returns
    ------------------
    a_edges_m
    alpha_edges_rad
    cell_probability_map

    Notes
    ----------
    This uses center based probability map as a first-order estimate
    of cell probabilites
    """
    _, _, _, _, probability_map = build_source_probability_map(model, mesh)
    _, _, a_edges_m, alpha_edges_rad = build_a_alpha_cell_centers_and_edges(model, mesh)

    cell_probability_map = probability_map / np.sum(probability_map)

    return a_edges_m, alpha_edges_rad, cell_probability_map