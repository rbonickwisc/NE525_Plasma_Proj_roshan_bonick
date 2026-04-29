from __future__ import annotations

import numpy as np

from tokamak_source_model.utils.parameters import SourceModelParameters, MeshParameters
from tokamak_source_model.utils.normalization import toroidal_volume_element_m3
from tokamak_source_model.utils.source_density import evaluate_profiles


def source_weighted_mean_radius(
    model: SourceModelParameters,
    mesh: MeshParameters,
) -> float:
    """
    Compute the source-weighted mean normalized radius

        <rho>_S = (∫ rho S dV) / (∫ S dV)

    where rho = a / A.

    Returns
    -------
    Source-weighted mean normalized radius.
    """
    a_mesh_m, _, _, _, _, dV_m3 = toroidal_volume_element_m3(model, mesh)

    a_grid_m = a_mesh_m[:, 0]
    rho_grid = a_grid_m / model.geometry.minor_radius_m

    evaluation = evaluate_profiles(a_grid_m, model)
    source_1d = evaluation.source_density_n_per_m3_per_s

    dV_shell_m3 = np.sum(dV_m3, axis=1)

    numerator = np.sum(rho_grid * source_1d * dV_shell_m3)
    denominator = np.sum(source_1d * dV_shell_m3)

    if denominator <= 0.0:
        raise ValueError("Total source weight is negative")

    return float(numerator / denominator)


def source_weighted_rms_radius(
    model: SourceModelParameters,
    mesh: MeshParameters,
) -> float:
    """
    Optional companion metric:
        sqrt( (∫ rho^2 S dV) / (∫ S dV) )
    """
    a_mesh_m, _, _, _, _, dV_m3 = toroidal_volume_element_m3(model, mesh)

    a_grid_m = a_mesh_m[:, 0]
    rho_grid = a_grid_m / model.geometry.minor_radius_m

    evaluation = evaluate_profiles(a_grid_m, model)
    source_1d = evaluation.source_density_n_per_m3_per_s
    dV_shell_m3 = np.sum(dV_m3, axis=1)

    numerator = np.sum((rho_grid**2) * source_1d * dV_shell_m3)
    denominator = np.sum(source_1d * dV_shell_m3)

    if denominator <= 0.0:
        raise ValueError("Total source weight is negative")

    return float(np.sqrt(numerator / denominator))