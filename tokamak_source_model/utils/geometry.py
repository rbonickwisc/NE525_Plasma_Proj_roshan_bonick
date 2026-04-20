from __future__ import annotations

import numpy as np

from .parameters import GeometryParameters, MeshParameters

def surface_to_rz(
        a_m: np.ndarray | float,
        alpha_rad: np.ndarray | float,
        geometry: GeometryParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map magnetic surface coordinates (a, alpha) to cylindrical (R, Z)

    Uses analytic tokamak shaping model:

        R = R0 + a*cos(alpha + delta*sin(alpha)) + esh*(1 - (a/A)^2)

        Z = El * a * sin(alpha)

    Parameters
    -------------------------
    a_m
        Radial magnetic surface coordinate [m]
    alpha_rad
        Poloidal angle [rad]
    geometry
        GeometryParameters instance

    Returns
    -------------------------
    R_m, Z_m
        Cylindrical coordinates [m]
    """

    R0 = geometry.major_radius_m
    A = geometry.minor_radius_m
    El = geometry.elongation
    delta = geometry.triangularity
    esh = geometry.shafranov_shift_m

    a_m = np.asarray(a_m, dtype = float)
    alpha_rad = np.asarray(alpha_rad, dtype=float)

    R_m = (
        R0 + a_m * np.cos(alpha_rad + delta * np.sin(alpha_rad)) + esh * (1.0 - (a_m/A) ** 2)
    )
    Z_m = El * a_m * np.sin(alpha_rad)

    return R_m, Z_m

def make_magnetic_surface_curve(
        surface_radius_m: float,
        alpha_rad: np.ndarray,
        geometry: GeometryParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a single close magnetic surface curve
    """
    
    return surface_to_rz(surface_radius_m, alpha_rad, geometry)

def make_a_alpha_grids(
        geometry: GeometryParameters,
        mesh: MeshParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create 1D a and alpha grids

    Returns
    ----------------------
    a_grid_m : np.ndarray
        1D array from a_min to minor_radius
    alpha_grid_rad : np.ndarray
        1D array from 0 to 2pi
    """
    a_grid_m = np.linspace(
        mesh.a_grid_min_m,
        geometry.minor_radius_m,
        mesh.num_a,
    )
    alpha_grid_rad = np.linspace(
        0.0,
        2.0 * np.pi,
        mesh.num_alpha,
        endpoint=False,
    )
    return a_grid_m, alpha_grid_rad