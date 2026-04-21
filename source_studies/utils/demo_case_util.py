from __future__ import annotations
from pathlib import Path
import numpy as np

from tokamak_source_model.utils.case_builder import build_default_mesh, build_a_mode_paper_model
from tokamak_source_model.utils.geometry import make_a_alpha_grids
from tokamak_source_model.utils.normalization import build_source_probability_map, estimate_total_neutron_rate_n_per_s, estimate_total_plasma_volume_m3
from tokamak_source_model.utils.plotting import plot_magnetic_surfaces, plot_probability_map_rz, plot_profiles_vs_a, plot_sampled_birth_points, plot_source_quantities_vs_a
from tokamak_source_model.utils.sampling import sample_source_particles
from tokamak_source_model.utils.source_density import evaluate_profiles
from tokamak_source_model.utils.validation import validate_source_model_parameters

def run_demo_case(
    model,
    output_dir: Path,
    mode_label: str,
    file_prefix: str,
    seed: int = 42,
    n_samples: int = 5000,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh = build_default_mesh()
    geometry = model.geometry
    profile = model.profile

    validate_source_model_parameters(model)

    a_grid_m, alpha_grid_rad = make_a_alpha_grids(geometry, mesh)
    evaluation = evaluate_profiles(a_grid_m, model)

    total_volume_m3 = estimate_total_plasma_volume_m3(model, mesh)
    total_rate_n_per_s = estimate_total_neutron_rate_n_per_s(model, mesh)

    _, _, R_m, Z_m, probability_map = build_source_probability_map(model, mesh)

    rng = np.random.default_rng(seed)
    samples = sample_source_particles(
        n_samples=n_samples,
        model=model,
        mesh=mesh,
        rng=rng,
    )

    direction_norms = np.sqrt(samples.u_x**2 + samples.u_y**2 + samples.u_z**2)

    print(f"{mode_label} demo case")
    print("-" * 32)
    print(f"n_i(center)        = {evaluation.ion_density_m3[0]:.6e} m^-3")
    print(f"T_i(center)        = {evaluation.ion_temp_keV[0]:.6e} keV")
    print(f"<sv>(center)       = {evaluation.reactivity_m3_per_s[0]:.6e} m^3/s")
    print(f"S(center)          = {evaluation.source_density_n_per_m3_per_s[0]:.6e} n/(m^3 s)")
    print(f"n_i(edge)          = {evaluation.ion_density_m3[-1]:.6e} m^-3")
    print(f"T_i(edge)          = {evaluation.ion_temp_keV[-1]:.6e} keV")
    print(f"S(max)             = {np.max(evaluation.source_density_n_per_m3_per_s):.6e} n/(m^3 s)")
    print(f"Plasma volume      = {total_volume_m3:.6e} m^3")
    print(f"Total neutron rate = {total_rate_n_per_s:.6e} n/s")
    print(f"Probability sum    = {np.sum(probability_map):.12f}")
    print(f"Mean energy [eV]   = {np.mean(samples.energy_eV):.6e}")
    print(f"Std energy  [eV]   = {np.std(samples.energy_eV):.6e}")
    print(f"Min energy  [eV]   = {np.min(samples.energy_eV):.6e}")
    print(f"Max energy  [eV]   = {np.max(samples.energy_eV):.6e}")
    print(f"Mean |u|           = {np.mean(direction_norms):.12f}")
    print(f"Weight sum         = {np.sum(samples.weight):.12f}")
    print(f"x range            = [{np.min(samples.x_m):.6e}, {np.max(samples.x_m):.6e}] m")
    print(f"z range            = [{np.min(samples.z_m):.6e}, {np.max(samples.z_m):.6e}] m")

    surface_radii_m = np.linspace(0.1 * geometry.minor_radius_m, geometry.minor_radius_m, 8)

    plot_magnetic_surfaces(
        geometry=geometry,
        surface_radii_m=surface_radii_m,
        alpha_rad=alpha_grid_rad,
        output_path=output_dir / f"{file_prefix}_magnetic_surfaces.png",
    )

    plot_profiles_vs_a(
        a_m=a_grid_m,
        geometry=geometry,
        profile=profile,
        output_path=output_dir / f"{file_prefix}_profiles.png",
    )

    plot_source_quantities_vs_a(
        evaluation=evaluation,
        output_path=output_dir / f"{file_prefix}_source_quantities.png",
    )

    plot_probability_map_rz(
        R_m=R_m,
        Z_m=Z_m,
        probability_map=probability_map,
        output_path=output_dir / f"{file_prefix}_probability.png",
    )

    plot_sampled_birth_points(
        x_m=samples.x_m,
        z_m=samples.z_m,
        output_path=output_dir / f"{file_prefix}_sampled_birth_points.png",
    )