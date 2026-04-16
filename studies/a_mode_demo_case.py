from __future__ import annotations

from pathlib import Path

import numpy as np

from tokamak_source_model.geometry import make_a_alpha_grids
from tokamak_source_model.normalization import build_source_probability_map, estimate_total_neutron_rate_n_per_s, estimate_total_plasma_volume_m3
from tokamak_source_model.parameters import FuelParameters, GeometryParameters, MeshParameters, ProfileParameters, SourceModelParameters
from tokamak_source_model.plotting import plot_magnetic_surfaces, plot_probability_map_rz, plot_profiles_vs_a, plot_sampled_birth_points, plot_source_quantities_vs_a
from tokamak_source_model.sampling import sample_source_particles
from tokamak_source_model.source_density import evaluate_profiles
from tokamak_source_model.validation import validate_source_model_parameters

def main() -> None:
    output_dir = Path("studies/output/a_mode")
    output_dir.mkdir(parents=True, exist_ok=True)

    geometry = GeometryParameters(
        major_radius_m=2.0,
        minor_radius_m=0.5,
        elongation=1.7,
        triangularity=0.33,
        shafranov_shift_m=0.1
    )

    profile = ProfileParameters(
        mode="pedestal", 
        ion_density_center_m3=1.09e20, 
        ion_temp_center_keV=45.9,
        alpha_n=1.0,
        alpha_T=8.06,
        pedestal_radius_m=0.8 * geometry.minor_radius_m,
        ion_density_pedestal_m3=1.09e20,
        ion_density_separatrix_m3=3.0e19,
        ion_temp_pedestal_keV=6.09,
        ion_temp_separatrix_keV=0.1,
        beta_T=6.0,
    )

    fuel = FuelParameters(
        deuterium_fraction=0.5,
        tritium_fraction=0.5,
    )

    model = SourceModelParameters(
        geometry=geometry,
        profile=profile,
        fuel=fuel,
    )

    validate_source_model_parameters(model)

    mesh = MeshParameters(
        num_a=200,
        num_alpha=360,
    )

    a_grid_m, alpha_grid_rad = make_a_alpha_grids(model.geometry, mesh)
    evaluation = evaluate_profiles(a_grid_m, model)

    total_volume_m3 = estimate_total_plasma_volume_m3(model, mesh)
    total_rate_n_per_s = estimate_total_neutron_rate_n_per_s(model, mesh)
    
    _, _, R_m, Z_m, probability_map = build_source_probability_map(model, mesh)

    rng = np.random.default_rng(42)
    samples = sample_source_particles(
        n_samples=5000,
        model=model,
        mesh=mesh,
        rng=rng,
    )

    direction_norms = np.sqrt(samples.u_x**2 + samples.u_y**2 + samples.u_z**2)

    print("A-mode demo case")
    print("-" * 25)
    print(f"n_i(center) = {evaluation.ion_density_m3[0]:.6e} m^-3")
    print(f"T_i(center) = {evaluation.ion_temp_keV[0]:.6e} keV")
    print(f"<sv>(center) = {evaluation.reactivity_m3_per_s[0]:.6e} m^3/s")
    print(
        f"S(center) ="
        f"{evaluation.source_density_n_per_m3_per_s[0]:.6e} n/(m^3 s)"
    )
    print(f"n_i(edge) = {evaluation.ion_density_m3[-1]:.6e} m^-3")
    print(f"T_i(edge) = {evaluation.ion_temp_keV[-1]:.6e} keV")

    nonzero_mask = evaluation.ion_temp_keV > 0.0
    if np.any(nonzero_mask):
        print(
            f"S(max)             ="
            f"{np.max(evaluation.source_density_n_per_m3_per_s):.6e} n/(m^3 s)"
        )

    print(f"Plasma volume        = {total_volume_m3:.6e} m^3")
    print(f"Total neutron rate   = {total_rate_n_per_s:.6e} n/s")
    print(f"Probability sum      = {np.sum(probability_map):.12f}")
    print(f"Mean sampled energy  = {np.mean(samples.energy_eV):.6e}")
    print(f"Mean direction norm  = {np.mean(direction_norms):.12f}")
    print(f"Weight sum           = {np.sum(samples.weight):.12f}")
    print(f"x range              = [{np.min(samples.x_m):.6e}, {np.max(samples.x_m):.6e}] m")
    print(f"z range              = [{np.min(samples.z_m):.6e}, {np.max(samples.z_m):.6e}] m")


    surface_radii_m = np.linspace(
        0.1 * geometry.minor_radius_m,
        geometry.major_radius_m,
        8
    )
        
    plot_magnetic_surfaces(
        geometry=geometry,
        surface_radii_m=surface_radii_m,
        alpha_rad=alpha_grid_rad,
        output_path=output_dir / "a_mode_magnetic_surfaces.png",
    )

    plot_profiles_vs_a(
        a_m=a_grid_m,
        geometry=geometry,
        profile=profile,
        output_path=output_dir / "a_mode_profiles.png"
    )

    plot_source_quantities_vs_a(
        evaluation=evaluation,
        output_path=output_dir / "_mode_source_quantities.png"
    )

    plot_probability_map_rz(
        R_m=R_m,
        Z_m=Z_m,
        probability_map=probability_map,
        output_path=output_dir / "a_mode_probability.png"
    )

    plot_sampled_birth_points(
        x_m=samples.x_m,
        z_m=samples.z_m,
        output_path = output_dir / "a_mode_sampled_birth_points.png"
    )

if __name__ == "__main__":
    main()