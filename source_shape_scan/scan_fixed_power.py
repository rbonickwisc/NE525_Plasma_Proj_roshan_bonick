from __future__ import annotations

from pathlib import Path
import math

import openmc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from tokamak_source_model.utils.case_builder import build_default_mesh
from tokamak_source_model.utils.normalization import estimate_total_neutron_rate_n_per_s
from tokamak_source_model.utils.openmc_adapter import build_openmc_independent_sources

from metrics import source_weighted_mean_radius
from profiles import build_a_mode_from_parameters, build_l_mode_reference


# -----------------------------
# OpenMC geometry constants
# -----------------------------
R0_CM = 200.0
PLASMA_R_CM = 20.0
FIRST_WALL_R_CM = 50.0
FLIBE_R_CM = 100.0
OUTER_SPHERE_R_CM = R0_CM + FLIBE_R_CM + 100.0

# FLiBe torus-shell volume for converting H3-production to g/(m^3 yr)
N_AV = 6.02214076e23
M_T_G_PER_MOL = 3.01604928
SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0
V_FLIBE_CM3 = 2.0 * math.pi**2 * R0_CM * (FLIBE_R_CM**2 - FIRST_WALL_R_CM**2)
V_FLIBE_M3 = V_FLIBE_CM3 * 1.0e-6


# -----------------------------
# OpenMC helpers
# -----------------------------
def build_materials() -> openmc.Materials:
    mats = openmc.Materials()

    steel = openmc.Material(name="steel")
    steel.set_density("g/cm3", 7.8)
    steel.add_element("Fe", 0.88)
    steel.add_element("Cr", 0.12)
    mats.append(steel)

    flibe = openmc.Material(name="FLiBe")
    flibe.set_density("g/cm3", 1.95)
    flibe.add_nuclide("Li6", 2.0 * 0.075)
    flibe.add_nuclide("Li7", 2.0 * 0.925)
    flibe.add_element("Be", 1.0)
    flibe.add_element("F", 4.0)
    mats.append(flibe)

    return mats


def build_geometry(materials: openmc.Materials) -> tuple[openmc.Geometry, openmc.Cell, openmc.Cell]:
    plasma_torus = openmc.ZTorus(x0=0.0, y0=0.0, z0=0.0, a=R0_CM, b=PLASMA_R_CM, c=PLASMA_R_CM)
    first_wall_torus = openmc.ZTorus(x0=0.0, y0=0.0, z0=0.0, a=R0_CM, b=FIRST_WALL_R_CM, c=FIRST_WALL_R_CM)
    flibe_torus = openmc.ZTorus(x0=0.0, y0=0.0, z0=0.0, a=R0_CM, b=FLIBE_R_CM, c=FLIBE_R_CM)
    outer_sphere = openmc.Sphere(r=OUTER_SPHERE_R_CM, boundary_type="vacuum")

    plasma_region = -plasma_torus
    first_wall_region = +plasma_torus & -first_wall_torus
    flibe_region = +first_wall_torus & -flibe_torus
    outside_region = +flibe_torus & -outer_sphere

    plasma_cell = openmc.Cell(name="Plasma", region=plasma_region, fill=None)
    first_wall_cell = openmc.Cell(name="First Wall", region=first_wall_region)
    flibe_cell = openmc.Cell(name="FLiBe Blanket", region=flibe_region)
    outside_cell = openmc.Cell(name="Outside Void", region=outside_region, fill=None)

    steel = next(mat for mat in materials if mat.name == "steel")
    flibe = next(mat for mat in materials if mat.name == "FLiBe")
    first_wall_cell.fill = steel
    flibe_cell.fill = flibe

    root_universe = openmc.Universe(cells=[plasma_cell, first_wall_cell, flibe_cell, outside_cell])
    geometry = openmc.Geometry(root_universe)

    return geometry, first_wall_cell, flibe_cell

def build_tallies(first_wall_cell: openmc.Cell, flibe_cell: openmc.Cell) -> openmc.Tallies:
    tallies = openmc.Tallies()

    fw_filter = openmc.CellFilter(first_wall_cell)
    flibe_filter = openmc.CellFilter(flibe_cell)
    energy_filter = openmc.EnergyFilter([0.0, 0.625, 1e5, 20e6])

    tbr = openmc.Tally(name="Tritium production in FLiBe")
    tbr.filters = [flibe_filter]
    tbr.scores = ["H3-production"]
    tallies.append(tbr)

    flibe_flux = openmc.Tally(name="Flux in FLiBe (E-binned)")
    flibe_flux.filters = [flibe_filter, energy_filter]
    flibe_flux.scores = ["flux"]
    tallies.append(flibe_flux)

    fw_flux = openmc.Tally(name="Flux in first wall (E-binned)")
    fw_flux.filters = [fw_filter, energy_filter]
    fw_flux.scores = ["flux"]
    tallies.append(fw_flux)

    return tallies


def build_settings(
    sources: list[openmc.IndependentSource],
    batches: int,
    particles: int,
) -> openmc.Settings:
    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.source = sources
    settings.batches = batches
    settings.inactive = 0
    settings.particles = particles
    settings.max_lost_particles = 1000
    settings.rel_max_lost_particles = 0.01
    settings.statepoint = {"batches": [batches]}
    return settings


def get_total_tally_mean_std(sp: openmc.StatePoint, tally_name: str) -> tuple[float, float]:
    tally = sp.get_tally(name=tally_name)
    df = tally.get_pandas_dataframe()
    mean = float(df["mean"].sum())
    std = float(np.sqrt(np.sum(df["std. dev."] ** 2)))
    return mean, std


def get_fast_flux_mean_std(sp: openmc.StatePoint, tally_name: str) -> tuple[float, float]:
    tally = sp.get_tally(name=tally_name)
    df = tally.get_pandas_dataframe()

    fast = df[
        (df["energy low [eV]"] >= 1.0e5)
        & (df["energy high [eV]"] <= 2.0e7)
    ]

    mean = float(fast["mean"].sum())
    std = float(np.sqrt(np.sum(fast["std. dev."] ** 2)))
    return mean, std


def h3_to_g_per_m3_per_yr(
    mean_per_source: float,
    std_per_source: float,
    ndot_target: float,
) -> tuple[float, float]:
    reactions_per_s = mean_per_source * ndot_target
    reactions_std_per_s = std_per_source * ndot_target

    g_per_s = reactions_per_s * (M_T_G_PER_MOL / N_AV)
    g_std_per_s = reactions_std_per_s * (M_T_G_PER_MOL / N_AV)

    g_per_m3_yr = g_per_s * SECONDS_PER_YEAR / V_FLIBE_M3
    g_std_per_m3_yr = g_std_per_s * SECONDS_PER_YEAR / V_FLIBE_M3

    return g_per_m3_yr, g_std_per_m3_yr


def flux_to_physical(
    mean_per_source: float,
    std_per_source: float,
    ndot_target: float,
) -> tuple[float, float]:
    return mean_per_source * ndot_target, std_per_source * ndot_target


# -----------------------------
# Main scan
# -----------------------------
def main() -> None:
    output_dir = Path(__file__).resolve().parent / "output" / "fixed_power_openmc_scan"
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh = build_default_mesh()

    # Use L-mode reference to define the fixed total fusion power / neutron rate
    l_model = build_l_mode_reference()
    ndot_target = estimate_total_neutron_rate_n_per_s(l_model, mesh)
    rho_l = source_weighted_mean_radius(l_model, mesh)

    # Start small so runtime stays reasonable
    pedestal_fracs = [0.55, 0.65, 0.75, 0.85, 0.95]

    batches = 20
    particles = 50000
    n_samples = 3000

    rows = []

    # Optional: include the L-mode reference itself
    cases = [("L-mode reference", None)] + [(f"A-like {frac:.2f}", frac) for frac in pedestal_fracs]

    for label, frac in cases:
        if frac is None:
            model_params = l_model
            case_tag = "l_mode_reference"
        else:
            model_params = build_a_mode_from_parameters(pedestal_fraction=float(frac))
            case_tag = f"ped_{str(frac).replace('.', 'p')}"

        case_dir = output_dir / case_tag
        case_dir.mkdir(parents=True, exist_ok=True)

        raw_ndot = estimate_total_neutron_rate_n_per_s(model_params, mesh)
        rho_s = source_weighted_mean_radius(model_params, mesh)

        rng = np.random.default_rng(42)
        sources = build_openmc_independent_sources(
            n_samples=n_samples,
            model=model_params,
            mesh=mesh,
            rng=rng,
        )

        materials = build_materials()
        geometry, first_wall_cell, flibe_cell = build_geometry(materials)
        tallies = build_tallies(first_wall_cell, flibe_cell)
        settings = build_settings(sources, batches=batches, particles=particles)

        model = openmc.Model(
            geometry=geometry,
            materials=materials,
            settings=settings,
            tallies=tallies,
        )
        model.export_to_xml(directory=case_dir)

        print(f"Running case: {label}")
        openmc.run(cwd=case_dir, output=True)

        sp = openmc.StatePoint(case_dir / f"statepoint.{batches}.h5")

        tbr_mean, tbr_std = get_total_tally_mean_std(sp, "Tritium production in FLiBe")
        flibe_fast_mean, flibe_fast_std = get_fast_flux_mean_std(sp, "Flux in FLiBe (E-binned)")
        fw_fast_mean, fw_fast_std = get_fast_flux_mean_std(sp, "Flux in first wall (E-binned)")

        tbr_phys, tbr_phys_std = h3_to_g_per_m3_per_yr(tbr_mean, tbr_std, ndot_target)
        flibe_fast_phys, flibe_fast_phys_std = flux_to_physical(flibe_fast_mean, flibe_fast_std, ndot_target)
        fw_fast_phys, fw_fast_phys_std = flux_to_physical(fw_fast_mean, fw_fast_std, ndot_target)

        rows.append(
            {
                "case": label,
                "pedestal_fraction": frac,
                "rho_source_mean": rho_s,
                "raw_total_neutron_rate_n_per_s": raw_ndot,
                "scale_to_fixed_power": ndot_target / raw_ndot,
                "blanket_tritium_g_m3_yr": tbr_phys,
                "blanket_tritium_std": tbr_phys_std,
                "blanket_fast_flux_n_cm2_s": flibe_fast_phys,
                "blanket_fast_flux_std": flibe_fast_phys_std,
                "first_wall_fast_flux_n_cm2_s": fw_fast_phys,
                "first_wall_fast_flux_std": fw_fast_phys_std,
                "tbr_to_fw_flux_ratio": tbr_phys / fw_fast_phys,
            }
        )

    df = pd.DataFrame(rows)
    csv_path = output_dir / "fixed_power_scan_results.csv"
    df.to_csv(csv_path, index=False)

    # A-like scan only for trend plots
    scan_df = df[df["pedestal_fraction"].notna()].copy()

    plt.figure(figsize=(8, 5))
    plt.plot(scan_df["rho_source_mean"], scan_df["blanket_tritium_g_m3_yr"], marker="o")
    plt.xlabel(r"Source-weighted mean radius $\langle \rho \rangle_S$")
    plt.ylabel(r"Blanket tritium production [g/(m$^3$ yr)]")
    plt.title("Blanket tritium production at fixed total fusion power")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "tbr_vs_rho_source_mean.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(scan_df["rho_source_mean"], scan_df["first_wall_fast_flux_n_cm2_s"], marker="o")
    plt.xlabel(r"Source-weighted mean radius $\langle \rho \rangle_S$")
    plt.ylabel(r"First-wall fast flux [n/(cm$^2$ s)]")
    plt.title("First-wall fast flux at fixed total fusion power")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "fw_fast_flux_vs_rho_source_mean.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(scan_df["rho_source_mean"], scan_df["tbr_to_fw_flux_ratio"], marker="o")
    plt.xlabel(r"Source-weighted mean radius $\langle \rho \rangle_S$")
    plt.ylabel(r"Tritium production / first-wall fast flux")
    plt.title("Breeding-to-wall-loading tradeoff at fixed total fusion power")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "tradeoff_vs_rho_source_mean.png", dpi=300)
    plt.close()

    print("\nFixed-power scan complete")
    print(df.to_string(index=False))
    print(f"\nSaved results to: {csv_path}")
    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()