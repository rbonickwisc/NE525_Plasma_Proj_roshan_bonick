from __future__ import annotations
from pathlib import Path
import math
import pandas as pd
import openmc
from tokamak_source_model.utils.case_builder import build_default_mesh, build_l_mode_model, build_a_mode_paper_model
from tokamak_source_model.utils.normalization import estimate_total_neutron_rate_n_per_s


N_AV = 6.02214076e23       
M_T_G_PER_MOL = 3.01604928
SECONDS_PER_YEAR = 365.25 * 24 * 3600

R0_CM = 200.0
R_IN_CM = 50.0
R_OUT_CM = 100.0
V_FLIBE_CM3 = 2.0 * math.pi**2 * R0_CM * (R_OUT_CM**2 - R_IN_CM**2)
V_FLIBE_M3 = V_FLIBE_CM3 * 1.0e-6


OUTPUT_ROOT = Path("openmc_tokamak_mode_comparison/output")

STATEPOINTS = {
    "L-mode": OUTPUT_ROOT / "torus_mode_l" / "statepoint.25.h5",
    "A-mode": OUTPUT_ROOT / "torus_mode_a" / "statepoint.25.h5",
}


def get_source_rates_n_per_s() -> dict[str, float]:
    """
    Compute total neutron production rates from the source model
    """
    mesh = build_default_mesh()

    l_model = build_l_mode_model()
    a_model = build_a_mode_paper_model()

    return {
        "L-mode": estimate_total_neutron_rate_n_per_s(l_model, mesh),
        "A-mode": estimate_total_neutron_rate_n_per_s(a_model, mesh),
    }


def get_total_tally_mean_std(
    sp: openmc.StatePoint,
    tally_name: str,
) -> tuple[float, float]:
    """
    sum all the bins of a tally and combine stds
    """
    tally = sp.get_tally(name=tally_name)
    df = tally.get_pandas_dataframe()

    mean = float(df["mean"].sum())
    std = float(math.sqrt((df["std. dev."] ** 2).sum()))
    return mean, std


def get_fast_flux_mean_std(
    sp: openmc.StatePoint,
    tally_name: str,
) -> tuple[float, float]:
    """
    extract only fast-energy bin
    """
    tally = sp.get_tally(name=tally_name)
    df = tally.get_pandas_dataframe()

    fast = df[
        (df["energy low [eV]"] >= 1.0e5)
        & (df["energy high [eV]"] <= 2.0e7)
    ]

    mean = float(fast["mean"].sum())
    std = float(math.sqrt((fast["std. dev."] ** 2).sum()))
    return mean, std


def scale_h3_to_g_per_m3_per_yr(
    mean_per_source: float,
    std_per_source: float,
    n_dot_n_per_s: float,
) -> tuple[float, float]:
    """
    convert H3-prod tally from reactions/source to avg. blanket tritium prod
    """
    reactions_per_s = mean_per_source * n_dot_n_per_s
    reactions_std_per_s = std_per_source * n_dot_n_per_s

    g_per_s = reactions_per_s * (M_T_G_PER_MOL / N_AV)
    g_std_per_s = reactions_std_per_s * (M_T_G_PER_MOL / N_AV)

    g_per_m3_yr = g_per_s * SECONDS_PER_YEAR / V_FLIBE_M3
    g_std_per_m3_yr = g_std_per_s * SECONDS_PER_YEAR / V_FLIBE_M3

    return g_per_m3_yr, g_std_per_m3_yr


def scale_flux_to_physical(
    mean_per_source: float,
    std_per_source: float,
    n_dot_n_per_s: float,
) -> tuple[float, float]:
    """
    convert openmc flux tally from source based to physical value
    """
    return mean_per_source * n_dot_n_per_s, std_per_source * n_dot_n_per_s


def relative_difference_and_2sigma(
    ref_mean: float,
    ref_std: float,
    test_mean: float,
    test_std: float,
) -> tuple[float, float]:
    """
    return relative difference (test - ref)/ref and 2*std uncertainty
    """
    if ref_mean == 0.0:
        return float("nan"), float("nan")

    r = (test_mean - ref_mean) / ref_mean

    var_r = (test_std / ref_mean) ** 2 + ((test_mean * ref_std) / (ref_mean ** 2)) ** 2
    two_sigma = 2.0 * math.sqrt(var_r)

    return r, two_sigma


def summarize_l_vs_a_same_power() -> pd.DataFrame:
    """
    compare L and A mode after normalizing A mode to the same fusion power as L mode
    """
    source_rates = get_source_rates_n_per_s()

    n_dot_l = source_rates["L-mode"]
    n_dot_a = source_rates["A-mode"]

    a_scale = n_dot_l / n_dot_a

    sp_l = openmc.StatePoint(STATEPOINTS["L-mode"])
    sp_a = openmc.StatePoint(STATEPOINTS["A-mode"])

    l_h3_mean, l_h3_std = get_total_tally_mean_std(sp_l, "Tritium production in FLiBe")
    a_h3_mean, a_h3_std = get_total_tally_mean_std(sp_a, "Tritium production in FLiBe")

    l_bf_mean, l_bf_std = get_fast_flux_mean_std(sp_l, "Flux in FLiBe (E-binned)")
    a_bf_mean, a_bf_std = get_fast_flux_mean_std(sp_a, "Flux in FLiBe (E-binned)")

    l_fw_mean, l_fw_std = get_fast_flux_mean_std(sp_l, "Flux in first wall (E-binned)")
    a_fw_mean, a_fw_std = get_fast_flux_mean_std(sp_a, "Flux in first wall (E-binned)")

    l_h3_phys, l_h3_phys_std = scale_h3_to_g_per_m3_per_yr(l_h3_mean, l_h3_std, n_dot_l)
    l_bf_phys, l_bf_phys_std = scale_flux_to_physical(l_bf_mean, l_bf_std, n_dot_l)
    l_fw_phys, l_fw_phys_std = scale_flux_to_physical(l_fw_mean, l_fw_std, n_dot_l)

    a_h3_same, a_h3_same_std = scale_h3_to_g_per_m3_per_yr(a_h3_mean, a_h3_std, n_dot_l)
    a_bf_same, a_bf_same_std = scale_flux_to_physical(a_bf_mean, a_bf_std, n_dot_l)
    a_fw_same, a_fw_same_std = scale_flux_to_physical(a_fw_mean, a_fw_std, n_dot_l)

    rows = []

    for metric, l_mean, l_std, a_mean, a_std in [
        ("Avg blanket tritium production [g/(m^3 yr)]", l_h3_phys, l_h3_phys_std, a_h3_same, a_h3_same_std),
        ("Avg blanket fast flux [n/(cm^2 s)]", l_bf_phys, l_bf_phys_std, a_bf_same, a_bf_same_std),
        ("Avg first-wall fast flux [n/(cm^2 s)]", l_fw_phys, l_fw_phys_std, a_fw_same, a_fw_same_std),
    ]:
        rel, two_sigma = relative_difference_and_2sigma(l_mean, l_std, a_mean, a_std)
        rows.append({
            "Metric": metric,
            "L-mode": l_mean,
            "A-mode (same fusion power)": a_mean,
            "(A-L)/L [%]": 100.0 * rel,
            "2σ [%]": 100.0 * two_sigma,
        })

    df = pd.DataFrame(rows)

    print("Source-rate normalization")
    print("-" * 25)
    print(f"L-mode total neutron rate = {n_dot_l:.6e} n/s")
    print(f"A-mode total neutron rate = {n_dot_a:.6e} n/s")
    print(f"A-mode scaling factor     = {a_scale:.6e}")
    print()

    return df


if __name__ == "__main__":
    df = summarize_l_vs_a_same_power()
    pd.set_option("display.float_format", lambda x: f"{x:.6e}")
    print(df.to_string(index=False))