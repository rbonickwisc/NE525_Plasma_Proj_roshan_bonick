from __future__ import annotations

from pathlib import Path
import math
import pandas as pd
import openmc

N_AV = 6.02214076e23
M_T_G_PER_MOL = 3.01604928
SECONDS_PER_YEAR = 365.25 * 24 * 3600

N_DOT = {
    "L-mode": 3.150941e18,
    "H-mode": 1.895748e19,
    "A-mode": 5.326079e19,
}

OUTPUT_ROOT = Path("openmc_tokamak_mode_comparison/output")
STATEPOINTS = {
    "L-mode": OUTPUT_ROOT / "torus_mode_l" / "statepoint.25.h5",
    "H-mode": OUTPUT_ROOT / "torus_mode_h" / "statepoint.25.h5",
    "A-mode": OUTPUT_ROOT / "torus_mode_a" / "statepoint.25.h5",
}

R0_CM = 200.0
R_IN_CM = 50.0
R_OUT_CM = 100.0
V_FLIBE_CM3 = 2.0 * math.pi**2 * R0_CM * (R_OUT_CM**2 - R_IN_CM**2)
V_FLIBE_M3 = V_FLIBE_CM3 * 1.0e-6


def get_fast_flux_mean_std(sp: openmc.StatePoint, tally_name: str) -> tuple[float, float]:
    tally = sp.get_tally(name=tally_name)
    df = tally.get_pandas_dataframe()

    fast = df[(df["energy low [eV]"] >= 1.0e5) & (df["energy high [eV]"] <= 2.0e7)]

    mean = float(fast["mean"].sum())
    std = float(math.sqrt((fast["std. dev."] ** 2).sum()))
    return mean, std


def get_total_tally_mean_std(sp: openmc.StatePoint, tally_name: str) -> tuple[float, float]:
    tally = sp.get_tally(name=tally_name)
    df = tally.get_pandas_dataframe()

    mean = float(df["mean"].sum())
    std = float(math.sqrt((df["std. dev."] ** 2).sum()))
    return mean, std


def physical_blanket_tritium_g_per_m3_yr(mean_h3_per_source: float, std_h3_per_source: float, n_dot: float):
    reactions_per_s = mean_h3_per_source * n_dot
    reactions_std_per_s = std_h3_per_source * n_dot

    g_per_s = reactions_per_s * (M_T_G_PER_MOL / N_AV)
    g_std_per_s = reactions_std_per_s * (M_T_G_PER_MOL / N_AV)

    g_per_m3_yr = g_per_s * SECONDS_PER_YEAR / V_FLIBE_M3
    g_std_per_m3_yr = g_std_per_s * SECONDS_PER_YEAR / V_FLIBE_M3
    return g_per_m3_yr, g_std_per_m3_yr


def physical_flux_n_per_cm2_s(mean_flux_per_source_cm2: float, std_flux_per_source_cm2: float, n_dot: float):
    return mean_flux_per_source_cm2 * n_dot, std_flux_per_source_cm2 * n_dot


def relative_difference_and_2sigma(ref_mean: float, ref_std: float, test_mean: float, test_std: float):
    if ref_mean == 0.0:
        return float("nan"), float("nan")

    r = (test_mean - ref_mean) / ref_mean
    var_r = (test_std / ref_mean) ** 2 + ((test_mean * ref_std) / (ref_mean ** 2)) ** 2
    two_sigma = 2.0 * math.sqrt(var_r)
    return r, two_sigma


def summarize_openmc_physical():
    raw = {}

    for mode, path in STATEPOINTS.items():
        sp = openmc.StatePoint(path)
        n_dot = N_DOT[mode]

        h3_mean, h3_std = get_total_tally_mean_std(sp, "Tritium production in FLiBe")
        h3_phys, h3_phys_std = physical_blanket_tritium_g_per_m3_yr(h3_mean, h3_std, n_dot)

        flibe_flux_mean, flibe_flux_std = get_fast_flux_mean_std(sp, "Flux in FLiBe (E-binned)")
        flibe_flux_phys, flibe_flux_phys_std = physical_flux_n_per_cm2_s(flibe_flux_mean, flibe_flux_std, n_dot)

        fw_flux_mean, fw_flux_std = get_fast_flux_mean_std(sp, "Flux in first wall (E-binned)")
        fw_flux_phys, fw_flux_phys_std = physical_flux_n_per_cm2_s(fw_flux_mean, fw_flux_std, n_dot)

        raw[mode] = {
            "Avg blanket tritium production [g/m^3/yr]": (h3_phys, h3_phys_std),
            "Avg blanket fast flux [n/cm^2/s]": (flibe_flux_phys, flibe_flux_phys_std),
            "Avg first-wall fast flux [n/cm^2/s]": (fw_flux_phys, fw_flux_phys_std),
        }

    rows = []
    metrics = list(raw["L-mode"].keys())

    for metric in metrics:
        l_mean, l_std = raw["L-mode"][metric]
        h_mean, h_std = raw["H-mode"][metric]
        a_mean, a_std = raw["A-mode"][metric]

        dh, dh_2s = relative_difference_and_2sigma(l_mean, l_std, h_mean, h_std)
        da, da_2s = relative_difference_and_2sigma(l_mean, l_std, a_mean, a_std)

        rows.append({
            "Metric": metric,
            "L-mode": l_mean,
            "H-mode": h_mean,
            "(H-L)/L [%]": 100.0 * dh,
            "2σ H [%]": 100.0 * dh_2s,
            "A-mode": a_mean,
            "(A-L)/L [%]": 100.0 * da,
            "2σ A [%]": 100.0 * da_2s,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = summarize_openmc_physical()
    pd.set_option("display.float_format", lambda x: f"{x:.6e}")
    print(df.to_string(index=False))