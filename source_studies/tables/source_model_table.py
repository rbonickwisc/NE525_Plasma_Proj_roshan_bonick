from __future__ import annotations

from dataclasses import asdict
import math

import numpy as np
import pandas as pd

from tokamak_source_model.utils.case_builder import (
    build_default_mesh,
    build_l_mode_model,
    build_generic_pedestal_model,
    build_a_mode_paper_model,
)
from tokamak_source_model.utils.normalization import estimate_total_neutron_rate_n_per_s
from tokamak_source_model.utils.source_density import evaluate_profiles


def summarize_source_model():
    mesh = build_default_mesh()

    models = {
        "L-mode": build_l_mode_model(),
        "H-mode": build_generic_pedestal_model(),
        "A-mode": build_a_mode_paper_model(),
    }

    rows = []

    for mode, model in models.items():
        a_grid_m = np.linspace(0.0, model.geometry.minor_radius_m, mesh.num_a)
        evaluation = evaluate_profiles(a_grid_m, model)

        row = {
            "Mode": mode,
            "n_i0 [m^-3]": model.profile.ion_density_center_m3,
            "T_i0 [keV]": model.profile.ion_temp_center_keV,
            "max <σv> [m^3/s]": float(np.max(evaluation.reactivity_m3_per_s)),
            "max S [n/(m^3 s)]": float(np.max(evaluation.source_density_n_per_m3_per_s)),
            "Total neutron rate [n/s]": float(estimate_total_neutron_rate_n_per_s(model, mesh)),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    l_rate = df.loc[df["Mode"] == "L-mode", "Total neutron rate [n/s]"].iloc[0]
    df["Rate / L-mode"] = df["Total neutron rate [n/s]"] / l_rate
    df["(Rate - L) / L [%]"] = 100.0 * (df["Total neutron rate [n/s]"] - l_rate) / l_rate

    return df


if __name__ == "__main__":
    df = summarize_source_model()
    pd.set_option("display.float_format", lambda x: f"{x:.6e}")
    print(df.to_string(index=False))