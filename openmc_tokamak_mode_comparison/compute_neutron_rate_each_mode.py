from __future__ import annotations
from tokamak_source_model.utils.case_builder import build_a_mode_paper_model, build_default_mesh, build_l_mode_model, build_generic_pedestal_model
from tokamak_source_model.utils.normalization import estimate_total_neutron_rate_n_per_s

def main() -> None:
    mesh = build_default_mesh()

    models = {
        "L-mode": build_l_mode_model(),
        "H-mode": build_generic_pedestal_model(),
        "A-mode": build_a_mode_paper_model(),
    }

    print("Mode total neutron rate comparison")
    print("-" * 25)

    rates = {}
    for label, model in models.items():
        rate = estimate_total_neutron_rate_n_per_s(model, mesh)
        rates[label] = rate
        print(f"{label:>8}: {rate:.6e} n/s")

    l_rate = rates["L-mode"]
    h_rate = rates["H-mode"]
    a_rate = rates["A-mode"]

    print()
    print("Relative ratios")
    print("---------------")
    print(f"H / L = {h_rate / l_rate:.6f}")
    print(f"A / L = {a_rate / l_rate:.6f}")
    print(f"A / H = {a_rate / h_rate:.6f}")


if __name__ == "__main__":
    main()