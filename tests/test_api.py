from __future__ import annotations

import numpy as np
import pytest

from tokamak_source_model import tokamak_source


def _common_kwargs() -> dict:
    return {
        "major_radius_m": 2.0,
        "minor_radius_m": 0.5,
        "elongation": 1.7,
        "triangularity": 0.33,
        "shafranov_shift_m": 0.1,
        "ion_density_center_m3": 2.0e20,
        "ion_temp_center_keV": 15.0,
        "alpha_n": 0.5,
        "alpha_T": 1.0,
        "deuterium_fraction": 0.5,
        "tritium_fraction": 0.5,
        "n_samples": 20,
        "num_a": 40,
        "num_alpha": 60,
        "num_R": 40,
        "num_Z": 40,
        "rng": np.random.default_rng(123),
    }


def _pedestal_kwargs() -> dict:
    return {
        "pedestal_radius_m": 0.4,
        "ion_density_pedestal_m3": 1.8e20,
        "ion_density_separatrix_m3": 3.0e19,
        "ion_temp_pedestal_keV": 4.0,
        "ion_temp_separatrix_keV": 0.1,
        "beta_T": 6.0,
    }


def test_tokamak_source_l_mode_builds_sources():
    sources = tokamak_source(
        mode="L",
        **_common_kwargs(),
        pedestal_radius_m=0.4,
        ion_density_pedestal_m3=1.8e20,
        ion_density_separatrix_m3=3.0e19,
        ion_temp_pedestal_keV=4.0,
        ion_temp_separatrix_keV=0.1,
        beta_T=6.0,
    )

    assert len(sources) == 20
    assert all(source is not None for source in sources)


def test_tokamak_source_h_mode_builds_sources():
    sources = tokamak_source(
        mode="H",
        **_common_kwargs(),
        **_pedestal_kwargs(),
    )

    assert len(sources) == 20
    assert all(source is not None for source in sources)


def test_tokamak_source_a_mode_builds_sources():
    sources = tokamak_source(
        mode="A",
        **_common_kwargs(),
        pedestal_radius_m=0.8 * 0.5,
        ion_density_pedestal_m3=1.09e20,
        ion_density_separatrix_m3=3.0e19,
        ion_temp_pedestal_keV=6.09,
        ion_temp_separatrix_keV=0.1,
        beta_T=6.0,
    )

    assert len(sources) == 20
    assert all(source is not None for source in sources)


def test_tokamak_source_h_mode_requires_pedestal_inputs():
    kwargs = _common_kwargs()

    with pytest.raises(ValueError, match="pedestal radius required for H/A mode"):
        tokamak_source(
            mode="H",
            **kwargs,
            ion_density_pedestal_m3=1.8e20,
            ion_density_separatrix_m3=3.0e19,
            ion_temp_pedestal_keV=4.0,
            ion_temp_separatrix_keV=0.1,
            beta_T=6.0,
        )


def test_tokamak_source_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode must be one of"):
        tokamak_source(
            mode="X",
            **_common_kwargs(),
            **_pedestal_kwargs(),
        )

def test_tokamak_source_strengths_sum_to_one_for_l_mode():
    sources = tokamak_source(
        mode="L",
        **_common_kwargs(),
        pedestal_radius_m=0.4,
        ion_density_pedestal_m3=1.8e20,
        ion_density_separatrix_m3=3.0e19,
        ion_temp_pedestal_keV=4.0,
        ion_temp_separatrix_keV=0.1,
        beta_T=6.0,
    )

    total_strength = sum(source.strength for source in sources)
    assert np.isclose(total_strength, 1.0)