"""
Tests for bloch_oscillations.simulation.

The ideal-simulation tests run against a small chain (L=3, layers_max=2)
with a noiseless AerSimulator so they complete in a few seconds without
requiring any hardware credentials.  The noisy-simulation tests are marked
``skip`` by default; set the ``skip`` flag to ``False`` once
``qiskit-ibm-runtime`` with fake backends is installed in the environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from domcsis_epic_tinker_box.bloch_oscillations.model import ModelParams, RunConfig
from domcsis_epic_tinker_box.bloch_oscillations.simulation import (
    run_ideal_simulation_counts,
    run_ideal_simulation_observables,
)


# ---------------------------------------------------------------------------
# Shared small-system fixtures
# ---------------------------------------------------------------------------

_SMALL_PARAMS = ModelParams(L=3, layers_max=2, J=0.25, ht=-0.15, hl=0.15, dt=0.25)
_IDEAL_CONFIG = RunConfig(
    backend_mode="ideal",
    use_cnot_zz=False,
    initial_state="all_up",
    optimization_level=0,
    use_parallel_u1=False,
    shots=64,
)


@pytest.fixture(autouse=True)
def tmp_sim_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect simulation data / log writes to a temp directory."""
    monkeypatch.chdir(tmp_path)


# ---------------------------------------------------------------------------
# Shape checks
# ---------------------------------------------------------------------------


def test_ideal_counts_returns_correct_shape() -> None:
    """run_ideal_simulation_counts returns arrays of shape (layers_max, L)."""
    mags, corr = run_ideal_simulation_counts(_SMALL_PARAMS, _IDEAL_CONFIG)
    assert mags.shape == (_SMALL_PARAMS.layers_max, _SMALL_PARAMS.L)
    assert corr.shape == (_SMALL_PARAMS.layers_max, _SMALL_PARAMS.L)


def test_ideal_observables_returns_correct_shape() -> None:
    """run_ideal_simulation_observables returns arrays of shape (layers_max, L)."""
    mags, corr = run_ideal_simulation_observables(_SMALL_PARAMS, _IDEAL_CONFIG)
    assert mags.shape == (_SMALL_PARAMS.layers_max, _SMALL_PARAMS.L)
    assert corr.shape == (_SMALL_PARAMS.layers_max, _SMALL_PARAMS.L)


# ---------------------------------------------------------------------------
# Physical sanity checks
# ---------------------------------------------------------------------------


def test_ideal_observables_magnetisations_bounded() -> None:
    """All \u27e8Z_j\u27e9 values must lie in [-1, 1] (eigenvalues of Z)."""
    mags, _ = run_ideal_simulation_observables(_SMALL_PARAMS, _IDEAL_CONFIG)
    assert np.all(mags >= -1.0 - 1e-9)
    assert np.all(mags <= 1.0 + 1e-9)


def test_ideal_counts_zero_layer_is_initial_state() -> None:
    """At Trotter step 0 (no gates), |0\u20260\u27e9 gives \u27e8Z_j\u27e9 = +1 for all j."""
    mags, _ = run_ideal_simulation_counts(_SMALL_PARAMS, _IDEAL_CONFIG)
    # Layer 0 means zero Trotter steps applied, so the state is still |000>.
    np.testing.assert_allclose(mags[0], np.ones(_SMALL_PARAMS.L), atol=0.15)


def test_ideal_observables_zero_layer_is_initial_state() -> None:
    """At Trotter step 0, exact statevector also gives \u27e8Z_j\u27e9 = +1."""
    mags, _ = run_ideal_simulation_observables(_SMALL_PARAMS, _IDEAL_CONFIG)
    np.testing.assert_allclose(mags[0], np.ones(_SMALL_PARAMS.L), atol=1e-9)


def test_ideal_counts_data_file_is_written(tmp_sim_dir: Any) -> None:
    """run_ideal_simulation_counts persists a data file to disk."""
    run_ideal_simulation_counts(_SMALL_PARAMS, _IDEAL_CONFIG)
    data_files = list(Path("simulation_data").glob("*.txt"))
    assert len(data_files) >= 1


# ---------------------------------------------------------------------------
# Noisy simulation (skipped unless fake backends installed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    True,
    reason="Requires qiskit-ibm-runtime with fake provider (set skip=False to enable)",
)
def test_noisy_counts_returns_correct_shape() -> None:
    """run_noisy_simulation_counts returns the expected array shapes."""
    from domcsis_epic_tinker_box.bloch_oscillations.simulation import (
        run_noisy_simulation_counts,
    )

    noisy_cfg = RunConfig(
        backend_mode="fake",
        fake_backend_name="brisbane",
        optimization_level=1,
        shots=64,
    )
    mags, corr = run_noisy_simulation_counts(_SMALL_PARAMS, noisy_cfg)
    assert mags.shape == (_SMALL_PARAMS.layers_max, _SMALL_PARAMS.L)
    assert corr.shape == (_SMALL_PARAMS.layers_max, _SMALL_PARAMS.L)
