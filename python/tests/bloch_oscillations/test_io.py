"""
Tests for bloch_oscillations.io.

Verifies that simulation data survives a round-trip through
``save_simulation_data`` / ``load_simulation_data``, and that the log
header writer creates a file with the expected section markers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from domcsis_epic_tinker_box.bloch_oscillations.model import ModelParams, RunConfig
from domcsis_epic_tinker_box.bloch_oscillations.io import (
    save_simulation_data,
    load_simulation_data,
    write_log_header,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture()
def tmp_sim_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Change the working directory to a temporary path so file helpers
    write to a controlled location that is cleaned up after each test."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture()
def small_arrays() -> tuple[np.ndarray[object, np.dtype[np.float64]], np.ndarray[object, np.dtype[np.float64]]]:
    """Return small deterministic magnetisation and correlator arrays."""
    rng = np.random.default_rng(42)
    mags = rng.uniform(-1.0, 1.0, (4, 3)).astype(np.float64)
    corr = rng.uniform(-0.5, 0.5, (4, 3)).astype(np.float64)
    return mags, corr


# ============================================================================
# save / load round-trip
# ============================================================================


def test_round_trip_preserves_arrays(
    tmp_sim_dir: Path,
    small_arrays: tuple[np.ndarray[object, np.dtype[np.float64]], np.ndarray[object, np.dtype[np.float64]]],
) -> None:
    """Data saved by save_simulation_data is recovered exactly by load."""
    params = ModelParams(L=3, layers_max=4)
    config = RunConfig()
    mags, corr = small_arrays

    path = save_simulation_data(params, config, mags, corr, "test_label")
    assert path.exists()

    mags_loaded, corr_loaded = load_simulation_data(path)

    np.testing.assert_array_almost_equal(mags, mags_loaded)
    np.testing.assert_array_almost_equal(corr, corr_loaded)


def test_load_raises_on_missing_file(tmp_sim_dir: Path) -> None:
    """load_simulation_data raises FileNotFoundError for non-existent paths."""
    with pytest.raises(FileNotFoundError):
        load_simulation_data(tmp_sim_dir / "no_such_file.txt")


def test_load_raises_on_malformed_file(tmp_sim_dir: Path) -> None:
    """load_simulation_data raises ValueError if section markers are absent."""
    bad_file = tmp_sim_dir / "bad.txt"
    bad_file.write_text("This file has no section markers.\n")
    with pytest.raises(ValueError, match="Could not find section markers"):
        load_simulation_data(bad_file)


# ============================================================================
# Log header
# ============================================================================


def test_write_log_header_creates_file(tmp_sim_dir: Path) -> None:
    """write_log_header creates the log file and returns its path."""
    params = ModelParams(L=3, layers_max=4)
    config = RunConfig()
    log_path = write_log_header(params, config, "test_log")
    assert log_path.exists()


def test_write_log_header_contains_section_markers(tmp_sim_dir: Path) -> None:
    """write_log_header writes all expected section headers."""
    params = ModelParams(L=3, layers_max=4)
    config = RunConfig()
    log_path = write_log_header(params, config, "test_log")
    text = log_path.read_text()
    assert "=== Simulation configuration ===" in text
    assert "=== Circuit information ===" in text
    assert "Model parameters:" in text
    assert "Run configuration:" in text


def test_write_log_header_records_param_values(tmp_sim_dir: Path) -> None:
    """write_log_header embeds the actual parameter values in the log."""
    params = ModelParams(L=5, layers_max=10, J=0.5)
    config = RunConfig(trotter_method="zig_zag")
    log_path = write_log_header(params, config, "param_check")
    text = log_path.read_text()
    assert "L: 5" in text
    assert "layers_max: 10" in text
    assert "J: 0.5" in text
    assert "zig_zag" in text
