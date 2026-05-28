import numpy as np
import pytest

from domcsis_epic_tinker_box.bloch_oscillations.io import (
    save_simulation_data,
    load_simulation_data,
    make_data_filename,
    log_simulation_header,
    append_layer_log,
)
from domcsis_epic_tinker_box.bloch_oscillations.model import ModelParams, RunConfig


@pytest.fixture
def small_data():
    params = ModelParams(L=3, layers_max=2)
    config = RunConfig()
    mags = np.array([[1.0, 0.0, -1.0], [0.5, 0.0, -0.5]])
    corr = np.array([[0.0, 0.1, 0.0], [0.0, 0.2, 0.0]])
    return params, config, mags, corr


def test_save_load_roundtrip(tmp_path, small_data):
    params, config, mags, corr = small_data
    filename = save_simulation_data(
        params, config, mags, corr, "test_run", base_dir=tmp_path
    )
    mags_loaded, corr_loaded = load_simulation_data(filename)
    np.testing.assert_allclose(mags_loaded, mags)
    np.testing.assert_allclose(corr_loaded, corr)


def test_save_creates_file(tmp_path, small_data):
    params, config, mags, corr = small_data
    filename = save_simulation_data(
        params, config, mags, corr, "test_run", base_dir=tmp_path
    )
    assert filename.exists()


def test_make_data_filename_contains_params(tmp_path):
    params = ModelParams(L=7, layers_max=40, J=0.25)
    config = RunConfig()
    filename = make_data_filename(params, config, "my_label", base_dir=tmp_path)
    assert "N_7" in str(filename)
    assert "layers_40" in str(filename)
    assert "J_0.25" in str(filename)
    assert "my_label" in str(filename)


def test_load_invalid_file_raises(tmp_path):
    bad_file = tmp_path / "bad.txt"
    bad_file.write_text("no sections here")
    with pytest.raises(ValueError, match="Could not find"):
        load_simulation_data(bad_file)


def test_log_simulation_header_creates_file(tmp_path):
    params = ModelParams(L=3, layers_max=2)
    config = RunConfig()
    log_file = log_simulation_header(
        params, config, "test_log", base_dir=tmp_path
    )
    assert log_file.exists()
    content = log_file.read_text()
    assert "Simulation configuration" in content
    assert "Model parameters" in content
