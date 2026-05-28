from domcsis_epic_tinker_box.bloch_oscillations.model import (
    ModelParams,
    RunConfig,
    center_index,
)


def test_model_params_defaults():
    params = ModelParams()
    assert params.L == 7
    assert params.layers_max == 40
    assert params.J == 1.0
    assert params.dt == 0.25


def test_run_config_defaults():
    config = RunConfig()
    assert config.backend_mode == "ideal"
    assert config.trotter_method == "even_odd"
    assert config.initial_state == "all_up"
    assert config.shots == 8192


def test_model_params_custom():
    params = ModelParams(L=5, J=0.25, dt=0.1)
    assert params.L == 5
    assert params.J == 0.25
    assert params.dt == 0.1


def test_center_index_odd():
    assert center_index(ModelParams(L=7)) == 3
    assert center_index(ModelParams(L=5)) == 2
    assert center_index(ModelParams(L=3)) == 1


def test_center_index_even():
    assert center_index(ModelParams(L=6)) == 2
    assert center_index(ModelParams(L=4)) == 1


def test_model_params_frozen():
    import pytest
    params = ModelParams()
    with pytest.raises(Exception):
        params.L = 10  # type: ignore[misc]
