import pytest
from domcsis_epic_tinker_box.bloch_oscillations.model import ModelParams
from domcsis_epic_tinker_box.bloch_oscillations.observables import (
    magnetisation_observables,
    correlator_observables,
)


@pytest.mark.parametrize("L", [3, 5, 7])
def test_magnetisation_observables_length(L):
    params = ModelParams(L=L)
    obs = magnetisation_observables(params)
    assert len(obs) == L


@pytest.mark.parametrize("L", [3, 5, 7])
def test_magnetisation_observables_single_z(L):
    """Each observable should be a single-Z Pauli string."""
    params = ModelParams(L=L)
    obs = magnetisation_observables(params)
    for i, op in enumerate(obs):
        label = op.paulis.to_labels()[0]
        assert label.count("Z") == 1
        # Qiskit Pauli string ordering is right-to-left (qubit 0 is rightmost)
        assert label[L - 1 - i] == "Z"


@pytest.mark.parametrize("L", [3, 5, 7])
def test_correlator_observables_length(L):
    params = ModelParams(L=L)
    obs = correlator_observables(params)
    assert len(obs) == L


@pytest.mark.parametrize("L", [3, 5, 7])
def test_correlator_observables_center_is_identity(L):
    """The center-site correlator should be all-identity (self-correlator excluded)."""
    import math
    params = ModelParams(L=L)
    obs = correlator_observables(params)
    center = math.ceil(L / 2 - 1)
    label = obs[center].paulis.to_labels()[0]
    assert label == "I" * L


@pytest.mark.parametrize("L", [5, 7])
def test_correlator_observables_two_z_off_center(L):
    """Off-center correlators should contain exactly two Z operators."""
    import math
    params = ModelParams(L=L)
    obs = correlator_observables(params)
    center = math.ceil(L / 2 - 1)
    for i, op in enumerate(obs):
        if i == center:
            continue
        label = op.paulis.to_labels()[0]
        assert label.count("Z") == 2
