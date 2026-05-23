from __future__ import annotations

import numpy as np
import pytest

from domcsis_epic_tinker_box.quantum_state.bell_states import bell_state


# ============================================================================
# String identifiers
# ============================================================================

def test_bell_state_phi_plus():
    state = bell_state("Phi+")
    expected = np.array(
        [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)],
        dtype=complex,
    )
    assert np.allclose(state.vector, expected)


def test_bell_state_phi_minus():
    state = bell_state("Phi-")
    expected = np.array(
        [1 / np.sqrt(2), 0, 0, -1 / np.sqrt(2)],
        dtype=complex,
    )
    assert np.allclose(state.vector, expected)


def test_bell_state_psi_plus():
    state = bell_state("Psi+")
    expected = np.array(
        [0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0],
        dtype=complex,
    )
    assert np.allclose(state.vector, expected)


def test_bell_state_psi_minus():
    state = bell_state("Psi-")
    expected = np.array(
        [0, 1 / np.sqrt(2), -1 / np.sqrt(2), 0],
        dtype=complex,
    )
    assert np.allclose(state.vector, expected)


@pytest.mark.parametrize(
    "label",
    ["phi+", "PHI+", "pHi+", "phi-", "psi+", "psi-"],
)
def test_bell_state_string_identifiers_are_case_insensitive(label):
    state = bell_state(label)
    assert np.isclose(np.linalg.norm(state.vector), 1.0)


def test_bell_state_invalid_string_raises():
    with pytest.raises(ValueError, match="Unrecognized Bell state identifier"):
        bell_state("not-a-bell-state")


# ============================================================================
# Integer identifiers
# ============================================================================

def test_bell_state_integer_two_qubits():
    state = bell_state(2)
    expected = np.array(
        [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)],
        dtype=complex,
    )
    assert np.allclose(state.vector, expected)


def test_bell_state_integer_three_qubits():
    state = bell_state(3)
    expected = np.zeros(8, dtype=complex)
    expected[0] = 1 / np.sqrt(2)
    expected[-1] = 1 / np.sqrt(2)
    assert np.allclose(state.vector, expected)


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4, 5])
def test_bell_state_integer_is_normalized(n_qubits):
    state = bell_state(n_qubits)
    assert np.isclose(np.linalg.norm(state.vector), 1.0)


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4, 5])
def test_bell_state_integer_has_only_two_nonzero_entries(n_qubits):
    state = bell_state(n_qubits)
    nonzero_indices = np.flatnonzero(np.abs(state.vector) > 1e-12)

    assert len(nonzero_indices) == 2
    assert set(nonzero_indices) == {0, 2**n_qubits - 1}


def test_bell_state_negative_integer_raises():
    with pytest.raises(ValueError, match="non-negative integer"):
        bell_state(-1)


def test_bell_state_too_large_integer_raises():
    with pytest.raises(ValueError, match="more than 20 qubits"):
        bell_state(21)


# ============================================================================
# Type errors
# ============================================================================

@pytest.mark.parametrize("bad_identifier", [None, 1.5, [], {}, np.array([1])])
def test_bell_state_invalid_type_raises(bad_identifier):
    with pytest.raises(ValueError, match="string or an integer"):
        bell_state(bad_identifier)
