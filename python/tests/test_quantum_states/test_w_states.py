from __future__ import annotations

import numpy as np
import pytest

from domcsis_epic_tinker_box.quantum_state.w_states import w_state


def test_w_state_one_qubit():
    state = w_state(1)
    expected = np.array([0, 1], dtype=complex)
    assert np.allclose(state.vector, expected)


def test_w_state_two_qubits():
    state = w_state(2)
    expected = np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0], dtype=complex)
    assert np.allclose(state.vector, expected)


def test_w_state_three_qubits():
    state = w_state(3)
    expected = np.array(
        [0, 1 / np.sqrt(3), 1 / np.sqrt(3), 0, 1 / np.sqrt(3), 0, 0, 0],
        dtype=complex,
    )
    assert np.allclose(state.vector, expected)


@pytest.mark.parametrize("no_qubits", [1, 2, 3, 4, 5])
def test_w_state_is_normalized(no_qubits):
    state = w_state(no_qubits)
    norm = np.linalg.norm(state.vector)
    assert norm == pytest.approx(1.0)


@pytest.mark.parametrize("no_qubits", [2, 3, 4, 5])
def test_w_state_has_exactly_no_qubits_nonzero_entries(no_qubits):
    state = w_state(no_qubits)
    nonzero_indices = np.flatnonzero(np.abs(state.vector) > 1e-12)

    assert len(nonzero_indices) == no_qubits
    assert np.allclose(
        state.vector[nonzero_indices],
        np.full(no_qubits, 1 / np.sqrt(no_qubits), dtype=complex),
    )


@pytest.mark.parametrize("no_qubits", [2, 3, 4, 5])
def test_w_state_support_is_single_excitation_basis_states(no_qubits):
    state = w_state(no_qubits)
    nonzero_indices = set(np.flatnonzero(np.abs(state.vector) > 1e-12))
    expected_indices = {1 << i for i in range(no_qubits)}

    assert nonzero_indices == expected_indices


@pytest.mark.parametrize("bad_no_qubits", [0, -1, -5])
def test_w_state_raises_for_nonpositive_qubits(bad_no_qubits):
    with pytest.raises(ValueError, match="positive integer"):
        w_state(bad_no_qubits)


def test_w_state_raises_for_too_many_qubits() -> None:
    with pytest.raises(ValueError, match="more than 20 qubits"):
        w_state(21)
