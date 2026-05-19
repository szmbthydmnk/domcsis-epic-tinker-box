from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from domcsis_epic_tinker_box.pauli_algebra import PauliOperators, pauli_matrix

# =============================================================================
# PauliOperators.all — construction and basic invariants
# =============================================================================

def test_all_single_qubit_n_qubits() -> None:
    p = PauliOperators.all(1)
    assert p.n_qubits == 1


def test_all_single_qubit_length() -> None:
    p = PauliOperators.all(1)
    assert len(p) == 4


def test_all_two_qubit_length() -> None:
    p = PauliOperators.all(2)
    assert len(p) == 16


def test_all_three_qubit_length() -> None:
    p = PauliOperators.all(3)
    assert len(p) == 4**3


def test_all_labels_are_strings() -> None:
    p = PauliOperators.all(1)
    assert all(isinstance(label, str) for label in p.labels)


def test_all_matrices_are_csr() -> None:
    p = PauliOperators.all(1)
    assert all(isinstance(m, csr_matrix) for m in p.matrices)


def test_all_labels_and_matrices_same_length() -> None:
    p = PauliOperators.all(2)
    assert len(p.labels) == len(p.matrices)


def test_all_single_qubit_labels_correct() -> None:
    p = PauliOperators.all(1)
    assert set(p.labels) == {"I", "X", "Y", "Z"}


def test_all_two_qubit_contains_II() -> None:
    p = PauliOperators.all(2)
    assert "II" in p.labels


def test_all_matrix_shape_single_qubit() -> None:
    p = PauliOperators.all(1)
    assert all(m.shape == (2, 2) for m in p.matrices)


def test_all_matrix_shape_two_qubits() -> None:
    p = PauliOperators.all(2)
    assert all(m.shape == (4, 4) for m in p.matrices)


def test_all_II_is_identity() -> None:
    p = PauliOperators.all(2)
    idx = p.labels.index("II")
    assert np.allclose(p.matrices[idx].toarray(), np.eye(4))


def test_all_invalid_n_qubits_raises() -> None:
    with pytest.raises(ValueError):
        PauliOperators.all(0)


# =============================================================================
# PauliOperators.from_strings — construction and basic invariants
# =============================================================================

def test_from_strings_returns_correct_type() -> None:
    p = PauliOperators.from_strings({"XI", "IZ"})
    assert isinstance(p, PauliOperators)


def test_from_strings_length() -> None:
    p = PauliOperators.from_strings({"XI", "IZ", "YY"})
    assert len(p) == 3


def test_from_strings_labels_match_input() -> None:
    p = PauliOperators.from_strings({"XI", "IZ"})
    assert set(p.labels) == {"XI", "IZ"}


def test_from_strings_matrices_are_csr() -> None:
    p = PauliOperators.from_strings({"XI", "IZ"})
    assert all(isinstance(m, csr_matrix) for m in p.matrices)


def test_from_strings_n_qubits_inferred() -> None:
    p = PauliOperators.from_strings({"XI", "IZ"})
    assert p.n_qubits == 2


def test_from_strings_invalid_character_raises() -> None:
    with pytest.raises(ValueError):
        PauliOperators.from_strings({"XA"})


def test_from_strings_XI_matrix_correct() -> None:
    p = PauliOperators.from_strings({"XI"})
    expected = np.kron(
        pauli_matrix("X").toarray(),
        pauli_matrix("I").toarray(),
    )
    idx = p.labels.index("XI")
    assert np.allclose(p.matrices[idx].toarray(), expected)


# =============================================================================
# Consistency: from_strings vs. PauliOperators.all
# =============================================================================

def test_from_strings_matches_all_for_same_operators() -> None:
    full = PauliOperators.all(2)
    subset = PauliOperators.from_strings({"XI", "YZ", "II"})
    full_dict = full.as_dict()
    for label, matrix in subset.items():
        assert np.allclose(matrix.toarray(), full_dict[label].toarray())


# =============================================================================
# PauliOperators.items — iteration
# =============================================================================

def test_items_yields_label_matrix_pairs() -> None:
    p = PauliOperators.all(1)
    for label, matrix in p.items():
        assert isinstance(label, str)
        assert isinstance(matrix, csr_matrix)


def test_items_count_matches_len() -> None:
    p = PauliOperators.all(2)
    assert sum(1 for _ in p.items()) == len(p)


def test_items_label_order_consistent_with_labels_tuple() -> None:
    p = PauliOperators.all(2)
    for (label, _), expected_label in zip(p.items(), p.labels):
        assert label == expected_label


# =============================================================================
# PauliOperators.as_dict
# =============================================================================

def test_as_dict_returns_dict() -> None:
    p = PauliOperators.all(1)
    assert isinstance(p.as_dict(), dict)


def test_as_dict_keys_match_labels() -> None:
    p = PauliOperators.all(1)
    assert set(p.as_dict().keys()) == set(p.labels)


def test_as_dict_values_match_matrices() -> None:
    p = PauliOperators.all(1)
    d = p.as_dict()
    for label, matrix in zip(p.labels, p.matrices):
        assert np.allclose(d[label].toarray(), matrix.toarray())


# =============================================================================
# Immutability (frozen=True)
# =============================================================================

def test_labels_are_immutable() -> None:
    p = PauliOperators.all(1)
    with pytest.raises((AttributeError, TypeError)):
        p.labels = ("I", "X", "Y", "Z")  # type: ignore[misc]


def test_matrices_are_immutable() -> None:
    p = PauliOperators.all(1)
    with pytest.raises((AttributeError, TypeError)):
        p.matrices = p.matrices  # type: ignore[misc]


# =============================================================================
# Physics: Pauli operators are Hermitian and square to identity (spot-checks)
# =============================================================================

def test_single_qubit_paulis_are_hermitian() -> None:
    p = PauliOperators.all(1)
    for label, matrix in p.items():
        if label == "I":
            continue
        m = matrix.toarray()
        assert np.allclose(m, m.conj().T, atol=1e-14), f"{label} not Hermitian"


def test_single_qubit_paulis_square_to_identity() -> None:
    p = PauliOperators.all(1)
    for label, matrix in p.items():
        m = matrix.toarray()
        assert np.allclose(m @ m, np.eye(2), atol=1e-14), f"{label}^2 != I"


def test_two_qubit_paulis_are_hermitian() -> None:
    p = PauliOperators.from_strings({"XY", "ZI", "YZ"})
    for label, matrix in p.items():
        m = matrix.toarray()
        assert np.allclose(m, m.conj().T, atol=1e-14), f"{label} not Hermitian"


def test_two_qubit_paulis_square_to_identity() -> None:
    p = PauliOperators.from_strings({"XY", "ZI", "YZ"})
    for label, matrix in p.items():
        m = matrix.toarray()
        assert np.allclose(m @ m, np.eye(4), atol=1e-14), f"{label}^2 != I"