import numpy as np
import pytest
from scipy.sparse import csr_matrix

from domcsis_epic_tinker_box.pauli_algebra import generate_all_pauli_strings, pauli_matrix


# =============================================================================
# generate_all_pauli_strings
# =============================================================================

def test_single_qubit_length() -> None:
    assert len(generate_all_pauli_strings(1)) == 4


def test_single_qubit_values() -> None:
    assert set(generate_all_pauli_strings(1)) == {"I", "X", "Y", "Z"}


def test_two_qubit_length() -> None:
    assert len(generate_all_pauli_strings(2)) == 16


def test_invalid_input() -> None:
    with pytest.raises(ValueError):
        generate_all_pauli_strings(0)


# =============================================================================
# pauli_matrix
# =============================================================================

def test_returns_csr_matrix() -> None:
    assert isinstance(pauli_matrix("I"), csr_matrix)


def test_shape_is_2x2() -> None:
    for label in ("I", "X", "Y", "Z"):
        assert pauli_matrix(label).shape == (2, 2)


def test_pauli_I_is_identity() -> None:
    assert np.allclose(pauli_matrix("I").toarray(), np.eye(2))


def test_pauli_X_values() -> None:
    expected = np.array([[0, 1], [1, 0]], dtype=complex)
    assert np.allclose(pauli_matrix("X").toarray(), expected)


def test_pauli_Y_values() -> None:
    expected = np.array([[0, -1j], [1j, 0]], dtype=complex)
    assert np.allclose(pauli_matrix("Y").toarray(), expected)


def test_pauli_Z_values() -> None:
    expected = np.array([[1, 0], [0, -1]], dtype=complex)
    assert np.allclose(pauli_matrix("Z").toarray(), expected)


def test_lowercase_labels_match_uppercase() -> None:
    for upper, lower in zip("IXYZ", "ixyz"):
        assert np.allclose(
            pauli_matrix(upper).toarray(),
            pauli_matrix(lower).toarray(),
        )


def test_integer_labels_match_string() -> None:
    for idx, label in enumerate("IXYZ"):
        assert np.allclose(
            pauli_matrix(idx).toarray(),
            pauli_matrix(label).toarray(),
        )


def test_paulis_are_hermitian() -> None:
    for label in ("X", "Y", "Z"):
        m = pauli_matrix(label).toarray()
        assert np.allclose(m, m.conj().T, atol=1e-14)


def test_paulis_square_to_identity() -> None:
    for label in ("X", "Y", "Z"):
        m = pauli_matrix(label).toarray()
        assert np.allclose(m @ m, np.eye(2), atol=1e-14)


def test_invalid_string_raises() -> None:
    with pytest.raises(ValueError):
        pauli_matrix("A")


def test_multi_char_string_raises() -> None:
    with pytest.raises(ValueError):
        pauli_matrix("XY")


def test_invalid_integer_raises() -> None:
    with pytest.raises(ValueError):
        pauli_matrix(4)