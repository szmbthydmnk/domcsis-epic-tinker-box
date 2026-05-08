from domcsis_epic_tinker_box.pauli_algebra.pauli_operators import generate_pauli_operators
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


def test_three_qubit_length() -> None:
    assert len(generate_all_pauli_strings(3)) == 4**3


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


# =============================================================================
# generate_pauli_operators — int input (list)
# =============================================================================

def test_int_input_returns_list() -> None:
    result = generate_pauli_operators(1)
    assert isinstance(result, list)


def test_int_input_correct_count() -> None:
    assert len(generate_pauli_operators(1)) == 4
    assert len(generate_pauli_operators(2)) == 16


def test_int_input_list_contains_csr_matrices() -> None:
    assert all(isinstance(m, csr_matrix) for m in generate_pauli_operators(1))  # type: ignore[union-attr]


def test_int_input_correct_shape() -> None:
    ops = generate_pauli_operators(2)
    assert all(m.shape == (4, 4) for m in ops)  # type: ignore[union-attr]


def test_int_input_first_operator_is_II() -> None:
    ops = generate_pauli_operators(2)
    assert isinstance(ops, list)
    assert np.allclose(ops[0].toarray(), np.eye(4))


# =============================================================================
# generate_pauli_operators — int input (dict)
# =============================================================================

def test_int_input_as_dict_returns_dict() -> None:
    result = generate_pauli_operators(1, as_dict=True)
    assert isinstance(result, dict)


def test_int_input_as_dict_correct_keys() -> None:
    result = generate_pauli_operators(1, as_dict=True)
    assert set(result.keys()) == {"I", "X", "Y", "Z"}  # type: ignore[union-attr]


def test_int_input_as_dict_values_are_csr_matrices() -> None:
    result = generate_pauli_operators(1, as_dict=True)
    assert all(isinstance(m, csr_matrix) for m in result.values())  # type: ignore[union-attr]


def test_int_input_as_dict_matches_list() -> None:
    ops_list = generate_pauli_operators(1)
    ops_dict = generate_pauli_operators(1, as_dict=True)
    strings = generate_all_pauli_strings(1)
    assert isinstance(ops_list, list)
    assert isinstance(ops_dict, dict)
    for i, s in enumerate(strings):
        assert np.allclose(
            ops_list[i].toarray(),
            ops_dict[s].toarray(),
        )


# =============================================================================
# generate_pauli_operators — set[str] input
# =============================================================================

def test_set_input_returns_dict() -> None:
    result = generate_pauli_operators({"XI", "IZ"})
    assert isinstance(result, dict)


def test_set_input_correct_keys() -> None:
    result = generate_pauli_operators({"XI", "IZ"})
    assert set(result.keys()) == {"XI", "IZ"}  # type: ignore[union-attr]


def test_set_input_correct_shape() -> None:
    result = generate_pauli_operators({"XI", "IZ"})
    assert all(m.shape == (4, 4) for m in result.values())  # type: ignore[union-attr]


def test_set_input_XI_correct_values() -> None:
    result = generate_pauli_operators({"XI"})
    assert isinstance(result, dict)
    expected = np.kron(
        pauli_matrix("X").toarray(),
        pauli_matrix("I").toarray(),
    )
    assert np.allclose(result["XI"].toarray(), expected)


def test_set_input_matches_full_generation() -> None:
    full = generate_pauli_operators(2, as_dict=True)
    subset = generate_pauli_operators({"XI", "YZ", "II"})
    assert isinstance(full, dict)
    assert isinstance(subset, dict)
    for s in ("XI", "YZ", "II"):
        assert np.allclose(
            full[s].toarray(),
            subset[s].toarray(),
        )


def test_set_input_invalid_string_raises() -> None:
    with pytest.raises(ValueError):
        generate_pauli_operators({"XA"})


def test_int_input_invalid_raises() -> None:
    with pytest.raises(ValueError):
        generate_pauli_operators(0)