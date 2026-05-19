from __future__ import annotations

import numpy as np
import pytest

from domcsis_epic_tinker_box.quantum_state.quantum_state import QuantumState


# ============================================================================
# Shared fixtures
# ============================================================================

@pytest.fixture
def ket_0() -> np.ndarray:
    return np.array([1.0, 0.0], dtype=complex)

@pytest.fixture
def ket_plus() -> np.ndarray:
    return np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)

@pytest.fixture
def ket_bell() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2)

@pytest.fixture
def rho_pure_0(ket_0) -> np.ndarray:
    return np.outer(ket_0, ket_0.conj())

@pytest.fixture
def rho_pure_plus(ket_plus) -> np.ndarray:
    return np.outer(ket_plus, ket_plus.conj())

@pytest.fixture
def rho_mixed() -> np.ndarray:
    return np.array([[0.5, 0.0], [0.0, 0.5]], dtype=complex)

@pytest.fixture
def rho_vec_pure_0(rho_pure_0) -> np.ndarray:
    return rho_pure_0.flatten(order="C")

@pytest.fixture
def rho_vec_mixed(rho_mixed) -> np.ndarray:
    return rho_mixed.flatten(order="C")


# ============================================================================
# from_vector — construction
# ============================================================================

def test_from_vector_state_type(ket_0) -> None:
    s = QuantumState.from_vector(ket_0)
    assert s.state_type == "vector"

def test_from_vector_n_qubits_single(ket_0) -> None:
    s = QuantumState.from_vector(ket_0)
    assert s.n_qubits == 1

def test_from_vector_n_qubits_two(ket_bell) -> None:
    s = QuantumState.from_vector(ket_bell)
    assert s.n_qubits == 2

def test_from_vector_stores_complex(ket_0) -> None:
    s = QuantumState.from_vector(ket_0)
    assert s.state.dtype == complex

def test_from_vector_values_preserved(ket_plus) -> None:
    s = QuantumState.from_vector(ket_plus)
    assert np.allclose(s.state, ket_plus)

def test_from_vector_not_1d_raises() -> None:
    with pytest.raises(ValueError, match="1D"):
        QuantumState.from_vector(np.array([[1.0, 0.0], [0.0, 0.0]]))

def test_from_vector_not_power_of_two_raises() -> None:
    with pytest.raises(ValueError):
        QuantumState.from_vector(np.array([1.0, 0.0, 0.0]) / np.sqrt(1))

def test_from_vector_unnormalised_raises() -> None:
    with pytest.raises(ValueError, match="normalised"):
        QuantumState.from_vector(np.array([1.0, 1.0], dtype=complex))

def test_from_vector_accepts_real_input() -> None:
    psi = np.array([1.0, 0.0])
    s = QuantumState.from_vector(psi)
    assert s.state.dtype == complex


# ============================================================================
# from_density_matrix — construction
# ============================================================================

def test_from_density_matrix_state_type(rho_pure_0) -> None:
    s = QuantumState.from_density_matrix(rho_pure_0)
    assert s.state_type == "density_matrix"

def test_from_density_matrix_n_qubits(rho_pure_0) -> None:
    s = QuantumState.from_density_matrix(rho_pure_0)
    assert s.n_qubits == 1

def test_from_density_matrix_stores_complex(rho_pure_0) -> None:
    s = QuantumState.from_density_matrix(rho_pure_0)
    assert s.state.dtype == complex

def test_from_density_matrix_values_preserved(rho_pure_0) -> None:
    s = QuantumState.from_density_matrix(rho_pure_0)
    assert np.allclose(s.state, rho_pure_0)

def test_from_density_matrix_mixed_accepted(rho_mixed) -> None:
    s = QuantumState.from_density_matrix(rho_mixed)
    assert s.state_type == "density_matrix"

def test_from_density_matrix_not_square_raises() -> None:
    with pytest.raises(ValueError):
        QuantumState.from_density_matrix(np.eye(2, 4, dtype=complex))

def test_from_density_matrix_not_hermitian_raises() -> None:
    rho = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=complex)
    with pytest.raises(ValueError, match="Hermitian"):
        QuantumState.from_density_matrix(rho)

def test_from_density_matrix_unnormalised_raises() -> None:
    rho = np.array([[0.6, 0.0], [0.0, 0.6]], dtype=complex)
    with pytest.raises(ValueError, match="normalised"):
        QuantumState.from_density_matrix(rho)

def test_from_density_matrix_not_power_of_two_raises() -> None:
    rho = np.eye(3, dtype=complex) / 3
    with pytest.raises(ValueError):
        QuantumState.from_density_matrix(rho)


# ============================================================================
# from_vectorized_density_matrix — construction
# ============================================================================

def test_from_vectorized_state_type(rho_vec_pure_0) -> None:
    s = QuantumState.from_vectorized_density_matrix(rho_vec_pure_0)
    assert s.state_type == "vectorized_density_matrix"

def test_from_vectorized_n_qubits(rho_vec_pure_0) -> None:
    s = QuantumState.from_vectorized_density_matrix(rho_vec_pure_0)
    assert s.n_qubits == 1

def test_from_vectorized_values_preserved(rho_vec_pure_0) -> None:
    s = QuantumState.from_vectorized_density_matrix(rho_vec_pure_0)
    assert np.allclose(s.state, rho_vec_pure_0)

def test_from_vectorized_mixed_accepted(rho_vec_mixed) -> None:
    s = QuantumState.from_vectorized_density_matrix(rho_vec_mixed)
    assert s.state_type == "vectorized_density_matrix"

def test_from_vectorized_not_1d_raises() -> None:
    with pytest.raises(ValueError, match="1D"):
        QuantumState.from_vectorized_density_matrix(np.eye(4, dtype=complex))

def test_from_vectorized_not_hermitian_raises() -> None:
    rho = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=complex)
    with pytest.raises(ValueError, match="Hermitian"):
        QuantumState.from_vectorized_density_matrix(rho.flatten())

def test_from_vectorized_unnormalised_raises() -> None:
    rho = np.array([[0.6, 0.0], [0.0, 0.6]], dtype=complex)
    with pytest.raises(ValueError, match="normalised"):
        QuantumState.from_vectorized_density_matrix(rho.flatten())


# ============================================================================
# dim property
# ============================================================================

def test_dim_single_qubit(ket_0) -> None:
    assert QuantumState.from_vector(ket_0).dim == 2

def test_dim_two_qubits(ket_bell) -> None:
    assert QuantumState.from_vector(ket_bell).dim == 4

def test_dim_from_density_matrix(rho_pure_0) -> None:
    assert QuantumState.from_density_matrix(rho_pure_0).dim == 2


# ============================================================================
# vector property
# ============================================================================

def test_vector_from_vector_returns_state(ket_plus) -> None:
    s = QuantumState.from_vector(ket_plus)
    assert np.allclose(s.vector, ket_plus)

def test_vector_from_pure_density_matrix(ket_0, rho_pure_0) -> None:
    s = QuantumState.from_density_matrix(rho_pure_0)
    assert np.allclose(np.abs(s.vector), np.abs(ket_0))

def test_vector_from_pure_vectorized(ket_0, rho_vec_pure_0) -> None:
    s = QuantumState.from_vectorized_density_matrix(rho_vec_pure_0)
    assert np.allclose(np.abs(s.vector), np.abs(ket_0))

def test_vector_from_mixed_density_matrix_raises(rho_mixed) -> None:
    s = QuantumState.from_density_matrix(rho_mixed)
    with pytest.raises(ValueError, match="not pure"):
        _ = s.vector

def test_vector_from_mixed_vectorized_raises(rho_vec_mixed) -> None:
    s = QuantumState.from_vectorized_density_matrix(rho_vec_mixed)
    with pytest.raises(ValueError, match="not pure"):
        _ = s.vector

def test_vector_is_normalised_from_density_matrix(rho_pure_plus) -> None:
    s = QuantumState.from_density_matrix(rho_pure_plus)
    assert np.isclose(np.dot(s.vector.conj(), s.vector).real, 1.0)


# ============================================================================
# density_matrix property
# ============================================================================

def test_density_matrix_from_vector(ket_0, rho_pure_0) -> None:
    s = QuantumState.from_vector(ket_0)
    assert np.allclose(s.density_matrix, rho_pure_0)

def test_density_matrix_is_hermitian_from_vector(ket_plus) -> None:
    s = QuantumState.from_vector(ket_plus)
    rho = s.density_matrix
    assert np.allclose(rho, rho.conj().T)

def test_density_matrix_trace_is_one_from_vector(ket_plus) -> None:
    s = QuantumState.from_vector(ket_plus)
    assert np.isclose(np.trace(s.density_matrix).real, 1.0)

def test_density_matrix_from_density_matrix_input(rho_pure_0) -> None:
    s = QuantumState.from_density_matrix(rho_pure_0)
    assert np.allclose(s.density_matrix, rho_pure_0)

def test_density_matrix_from_vectorized(rho_vec_pure_0, rho_pure_0) -> None:
    s = QuantumState.from_vectorized_density_matrix(rho_vec_pure_0)
    assert np.allclose(s.density_matrix, rho_pure_0)


# ============================================================================
# purity property
# ============================================================================

def test_purity_pure_vector(ket_plus) -> None:
    s = QuantumState.from_vector(ket_plus)
    assert np.isclose(s.purity, 1.0)

def test_purity_pure_density_matrix(rho_pure_0) -> None:
    s = QuantumState.from_density_matrix(rho_pure_0)
    assert np.isclose(s.purity, 1.0)

def test_purity_mixed_density_matrix(rho_mixed) -> None:
    s = QuantumState.from_density_matrix(rho_mixed)
    assert s.purity < 1.0

def test_purity_mixed_is_correct(rho_mixed) -> None:
    s = QuantumState.from_density_matrix(rho_mixed)
    assert np.isclose(s.purity, 0.5)

def test_purity_pure_vectorized(rho_vec_pure_0) -> None:
    s = QuantumState.from_vectorized_density_matrix(rho_vec_pure_0)
    assert np.isclose(s.purity, 1.0)

def test_purity_mixed_vectorized(rho_vec_mixed) -> None:
    s = QuantumState.from_vectorized_density_matrix(rho_vec_mixed)
    assert s.purity < 1.0


# ============================================================================
# Immutability
# ============================================================================

def test_state_is_immutable(ket_0) -> None:
    s = QuantumState.from_vector(ket_0)
    with pytest.raises((AttributeError, TypeError)):
        s.state = ket_0  # type: ignore[misc]

def test_n_qubits_is_immutable(ket_0) -> None:
    s = QuantumState.from_vector(ket_0)
    with pytest.raises((AttributeError, TypeError)):
        s.n_qubits = 2  # type: ignore[misc]


# ============================================================================
# Cross-constructor consistency
# ============================================================================

def test_vector_and_density_matrix_give_same_rho(ket_plus, rho_pure_plus) -> None:
    sv = QuantumState.from_vector(ket_plus)
    sd = QuantumState.from_density_matrix(rho_pure_plus)
    assert np.allclose(sv.density_matrix, sd.density_matrix)

def test_density_matrix_and_vectorized_give_same_purity(rho_pure_0, rho_vec_pure_0) -> None:
    sd = QuantumState.from_density_matrix(rho_pure_0)
    sv = QuantumState.from_vectorized_density_matrix(rho_vec_pure_0)
    assert np.isclose(sd.purity, sv.purity)

def test_all_three_constructors_same_n_qubits(ket_0, rho_pure_0, rho_vec_pure_0) -> None:
    s1 = QuantumState.from_vector(ket_0)
    s2 = QuantumState.from_density_matrix(rho_pure_0)
    s3 = QuantumState.from_vectorized_density_matrix(rho_vec_pure_0)
    assert s1.n_qubits == s2.n_qubits == s3.n_qubits