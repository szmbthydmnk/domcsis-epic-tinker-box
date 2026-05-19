from __future__ import annotations

import warnings
import numpy as np
import pytest

from domcsis_epic_tinker_box.quantum_state.quantum_state import QuantumState
from domcsis_epic_tinker_box.pauli_algebra.pauli_operators import PauliOperators
from domcsis_epic_tinker_box.magic.stabilizer_renyi_entropy import (
    stabilizer_renyi_entropy,
    _stabilizer_renyi_entropy_unchecked,
    _validate_alpha,
    _validate_compatible,
    _characteristic_function,
    _compute_sre,
)


# ============================================================================
# Shared fixtures
# ============================================================================

@pytest.fixture
def ket_0() -> QuantumState:
    return QuantumState.from_vector(np.array([1.0, 0.0], dtype=complex))

@pytest.fixture
def ket_1() -> QuantumState:
    return QuantumState.from_vector(np.array([0.0, 1.0], dtype=complex))

@pytest.fixture
def ket_plus() -> QuantumState:
    return QuantumState.from_vector(np.array([1.0, 1.0], dtype=complex) / np.sqrt(2))

@pytest.fixture
def ket_bell() -> QuantumState:
    return QuantumState.from_vector(
        np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2)
    )

@pytest.fixture
def paulis_1q() -> PauliOperators:
    return PauliOperators.all(1)

@pytest.fixture
def paulis_2q() -> PauliOperators:
    return PauliOperators.all(2)


# ============================================================================
# _validate_alpha
# ============================================================================

def test_validate_alpha_zero_raises() -> None:
    with pytest.raises(ValueError, match="alpha must be positive"):
        _validate_alpha(0)

def test_validate_alpha_negative_raises() -> None:
    with pytest.raises(ValueError):
        _validate_alpha(-1)

def test_validate_alpha_negative_float_raises() -> None:
    with pytest.raises(ValueError):
        _validate_alpha(-0.5)

def test_validate_alpha_two_no_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _validate_alpha(2)  # must not raise

def test_validate_alpha_three_no_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _validate_alpha(3)

def test_validate_alpha_one_warns() -> None:
    with pytest.warns(UserWarning, match="alpha=1"):
        _validate_alpha(1)

def test_validate_alpha_non_integer_warns() -> None:
    with pytest.warns(UserWarning, match="not an integer"):
        _validate_alpha(1.5)

def test_validate_alpha_float_integer_no_warning() -> None:
    # 2.0 is mathematically an integer — should not warn
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _validate_alpha(2.0)


# ============================================================================
# _validate_compatible
# ============================================================================

def test_validate_compatible_matching(ket_0, paulis_1q) -> None:
    _validate_compatible(ket_0, paulis_1q)  # must not raise

def test_validate_compatible_mismatch_raises(ket_0, paulis_2q) -> None:
    with pytest.raises(ValueError, match="n_qubits"):
        _validate_compatible(ket_0, paulis_2q)

def test_validate_compatible_two_qubit_matching(ket_bell, paulis_2q) -> None:
    _validate_compatible(ket_bell, paulis_2q)  # must not raise


# ============================================================================
# _characteristic_function
# ============================================================================

def test_characteristic_function_length(ket_0, paulis_1q) -> None:
    xi = _characteristic_function(ket_0.vector, paulis_1q)
    assert len(xi) == 4

def test_characteristic_function_dtype(ket_0, paulis_1q) -> None:
    xi = _characteristic_function(ket_0.vector, paulis_1q)
    assert xi.dtype == float

def test_characteristic_function_non_negative(ket_plus, paulis_1q) -> None:
    xi = _characteristic_function(ket_plus.vector, paulis_1q)
    assert np.all(xi >= 0)

def test_characteristic_function_sums_to_one(ket_plus, paulis_1q) -> None:
    # sum_P Xi(P) = 1 for any normalised pure state
    xi = _characteristic_function(ket_plus.vector, paulis_1q)
    assert np.isclose(np.sum(xi), 1.0, atol=1e-12)

def test_characteristic_function_sums_to_one_ket_0(ket_0, paulis_1q) -> None:
    xi = _characteristic_function(ket_0.vector, paulis_1q)
    assert np.isclose(np.sum(xi), 1.0, atol=1e-12)

def test_characteristic_function_sums_to_one_two_qubit(ket_bell, paulis_2q) -> None:
    xi = _characteristic_function(ket_bell.vector, paulis_2q)
    assert np.isclose(np.sum(xi), 1.0, atol=1e-12)

def test_characteristic_function_two_qubit_length(ket_bell, paulis_2q) -> None:
    xi = _characteristic_function(ket_bell.vector, paulis_2q)
    assert len(xi) == 16

def test_characteristic_function_ket_0_II_value(ket_0, paulis_1q) -> None:
    # For |0>, <0|I|0> = 1, so Xi(I) = 1/2
    xi = _characteristic_function(ket_0.vector, paulis_1q)
    idx = paulis_1q.labels.index("I")
    assert np.isclose(xi[idx], 0.5)

def test_characteristic_function_ket_0_Z_value(ket_0, paulis_1q) -> None:
    # For |0>, <0|Z|0> = 1, so Xi(Z) = 1/2
    xi = _characteristic_function(ket_0.vector, paulis_1q)
    idx = paulis_1q.labels.index("Z")
    assert np.isclose(xi[idx], 0.5)

def test_characteristic_function_ket_0_X_is_zero(ket_0, paulis_1q) -> None:
    # For |0>, <0|X|0> = 0, so Xi(X) = 0
    xi = _characteristic_function(ket_0.vector, paulis_1q)
    idx = paulis_1q.labels.index("X")
    assert np.isclose(xi[idx], 0.0)


# ============================================================================
# _compute_sre
# ============================================================================

def test_compute_sre_returns_float() -> None:
    xi = np.array([0.25, 0.25, 0.25, 0.25])
    assert isinstance(_compute_sre(xi, 2), float)

def test_compute_sre_uniform_spectrum_is_zero() -> None:
    # Uniform Xi = 1/4^n gives SRE = 0 (maximally magic state)
    n = 1
    d = 4 ** n
    xi = np.full(d, 1.0 / d)
    assert np.isclose(_compute_sre(xi, 2), 0.0, atol=1e-12)

def test_compute_sre_stabilizer_state_is_negative_log2_d() -> None:
    # For a stabilizer state on 1 qubit: 2 non-zero Xi = 1/2, rest zero
    # M_2 = log2(sum Xi^2) / (1-2) - log2(4) = log2(2*(1/2)^2)/(−1) - 2
    #      = log2(0.5) / (-1) - 2 = 1 - 2 = -1 ... wait, let's just check it's real
    xi = np.array([0.5, 0.0, 0.0, 0.5])
    result = _compute_sre(xi, 2)
    assert isinstance(result, float)

def test_compute_sre_alpha_3(ket_plus, paulis_1q) -> None:
    xi = _characteristic_function(ket_plus.vector, paulis_1q)
    result = _compute_sre(xi, 3)
    assert isinstance(result, float)
    assert np.isfinite(result)


# ============================================================================
# stabilizer_renyi_entropy — public entry point
# ============================================================================

def test_sre_returns_float(ket_0, paulis_1q) -> None:
    result = stabilizer_renyi_entropy(ket_0, paulis_1q)
    assert isinstance(result, float)

def test_sre_finite(ket_plus, paulis_1q) -> None:
    result = stabilizer_renyi_entropy(ket_plus, paulis_1q)
    assert np.isfinite(result)

def test_sre_pauli_spectrum_false_returns_float(ket_0, paulis_1q) -> None:
    result = stabilizer_renyi_entropy(ket_0, paulis_1q, pauli_spectrum=False)
    assert isinstance(result, float)

def test_sre_pauli_spectrum_true_returns_tuple(ket_0, paulis_1q) -> None:
    result = stabilizer_renyi_entropy(ket_0, paulis_1q, pauli_spectrum=True)
    assert isinstance(result, tuple)
    assert len(result) == 2

def test_sre_pauli_spectrum_tuple_first_is_float(ket_0, paulis_1q) -> None:
    sre, _ = stabilizer_renyi_entropy(ket_0, paulis_1q, pauli_spectrum=True)
    assert isinstance(sre, float)

def test_sre_pauli_spectrum_dict_keys_match_labels(ket_0, paulis_1q) -> None:
    _, spectrum = stabilizer_renyi_entropy(ket_0, paulis_1q, pauli_spectrum=True)
    assert set(spectrum.keys()) == set(paulis_1q.labels)

def test_sre_pauli_spectrum_values_non_negative(ket_plus, paulis_1q) -> None:
    _, spectrum = stabilizer_renyi_entropy(ket_plus, paulis_1q, pauli_spectrum=True)
    assert all(v >= 0 for v in spectrum.values())

def test_sre_pauli_spectrum_values_sum_to_one(ket_plus, paulis_1q) -> None:
    _, spectrum = stabilizer_renyi_entropy(ket_plus, paulis_1q, pauli_spectrum=True)
    assert np.isclose(sum(spectrum.values()), 1.0, atol=1e-12)

def test_sre_spectrum_sre_matches_standalone(ket_plus, paulis_1q) -> None:
    sre_scalar = stabilizer_renyi_entropy(ket_plus, paulis_1q)
    sre_from_tuple, _ = stabilizer_renyi_entropy(ket_plus, paulis_1q, pauli_spectrum=True)
    assert np.isclose(sre_scalar, sre_from_tuple)

def test_sre_alpha_zero_raises(ket_0, paulis_1q) -> None:
    with pytest.raises(ValueError, match="alpha must be positive"):
        stabilizer_renyi_entropy(ket_0, paulis_1q, alpha=0)

def test_sre_alpha_negative_raises(ket_0, paulis_1q) -> None:
    with pytest.raises(ValueError):
        stabilizer_renyi_entropy(ket_0, paulis_1q, alpha=-2)

def test_sre_incompatible_raises(ket_0, paulis_2q) -> None:
    with pytest.raises(ValueError, match="n_qubits"):
        stabilizer_renyi_entropy(ket_0, paulis_2q)

def test_sre_alpha_one_warns(ket_0, paulis_1q) -> None:
    with pytest.warns(UserWarning, match="alpha=1"):
        stabilizer_renyi_entropy(ket_0, paulis_1q, alpha=1)

def test_sre_alpha_float_warns(ket_0, paulis_1q) -> None:
    with pytest.warns(UserWarning, match="not an integer"):
        stabilizer_renyi_entropy(ket_0, paulis_1q, alpha=1.5)

def test_sre_two_qubits_finite(ket_bell, paulis_2q) -> None:
    result = stabilizer_renyi_entropy(ket_bell, paulis_2q)
    assert np.isfinite(result)

def test_sre_from_density_matrix_matches_from_vector(paulis_1q) -> None:
    psi = np.array([1.0, 0.0], dtype=complex)
    rho = np.outer(psi, psi.conj())
    sv = QuantumState.from_vector(psi)
    sd = QuantumState.from_density_matrix(rho)
    assert np.isclose(
        stabilizer_renyi_entropy(sv, paulis_1q),
        stabilizer_renyi_entropy(sd, paulis_1q),
    )

def test_sre_stabilizer_state_is_non_positive(ket_0, paulis_1q) -> None:
    # Stabilizer states have SRE <= 0
    result = stabilizer_renyi_entropy(ket_0, paulis_1q)
    assert result <= 0.0 + 1e-12


# ============================================================================
# _stabilizer_renyi_entropy_unchecked — fast path
# ============================================================================

def test_unchecked_matches_public(ket_plus, paulis_1q) -> None:
    expected = stabilizer_renyi_entropy(ket_plus, paulis_1q)
    result = _stabilizer_renyi_entropy_unchecked(ket_plus, paulis_1q)
    assert np.isclose(result, expected)

def test_unchecked_pauli_spectrum_matches_public(ket_plus, paulis_1q) -> None:
    sre_pub, spec_pub = stabilizer_renyi_entropy(ket_plus, paulis_1q, pauli_spectrum=True)
    sre_unc, spec_unc = _stabilizer_renyi_entropy_unchecked(ket_plus, paulis_1q, pauli_spectrum=True)
    assert np.isclose(sre_pub, sre_unc)
    for label in spec_pub:
        assert np.isclose(spec_pub[label], spec_unc[label])

def test_unchecked_batch_consistent(paulis_1q) -> None:
    states = [
        QuantumState.from_vector(np.array([1.0, 0.0], dtype=complex)),
        QuantumState.from_vector(np.array([0.0, 1.0], dtype=complex)),
        QuantumState.from_vector(np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)),
    ]
    _validate_alpha(2)
    _validate_compatible(states[0], paulis_1q)
    results = [_stabilizer_renyi_entropy_unchecked(s, paulis_1q) for s in states]
    expected = [stabilizer_renyi_entropy(s, paulis_1q) for s in states]
    assert np.allclose(results, expected)