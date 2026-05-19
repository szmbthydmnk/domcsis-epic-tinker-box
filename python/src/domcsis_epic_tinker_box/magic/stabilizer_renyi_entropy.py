"""
This module implements the Stabilizer Rényi Entropy (SRE) for quantum states.
"""


from __future__ import annotations

import warnings
import numpy as np
from typing import overload, Literal

from ..pauli_algebra.pauli_operators import PauliOperators
from ..quantum_state.quantum_state import QuantumState


# ============================================================================
# Overloads — public entry point
# ============================================================================

@overload
def stabilizer_renyi_entropy(
    state: QuantumState,
    paulis: PauliOperators,
    alpha: int | float = ...,
    pauli_spectrum: Literal[False] = ...,
) -> float: ...

@overload
def stabilizer_renyi_entropy(
    state: QuantumState,
    paulis: PauliOperators,
    alpha: int | float = ...,
    pauli_spectrum: Literal[True] = ...,
) -> tuple[float, dict[str, float]]: ...


# ============================================================================
# Public entry point — validated
# ============================================================================

def stabilizer_renyi_entropy(
    state: QuantumState,
    paulis: PauliOperators,
    alpha: int | float = 2,
    pauli_spectrum: bool = False,
) -> float | tuple[float, dict[str, float]]:
    """
    Compute the alpha-Stabilizer Rényi Entropy (SRE) for a pure quantum state.

    Validated entry point. For hot loops over many states with fixed paulis
    and alpha, validate once and use _stabilizer_renyi_entropy_unchecked
    directly to avoid redundant checks.

    The SRE is defined as:
        M_alpha(|psi>) = (1 / (1 - alpha)) * log( sum_{P in P_n} Xi(P)^alpha )
    where the characteristic function is:
        Xi(P) = |<psi|P|psi>|^2 / 2^n

    Special values of alpha:
        alpha=2 (default): Linear Magic M_2, the most common SRE variant.
        alpha=1:           Linear Magic M_1; a warning is issued.
        alpha>0, non-int:  general alpha-SRE; a warning is issued.
        alpha<=0:          undefined; raises ValueError.

    Args:
        state:
            Pure quantum state as a QuantumState instance. Normalisation,
            dimension, and purity are guaranteed by construction.
        paulis:
            PauliOperators instance. Must match state.n_qubits.
        alpha:
            Rényi order. Must be positive. Default is 2 (Linear Magic M_2).
        pauli_spectrum:
            If False (default), return only the SRE value.
            If True, return (SRE, spectrum) where spectrum maps each Pauli
            label to its characteristic function value Xi(P).

    Returns:
        float:
            The SRE value (if pauli_spectrum=False).
        tuple[float, dict[str, float]]:
            (SRE, Xi spectrum) if pauli_spectrum=True.

    Raises:
        ValueError: If state.n_qubits != paulis.n_qubits.
        ValueError: If alpha <= 0.
    """
    _validate_alpha(alpha)
    _validate_compatible(state, paulis)
    return _stabilizer_renyi_entropy_unchecked(state, paulis, alpha, pauli_spectrum)


# ============================================================================
# Fast path — unchecked
# ============================================================================

def _stabilizer_renyi_entropy_unchecked(
    state: QuantumState,
    paulis: PauliOperators,
    alpha: int | float = 2,
    pauli_spectrum: bool = False,
) -> float | tuple[float, dict[str, float]]:
    """
    Fast path SRE computation — no validation.

    Caller is responsible for ensuring:
        - alpha > 0
        - state.n_qubits == paulis.n_qubits
        - state is a valid QuantumState (normalised, pure)

    Intended for hot loops over many states with fixed paulis and alpha.
    Validate once with the public entry point, then call this directly:

        _validate_alpha(alpha)
        _validate_compatible(states[0], paulis)
        results = [
            _stabilizer_renyi_entropy_unchecked(state, paulis, alpha)
            for state in states
        ]

    Args:
        state:        QuantumState instance (unchecked).
        paulis:       PauliOperators instance (unchecked).
        alpha:        Rényi order (unchecked, assumed positive).
        pauli_spectrum: See stabilizer_renyi_entropy.

    Returns:
        float | tuple[float, dict[str, float]]: See stabilizer_renyi_entropy.
    """
    xi = _characteristic_function(state.vector, paulis)
    sre = _compute_sre(xi, alpha)
    if pauli_spectrum:
        return sre, dict(zip(paulis.labels, xi.tolist()))
    return sre


# ============================================================================
# Private helpers
# ============================================================================

def _validate_alpha(
    alpha: int | float,
) -> None:
    """
    Validate and characterise the Rényi order alpha.

    Rules:
        alpha <= 0:          raises ValueError (undefined).
        alpha == 1:          Linear Magic M_1; issues UserWarning.
        alpha == 2:          Linear Magic M_2; valid, no warning.
        alpha > 0, non-int:  valid; issues UserWarning.
        other positive int:  general alpha-SRE; valid, no warning.
    """
    if alpha <= 0:
        raise ValueError(
            f"alpha must be positive, got {alpha}. "
            "SRE is not defined for alpha <= 0."
        )
    if alpha == 1:
        warnings.warn(
            "alpha=1: computing Linear Magic M_1. "
            "Note that alpha -> 1 also recovers the Shannon entropy in the limit; "
            "here the finite formula is used directly.",
            UserWarning,
            stacklevel=3,
        )
    if not isinstance(alpha, int) and alpha != int(alpha):
        warnings.warn(
            f"alpha={alpha} is not an integer. "
            "Computing the general alpha-Rényi entropy. "
            "Ensure this is intentional.",
            UserWarning,
            stacklevel=3,
        )


def _validate_compatible(
    state: QuantumState, 
    paulis: PauliOperators,
) -> None:
    if state.n_qubits != paulis.n_qubits:
        raise ValueError(
            f"state.n_qubits={state.n_qubits} does not match "
            f"paulis.n_qubits={paulis.n_qubits}."
        )


def _characteristic_function(
    psi: np.ndarray,
    paulis: PauliOperators,
) -> np.ndarray:
    """
    Compute Xi(P) = |<psi|P|psi>|^2 / 2^n for all P in paulis.

    Args:
        psi:    Normalised state vector of shape (2^n,), from QuantumState.vector.
        paulis: PauliOperators instance for n qubits.

    Returns:
        np.ndarray of shape (4^n,) with Xi values in paulis label order.
    """
    norm = 2 ** paulis.n_qubits
    return np.array(
        [abs(psi.conj() @ (P @ psi)) ** 2 / norm for _, P in paulis.items()],
        dtype=float,
    )


def _compute_sre(
    xi: np.ndarray,
    alpha: int | float,
) -> float:
    """
    Compute M_alpha = log2( sum(Xi^alpha) ) / (1 - alpha) - log2(d).

    For alpha=1, the formula degenerates (division by zero) and the
    Shannon entropy limit is used instead:
        M_1 = -sum_P Xi(P) * log2(Xi(P)) - log2(d)
    where 0 * log2(0) := 0 by convention.

    Args:
        xi:    np.ndarray of characteristic function values Xi(P).
        alpha: Rényi order (positive).

    Returns:
        float: The SRE value.
    """
    d = xi.size
    if alpha == 1:
        # Shannon entropy limit: 0 * log2(0) = 0 by convention
        with np.errstate(divide="ignore"):
            log_xi = np.where(xi > 0, np.log2(xi), 0.0)
        return float(-np.dot(xi, log_xi) - np.log2(d))
    
    return float((np.log2(np.sum(xi ** alpha)) / (1 - alpha)) - np.log2(d))