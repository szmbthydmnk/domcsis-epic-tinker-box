"""
Observable builders for the Bloch-oscillations simulation.

This module constructs the ``SparsePauliOp`` objects used to measure:

* **Site magnetisations**  ⟨Z_j⟩  for each qubit ``j``.
* **Connected two-point correlators**  ⟨Z_c Z_j⟩ − ⟨Z_c⟩⟨Z_j⟩  where
  ``c`` is the centre site returned by :func:`~.model.center_index`.

Qiskit's ``SparsePauliOp`` uses a *little-endian* string convention: the
rightmost character in the string corresponds to qubit 0.  The string
construction here follows that convention explicitly.

No I/O and no circuit construction live in this module.
"""

from __future__ import annotations

from qiskit.quantum_info import SparsePauliOp  # type: ignore[import-untyped]

from .model import ModelParams, center_index


# ============================================================================
# Magnetisation observables
# ============================================================================


def magnetisation_observables(params: ModelParams) -> list[SparsePauliOp]:
    """Build the list of single-site Z observables.

    Returns one ``SparsePauliOp`` per lattice site ``j``, representing the
    Pauli-Z operator acting on site ``j`` and the identity on all other
    sites.  The expectation value gives the local magnetisation ⟨Z_j⟩.

    Args:
        params: Model parameters (only ``L`` is used).

    Returns:
        A list of length ``L`` of ``SparsePauliOp`` objects, ordered by
        site index (index 0 → leftmost qubit).
    """
    observables: list[SparsePauliOp] = []

    for qubit in range(params.L):
        # Start from an all-identity string of length L.
        label = list("I" * params.L)
        # Insert a Z at position ``qubit`` (0-indexed from the left).
        label[qubit] = "Z"
        observables.append(SparsePauliOp("".join(label)))

    return observables


# ============================================================================
# Connected correlator observables
# ============================================================================


def correlator_observables(params: ModelParams) -> list[SparsePauliOp]:
    """Build the list of two-site Z_c Z_j observables.

    Returns one ``SparsePauliOp`` per lattice site ``j``, representing the
    product Z_c Z_j where ``c`` is the centre site.  When ``j == c`` the
    product Z_c Z_c = I and the corresponding observable is all identities
    (so that ``⟨Z_c Z_c⟩ − ⟨Z_c⟩² = 1 − ⟨Z_c⟩²`` simplifies correctly
    in the post-processing step).

    Args:
        params: Model parameters (only ``L`` is used).

    Returns:
        A list of length ``L`` of ``SparsePauliOp`` objects.
    """
    c = center_index(params)  # Index of the centre (reference) site.
    observables: list[SparsePauliOp] = []

    for qubit in range(params.L):
        if qubit == c:
            # Z_c * Z_c = I  →  return the all-identity operator.
            label = "I" * params.L
        else:
            # Place a Z at the centre and at qubit j.
            chars = list("I" * params.L)
            chars[c] = "Z"
            chars[qubit] = "Z"
            label = "".join(chars)

        observables.append(SparsePauliOp(label))

    return observables
