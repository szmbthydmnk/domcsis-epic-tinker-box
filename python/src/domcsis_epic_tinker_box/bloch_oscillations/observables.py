"""
Observable construction for the Bloch-oscillations simulation.

This module builds the ``SparsePauliOp`` lists used to extract physical
observables from the simulation output:

* **Site magnetisations** ⟨Z_j⟩ for j = 0, …, L-1.
* **Connected two-point correlators** ⟨Z_c Z_j⟩ − ⟨Z_c⟩⟨Z_j⟩  where c is
  the centre site index.

All functions are pure (no side effects, no I/O) and have no dependency on
the Qiskit simulator stack beyond ``qiskit.quantum_info.SparsePauliOp``.
"""

from __future__ import annotations

from qiskit.quantum_info import SparsePauliOp  # type: ignore[import-untyped]

from .model import ModelParams, center_index


def magnetisation_observables(params: ModelParams) -> list[SparsePauliOp]:
    """Build one ``SparsePauliOp`` per site for measuring ⟨Z_j⟩.

    The observable for site *j* is the Pauli string with ``Z`` at position
    *j* and ``I`` everywhere else.

    Args:
        params: Model parameters providing the chain length ``L``.

    Returns:
        A list of ``L`` ``SparsePauliOp`` objects, one per site.
    """
    observables: list[SparsePauliOp] = []
    for j in range(params.L):
        # Build 'I…IZI…I' with Z at position j.
        chars = ["I"] * params.L
        chars[j] = "Z"
        observables.append(SparsePauliOp("".join(chars)))
    return observables


def correlator_observables(params: ModelParams) -> list[SparsePauliOp]:
    """Build one ``SparsePauliOp`` per site for measuring ⟨Z_c Z_j⟩.

    The observable for site *j* is the Pauli string with ``Z`` at both the
    centre site *c* and position *j*.  At the centre site itself (j == c)
    the product Z_c Z_c = I, so the observable collapses to the all-identity
    string (expectation value ≡ 1 before subtracting the disconnected part).

    The connected correlator is computed by the caller as::

        corr_connected[j] = ⟨Z_c Z_j⟩ − ⟨Z_c⟩ · ⟨Z_j⟩

    Args:
        params: Model parameters providing the chain length ``L``.

    Returns:
        A list of ``L`` ``SparsePauliOp`` objects.
    """
    c = center_index(params)
    observables: list[SparsePauliOp] = []
    for j in range(params.L):
        chars = ["I"] * params.L
        if j != c:
            chars[c] = "Z"
            chars[j] = "Z"
        # When j == c: Z_c * Z_c = I, so chars stays all-'I'.
        observables.append(SparsePauliOp("".join(chars)))
    return observables
