"""
Data-model definitions for the Bloch-oscillations simulation.

This module is intentionally free of Qiskit imports so that it can be
imported, instantiated, and unit-tested without a full Qiskit installation.
All physics parameters and run-time configuration live here as frozen-style
dataclasses (``replace``-safe, ``asdict``-serialisable).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ============================================================================
# Physics parameters
# ============================================================================


@dataclass
class ModelParams:
    """Physical Hamiltonian parameters for the 1-D Ising chain.

    The Hamiltonian used in the Bloch-oscillation simulation is::

        H = J * sum_{i} Z_i Z_{i+1}  +  ht * sum_i X_i  +  hl * sum_i Z_i

    All fields have physically reasonable defaults that reproduce the
    Bloch-oscillation regime studied in the original notebook.

    Attributes:
        J:           Nearest-neighbour ZZ coupling strength.
        ht:          Transverse (X) field amplitude.
        hl:          Longitudinal (Z) field amplitude.
        dt:          Trotter time step.
        L:           Number of lattice sites (qubits).
        layers_max:  Maximum number of Trotter layers to simulate.
    """

    J: float = 1.0
    ht: float = -0.15
    hl: float = 0.15
    dt: float = 0.25
    L: int = 7
    layers_max: int = 40


# ============================================================================
# Run-time configuration
# ============================================================================


@dataclass
class RunConfig:
    """Simulation run-time configuration.

    Separates execution choices (backend, shots, initial state, Trotter
    decomposition strategy) from the physical model so that the same
    ``ModelParams`` object can be re-used across many configurations.

    Attributes:
        backend_mode:       ``"ideal"`` uses a noiseless ``AerSimulator``;
                            ``"fake"`` builds a density-matrix simulator
                            from a ``FakeBackend`` noise model.
        fake_backend_name:  Name of the fake backend to use when
                            ``backend_mode="fake"``.  Recognised values:
                            ``"brisbane"``, ``"sherbrooke"``, ``"almaden"``.
                            Must be set when ``backend_mode="fake"``.
        initial_state:      Computational-basis product state used as the
                            time-zero wave-function.  ``"all_up"`` prepares
                            |0…0⟩; ``"all_down"`` prepares |1…1⟩.
        use_cnot_zz:        When ``True`` the ZZ gate is decomposed into
                            ``CX – RZ – CX`` explicitly rather than using
                            Qiskit's ``PauliEvolutionGate``.
        shots:              Number of measurement shots per circuit.
        optimization_level: Qiskit transpiler optimisation level (0–3).
        use_parallel_u1:    When ``True`` the single-qubit layer is applied
                            as separate ``RX`` / ``RZ`` sweeps so the
                            transpiler can schedule them in parallel.
        trotter_method:     Ordering of two-qubit bonds in each Trotter
                            layer.  ``"even_odd"`` applies even bonds then
                            odd bonds; ``"odd_even"`` reverses the order;
                            ``"zig_zag"`` sweeps left-to-right then
                            right-to-left.
    """

    backend_mode: str = "ideal"
    fake_backend_name: str | None = None
    initial_state: str = "all_up"
    use_cnot_zz: bool = False
    shots: int = 8192
    optimization_level: int = 1
    use_parallel_u1: bool = False
    trotter_method: str = "even_odd"


# ============================================================================
# Derived geometry helpers
# ============================================================================


def center_index(params: ModelParams) -> int:
    """Return the index of the centre lattice site.

    The centre site is used as the reference qubit when computing connected
    two-point correlators  ⟨Z_c Z_j⟩ − ⟨Z_c⟩⟨Z_j⟩.

    The formula ``ceil(L/2 - 1)`` reproduces the notebook convention for
    both odd and even chain lengths:

    * L=7  →  site 2  (0-indexed)
    * L=6  →  site 2  (0-indexed)

    Args:
        params: Model parameters containing the chain length ``L``.

    Returns:
        Zero-based index of the centre site.
    """
    # Subtract 1 before the ceiling so even L also lands on the left-centre.
    return math.ceil(params.L / 2 - 1)
