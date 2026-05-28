"""
Quantum-circuit construction for the Bloch-oscillations Trotter simulation.

This module contains all gate-level and circuit-level building blocks:

* Single-qubit evolution gates (native ``PauliEvolutionGate`` or explicit
  ``RX``/``RZ`` decomposition).
* Two-qubit ZZ evolution gates (native or ``CX``-based decomposition).
* Circuit initialisation (register allocation + optional |1…1⟩ preparation).
* Trotterized time-evolution layer appended in-place to a circuit.

No I/O, no simulation execution, and no backend construction live here.
The only external dependency is Qiskit; NumPy is used only for angle
arithmetic.

Trotter methods
---------------
``"even_odd"``
    Bond (0,1), (2,3), … then (1,2), (3,4), …  (standard TEBD ordering).
``"odd_even"``
    Bond (1,2), (3,4), … then (0,1), (2,3), …  (reversed ordering).
``"zig_zag"``
    Bond (0,1), (1,2), …, (L-2, L-1) then (L-2, L-1), …, (0,1)  (sweeps
    left-to-right then right-to-left).
"""

from __future__ import annotations

from typing import Union

# qiskit ships no PEP 561 stubs; mypy is configured to ignore missing imports
# for the entire qiskit* namespace via [[tool.mypy.overrides]] in pyproject.toml.
# The type: ignore comments that were previously needed here have been removed
# because the per-module override in pyproject.toml is the single point of truth.
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Instruction
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp

from .model import ModelParams, RunConfig

# Type alias for any gate object that can be appended to a circuit.
# Using a Union rather than a Protocol keeps the dependency on Qiskit
# minimal and avoids structural subtyping pitfalls with untyped third-party
# classes.
GateOrInstruction = Union[PauliEvolutionGate, Instruction]


# ============================================================================
# Single-qubit gate builders
# ============================================================================


def build_single_qubit_gate_native(params: ModelParams) -> PauliEvolutionGate:
    """Build the single-site evolution gate via ``PauliEvolutionGate``.

    Implements  exp(-i (ht X + hl Z) dt)  as a single Qiskit
    ``PauliEvolutionGate`` so the transpiler can choose the best native
    decomposition for the target backend.

    Args:
        params: Model parameters supplying ``ht``, ``hl``, and ``dt``.

    Returns:
        A ``PauliEvolutionGate`` representing one Trotter step of the
        single-qubit part of the Hamiltonian.
    """
    x_op = SparsePauliOp("X")
    z_op = SparsePauliOp("Z")
    # SparsePauliOp arithmetic is not yet typed upstream; suppress the error.
    h1: SparsePauliOp = params.ht * x_op + params.hl * z_op  
    return PauliEvolutionGate(h1, time=params.dt)


def build_single_qubit_gate_decomposed(params: ModelParams) -> Instruction:
    """Build the single-site evolution gate as an explicit ``RX``/``RZ`` circuit.

    This explicit decomposition is equivalent to
    :func:`build_single_qubit_gate_native` but expressed as concrete rotations,
    which can be useful for backends that do not support
    ``PauliEvolutionGate`` directly or when the transpiler's synthesis is
    undesirable for circuit-inspection purposes.

    The decomposition is::

        U_1 = RZ(2 hl dt) RX(2 ht dt)

    Note that the factor of 2 comes from the convention
    ``exp(-i theta/2 P)`` used by Qiskit's rotation gates.

    Args:
        params: Model parameters supplying ``ht``, ``hl``, and ``dt``.

    Returns:
        A one-qubit ``Instruction`` named ``"U1"`` wrapping the RX/RZ circuit.
    """
    circ = QuantumCircuit(1, name="U1")
    theta_x = 2.0 * params.ht * params.dt
    theta_z = 2.0 * params.hl * params.dt
    circ.rx(theta_x, 0)
    circ.rz(theta_z, 0)
    return circ.to_instruction()  


# ============================================================================
# Two-qubit ZZ gate builders
# ============================================================================


def build_two_qubit_gate_native(params: ModelParams) -> PauliEvolutionGate:
    """Build the two-site ZZ evolution gate via ``PauliEvolutionGate``.

    Implements  exp(-i J Z_i Z_{i+1} dt/2)  using a ``PauliEvolutionGate``
    on two qubits.  The time argument is ``dt/2`` because in the even-odd
    decomposition each bond appears twice per full Trotter step.

    Args:
        params: Model parameters supplying ``J`` and ``dt``.

    Returns:
        A two-qubit ``PauliEvolutionGate`` for half a Trotter ZZ step.
    """
    zz_op = SparsePauliOp("ZZ")
    h2: SparsePauliOp = params.J * zz_op  
    return PauliEvolutionGate(h2, time=0.5 * params.dt)


def build_two_qubit_gate_decomposed(params: ModelParams) -> Instruction:
    """Build the ZZ evolution gate as an explicit ``CX``-based circuit.

    Implements  exp(-i J Z Z dt/2)  using the exact two-CNOT decomposition::

        CX(0→1) ── RZ(-2γ, qubit=1) ── CX(0→1)

    where  γ = J * dt / 2.  The minus sign in ``RZ`` arises from Qiskit's
    convention  RZ(θ) = exp(-i θ/2 Z)  combined with the Hamiltonian sign.

    Args:
        params: Model parameters supplying ``J`` and ``dt``.

    Returns:
        A two-qubit ``Instruction`` named ``"ZZ_decomposed"``.
    """
    gamma = 0.5 * params.J * params.dt
    circ = QuantumCircuit(2, name="ZZ_decomposed")
    circ.cx(0, 1)
    circ.rz(-2.0 * gamma, 1)
    circ.cx(0, 1)
    return circ.to_instruction()  


# ============================================================================
# Gate selection helper
# ============================================================================


def build_evolution_gates(
    params: ModelParams,
    config: RunConfig,
) -> tuple[GateOrInstruction, GateOrInstruction | None]:
    """Select and build the ZZ and single-qubit evolution gates.

    Returns ``(U2, U1)`` where:

    * ``U2`` is the two-qubit ZZ gate (native or CNOT-decomposed).
    * ``U1`` is the single-qubit gate, or ``None`` when
      ``config.use_parallel_u1`` is ``True`` (in that mode the gates are
      appended individually inside :func:`append_trotter_layer` to expose
      independent-qubit parallelism to the transpiler).

    Args:
        params: Model parameters.
        config: Run configuration.

    Returns:
        A tuple ``(U2, U1)``.
    """
    u2: GateOrInstruction = (
        build_two_qubit_gate_decomposed(params)
        if config.use_cnot_zz
        else build_two_qubit_gate_native(params)
    )
    u1: GateOrInstruction | None = (
        None
        if config.use_parallel_u1
        else build_single_qubit_gate_native(params)
    )
    return u2, u1


# ============================================================================
# Circuit initialisation
# ============================================================================


def init_circuit(
    params: ModelParams,
    config: RunConfig,
    with_classical: bool = True,
) -> QuantumCircuit:
    """Allocate a fresh quantum circuit with the correct register sizes.

    Optionally prepares the |1…1⟩ initial state when
    ``config.initial_state == "all_down"``.

    Args:
        params:          Model parameters (only ``L`` is used).
        config:          Run configuration (``initial_state`` field).
        with_classical:  When ``True`` (default) a classical register of
                         size ``L`` is added for mid-circuit measurements.
                         Pass ``False`` when using statevector simulation
                         which does not require measurement instructions.

    Returns:
        A freshly initialised ``QuantumCircuit``.

    Raises:
        ValueError: If ``config.initial_state`` is not one of the
                    supported values (``"all_up"``, ``"all_down"``).
    """
    qr = QuantumRegister(params.L)
    if with_classical:
        cr = ClassicalRegister(params.L)
        circ: QuantumCircuit = QuantumCircuit(qr, cr)
    else:
        circ = QuantumCircuit(qr)

    if config.initial_state == "all_up":
        pass  # |0…0⟩ is the default state; nothing to add.
    elif config.initial_state == "all_down":
        for q in range(params.L):
            circ.x(q)
    else:
        raise ValueError(
            f"Unknown initial_state: '{config.initial_state}'. "
            "Supported values: 'all_up', 'all_down'."
        )

    return circ


# ============================================================================
# Trotterized time-evolution layer
# ============================================================================


def append_trotter_layer(
    circ: QuantumCircuit,
    params: ModelParams,
    config: RunConfig,
    u2: GateOrInstruction,
    u1: GateOrInstruction | None,
    layers: int,
    measure: bool = True,
) -> None:
    """Append ``layers`` Trotter steps to ``circ`` in-place.

    Each step applies the two-qubit ZZ layer (bond ordering selected by
    ``config.trotter_method``) followed by the single-qubit layer, then
    inserts a barrier for visual separation in circuit diagrams.

    When ``u1`` is ``None`` the single-qubit rotation is expanded as
    individual ``RX`` / ``RZ`` calls so the transpiler can schedule them
    in parallel across qubits.

    Args:
        circ:    The circuit to modify in-place.
        params:  Model parameters.
        config:  Run configuration (``trotter_method``, ``use_parallel_u1``,
                 ``ht``, ``hl``, ``dt``).
        u2:      Two-qubit ZZ gate.
        u1:      Single-qubit gate, or ``None`` for parallel-U1 mode.
        layers:  Number of Trotter layers to append.
        measure: When ``True`` a full-register measurement instruction is
                 appended after the last layer.

    Raises:
        ValueError: If ``config.trotter_method`` is not one of
                    ``"even_odd"``, ``"odd_even"``, ``"zig_zag"``.
    """
    method = config.trotter_method

    for _ in range(layers):
        if method == "even_odd":
            for q in range(0, params.L - 1, 2):
                circ.append(u2, [q, q + 1])
            for q in range(1, params.L - 1, 2):
                circ.append(u2, [q, q + 1])
        elif method == "odd_even":
            for q in range(1, params.L - 1, 2):
                circ.append(u2, [q, q + 1])
            for q in range(0, params.L - 1, 2):
                circ.append(u2, [q, q + 1])
        elif method == "zig_zag":
            for q in range(params.L - 1):
                circ.append(u2, [q, q + 1])
            for q in reversed(range(params.L - 1)):
                circ.append(u2, [q, q + 1])
        else:
            raise ValueError(
                f"Unknown trotter_method: '{method}'. "
                "Supported values: 'even_odd', 'odd_even', 'zig_zag'."
            )

        if u1 is None:
            # Parallel mode: emit individual RX/RZ so the transpiler sees
            # independent single-qubit operations and can schedule them
            # concurrently.
            theta_x = 2.0 * params.ht * params.dt
            theta_z = 2.0 * params.hl * params.dt
            for q in range(params.L):
                circ.rx(theta_x, q)
            for q in range(params.L):
                circ.rz(theta_z, q)
        else:
            for q in range(params.L):
                circ.append(u1, [q])

        circ.barrier()

    if measure:
        circ.measure(range(params.L), range(params.L))
