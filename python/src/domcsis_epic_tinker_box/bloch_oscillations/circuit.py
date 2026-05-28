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

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister  # type: ignore[import-untyped]
from qiskit.circuit import Instruction  # type: ignore[import-untyped]
from qiskit.circuit.library import PauliEvolutionGate  # type: ignore[import-untyped]
from qiskit.quantum_info import SparsePauliOp  # type: ignore[import-untyped]

from .model import ModelParams, RunConfig


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
    # Build the 1-qubit Hamiltonian  H_1 = ht X + hl Z.
    x_op = SparsePauliOp("X")
    z_op = SparsePauliOp("Z")
    h1 = params.ht * x_op + params.hl * z_op

    # Wrap in a PauliEvolutionGate for the full Trotter step time dt.
    return PauliEvolutionGate(h1, time=params.dt)


def build_single_qubit_gate_decomposed(params: ModelParams) -> Instruction:
    """Build the single-site evolution gate as an explicit ``RX``/``RZ`` circuit.

    This explicit decomposition is equivalent to ``build_single_qubit_gate_native``
    but expressed as concrete rotations, which can be useful for backends that
    do not support ``PauliEvolutionGate`` directly or when the transpiler's
    synthesis is undesirable for circuit-inspection purposes.

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

    # Rotation angles: the factor 2 comes from Qiskit's R(theta) = exp(-i theta/2 P).
    theta_x = 2.0 * params.ht * params.dt
    theta_z = 2.0 * params.hl * params.dt

    circ.rx(theta_x, 0)
    circ.rz(theta_z, 0)

    return circ.to_instruction()  # type: ignore[return-value]


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
    # Build H_2 = J * ZZ on two qubits.
    zz_op = SparsePauliOp("ZZ")
    h2 = params.J * zz_op

    # Use half the step time so two applications per full layer give exp(-i J ZZ dt).
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
    # Half the Trotter step contributes γ to the rotation angle.
    gamma = 0.5 * params.J * params.dt

    circ = QuantumCircuit(2, name="ZZ_decomposed")

    # Standard CX–RZ–CX synthesis of exp(-i γ ZZ).
    circ.cx(0, 1)
    circ.rz(-2.0 * gamma, 1)
    circ.cx(0, 1)

    return circ.to_instruction()  # type: ignore[return-value]


# ============================================================================
# Gate selection helper
# ============================================================================


def build_evolution_gates(
    params: ModelParams,
    config: RunConfig,
) -> tuple[PauliEvolutionGate | Instruction, PauliEvolutionGate | Instruction | None]:
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
    # Choose the ZZ gate implementation.
    if config.use_cnot_zz:
        u2: PauliEvolutionGate | Instruction = build_two_qubit_gate_decomposed(params)
    else:
        u2 = build_two_qubit_gate_native(params)

    # The single-qubit gate is omitted (None) when parallel scheduling is
    # requested — the raw RX/RZ gates are appended qubit-by-qubit instead.
    if config.use_parallel_u1:
        u1: PauliEvolutionGate | Instruction | None = None
    else:
        u1 = build_single_qubit_gate_native(params)

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
                         the same size as the quantum register is added.
                         Set to ``False`` for statevector / density-matrix
                         simulations that do not use measurements.

    Returns:
        A freshly allocated ``QuantumCircuit``.

    Raises:
        ValueError: If ``config.initial_state`` is not recognised.
    """
    qr = QuantumRegister(params.L)

    if with_classical:
        cr = ClassicalRegister(params.L)
        circ: QuantumCircuit = QuantumCircuit(qr, cr)
    else:
        circ = QuantumCircuit(qr)

    if config.initial_state == "all_up":
        # |0…0⟩ is the default computational-basis state; no gates needed.
        pass
    elif config.initial_state == "all_down":
        # Flip every qubit to prepare |1…1⟩.
        for q in range(params.L):
            circ.x(q)
    else:
        raise ValueError(
            f"Unknown initial_state: '{config.initial_state}'. "
            "Supported values: 'all_up', 'all_down'."
        )

    return circ


# ============================================================================
# Trotter layer appender
# ============================================================================


def append_trotter_layer(
    circ: QuantumCircuit,
    params: ModelParams,
    config: RunConfig,
    u2: PauliEvolutionGate | Instruction,
    u1: PauliEvolutionGate | Instruction | None,
    layers: int,
    *,
    measure: bool = True,
) -> None:
    """Append ``layers`` Trotter steps in-place to ``circ``.

    The bond ordering within each step is controlled by
    ``config.trotter_method``.  After the two-qubit bonds, the single-qubit
    evolution is applied either as a packaged ``u1`` instruction or as
    individual ``RX``/``RZ`` sweeps when ``config.use_parallel_u1`` is
    ``True``.  A ``barrier`` is inserted after each full Trotter step to
    prevent the transpiler from merging layers into each other.

    If ``measure`` is ``True`` a final measurement of all qubits into the
    classical register is appended after all layers.

    Args:
        circ:    Target circuit to modify in-place.
        params:  Model parameters (chain length and time step).
        config:  Run configuration (Trotter method, parallel-U1 flag).
        u2:      Two-qubit ZZ evolution gate.
        u1:      Single-qubit evolution gate, or ``None`` when
                 ``config.use_parallel_u1`` is ``True``.
        layers:  Number of Trotter steps to append.
        measure: Whether to append a measurement at the end.

    Raises:
        ValueError: If ``config.trotter_method`` is not recognised.
    """
    for _layer in range(layers):
        # ------------------------------------------------------------------
        # Two-qubit bond layer — ordering depends on the Trotter method.
        # ------------------------------------------------------------------
        if config.trotter_method == "even_odd":
            # Apply even bonds (0-1, 2-3, …) first, then odd bonds (1-2, 3-4, …).
            for q in range(0, params.L - 1, 2):
                circ.append(u2, [q, q + 1])
            for q in range(1, params.L - 1, 2):
                circ.append(u2, [q, q + 1])

        elif config.trotter_method == "odd_even":
            # Apply odd bonds first, then even bonds — mirror of even_odd.
            for q in range(1, params.L - 1, 2):
                circ.append(u2, [q, q + 1])
            for q in range(0, params.L - 1, 2):
                circ.append(u2, [q, q + 1])

        elif config.trotter_method == "zig_zag":
            # Sweep all bonds left-to-right, then right-to-left.
            for q in range(params.L - 1):
                circ.append(u2, [q, q + 1])
            for q in reversed(range(params.L - 1)):
                circ.append(u2, [q, q + 1])

        else:
            raise ValueError(
                f"Unknown trotter_method: '{config.trotter_method}'. "
                "Supported values: 'even_odd', 'odd_even', 'zig_zag'."
            )

        # ------------------------------------------------------------------
        # Single-qubit evolution layer.
        # ------------------------------------------------------------------
        if config.use_parallel_u1:
            # Separate sweeps allow the transpiler to schedule all RX gates
            # simultaneously, then all RZ gates simultaneously, maximising
            # hardware parallelism.
            theta_x = 2.0 * params.ht * params.dt
            theta_z = 2.0 * params.hl * params.dt
            for q in range(params.L):
                circ.rx(theta_x, q)
            for q in range(params.L):
                circ.rz(theta_z, q)
        else:
            # Apply the packaged single-qubit instruction to every site.
            for q in range(params.L):
                circ.append(u1, [q])  # type: ignore[arg-type]

        # Barrier prevents the transpiler from fusing adjacent Trotter steps.
        circ.barrier()

    # Append terminal measurements if requested.
    if measure:
        circ.measure(range(params.L), range(params.L))
