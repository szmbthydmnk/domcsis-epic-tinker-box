"""
Tests for bloch_oscillations.circuit.

Verifies the gate-builder return types, the circuit-initialisation helper,
and the Trotter-layer appender for all three bond-ordering methods on a
small chain (L=3) to keep the test suite fast.
"""

from __future__ import annotations

import pytest

from qiskit.circuit import Instruction  # type: ignore[import-untyped]
from qiskit.circuit.library import PauliEvolutionGate  # type: ignore[import-untyped]
from qiskit import QuantumCircuit  # type: ignore[import-untyped]

from domcsis_epic_tinker_box.bloch_oscillations.model import ModelParams, RunConfig
from domcsis_epic_tinker_box.bloch_oscillations.circuit import (
    build_single_qubit_gate_native,
    build_single_qubit_gate_decomposed,
    build_two_qubit_gate_native,
    build_two_qubit_gate_decomposed,
    build_evolution_gates,
    init_circuit,
    append_trotter_layer,
)


# Minimal parameters for fast tests.
_PARAMS = ModelParams(L=3, layers_max=2)


# ============================================================================
# Gate builders
# ============================================================================


def test_single_qubit_native_returns_pauli_evolution_gate() -> None:
    """build_single_qubit_gate_native returns a PauliEvolutionGate."""
    gate = build_single_qubit_gate_native(_PARAMS)
    assert isinstance(gate, PauliEvolutionGate)


def test_single_qubit_decomposed_returns_instruction() -> None:
    """build_single_qubit_gate_decomposed returns an Instruction."""
    gate = build_single_qubit_gate_decomposed(_PARAMS)
    assert isinstance(gate, Instruction)


def test_two_qubit_native_returns_pauli_evolution_gate() -> None:
    """build_two_qubit_gate_native returns a PauliEvolutionGate."""
    gate = build_two_qubit_gate_native(_PARAMS)
    assert isinstance(gate, PauliEvolutionGate)


def test_two_qubit_decomposed_returns_instruction() -> None:
    """build_two_qubit_gate_decomposed returns an Instruction."""
    gate = build_two_qubit_gate_decomposed(_PARAMS)
    assert isinstance(gate, Instruction)


# ============================================================================
# build_evolution_gates
# ============================================================================


def test_build_evolution_gates_native_u1_not_none() -> None:
    """When use_parallel_u1=False, U1 is not None."""
    config = RunConfig(use_parallel_u1=False)
    _u2, u1 = build_evolution_gates(_PARAMS, config)
    assert u1 is not None


def test_build_evolution_gates_parallel_u1_is_none() -> None:
    """When use_parallel_u1=True, U1 is None."""
    config = RunConfig(use_parallel_u1=True)
    _u2, u1 = build_evolution_gates(_PARAMS, config)
    assert u1 is None


# ============================================================================
# init_circuit
# ============================================================================


def test_init_circuit_all_up_has_no_x_gates() -> None:
    """all_up initial state: circuit has no X gates (|0...0> is the default)."""
    config = RunConfig(initial_state="all_up")
    circ = init_circuit(_PARAMS, config)
    ops = dict(circ.count_ops())
    assert "x" not in ops


def test_init_circuit_all_down_has_x_gates() -> None:
    """all_down initial state: L X gates are prepended."""
    config = RunConfig(initial_state="all_down")
    circ = init_circuit(_PARAMS, config)
    ops = dict(circ.count_ops())
    assert ops.get("x", 0) == _PARAMS.L


def test_init_circuit_unknown_state_raises() -> None:
    """Unknown initial_state raises ValueError."""
    config = RunConfig(initial_state="neel")
    with pytest.raises(ValueError, match="Unknown initial_state"):
        init_circuit(_PARAMS, config)


# ============================================================================
# append_trotter_layer
# ============================================================================


@pytest.mark.parametrize("method", ["even_odd", "odd_even", "zig_zag"])
def test_append_trotter_layer_all_methods_run(
    method: str,
) -> None:
    """All three Trotter methods append without error for L=3, 2 layers."""
    config = RunConfig(trotter_method=method, use_parallel_u1=False)
    u2, u1 = build_evolution_gates(_PARAMS, config)
    circ = init_circuit(_PARAMS, config)
    append_trotter_layer(circ, _PARAMS, config, u2, u1, layers=2, measure=True)
    assert isinstance(circ, QuantumCircuit)


def test_append_trotter_layer_parallel_u1_runs() -> None:
    """Parallel U1 mode (u1=None) executes without error."""
    config = RunConfig(trotter_method="even_odd", use_parallel_u1=True)
    u2, u1 = build_evolution_gates(_PARAMS, config)
    circ = init_circuit(_PARAMS, config)
    append_trotter_layer(circ, _PARAMS, config, u2, u1, layers=1, measure=True)
    assert isinstance(circ, QuantumCircuit)


def test_append_trotter_layer_unknown_method_raises() -> None:
    """Unknown trotter_method raises ValueError."""
    config = RunConfig(trotter_method="unknown")
    u2, u1 = build_evolution_gates(_PARAMS, config)
    circ = init_circuit(_PARAMS, RunConfig())
    with pytest.raises(ValueError, match="Unknown trotter_method"):
        append_trotter_layer(circ, _PARAMS, config, u2, u1, layers=1, measure=False)


def test_append_trotter_layer_barrier_count() -> None:
    """Number of barriers equals the number of layers appended."""
    layers = 3
    config = RunConfig(trotter_method="even_odd", use_parallel_u1=False)
    u2, u1 = build_evolution_gates(_PARAMS, config)
    circ = init_circuit(_PARAMS, config)
    append_trotter_layer(circ, _PARAMS, config, u2, u1, layers=layers, measure=False)
    barrier_count = dict(circ.count_ops()).get("barrier", 0)
    assert barrier_count == layers
