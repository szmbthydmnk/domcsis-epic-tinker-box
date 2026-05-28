import pytest
from qiskit import QuantumCircuit

from domcsis_epic_tinker_box.bloch_oscillations.circuit import (
    build_single_qubit_gate_decomposed,
    build_single_qubit_gate_native,
    build_two_qubit_gate_decomposed,
    build_two_qubit_gate_native,
    circuit_init,
    time_evolution_operator,
    trotterized_circuit,
)
from domcsis_epic_tinker_box.bloch_oscillations.model import ModelParams, RunConfig


def test_build_two_qubit_gate_decomposed_name():
    gate = build_two_qubit_gate_decomposed(ModelParams())
    assert gate.name == "ZZ_decomposed"


def test_build_single_qubit_gate_decomposed_name():
    gate = build_single_qubit_gate_decomposed(ModelParams())
    assert gate.name == "U1"


def test_build_two_qubit_gate_native_returns_gate():
    from qiskit.circuit.library import PauliEvolutionGate
    gate = build_two_qubit_gate_native(ModelParams())
    assert isinstance(gate, PauliEvolutionGate)


def test_build_single_qubit_gate_native_returns_gate():
    from qiskit.circuit.library import PauliEvolutionGate
    gate = build_single_qubit_gate_native(ModelParams())
    assert isinstance(gate, PauliEvolutionGate)


def test_circuit_init_registers():
    params = ModelParams(L=4)
    circ = circuit_init(params)
    assert circ.num_qubits == 4
    assert circ.num_clbits == 4


@pytest.mark.parametrize("method", ["even_odd", "odd_even", "zig_zag"])
def test_trotterized_circuit_all_methods(method):
    params = ModelParams(L=4, layers_max=2)
    config = RunConfig(trotter_method=method)
    u2, u1 = time_evolution_operator(params, config)
    circ = QuantumCircuit(params.L, params.L)
    trotterized_circuit(circ, params, u2, u1, 2, config=config, measure=True)
    assert circ.count_ops().get("measure", 0) == 4


def test_trotterized_circuit_zero_layers_only_measures():
    params = ModelParams(L=3, layers_max=1)
    config = RunConfig(trotter_method="even_odd")
    u2, u1 = time_evolution_operator(params, config)
    circ = QuantumCircuit(params.L, params.L)
    trotterized_circuit(circ, params, u2, u1, 0, config=config, measure=True)
    assert circ.count_ops().get("measure", 0) == 3


def test_trotterized_circuit_unknown_method_raises():
    params = ModelParams(L=3)
    config = RunConfig(trotter_method="bad_method")
    u2, u1 = time_evolution_operator(params, config)
    circ = QuantumCircuit(params.L, params.L)
    with pytest.raises(ValueError, match="Unknown Trotter method"):
        trotterized_circuit(circ, params, u2, u1, 1, config=config)


def test_time_evolution_operator_cnot_zz():
    config = RunConfig(use_cnot_zz=True)
    u2, u1 = time_evolution_operator(ModelParams(), config)
    assert u2.name == "ZZ_decomposed"


def test_time_evolution_operator_parallel_u1_is_none():
    config = RunConfig(use_parallel_u1=True)
    _, u1 = time_evolution_operator(ModelParams(), config)
    assert u1 is None
