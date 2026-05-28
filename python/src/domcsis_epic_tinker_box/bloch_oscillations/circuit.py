from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp

from .model import ModelParams, RunConfig


def build_single_qubit_gate_decomposed(params: ModelParams):
    circ = QuantumCircuit(1, name="U1")
    theta_x = 2 * params.ht * params.dt
    theta_z = 2 * params.hl * params.dt
    circ.rx(theta_x, 0)
    circ.rz(theta_z, 0)
    return circ.to_instruction()


def build_single_qubit_gate_native(params: ModelParams):
    z = SparsePauliOp("Z")
    x = SparsePauliOp("X")
    h1 = params.ht * x + params.hl * z
    return PauliEvolutionGate(h1, time=params.dt)


def build_two_qubit_gate_native(params: ModelParams):
    z = SparsePauliOp("Z")
    h2 = params.J * (z ^ z)
    return PauliEvolutionGate(h2, time=0.5 * params.dt)


def build_two_qubit_gate_decomposed(params: ModelParams):
    gamma = 0.5 * params.J * params.dt
    circ = QuantumCircuit(2, name="ZZ_decomposed")
    circ.cx(0, 1)
    circ.rz(-2.0 * gamma, 1)
    circ.cx(0, 1)
    return circ.to_instruction()


def time_evolution_operator(params: ModelParams, config: RunConfig):
    if config.use_cnot_zz:
        u2 = build_two_qubit_gate_decomposed(params)
    else:
        u2 = build_two_qubit_gate_native(params)

    if config.use_parallel_u1:
        u1 = None
    else:
        u1 = build_single_qubit_gate_native(params)

    return u2, u1


def circuit_init(params: ModelParams):
    qr = QuantumRegister(params.L)
    cr = ClassicalRegister(params.L)
    return QuantumCircuit(qr, cr)


def _append_parallel_u1_layer(circ: QuantumCircuit, params: ModelParams):
    theta_x = 2 * params.ht * params.dt
    theta_z = 2 * params.hl * params.dt
    for qubit in range(params.L):
        circ.rx(theta_x, qubit)
    for qubit in range(params.L):
        circ.rz(theta_z, qubit)


def _append_serial_u1_layer(circ: QuantumCircuit, params: ModelParams, u1):
    for qubit in range(params.L):
        circ.append(u1, [qubit])


def trotterized_circuit(
    circ: QuantumCircuit,
    params: ModelParams,
    u2,
    u1,
    layers: int,
    config: RunConfig,
    measure: bool = True,
) -> None:
    method = config.trotter_method

    for _layer in range(layers):
        if method == "zig_zag":
            for qubit in range(params.L - 1):
                circ.append(u2, [qubit, qubit + 1])
            for qubit in reversed(range(params.L - 1)):
                circ.append(u2, [qubit, qubit + 1])
        elif method == "even_odd":
            for qubit in range(0, params.L - 1, 2):
                circ.append(u2, [qubit, qubit + 1])
            for qubit in range(1, params.L - 1, 2):
                circ.append(u2, [qubit, qubit + 1])
        elif method == "odd_even":
            for qubit in range(1, params.L - 1, 2):
                circ.append(u2, [qubit, qubit + 1])
            for qubit in range(0, params.L - 1, 2):
                circ.append(u2, [qubit, qubit + 1])
        else:
            raise ValueError(f"Unknown Trotter method: {method}")

        if config.use_parallel_u1:
            _append_parallel_u1_layer(circ, params)
        else:
            _append_serial_u1_layer(circ, params, u1)

        circ.barrier()

    if measure:
        circ.measure(range(params.L), range(params.L))
