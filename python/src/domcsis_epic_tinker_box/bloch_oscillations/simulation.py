import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit_aer import AerSimulator

from .backends import get_fake_backend, make_noisy_density_sim
from .circuit import circuit_init, time_evolution_operator, trotterized_circuit
from .io import (
    append_layer_log,
    log_simulation_header,
    save_simulation_data,
)
from .model import ModelParams, RunConfig, center_index
from .observables import magnetisation_observables, correlator_observables


def _prepare_initial_state(circuit: QuantumCircuit, config: RunConfig, n_qubits: int):
    if config.initial_state == "all_up":
        return
    if config.initial_state == "all_down":
        for q in range(n_qubits):
            circuit.x(q)
        return
    raise ValueError(f"Unsupported initial_state: {config.initial_state}")


def build_circuit_list(params: ModelParams, config: RunConfig, u2, u1):
    circuits_list = []

    if config.backend_mode == "ideal":
        backend = AerSimulator()
    elif config.backend_mode == "fake":
        if config.fake_backend_name is None:
            raise ValueError("fake_backend_name must be set for backend_mode='fake'")
        backend = get_fake_backend(config.fake_backend_name)
    else:
        raise ValueError(f"Unknown backend_mode: {config.backend_mode}")

    for layers in range(params.layers_max):
        circuit = circuit_init(params)
        _prepare_initial_state(circuit, config, params.L)

        trotterized_circuit(
            circuit,
            params,
            u2,
            u1,
            layers,
            config=config,
            measure=True,
        )

        transpiled_circuit = transpile(
            circuit,
            backend=backend,
            optimization_level=config.optimization_level,
        )
        circuits_list.append(transpiled_circuit)

    return circuits_list


def _counts_to_observables(counts, params: ModelParams):
    shots = sum(counts.values())
    mags = np.zeros(params.L)
    corr = np.zeros(params.L)
    center = center_index(params)

    for bitstring, n in counts.items():
        bits = bitstring[::-1]
        z = np.array([1 if b == "0" else -1 for b in bits[: params.L]])
        mags += n * z
        corr += n * (z[center] * z)

    mags /= shots
    corr /= shots
    corr -= mags[center] * mags
    return mags, corr


def run_ideal_simulation_counts(params: ModelParams, config: RunConfig):
    u2, u1 = time_evolution_operator(params, config)
    circuits = []

    for layers in range(params.layers_max):
        circuit = circuit_init(params)
        _prepare_initial_state(circuit, config, params.L)

        trotterized_circuit(
            circuit,
            params,
            u2,
            u1,
            layers,
            config=config,
            measure=True,
        )
        circuits.append(circuit)

    backend = AerSimulator()
    circuits = transpile(
        circuits,
        backend=backend,
        optimization_level=config.optimization_level,
    )
    result = backend.run(circuits, shots=config.shots).result()

    magnetizations = []
    correlators = []

    log_file = log_simulation_header(params, config, "ideal_counts")
    for layer, circuit in enumerate(circuits):
        append_layer_log(log_file, layer, circuit, "ideal_counts")

    for layer in range(params.layers_max):
        counts = result.get_counts(layer)
        mags, corr = _counts_to_observables(counts, params)
        magnetizations.append(mags)
        correlators.append(corr)

    magnetizations = np.array(magnetizations)
    correlators = np.array(correlators)

    save_simulation_data(
        params,
        config,
        magnetizations,
        correlators,
        "ideal_counts",
    )
    return magnetizations, correlators


def run_ideal_simulation_observables(params: ModelParams, config: RunConfig):
    u2, u1 = time_evolution_operator(params, config)
    mag_observables = magnetisation_observables(params)
    corr_observables = correlator_observables(params)

    magnetizations = []
    correlators = []
    log_file = log_simulation_header(params, config, "ideal_observables")
    center = center_index(params)

    for layers in range(params.layers_max):
        circuit = QuantumCircuit(params.L)
        _prepare_initial_state(circuit, config, params.L)

        trotterized_circuit(
            circuit,
            params,
            u2,
            u1,
            layers,
            config=config,
            measure=False,
        )

        circuit = transpile(circuit, basis_gates=["cx", "rz", "sx", "x"])
        append_layer_log(log_file, layers, circuit, "ideal")
        state = Statevector.from_instruction(circuit)

        mags = np.array([
            np.real(state.expectation_value(obs))
            for obs in mag_observables
        ])
        corr_raw = np.array([
            np.real(state.expectation_value(obs))
            for obs in corr_observables
        ])
        corr_connected = corr_raw - mags[center] * mags

        magnetizations.append(mags)
        correlators.append(corr_connected)

    magnetizations = np.array(magnetizations)
    correlators = np.array(correlators)

    save_simulation_data(
        params,
        config,
        magnetizations,
        correlators,
        "ideal_observables",
    )
    return magnetizations, correlators


def run_noisy_simulation_counts(params: ModelParams, config: RunConfig):
    if config.fake_backend_name is None:
        raise ValueError("fake_backend_name must be specified for noisy simulations")

    fake_backend = get_fake_backend(config.fake_backend_name)
    backend = make_noisy_density_sim(fake_backend)

    u2, u1 = time_evolution_operator(params, config)
    circuits = build_circuit_list(params, config, u2, u1)
    result = backend.run(circuits, shots=config.shots).result()

    magnetizations = []
    correlators = []

    log_file = log_simulation_header(
        params,
        config,
        f"noisy_counts_{config.fake_backend_name}",
    )
    for layer, circuit in enumerate(circuits):
        append_layer_log(log_file, layer, circuit, "noisy_counts")

    for layer in range(params.layers_max):
        counts = result.get_counts(layer)
        mags, corr = _counts_to_observables(counts, params)
        magnetizations.append(mags)
        correlators.append(corr)

    magnetizations = np.array(magnetizations)
    correlators = np.array(correlators)

    save_simulation_data(
        params,
        config,
        magnetizations,
        correlators,
        f"noisy_counts_{config.fake_backend_name}",
    )
    return magnetizations, correlators


def run_noisy_simulation_observables(params: ModelParams, config: RunConfig):
    if config.fake_backend_name is None:
        raise ValueError("fake_backend_name must be specified for noisy simulations")

    fake_backend = get_fake_backend(config.fake_backend_name)
    backend = make_noisy_density_sim(fake_backend)

    u2, u1 = time_evolution_operator(params, config)
    logical_mag_observables = magnetisation_observables(params)
    logical_corr_observables = correlator_observables(params)

    magnetizations = []
    correlators = []
    center = center_index(params)

    log_file = log_simulation_header(
        params,
        config,
        f"noisy_observables_{config.fake_backend_name}",
    )

    for layers in range(params.layers_max):
        circuit = QuantumCircuit(params.L)
        _prepare_initial_state(circuit, config, params.L)

        trotterized_circuit(
            circuit,
            params,
            u2,
            u1,
            layers,
            config=config,
            measure=False,
        )

        circuit = transpile(
            circuit,
            backend=backend,
            optimization_level=config.optimization_level,
        )
        append_layer_log(log_file, layers, circuit, "noisy")

        mag_observables = [
            obs.apply_layout(circuit.layout)
            for obs in logical_mag_observables
        ]
        corr_observables = [
            obs.apply_layout(circuit.layout)
            for obs in logical_corr_observables
        ]

        circuit.save_density_matrix()
        result = backend.run(circuit).result()
        rho = DensityMatrix(result.data(0)["density_matrix"])

        mags = np.array([
            np.real(rho.expectation_value(obs))
            for obs in mag_observables
        ])
        corr_raw = np.array([
            np.real(rho.expectation_value(obs))
            for obs in corr_observables
        ])
        corr_connected = corr_raw - mags[center] * mags

        magnetizations.append(mags)
        correlators.append(corr_connected)

    magnetizations = np.array(magnetizations)
    correlators = np.array(correlators)

    save_simulation_data(
        params,
        config,
        magnetizations,
        correlators,
        f"noisy_observables_{config.fake_backend_name}",
    )
    return magnetizations, correlators
