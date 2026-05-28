"""
High-level simulation drivers for the Bloch-oscillations experiment.

This module provides four public functions, one for each combination of:

* **Noise model** — ideal (noiseless ``AerSimulator``) vs. noisy (density-
  matrix simulator seeded from a fake IBM-backend noise model).
* **Observable extraction** — ``counts`` (post-process measurement bit-
  strings) vs. ``observables`` (exact expectation values via
  ``Statevector`` / ``DensityMatrix``).

+--------------------+---------+------------+
| Function           | Noise   | Extraction |
+====================+=========+============+
| run_ideal_...count | None    | Counts     |
| run_ideal_...obs   | None    | Statevec   |
| run_noisy_...count | Fake BK | Counts     |
| run_noisy_...obs   | Fake BK | DensityMat |
+--------------------+---------+------------+

Each driver:

1. Builds the evolution gates via :func:`~.circuit.build_evolution_gates`.
2. Constructs and transpiles one circuit per Trotter layer.
3. Runs the circuits on the target backend.
4. Extracts site magnetisations ⟨Z_j⟩ and connected correlators
   ⟨Z_c Z_j⟩ − ⟨Z_c⟩⟨Z_j⟩.
5. Persists the results with :func:`~.io.save_simulation_data`.

All drivers return ``(magnetizations, correlators)`` as 2-D NumPy arrays
of shape ``(layers_max, L)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from qiskit import transpile 
from qiskit.quantum_info import SparsePauliOp, Statevector, DensityMatrix 
from qiskit_aer import AerSimulator  

from .model import ModelParams, RunConfig, center_index
from .backends import get_fake_backend, make_noisy_density_simulator
from .circuit import build_evolution_gates, init_circuit, append_trotter_layer
from .observables import magnetisation_observables, correlator_observables
from .io import save_simulation_data, write_log_header, append_layer_log


# ============================================================================
# Internal helpers
# ============================================================================


def _extract_from_counts(
    counts: dict[str, int],
    shots: int,
    params: ModelParams,
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]]:
    """Compute ⟨Z_j⟩ and connected correlators from a counts dictionary.

    Each bit-string in ``counts`` is interpreted as a computational-basis
    measurement outcome.  The convention used by Qiskit is *little-endian*:
    the leftmost character of the reversed string corresponds to qubit 0.

    Bit value mapping:  ``'0'`` → +1,  ``'1'`` − −1.

    Args:
        counts: Qiskit result counts dictionary mapping bit-strings to
                multiplicity.
        shots:  Total number of measurement shots (= sum of multiplicities).
        params: Model parameters providing chain length ``L`` and the
                centre-site index.

    Returns:
        A tuple ``(mags, corr)`` where both arrays have shape ``(L,)``.
        ``corr`` is the *connected* correlator
        ⟨Z_c Z_j⟩ − ⟨Z_c⟩⟨Z_j⟩ using the pre-computed single-site
        averages.
    """
    c = center_index(params)
    mags: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(params.L, dtype=np.float64)
    corr_raw: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(params.L, dtype=np.float64)

    for bitstring, n in counts.items():
        bits = bitstring[::-1]
        z: np.ndarray[Any, np.dtype[np.float64]] = np.array(
            [1.0 if b == "0" else -1.0 for b in bits[: params.L]], dtype=np.float64
        )
        mags += n * z
        corr_raw += n * (z[c] * z)

    mags /= shots
    corr_raw /= shots
    corr_connected: np.ndarray[Any, np.dtype[np.float64]] = corr_raw - mags[c] * mags
    return mags, corr_connected


# ============================================================================
# Ideal simulations
# ============================================================================


def run_ideal_simulation_counts(
    params: ModelParams,
    config: RunConfig,
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]]:
    """Run the ideal (noiseless) simulation and extract results from counts.

    Builds one circuit per Trotter layer (0 … layers_max-1), transpiles
    them, runs a single batched job on the noiseless ``AerSimulator``, and
    post-processes each result's counts dictionary to extract site
    magnetisations and connected correlators.

    Args:
        params: Physical model parameters.
        config: Run configuration.

    Returns:
        A tuple ``(magnetizations, correlators)`` of shape
        ``(layers_max, L)``.
    """
    u2, u1 = build_evolution_gates(params, config)
    circuits = []

    for layers in range(params.layers_max):
        circ = init_circuit(params, config, with_classical=True)
        append_trotter_layer(circ, params, config, u2, u1, layers=layers, measure=True)
        circuits.append(circ)

    backend: AerSimulator = AerSimulator()
    transpiled = transpile(circuits, backend=backend, optimization_level=config.optimization_level)

    log_file = write_log_header(params, config, "ideal_counts")
    for layer, circuit in enumerate(transpiled):
        append_layer_log(log_file, layer, circuit, "ideal_counts")

    result = backend.run(transpiled, shots=config.shots).result()

    magnetizations_list: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    correlators_list: list[np.ndarray[Any, np.dtype[np.float64]]] = []

    for layer in range(params.layers_max):
        counts: dict[str, int] = result.get_counts(layer)  
        shots = sum(counts.values())
        mags, corr = _extract_from_counts(counts, shots, params)
        magnetizations_list.append(mags)
        correlators_list.append(corr)

    magnetizations: np.ndarray[Any, np.dtype[np.float64]] = np.array(magnetizations_list)
    correlators: np.ndarray[Any, np.dtype[np.float64]] = np.array(correlators_list)

    save_simulation_data(params, config, magnetizations, correlators, "ideal_counts")
    return magnetizations, correlators


def run_ideal_simulation_observables(
    params: ModelParams,
    config: RunConfig,
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]]:
    """Run the ideal simulation using exact statevector expectation values.

    For each Trotter layer a circuit without measurement instructions is
    transpiled, converted to a ``Statevector``, and queried for the
    expectation value of each observable directly.  This avoids shot noise
    and is suitable for benchmarking / reference data.

    Args:
        params: Physical model parameters.
        config: Run configuration.

    Returns:
        A tuple ``(magnetizations, correlators)`` of shape
        ``(layers_max, L)``.
    """
    u2, u1 = build_evolution_gates(params, config)
    mag_obs: list[SparsePauliOp] = magnetisation_observables(params)
    corr_obs: list[SparsePauliOp] = correlator_observables(params)
    c = center_index(params)

    log_file = write_log_header(params, config, "ideal_observables")

    magnetizations_list: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    correlators_list: list[np.ndarray[Any, np.dtype[np.float64]]] = []

    for layers in range(params.layers_max):
        circ = init_circuit(params, config, with_classical=False)
        append_trotter_layer(circ, params, config, u2, u1, layers=layers, measure=False)

        circ = transpile(circ, basis_gates=["cx", "rz", "sx", "x"])  

        append_layer_log(log_file, layers, circ, "ideal_observables")

        state: Statevector = Statevector.from_instruction(circ)

        mags: np.ndarray[Any, np.dtype[np.float64]] = np.array(
            [np.real(state.expectation_value(obs)) for obs in mag_obs],
            dtype=np.float64,
        )
        corr_raw: np.ndarray[Any, np.dtype[np.float64]] = np.array(
            [np.real(state.expectation_value(obs)) for obs in corr_obs],
            dtype=np.float64,
        )
        corr_connected: np.ndarray[Any, np.dtype[np.float64]] = corr_raw - mags[c] * mags

        magnetizations_list.append(mags)
        correlators_list.append(corr_connected)

    magnetizations: np.ndarray[Any, np.dtype[np.float64]] = np.array(magnetizations_list)
    correlators: np.ndarray[Any, np.dtype[np.float64]] = np.array(correlators_list)

    save_simulation_data(params, config, magnetizations, correlators, "ideal_observables")
    return magnetizations, correlators


# ============================================================================
# Noisy simulations
# ============================================================================


def run_noisy_simulation_counts(
    params: ModelParams,
    config: RunConfig,
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]]:
    """Run the noisy simulation (density-matrix) and extract from counts.

    Args:
        params: Physical model parameters.
        config: Run configuration.  ``config.fake_backend_name`` must be set.

    Returns:
        A tuple ``(magnetizations, correlators)`` of shape
        ``(layers_max, L)``.

    Raises:
        ValueError: If ``config.fake_backend_name`` is ``None``.
    """
    if config.fake_backend_name is None:
        raise ValueError(
            "fake_backend_name must be specified for noisy simulations."
        )

    fake_backend = get_fake_backend(config.fake_backend_name)
    backend: AerSimulator = make_noisy_density_simulator(fake_backend)

    u2, u1 = build_evolution_gates(params, config)
    circuits = []

    for layers in range(params.layers_max):
        circ = init_circuit(params, config, with_classical=True)
        append_trotter_layer(circ, params, config, u2, u1, layers=layers, measure=True)
        circuits.append(circ)

    transpiled = transpile(
        circuits,
        backend=backend,
        optimization_level=config.optimization_level,
    )

    label = f"noisy_counts_{config.fake_backend_name}"
    log_file = write_log_header(params, config, label)
    for layer, circuit in enumerate(transpiled):
        append_layer_log(log_file, layer, circuit, "noisy_counts")

    result = backend.run(transpiled, shots=config.shots).result()

    magnetizations_list: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    correlators_list: list[np.ndarray[Any, np.dtype[np.float64]]] = []

    for layer in range(params.layers_max):
        counts: dict[str, int] = result.get_counts(layer)  
        shots = sum(counts.values())
        mags, corr = _extract_from_counts(counts, shots, params)
        magnetizations_list.append(mags)
        correlators_list.append(corr)

    magnetizations: np.ndarray[Any, np.dtype[np.float64]] = np.array(magnetizations_list)
    correlators: np.ndarray[Any, np.dtype[np.float64]] = np.array(correlators_list)

    save_simulation_data(params, config, magnetizations, correlators, label)
    return magnetizations, correlators


def run_noisy_simulation_observables(
    params: ModelParams,
    config: RunConfig,
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]]:
    """Run the noisy simulation using density-matrix expectation values.

    Transpiles each circuit to the fake backend's native gate set, saves
    the density matrix explicitly, and queries it for observable expectation
    values.  Qubit layout remapping is handled by
    ``SparsePauliOp.apply_layout``.

    Args:
        params: Physical model parameters.
        config: Run configuration.  ``config.fake_backend_name`` must be set.

    Returns:
        A tuple ``(magnetizations, correlators)`` of shape
        ``(layers_max, L)``.

    Raises:
        ValueError: If ``config.fake_backend_name`` is ``None``.
    """
    if config.fake_backend_name is None:
        raise ValueError(
            "fake_backend_name must be specified for noisy simulations."
        )

    fake_backend = get_fake_backend(config.fake_backend_name)
    backend: AerSimulator = make_noisy_density_simulator(fake_backend)

    u2, u1 = build_evolution_gates(params, config)
    logical_mag_obs: list[SparsePauliOp] = magnetisation_observables(params)
    logical_corr_obs: list[SparsePauliOp] = correlator_observables(params)
    c = center_index(params)

    label = f"noisy_observables_{config.fake_backend_name}"
    log_file = write_log_header(params, config, label)

    magnetizations_list: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    correlators_list: list[np.ndarray[Any, np.dtype[np.float64]]] = []

    for layers in range(params.layers_max):
        circ = init_circuit(params, config, with_classical=False)
        append_trotter_layer(circ, params, config, u2, u1, layers=layers, measure=False)

        circ = transpile(
            circ,
            backend=backend,
            optimization_level=config.optimization_level,
        )  

        append_layer_log(log_file, layers, circ, "noisy_observables")

        # Remap logical observables to the physical qubit layout chosen by the
        # transpiler.
        mag_obs: list[SparsePauliOp] = [
            obs.apply_layout(circ.layout) for obs in logical_mag_obs  
        ]
        corr_obs: list[SparsePauliOp] = [
            obs.apply_layout(circ.layout) for obs in logical_corr_obs  
        ]

        circ.save_density_matrix()  
        result = backend.run(circ).result()
        rho: DensityMatrix = DensityMatrix(result.data(0)["density_matrix"])

        mags: np.ndarray[Any, np.dtype[np.float64]] = np.array(
            [np.real(rho.expectation_value(obs)) for obs in mag_obs],
            dtype=np.float64,
        )
        corr_raw: np.ndarray[Any, np.dtype[np.float64]] = np.array(
            [np.real(rho.expectation_value(obs)) for obs in corr_obs],
            dtype=np.float64,
        )
        corr_connected: np.ndarray[Any, np.dtype[np.float64]] = corr_raw - mags[c] * mags

        magnetizations_list.append(mags)
        correlators_list.append(corr_connected)

    magnetizations: np.ndarray[Any, np.dtype[np.float64]] = np.array(magnetizations_list)
    correlators: np.ndarray[Any, np.dtype[np.float64]] = np.array(correlators_list)

    save_simulation_data(params, config, magnetizations, correlators, label)
    return magnetizations, correlators
