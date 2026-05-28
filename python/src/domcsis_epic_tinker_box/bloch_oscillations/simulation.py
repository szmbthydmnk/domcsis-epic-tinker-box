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

import numpy as np
from qiskit import transpile  # type: ignore[import-untyped]
from qiskit.quantum_info import SparsePauliOp, Statevector, DensityMatrix  # type: ignore[import-untyped]
from qiskit_aer import AerSimulator  # type: ignore[import-untyped]

from .model import ModelParams, RunConfig, center_index
from .backends import get_fake_backend, make_noisy_density_simulator
from .circuit import build_evolution_gates, init_circuit, append_trotter_layer
from .observables import magnetisation_observables, correlator_observables
from .io import save_simulation_data, write_log_header, append_layer_log


# ============================================================================
# Internal helpers
# ============================================================================


def _extract_magnetisations_and_correlators_from_counts(
    counts: dict[str, int],
    shots: int,
    params: ModelParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ⟨Z_j⟩ and connected correlators from a counts dictionary.

    Each bit-string in ``counts`` is interpreted as a computational-basis
    measurement outcome.  The convention used by Qiskit is *little-endian*:
    the leftmost character of the reversed string corresponds to qubit 0.

    Bit value mapping:  ``'0'`` → +1,  ``'1'`` → −1.

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
    mags = np.zeros(params.L)
    corr_raw = np.zeros(params.L)

    for bitstring, n in counts.items():
        # Reverse so that index 0 = qubit 0 (Qiskit little-endian convention).
        bits = bitstring[::-1]
        # Map '0' → +1,  '1' → −1.
        z = np.array([1 if b == "0" else -1 for b in bits[: params.L]])

        mags += n * z
        corr_raw += n * (z[c] * z)

    # Normalise by the total shot count.
    mags /= shots
    corr_raw /= shots

    # Subtract the disconnected part to obtain the connected correlator.
    corr_connected = corr_raw - mags[c] * mags

    return mags, corr_connected


# ============================================================================
# Ideal simulations
# ============================================================================


def run_ideal_simulation_counts(
    params: ModelParams,
    config: RunConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Run an ideal (noiseless) Trotter simulation using measurement counts.

    Builds one transpiled circuit for each Trotter depth from 0 to
    ``params.layers_max - 1``, runs all circuits in a single batch job on
    ``AerSimulator``, and post-processes the measurement bit-strings to
    extract site magnetisations and connected correlators.

    Args:
        params: Physical model parameters.
        config: Run configuration.  ``backend_mode`` is ignored; the
                noiseless ``AerSimulator`` is always used.

    Returns:
        A tuple ``(magnetizations, correlators)`` where each element is a
        2-D ``np.ndarray`` of shape ``(layers_max, L)``.
    """
    backend = AerSimulator()
    u2, u1 = build_evolution_gates(params, config)

    # ---- Build and transpile all circuits up-front ----
    circuits = []
    for layers in range(params.layers_max):
        circ = init_circuit(params, config, with_classical=True)
        append_trotter_layer(circ, params, config, u2, u1, layers, measure=True)
        circuits.append(circ)

    transpiled = transpile(
        circuits,
        backend=backend,
        optimization_level=config.optimization_level,
    )

    # ---- Log circuit info for every layer ----
    log_file = write_log_header(params, config, "ideal_counts")
    for layer, circ in enumerate(transpiled):
        append_layer_log(log_file, layer, circ, "ideal_counts")

    # ---- Execute the whole batch in one shot ----
    result = backend.run(transpiled, shots=config.shots).result()

    # ---- Post-process each layer ----
    magnetizations: list[np.ndarray] = []
    correlators: list[np.ndarray] = []
    c = center_index(params)

    for layer in range(params.layers_max):
        counts = result.get_counts(layer)
        shots_total = sum(counts.values())
        mags, corr = _extract_magnetisations_and_correlators_from_counts(
            counts, shots_total, params
        )
        magnetizations.append(mags)
        correlators.append(corr)

    mag_array = np.array(magnetizations)
    corr_array = np.array(correlators)

    # Persist results to disk.
    save_simulation_data(params, config, mag_array, corr_array, "ideal_counts")

    return mag_array, corr_array


def run_ideal_simulation_observables(
    params: ModelParams,
    config: RunConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Run an ideal (noiseless) Trotter simulation using exact expectation values.

    For each Trotter depth the circuit is transpiled to ``['cx', 'rz', 'sx',
    'x']`` and then evaluated as a ``Statevector``.  This avoids shot noise
    entirely at the cost of exponential memory in ``L``.

    Args:
        params: Physical model parameters.
        config: Run configuration.

    Returns:
        A tuple ``(magnetizations, correlators)`` where each element is a
        2-D ``np.ndarray`` of shape ``(layers_max, L)``.
    """
    u2, u1 = build_evolution_gates(params, config)
    mag_obs = magnetisation_observables(params)
    corr_obs = correlator_observables(params)
    c = center_index(params)

    log_file = write_log_header(params, config, "ideal_observables")

    magnetizations: list[np.ndarray] = []
    correlators: list[np.ndarray] = []

    for layers in range(params.layers_max):
        # Build a measurement-free circuit for statevector evaluation.
        circ = init_circuit(params, config, with_classical=False)
        append_trotter_layer(circ, params, config, u2, u1, layers, measure=False)

        # Transpile to a standard basis for reproducible circuit structure.
        transpiled = transpile(circ, basis_gates=["cx", "rz", "sx", "x"])

        append_layer_log(log_file, layers, transpiled, "ideal_observables")

        # Evaluate the full statevector and compute exact expectations.
        state = Statevector.from_instruction(transpiled)

        mags = np.array(
            [np.real(state.expectation_value(obs)) for obs in mag_obs]
        )

        # Raw ⟨Z_c Z_j⟩ from the tensor-product observable.
        corr_raw = np.array(
            [np.real(state.expectation_value(obs)) for obs in corr_obs]
        )

        # Subtract the disconnected part ⟨Z_c⟩⟨Z_j⟩.
        corr_connected = corr_raw - mags[c] * mags

        magnetizations.append(mags)
        correlators.append(corr_connected)

    mag_array = np.array(magnetizations)
    corr_array = np.array(correlators)

    save_simulation_data(params, config, mag_array, corr_array, "ideal_observables")

    return mag_array, corr_array


# ============================================================================
# Noisy simulations
# ============================================================================


def run_noisy_simulation_counts(
    params: ModelParams,
    config: RunConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a noisy Trotter simulation using measurement counts.

    Constructs a density-matrix ``AerSimulator`` seeded from the noise model
    of the fake backend specified in ``config.fake_backend_name``, then
    follows the same batch-execution and count post-processing pipeline as
    :func:`run_ideal_simulation_counts`.

    Args:
        params: Physical model parameters.
        config: Run configuration.  ``fake_backend_name`` must be set.

    Returns:
        A tuple ``(magnetizations, correlators)`` where each element is a
        2-D ``np.ndarray`` of shape ``(layers_max, L)``.

    Raises:
        ValueError: If ``config.fake_backend_name`` is ``None``.
    """
    if config.fake_backend_name is None:
        raise ValueError(
            "config.fake_backend_name must be set for noisy simulations."
        )

    fake_backend = get_fake_backend(config.fake_backend_name)
    backend = make_noisy_density_simulator(fake_backend)
    u2, u1 = build_evolution_gates(params, config)

    label = f"noisy_counts_{config.fake_backend_name}"
    log_file = write_log_header(params, config, label)

    # ---- Build, transpile, and log circuits ----
    circuits = []
    for layers in range(params.layers_max):
        circ = init_circuit(params, config, with_classical=True)
        append_trotter_layer(circ, params, config, u2, u1, layers, measure=True)
        circuits.append(circ)

    transpiled = transpile(
        circuits,
        backend=backend,
        optimization_level=config.optimization_level,
    )

    for layer, circ in enumerate(transpiled):
        append_layer_log(log_file, layer, circ, "noisy_counts")

    # ---- Execute ----
    result = backend.run(transpiled, shots=config.shots).result()

    # ---- Post-process ----
    magnetizations: list[np.ndarray] = []
    correlators: list[np.ndarray] = []

    for layer in range(params.layers_max):
        counts = result.get_counts(layer)
        shots_total = sum(counts.values())
        mags, corr = _extract_magnetisations_and_correlators_from_counts(
            counts, shots_total, params
        )
        magnetizations.append(mags)
        correlators.append(corr)

    mag_array = np.array(magnetizations)
    corr_array = np.array(correlators)

    save_simulation_data(params, config, mag_array, corr_array, label)

    return mag_array, corr_array


def run_noisy_simulation_observables(
    params: ModelParams,
    config: RunConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a noisy Trotter simulation using density-matrix expectation values.

    For each Trotter depth a single circuit is transpiled onto the fake
    backend's coupling map and noise model, then run with
    ``save_density_matrix()`` to extract the full mixed state.  Physical
    observables are mapped through ``apply_layout`` to account for qubit
    remapping by the transpiler.

    Args:
        params: Physical model parameters.
        config: Run configuration.  ``fake_backend_name`` must be set.

    Returns:
        A tuple ``(magnetizations, correlators)`` where each element is a
        2-D ``np.ndarray`` of shape ``(layers_max, L)``.

    Raises:
        ValueError: If ``config.fake_backend_name`` is ``None``.
    """
    if config.fake_backend_name is None:
        raise ValueError(
            "config.fake_backend_name must be set for noisy simulations."
        )

    fake_backend = get_fake_backend(config.fake_backend_name)
    backend = make_noisy_density_simulator(fake_backend)
    u2, u1 = build_evolution_gates(params, config)

    # Build *logical* observables once; they will be re-mapped per layer.
    logical_mag_obs = magnetisation_observables(params)
    logical_corr_obs = correlator_observables(params)
    c = center_index(params)

    label = f"noisy_observables_{config.fake_backend_name}"
    log_file = write_log_header(params, config, label)

    magnetizations: list[np.ndarray] = []
    correlators: list[np.ndarray] = []

    for layers in range(params.layers_max):
        # Build and transpile one measurement-free circuit per depth.
        circ = init_circuit(params, config, with_classical=False)
        append_trotter_layer(circ, params, config, u2, u1, layers, measure=False)

        transpiled = transpile(
            circ,
            backend=backend,
            optimization_level=config.optimization_level,
        )

        append_layer_log(log_file, layers, transpiled, "noisy_observables")

        # Remap logical observables to the physical qubit layout chosen by
        # the transpiler so that expectation values are computed on the
        # correct physical qubits.
        mag_obs = [
            obs.apply_layout(transpiled.layout)
            for obs in logical_mag_obs
        ]
        corr_obs = [
            obs.apply_layout(transpiled.layout)
            for obs in logical_corr_obs
        ]

        # Request that Aer saves the density matrix in the result data dict.
        transpiled.save_density_matrix()

        result = backend.run(transpiled).result()
        rho = DensityMatrix(result.data(0)["density_matrix"])

        mags = np.array(
            [np.real(rho.expectation_value(obs)) for obs in mag_obs]
        )

        corr_raw = np.array(
            [np.real(rho.expectation_value(obs)) for obs in corr_obs]
        )

        # Subtract the disconnected part ⟨Z_c⟩⟨Z_j⟩.
        corr_connected = corr_raw - mags[c] * mags

        magnetizations.append(mags)
        correlators.append(corr_connected)

    mag_array = np.array(magnetizations)
    corr_array = np.array(correlators)

    save_simulation_data(params, config, mag_array, corr_array, label)

    return mag_array, corr_array
