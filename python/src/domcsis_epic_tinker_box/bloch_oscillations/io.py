"""
I/O helpers for the Bloch-oscillations simulation.

This module is responsible for every side-effect involving the file system:

* **Simulation data** — saving and loading the ``magnetizations`` and
  ``correlators`` arrays produced by the simulation drivers.
* **Circuit logs** — writing a structured text log that records the
  configuration header and, for each Trotter layer, the circuit depth,
  gate counts, and a human-readable circuit diagram.

Separating I/O from computation keeps the simulation drivers in
``simulation.py`` pure enough to be tested without touching the file system.

Directory layout
----------------
Data files are written to ``simulation_data/``, log files to
``simulation_logs/``.  Both directories are created on first use.

Filename conventions
--------------------
Data files:  ``N_{L}_layers_{layers_max}_J_{J}_ht_{ht}_hl_{hl}_dt_{dt}_{label}.txt``
Log files:   ``N_{L}_layers_{layers_max}_{label}.txt``
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .model import ModelParams, RunConfig


# ============================================================================
# Data file helpers
# ============================================================================


def _data_filename(
    params: ModelParams,
    config: RunConfig,  # noqa: ARG001  — reserved for future per-config filenames
    run_label: str,
) -> Path:
    """Construct the full path for a simulation-data file.

    The directory ``simulation_data/`` is created if it does not exist.

    Args:
        params:    Model parameters used to build the filename stem.
        config:    Run configuration (currently unused in the stem but
                   kept for API consistency with future extensions).
        run_label: Short string identifying the simulation variant
                   (e.g. ``"ideal_counts"`` or ``"noisy_brisbane"``).

    Returns:
        A ``pathlib.Path`` pointing to the target file.
    """
    data_dir = Path("simulation_data")
    data_dir.mkdir(exist_ok=True)

    # Embed all physically relevant parameters in the filename so that
    # results from different parameter sweeps never collide.
    stem = (
        f"N_{params.L}_"
        f"layers_{params.layers_max}_"
        f"J_{params.J}_"
        f"ht_{params.ht}_"
        f"hl_{params.hl}_"
        f"dt_{params.dt}_"
        f"{run_label}.txt"
    )
    return data_dir / stem


def save_simulation_data(
    params: ModelParams,
    config: RunConfig,
    magnetizations: np.ndarray,
    correlators: np.ndarray,
    run_label: str,
) -> Path:
    """Write magnetisation and correlator arrays to a plain-text file.

    The file begins with a human-readable configuration header so that a
    data file is self-describing.  The two arrays are separated by a labelled
    section marker so that :func:`load_simulation_data` can reliably parse
    them back.

    Args:
        params:         Physical model parameters.
        config:         Run configuration.
        magnetizations: 2-D array of shape ``(layers_max, L)`` containing
                        ⟨Z_j⟩ at each Trotter step.
        correlators:    2-D array of shape ``(layers_max, L)`` containing
                        connected correlators at each Trotter step.
        run_label:      Short identifier used in the filename and header.

    Returns:
        The ``Path`` to the written file.
    """
    path = _data_filename(params, config, run_label)

    with open(path, "w") as fh:
        # --- Configuration header ---
        fh.write("=== Simulation configuration ===\n\n")

        fh.write("Model parameters:\n")
        for key, value in asdict(params).items():
            fh.write(f"{key}: {value}\n")

        fh.write("\nRun configuration:\n")
        for key, value in asdict(config).items():
            fh.write(f"{key}: {value}\n")

        # --- Numeric payload ---
        fh.write("\n=== Magnetizations ===\n")
        np.savetxt(fh, magnetizations)

        fh.write("\n=== Correlators ===\n")
        np.savetxt(fh, correlators)

    return path


def load_simulation_data(filename: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load magnetisation and correlator arrays from a simulation-data file.

    Parses a file written by :func:`save_simulation_data`.  The function
    scans for the ``=== Magnetizations ===`` and ``=== Correlators ===``
    section markers and extracts the numeric blocks between them.

    Args:
        filename: Path to the ``.txt`` file produced by
                  :func:`save_simulation_data`.

    Returns:
        A tuple ``(magnetizations, correlators)`` where each element is a
        2-D ``np.ndarray`` of shape ``(layers_max, L)``.

    Raises:
        FileNotFoundError: If ``filename`` does not exist.
        ValueError:        If the expected section markers are not found
                           in the file.
    """
    with open(filename) as fh:
        lines = fh.readlines()

    # Locate the line indices of the two section markers.
    mag_start: int | None = None
    corr_start: int | None = None
    for i, line in enumerate(lines):
        if "=== Magnetizations ===" in line:
            mag_start = i + 1
        elif "=== Correlators ===" in line:
            corr_start = i + 1

    if mag_start is None or corr_start is None:
        raise ValueError(
            f"Could not find section markers in '{filename}'. "
            "Is this a valid simulation-data file?"
        )

    # Slice out the text rows for each block and parse with NumPy.
    mag_lines = lines[mag_start : corr_start - 1]
    corr_lines = lines[corr_start:]

    magnetizations = np.loadtxt(mag_lines)  
    correlators = np.loadtxt(corr_lines)  

    return magnetizations, correlators


# ============================================================================
# Circuit log helpers
# ============================================================================


def _log_filename(
    params: ModelParams,
    config: RunConfig,  # noqa: ARG001
    run_label: str,
) -> Path:
    """Construct the full path for a simulation-log file.

    The directory ``simulation_logs/`` is created if it does not exist.

    Args:
        params:    Model parameters (``L`` and ``layers_max`` used in stem).
        config:    Run configuration (reserved for future use).
        run_label: Short identifier for the run variant.

    Returns:
        A ``pathlib.Path`` pointing to the target log file.
    """
    log_dir = Path("simulation_logs")
    log_dir.mkdir(exist_ok=True)

    stem = f"N_{params.L}_layers_{params.layers_max}_{run_label}.txt"
    return log_dir / stem


def write_log_header(
    params: ModelParams,
    config: RunConfig,
    run_label: str,
) -> Path:
    """Create a log file and write the configuration header.

    Must be called once before :func:`append_layer_log` is called for the
    first time.  The file is (over-)written so that re-running a simulation
    with the same parameters produces a clean log.

    Args:
        params:    Model parameters.
        config:    Run configuration.
        run_label: Short identifier used in the filename.

    Returns:
        The ``Path`` to the created log file (pass it to
        :func:`append_layer_log`).
    """
    log_file = _log_filename(params, config, run_label)

    with open(log_file, "w") as fh:
        fh.write("=== Simulation configuration ===\n\n")

        fh.write("Model parameters:\n")
        for key, value in asdict(params).items():
            fh.write(f"{key}: {value}\n")

        fh.write("\nRun configuration:\n")
        for key, value in asdict(config).items():
            fh.write(f"{key}: {value}\n")

        fh.write("\n=== Circuit information ===\n\n")

    return log_file


def append_layer_log(
    log_file: Path,
    layer: int,
    circuit: Any,
    sim_type: str,
) -> None:
    """Append a single-layer entry to an existing circuit log file.

    Records the circuit depth, gate-count dictionary, and a 120-character-
    wide text diagram of the transpiled circuit.

    Args:
        log_file: Path returned by :func:`write_log_header`.
        layer:    Zero-based Trotter-layer index.
        circuit:  Transpiled ``QuantumCircuit`` for this layer.  Typed as
                  ``Any`` because ``qiskit`` ships no PEP 561 stubs; the
                  concrete type is ``qiskit.QuantumCircuit`` at runtime.
        sim_type: Short label for the simulation type column
                  (e.g. ``"ideal_counts"`` or ``"noisy"``).
    """
    with open(log_file, "a") as fh:
        # Summary line: type, layer index, depth, and gate counts.
        fh.write(
            f"{sim_type:>16} | "
            f"layer={layer:2d} | "
            f"depth={circuit.depth():4d} | "  
            f"ops={dict(circuit.count_ops())}\n"  
        )
        fh.write("\n")
        fh.write(f"--- Circuit layer {layer} ---\n")

        # Use Qiskit's text drawer with a generous line width so multi-qubit
        # connections are not broken across lines unnecessarily.
        circuit_text = circuit.draw(output="text", fold=120, idle_wires=False)  
        fh.write(str(circuit_text))
        fh.write("\n\n")
        fh.write("=" * 100)
        fh.write("\n\n")
