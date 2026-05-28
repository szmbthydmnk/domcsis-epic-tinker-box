from dataclasses import asdict
from pathlib import Path
import numpy as np

from .model import ModelParams, RunConfig


def make_data_filename(
    params: ModelParams,
    config: RunConfig,
    run_label: str,
    base_dir: str | Path = "simulation_data",
):
    data_dir = Path(base_dir)
    data_dir.mkdir(exist_ok=True)

    filename = (
        f"N_{params.L}_"
        f"layers_{params.layers_max}_"
        f"J_{params.J}_"
        f"ht_{params.ht}_"
        f"hl_{params.hl}_"
        f"dt_{params.dt}_"
        f"{run_label}.txt"
    )
    return data_dir / filename


def save_simulation_data(
    params: ModelParams,
    config: RunConfig,
    magnetizations: np.ndarray,
    correlators: np.ndarray,
    run_label: str,
    base_dir: str | Path = "simulation_data",
):
    data_file = make_data_filename(params, config, run_label, base_dir=base_dir)

    with open(data_file, "w", encoding="utf-8") as f:
        f.write("=== Simulation configuration ===\n\n")
        f.write("Model parameters:\n")
        for key, value in asdict(params).items():
            f.write(f"{key}: {value}\n")

        f.write("\nRun configuration:\n")
        for key, value in asdict(config).items():
            f.write(f"{key}: {value}\n")

        f.write("\n=== Magnetizations ===\n")
        np.savetxt(f, magnetizations)

        f.write("\n=== Correlators ===\n")
        np.savetxt(f, correlators)

    return data_file


def load_simulation_data(filename: str | Path):
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    mag_start = None
    corr_start = None

    for i, line in enumerate(lines):
        if "=== Magnetizations ===" in line:
            mag_start = i + 1
        elif "=== Correlators ===" in line:
            corr_start = i + 1

    if mag_start is None or corr_start is None:
        raise ValueError("Could not find magnetization/correlator sections in file.")

    mag_lines = lines[mag_start : corr_start - 1]
    corr_lines = lines[corr_start:]

    magnetizations = np.loadtxt(mag_lines)
    correlators = np.loadtxt(corr_lines)

    return magnetizations, correlators


def make_log_filename(
    params: ModelParams,
    config: RunConfig,
    run_label: str,
    base_dir: str | Path = "simulation_logs",
):
    log_dir = Path(base_dir)
    log_dir.mkdir(exist_ok=True)

    filename = (
        f"N_{params.L}_"
        f"layers_{params.layers_max}_"
        f"{run_label}.txt"
    )
    return log_dir / filename


def log_simulation_header(
    params: ModelParams,
    config: RunConfig,
    run_label: str,
    base_dir: str | Path = "simulation_logs",
):
    log_file = make_log_filename(params, config, run_label, base_dir=base_dir)

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=== Simulation configuration ===\n\n")
        f.write("Model parameters:\n")
        for key, value in asdict(params).items():
            f.write(f"{key}: {value}\n")

        f.write("\nRun configuration:\n")
        for key, value in asdict(config).items():
            f.write(f"{key}: {value}\n")

        f.write("\n=== Circuit information ===\n\n")

    return log_file


def append_layer_log(log_file, layer, circuit, sim_type):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(
            f"{sim_type:>12} | "
            f"layer={layer:2d} | "
            f"depth={circuit.depth():4d} | "
            f"ops={dict(circuit.count_ops())}\n"
        )
        f.write("\n")
        f.write(f"--- Circuit layer {layer} ---\n")
        circuit_text = circuit.draw(output="text", fold=120, idle_wires=False)
        f.write(str(circuit_text))
        f.write("\n\n")
        f.write("=" * 100)
        f.write("\n\n")
