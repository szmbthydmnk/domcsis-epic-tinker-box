"""
Layer 2 — Scaling study.

Run:
    python python/benchmarks/bench_pauli_scaling.py
    python python/benchmarks/bench_pauli_scaling.py --max-n 6 --csv
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
import tracemalloc
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PYTHON_DIR / "benchmarks" / "results"
sys.path.insert(0, str(PYTHON_DIR / "src"))

from domcsis_epic_tinker_box.pauli_algebra import generate_pauli_operators  # noqa: E402


def time_and_memory(n_qubits: int, repeats: int = 5) -> tuple[float, float]:
    times: list[float] = []
    peak_kb = 0.0

    for _ in range(repeats):
        gc.collect()
        gc.disable()

        tracemalloc.start()
        t0 = time.perf_counter()

        generate_pauli_operators(n_qubits)

        t1 = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        gc.enable()

        times.append(t1 - t0)
        peak_kb = max(peak_kb, peak / 1024)

    return min(times), peak_kb


def run_scaling_study(max_n: int = 5, save_csv: bool = False) -> None:
    rows: list[tuple[int, int, str, float, float]] = []

    print("\n" + "=" * 65)
    print("  SCALING STUDY: generate_pauli_operators(n_qubits)")
    print("=" * 65)
    print(
        f"  {'n':>4}  {'operators':>12}  {'matrix size':>12}  "
        f"{'time (ms)':>12}  {'peak mem (KB)':>14}"
    )
    print("-" * 65)

    for n in range(1, max_n + 1):
        n_operators = 4**n
        matrix_size = f"{2**n}x{2**n}"
        t_min, mem_peak = time_and_memory(n)
        t_ms = t_min * 1000

        rows.append((n, n_operators, matrix_size, t_ms, mem_peak))
        print(
            f"  {n:>4}  {n_operators:>12}  {matrix_size:>12}  "
            f"{t_ms:>12.3f}  {mem_peak:>14.1f}"
        )

    print("=" * 65 + "\n")

    if save_csv:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = RESULTS_DIR / "scaling.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("n_qubits,n_operators,matrix_size,time_ms,peak_mem_kb\n")
            for n, ops, size, t, mem in rows:
                f.write(f"{n},{ops},{size},{t:.6f},{mem:.2f}\n")
        print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scaling study for generate_pauli_operators."
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=5,
        help="Maximum number of qubits to benchmark.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Save results to python/benchmarks/results/scaling.csv",
    )
    args = parser.parse_args()

    run_scaling_study(max_n=args.max_n, save_csv=args.csv)