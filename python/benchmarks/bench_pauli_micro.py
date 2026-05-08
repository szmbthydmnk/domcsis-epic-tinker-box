"""
Layer 1 — Micro-benchmarks with pytest-benchmark.

Run:
    pytest python/benchmarks/bench_pauli_micro.py -v
    pytest python/benchmarks/bench_pauli_micro.py --benchmark-compare
    pytest python/benchmarks/bench_pauli_micro.py --benchmark-save=baseline
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PYTHON_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PYTHON_DIR / "src"))

from domcsis_epic_tinker_box.pauli_algebra import (  # noqa: E402
    generate_all_pauli_strings,
    generate_pauli_operators,
)


def test_bench_single_qubit(benchmark: Any) -> None:
    benchmark(generate_pauli_operators, 1)


def test_bench_two_qubit(benchmark: Any) -> None:
    benchmark(generate_pauli_operators, 2)


def test_bench_three_qubit(benchmark: Any) -> None:
    benchmark(generate_pauli_operators, 3)


def test_bench_four_qubit(benchmark: Any) -> None:
    benchmark(generate_pauli_operators, 4)


def test_bench_three_qubit_as_dict(benchmark: Any) -> None:
    benchmark(generate_pauli_operators, 3, True)


def test_bench_subset_small(benchmark: Any) -> None:
    subset = {"XII", "IZI", "YYX"}
    benchmark(generate_pauli_operators, subset)


def test_bench_subset_full(benchmark: Any) -> None:
    all_3q = set(generate_all_pauli_strings(3))
    benchmark(generate_pauli_operators, all_3q)