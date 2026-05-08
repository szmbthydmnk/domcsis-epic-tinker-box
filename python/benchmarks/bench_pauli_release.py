"""
Layer 3 — Release benchmark with pyperf.

Run:
    python python/benchmarks/bench_pauli_release.py \
        --output python/benchmarks/results/release_v0.1.4.json

Compare:
    python -m pyperf compare_to \
        python/benchmarks/results/release_v0.1.4.json \
        python/benchmarks/results/release_v0.2.0.json
        
Ispect results:
    python -m pyperf stats python/benchmarks/results/release_v0.1.4.json
    python -m pyperf hist python/benchmarks/results/release_v0.1.4.json
"""

from __future__ import annotations

import sys
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PYTHON_DIR / "src"))

import pyperf  # noqa: E402
from domcsis_epic_tinker_box.pauli_algebra import (  # noqa: E402
    generate_pauli_operators,
)


def main() -> None:
    runner = pyperf.Runner()

    for n in [1, 2, 3, 4]:
        runner.bench_func(
            f"generate_pauli_operators(n={n})",
            generate_pauli_operators,
            n,
        )

    runner.bench_func(
        "generate_pauli_operators(n=3, as_dict=True)",
        generate_pauli_operators,
        3,
        True,
    )

    runner.bench_func(
        "generate_pauli_operators(subset=3 strings)",
        generate_pauli_operators,
        {"XII", "IZI", "YYX"},
    )


if __name__ == "__main__":
    main()