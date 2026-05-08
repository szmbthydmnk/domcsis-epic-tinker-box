"""
Benchmark suite for generate_pauli_operators.

Layer 1 — Micro-benchmarks (pytest-benchmark)
    Run with:   pytest python/benchmarks/bench_pauli_operators.py -v
    Compare:    pytest python/benchmarks/bench_pauli_operators.py --benchmark-compare
    Save:       pytest python/benchmarks/bench_pauli_operators.py --benchmark-save=baseline

Layer 2 — Scaling study (standalone script)
    Run with:   python python/benchmarks/bench_pauli_operators.py
                python python/benchmarks/bench_pauli_operators.py --max-n 6 --csv

Layer 3 — Release benchmark (pyperf, statistical)
    Run with:   python python/benchmarks/bench_pauli_operators.py --release
    Compare:    python -m pyperf compare_to results/release_v0.1.4.json results/release_v0.2.0.json

Dependencies:
    pip install pytest pytest-benchmark pyperf
"""

# =============================================================================
# Why these imports?
#
# pytest-benchmark:
#   Provides the `benchmark` fixture that wraps a callable, handles warm-up,
#   disables GC during measurement, repeats automatically, and reports
#   min/mean/stddev/median. Integrates with pytest so benchmarks live
#   alongside tests but are only run when explicitly requested.
#
# pyperf:
#   The benchmarking library used by CPython's own benchmark suite. Runs
#   the benchmark in a subprocess (to get a clean process state), calibrates
#   the number of iterations automatically, and detects unstable results.
#   Produces a JSON output that can be compared with `python -m pyperf compare`.
#
#   IMPORTANT — pyperf.Runner() hijacks sys.argv at instantiation time.
#   This means it must never be instantiated before our own argparse has
#   finished and cleaned up sys.argv. See _run_release_benchmark() for how
#   we handle this.
#
# tracemalloc:
#   Python stdlib memory tracer. Captures peak memory allocated during a
#   code block at the level of individual Python objects. No external
#   dependency. Used for Layer 2 memory reporting.
#
# time.perf_counter:
#   Monotonic, high-resolution wall-clock timer. The correct clock for
#   benchmarking — unlike time.time() it is not affected by system clock
#   adjustments (NTP, DST). Resolution is typically < 100 ns on modern
#   hardware.
# =============================================================================

from __future__ import annotations

import argparse
import gc
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# sys.path manipulation is necessary here to make the package importable when
# this file is run directly as a script (Layer 2 / 3), without requiring the
# package to be installed via pip.
#
# File location:  python/benchmarks/bench_pauli_operators.py
# .parent         → python/benchmarks/
# .parent.parent  → python/
# .parent.parent.parent → repo root
#
# We add python/src to sys.path so Python can find domcsis_epic_tinker_box.
#
# The # noqa: E402 comments below suppress ruff's E402 warning
# ("module level import not at top of file") for the package imports that
# follow the sys.path modification. This is a well-known and accepted
# exception — the path modification MUST precede the import it enables.
# The per-file ignore in pyproject.toml [tool.ruff.lint.per-file-ignores]
# provides the same suppression project-wide for this file.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "python" / "src"))

from domcsis_epic_tinker_box.pauli_algebra import (  # noqa: E402
    generate_all_pauli_strings,
    generate_pauli_operators,
)


# =============================================================================
# LAYER 1 — Micro-benchmarks (pytest-benchmark)
#
# Each function prefixed with `test_` is discovered by pytest.
# The `benchmark` argument is a pytest-benchmark fixture — it wraps the
# callable, runs it repeatedly, collects timings, disables GC, and handles
# warm-up automatically.
#
# Key pytest-benchmark concepts:
#
#   benchmark(fn, *args)
#       Calls fn(*args) in a loop, measures wall time per call.
#       Automatically calibrates the number of rounds and iterations.
#
#   benchmark.pedantic(fn, args=(), rounds=20, warmup_rounds=2)
#       More control: explicit round and warm-up counts.
#       Use this when the function has setup cost that should not be counted.
#
# Why separate benchmarks for each n?
#   generate_pauli_operators(n) has exponential cost O(4^n * 2^n * 2^n).
#   A single benchmark at n=3 tells you nothing about scaling behaviour.
#   Benchmarking n=1 through n=5 lets you verify that the exponent is
#   correct and catch any super-exponential regression.
#
# Why benchmark list vs dict separately?
#   The two code paths in generate_pauli_operators differ only in whether
#   they build a dict or a list. The dict path does one extra string lookup
#   per matrix. If that overhead is significant (it should not be, but worth
#   verifying), it shows up here.
# =============================================================================


def test_bench_single_qubit(benchmark: Any) -> None:
    """Micro-benchmark: all 4^1 = 4 single-qubit Pauli operators (list)."""
    benchmark(generate_pauli_operators, 1)


def test_bench_two_qubit(benchmark: Any) -> None:
    """Micro-benchmark: all 4^2 = 16 two-qubit Pauli operators (list)."""
    benchmark(generate_pauli_operators, 2)


def test_bench_three_qubit(benchmark: Any) -> None:
    """Micro-benchmark: all 4^3 = 64 three-qubit Pauli operators (list)."""
    benchmark(generate_pauli_operators, 3)


def test_bench_four_qubit(benchmark: Any) -> None:
    """Micro-benchmark: all 4^4 = 256 four-qubit Pauli operators (list)."""
    benchmark(generate_pauli_operators, 4)


def test_bench_three_qubit_as_dict(benchmark: Any) -> None:
    """
    Micro-benchmark: 4^3 = 64 three-qubit Pauli operators returned as dict.

    Compared against test_bench_three_qubit to isolate dict construction
    overhead vs list construction overhead.
    """
    benchmark(generate_pauli_operators, 3, True)


def test_bench_subset_small(benchmark: Any) -> None:
    """
    Micro-benchmark: set[str] path with a small subset (3 operators).

    Verifies that the set path avoids computing all 4^3 = 64 operators
    when only 3 are needed. Should be ~20x faster than test_bench_three_qubit.
    """
    subset = {"XII", "IZI", "YYX"}
    benchmark(generate_pauli_operators, subset)


def test_bench_subset_full(benchmark: Any) -> None:
    """
    Micro-benchmark: set[str] path with ALL 3-qubit strings.

    This is the worst case for the set path — it generates the same operators
    as test_bench_three_qubit but via a different code path. Comparing the two
    reveals the overhead of the set path (dict construction, string validation)
    vs the int path (list construction, no validation per string).
    """
    all_3q = set(generate_all_pauli_strings(3))
    benchmark(generate_pauli_operators, all_3q)


# =============================================================================
# LAYER 2 — Scaling study (standalone)
#
# Measures wall time AND peak memory across n_qubits = 1..max_n.
# Outputs a human-readable table and optionally a CSV for plotting.
#
# Why manual timing here instead of pytest-benchmark?
#   pytest-benchmark is designed for single-function micro-benchmarks.
#   A scaling study requires running the same function at many input sizes
#   and collecting structured output (a table / CSV). That is easier to
#   do with explicit loops and time.perf_counter() than with pytest fixtures.
#
# Why disable GC explicitly?
#   Python's cyclic garbage collector can fire unpredictably mid-measurement,
#   inflating timings. We disable it before each measurement and re-enable
#   it after. This is the same thing pytest-benchmark does internally.
#
# Why use tracemalloc for memory?
#   scipy.sparse matrices allocate memory in C extensions. tracemalloc
#   captures Python-level allocations, which includes the Python objects
#   wrapping the C arrays but not the raw C buffers themselves. It therefore
#   gives a lower bound on memory usage, not the true peak. For true peak
#   memory (including C allocations) you would need memory_profiler or
#   /proc/self/status on Linux. For our purposes tracemalloc is sufficient
#   to observe the scaling trend.
#
# Expected scaling:
#   Time:   O(4^n) operators, each built by (n-1) kron products of 2^k x 2^k
#           sparse matrices. Total cost roughly O(n * 4^n * 4^n) = O(n * 16^n).
#   Memory: Each operator is 2^n x 2^n complex128. For n=5: 32x32 = 1024
#           elements * 16 bytes = 16 KB per operator, 1024 operators = 16 MB.
# =============================================================================

def _time_and_memory(n_qubits: int, repeats: int = 5) -> tuple[float, float]:
    """
    Measure minimum wall time (seconds) and peak memory (KB) for
    generate_pauli_operators(n_qubits) over `repeats` independent runs.

    Returns:
        (min_time_seconds, peak_memory_kb)
    """
    times: list[float] = []

    for _ in range(repeats):
        # Collect garbage before each run so GC pressure from the previous
        # run does not affect this one.
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

    # Min time is the best estimate of true execution cost — it is the run
    # least affected by OS scheduling noise and GC pauses.
    return min(times), peak / 1024


def _run_scaling_study(max_n: int = 5, save_csv: bool = False) -> None:
    """
    Run the scaling study for n_qubits = 1..max_n and print a results table.

    Args:
        max_n:     Maximum number of qubits to benchmark.
        save_csv:  If True, write results to benchmarks/results/scaling.csv
    """
    print("\n" + "=" * 65)
    print("  SCALING STUDY: generate_pauli_operators(n_qubits)")
    print("=" * 65)
    print(f"  {'n':>4}  {'operators':>12}  {'matrix size':>12}  "
          f"{'time (ms)':>12}  {'peak mem (KB)':>14}")
    print("-" * 65)

    rows: list[tuple[int, int, str, float, float]] = []

    for n in range(1, max_n + 1):
        n_operators = 4 ** n
        matrix_size = f"{2**n}x{2**n}"
        t_min, mem_peak = _time_and_memory(n)
        t_ms = t_min * 1000

        rows.append((n, n_operators, matrix_size, t_ms, mem_peak))
        print(f"  {n:>4}  {n_operators:>12}  {matrix_size:>12}  "
              f"{t_ms:>12.3f}  {mem_peak:>14.1f}")

    print("=" * 65 + "\n")

    if save_csv:
        csv_path = _REPO_ROOT / "python" / "benchmarks" / "results" / "scaling.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w") as f:
            f.write("n_qubits,n_operators,matrix_size,time_ms,peak_mem_kb\n")
            for n, ops, size, t, mem in rows:
                f.write(f"{n},{ops},{size},{t:.6f},{mem:.2f}\n")
        print(f"  Results saved to {csv_path}")


# =============================================================================
# LAYER 3 — Release benchmark (pyperf)
#
# pyperf is the tool used by CPython's own benchmark suite (pyperformance).
# It is the most statistically rigorous Python benchmarking tool available.
#
# What makes pyperf different from timeit/pytest-benchmark:
#
#   1. Subprocess isolation
#      Each benchmark run is executed in a fresh subprocess. This eliminates
#      JIT warm-up effects, import caching, and memory fragmentation from
#      previous runs contaminating results.
#
#   2. Automatic calibration
#      pyperf measures how many inner iterations fit in a target time window
#      and adjusts automatically. You do not need to guess `number=`.
#
#   3. Stability detection
#      pyperf checks whether the distribution of timings is stable. If
#      standard deviation > 10% of mean, it warns you that results are noisy
#      and should not be trusted. timeit never does this.
#
#   4. JSON output
#      Results are saved as machine-readable JSON. You can then run:
#        python -m pyperf compare_to baseline.json new.json
#      to get a statistically grounded comparison with confidence intervals.
#
# IMPORTANT — pyperf.Runner() sys.argv conflict:
#   pyperf.Runner() immediately calls argparse on sys.argv when instantiated.
#   If our own "--release" flag is still in sys.argv at that point, pyperf
#   raises "unrecognized arguments: --release" and exits.
#
#   The fix has two parts:
#     1. Parse and remove our own flags from sys.argv BEFORE instantiating
#        pyperf.Runner(). We do this by resetting sys.argv to only the script
#        name before calling pyperf.Runner().
#     2. Pass the output file path to pyperf via its own --output flag by
#        injecting it into sys.argv after we have cleared our own flags.
#        pyperf natively understands --output and writes the JSON there.
#
# How to run:
#   python python/benchmarks/bench_pauli_operators.py --release
#   python -m pyperf compare_to benchmarks/results/release_v0.1.4.json \
#                                benchmarks/results/release_v0.2.0.json
# =============================================================================

def _run_release_benchmark(output_path: Path) -> None:
    """
    Run the Layer 3 release benchmark using pyperf.

    Args:
        output_path: Path to write the pyperf JSON results file.
    """
    try:
        import pyperf  # type: ignore[import-untyped]  # noqa: PLC0415
    except ImportError:
        print("pyperf is not installed. Run: pip install pyperf")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Reconstruct sys.argv with only the script name and pyperf's own --output
    # flag. pyperf.Runner() will parse this clean argv and write results to the
    # specified file. Without this reset, pyperf sees "--release" and crashes.
    sys.argv = [sys.argv[0], "--output", str(output_path)]

    runner = pyperf.Runner()

    # pyperf.Runner.bench_func() takes a callable and its arguments.
    # It handles subprocess isolation, calibration, and JSON output internally.
    # Each call below produces one named benchmark entry in the output JSON.
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

    print(f"\nResults saved to: {output_path}")


# =============================================================================
# Entry point
#
# When run as a script:
#   python benchmarks/bench_pauli_operators.py           → Layer 2 scaling study
#   python benchmarks/bench_pauli_operators.py --release → Layer 3 pyperf
#   python benchmarks/bench_pauli_operators.py --csv     → Layer 2 + save CSV
#
# When imported by pytest:
#   Only the test_bench_* functions are collected (Layer 1).
#   The __main__ block is never executed.
# =============================================================================

if __name__ == "__main__":
    # Parse our own flags FIRST, before pyperf can touch sys.argv.
    parser = argparse.ArgumentParser(
        description="Benchmark suite for generate_pauli_operators."
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="Run Layer 3 release benchmark using pyperf.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Save Layer 2 scaling study results to CSV.",
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=5,
        help="Maximum number of qubits for the scaling study (default: 5).",
    )
    args = parser.parse_args()

    if args.release:
        # Derive the output path from the package version if available.
        try:
            from domcsis_epic_tinker_box import __version__  # type: ignore[attr-defined]  # noqa: PLC0415
            version_tag = f"v{__version__}"
        except ImportError:
            version_tag = "unknown"

        output_path = (
            _REPO_ROOT
            / "python"
            / "benchmarks"
            / "results"
            / f"release_{version_tag}.json"
        )
        _run_release_benchmark(output_path)
    else:
        _run_scaling_study(max_n=args.max_n, save_csv=args.csv)
