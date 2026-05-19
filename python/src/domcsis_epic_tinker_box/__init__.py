"""Domcsi's Epic Tinker Box — quantum simulation and statistics utilities."""

from .pauli_algebra import (
    pauli_matrix,
    generate_all_pauli_strings,
    generate_pauli_operators,
    PauliOperators,
)
from .quantum_state import QuantumState
from .magic import (
    stabilizer_renyi_entropy,
)
from .statistics import (
    generate_gaussian_rnd_numbers,
    generate_uniform_rnd_numbers,
    generate_exponential_rnd_numbers,
    generate_wigner_surmise_rnd_numbers,
    generate_random_numbers,
    mean,
    mean_complex,
    variance,
    skewness,
)

__all__ = [
    # pauli_algebra
    "pauli_matrix",
    "generate_all_pauli_strings",
    "generate_pauli_operators",
    "PauliOperators",
    # quantum_state
    "QuantumState",
    # magic
    "stabilizer_renyi_entropy",
    # statistics
    "generate_gaussian_rnd_numbers",
    "generate_uniform_rnd_numbers",
    "generate_exponential_rnd_numbers",
    "generate_wigner_surmise_rnd_numbers",
    "generate_random_numbers",
    "mean",
    "mean_complex",
    "variance",
    "skewness",
]
