[![Python pytest](https://github.com/szmbthydmnk/domcsis-epic-tinker-box/actions/workflows/python-pytest.yml/badge.svg)](https://github.com/szmbthydmnk/domcsis-epic-tinker-box/actions/workflows/python-pytest.yml)
[![PyPI version](https://img.shields.io/pypi/v/domcsis-epic-tinker-box)](https://pypi.org/project/domcsis-epic-tinker-box/)
[![python mypy](https://github.com/szmbthydmnk/domcsis-epic-tinker-box/actions/workflows/python-mypy.yml/badge.svg?branch=main&event=push)](https://github.com/szmbthydmnk/domcsis-epic-tinker-box/actions/workflows/python-mypy.yml)
[![Python Ruff](https://github.com/szmbthydmnk/domcsis-epic-tinker-box/actions/workflows/python-ruff.yml/badge.svg?branch=main&event=push)](https://github.com/szmbthydmnk/domcsis-epic-tinker-box/actions/workflows/python-ruff.yml)

---

[![Julia tests](https://github.com/szmbthydmnk/domcsis-epic-tinker-box/actions/workflows/julia-tests.yml/badge.svg)](https://github.com/szmbthydmnk/domcsis-epic-tinker-box/actions/workflows/julia-tests.yml)
[![Julia version](https://img.shields.io/badge/Julia%20registry-v0.1.0-blueviolet)](https://github.com/JuliaRegistries/General)
[![Julia format](https://github.com/szmbthydmnk/domcsis-epic-tinker-box/actions/workflows/julia-format.yml/badge.svg)](https://github.com/szmbthydmnk/domcsis-epic-tinker-box/actions/workflows/julia-format.yml)
[![Julia docs](https://github.com/szmbthydmnk/domcsis-epic-tinker-box/actions/workflows/julia-docs.yml/badge.svg)](https://github.com/szmbthydmnk/domcsis-epic-tinker-box/actions/workflows/julia-docs.yml)
[![Julia LoC metrics](https://github.com/szmbthydmnk/domcsis-epic-tinker-box/actions/workflows/julia-loc.yml/badge.svg)](https://github.com/szmbthydmnk/domcsis-epic-tinker-box/actions/workflows/julia-loc.yml)

# Domcsi's Epic Tinker Box
Hello stranger!

Domcsi's Epic Tinker Box is my little repository of Python and Julia libraries filled (gradually) with code and routines that I use throughout my coding, scientific and research journey.

This is also a playground for myself to learn and grow as a programmer.

Table of contents:
- [Description](#description)
- [Implemented Features (Python)](#implemented-features-python)
  - [`statistics`](#statistics)
  - [`pauli_algebra`](#pauli_algebra)
  - [`quantum_state`](#quantum_state)
  - [`magic`](#magic)
- [Metrics](#metrics)
  - [Number of Tests](#number-of-tests)
  - [LOC metrics](#loc-metrics)
---

## Implemented Features (Python)

### `statistics`

| Function | Description |
|---|---|
| `generate_gaussian_rnd_numbers` | Samples from 𝒩(μ, σ²) |
| `generate_uniform_rnd_numbers` | Samples from Uniform(a, b) |
| `generate_exponential_rnd_numbers` | Samples from Exp(λ) |
| `generate_wigner_surmise_rnd_numbers` | Samples from the generalised Wigner surmise (GOE/GUE/GSE and beyond) |
| `generate_random_numbers` | Unified dispatch wrapper for all distributions |
| `mean_real` | Population mean of real samples |
| `mean_complex` | Population mean of complex samples; returns `(real_part, complex_mean)` |
| `variance` | Population variance (N denominator) |
| `skewness` | Pearson's moment coefficient of skewness |

All samplers support optional `rng` injection or `seed` for reproducible, non-global-state sampling.

---

### `pauli_algebra`

| Function / Class | Description |
|---|---|
| `pauli_matrix(which)` | Single-qubit Pauli matrix in sparse CSR format; accepts string (`"I"`,`"X"`,`"Y"`,`"Z"`) or integer (0–3) |
| `generate_all_pauli_strings(n)` | All 4ⁿ Pauli strings for n qubits in lexicographic order |
| `generate_pauli_operators(n)` | All 4ⁿ sparse Pauli matrices; returns list or dict |
| `generate_pauli_operators(set)` | Sparse matrices for an explicit subset of Pauli strings |
| `PauliOperators` | Immutable validated collection; factories `PauliOperators.all(n)` and `PauliOperators.from_strings(set)`; iterable as `(label, matrix)` pairs |

---

### `quantum_state`

| Constructor | Canonical form | Notes |
|---|---|---|
| `QuantumState.from_vector(ψ)` | State vector `(2ᴺ,)` | Pure states only |
| `QuantumState.from_density_matrix(ρ)` | Density matrix `(2ᴺ, 2ᴺ)` | Pure and mixed |
| `QuantumState.from_vectorized_density_matrix(ρ_vec)` | vec(ρ) `(4ᴺ,)` | Pure and mixed |

Derived quantities computed on demand from the canonical form:

| Property | Returns |
|---|---|
| `.vector` | State vector; raises for mixed states |
| `.density_matrix` | ρ = \|ψ⟩⟨ψ\| or stored ρ |
| `.purity` | Tr(ρ²); `1.0` immediately for vector-canonical states |
| `.dim` | Hilbert space dimension 2ᴺ |

---

### `magic`

| Function | Description |
|---|---|
| `stabilizer_renyi_entropy(state, paulis, α)` | α-Stabilizer Rényi Entropy M_α; validated entry point |
| `_stabilizer_renyi_entropy_unchecked(...)` | Fast path for hot loops; skip validation after a single upfront check |
| `characteristic_function(ψ, paulis)` | Ξ(P) = \|⟨ψ\|P\|ψ⟩\|² / 2ⁿ for all P |
| `validate_alpha(α)` | Standalone α validator; warns for α=1 and non-integer α |
| `validate_compatible(state, paulis)` | Checks n_qubits agreement |

`stabilizer_renyi_entropy` returns a `float` by default, or a `(float, dict[str, float])` Pauli spectrum tuple when `pauli_spectrum=True`.

---

## Metrics

### Number of Tests

- Python tests: <!-- PYTHON_TESTS -->293 tests<!-- /PYTHON_TESTS -->

### LOC metrics
Now this is not important, and I do not think that it is a good idea to attribute quality, effort or productiveness to the following metric, but it is like the first, easy-peasy GitHub Action that one can set up, so here it is:

```
LOC is lines of code
LOCo is lines of comments
code share is LOC/(LOC + LOCo)
```

- Python: <!-- PYTHON_METRIC -->1807 LOC, code share = 71.7%<!-- /PYTHON_METRIC -->
- Julia: <!-- JULIA_METRIC -->36 LOC, code share = 97.3%<!-- /JULIA_METRIC -->

---
