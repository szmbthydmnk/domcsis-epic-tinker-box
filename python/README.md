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
- [Implemented Features](#implemented-features)
  - [`statistics`](#statistics)
  - [`pauli_algebra`](#pauli_algebra)
  - [`quantum_state`](#quantum_state)
  - [`magic`](#magic)
- [Metrics](#metrics)
  - [Number of Tests](#number-of-tests)
  - [LOC metrics](#loc-metrics)
---

## Implemented Features

### `statistics`

| Function | Description | Python | Julia |
|---|---|:---:|:---:|
| `mean` | Arithmetic mean of a numeric vector | ✅ | ✅ |
| `mean_complex` | Population mean of complex samples; returns `(real_part, complex_mean)` | ✅ | ❌ |
| `variance` | Population variance (N denominator) | ✅ | ❌ |
| `skewness` | Pearson's moment coefficient of skewness | ✅ | ❌ |
| `generate_gaussian_rnd_numbers` | Samples from 𝒩(μ, σ²) | ✅ | ❌ |
| `generate_uniform_rnd_numbers` | Samples from Uniform(a, b) | ✅ | ❌ |
| `generate_exponential_rnd_numbers` | Samples from Exp(λ) | ✅ | ❌ |
| `generate_wigner_surmise_rnd_numbers` | Samples from the generalised Wigner surmise (GOE/GUE/GSE and beyond) | ✅ | ❌ |
| `generate_random_numbers` | Unified dispatch wrapper for all distributions | ✅ | ❌ |

All Python samplers support optional `rng` injection or `seed` for reproducible, non-global-state sampling.

---

### `pauli_algebra`

| Function / Class | Description | Python | Julia |
|---|---|:---:|:---:|
| `pauli_matrix(which)` | Single-qubit Pauli matrix in sparse CSR format | ✅ | ❌ |
| `generate_all_pauli_strings(n)` | All 4ⁿ Pauli strings for n qubits in lexicographic order | ✅ | ❌ |
| `generate_pauli_operators(n)` | All 4ⁿ sparse Pauli matrices | ✅ | ❌ |
| `PauliOperators` | Immutable validated collection with factories `all(n)` and `from_strings(set)` | ✅ | ❌ |

---

### `quantum_state`

| Constructor | Description | Python | Julia |
|---|---|:---:|:---:|
| `QuantumState.from_vector(ψ)` | Pure state from state vector `(2ᴺ,)` | ✅ | ❌ |
| `QuantumState.from_density_matrix(ρ)` | State from density matrix `(2ᴺ, 2ᴺ)` | ✅ | ❌ |
| `QuantumState.from_vectorized_density_matrix(ρ_vec)` | State from vec(ρ) `(4ᴺ,)` | ✅ | ❌ |

Derived quantities computed on demand:

| Property | Returns | Python | Julia |
|---|---|:---:|:---:|
| `.vector` | State vector; raises for mixed states | ✅ | ❌ |
| `.density_matrix` | ρ = \|ψ⟩⟨ψ\| or stored ρ | ✅ | ❌ |
| `.purity` | Tr(ρ²) | ✅ | ❌ |
| `.dim` | Hilbert space dimension 2ᴺ | ✅ | ❌ |

---

### `magic`

| Function | Description | Python | Julia |
|---|---|:---:|:---:|
| `stabilizer_renyi_entropy(state, paulis, α)` | α-Stabilizer Rényi Entropy M_α | ✅ | ❌ |
| `_stabilizer_renyi_entropy_unchecked(...)` | Fast path for hot loops | ✅ | ❌ |
| `characteristic_function(ψ, paulis)` | Ξ(P) = \|⟨ψ\|P\|ψ⟩\|² / 2ⁿ for all P | ✅ | ❌ |
| `validate_alpha(α)` | Standalone α validator | ✅ | ❌ |
| `validate_compatible(state, paulis)` | Checks n_qubits agreement | ✅ | ❌ |

---

## Metrics

### Number of Tests

- Python tests: <!-- PYTHON_TESTS -->293 tests<!-- /PYTHON_TESTS -->
- Julia tests: <!-- JULIA_TESTS -->6 tests<!-- /JULIA_TESTS -->

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
