## [0.4.0] - 2026-05-19

### Added
- `QuantumState` — immutable validated quantum state container with three
  canonical representations: state vector, density matrix, and vectorised
  density matrix. Supports pure and mixed states. Derived quantities
  (`vector`, `density_matrix`, `purity`) computed on demand from whichever
  form was provided at construction.
- `PauliOperators` — immutable validated collection of N-qubit Pauli operators
  built from sparse CSR matrices. Factory constructors `all(n)` and
  `from_strings(set)`. Iterable as `(label, matrix)` pairs.
- `stabilizer_renyi_entropy` — validated public entry point for the
  α-Stabilizer Rényi Entropy (SRE) of a pure quantum state. Returns the SRE
  scalar or `(SRE, spectrum)` tuple when `pauli_spectrum=True`.
- `_stabilizer_renyi_entropy_unchecked` — fast unchecked path for hot loops
  over many states with fixed Paulis and α.
- `characteristic_function` — computes Ξ(P) = |⟨ψ|P|ψ⟩|² / 2ⁿ for all
  Paulis in a `PauliOperators` instance.
- `validate_alpha` / `validate_compatible` — standalone validation helpers
  exposed for use in performance-critical call sites.
- `statistical_moments` — population mean (real and complex), population
  variance, and skewness for finite-sample ensembles.
- Full pytest coverage for all new modules (243 tests total).

### Changed
- `_check_normalised` in `quantum_state` now accepts a `float` norm rather
  than a complex scalar, eliminating spurious imaginary-part warnings.
- `_compute_sre` now handles `alpha=1` via the Shannon-entropy limit
  (`0 log 0 := 0`) instead of triggering a divide-by-zero `RuntimeWarning`.

---

## [0.2.0] - 2026-05-17

### Added
- Random number generation utilities for Gaussian, uniform, exponential, and Wigner surmise distributions.
- Support for Wigner surmise sampling through the generic distribution wrapper.
- Optional local seed and RNG injection for reproducible sampling.

### Changed
- Improved validation and NumPy-style docstrings for random-number-generation utilities.
- Expanded pytest coverage for the random number generation module.
