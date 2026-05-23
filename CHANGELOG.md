## [Unreleased] - 2026-05-23

### Added (Python)
- `bell_state` — constructs 2-qubit Bell states by string identifier (`'Phi+'`, `'Phi-'`, `'Psi+'`, `'Psi-'`, case-insensitive) or N-qubit edge Bell state by integer. Validates input; supports up to 20 qubits.
- `w_state` — generates N-qubit W states as normalised `QuantumState` vectors. Validates input; supports up to 20 qubits.

### Added (Julia)
- Initial Julia package scaffold `DomcsisEpicTinkerBox.jl` with full CI: tests (Julia 1.9, 1.10, nightly), formatter (JuliaFormatter), Documenter.jl docs deployment to GitHub Pages, and LoC metrics.
- `statistics.jl` — `mean(xs)` computes the arithmetic mean of any `AbstractVector{<:Number}`; throws `ArgumentError` on empty input.
- TagBot workflow for automated GitHub Release creation on General registry merge.
- 5 Julia tests covering `mean` (standard cases, edge cases, empty-vector error) and `greet` smoke test.

### Fixed (CI)
- Python publish workflow: added `if: !startsWith(github.ref_name, 'julia-')` guard so Julia releases no longer trigger PyPI upload.
- Python publish workflow: added `skip-existing: true` so re-runs for an already-uploaded Python version pass silently instead of erroring.
- Julia docs workflow: added `pages: write` and `id-token: write` permissions required for Documenter.jl GitHub Pages deployment.
- Julia CI: removed committed `Manifest.toml` that broke resolution on Julia 1.9/1.10; added `.gitignore` to prevent future commits.
- Julia CI: fixed docs environment setup (`dev` local package into docs env + correct UUID in docs `Project.toml`).
- `loc.yml`: fixed Julia test counter `UndefVarError` by wrapping count logic in a named function to avoid top-level soft-scope ambiguity.

### Changed
- README: added Python/Julia ✅/❌ columns to all feature tables; added Julia test count metric (`<!-- JULIA_TESTS -->`) updated automatically by `loc.yml`; added Julia registry version badge.
- `loc.yml`: extended to count Julia tests via macro-pattern scan of `julia/test/**/*.jl` and inject the count into README.

---

## [0.4.1] - 2026-05-22

### Added
- Reference stabilizer-state enumerations as text files: `1q_stabilizer_states.txt`, `2q_stabilizer_states.txt`, `3q_stabilizer_states.txt` — complete lists of single-, two-, and three-qubit stabilizer states for use in tests and benchmarks.

---

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
- Module stubs for `mana.py` and `robustness_of_magic.py` (placeholders for future implementations).
- Populated all previously empty `__init__.py` files across Python submodules.
- Full pytest coverage for all new modules (243 tests total).

### Changed
- `_check_normalised` in `quantum_state` now accepts a `float` norm rather
  than a complex scalar, eliminating spurious imaginary-part warnings.
- `_compute_sre` now handles `alpha=1` via the Shannon-entropy limit
  (`0 log 0 := 0`) instead of triggering a divide-by-zero `RuntimeWarning`.
- `PauliOperators`: updated `pauli_operators.py` to fix iteration and
  `as_dict` behaviour (multiple refinement commits).

---

## [0.2.0] - 2026-05-17

### Added
- Random number generation utilities for Gaussian, uniform, exponential, and Wigner surmise distributions.
- Support for Wigner surmise sampling through the generic distribution wrapper.
- Optional local seed and RNG injection for reproducible sampling.

### Changed
- Improved validation and NumPy-style docstrings for random-number-generation utilities.
- Expanded pytest coverage for the random number generation module.
