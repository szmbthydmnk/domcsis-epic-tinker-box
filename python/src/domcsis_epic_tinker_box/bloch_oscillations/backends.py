"""
Backend factory helpers for the Bloch-oscillations simulation.

This module isolates all backend-construction logic so that the simulation
drivers in ``simulation.py`` remain easy to unit-test: replace these
factories with fakes or mocks and the simulation layer requires no changes.

Supported fake backends
-----------------------
Backend resolution is intentionally lenient: it tries the ``V2`` variant
first and falls back to the legacy class name.  This makes the code forward-
compatible with future ``qiskit_ibm_runtime`` releases that may drop ``V2``
suffixes.

Recognised ``fake_backend_name`` values (case-insensitive):
    - ``"brisbane"``
    - ``"sherbrooke"``
    - ``"almaden"``
"""

from __future__ import annotations

from qiskit_aer import AerSimulator  # type: ignore[import-untyped]
from qiskit_aer.noise import NoiseModel  # type: ignore[import-untyped]


# ============================================================================
# Fake-backend resolution
# ============================================================================

# Mapping from normalised name to the ordered list of class names to try.
# The V2 variant is tried first; the legacy name is the fallback.
_FAKE_BACKEND_CANDIDATES: dict[str, list[str]] = {
    "brisbane": ["FakeBrisbaneV2", "FakeBrisbane"],
    "sherbrooke": ["FakeSherbrookeV2", "FakeSherbrooke"],
    "almaden": ["FakeAlmadenV2", "FakeAlmaden"],
}


def get_fake_backend(name: str) -> object:
    """Instantiate a fake IBM backend by human-readable name.

    Tries the ``V2`` class variant first, then falls back to the legacy
    class name.  Raises ``ValueError`` if the name is not in the supported
    set, and ``RuntimeError`` if none of the candidate class names can be
    found in the installed ``qiskit_ibm_runtime.fake_provider`` module.

    Args:
        name: Human-readable backend name (case-insensitive).  Recognised
              values: ``"brisbane"``, ``"sherbrooke"``, ``"almaden"``.

    Returns:
        An instantiated fake backend object (concrete type varies by
        installed ``qiskit_ibm_runtime`` version).

    Raises:
        ValueError:    If ``name`` is not a recognised fake backend.
        RuntimeError:  If no matching class is found in
                       ``qiskit_ibm_runtime.fake_provider``.
    """
    from qiskit_ibm_runtime import fake_provider  # type: ignore[import-untyped]

    # Normalise the name so callers are not case-sensitive.
    key = name.lower()
    if key not in _FAKE_BACKEND_CANDIDATES:
        raise ValueError(
            f"Unknown fake backend name: '{name}'. "
            f"Supported values: {sorted(_FAKE_BACKEND_CANDIDATES)}"
        )

    # Walk the candidate list; return the first class that exists in the
    # installed version of qiskit_ibm_runtime.
    for cls_name in _FAKE_BACKEND_CANDIDATES[key]:
        cls = getattr(fake_provider, cls_name, None)
        if cls is not None:
            return cls()  # type: ignore[no-any-return]

    raise RuntimeError(
        f"Could not find a fake-backend class for '{name}' in the installed "
        "qiskit_ibm_runtime.  Try upgrading: pip install -U qiskit-ibm-runtime"
    )


# ============================================================================
# Noisy density-matrix simulator factory
# ============================================================================


def make_noisy_density_simulator(fake_backend: object) -> AerSimulator:
    """Build an ``AerSimulator`` in density-matrix mode from a fake backend.

    The noise model is extracted from the fake backend using
    ``NoiseModel.from_backend``.  The ``basis_gates`` list is intentionally
    left to Aer's own defaults so that scheduling instructions (``delay``,
    ``for_loop``, ``if_else``) that appear in some fake-backend gate sets
    do not cause transpilation failures.

    Args:
        fake_backend: An instantiated fake backend object as returned by
                      :func:`get_fake_backend`.

    Returns:
        An ``AerSimulator`` configured for density-matrix simulation with
        the noise model of the given fake backend.
    """
    # Extract the noise model; AerSimulator accepts it as a constructor kwarg.
    noise_model = NoiseModel.from_backend(fake_backend)  # type: ignore[arg-type]

    return AerSimulator(
        method="density_matrix",
        noise_model=noise_model,
    )
