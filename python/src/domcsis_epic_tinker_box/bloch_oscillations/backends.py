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

from typing import Any

from qiskit_aer import AerSimulator  
from qiskit_aer.noise import NoiseModel  

from qiskit_ibm_runtime import fake_provider  

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


def get_fake_backend(name: str) -> Any:
    """Instantiate a fake IBM backend by human-readable name.

    Tries the ``V2`` class variant first, then falls back to the legacy
    class name.  Raises ``ValueError`` if the name is not in the supported
    set, and ``RuntimeError`` if none of the candidate class names can be
    found in the installed ``qiskit_ibm_runtime.fake_provider`` module.

    The return type is ``Any`` because the concrete class varies with the
    installed version of ``qiskit_ibm_runtime`` and carries no public type
    stubs.  Callers that need a specific interface should cast the result.

    Args:
        name: Human-readable backend name (case-insensitive).  Recognised
              values: ``"brisbane"``, ``"sherbrooke"``, ``"almaden"``.

    Returns:
        An instantiated fake backend object.

    Raises:
        ValueError:    If ``name`` is not a recognised fake backend.
        RuntimeError:  If no matching class is found in
                       ``qiskit_ibm_runtime.fake_provider``.
    """

    key = name.lower()
    if key not in _FAKE_BACKEND_CANDIDATES:
        raise ValueError(
            f"Unknown fake backend name: '{name}'. "
            f"Supported values: {sorted(_FAKE_BACKEND_CANDIDATES)}"
        )

    for cls_name in _FAKE_BACKEND_CANDIDATES[key]:
        cls = getattr(fake_provider, cls_name, None)
        if cls is not None:
            return cls()

    raise RuntimeError(
        f"Could not find a fake-backend class for '{name}' in the installed "
        "qiskit_ibm_runtime.  Try upgrading: pip install -U qiskit-ibm-runtime"
    )


# ============================================================================
# Noisy density-matrix simulator factory
# ============================================================================


def make_noisy_density_simulator(fake_backend: Any) -> AerSimulator:
    """Build an ``AerSimulator`` in density-matrix mode from a fake backend.

    The noise model is extracted from the fake backend using
    ``NoiseModel.from_backend``.  The ``basis_gates`` list is intentionally
    left to Aer's own defaults so that scheduling instructions (``delay``,
    ``for_loop``, ``if_else``) that appear in some fake-backend gate sets
    do not cause transpilation failures.

    Args:
        fake_backend: An instantiated fake backend object as returned by
                      :func:`get_fake_backend`.  Typed as ``Any`` because
                      the concrete class carries no public stubs.

    Returns:
        An ``AerSimulator`` configured for density-matrix simulation with
        the noise model of the given fake backend.
    """
    noise_model = NoiseModel.from_backend(fake_backend)
    return AerSimulator(
        method="density_matrix",
        noise_model=noise_model,
    )
