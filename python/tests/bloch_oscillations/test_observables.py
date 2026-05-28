"""
Tests for bloch_oscillations.observables.

Verifies that the Pauli-string lists have the correct length, that the
Z operators are placed on the correct qubit positions, and that the
centre-site correlator observable correctly reduces to the identity.
"""

from __future__ import annotations

import pytest

from domcsis_epic_tinker_box.bloch_oscillations.model import ModelParams, center_index
from domcsis_epic_tinker_box.bloch_oscillations.observables import (
    magnetisation_observables,
    correlator_observables,
)


# ============================================================================
# Helpers
# ============================================================================


def _pauli_string(obs: object) -> str:
    """Extract the single Pauli label string from a SparsePauliOp."""
    # SparsePauliOp.to_list() returns [(label, coeff), …].
    items = obs.to_list()  # type: ignore[attr-defined]
    assert len(items) == 1
    return str(items[0][0])


# ============================================================================
# magnetisation_observables
# ============================================================================


@pytest.mark.parametrize("L", [3, 5, 7])
def test_magnetisation_length(L: int) -> None:
    """Returns exactly L observables."""
    p = ModelParams(L=L)
    obs = magnetisation_observables(p)
    assert len(obs) == L


@pytest.mark.parametrize("L", [3, 5, 7])
def test_magnetisation_pauli_strings(L: int) -> None:
    """Each observable has a single Z at the correct position, rest I."""
    p = ModelParams(L=L)
    for j, obs in enumerate(magnetisation_observables(p)):
        label = _pauli_string(obs)
        assert len(label) == L
        assert label[j] == "Z", f"Expected Z at index {j}, got '{label}'"
        for k, ch in enumerate(label):
            if k != j:
                assert ch == "I", f"Expected I at index {k}, got '{ch}'"


# ============================================================================
# correlator_observables
# ============================================================================


@pytest.mark.parametrize("L", [3, 5, 7])
def test_correlator_length(L: int) -> None:
    """Returns exactly L observables."""
    p = ModelParams(L=L)
    obs = correlator_observables(p)
    assert len(obs) == L


@pytest.mark.parametrize("L", [3, 5, 7])
def test_correlator_centre_site_is_identity(L: int) -> None:
    """The correlator at the centre site is the all-identity operator."""
    p = ModelParams(L=L)
    c = center_index(p)
    obs = correlator_observables(p)
    label = _pauli_string(obs[c])
    assert label == "I" * L


@pytest.mark.parametrize("L", [5, 7])
def test_correlator_off_centre_has_two_z(L: int) -> None:
    """Off-centre correlator observables carry exactly two Z characters."""
    p = ModelParams(L=L)
    c = center_index(p)
    for j, obs in enumerate(correlator_observables(p)):
        if j == c:
            continue
        label = _pauli_string(obs)
        z_count = label.count("Z")
        assert z_count == 2, f"Expected 2 Z's for j={j}, got {z_count} in '{label}'"
