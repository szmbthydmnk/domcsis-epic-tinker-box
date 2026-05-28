"""
Tests for bloch_oscillations.model.

Verifies dataclass default values, field mutability via ``replace``,
and the ``center_index`` geometry helper for odd and even chain lengths.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from domcsis_epic_tinker_box.bloch_oscillations.model import (
    ModelParams,
    RunConfig,
    center_index,
)


# ============================================================================
# ModelParams
# ============================================================================


class TestModelParams:
    """Tests for the ModelParams dataclass."""

    def test_defaults(self) -> None:
        """Default field values match the documented physical defaults."""
        p = ModelParams()
        assert p.J == 1.0
        assert p.ht == -0.15
        assert p.hl == 0.15
        assert p.dt == 0.25
        assert p.L == 7
        assert p.layers_max == 40

    def test_custom_values(self) -> None:
        """Custom field values are stored correctly."""
        p = ModelParams(L=3, layers_max=5, J=0.5, ht=-0.25, hl=0.25, dt=0.1)
        assert p.L == 3
        assert p.layers_max == 5

    def test_replace_leaves_original_unchanged(self) -> None:
        """``dataclasses.replace`` creates a new instance without mutation."""
        original = ModelParams()
        modified = replace(original, L=9)
        assert original.L == 7
        assert modified.L == 9


# ============================================================================
# RunConfig
# ============================================================================


class TestRunConfig:
    """Tests for the RunConfig dataclass."""

    def test_defaults(self) -> None:
        """Default field values match the documented defaults."""
        c = RunConfig()
        assert c.backend_mode == "ideal"
        assert c.fake_backend_name is None
        assert c.initial_state == "all_up"
        assert c.use_cnot_zz is False
        assert c.shots == 8192
        assert c.optimization_level == 1
        assert c.use_parallel_u1 is False
        assert c.trotter_method == "even_odd"

    def test_custom_values(self) -> None:
        """Custom field values are stored and readable."""
        c = RunConfig(
            backend_mode="fake",
            fake_backend_name="brisbane",
            initial_state="all_down",
            trotter_method="zig_zag",
        )
        assert c.backend_mode == "fake"
        assert c.fake_backend_name == "brisbane"
        assert c.trotter_method == "zig_zag"


# ============================================================================
# center_index
# ============================================================================


@pytest.mark.parametrize(
    ("L", "expected"),
    [
        (7, 3),  # ceil(7/2 - 1) = ceil(3.5 - 1) = ceil(2.5) = 3
        (6, 2),  # ceil(6/2 - 1) = ceil(3.0 - 1) = ceil(2.0) = 2
        (3, 1),  # ceil(3/2 - 1) = ceil(1.5 - 1) = ceil(0.5) = 1
        (1, 0),  # ceil(1/2 - 1) = ceil(0.5 - 1) = ceil(-0.5) = 0
    ],
)
def test_center_index(L: int, expected: int) -> None:
    """center_index reproduces the notebook's ceil(L/2 - 1) formula."""
    p = ModelParams(L=L)
    result = center_index(p)
    assert result == math.ceil(L / 2 - 1)
    assert result == expected
