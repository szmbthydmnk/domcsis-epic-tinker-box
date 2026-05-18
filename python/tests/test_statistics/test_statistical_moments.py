from __future__ import annotations

import numpy as np
import pytest

from domcsis_epic_tinker_box.statistics.statistical_moments import (
    mean,
    mean_complex,
    skewness,
    variance,
)


# =============================================================================
# mean
# =============================================================================

def test_mean_list() -> None:
    data = [1.0, 2.0, 3.0, 4.0]
    assert mean(data) == pytest.approx(2.5)


def test_mean_numpy_array() -> None:
    data = np.array([1.0, 2.0, 3.0, 4.0])
    assert mean(data) == pytest.approx(2.5)


def test_mean_single_value() -> None:
    data = [7.5]
    assert mean(data) == pytest.approx(7.5)


def test_mean_negative_values() -> None:
    data = [-2.0, -1.0, 0.0, 1.0, 2.0]
    assert mean(data) == pytest.approx(0.0)


def test_mean_empty_list_raises() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        mean([])


def test_mean_empty_numpy_array_raises() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        mean(np.array([]))


def test_mean_complex_list_raises() -> None:
    data = [1.0, 2.0 + 1.0j, 3.0]
    with pytest.raises(ValueError, match="real numbers only"):
        mean(data)


def test_mean_complex_numpy_array_raises() -> None:
    data = np.array([1.0 + 0.0j, 2.0 + 0.0j])
    with pytest.raises(ValueError, match="real numbers only"):
        mean(data)


# =============================================================================
# mean_complex
# =============================================================================

def test_mean_complex_list() -> None:
    data = [1.0 + 2.0j, 3.0 + 4.0j]
    real_mean, complex_mean = mean_complex(data)

    assert real_mean == pytest.approx(2.0)
    assert complex_mean == pytest.approx(2.0 + 3.0j)


def test_mean_complex_numpy_array() -> None:
    data = np.array([1.0 + 2.0j, 3.0 + 4.0j])
    real_mean, complex_mean = mean_complex(data)

    assert real_mean == pytest.approx(2.0)
    assert complex_mean == pytest.approx(2.0 + 3.0j)


def test_mean_complex_single_value() -> None:
    data = [2.0 - 5.0j]
    real_mean, complex_mean = mean_complex(data)

    assert real_mean == pytest.approx(2.0)
    assert complex_mean == pytest.approx(2.0 - 5.0j)


def test_mean_complex_empty_list_raises() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        mean_complex([])


def test_mean_complex_empty_numpy_array_raises() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        mean_complex(np.array([]))


def test_mean_complex_real_list_raises() -> None:
    data = [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="complex numbers only"):
        mean_complex(data)


def test_mean_complex_real_numpy_array_raises() -> None:
    data = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="complex numbers only"):
        mean_complex(data)


def test_mean_complex_mixed_list_raises() -> None:
    data = [1.0 + 1.0j, 2.0]
    with pytest.raises(ValueError, match="complex numbers only"):
        mean_complex(data)


# =============================================================================
# variance
# =============================================================================

def test_variance_list() -> None:
    data = [1.0, 2.0, 3.0]
    expected = 2.0 / 3.0
    assert variance(data) == pytest.approx(expected)


def test_variance_numpy_array() -> None:
    data = np.array([1.0, 2.0, 3.0])
    expected = 2.0 / 3.0
    assert variance(data) == pytest.approx(expected)


def test_variance_single_value_is_zero() -> None:
    data = [5.0]
    assert variance(data) == pytest.approx(0.0)


def test_variance_constant_data_is_zero() -> None:
    data = [4.0, 4.0, 4.0, 4.0]
    assert variance(data) == pytest.approx(0.0)


def test_variance_known_values() -> None:
    data = [-2.0, -1.0, 0.0, 1.0, 2.0]
    expected = 2.0
    assert variance(data) == pytest.approx(expected)


def test_variance_empty_list_raises() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        variance([])


def test_variance_empty_numpy_array_raises() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        variance(np.array([]))


def test_variance_complex_list_raises() -> None:
    data = [1.0, 2.0 + 1.0j]
    with pytest.raises(ValueError, match="real numbers only"):
        variance(data)


def test_variance_complex_numpy_array_raises() -> None:
    data = np.array([1.0 + 0.0j, 2.0 + 0.0j])
    with pytest.raises(ValueError, match="real numbers only"):
        variance(data)


# =============================================================================
# skewness
# =============================================================================

def test_skewness_symmetric_data_is_zero() -> None:
    data = [-1.0, 0.0, 1.0]
    assert skewness(data) == pytest.approx(0.0)


def test_skewness_numpy_array_symmetric_data_is_zero() -> None:
    data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert skewness(data) == pytest.approx(0.0)


def test_skewness_positive_for_right_skewed_data() -> None:
    data = [1.0, 1.0, 2.0, 10.0]
    assert skewness(data) > 0.0


def test_skewness_negative_for_left_skewed_data() -> None:
    data = [-10.0, -2.0, -1.0, -1.0]
    assert skewness(data) < 0.0


def test_skewness_known_value() -> None:
    data = [1.0, 2.0, 3.0]
    expected = 0.0
    assert skewness(data) == pytest.approx(expected)


def test_skewness_empty_list_raises() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        skewness([])


def test_skewness_empty_numpy_array_raises() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        skewness(np.array([]))


def test_skewness_complex_list_raises() -> None:
    data = [1.0, 2.0 + 1.0j]
    with pytest.raises(ValueError, match="real numbers only"):
        skewness(data)


def test_skewness_complex_numpy_array_raises() -> None:
    data = np.array([1.0 + 0.0j, 2.0 + 0.0j])
    with pytest.raises(ValueError, match="real numbers only"):
        skewness(data)


def test_skewness_zero_variance_raises() -> None:
    data = [3.0, 3.0, 3.0]
    with pytest.raises(ValueError, match="Variance of the data must be non-zero"):
        skewness(data)
