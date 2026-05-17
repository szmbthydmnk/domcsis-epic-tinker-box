from __future__ import annotations

import math
import random

import pytest

from domcsis_epic_tinker_box.statistics.random_number_generation import (
    generate_exponential_rnd_numbers,
    generate_gaussian_rnd_numbers,
    generate_random_numbers,
    generate_uniform_rnd_numbers,
    generate_wigner_surmise_rnd_numbers,
)


def _sample_mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _sample_std(values: list[float]) -> float:
    mean = _sample_mean(values)
    return math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))


# =============================================================================
# generate_gaussian_rnd_numbers

def test_gaussian_rnd_numbers_length() -> None:
    n = 5
    numbers = generate_gaussian_rnd_numbers(n=n)
    assert len(numbers) == n


def test_gaussian_rnd_numbers_invalid_sigma_negative() -> None:
    with pytest.raises(ValueError):
        generate_gaussian_rnd_numbers(sigma=-1, n=10)


def test_gaussian_rnd_numbers_invalid_sigma_zero() -> None:
    with pytest.raises(ValueError):
        generate_gaussian_rnd_numbers(sigma=0, n=10)


def test_gaussian_rnd_numbers_zero_samples() -> None:
    with pytest.raises(ValueError):
        generate_gaussian_rnd_numbers(n=0)


def test_gaussian_rnd_numbers_mean_and_std() -> None:
    mu = 0.0
    sigma = 1.0
    n = 10_000
    numbers = generate_gaussian_rnd_numbers(mu=mu, sigma=sigma, n=n, seed=123)

    assert abs(_sample_mean(numbers) - mu) < 0.05
    assert abs(_sample_std(numbers) - sigma) < 0.05


def test_gaussian_rnd_numbers_seed_reproducible() -> None:
    numbers_1 = generate_gaussian_rnd_numbers(mu=1.2, sigma=0.7, n=20, seed=42)
    numbers_2 = generate_gaussian_rnd_numbers(mu=1.2, sigma=0.7, n=20, seed=42)
    assert numbers_1 == numbers_2


def test_gaussian_rnd_numbers_rng_reproducible() -> None:
    rng_1 = random.Random(42)
    rng_2 = random.Random(42)

    numbers_1 = generate_gaussian_rnd_numbers(mu=0.5, sigma=2.0, n=20, rng=rng_1)
    numbers_2 = generate_gaussian_rnd_numbers(mu=0.5, sigma=2.0, n=20, rng=rng_2)

    assert numbers_1 == numbers_2


def test_gaussian_rnd_numbers_rng_and_seed_conflict() -> None:
    with pytest.raises(ValueError):
        generate_gaussian_rnd_numbers(n=10, rng=random.Random(1), seed=1)


# =============================================================================
# generate_uniform_rnd_numbers

def test_uniform_rnd_numbers_length() -> None:
    n = 5
    numbers = generate_uniform_rnd_numbers(n=n)
    assert len(numbers) == n


def test_uniform_rnd_numbers_invalid_range() -> None:
    with pytest.raises(ValueError):
        generate_uniform_rnd_numbers(a=1, b=0, n=10)


def test_uniform_rnd_numbers_zero_samples() -> None:
    with pytest.raises(ValueError):
        generate_uniform_rnd_numbers(n=0)


def test_uniform_rnd_numbers_range() -> None:
    a = -2.0
    b = 3.0
    n = 10_000
    numbers = generate_uniform_rnd_numbers(a=a, b=b, n=n, seed=123)

    assert all(a <= x <= b for x in numbers)


def test_uniform_rnd_numbers_mean() -> None:
    a = 2.0
    b = 6.0
    n = 10_000
    numbers = generate_uniform_rnd_numbers(a=a, b=b, n=n, seed=123)

    expected_mean = 0.5 * (a + b)
    assert abs(_sample_mean(numbers) - expected_mean) < 0.05


def test_uniform_rnd_numbers_seed_reproducible() -> None:
    numbers_1 = generate_uniform_rnd_numbers(a=-1, b=1, n=20, seed=42)
    numbers_2 = generate_uniform_rnd_numbers(a=-1, b=1, n=20, seed=42)
    assert numbers_1 == numbers_2


def test_uniform_rnd_numbers_rng_and_seed_conflict() -> None:
    with pytest.raises(ValueError):
        generate_uniform_rnd_numbers(n=10, rng=random.Random(1), seed=1)


# =============================================================================
# generate_exponential_rnd_numbers

def test_exponential_rnd_numbers_length() -> None:
    n = 5
    numbers = generate_exponential_rnd_numbers(n=n)
    assert len(numbers) == n


def test_exponential_rnd_numbers_invalid_lambda() -> None:
    with pytest.raises(ValueError):
        generate_exponential_rnd_numbers(lambd=0, n=10)


def test_exponential_rand_numbers_zero_samples() -> None:
    with pytest.raises(ValueError):
        generate_exponential_rnd_numbers(n=0)


def test_exponential_rnd_numbers_non_negative() -> None:
    numbers = generate_exponential_rnd_numbers(lambd=2.0, n=1000, seed=123)
    assert all(x >= 0 for x in numbers)


def test_exponential_rnd_numbers_mean() -> None:
    lambd = 2.0
    n = 10_000
    numbers = generate_exponential_rnd_numbers(lambd=lambd, n=n, seed=123)

    assert abs(_sample_mean(numbers) - 1 / lambd) < 0.03


def test_exponential_rnd_numbers_seed_reproducible() -> None:
    numbers_1 = generate_exponential_rnd_numbers(lambd=1.5, n=20, seed=42)
    numbers_2 = generate_exponential_rnd_numbers(lambd=1.5, n=20, seed=42)
    assert numbers_1 == numbers_2


def test_exponential_rnd_numbers_rng_and_seed_conflict() -> None:
    with pytest.raises(ValueError):
        generate_exponential_rnd_numbers(n=10, rng=random.Random(1), seed=1)


# =============================================================================
# generate_wigner_surmise_rnd_numbers

def test_wigner_surmise_rnd_numbers_length() -> None:
    n = 5
    numbers = generate_wigner_surmise_rnd_numbers(beta=2, n=n)
    assert len(numbers) == n


def test_wigner_surmise_rnd_numbers_invalid_beta() -> None:
    with pytest.raises(ValueError):
        generate_wigner_surmise_rnd_numbers(beta=0, n=10)


def test_wigner_surmise_rnd_numbers_zero_samples() -> None:
    with pytest.raises(ValueError):
        generate_wigner_surmise_rnd_numbers(beta=2, n=0)


def test_wigner_surmise_rnd_numbers_non_negative() -> None:
    numbers = generate_wigner_surmise_rnd_numbers(beta=2, n=1000, seed=123)
    assert all(x >= 0 for x in numbers)


def test_wigner_surmise_rnd_numbers_unit_mean_approximately() -> None:
    n = 20_000
    numbers = generate_wigner_surmise_rnd_numbers(beta=2, n=n, seed=123)

    assert abs(_sample_mean(numbers) - 1.0) < 0.03


@pytest.mark.parametrize("beta", [1, 2, 4])
def test_wigner_surmise_standard_betas_unit_mean(beta: int) -> None:
    numbers = generate_wigner_surmise_rnd_numbers(beta=beta, n=20_000, seed=123)
    assert abs(_sample_mean(numbers) - 1.0) < 0.03


def test_wigner_surmise_rnd_numbers_seed_reproducible() -> None:
    numbers_1 = generate_wigner_surmise_rnd_numbers(beta=4, n=20, seed=42)
    numbers_2 = generate_wigner_surmise_rnd_numbers(beta=4, n=20, seed=42)
    assert numbers_1 == numbers_2


def test_wigner_surmise_rnd_numbers_rng_and_seed_conflict() -> None:
    with pytest.raises(ValueError):
        generate_wigner_surmise_rnd_numbers(beta=2, n=10, rng=random.Random(1), seed=1)


# =============================================================================
# generate_random_numbers

def test_generate_random_numbers_gaussian() -> None:
    distribution = "gaussian"
    params = {"mu": 0.0, "sigma": 1.0}
    n = 10_000
    numbers = generate_random_numbers(distribution, params, n, seed=123)

    assert len(numbers) == n
    assert abs(_sample_mean(numbers) - params["mu"]) < 0.05
    assert abs(_sample_std(numbers) - params["sigma"]) < 0.05


def test_generate_random_numbers_uniform() -> None:
    distribution = "uniform"
    params = {"a": 0.0, "b": 1.0}
    n = 10_000
    numbers = generate_random_numbers(distribution, params, n, seed=123)

    assert len(numbers) == n
    assert all(params["a"] <= x <= params["b"] for x in numbers)


def test_generate_random_numbers_exponential() -> None:
    distribution = "exponential"
    params = {"lambda": 1.0}
    n = 10_000
    numbers = generate_random_numbers(distribution, params, n, seed=123)

    assert len(numbers) == n
    assert abs(_sample_mean(numbers) - 1 / params["lambda"]) < 0.05


def test_generate_random_numbers_wigner_surmise() -> None:
    distribution = "wigner_surmise"
    params = {"beta": 2}
    n = 20_000
    numbers = generate_random_numbers(distribution, params, n, seed=123)

    assert len(numbers) == n
    assert all(x >= 0 for x in numbers)
    assert abs(_sample_mean(numbers) - 1.0) < 0.03


def test_generate_random_numbers_wigner_alias() -> None:
    numbers_1 = generate_random_numbers("wigner", {"beta": 2}, 20, seed=42)
    numbers_2 = generate_random_numbers("wigner", {"beta": 2}, 20, seed=42)
    assert numbers_1 == numbers_2


def test_generate_random_numbers_invalid_distribution() -> None:
    distribution = "invalid_dist"
    params: dict[str, float] = {}
    n = 10

    with pytest.raises(ValueError):
        generate_random_numbers(distribution, params, n)


def test_generate_random_numbers_zero_samples() -> None:
    distribution = "gaussian"
    params = {"mu": 0.0, "sigma": 1.0}
    n = 0

    with pytest.raises(ValueError):
        generate_random_numbers(distribution, params, n)


def test_generate_random_numbers_seed_reproducible() -> None:
    params = {"mu": 1.0, "sigma": 2.0}
    numbers_1 = generate_random_numbers("gaussian", params, 20, seed=42)
    numbers_2 = generate_random_numbers("gaussian", params, 20, seed=42)
    assert numbers_1 == numbers_2


def test_generate_random_numbers_rng_and_seed_conflict() -> None:
    with pytest.raises(ValueError):
        generate_random_numbers(
            "gaussian",
            {"mu": 0.0, "sigma": 1.0},
            10,
            rng=random.Random(1),
            seed=1,
        )


def test_generate_random_numbers_uses_defaults_when_params_missing() -> None:
    numbers = generate_random_numbers("gaussian", {}, 10, seed=42)
    assert len(numbers) == 10


@pytest.mark.parametrize(
    ("distribution", "params"),
    [
        ("gaussian", {"mu": 0.0, "sigma": -1.0}),
        ("uniform", {"a": 2.0, "b": 1.0}),
        ("exponential", {"lambda": 0.0}),
        ("wigner_surmise", {"beta": 0}),
    ],
)
def test_generate_random_numbers_propagates_parameter_validation(
    distribution: str,
    params: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        generate_random_numbers(distribution, params, 10)
