"""
This module provides functions for generating random numbers from various
probability distributions, including Gaussian, uniform, exponential, and
Wigner surmise distributions.

Each function supports generation of multiple samples, validates its input
parameters, and optionally accepts either a dedicated random number generator
or a local seed for reproducible sampling without mutating global random state.
"""

from __future__ import annotations

from typing import Optional

import math
import random


def _resolve_rng(
    rng: Optional[random.Random] = None,
    seed: Optional[int] = None,
) -> random.Random:
    """
    Return a random number generator instance.

    Parameters
    ----------
    rng : random.Random | None, default=None
        Existing random number generator instance.
    seed : int | None, default=None
        Seed used to create a local ``random.Random`` instance.

    Returns
    -------
    random.Random
        Random number generator used by the sampling functions.

    Raises
    ------
    ValueError
        If both ``rng`` and ``seed`` are provided.

    Notes
    -----
    This helper avoids reseeding Python's global random state inside library
    functions. If ``seed`` is provided, a local generator is created via
    ``random.Random(seed)``.
    """
    if rng is not None and seed is not None:
        raise ValueError("Pass either 'rng' or 'seed', not both.")
    if rng is not None:
        return rng
    if seed is not None:
        return random.Random(seed)
    return random


def generate_gaussian_rnd_numbers(
    mu: float = 0,
    sigma: float = 1,
    n: int = 1,
    rng: Optional[random.Random] = None,
    seed: Optional[int] = None,
) -> list[float]:
    """
    Generate ``n`` random variables from a Gaussian distribution.

    Parameters
    ----------
    mu : float, default=0
        Mean of the Gaussian distribution.
    sigma : float, default=1
        Standard deviation of the Gaussian distribution. Must be positive.
    n : int, default=1
        Number of random variables to generate.
    rng : random.Random | None, default=None
        Optional random number generator instance.
    seed : int | None, default=None
        Optional seed for constructing a local random number generator.

    Returns
    -------
    list[float]
        A list of ``n`` samples drawn from the specified Gaussian distribution.

    Raises
    ------
    ValueError
        If ``sigma`` is not positive, if ``n`` is not a positive integer,
        or if both ``rng`` and ``seed`` are provided.

    Notes
    -----
    The Gaussian probability density function is

    ``f(x) = (1 / (sigma * sqrt(2*pi))) * exp(-(x - mu)^2 / (2*sigma^2))``

    for ``x in R``, with mean ``mu`` and standard deviation ``sigma``.
    """
    if sigma <= 0:
        raise ValueError(f"Expected positive standard deviation, got {sigma}")
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"Expected positive integer number of samples, got {n!r}")

    local_rng = _resolve_rng(rng=rng, seed=seed)
    return [local_rng.gauss(mu, sigma) for _ in range(n)]


def generate_uniform_rnd_numbers(
    a: float = 0,
    b: float = 1,
    n: int = 1,
    rng: Optional[random.Random] = None,
    seed: Optional[int] = None,
) -> list[float]:
    """
    Generate ``n`` random variables from a uniform distribution over ``[a, b]``.

    Parameters
    ----------
    a : float, default=0
        Lower bound of the uniform distribution.
    b : float, default=1
        Upper bound of the uniform distribution.
    n : int, default=1
        Number of random variables to generate.
    rng : random.Random | None, default=None
        Optional random number generator instance.
    seed : int | None, default=None
        Optional seed for constructing a local random number generator.

    Returns
    -------
    list[float]
        A list of ``n`` samples drawn from the specified uniform distribution.

    Raises
    ------
    ValueError
        If ``a > b``, if ``n`` is not a positive integer, or if both ``rng``
        and ``seed`` are provided.

    Notes
    -----
    The uniform probability density function on ``[a, b]`` is

    ``f(x) = 1 / (b - a)``

    for ``a <= x <= b``, and ``f(x) = 0`` otherwise.
    """
    if a > b:
        raise ValueError(
            f"Expected lower bound a to be less than or equal to upper bound b, got a={a}, b={b}"
        )
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"Expected positive integer number of samples, got {n!r}")

    local_rng = _resolve_rng(rng=rng, seed=seed)
    return [local_rng.uniform(a, b) for _ in range(n)]


def generate_exponential_rnd_numbers(
    lambd: float = 1,
    n: int = 1,
    rng: Optional[random.Random] = None,
    seed: Optional[int] = None,
) -> list[float]:
    """
    Generate ``n`` random variables from an exponential distribution.

    Parameters
    ----------
    lambd : float, default=1
        Rate parameter of the exponential distribution. Must be positive.
    n : int, default=1
        Number of random variables to generate.
    rng : random.Random | None, default=None
        Optional random number generator instance.
    seed : int | None, default=None
        Optional seed for constructing a local random number generator.

    Returns
    -------
    list[float]
        A list of ``n`` samples drawn from the specified exponential distribution.

    Raises
    ------
    ValueError
        If ``lambd`` is not positive, if ``n`` is not a positive integer,
        or if both ``rng`` and ``seed`` are provided.

    Notes
    -----
    The exponential probability density function is

    ``f(x) = lambd * exp(-lambd * x)``

    for ``x >= 0``, and ``f(x) = 0`` otherwise.
    Its mean is ``1 / lambd``.
    """
    if lambd <= 0:
        raise ValueError(f"Expected positive rate parameter, got {lambd}")
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"Expected positive integer number of samples, got {n!r}")

    local_rng = _resolve_rng(rng=rng, seed=seed)
    return [local_rng.expovariate(lambd) for _ in range(n)]


def generate_wigner_surmise_rnd_numbers(
    beta: int = 2,
    n: int = 1,
    rng: Optional[random.Random] = None,
    seed: Optional[int] = None,
) -> list[float]:
    """
    Generate ``n`` samples from the generalized Wigner surmise distribution.

    Parameters
    ----------
    beta : int, default=2
        Dyson index / surmise order. Standard choices are:
        - 1: GOE
        - 2: GUE
        - 4: GSE
        
        Any positive integer is accepted for the generalized surmise.
    n : int, default=1
        Number of random variables to generate.
    rng : random.Random | None, default=None
        Optional random number generator instance.
    seed : int | None, default=None
        Optional seed for constructing a local random number generator.

    Returns
    -------
    list[float]
        A list of ``n`` samples drawn from the specified Wigner surmise distribution.

    Raises
    ------
    ValueError
        If ``beta`` is not a positive integer, if ``n`` is not a positive
        integer, or if both ``rng`` and ``seed`` are provided.

    Notes
    -----
    The generalized unit-mean Wigner surmise density is

    ``p(s) = a_beta * s**beta * exp(-b_beta * s**2)``,  for ``s >= 0``

    where

    ``b_beta = [Gamma((beta + 2)/2) / Gamma((beta + 1)/2)]**2``

    and

    ``a_beta = 2 * b_beta**((beta + 1)/2) / Gamma((beta + 1)/2)``.

    Sampling is performed by drawing

    ``Y ~ Gamma(shape=(beta + 1)/2, scale=1 / b_beta)``

    and returning ``S = sqrt(Y)``.
    """
    if not isinstance(beta, int) or beta <= 0:
        raise ValueError(f"Expected a positive integer for beta, got {beta!r}")
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"Expected a positive integer for n, got {n!r}")

    local_rng = _resolve_rng(rng=rng, seed=seed)

    shape = 0.5 * (beta + 1)
    b_beta = (
        math.gamma(0.5 * (beta + 2)) / math.gamma(0.5 * (beta + 1))
    ) ** 2
    scale = 1.0 / b_beta

    return [math.sqrt(local_rng.gammavariate(shape, scale)) for _ in range(n)]


def generate_random_numbers(
    distribution: str,
    params: Optional[dict[str, float]] = None,
    n: int = 1,
    rng: Optional[random.Random] = None,
    seed: Optional[int] = None,
) -> list[float]:
    """
    Generate ``n`` random variables from a specified distribution.

    Parameters
    ----------
    distribution : str
        Name of the distribution to sample from. Supported values are
        ``'gaussian'``, ``'uniform'``, ``'exponential'``, and
        ``'wigner_surmise'``.
    params : dict[str, float] | None, default=None
        Dictionary of distribution parameters.

        Expected keys:
        - Gaussian: ``'mu'``, ``'sigma'``
        - Uniform: ``'a'``, ``'b'``
        - Exponential: ``'lambda'``
        - Wigner surmise: ``'beta'``

        Missing parameters fall back to the defaults of the corresponding
        generator function.
    n : int, default=1
        Number of random variables to generate.
    rng : random.Random | None, default=None
        Optional random number generator instance.
    seed : int | None, default=None
        Optional seed for constructing a local random number generator.

    Returns
    -------
    list[float]
        A list of ``n`` samples drawn from the specified distribution.

    Raises
    ------
    ValueError
        If the distribution is unsupported, if ``n`` is not a positive integer,
        or if both ``rng`` and ``seed`` are provided.

    Notes
    -----
    This is a convenience wrapper around the individual generator functions.
    It forwards all validated arguments to the appropriate distribution-specific
    implementation and uses a shared local random number generator for the
    sampling call.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"Expected positive integer number of samples, got {n!r}")

    params = {} if params is None else params
    local_rng = _resolve_rng(rng=rng, seed=seed)
    distribution = distribution.lower()

    if distribution == "gaussian":
        return generate_gaussian_rnd_numbers(
            mu=params.get("mu", 0.0),
            sigma=params.get("sigma", 1.0),
            n=n,
            rng=local_rng,
        )

    if distribution == "uniform":
        return generate_uniform_rnd_numbers(
            a=params.get("a", 0.0),
            b=params.get("b", 1.0),
            n=n,
            rng=local_rng,
        )

    if distribution == "exponential":
        return generate_exponential_rnd_numbers(
            lambd=params.get("lambda", 1.0),
            n=n,
            rng=local_rng,
        )

    if distribution in {"wigner_surmise", "wigner"}:
        return generate_wigner_surmise_rnd_numbers(
            beta=int(params.get("beta", 2)),
            n=n,
            rng=local_rng,
        )

    raise ValueError(
        "Unsupported/not implemented distribution: "
        f"{distribution!r}. Supported values are 'gaussian', 'uniform', "
        "'exponential', and 'wigner_surmise'."
    )
