"""
This module provides functions for generating random numbers from various distributions, including Gaussian, uniform, and exponential distributions. Each function allows the user to specify parameters such as mean, standard deviation, bounds, and rate parameter, as well as the number of random variables to generate. The functions also include error handling to ensure that the input parameters are valid.
"""

import random

def generate_gaussian_rnd_numbers(mu: float = 0, sigma: float = 1, n: int = 1) -> list[float]:
    """
    Generates n random variables from a Gaussian distribution with a mean of mu and a standard deviation of sigma.
    
    Args:
        mu: float
            The mean of the Gaussian distribution.
        sigma: float
            The standard deviation of the Gaussian distribution.
        n: int
            The number of random variables to generate.
    
    Returns:
            A list of n random variables drawn from the specified Gaussian distribution.
    Raises:
        ValueError:
            If sigma is not positive or if n is not a positive integer.
    """
    
    if sigma < 0:
        raise ValueError(f"Expected positive standard deviation, got {sigma}")
    if n <= 0:
        raise ValueError(f"Expected positive number of samples, got {n}")
    
    return [random.gauss(mu, sigma) for _ in range(n)]


def generate_uniform_rnd_numbers(a: float = 0, b: float = 1, n: int = 1) -> list[float]:
    """
    Generates n random variables from a uniform distribution over the interval [a, b].
    
    Args:
        a: float
            The lower bound of the uniform distribution.
        b: float
            The upper bound of the uniform distribution.
        n: int
            The number of random variables to generate.
    
    Returns:
            A list of n random variables drawn from the specified uniform distribution.
    Raises:
        ValueError:
            If a is greater than b or if n is not a positive integer.
    """
    
    if a > b:
        raise ValueError(f"Expected lower bound a to be less than or equal to upper bound b, got a={a}, b={b}")
    if n <= 0:
        raise ValueError(f"Expected positive number of samples, got {n}")
    
    return [random.uniform(a, b) for _ in range(n)]


def generate_exponential_rnd_numbers(lambd: float = 1, n: int = 1) -> list[float]:
    """
    Generates n random variables from an exponential distribution with rate parameter lambd.
    
    Args:
        lambd: float
            The rate parameter of the exponential distribution (must be positive).
        n: int
            The number of random variables to generate.
    Returns:
            A list of n random variables drawn from the specified exponential distribution.
    Raises:
        ValueError:
            If lambd is not positive or if n is not a positive integer.
    """
    if lambd <= 0:
        raise ValueError(f"Expected positive rate parameter, got {lambd}")
    if n <= 0:
        raise ValueError(f"Expected positive number of samples, got {n}")
    
    return [random.expovariate(lambd) for _ in range(n)]    



def generate_random_numbers(distribution: str, params: dict, n: int = 1) -> list[float]:
    """
    Generates n random variables from a specified distribution with given parameters.
    
    Args:
        distribution: str
            The name of the distribution to sample from. Supported values are 'gaussian', 'uniform', and 'exponential'.
        params: dict
            A dictionary containing the parameters for the specified distribution. For 'gaussian', expected keys are 'mu' and 'sigma'. For 'uniform', expected keys are 'a' and 'b'. For 'exponential', expected key is 'lambda'.
        n: int
            The number of random variables to generate.
    Returns:
            A list of n random variables drawn from the specified distribution.
    Raises:
        ValueError:
            If the distribution is unsupported, if required parameters are missing, or if n is not a positive integer.  
    """
    
    # Sample size validation
    if n <= 0:
        raise ValueError(f"Expected positive number of samples, got {n}")
    
    # Distribution selection and parameter validation
    if distribution == 'gaussian':
        if 'mu' not in params or 'sigma' not in params:
            raise ValueError("Missing parameters for Gaussian distribution. Expected keys: 'mu', 'sigma'")
        return generate_gaussian_rnd_numbers(params['mu'], params['sigma'], n)
    elif distribution == 'uniform':
        if 'a' not in params or 'b' not in params:
            raise ValueError("Missing parameters for Uniform distribution. Expected keys: 'a', 'b'")
        return generate_uniform_rnd_numbers(params['a'], params['b'], n)
    elif distribution == 'exponential':
        if 'lambda' not in params:
            raise ValueError("Missing parameter for Exponential distribution. Expected key: 'lambda'")
        return generate_exponential_rnd_numbers(params['lambda'], n)
    else:
        raise ValueError(f"Unsupported/not implemented distribution: '{distribution}'. Supported values are 'gaussian', 'uniform', and 'exponential'.") 
    
    if distribution == 'gaussian':
        return generate_gaussian_rnd_numbers(params['mu'], params['sigma'], n)
    elif distribution == 'uniform':
        return generate_uniform_rnd_numbers(params['a'], params['b'], n)
    elif distribution == 'exponential':
        return generate_exponential_rnd_numbers(params['lambda'], n)
    
    