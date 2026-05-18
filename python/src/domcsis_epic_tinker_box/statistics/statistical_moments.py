"""
This module contains function for calulcuating statistical moments of finite sample random ensembles, such as mean, variance, skewness, kurtosis, and autocorrelation. These functions are useful for analyzing the properties of random variables and their distributions.

While these are very much standard functions in any library, there are some subtelties in their implementaiton. In order to avoid ambiguities, I implement these. 

However, keep in mind that these functions are not opitimized for performance but for clarity. In case calculation requires speed, I would recommend using libraries such as NumPy or SciPy, which have highly optimized implementations of these functions.
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np

def mean(data: List[float] | np.ndarray) -> float:
    """
    Calculates the mean(average) of a list of real numbers.
    
    Arguments:
    data -- A list or numpy array of real numbers.
    
    Returns:
    The mean of the input data.
    
    Raises:
    ValueError -- If the input data is empty or contains complex numbers.
    """
    
    if len(data) == 0:
        raise ValueError("Data list must not be empty")
    
    if isinstance(data, np.ndarray):
        if np.iscomplexobj(data):
            raise ValueError("Data list must contain real numbers only, use mean_complex for complex numbers")
    else:
        if any(isinstance(x, complex) for x in data):
            raise ValueError("Data list must contain real numbers only, use mean_complex for complex numbers")
    return sum(data) / len(data)
    
    
def mean_complex(data: List[complex] | np.ndarray) -> Tuple[float, complex]:
    """
    Calculates the mean (average) of a list of complex numbers.
    
    Arguments:
    data -- A list or numpy array of complex numbers.
    
    Returns:
    A tuple containing the real part of the mean and the complex mean itself.
    
    Raises:
    ValueError -- If the input data is empty or contains non-complex numbers.
    """
    
    if len(data) == 0:
        raise ValueError("Data list must not be empty")
    
    if isinstance(data, np.ndarray):
        if not np.iscomplexobj(data):
            raise ValueError("Data list must contain complex numbers only, use mean for real numbers")
    else:
        if any(not isinstance(x, complex) for x in data):
            raise ValueError("Data list must contain complex numbers only, use mean for real numbers")
    
    real_mean = mean([x.real for x in data])
    imag_mean = mean([x.imag for x in data])
    
    return real_mean, complex(real_mean, imag_mean)


def variance(data: List[float] | np.ndarray) -> float:
    """
    Calculates the variance of a list of real numbers.
    
    Arguments:
    data -- A list or numpy array of real numbers.
    
    Returns:
    The variance of the input data.
    
    Raises:
    ValueError -- If the input data is empty or contains complex numbers.
    """
    
    if len(data) == 0:
        raise ValueError("Data list must not be empty")
    
    if isinstance(data, np.ndarray):
        if np.iscomplexobj(data):
            raise ValueError("Data list must contain real numbers only")
    else:
        if any(isinstance(x, complex) for x in data):
            raise ValueError("Data list must contain real numbers only")
    
    data_mean = mean(data)
    return sum((x - data_mean) ** 2 for x in data) / len(data)



def skewness(data: List[float] | np.ndarray) -> float:
    """
    Calculates the skewness of a list of real numbers.
    
    Arguments:
    data -- A list or numpy array of real numbers.
    
    Returns:
    The skewness of the input data.
    
    Raises:
    ValueError -- If the input data is empty or contains complex numbers.
    
        Skewness is a measure of the asymmetry of the probability distribution of a real-valued random variable about its mean. A positive skewness indicates that the distribution has a longer tail on the right side, while a negative skewness indicates a longer tail on the left side. A skewness of zero indicates a perfectly symmetrical distribution.
        
        Expression:
        skewness = (1/n) * sum((x_i - mean)^3) / (variance^(3/2))
    """
    
    if len(data) == 0:
        raise ValueError("Data list must not be empty")
    
    if isinstance(data, np.ndarray):
        if np.iscomplexobj(data):
            raise ValueError("Data list must contain real numbers only")
    else:
        if any(isinstance(x, complex) for x in data):
            raise ValueError("Data list must contain real numbers only")
    
    data_mean = mean(data)
    data_variance = variance(data)
    
    if data_variance == 0:
        raise ValueError("Variance of the data must be non-zero to calculate skewness")
    
    return sum((x - data_mean) ** 3 for x in data) / (len(data) * (data_variance ** 1.5))


