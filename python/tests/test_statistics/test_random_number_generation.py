from domcsis_epic_tinker_box.statistics.random_number_generation import generate_random_numbers, generate_gaussian_rnd_numbers, generate_uniform_rnd_numbers, generate_exponential_rnd_numbers
import pytest   

# =============================================================================
# generate_gaussian_rnd_numbers

def test_gaussian_rnd_numbers_length() -> None:
    n = 5
    numbers = generate_gaussian_rnd_numbers(n=n)
    assert len(numbers) == n
    

def test_gaussian_rnd_numbers_invalid_sigma() -> None:
    with pytest.raises(ValueError):
        generate_gaussian_rnd_numbers(sigma=-1, n=10)
        

def test_gaussian_rnd_numbers_zero_samples() -> None:
    with pytest.raises(ValueError):
        generate_gaussian_rnd_numbers(n=0)
        
        
def test_gaussian_rnd_numbers_mean_and_std() -> None:
    mu = 0
    sigma = 1
    n = 10000
    numbers = generate_gaussian_rnd_numbers(mu=mu, sigma=sigma, n=n)
    assert abs(sum(numbers)/n - mu) < 0.1  # Check if mean is close to mu
    assert abs((sum((x - mu)**2 for x in numbers)/n)**0.5 - sigma) < 0.1  # Check if std is close to sigma  
    
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
    a = 0
    b = 1
    n = 10000
    numbers = generate_uniform_rnd_numbers(a=a, b=b, n=n)
    assert all(a <= x <= b for x in numbers)  # Check if all numbers are in the range [a, b]
    
# =============================================================================
# generate_exponential_rnd_numbers

def test_exponential_rnd_numbers_length() -> None:
    n = 5
    numbers = generate_exponential_rnd_numbers(n=n)
    assert len(numbers) == n
    
def test_exponential_rand_numbers_zero_samples() -> None:
    with pytest.raises(ValueError):
        generate_exponential_rnd_numbers(n=0)

def test_exponential_rnd_numbers_mean() -> None:
    lambd = 1
    n = 10000
    numbers = generate_exponential_rnd_numbers(lambd=lambd, n=n)
    assert abs(sum(numbers)/n - 1/lambd) < 0.1  # Check if mean is close to 1/lambda
    
# =============================================================================
# generate_random_numbers

def test_generate_random_numbers_gaussian() -> None:
    distribution = 'gaussian'
    params = {'mu': 0, 'sigma': 1}
    n = 10000
    numbers = generate_random_numbers(distribution, params, n)
    assert len(numbers) == n
    assert abs(sum(numbers)/n - params['mu']) < 0.1  # Check if mean is close to mu
    assert abs((sum((x - params['mu'])**2 for x in numbers)/n)**0.5 - params['sigma']) < 0.1  # Check if std is close to sigma
    
    
def test_generate_random_numbers_uniform() -> None:
    distribution = 'uniform'
    params = {'a': 0, 'b': 1}
    n = 10000
    numbers = generate_random_numbers(distribution, params, n)
    assert len(numbers) == n
    assert all(params['a'] <= x <= params['b'] for x in numbers)  # Check if all numbers are in the range [a, b]    
    
    
def test_generate_random_numbers_exponential() -> None:
    distribution = 'exponential'
    params = {'lambda': 1}
    n = 10000
    numbers = generate_random_numbers(distribution, params, n)
    assert len(numbers) == n
    assert abs(sum(numbers)/n - 1/params['lambda']) < 0.1  # Check if mean is close to 1/lambda
    
    
def test_generate_random_numbers_invalid_distribution() -> None:
    distribution = 'invalid_dist'
    params = {}
    n = 10
    with pytest.raises(ValueError):
        generate_random_numbers(distribution, params, n)    
        
def test_generate_random_numbers_zero_samples() -> None:
    distribution = 'gaussian'
    params = {'mu': 0, 'sigma': 1}
    n = 0
    with pytest.raises(ValueError):
        generate_random_numbers(distribution, params, n)