module DomcsisEpicTinkerBox

export resolve_rng,
       generate_gaussian_rnd_numbers,
       generate_uniform_rnd_numbers,
       generate_exponential_rnd_numbers,
       generate_wigner_surmise_rnd_numbers,
       generate_random_numbers

using Random, Distributions

# Global RNG fallback, analogous to Python's module-level Random()
const GLOBAL_RNG = Random.GLOBAL_RNG

"""
    resolve_rng(; rng::AbstractRNG = GLOBAL_RNG, seed::Union{Nothing,Int} = nothing)

Return a random number generator instance.

If `seed` is provided, a new `MersenneTwister` is constructed.
If `rng` is provided (and `seed` is nothing), that RNG is used.
If both are provided, an error is thrown.
"""
function resolve_rng(; rng::AbstractRNG = GLOBAL_RNG, seed::Union{Nothing,Int} = nothing)
    if rng !== GLOBAL_RNG && seed !== nothing
        throw(ArgumentError("Pass either `rng` or `seed`, not both."))
    end
    if seed !== nothing
        return MersenneTwister(seed)
    end
    return rng
end

"""
    generate_gaussian_rnd_numbers(; μ::Real = 0, σ::Real = 1, n::Integer = 1, rng::AbstractRNG = GLOBAL_RNG, seed::Union{Nothing,Int} = nothing) -> Vector{Float64}

Generate `n` random variables from a Gaussian distribution `𝒩(μ, σ²)`.
"""
function generate_gaussian_rnd_numbers(; μ::Real = 0, σ::Real = 1, n::Integer = 1,
                                       rng::AbstractRNG = GLOBAL_RNG,
                                       seed::Union{Nothing,Int} = nothing)
    σ <= 0 && throw(ArgumentError("Expected positive standard deviation, got $σ"))
    n <= 0 && throw(ArgumentError("Expected positive integer number of samples, got $n"))

    local_rng = resolve_rng(rng = rng, seed = seed)
    dist = Normal(μ, σ)
    return rand(local_rng, dist, n)
end

"""
    generate_uniform_rnd_numbers(; a::Real = 0, b::Real = 1, n::Integer = 1, rng::AbstractRNG = GLOBAL_RNG, seed::Union{Nothing,Int} = nothing) -> Vector{Float64}

Generate `n` random variables from a uniform distribution on `[a, b]`.
"""
function generate_uniform_rnd_numbers(; a::Real = 0, b::Real = 1, n::Integer = 1,
                                      rng::AbstractRNG = GLOBAL_RNG,
                                      seed::Union{Nothing,Int} = nothing)
    a > b && throw(ArgumentError("Expected a ≤ b, got a=$a, b=$b"))
    n <= 0 && throw(ArgumentError("Expected positive integer number of samples, got $n"))

    local_rng = resolve_rng(rng = rng, seed = seed)
    dist = Uniform(a, b)
    return rand(local_rng, dist, n)
end

"""
    generate_exponential_rnd_numbers(; λ::Real = 1, n::Integer = 1, rng::AbstractRNG = GLOBAL_RNG, seed::Union{Nothing,Int} = nothing) -> Vector{Float64}

Generate `n` random variables from an exponential distribution with rate `λ`.
"""
function generate_exponential_rnd_numbers(; λ::Real = 1, n::Integer = 1,
                                          rng::AbstractRNG = GLOBAL_RNG,
                                          seed::Union{Nothing,Int} = nothing)
    λ <= 0 && throw(ArgumentError("Expected positive rate parameter, got $λ"))
    n <= 0 && throw(ArgumentError("Expected positive integer number of samples, got $n"))

    local_rng = resolve_rng(rng = rng, seed = seed)
    dist = Exponential(λ)
    return rand(local_rng, dist, n)
end

"""
    generate_wigner_surmise_rnd_numbers(; β::Integer = 2, n::Integer = 1, rng::AbstractRNG = GLOBAL_RNG, seed::Union{Nothing,Int} = nothing) -> Vector{Float64}

Generate `n` samples from the generalized unit-mean Wigner surmise distribution

    p(s) = a_β s^β exp(-b_β s^2),

using the gamma–square-root construction.
"""
function generate_wigner_surmise_rnd_numbers(; β::Integer = 2, n::Integer = 1,
                                             rng::AbstractRNG = GLOBAL_RNG,
                                             seed::Union{Nothing,Int} = nothing)
    β <= 0 && throw(ArgumentError("Expected a positive integer for β, got $β"))
    n <= 0 && throw(ArgumentError("Expected a positive integer for n, got $n"))

    local_rng = resolve_rng(rng = rng, seed = seed)

    shape = 0.5 * (β + 1)
    b_β = (gamma(0.5 * (β + 2)) / gamma(0.5 * (β + 1)))^2
    scale = 1 / b_β
    dist = Gamma(shape, scale)

    y = rand(local_rng, dist, n)
    return sqrt.(y)
end

"""
    generate_random_numbers(distribution::AbstractString; params::Dict{String,Float64} = Dict(), n::Integer = 1, rng::AbstractRNG = GLOBAL_RNG, seed::Union{Nothing,Int} = nothing) -> Vector{Float64}

Convenience wrapper to sample from different distributions by name.

Supported `distribution` values:
- "gaussian"
- "uniform"
- "exponential"
- "wigner_surmise" or "wigner"
"""
function generate_random_numbers(distribution::AbstractString;
                                 params::Dict{String,Float64} = Dict{String,Float64}(),
                                 n::Integer = 1,
                                 rng::AbstractRNG = GLOBAL_RNG,
                                 seed::Union{Nothing,Int} = nothing)
    n <= 0 && throw(ArgumentError("Expected positive integer number of samples, got $n"))

    local_rng = resolve_rng(rng = rng, seed = seed)
    dist_lc = lowercase(distribution)

    if dist_lc == "gaussian"
        μ = get(params, "mu", 0.0)
        σ = get(params, "sigma", 1.0)
        return generate_gaussian_rnd_numbers(μ = μ, σ = σ, n = n, rng = local_rng)
    elseif dist_lc == "uniform"
        a = get(params, "a", 0.0)
        b = get(params, "b", 1.0)
        return generate_uniform_rnd_numbers(a = a, b = b, n = n, rng = local_rng)
    elseif dist_lc == "exponential"
        λ = get(params, "lambda", 1.0)
        return generate_exponential_rnd_numbers(λ = λ, n = n, rng = local_rng)
    elseif dist_lc == "wigner_surmise" || dist_lc == "wigner"
        β = Int(get(params, "beta", 2.0))
        return generate_wigner_surmise_rnd_numbers(β = β, n = n, rng = local_rng)
    else
        throw(ArgumentError("Unsupported distribution: " * repr(distribution) *
                            ". Supported values are 'gaussian', 'uniform', " *
                            "'exponential', and 'wigner_surmise'."))
    end
end

end # module DomcsisEpicTinkerBox
