"""
Statistical moments for finite-sample real and complex ensembles.

All functions operate on any `AbstractVector`; both real and complex
elements are handled via separate dispatch where semantics differ.
Functions match the Python implementations exactly in their definitions,
but are expressed idiomatically in Julia.
"""
module StatisticalMoments

export mean_real, mean_complex, variance, skewness


"""
    mean_real(data::AbstractVector{<:Real}) -> Float64

Arithmetic mean of a vector of real numbers.

# Arguments
- `data`: Non-empty vector of real numbers.

# Throws
- `ArgumentError` if `data` is empty.
"""
function mean_real(data::AbstractVector{<:Real})::Float64
    isempty(data) && throw(ArgumentError("Data vector must not be empty."))
    return sum(data) / length(data)
end


"""
    mean_complex(data::AbstractVector{<:Complex}) -> Tuple{Float64, ComplexF64}

Return `(real_part_of_mean, complex_mean)` for a vector of complex numbers.

# Arguments
- `data`: Non-empty vector of complex numbers.

# Returns
A `Tuple{Float64, ComplexF64}` of `(real(mean), mean)`.

# Throws
- `ArgumentError` if `data` is empty.
"""
function mean_complex(data::AbstractVector{<:Complex})::Tuple{Float64, ComplexF64}
    isempty(data) && throw(ArgumentError("Data vector must not be empty."))
    m = sum(data) / length(data)
    return real(m), ComplexF64(m)
end


"""
    variance(data::AbstractVector{<:Real}) -> Float64

Population variance (divided by N) of a vector of real numbers.

# Arguments
- `data`: Non-empty vector of real numbers.

# Throws
- `ArgumentError` if `data` is empty.

The population variance is

    var = (1/n) \u00d7 \u03a3(x_i - \u03bc)^2
"""
function variance(data::AbstractVector{<:Real})::Float64
    isempty(data) && throw(ArgumentError("Data vector must not be empty."))
    \u03bc = mean_real(data)
    return sum((x - \u03bc)^2 for x in data) / length(data)
end


"""
    skewness(data::AbstractVector{<:Real}) -> Float64

Pearson's moment coefficient of skewness (population, N denominator).

    skewness = (1/n) \u00d7 \u03a3(x_i - \u03bc)^3 / var^(3/2)

# Arguments
- `data`: Non-empty vector of real numbers with non-zero variance.

# Throws
- `ArgumentError` if `data` is empty or has zero variance.
"""
function skewness(data::AbstractVector{<:Real})::Float64
    isempty(data) && throw(ArgumentError("Data vector must not be empty."))
    \u03bc  = mean_real(data)
    var = variance(data)
    var == 0 && throw(ArgumentError("Variance is zero; skewness is undefined."))
    return sum((x - \u03bc)^3 for x in data) / (length(data) * var^1.5)
end

end # module StatisticalMoments
