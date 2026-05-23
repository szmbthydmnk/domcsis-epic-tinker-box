"""Compute the arithmetic mean of a numeric vector."""
function mean(xs::AbstractVector{<:Number})::Float64
    isempty(xs) && throw(ArgumentError("cannot compute mean of empty vector"))
    return sum(xs) / length(xs)
end
