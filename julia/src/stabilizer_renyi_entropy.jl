"""
Stabilizer Rényi Entropy (SRE) for quantum states.

Implements the \u03b1-SRE

    M_\u03b1(|\u03c8\u27e9) = [log\u2082(\u03a3_P \u039e(P)^\u03b1) / (1 - \u03b1)] - log\u2082 d

where the characteristic function is

    \u039e(P) = |\u27e8\u03c8|P|\u03c8\u27e9|\u00b2 / 2^n.

For \u03b1 = 1 the Shannon-entropy limit is used:

    M_1 = -\u03a3_P \u039e(P) log\u2082 \u039e(P) - log\u2082 d,

with the convention 0 log 0 = 0.

A fast (unchecked) path is provided for hot loops over many states
with fixed Paulis and \u03b1; validate once outside the loop.
"""
module Magic

export stabilizer_renyi_entropy,
       stabilizer_renyi_entropy_unchecked,
       validate_alpha,
       validate_compatible,
       characteristic_function

using LinearAlgebra
using ..PauliAlgebra: PauliOperators
using ..QuantumStates: QuantumState, state_vector


# ============================================================================
# Validation helpers
# ============================================================================

"""
    validate_alpha(\u03b1::Real)

Validate the Rényi order \u03b1.

- \u03b1 \u2264 0        : throws `ArgumentError`.
- \u03b1 = 1        : issues a warning (Shannon-entropy limit).
- \u03b1 non-integer: issues a warning (generalised \u03b1-SRE).
- \u03b1 = 2, 3, \u2026  : valid, no warning.
"""
function validate_alpha(\u03b1::Real)
    \u03b1 <= 0 && throw(ArgumentError(
        "\u03b1 must be positive, got $(\u03b1). SRE is not defined for \u03b1 \u2264 0."
    ))
    if \u03b1 == 1
        @warn "\u03b1=1: computing M\u2081 directly (Shannon-entropy limit of the SRE)."
    elseif !isinteger(\u03b1)
        @warn "\u03b1=$(\u03b1) is not an integer. Computing the generalised \u03b1-Rényi entropy."
    end
end


"""
    validate_compatible(state::QuantumState, paulis::PauliOperators)

Throw an `ArgumentError` if `state.n_qubits \u2260 paulis.n_qubits`.
"""
function validate_compatible(state::QuantumState, paulis::PauliOperators)
    state.n_qubits != paulis.n_qubits && throw(ArgumentError(
        "state.n_qubits=$(state.n_qubits) \u2260 paulis.n_qubits=$(paulis.n_qubits)."
    ))
end


# ============================================================================
# Core computation
# ============================================================================

"""
    characteristic_function(\u03c8::AbstractVector{<:Complex}, paulis::PauliOperators) -> Vector{Float64}

Compute \u039e(P) = |\u27e8\u03c8|P|\u03c8\u27e9|\u00b2 / 2^n for every P in `paulis`.

Returns a `Vector{Float64}` of length `4^n` in label order.
The matrix-vector products `P * \u03c8` exploit sparsity via `SparseMatrixCSC`.
"""
function characteristic_function(
    \u03c8::AbstractVector{<:Complex},
    paulis::PauliOperators
)::Vector{Float64}
    norm_factor = 2^paulis.n_qubits
    return Float64[abs2(dot(\u03c8, P * \u03c8)) / norm_factor for (_, P) in paulis]
end


"""
    _compute_sre(\u03be::AbstractVector{Float64}, \u03b1::Real) -> Float64

Evaluate M_\u03b1 from the characteristic function vector \u03be.

For \u03b1 = 1 uses the Shannon-entropy formula with `0 log 0 := 0`.
"""
function _compute_sre(\u03be::AbstractVector{Float64}, \u03b1::Real)::Float64
    d = length(\u03be)
    if \u03b1 == 1
        return -sum(x > 0 ? x * log2(x) : 0.0 for x in \u03be) - log2(d)
    end
    return log2(sum(x^\u03b1 for x in \u03be)) / (1 - \u03b1) - log2(d)
end


# ============================================================================
# Unchecked fast path
# ============================================================================

"""
    stabilizer_renyi_entropy_unchecked(state, paulis, \u03b1 = 2; pauli_spectrum = false)

Fast-path SRE computation with no validation.

The caller is responsible for ensuring:
- `\u03b1 > 0`
- `state.n_qubits == paulis.n_qubits`
- `state` is a valid, normalised pure `QuantumState`

Intended for hot loops:

```julia
validate_alpha(\u03b1)
validate_compatible(states[1], paulis)
results = [stabilizer_renyi_entropy_unchecked(s, paulis, \u03b1) for s in states]
```

# Returns
- `Float64` if `pauli_spectrum = false` (default).
- `Tuple{Float64, Dict{String, Float64}}` if `pauli_spectrum = true`.
"""
function stabilizer_renyi_entropy_unchecked(
    state      ::QuantumState,
    paulis     ::PauliOperators,
    \u03b1          ::Real = 2;
    pauli_spectrum::Bool = false
)
    \u03be  = characteristic_function(state_vector(state), paulis)
    sre = _compute_sre(\u03be, \u03b1)
    pauli_spectrum || return sre
    spectrum = Dict(lbl => \u03be[i] for (i, (lbl, _)) in enumerate(paulis))
    return sre, spectrum
end


# ============================================================================
# Public validated entry point
# ============================================================================

"""
    stabilizer_renyi_entropy(state, paulis, \u03b1 = 2; pauli_spectrum = false)

Compute the \u03b1-Stabilizer Rényi Entropy for a pure quantum state.

Validated entry point. For hot loops over many states with fixed
`paulis` and `\u03b1`, validate once then call `stabilizer_renyi_entropy_unchecked`.

# Arguments
- `state::QuantumState`         \u2014 pure quantum state.
- `paulis::PauliOperators`      \u2014 Pauli operator set matching `state.n_qubits`.
- `\u03b1::Real`                    \u2014 Rényi order (default 2).
- `pauli_spectrum::Bool = false`\u2014 if true, also return the \u039e spectrum.

# Returns
- `Float64` if `pauli_spectrum = false`.
- `Tuple{Float64, Dict{String, Float64}}` if `pauli_spectrum = true`.

# Throws
- `ArgumentError` if `\u03b1 \u2264 0` or `state.n_qubits \u2260 paulis.n_qubits`.
"""
function stabilizer_renyi_entropy(
    state      ::QuantumState,
    paulis     ::PauliOperators,
    \u03b1          ::Real = 2;
    pauli_spectrum::Bool = false
)
    validate_alpha(\u03b1)
    validate_compatible(state, paulis)
    return stabilizer_renyi_entropy_unchecked(state, paulis, \u03b1; pauli_spectrum = pauli_spectrum)
end

end # module Magic
