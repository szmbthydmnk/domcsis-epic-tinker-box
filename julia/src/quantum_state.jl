"""
Immutable, validated quantum state container.

Three canonical representations are supported:
- `:vector`                   \u2014 state vector of shape (2^N,)
- `:density_matrix`           \u2014 density matrix of shape (2^N, 2^N)
- `:vectorized_density_matrix`\u2014 row-stacked vec(\u03c1) of shape (4^N,)

The non-canonical forms are computed on demand.  Mixed states (purity < 1)
are fully supported when constructed from a density matrix or its
vectorised form; accessing `state_vector` on a mixed state throws.
"""
module QuantumStates

export QuantumState,
       from_vector,
       from_density_matrix,
       from_vectorized_density_matrix,
       state_vector,
       density_matrix,
       purity

using LinearAlgebra


# ============================================================================
# Type definition
# ============================================================================

"""
    QuantumState

Immutable validated quantum state.

# Fields
- `n_qubits::Int`
- `state::Union{Vector{ComplexF64}, Matrix{ComplexF64}}`
- `state_type::Symbol`  \u2014 one of `:vector`, `:density_matrix`, `:vectorized_density_matrix`
"""
struct QuantumState
    n_qubits  ::Int
    state     ::Union{Vector{ComplexF64}, Matrix{ComplexF64}}
    state_type::Symbol
end


# ============================================================================
# Private helpers
# ============================================================================

function _check_power_of_two(dim::Int)::Int
    n = round(Int, log2(dim))
    2^n != dim && throw(ArgumentError("Dimension $(dim) is not a power of 2."))
    return n
end

function _check_normalised(norm::Number, atol::Float64, label::String)
    if abs(real(norm) - 1.0) > atol || abs(imag(norm)) > atol
        throw(ArgumentError(
            "State is not normalised: $(label) norm = $(norm), expected 1.0."
        ))
    end
end


# ============================================================================
# Factory constructors
# ============================================================================

"""
    from_vector(psi::AbstractVector; atol::Float64 = 1e-14) -> QuantumState

Construct a `QuantumState` from a state vector. Validates that the dimension
is a power of 2 and that the vector is normalised.

# Throws
- `ArgumentError` if dimension is not a power of 2.
- `ArgumentError` if the vector is not normalised.
"""
function from_vector(
    psi::AbstractVector;
    atol::Float64 = 1e-14
)::QuantumState
    \u03c8 = Vector{ComplexF64}(psi)
    n_qubits = _check_power_of_two(length(\u03c8))
    _check_normalised(dot(\u03c8, \u03c8), atol, "state vector")
    return QuantumState(n_qubits, \u03c8, :vector)
end


"""
    from_density_matrix(\u03c1::AbstractMatrix; atol::Float64 = 1e-14) -> QuantumState

Construct a `QuantumState` from a density matrix.
Validates shape (square, power-of-2 dimension), Hermiticity, and trace.
Both pure and mixed states are accepted.

# Throws
- `ArgumentError` if the matrix is not square or its dimension is not a power of 2.
- `ArgumentError` if the matrix is not Hermitian.
- `ArgumentError` if `Tr(\u03c1) \u2260 1`.
"""
function from_density_matrix(
    \u03c1::AbstractMatrix;
    atol::Float64 = 1e-14
)::QuantumState
    size(\u03c1, 1) != size(\u03c1, 2) && throw(ArgumentError(
        "Density matrix must be square, got shape $(size(\u03c1))."
    ))
    n_qubits = _check_power_of_two(size(\u03c1, 1))
    isapprox(\u03c1, \u03c1'; atol = atol) || throw(ArgumentError("Density matrix is not Hermitian."))
    _check_normalised(tr(\u03c1), atol, "density matrix trace")
    return QuantumState(n_qubits, Matrix{ComplexF64}(\u03c1), :density_matrix)
end


"""
    from_vectorized_density_matrix(\u03c1_vec::AbstractVector; atol::Float64 = 1e-14) -> QuantumState

Construct a `QuantumState` from a vectorised (row-stacked) density matrix.
Validates that the length is a perfect square with a power-of-2 dimension,
and that the reshaped matrix is Hermitian with trace 1.

# Throws
- `ArgumentError` on shape, Hermiticity, or trace violations.
"""
function from_vectorized_density_matrix(
    \u03c1_vec::AbstractVector;
    atol::Float64 = 1e-14
)::QuantumState
    dim = round(Int, sqrt(length(\u03c1_vec)))
    dim^2 != length(\u03c1_vec) && throw(ArgumentError(
        "Length of vectorised density matrix ($(length(\u03c1_vec))) is not a perfect square."
    ))
    n_qubits = _check_power_of_two(dim)
    \u03c1 = reshape(Vector{ComplexF64}(\u03c1_vec), dim, dim)
    isapprox(\u03c1, \u03c1'; atol = atol) || throw(ArgumentError("Density matrix is not Hermitian."))
    _check_normalised(tr(\u03c1), atol, "density matrix trace")
    return QuantumState(n_qubits, Vector{ComplexF64}(\u03c1_vec), :vectorized_density_matrix)
end


# ============================================================================
# Derived quantities
# ============================================================================

"""
    dim(s::QuantumState) -> Int

Hilbert space dimension 2^n.
"""
dim(s::QuantumState)::Int = 2^s.n_qubits


"""
    state_vector(s::QuantumState) -> Vector{ComplexF64}

Return the state vector. For density-matrix or vectorised forms the
state vector is extracted via eigendecomposition (pure states only).

# Throws
- `ArgumentError` if the state is mixed (purity < 1).
"""
function state_vector(s::QuantumState)::Vector{ComplexF64}
    s.state_type === :vector && return s.state::Vector{ComplexF64}

    d = dim(s)
    \u03c1 = s.state_type === :density_matrix ?
        s.state::Matrix{ComplexF64} :
        reshape(s.state::Vector{ComplexF64}, d, d)

    isapprox(real(tr(\u03c1 * \u03c1)), 1.0; atol = 1e-14) ||
        throw(ArgumentError("State is not pure; state vector is undefined for mixed states."))

    F = eigen(Hermitian(\u03c1))
    idx = argmax(real.(F.values))
    return Vector{ComplexF64}(F.vectors[:, idx])
end


"""
    density_matrix(s::QuantumState) -> Matrix{ComplexF64}

Return the density matrix. For vector-canonical states, computes \u03c1 = |\u03c8\u27e9\u27e8\u03c8|.
For vectorised states, reshapes the stored array.
"""
function density_matrix(s::QuantumState)::Matrix{ComplexF64}
    if s.state_type === :vector
        \u03c8 = s.state::Vector{ComplexF64}
        return \u03c8 * \u03c8'
    elseif s.state_type === :density_matrix
        return s.state::Matrix{ComplexF64}
    else
        d = dim(s)
        return reshape(s.state::Vector{ComplexF64}, d, d)
    end
end


"""
    purity(s::QuantumState) -> Float64

Purity `Tr(\u03c1\u00b2)`. Returns `1.0` immediately for vector-canonical states.
"""
function purity(s::QuantumState)::Float64
    s.state_type === :vector && return 1.0
    \u03c1 = density_matrix(s)
    return real(tr(\u03c1 * \u03c1))
end

end # module QuantumStates
