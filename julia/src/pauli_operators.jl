"""
Pauli operator algebra for N-qubit systems.

Provides single-qubit Pauli matrices in sparse format, multi-qubit Pauli
string construction via tensor product, and an immutable `PauliOperators`
collection that bundles labels with their corresponding matrices.

Sparse matrices use `SparseMatrixCSC{ComplexF64, Int}` throughout,
matching the scipy CSR representation in the Python source. Kronecker
products are computed via `kron` from `SparseArrays`.
"""
module PauliAlgebra

export pauli_matrix,
       generate_all_pauli_strings,
       generate_pauli_operators,
       PauliOperators

using SparseArrays
using LinearAlgebra

# ============================================================================
# Single-qubit Pauli matrices — module-level constants
# ============================================================================

const PAULI_I = sparse(ComplexF64[1 0; 0 1])
const PAULI_X = sparse(ComplexF64[0 1; 1 0])
const PAULI_Y = sparse(ComplexF64[0 -1im; 1im 0])
const PAULI_Z = sparse(ComplexF64[1 0; 0 -1])


"""
    pauli_matrix(which::Union{AbstractString, Integer}) -> SparseMatrixCSC{ComplexF64, Int}

Return a single-qubit Pauli matrix in sparse format.

Supported identifiers:
- `"I"`, `"i"`, or `0` \u2192 identity
- `"X"`, `"x"`, or `1` \u2192 Pauli X
- `"Y"`, `"y"`, or `2` \u2192 Pauli Y
- `"Z"`, `"z"`, or `3` \u2192 Pauli Z

# Throws
- `ArgumentError` for unsupported labels.
"""
function pauli_matrix(which::AbstractString)::SparseMatrixCSC{ComplexF64, Int}
    length(which) != 1 && throw(ArgumentError("Pauli label must have length 1, got '$(which)'"))
    c = uppercase(which[1])
    c == 'I' && return PAULI_I
    c == 'X' && return PAULI_X
    c == 'Y' && return PAULI_Y
    c == 'Z' && return PAULI_Z
    throw(ArgumentError("Unsupported Pauli label: '$(which)'"))
end

function pauli_matrix(which::Integer)::SparseMatrixCSC{ComplexF64, Int}
    which == 0 && return PAULI_I
    which == 1 && return PAULI_X
    which == 2 && return PAULI_Y
    which == 3 && return PAULI_Z
    throw(ArgumentError("Unsupported Pauli index: $(which). Expected 0\u20133."))
end


# ============================================================================
# Multi-qubit Pauli string construction
# ============================================================================

"""
    _pauli_string_to_matrix(pauli_string::AbstractString) -> SparseMatrixCSC{ComplexF64, Int}

Build the 2^n \u00d7 2^n sparse matrix for a Pauli string via tensor product.
"""
function _pauli_string_to_matrix(pauli_string::AbstractString)::SparseMatrixCSC{ComplexF64, Int}
    return foldl(kron, (pauli_matrix(string(c)) for c in pauli_string))
end


"""
    generate_all_pauli_strings(n_qubits::Integer) -> Vector{String}

Return all 4^n Pauli strings for `n_qubits` qubits, in lexicographic order
`II\u2026I, II\u2026X, \u2026, ZZ\u2026Z`.

# Throws
- `ArgumentError` if `n_qubits <= 0`.
"""
function generate_all_pauli_strings(n_qubits::Integer)::Vector{String}
    n_qubits <= 0 && throw(ArgumentError("Number of qubits must be positive, got $(n_qubits)."))
    letters = ("I", "X", "Y", "Z")
    return [join(t) for t in Iterators.product(fill(letters, n_qubits)...)]
end


"""
    generate_pauli_operators(n_qubits::Integer) -> OrderedDict{String, SparseMatrixCSC{ComplexF64, Int}}
    generate_pauli_operators(pauli_strings::AbstractSet{<:AbstractString}) -> OrderedDict{String, SparseMatrixCSC{ComplexF64, Int}}

Generate sparse matrix representations of multi-qubit Pauli operators.

When called with an integer, generates all 4^n Pauli operators.
When called with a set of Pauli strings, generates only those operators.
Always returns an ordered dict mapping label \u2192 matrix.

# Throws
- `ArgumentError` for non-positive `n_qubits` or invalid Pauli strings.
"""
function generate_pauli_operators(
    n_qubits::Integer
)::Vector{Pair{String, SparseMatrixCSC{ComplexF64, Int}}}
    strings = generate_all_pauli_strings(n_qubits)
    return [s => _pauli_string_to_matrix(s) for s in strings]
end

function generate_pauli_operators(
    pauli_strings::AbstractSet{<:AbstractString}
)::Vector{Pair{String, SparseMatrixCSC{ComplexF64, Int}}}
    for s in pauli_strings
        if !all(c -> c \u2208 "IXYZ", uppercase(s))
            throw(ArgumentError("Invalid Pauli string: '$(s)'"))
        end
    end
    return [uppercase(s) => _pauli_string_to_matrix(uppercase(s)) for s in pauli_strings]
end


# ============================================================================
# PauliOperators — immutable validated collection
# ============================================================================

"""
    PauliOperators

Immutable, validated collection of N-qubit Pauli operators.

Bundles `labels` (tuple of Pauli strings) with their corresponding
`matrices` (tuple of sparse matrices). The only supported constructors
are `PauliOperators.all(n)` and `PauliOperators.from_strings(...)`.  
Direct field construction is intentionally not exported.

# Fields
- `n_qubits::Int`: Number of qubits.
- `labels::NTuple{N, String}`: Ordered Pauli string labels.
- `matrices::NTuple{N, SparseMatrixCSC{ComplexF64, Int}}`: Corresponding matrices.
"""
struct PauliOperators{N}
    n_qubits::Int
    labels  ::NTuple{N, String}
    matrices::NTuple{N, SparseMatrixCSC{ComplexF64, Int}}
end

"""
    PauliOperators_all(n_qubits::Integer) -> PauliOperators

Construct a `PauliOperators` containing all 4^n Pauli operators for `n_qubits`.
"""
function PauliOperators_all(n_qubits::Integer)::PauliOperators
    pairs = generate_pauli_operators(n_qubits)
    lbls  = Tuple(first.(pairs))
    mats  = Tuple(last.(pairs))
    return PauliOperators(n_qubits, lbls, mats)
end

"""
    PauliOperators_from_strings(pauli_strings::AbstractSet{<:AbstractString}) -> PauliOperators

Construct a `PauliOperators` from an explicit subset of Pauli strings.
"""
function PauliOperators_from_strings(
    pauli_strings::AbstractSet{<:AbstractString}
)::PauliOperators
    pairs = generate_pauli_operators(pauli_strings)
    n     = length(first(first(pairs)))   # length of first label = n_qubits
    lbls  = Tuple(first.(pairs))
    mats  = Tuple(last.(pairs))
    return PauliOperators(n, lbls, mats)
end

# Convenience iteration
Base.length(p::PauliOperators) = length(p.labels)
Base.iterate(p::PauliOperators, state=1) = state > length(p) ? nothing : ((p.labels[state], p.matrices[state]), state + 1)

"""
    as_dict(p::PauliOperators) -> Dict{String, SparseMatrixCSC{ComplexF64, Int}}
"""
function as_dict(p::PauliOperators)
    return Dict(lbl => mat for (lbl, mat) in p)
end

end # module PauliAlgebra
