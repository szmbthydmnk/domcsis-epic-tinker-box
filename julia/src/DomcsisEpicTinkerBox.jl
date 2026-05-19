module DomcsisEpicTinkerBox

include("statistical_moments.jl")
include("random_number_generation.jl")
include("pauli_operators.jl")
include("quantum_state.jl")
include("stabilizer_renyi_entropy.jl")

using .StatisticalMoments
using .PauliAlgebra
using .QuantumStates
using .Magic

export
    # StatisticalMoments
    mean_real, mean_complex, variance, skewness,
    # RandomNumberGeneration (already exported from its module)
    resolve_rng,
    generate_gaussian_rnd_numbers,
    generate_uniform_rnd_numbers,
    generate_exponential_rnd_numbers,
    generate_wigner_surmise_rnd_numbers,
    generate_random_numbers,
    # PauliAlgebra
    pauli_matrix,
    generate_all_pauli_strings,
    generate_pauli_operators,
    PauliOperators,
    PauliOperators_all,
    PauliOperators_from_strings,
    # QuantumStates
    QuantumState,
    from_vector,
    from_density_matrix,
    from_vectorized_density_matrix,
    state_vector,
    density_matrix,
    purity,
    # Magic
    stabilizer_renyi_entropy,
    stabilizer_renyi_entropy_unchecked,
    validate_alpha,
    validate_compatible,
    characteristic_function

end # module DomcsisEpicTinkerBox
