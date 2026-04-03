## Clifford Gates and Circuits

- generate_single_qubit_clifford()
- generate_n_qubit_clifford()
- generate_random_clifford(n_qubits)
- generate_clifford_from_symplectic(symplectic_matrix)
- generate_clifford_from_tableau(tableau)

- clifford_gate_set()
- minimal_clifford_generators()

- compose_clifford(C1, C2)
- invert_clifford(C)
- conjugate_pauli_by_clifford(P, C)

- generate_brickwall_clifford(n_qubits, depth)
- generate_random_clifford_layer(n_qubits)

- clifford_to_unitary(C)
- clifford_to_tableau(C)
- tableau_to_clifford(tableau)

- stabilizer_state_from_clifford()
- random_stabilizer_state(n_qubits)

- stabilizer_measurement_update()
- stabilizer_partial_trace()

- compute_clifford_depth(circuit)
- compute_clifford_width(circuit)

- decompose_clifford_to_gates()
- synthesize_clifford_from_pauli_maps()

## Pauli String Methods

- generate_pauli_string(n_qubits)
- generate_all_pauli_strings(n_qubits)

- pauli_string_multiply(P1, P2)
- pauli_string_commutes(P1, P2)
- pauli_string_anticommutes(P1, P2)

- pauli_string_to_binary_vector()
- binary_vector_to_pauli_string()

- generate_pauli_basis(n_qubits)

- pauli_weight(P)
- pauli_support(P)

- random_pauli_string(n_qubits)

## Pauli Orbits

- compute_pauli_orbit(P, group)
- compute_full_pauli_orbits(group)

- orbit_representative(P)
- orbit_size(P)

- stabilizer_subgroup(P)

## Magic Measures

- compute_stabilizer_renyi_entropy(rho, alpha)

- compute_mana(rho)
- compute_log_mana(rho)

- compute_robustness_of_magic(rho)

- compute_relative_entropy_of_magic(rho)

- compute_stabilizer_extent(rho)

- compute_wigner_function(rho)

- compute_negative_volume_wigner(rho)

- compute_magic_monotones()

- compute_magic_growth_rate()

- magic_measure_bootstrap()

## Entanglement Measures

### Pure States

- von_neumann_entropy(psi)
- renyi_entropy(psi, alpha)

- schmidt_decomposition(psi)

- entanglement_entropy(psi)

### Mixed States

- entanglement_of_formation(rho)

- concurrence(rho)

- negativity(rho)
- logarithmic_negativity(rho)

- relative_entropy_of_entanglement(rho)

- squashed_entanglement(rho)

### Mutual Information

- quantum_mutual_information(rho)

- conditional_mutual_information(rho)

- classical_mutual_information(p)

### Multipartite

- geometric_entanglement(psi)

- global_entanglement_measure()

- tangle_measure()


## Pure State Generators

- generate_random_state(n_qubits)

- generate_haar_random_state(n_qubits)

- generate_product_state()

- generate_ghz_state()

- generate_w_state()

- generate_cluster_state()

- generate_random_stabilizer_state()

- generate_magic_state()

## Mixed States

- generate_random_density_matrix()

- generate_thermal_state(H, beta)

- generate_depolarized_state()

- generate_classical_mixture()

- generate_maximally_mixed_state()

## Density Matrix Tools

- partial_trace(rho, subsystem)

- partial_transpose(rho)

- purity(rho)

- fidelity(rho1, rho2)

- trace_distance(rho1, rho2)

- quantum_relative_entropy(rho1, rho2)

- diagonalize_density_matrix()

- eigenvalue_spectrum(rho)

- normalize_density_matrix()

- check_density_matrix_validity()

- tensor_product_states()

- reshape_density_matrix()

- vectorize_density_matrix()

## Unitary Generation

- generate_random_unitary(n)

- generate_haar_unitary(n)

- generate_random_SU2()

- generate_random_U2()

- generate_random_SU(n)

- generate_random_U(n)

- generate_circular_unitary_ensemble()

- generate_circular_orthogonal_ensemble()

- generate_circular_symplectic_ensemble()

## Hermitian

- generate_random_hermitian()

- generate_gue_matrix()

- generate_goe_matrix()

- generate_random_positive_matrix()


## Spin Hamiltonians

- generate_ising_hamiltonian()

- generate_xyz_hamiltonian()

- generate_heisenberg_hamiltonian()

- generate_transverse_field_ising()

- generate_random_local_hamiltonian()

- generate_k_local_hamiltonian()

- generate_sparse_spin_hamiltonian()

- generate_dense_spin_hamiltonian()

- vectorized_spin_hamiltonian()

- generate_periodic_boundary_conditions()

- generate_open_boundary_conditions()


## Lindblad Dynamics

- generate_lindbladian()

- lindblad_superoperator()

- vectorized_lindblad_form()

- compute_steady_state()

- lindblad_time_evolve()

- unravel_lindblad_trajectory()

- generate_jump_operators()

- compute_liouvillian_spectrum()

- dissipative_gap()


## Time Evolution

### Exact

- matrix_exponential()

- diagonalization_evolution()

### Numerical Integrators

- runge_kutta_4()

- adaptive_runge_kutta()

- krylov_time_evolution()

### Magnus Methods

- magnus_expansion_order_1()

- magnus_expansion_order_2()

- magnus_expansion_order_3()

- magnus_expansion_general()

### Dyson Series

- dyson_series_order_1()

- dyson_series_order_2()

- dyson_series_general()

### Trotterization

- trotter_step()

- suzuki_trotter_order_2()

- suzuki_trotter_order_4()

- randomized_trotter()

### Operator Builders

- time_evolution_operator()

- time_ordered_exponential()

- midpoint_hamiltonian_expansion()


## Statistical Tools

- bootstrap_resample()

- jackknife_resample()

- histogram()

- kernel_density_estimate()

- cumulative_distribution()

- empirical_distribution_function()

- confidence_interval()

- percentile_interval()

- moving_average()

- weighted_average()

- variance_estimator()

- covariance_matrix()

- principal_component_analysis()


## Probability Estimation

- maximum_likelihood_estimation()

- bayesian_update()

- compute_posterior()

- likelihood_ratio_test()

- bootstrap_error_estimation()

- monte_carlo_sampling()

- importance_sampling()

- rejection_sampling()

## Error Mitigation

- zero_noise_extrapolation()

- probabilistic_error_cancellation()

- measurement_error_mitigation()

- readout_error_correction()

- noise_model_estimation()

- randomized_compiling()


## Information Measures

- fisher_information()

- quantum_fisher_information()

- classical_fisher_information()

- shannon_entropy()

- cross_entropy()

- kullback_leibler_divergence()

- mutual_information()

- conditional_entropy()

- channel_capacity_estimation()

## Group Theory

- generate_group_generators()

- group_multiply()

- group_inverse()

- subgroup_generation()

- coset_decomposition()

- orbit_generation()

- stabilizer_group()

- compute_character_table()

- irreducible_representation()

- group_action_on_set()

## Linear Algebra

- eigen_decomposition()

- singular_value_decomposition()

- qr_decomposition()

- polar_decomposition()

- matrix_logarithm()

- matrix_square_root()

- kronecker_product()

- commutator()

- anti_commutator()

- projector()

- matrix_norm()

- spectral_norm()

## Visualization

- plot_histogram()

- plot_heatmap()

- plot_matrix_spectrum()

- plot_pauli_distribution()

- plot_entanglement_growth()

- plot_magic_growth()

- plot_time_evolution_observable()


## Performance Tools

- sparse_matrix_conversion()

- batch_matrix_operations()

- parallel_map()

- memory_profile()

- runtime_benchmark()

- vectorized_expectation_values()

- jit_compile_helpers()


DomcsisEpicTinkerBox/

├── python/
│   ├── clifford/
│   │   ├── generation.py
│   │   ├── tableau.py
│   │   ├── synthesis.py
│   │   └── circuits.py
│
│   ├── pauli/
│   │   ├── strings.py
│   │   ├── binary_representation.py
│   │   ├── orbits.py
│   │   └── basis.py
│
│   ├── magic/
│   │   ├── sre.py
│   │   ├── mana.py
│   │   ├── robustness.py
│   │   └── wigner.py
│
│   ├── entanglement/
│   │   ├── entropy.py
│   │   ├── negativity.py
│   │   ├── concurrence.py
│   │   └── multipartite.py
│
│   ├── states/
│   │   ├── pure.py
│   │   ├── mixed.py
│   │   └── stabilizer.py
│
│   ├── density/
│   │   ├── trace.py
│   │   ├── fidelity.py
│   │   ├── validation.py
│   │   └── transformations.py
│
│   ├── random_matrices/
│   │   ├── unitary.py
│   │   ├── hermitian.py
│   │   └── ensembles.py
│
│   ├── hamiltonians/
│   │   ├── spin.py
│   │   ├── local.py
│   │   └── boundary_conditions.py
│
│   ├── lindblad/
│   │   ├── superoperators.py
│   │   ├── time_evolution.py
│   │   └── steady_states.py
│
│   ├── evolution/
│   │   ├── magnus.py
│   │   ├── dyson.py
│   │   ├── trotter.py
│   │   ├── integrators.py
│   │   └── operators.py
│
│   ├── statistics/
│   │   ├── bootstrap.py
│   │   ├── histogram.py
│   │   ├── distributions.py
│   │   └── estimators.py
│
│   ├── probability/
│   │   ├── inference.py
│   │   ├── monte_carlo.py
│   │   └── sampling.py
│
│   ├── error_mitigation/
│   │   ├── noise_models.py
│   │   ├── mitigation.py
│   │   └── calibration.py
│
│   ├── information/
│   │   ├── fisher.py
│   │   ├── entropy.py
│   │   └── mutual_information.py
│
│   ├── group_theory/
│   │   ├── groups.py
│   │   ├── representations.py
│   │   └── actions.py
│
│   ├── linear_algebra/
│   │   ├── decompositions.py
│   │   ├── matrix_functions.py
│   │   └── tensor_products.py
│
│   ├── visualization/
│   │   ├── plots.py
│   │   └── heatmaps.py
│
│   ├── performance/
│   │   ├── parallel.py
│   │   ├── profiling.py
│   │   └── vectorization.py
│
│   └── utils/
│       ├── constants.py
│       ├── validation.py
│       └── logging.py
│
├── julia/
│   └── (mirrored structure)
│
├── tests/
│   ├── python/
│   └── julia/
│
├── benchmarks/
│
├── examples/
│
├── docs/
│
├── notebooks/
│
├── LICENSE
├── README.md
└── pyproject.toml




- tensor_networks/
    - mpo_generation()
    - mps_operations()
    - dmrg_tools()

- quantum_channels/
    - kraus_conversion()
    - channel_distance()

- symmetry_reduction/
    - symmetry_sector_projection()