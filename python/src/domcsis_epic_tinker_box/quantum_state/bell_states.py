"""
This file contains functions helping to generate Bell states.

The Bell states are a specific type of entangled quantum state that involve two qubits. There are four Bell states, which can be expressed as follows:
|Φ+⟩ = (1/√2) * (|00�⟩ + |11⟩)
|Φ-⟩ = (1/√2) * (|00⟩ - |11⟩)
|Ψ+⟩ = (1/√2) * (|01⟩ + |10⟩)
|Ψ-⟩ = (1/√2) * (|01⟩ - |10⟩)   

References:
- Nielsen, M. A., & Chuang, I. L. (2010). Quantum computation and quantum information. Cambridge University Press.
- Bell, J. S. (1964). On the Einstein Podolsky Rosen paradox. Physics Physique Физика, 1(3), 195-200.

"""

from __future__ import annotations

import numpy as np

from .quantum_state import QuantumState

def bell_state(state_identifier: str | int) -> QuantumState:
    """
    Generates a Bell state based on the provided identifier.

    Arguments:
        state_identifier: A string ("Phi+", "Phi-", "Psi+", "Psi-") or an integer (0, 1, 2, 3) that identifies the Bell state to generate.

    Returns:
        A QuantumState instance representing the specified Bell state.

    Raises:
        ValueError: If the state_identifier is not recognized.
    """
    
    if isinstance(state_identifier, str):
        
        state_identifier = state_identifier.upper()
        if state_identifier == "PHI+":
            return QuantumState.from_vector(np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex))
        elif state_identifier == "PHI-":
            return QuantumState.from_vector(np.array([1/np.sqrt(2), 0, 0, -1/np.sqrt(2)], dtype=complex))
        elif state_identifier == "PSI+":
            return QuantumState.from_vector(np.array([0, 1/np.sqrt(2), 1/np.sqrt(2), 0], dtype=complex))
        elif state_identifier == "PSI-":
            return QuantumState.from_vector(np.array([0, 1/np.sqrt(2), -1/np.sqrt(2), 0], dtype=complex))
        else:
            raise ValueError(f"Unrecognized Bell state identifier: {state_identifier}. Valid options are 'Phi+', 'Phi-', 'Psi+', 'Psi-'.")
    elif isinstance(state_identifier, int):
        # If instead of a string an int is given, then the general N-qubit bell state is generated. 
        if state_identifier < 0:
            raise ValueError("State identifier must be a non-negative integer.")
        if state_identifier > 20:
            raise ValueError("Generating Bell states for more than 20 qubits is not supported due to memory constraints.")
        n_qubits = state_identifier
        bell_vector = np.zeros(2 ** n_qubits, dtype=complex)
        bell_vector[0] = 1 / np.sqrt(2)
        bell_vector[-1] = 1 / np.sqrt(2)
        return QuantumState.from_vector(bell_vector)
    else:
        raise ValueError(f"State identifier must be a string or an integer, got {type(state_identifier)}.")
