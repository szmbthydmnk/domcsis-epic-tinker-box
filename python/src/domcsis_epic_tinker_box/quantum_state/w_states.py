"""
This file contains functions helping to generate W states.

The W state is a specific type of entangled quantum state that is a superposition of states where exactly one qubit is in the state |1⟩ and the rest are in the state |0⟩. For n qubits, the W state can be expressed as:
|W_n⟩ = (1/√n) * (|100...0⟩ + |010...0⟩ + ... + |000...1⟩)

References:
- Dür, W., Vidal, G., & Cirac, J. I. (2000). Three qubits can be entangled in two inequivalent ways. Physical Review A, 62(6), 062314. 
- Zeilinger, A., Horne, M. A., & Greenberger, D. M. (1993). Higher-order quantum entanglement. In Proceedings of the 1993 Symposium on the Foundations of Modern Physics (pp. 73-80). World Scientific. 
"""

from __future__ import annotations

import numpy as np

from .quantum_state import QuantumState

def w_state(no_qubits: int) -> QuantumState:
    """
    Generates the W state for a given number of qubits.
    
    Argsuments:
        no_qubits: The number of qubits in the W state. Must be a positive integer.
    
    Returns:
        A QuantumState instance representing the W state.
        
    Raises:
        ValueError: If no_qubits is not a positive integer.
        ValueError: If no_qubits is greater than 20 due to memory constraints.
    """
    
    if no_qubits <= 0:
        raise ValueError("Number of qubits must be a positive integer.")
    if no_qubits > 20:
        raise ValueError("Generating W states for more than 20 qubits is not supported due to memory constraints.")
    
    # Create the W state vector
    w_vector = np.zeros(2 ** no_qubits, dtype=complex)
    for i in range(no_qubits):
        w_vector[1 << i] = 1 / np.sqrt(no_qubits)
        
    return QuantumState.from_vector(w_vector)
