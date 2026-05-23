from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class QuantumState:
    """
    Immutable, validated quantum state.
    
    Stores either the state vector or the density matrix, but not both. the other representation is computed on demand and cached. For pure states, the state vector is the canonical representation. The density matrix is computed as rho = |psi><psi| when requested.
    
    For mixed states, the density matrix is the canonical representation. The state vector is not defined for mixed states and will raise an error if accessed.


    Attributes:
        n_qubits: Number of qubits.
        state: State represetnation - state vector (2^N), density matrix (2^N x 2^N) or vectorized density matrix (4^N).
        state_type: "vector" if state is a state vector, "density_matrix" if state is a density matrix, "vectorized_density_matrix" if state is a vectorized density matrix.
    
    Example usage:
        psi = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)  # Bell state |Φ+⟩
        state = QuantumState.from_vector(psi)
        print("State vector:", state.vector)
        print("Density matrix:\n", state.density_matrix)
        print("Purity:", state.purity)
        
    """
    
    n_qubits: int
    state: np.ndarray
    state_type: str  # "vector", "density_matrix", or "vectorized_density_matrix"

    # ------------------------------------------------------------------ #
    # Factory constructors                                               #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_vector(cls, psi: np.ndarray, atol: float = 1e-14) -> QuantumState:
        """
        Construct from a state vector. Validates shape and normalisation.

        Args:
            psi:  1D array of shape (2^N).
            atol: Tolerance for normalisation check.

        Raises:
            ValueError: If dimension is not a power of 2.
            ValueError: If state is not normalised.
        """
        
        psi = np.asarray(psi, dtype=complex)
        if psi.ndim != 1:
            raise ValueError(
                f"State vector must be 1D, got shape {psi.shape}."
            )
            
        n_qubits = _check_power_of_two(psi.shape[0])
        
        _check_normalised(psi @ psi.conj(), atol, "state vector")
        return cls(n_qubits=n_qubits, state=psi, state_type="vector")

    @classmethod
    def from_density_matrix(cls, rho: np.ndarray, atol: float = 1e-14) -> QuantumState:
        """
        Construct from a density matrix. Validates shape,
        Hermiticity, and trace.

        Args:
            rho:  2D array of shape (2^N, 2^N).
            atol: Tolerance for all checks.

        Raises:
            ValueError: If shape is not square or dimension not a power of 2.
            ValueError: If rho is not Hermitian.
            ValueError: If Tr(rho) != 1.
        """
        rho = np.asarray(rho, dtype=complex)
        
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
            raise ValueError(
                f"Density matrix must be square 2D array, got shape {rho.shape}."
            )
            
        n_qubits = _check_power_of_two(rho.shape[0])
        
        if not np.allclose(rho, rho.conj().T, atol=atol):
            raise ValueError("Density matrix is not Hermitian.")
        
        _check_normalised(np.trace(rho), atol, "density matrix trace")
        
        
        return cls(n_qubits=n_qubits, 
                   state=rho, 
                   state_type="density_matrix",
                   )


    @classmethod
    def from_vectorized_density_matrix(cls, rho_vec: np.ndarray, atol: float = 1e-14) -> QuantumState:
        """
        Constructs from a vectorized density matrix. Validates shape, Hermiticity, and trace.
        
        Args:
            rho_vec: 1D array of shape (4^N,).
            atol: Tolerance for all checks.
            
        Raises:
            ValueError: If shape is not 1D or dimension not a power of 4.
            ValueError: If reshaped rho is not Hermitian.
            ValueError: If Tr(rho) != 1.        
        """
        
        rho_vec = np.asarray(rho_vec, dtype=complex)
        
        if rho_vec.ndim != 1:
            raise ValueError(
                f"Vectorized density matrix must be 1D array, got shape {rho_vec.shape}."
            )
            
        n_qubits = _check_power_of_two(int(np.sqrt(rho_vec.shape[0])))
        
        rho = rho_vec.reshape((2 ** n_qubits, 2 ** n_qubits))
        
        if not np.allclose(rho, rho.conj().T, atol=atol):
            raise ValueError("Density matrix is not Hermitian.")
        
        _check_normalised(np.trace(rho), atol, "density matrix trace")
        
        return cls(
            n_qubits=n_qubits, 
            state=rho_vec, 
            state_type="vectorized_density_matrix",
            )
        
    # ------------------------------------------------------------------ #
    # Derived properties                                                   #
    # ------------------------------------------------------------------ #

    @property
    def dim(self) -> int:
        """Hilbert space dimension 2^n."""
        return int(2 ** self.n_qubits)

    @property
    def density_matrix(self) -> np.ndarray:
        """Compute rho = |psi><psi| on demand."""
        return np.outer(self.vector, self.vector.conj())

    @property
    def vector(self) -> np.ndarray:
        """
        Returns the state vector from a density matrix or a vectorized density matrix if that is pure, else raises an error.
        """
        
        if self.state_type == "vector":
            return self.state
        
        if self.state_type == "density_matrix":
            rho = self.state
        elif self.state_type == "vectorized_density_matrix":
            rho = self.state.reshape((self.dim, self.dim))
        else:
            raise ValueError(f"Invalid state_type: {self.state_type}")
        
        # Check purity: Tr(rho^2) == 1 for pure states
        if not np.isclose(np.trace(rho @ rho), 1.0, atol=1e-14):
            raise ValueError("State is not pure, cannot extract state vector.")
        
        # For a pure state, the eigenvector with eigenvalue 1 is the state vector
        eigvals, eigvecs = np.linalg.eigh(rho)
        idx = np.argmax(eigvals)
        return eigvecs[:, idx]
    
    
    # =============================================================================
    # State manipulation methods (e.g. partial trace) could be added here in the future. 
    # For now, this class is just a validated container for quantum states. 
    # ============================================================================
    
    @property
    def purity(self) -> float:
        """Purity Tr(rho^2). 1 for pure states, <1 for mixed states."""
        if self.state_type == "vector":
            return 1.0
        elif self.state_type == "density_matrix":
            rho = self.state
        elif self.state_type == "vectorized_density_matrix":
            rho = self.state.reshape((self.dim, self.dim))
        else:
            raise ValueError(f"Invalid state_type: {self.state_type}")
        
        return float(np.real(np.trace(rho @ rho)))
    

# ============================================================================
# Private helpers
# ============================================================================

def _check_power_of_two(dim: int) -> int:
    """Return log2(dim) if dim is a power of 2, else raise."""
    n = int(np.log2(dim))
    if 2 ** n != dim:
        raise ValueError(
            f"State dimension {dim} is not a power of 2."
        )
    return n


def _check_normalised(norm: complex, atol: float, label: str) -> None:
    if not np.isclose(norm.real, 1.0, atol=atol) or abs(norm.imag) > atol:
        raise ValueError(
            f"State is not normalised: {label} norm = {norm:.7f}, expected 1.0."
        )