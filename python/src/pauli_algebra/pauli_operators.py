# This code is part of...

# Last updated: 2026.05.08


from scipy.sparse import identity, csr_matrix       # Sparse identity matrix; general sparse array
            # There is a difference between csc and csr matrix:
                # One is compressed through rows the other through columns, for Paulis there is no different.


# Defining the Pauli matrices 
_PAULI_I = identity(2, dtype=complex, format="csr")
_PAULI_X = csr_matrix([[0, 1], [1, 0]], dtype=complex)
_PAULI_Y = csr_matrix([[0, 0 - 1j], [0 + 1j, 0]], dtype=complex)
_PAULI_Z = csr_matrix([[1, 0], [0, -1]], dtype=complex)

# A "single-qubit" Pauli generator wrapper for the above defined Pauli matrices.
def pauli_matrix(which: str | int) -> csr_matrix:
    """
    Return a single-qubit Pauli matrix in sparse CSR format.
    Supported labels:
        - 'I', 'i', or 0
        - 'X', 'x', or 1
        - 'Y', 'y', or 2
        - 'Z', 'z', or 3
    Args:
        which:
            Pauli matrix identifier.
    Returns:
        scipy.sparse.csr_matrix:
            2x2 sparse Pauli matrix.
    Raises:
        ValueError:
            If the label is unsupported.
    """
    if isinstance(which, str):
        if len(which) != 1:
            raise ValueError(f"Pauli label must be have length 1, got '{which}'")
        
    if which in ('I', 'i', 0):
        return _PAULI_I
    if which in ('X', 'x', 1):
        return _PAULI_X
    if which in ('Y', 'y', 2):
        return _PAULI_Y
    if which in ('Z', 'z', 3):
        return _PAULI_Z

    raise ValueError(f"Unsupported Pauli label: '{which}'")

