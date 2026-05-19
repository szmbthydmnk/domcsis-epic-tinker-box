"""
This file handles Pauli operator and Pauli string generation and manipulation.

public functions:
- pauli_matrix
- generate_all_pauli_strings
- generate_pauli_operators

private functions:

"""
# =====================================================================================
from __future__ import annotations
    # functools.reduce is used to build the multi-qubit Pauli matrices via tensor 
    # product 
from functools import reduce
    # itertools.product is used to generate all combinations of Pauli labels for a 
    # given number of qubits
from itertools import product
    # typing.overload is used to provide type hints for the generate_pauli_operators 
    # function, which has different return types based on the input arguments 
from typing import overload, cast
    #
from dataclasses import dataclass
# =====================================================================================
    # Sparse identity matrix; general sparse array manipulation; sparse Kronecker product
from scipy.sparse import identity, csr_matrix, kron   
# =====================================================================================

# Defining the sngle-qubit Pauli matrices 
_PAULI_I = identity(2, dtype=complex, format="csr")
_PAULI_X = csr_matrix([[0, 1], [1, 0]], dtype=complex)
_PAULI_Y = csr_matrix([[0, 0 - 1j], [0 + 1j, 0]], dtype=complex)
_PAULI_Z = csr_matrix([[1, 0], [0, -1]], dtype=complex)

@dataclass(frozen=True)
class  PauliOperators:
    """
    Immutable, validated collection of N-qubit operators.
    Always store the sparse matrices and labels together.
    """
    
    n_qubits: int
    labels: tuple[str, ...]             # ("II", "IX", ..., "ZZ")
    matrices: tuple[csr_matrix, ...]    # matching sparse matrices
    
    # ------------------------------------------------------------------ #
    # Factory constructors — the only way to build a valid instance      #
    # ------------------------------------------------------------------ #
    @classmethod
    def all(cls, n_qubits: int) -> "PauliOperators":
        """
        Generates the full 4^N Pauli space.
        """
        
        operators: dict[str, csr_matrix] = cast(dict[str, csr_matrix], generate_pauli_operators(n_qubits, as_dict=True))
        return cls(
            n_qubits=n_qubits,
            labels=tuple(operators.keys()),
            matrices=tuple(operators.values()),
        )
    
    
    @classmethod
    def from_strings(cls, pauli_strings: set[str]) -> "PauliOperators":
        """
        Generates operators for an explicit subset of Pauli strings.
        """
        
        operators: dict[str, csr_matrix] = cast(dict[str, csr_matrix], generate_pauli_operators(pauli_strings))
        m: int = len(next(iter(pauli_strings)))
        
        return cls(
            n_qubits=m,
            labels=tuple(operators.keys()),
            matrices=tuple(operators.values())
        )

    # ------------------------------------------------------------------ #
    # Convinience access                                                 #
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.labels)
    
    
    def items(self) -> zip[tuple[str, csr_matrix]]:
        """
        Iterate as (label, matrix) pairs - like dict.items().
        """
        return zip(self.labels, self.matrices)
    
    
    def as_dict(self) -> dict[str, csr_matrix]:
        return dict(zip(self.labels, self.matrices))
    
# Internal helpers
def _pauli_string_to_matrix(pauli_string: str) -> csr_matrix:
    """
    Build the sparse matrix for a multi-qubit Pauli string via tensor product.

    Args:
        pauli_string: A string of Pauli labels, e.g. 'XYZ'.

    Returns:
        scipy.sparse.csr_matrix: 2^n x 2^n sparse matrix.
    """
    return reduce(
        lambda a, b: kron(a, b, format="csr"),
        (pauli_matrix(p) for p in pauli_string),
    )


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


def generate_all_pauli_strings(no_qubits: int) -> list[str]:
    """
    Returns a list with the N-qubit Pauli strings as strings.

    Args:
        no_qubits: int
            number of qubits.
    
    Returns:
        list[str]

    Raises:
        ValueError:
            If the number of qubits is not positive or not an integer.
    """
    if no_qubits <= 0:
        raise ValueError(f"Number of qubits must be positive, got '{no_qubits}'")
    
    _PAULI_LETTERS = ["I", "X", "Y", "Z"]

    return ["".join(s) for s in product(_PAULI_LETTERS, repeat = no_qubits)]
    

@overload
def generate_pauli_operators(
    which: int,
    as_dict: bool = ...,
) -> list[csr_matrix] | dict[str, csr_matrix]: ...

@overload
def generate_pauli_operators(
    which: set[str],
) -> list[csr_matrix] | dict[str, csr_matrix]: ...

def generate_pauli_operators(
        which: int | set[str],
        as_dict: bool = False,
) -> list[csr_matrix] | dict[str, csr_matrix]:
    """
    Generate sparse matrix representations of multi-qubit Pauli operators.

    When called with an int:
        Generates all 4^n Pauli operators for n qubits.
        Returns an ordered list by default, or a dict if as_dict=True.

    When called with a set of strings:
        Generates operators only for the provided Pauli strings.
        Always returns a dict mapping string -> matrix.
        The as_dict argument is ignored in this case.

    Args:
        which:
            - int: number of qubits; generates all 4^n operators.
            - set[str]: explicit set of Pauli strings to generate.
        as_dict:
            Only relevant when which is an int. If True, return a dict
            instead of a list. Default is False.

    Returns:
        list[csr_matrix]:
            Ordered list of operators (only when which is an int and
            as_dict=False).
        dict[str, csr_matrix]:
            Mapping from Pauli string to sparse matrix.

    Raises:
        ValueError: If which is an int that is not positive.
        ValueError: If which is a set containing invalid Pauli strings.

    Examples:
        >>> ops = generate_pauli_operators(1)          # list of 4 matrices
        >>> ops = generate_pauli_operators(2, as_dict=True)   # dict, 16 entries
        >>> ops = generate_pauli_operators({"XI", "IZ"})      # dict, 2 entries
    """

    if isinstance(which, int):
        strings = generate_all_pauli_strings(which)
        if as_dict:
            return {s: _pauli_string_to_matrix(s) for s in strings}
        return [_pauli_string_to_matrix(s) for s in strings]

    # set[str] branch
    result: dict[str, csr_matrix] = {}
    for s in which:
        if not all(c in "IXYZ" for c in s.upper()):
            raise ValueError(f"Invalid Pauli string: '{s}'")
        result[s] = _pauli_string_to_matrix(s.upper())
    return result