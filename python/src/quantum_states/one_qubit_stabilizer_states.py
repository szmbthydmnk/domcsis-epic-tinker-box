# This code is part of Domcsi's Epic Tinker Box

import numpy as np

def one_qubit_stabilizer_state(identifier: str) -> np.ndarray:
    """
    
    """
    asdasdasdasd
    
    if not isinstance(id, str):
        raise TypeError(f"identifier must be of type string, got {type(identifier).__name__}")
    if id == "":
        raise ValueError("identifier was empty")
    
    if identifier == "x":
        return _x_positive_eigenstate()


def _x_positive_eigenstate() -> np.ndarray:
    """ Internal helper; gives back the  """
    return np.array(1/np.sqrt(2) * np.array([1, 1]))
    