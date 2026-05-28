import math
from qiskit.quantum_info import SparsePauliOp

from .model import ModelParams


def magnetisation_observables(params: ModelParams):
    labels = []
    for qubit in range(params.L):
        string = "I" * params.L
        string = string[:qubit] + "Z" + string[qubit + 1 :]
        labels.append(string)
    return [SparsePauliOp(label) for label in labels]


def correlator_observables(params: ModelParams):
    labels = []
    center = math.ceil(params.L / 2 - 1)

    for qubit in range(params.L):
        string_list = list("I" * params.L)
        string_list[center] = "Z"
        string_list[qubit] = "Z"
        if qubit == center:
            string_list = list("I" * params.L)
        labels.append("".join(string_list))

    return [SparsePauliOp(label) for label in labels]
