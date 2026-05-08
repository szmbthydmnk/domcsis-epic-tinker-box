from domcsis_epic_tinker_box.pauli_algebra import generate_all_pauli_strings


def test_single_qubit_length():
    result = generate_all_pauli_strings(1)
    assert len(result) == 4


def test_single_qubit_values():
    result = generate_all_pauli_strings(1)
    assert set(result) == {"I", "X", "Y", "Z"}


def test_two_qubit_length():
    result = generate_all_pauli_strings(2)
    assert len(result) == 16


def test_invalid_input():
    import pytest
    with pytest.raises(ValueError):
        generate_all_pauli_strings(0)
