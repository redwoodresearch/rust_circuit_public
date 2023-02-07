import uuid

import pytest

from rust_circuit import Add, Array, Cumulant, Einsum, Symbol, circuit_is_leaf, circuit_is_var, print_circuit_type_check


def test_circuit_type():
    assert print_circuit_type_check(Cumulant) is Cumulant
    assert isinstance(Symbol([2, 3], uuid.uuid4(), name="hi"), Symbol)
    assert print_circuit_type_check(Symbol) is Symbol
    assert print_circuit_type_check(Array) is Array
    assert print_circuit_type_check(Einsum) is Einsum
    assert print_circuit_type_check(Add) is Add
    with pytest.raises(TypeError):
        print_circuit_type_check(int)  # type:ignore
    with pytest.raises(TypeError):
        print_circuit_type_check(None)  # type:ignore
    assert circuit_is_leaf(Symbol([2, 3], uuid.uuid4(), name="hi"))
    assert not circuit_is_var(Symbol([2, 3], uuid.uuid4(), name="hi"))


if __name__ == "__main__":
    test_circuit_type()
