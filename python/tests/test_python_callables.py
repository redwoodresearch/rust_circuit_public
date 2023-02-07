import pytest

import rust_circuit as rc


def erroring_f(c):
    assert False


def thunk():
    pass


def c2c(c):
    return c


def ci2c(i, c):
    return c.rename(str(i))


def printy(c):
    print("found", c)


def test_python_callables():
    circuit = rc.Einsum.from_einsum_string("ab,bc->ac", rc.Array.randn(2, 3), rc.Array.randn(3, 4))
    print(circuit.get_compatible_device_dtype())
    print(circuit.map_children(c2c))
    print(circuit.map_children_enumerate(ci2c))

    with pytest.raises(AssertionError):
        print(circuit.map_children(erroring_f))
    with pytest.raises(TypeError):
        print(circuit.map_children(thunk))  # type: ignore
    circuit.visit(printy)
    with pytest.raises(TypeError):
        circuit.visit(c2c)  # type: ignore


if __name__ == "__main__":
    test_python_callables()
