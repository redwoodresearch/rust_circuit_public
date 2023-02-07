import pytest

from rust_circuit import *

circs = [
    Add(Array.randn(128, 32, 32), Scalar(1.0, (), None)),
    Add(Add(Scalar(5.5, (2, 2), None), Scalar(1.0, (), None)), Scalar(1.0, (), None)),
]


def erroring_function(*args, **kwargs):
    assert False


def test_filter():
    assert list(filter_nodes(circs[0], lambda x: isinstance(x, Scalar))) == [Scalar(1.0, (), None)]
    with pytest.raises(Exception):
        list(filter_nodes(circs[0], erroring_function))


def map_scalarconstant_plus_one(x):
    if isinstance(x, Scalar):
        return Scalar(x.value + 1.0, (), None)
    else:
        return x


def test_deep_map():

    circs[0].print()
    deep_map(circs[0], map_scalarconstant_plus_one).print()
    assert deep_map(circs[0], map_scalarconstant_plus_one) == Add(circs[0].children[0], Scalar(2.0, (), None))
    with pytest.raises(Exception):
        deep_map(circs[0], erroring_function)

    def exact_mapper(x):
        if x == Add(Scalar(5.5, (2, 2), None), Scalar(1.0, (), None)):
            return Scalar(9.9, (2, 2), None)
        else:
            return map_scalarconstant_plus_one(x)

    deep_map(circs[1], exact_mapper).print()
    assert deep_map_preorder(circs[1], exact_mapper) == Add(Scalar(9.9, (2, 2), None), Scalar(2.0, (), None))
    assert deep_map(circs[1], exact_mapper) == deep_map(circs[1], map_scalarconstant_plus_one)


def test_path_get():
    path_get(circs[1], [0, 0]).print()
    assert path_get(circs[1], [0, 0]) == Scalar(5.5, (2, 2), None)


def test_update():
    circs[1].print()
    updated = update_nodes(circs[1], lambda x: x == Scalar(5.5, (2, 2), None), lambda x: Scalar(77.0, (2, 2), None))
    updated.print()
    updated2 = update_path(circs[1], [0, 0], lambda x: Scalar(77.0, (2, 2), None))
    updated2.print()


def test_device_dtype_op_repr_and_conversion():
    tdd = TorchDeviceDtype("cuda:0", "int32")
    assert repr(tdd) == 'TorchDeviceDtype(device="cuda:0", dtype="int32")'

    tdd.op()
    assert repr(tdd.op()) == 'TorchDeviceDtypeOp(device="cuda:0", dtype="int32")'

    assert repr(TorchDeviceDtypeOp()) == "TorchDeviceDtypeOp(device=None, dtype=None)"
    assert repr(TorchDeviceDtypeOp(dtype="int32")) == 'TorchDeviceDtypeOp(device=None, dtype="int32")'
    assert repr(TorchDeviceDtypeOp(device="cpu")) == 'TorchDeviceDtypeOp(device="cpu", dtype=None)'


if __name__ == "__main__":
    test_filter()
    test_deep_map()
    test_path_get()
    test_update()
