from rust_circuit import Array
from rust_circuit.py_utils import timed


def test_serialization():
    shape = (1024, 10)
    with timed("init"):
        arr = Array.randn(*shape)
    with timed("save"):
        arr.save_rrfs()
    hash_base16 = arr.tensor_hash_base16()
    print(hash_base16)
    with timed("load"):
        arr2 = Array.from_hash(None, hash_base16)
    print(arr == arr2)
    arr.print()
    arr2.print()


if __name__ == "__main__":
    test_serialization()
    print("done")
