import pytest

from rust_circuit import *


def test_schedule_serialize():
    circ = Einsum.from_einsum_string("ab,bc->ac", Array.randn(2, 3), Array.randn(3, 4))
    schedule = optimize_to_schedule(circ)
    serialized = schedule.serialize()
    print(serialized)
    deserialized = Schedule.deserialize(serialized)
    print(deserialized)


@pytest.mark.skip(reason="we don't have server test set up yet")
def test_remote_execution():
    circ = Einsum.from_einsum_string("ab,bc->ac", Array.randn(2, 3), Array.randn(3, 4))
    schedule = optimize_to_schedule(circ)
    result = schedule.evaluate_remote("http://0.0.0.0:9876")
    print(result)


if __name__ == "__main__":
    # test_schedule_serialize()
    test_remote_execution()
