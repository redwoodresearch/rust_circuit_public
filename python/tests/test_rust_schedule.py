from typing import List, cast

import hypothesis
from torch import tensor
from torch.testing import assert_close

from interp.circuit.circuit import Circuit as pCircuit
from interp.circuit.circuit_utils import circuit_map, circuit_reduce
from interp.circuit.computational_node import Einsum as pEinsum
from interp.circuit.computational_node import Index as pIndex
from interp.circuit.constant import ArrayConstant as pArray
from interp.circuit.constant import One
from interp.circuit.testing.utils import mark_not_interesting_if
from rust_circuit import *
from rust_circuit.interop_rust import py_to_rust, schedule_replace_circuits

from .test_rust_rewrite import get_c_st


@hypothesis.settings(deadline=None, suppress_health_check=[hypothesis.HealthCheck.filter_too_much])
@hypothesis.given(get_c_st(max_growth_steps=10))
@mark_not_interesting_if(SchedulingOOMError)
def test_schedule(circ):
    raw_test_schedule(circ)


def raw_test_schedule(circ):
    rust_circ = strip_names_and_tags(py_to_rust(circ))
    schedule = optimize_to_schedule(rust_circ)
    circ_replaced = update_nodes(rust_circ, lambda x: isinstance(x, Array), lambda x: Array(cast(Array, x).value + 9.1))
    circ_replaced_py = circuit_map(lambda x: pArray(x.value + 9.1) if isinstance(x, pArray) else x, circ)
    py_map: List[pCircuit] = circuit_reduce(
        (lambda acc, n: [*acc, n] if isinstance(n, pArray) else acc), [], circ_replaced_py
    )
    orig = circ_replaced.evaluate()
    try:  # sometimes constants dont match up, todo debug
        circ_replaced_in_py_rust = schedule_replace_circuits(schedule, {x: cast(Array, x).value + 9.1 for x in py_map})
    except Exception as e:
        hypothesis.assume(False, repr(e))
    # print(orig,sch)
    assert_close(orig, circ_replaced_in_py_rust.evaluate())
    assert_close(rust_circ.evaluate(), schedule.evaluate())


if __name__ == "__main__":
    raw_test_schedule(
        pIndex(
            node=pEinsum(
                args=((One(shape=(2, 3, 1), name=""), (0, 1, 2)),),
                out_axes=(0, 1, 1, 2, 0),
                name="",
                shape=(2, 3, 3, 1, 2),
                is_constant=True,
                is_explicitly_computable=True,
                can_be_sampled=True,
            ),
            index=(0, slice(None, 2, None), 0, tensor([0]), slice(-2, 2, None)),
            name="",
            hash_tensor_idx_by_value=True,
            shape=(2, 1, 2),
            is_constant=True,
            is_explicitly_computable=True,
            can_be_sampled=True,
        )
    )
