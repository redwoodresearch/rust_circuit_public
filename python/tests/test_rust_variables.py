from uuid import uuid4

import attrs
import hypothesis
import hypothesis.extra.numpy as st_np
import torch

from interp.circuit.circuit import Circuit
from interp.circuit.circuit_compiler.util import RearrangeSpec
from interp.circuit.circuit_utils import cast_circuit, circuit_map, get_nodes
from interp.circuit.computational_node import Add, GeneralFunction, Index, UnaryRearrange
from interp.circuit.constant import ArrayConstant, One, Zero
from interp.circuit.get_update_node import FunctionIterativeNodeMatcher as F
from interp.circuit.get_update_node import NodeUpdater as NU
from interp.circuit.print_circuit import lambda_notation_circuit
from interp.circuit.testing.topdown_circuit import CircuitProbs as CP
from interp.circuit.testing.topdown_circuit import st_Circuit
from interp.circuit.var import AutoTag
from rust_circuit import Scalar as rScalar
from rust_circuit import Tag as rTag
from rust_circuit.interop_rust import py_to_rust


@hypothesis.settings(max_examples=200)
@hypothesis.given(
    circ=st_Circuit(
        st_np.array_shapes(min_dims=0),
        must_be_constant=False,
        must_be_explicitly_computable=False,
        growth_order="breadth_first",
        possibly_different_name=True,
        probs_default=CP.kw(
            all=1,
            Zero=0,
            One=0,
            GeneralFunction=0,
            Add=(1, dict(use_weights=False)),
            Index=(1, dict(allow_out_of_bounds_slice=False)),
        ),  # TODO: fix rearrange maybe
    ),
)
def test_rust_and_back(circ: Circuit):
    all_circs = get_nodes(circ)
    for c in all_circs:
        hypothesis.assume(not isinstance(c, (Zero, One, GeneralFunction)), "How did we get here???")
        if isinstance(c, Add):
            hypothesis.assume(all(x == 1.0 for x in c.items.values()), "How did we get here???")

    raw_test_rust_and_back(circ)


def raw_test_rust_and_back(circ: Circuit):
    circ = cast_circuit(circ, dtype=torch.float64)
    circ = circuit_map(
        lambda x: UnaryRearrange.from_spec(
            x.node,
            RearrangeSpec.from_rust(
                x.get_spec().to_rust().canonicalize().fill_empty_ints().conform_to_input_shape(x.node.shape)
            ),
            name=x.name,
        )
        if isinstance(x, UnaryRearrange)
        else x,
        circ,
    )
    circ_tagged = circuit_map(lambda x: AutoTag(x, x.uuid) if isinstance(x, ArrayConstant) else x, circ)
    rust_circ = py_to_rust(circ_tagged)
    back_py = rust_circ.to_py()
    back_py = NU(
        lambda x: attrs.evolve(x.node, uuid=x.uuid) if isinstance(x, AutoTag) else x,
        F(lambda x: isinstance(x, AutoTag) and isinstance(x.node, ArrayConstant)),
        assert_exists=False,
        assert_unique=False,
    )(back_py)
    back_py = circuit_map(
        lambda x: attrs.evolve(x, hash_tensor_idx_by_value=True) if isinstance(x, Index) else x, back_py
    )

    if back_py != circ:
        print(lambda_notation_circuit(back_py, tensors_as_randn=True))
        print(lambda_notation_circuit(circ, tensors_as_randn=True))
    assert back_py == circ


def test_auto_tag():
    assert rTag.new_with_random_uuid(rScalar(1)) != rTag.new_with_random_uuid(rScalar(1))
    common_uuid = uuid4()
    assert rTag(rScalar(1), common_uuid) == rTag(rScalar(1), common_uuid)
