from typing import Tuple

import hypothesis
import hypothesis.extra.numpy as st_np
import hypothesis.strategies as st
import torch
from torch.testing import assert_close

from interp.circuit.algebric_rewrite import (
    MulRearrangeSpec,
    MulRearrangeSpecSub,
    MulRestType,
    NamedItem,
    get_einsum_ordering_items,
)
from interp.circuit.circuit_utils import cast_circuit
from interp.circuit.test_algebric_rewrite import partitioned_or_rest
from interp.circuit.testing.topdown_circuit import CircuitProbs as CP
from interp.circuit.testing.topdown_circuit import st_Circuit
from interp.circuit.testing.utils import mark_not_interesting_if
from rust_circuit import Add, Einsum, NestAddsSpecSub, NestEinsumsSpecSub
from rust_circuit import NestRest as Rest
from rust_circuit import add_flatten, nest_adds, nest_einsums
from rust_circuit.interop_rust import py_to_rust

NestEinsumArgs = Tuple[Einsum, NestEinsumsSpecSub]


@st.composite
def st_nest_einsum_args(draw: st.DrawFn, must_be_explicitly_computable: bool = True) -> NestEinsumArgs:
    """TODO: currently does not generate any :class:`NamedItem` s or terminating node specs."""
    st_circuit = st_Circuit(
        st_np.array_shapes(min_dims=0),
        must_be_explicitly_computable=must_be_explicitly_computable,
        max_growth_steps=4,
        probs_default=CP.kw(all=1, Einsum=70),
        probs_per_depth=[CP.kw(all=0, Einsum=1), CP.kw(all=1, Einsum=70)],
        possibly_different_name=True,
    )
    circuit = draw(st_circuit)
    from interp.circuit.computational_node import Einsum as cEinsum

    assert isinstance(circuit, cEinsum)
    # TODO: improve this testing as needed!
    spec: MulRearrangeSpecSub
    if (n := len(get_einsum_ordering_items(circuit)[0].args)) == 0:
        spec = ()
    else:
        idxs = draw(st.permutations(range(n)))
        spec, _ = draw(partitioned_or_rest(idxs, can_have_rest=True))

    def convert(spec: MulRearrangeSpec) -> NestEinsumsSpecSub:
        if isinstance(spec, int):
            return spec
        if isinstance(spec, tuple):
            return [convert(s) for s in spec]
        if isinstance(spec, MulRestType):
            return Rest()
        assert not isinstance(spec, NamedItem)

        assert False

    spec_out = convert(spec)

    circuit = cast_circuit(circuit, dtype=torch.float64)
    r_circ = py_to_rust(circuit)
    assert isinstance(r_circ, Einsum)

    return r_circ, spec_out


NestAddArgs = Tuple[Add, NestAddsSpecSub]


# TODO: dedup with above
@st.composite
def st_nest_add_args(draw: st.DrawFn, must_be_explicitly_computable: bool = True) -> NestAddArgs:
    """TODO: currently does not generate any :class:`NamedItem` s or terminating node specs."""
    st_circuit = st_Circuit(
        st_np.array_shapes(min_dims=0),
        must_be_explicitly_computable=must_be_explicitly_computable,
        max_growth_steps=4,
        probs_default=CP.kw(all=1, Add=(70, dict(use_weights=False))),
        probs_per_depth=[
            CP.kw(all=0, Add=(1, dict(use_weights=False))),
            CP.kw(all=1, Add=(70, dict(use_weights=False))),
        ],
        possibly_different_name=True,
    )
    circuit = draw(st_circuit)
    from interp.circuit.computational_node import Add as cAdd

    assert isinstance(circuit, cAdd)

    circuit = cast_circuit(circuit, dtype=torch.float64)
    r_circ = py_to_rust(circuit)
    assert isinstance(r_circ, Add)

    # TODO: improve this testing as needed!
    spec: MulRearrangeSpecSub
    if (n := len(add_flatten(r_circ).children)) == 0:
        spec = ()
    else:
        idxs = draw(st.permutations(range(n)))
        spec, _ = draw(partitioned_or_rest(idxs, can_have_rest=True))

    def convert(spec: MulRearrangeSpec) -> NestAddsSpecSub:
        if isinstance(spec, int):
            return spec
        if isinstance(spec, tuple):
            return [convert(s) for s in spec]
        if isinstance(spec, MulRestType):
            return Rest()
        assert not isinstance(spec, NamedItem)

        assert False

    spec_out = convert(spec)

    return r_circ, spec_out


@hypothesis.given(nest_einsum_args=st_nest_einsum_args())
@mark_not_interesting_if(
    ValueError, message="einsum(): subscript in subscript list is not within the valid range [0, 52)"
)
def test_nest_einsums_hypothesis(nest_einsum_args: NestEinsumArgs):
    circuit, spec = nest_einsum_args
    nested = nest_einsums(circuit, spec)
    # circuit.print()
    # nested.print()
    assert_close(
        nested.evaluate(),
        circuit.evaluate(),
        rtol=1e-4,
        atol=1e-4,
    )


@hypothesis.given(nest_add_args=st_nest_add_args())
def test_nest_adds_hypothesis(nest_add_args: NestAddArgs):
    circuit, spec = nest_add_args
    nested = nest_adds(circuit, spec)
    # circuit.print()
    # nested.print()
    assert_close(
        nested.evaluate(),
        circuit.evaluate(),
        rtol=1e-4,
        atol=1e-4,
    )
