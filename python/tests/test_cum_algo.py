from datetime import timedelta
from uuid import UUID

import hypothesis
import hypothesis.extra.numpy as st_np
import pytest
import torch
from torch import tensor

from interp.circuit.computational_node import (
    Add,
    Concat,
    Einsum,
    GeneralFunction,
    Index,
    UnaryRearrange,
    softmax_fn,
    softmax_jacobian,
)
from interp.circuit.constant import ArrayConstant, FloatConstant, One, Zero
from interp.circuit.cum_algo import eps_attrib
from interp.circuit.cumulant import Cumulant
from interp.circuit.test_circuit import make_discrete_var
from interp.circuit.testing.topdown_circuit import CircuitProbs as CP
from interp.circuit.testing.topdown_circuit import st_Circuit
from interp.circuit.var import AutoTag, DiscreteVar
from rust_circuit import Circuit as rCircuit
from rust_circuit import Cumulant as rCumulant
from rust_circuit import DiscreteVar as rDiscreteVar
from rust_circuit import Einsum as rEinsum
from rust_circuit import GeneralFunction as rGeneralFunction
from rust_circuit import RunDiscreteVarAllSpec, Sampler
from rust_circuit import StoredCumulantVar as rStoredCumulantVar
from rust_circuit import rewrite_cum_to_circuit_of_cum
from rust_circuit.cum_algo import eps_attrib as r_eps_attrib
from rust_circuit.interop_rust import py_to_rust


@hypothesis.example(  # Two DiscreteVars
    cumulant=(
        lambda c0: Cumulant(
            circuits=(
                DiscreteVar(
                    values=Zero(shape=(1, 1), name=""),
                    probs_and_group=AutoTag(
                        node=c0, uuid=UUID("51b47e37-4766-41dd-9f2d-9f0a8c3742a4"), name="uniform_probs_and_group"
                    ),
                    name="",
                ),
                DiscreteVar(
                    values=Zero(shape=(1, 2), name=""),
                    probs_and_group=AutoTag(
                        node=c0, uuid=UUID("75ebd054-4fd9-4086-b215-2ac363445cf8"), name="uniform_probs_and_group"
                    ),
                    name="",
                ),
            ),
            name="",
            is_explicitly_computable=False,
            can_be_sampled=True,
        )
    )(FloatConstant(value=1.0, shape=(1,), name="uniform_probs")),
)
@hypothesis.example(  # 4 DiscreteVars
    cumulant=(
        lambda c0: Cumulant(
            circuits=tuple(
                DiscreteVar(
                    values=FloatConstant(i / 4, shape=(1, i + 4), name=""),
                    probs_and_group=c0,
                )
                for i in range(1, 5)
            ),
            name="",
            is_explicitly_computable=False,
            can_be_sampled=True,
        )
    )(FloatConstant(value=1.0, shape=(1,), name="uniform_probs")),
)
@hypothesis.example(
    cumulant=Cumulant(
        circuits=(
            Index(
                node=Einsum(
                    args=(
                        (One(shape=(2, 1), name=""), (0, 1)),
                        (
                            ArrayConstant(
                                value=tensor([1.5410]),
                                shape=(1,),
                                uuid=UUID("f728b4fa-4248-5e3a-0a5d-2f346baa9455"),
                                name="",
                            ),
                            (2,),
                        ),
                    ),
                    out_axes=(0, 1),
                    name="",
                    shape=(2, 1),
                    is_constant=True,
                    is_explicitly_computable=True,
                    can_be_sampled=True,
                ),
                index=(0, slice(0, 1, None)),
                name="",
                shape=(1,),
                is_constant=True,
                is_explicitly_computable=True,
                can_be_sampled=True,
            ),
            DiscreteVar(
                values=One(shape=(2,), name=""),
                probs_and_group=AutoTag(
                    node=FloatConstant(value=0.5, shape=(2,), name="uniform_probs"),
                    uuid=UUID("de5bfc0e-b11d-4c42-93bf-315db41f587a"),
                    name="uniform_probs_and_group",
                ),
                name="",
            ),
        ),
        name="",
        is_explicitly_computable=False,
        can_be_sampled=True,
    ),
)
@hypothesis.example(
    cumulant=(
        lambda c0: Cumulant(
            circuits=(
                c0,
                DiscreteVar(
                    values=Zero(shape=(2, 1), name=""),
                    probs_and_group=AutoTag(
                        node=FloatConstant(value=0.5, shape=(2,), name="uniform_probs"),
                        uuid=UUID("79afd6c2-c9a8-4f5b-a258-95ba1bc6a10f"),
                        name="uniform_probs_and_group",
                    ),
                    name="",
                ),
                c0,
                One(shape=(4,), name=""),
                Zero(shape=(1, 1), name=""),
                One(shape=(2,), name=""),
                Zero(shape=(5,), name=""),
            ),
            name="",
            is_explicitly_computable=False,
            can_be_sampled=True,
        )
    )(Zero(shape=(), name="A")),
)
@hypothesis.example(  # AutoTag
    cumulant=Cumulant(
        circuits=(
            AutoTag(
                node=DiscreteVar(
                    values=Zero(shape=(1,), name=""),
                    probs_and_group=AutoTag(
                        node=FloatConstant(value=1.0, shape=(1,), name="uniform_probs"),
                        uuid=UUID("cf4f2809-1c44-4947-ba57-f7bf6b4d9a26"),
                        name="uniform_probs_and_group",
                    ),
                    name="",
                ),
                uuid=UUID("e3e70682-c209-4cac-629f-6fbed82c07cd"),
                name="",
            ),
        ),
        name="",
        is_explicitly_computable=False,
        can_be_sampled=True,
    ),
)
@hypothesis.example(
    cumulant=Cumulant(
        circuits=(
            Einsum(
                args=(
                    (
                        DiscreteVar(
                            values=One(shape=(1, 2), name=""),
                            probs_and_group=GeneralFunction(
                                node=Zero(shape=(1,), name=""),
                                function=softmax_fn,
                                get_jacobian=softmax_jacobian,
                                name="_softmax",
                                allows_batching=True,
                                non_batch_dims=(-1,),
                                shape=(1,),
                                is_constant=True,
                                is_explicitly_computable=True,
                                can_be_sampled=True,
                            ),
                            name="",
                        ),
                        (0,),
                    ),
                ),
                out_axes=(),
                name="",
                shape=(),
                is_constant=False,
                is_explicitly_computable=False,
                can_be_sampled=True,
            ),
        ),
        name="",
        is_explicitly_computable=False,
        can_be_sampled=True,
    ),
)
@hypothesis.example(
    cumulant=Cumulant(
        circuits=(
            Einsum(
                args=((One(shape=(1,), name=""), (0,)),),
                out_axes=(0, 0),
                name="",
                shape=(1, 1),
                is_constant=True,
                is_explicitly_computable=True,
                can_be_sampled=True,
            ),
        ),
        name="",
        is_explicitly_computable=True,
        can_be_sampled=True,
    ),
)
@hypothesis.example(
    cumulant=Cumulant(
        circuits=(
            UnaryRearrange(
                node=Einsum(
                    args=(
                        (
                            UnaryRearrange(
                                node=Concat(
                                    circuits=(
                                        One(shape=(2, 1, 1, 1, 1), name=""),
                                        Index(
                                            node=DiscreteVar(
                                                values=Zero(shape=(3, 1, 1, 1, 1, 1), name=""),
                                                probs_and_group=AutoTag(
                                                    node=FloatConstant(
                                                        value=0.3333333333333333, shape=(3,), name="uniform_probs"
                                                    ),
                                                    uuid=UUID("6ab46340-445c-4633-8aff-e6634e336d57"),
                                                    name="uniform_probs_and_group",
                                                ),
                                                name="",
                                            ),
                                            index=(
                                                slice(0, 1, None),
                                                slice(0, 1, None),
                                                slice(0, 1, None),
                                                slice(0, 1, None),
                                                slice(0, 1, None),
                                            ),
                                            name="",
                                            shape=(1, 1, 1, 1, 1),
                                            is_constant=False,
                                            is_explicitly_computable=False,
                                            can_be_sampled=True,
                                        ),
                                    ),
                                    axis=0,
                                    name="",
                                    shape=(3, 1, 1, 1, 1),
                                    is_constant=False,
                                    is_explicitly_computable=False,
                                    can_be_sampled=True,
                                ),
                                op_string="s0 s1 s2 s3 s4 -> s0 s1 s2 s3 s4",
                                axes_lengths=(),
                                name="",
                                shape=(3, 1, 1, 1, 1),
                                is_constant=False,
                                is_explicitly_computable=False,
                                can_be_sampled=True,
                            ),
                            (0, 1, 2, 3, 4),
                        ),
                    ),
                    out_axes=(),
                    name="",
                    shape=(),
                    is_constant=False,
                    is_explicitly_computable=False,
                    can_be_sampled=True,
                ),
                op_string="->",
                axes_lengths=(),
                name="",
                shape=(),
                is_constant=False,
                is_explicitly_computable=False,
                can_be_sampled=True,
            ),
            One(shape=(1,), name=""),
        ),
        name="",
        is_explicitly_computable=False,
        can_be_sampled=True,
    ),
)
@hypothesis.example(
    cumulant=Cumulant(
        circuits=(
            Add(
                items={
                    Index(
                        node=One(shape=(), name=""),
                        index=(),
                        name="",
                        shape=(),
                        is_constant=True,
                        is_explicitly_computable=True,
                        can_be_sampled=True,
                    ): 1.0,
                    DiscreteVar(
                        values=One(shape=(1,), name=""),
                        probs_and_group=AutoTag(
                            node=FloatConstant(value=1.0, shape=(1,), name="uniform_probs"),
                            uuid=UUID("b95463a1-b440-4d3e-a7cf-5732b90a680a"),
                            name="uniform_probs_and_group",
                        ),
                        name="",
                    ): 0.0,
                },
                name="",
                shape=(),
                is_constant=False,
                is_explicitly_computable=False,
                can_be_sampled=True,
            ),
        ),
        name="",
        is_explicitly_computable=False,
        can_be_sampled=True,
    ),
)
@hypothesis.given(
    cumulant=st_Circuit(
        st_np.array_shapes(min_dims=0, max_dims=3),
        max_growth_steps=10,
        probs_per_depth=[
            CP.kw(all=0, Cumulant=1),
            # Only allow DiscreteVars as random variables, otherwise draw some branches. Do not draw GeneralFunctions.
            CP.kw(all=1, StoredCumulantVar=0, DiscreteVar=5, leaves=1, branches=5, randoms=0, GeneralFunction=0),
        ],
        # At depth >2, GeneralFunctions are now fair game
        probs_default=CP.kw(all=1, StoredCumulantVar=0, DiscreteVar=5, leaves=1, branches=5, randoms=0),
    )
)
@hypothesis.settings(deadline=timedelta(seconds=3))
@pytest.mark.skip(reason="TODO unskip; Bug found by fuzzing")
def test_rewrite_cum_to_circuit_of_cum(cumulant: Cumulant):
    raw_test_rewrite_cum_to_circuit_of_cum(py_to_rust(cumulant).cast_cumulant())


def raw_test_rewrite_cum_to_circuit_of_cum(cumulant: rCumulant):
    to_expand = cumulant.children[0]

    hypothesis.assume(
        not isinstance(to_expand, (rGeneralFunction, rDiscreteVar, rStoredCumulantVar)),
        "Do not try to expand things that cannot be expanded",
    )

    hypothesis.assume(
        not isinstance(to_expand, rEinsum) or len(to_expand.args) <= 3,
        "Expanding an Einsum with more than 3 children is too expensive",
    )

    any_constant_zero = any(c.is_constant for c in cumulant.children) and cumulant.num_children > 1
    hypothesis.assume(not any_constant_zero, "trivial if more than 1 inputs are constant")

    hypothesis.assume(
        cumulant.num_children <= 3 or to_expand.is_constant, "only cumulants of order <= 3 -- otherwise too expensive"
    )

    substituted_cumulant: rCircuit = rewrite_cum_to_circuit_of_cum(cumulant, to_expand, lambda _: "")
    assert not (isinstance(substituted_cumulant, rCumulant) and not isinstance(to_expand, rCumulant))

    sample_spec = RunDiscreteVarAllSpec.create_full_from_circuits(cumulant)
    try:
        sampled_cumulant = Sampler(sample_spec)(cumulant)
    except:
        hypothesis.assume(False, "circuit too weird to be sampled")
        # For example
        # 0 '' [] Cumulant
        #   1 '' [] Rearrange  ->
        #     2 '' [] DiscreteVar
        #     3 '' [1] Cumulant
        #         4 '' [1] Scalar 0
        #     5 '_softmax' [1] GeneralFunction softmax
        #         3

    sampled_substituted_cumulant = Sampler(sample_spec)(substituted_cumulant)

    atol = 1e-6
    rtol = 1e-5
    torch.testing.assert_allclose(
        sampled_cumulant.evaluate(), sampled_substituted_cumulant.evaluate(), atol=atol, rtol=rtol
    )


def compare_eps_attrib_values(c: Cumulant):
    for non_centered in [False, True]:
        to_expand = py_to_rust(eps_attrib(c, non_centered=non_centered))
        r_to_expand = r_eps_attrib(py_to_rust(c).cast_cumulant(), non_centered=non_centered)

        sample_spec = RunDiscreteVarAllSpec.create_full_from_circuits(to_expand)
        sampled = Sampler(sample_spec)(to_expand).evaluate()
        r_sample_spec = RunDiscreteVarAllSpec.create_full_from_circuits(r_to_expand)
        # TODO: make sampling work?
        r_sampled = Sampler(r_sample_spec)(r_to_expand.cast_module().substitute()).evaluate()

        atol = 1e-6
        rtol = 1e-5
        torch.testing.assert_allclose(sampled, r_sampled, atol=atol, rtol=rtol)


def test_eps_attrib_values():
    x, _, _ = make_discrete_var(values_shape=(2, 3), name="x", batch_size=17)
    y, _, _ = make_discrete_var(values_shape=(3, 2), name="y", batch_size=17, group=x.group)
    z, _, _ = make_discrete_var(values_shape=(4,), name="z", batch_size=17, group=x.group)

    vals = [
        Cumulant(()),
        Cumulant((x,)),
        Cumulant((x, z)),
        Cumulant((z, x)),
        Cumulant((y, z)),
        Cumulant((x, y)),
        Cumulant((x, y, z)),
    ]

    for val in vals:
        compare_eps_attrib_values(val)


if __name__ == "__main__":
    test_eps_attrib_values()
    # import rust_circuit as rc
    # s, *_ = rc.symbolic_sizes()
    # rc.Symbol.new_with_random_uuid((s,)).print()
