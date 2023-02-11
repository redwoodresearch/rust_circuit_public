import functools
import itertools
import math
import random
from copy import copy
from datetime import timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast
from uuid import UUID, uuid4

import hypothesis
import hypothesis.extra.numpy as st_np
import hypothesis.strategies as st
import pytest
import torch
from hypothesis import note
from torch import tensor
from torch.testing import assert_close

import rust_circuit as rc
import rust_circuit.optional as op
from interp.circuit.circuit import Circuit, MemoizedFn, Shape
from interp.circuit.circuit_compiler.util import RearrangeSpec as pRearrangeSpec
from interp.circuit.circuit_utils import cast_circuit, evaluate_fn
from interp.circuit.computational_node import Concat, Einsum, GeneralFunction, Index, UnaryRearrange, sigmoid_fn
from interp.circuit.constant import ArrayConstant, FloatConstant, One, Zero
from interp.circuit.testing import strategies as st_c
from interp.circuit.testing.topdown_circuit import CircuitProbs as CP
from interp.circuit.testing.topdown_circuit import st_Circuit
from interp.circuit.testing.utils import mark_not_interesting_if
from interp.tools.indexer import I
from rust_circuit import Add as rAdd
from rust_circuit import Array as rArray
from rust_circuit import Circuit as rCircuit
from rust_circuit import Concat as rConcat
from rust_circuit import ConstructArrayHasReservedSymbolicShapeError, DistributeNoopError
from rust_circuit import Einsum as rEinsum
from rust_circuit import GeneralFunction as rGeneralFunction
from rust_circuit import Index as rIndex
from rust_circuit import (
    OptimizationSettings,
    Parser,
    PrintOptions,
    PushDownIndexEinsumNoopError,
    PushDownIndexNoopOnConcatError,
    PushDownIndexNoopOnGeneralFunctionError,
    PushDownIndexRearrangeNotPossibleError,
    PushDownIndexScatterNoopError,
    PushDownIndexThroughIndexError,
    PushDownIndexUnimplementedTypeError,
)
from rust_circuit import Rearrange as rRearrange
from rust_circuit import RearrangeSpec as rRearrangeSpec
from rust_circuit import Scalar as rScalar
from rust_circuit import Scatter as rScatter
from rust_circuit import SchedulingOOMError
from rust_circuit import Symbol as rSymbol
from rust_circuit import (
    TorchDeviceDtypeOp,
    add_collapse_scalar_inputs,
    add_deduplicate,
    add_elim_removable_axes_weak,
    add_elim_zeros,
    add_flatten_once,
    add_fuse_scalar_multiples,
    add_make_broadcasts_explicit,
    add_outer_product_broadcasts_on_top,
    add_pull_diags,
    add_pull_removable_axes,
    add_pull_scatter,
    all_children,
    batch_to_concat,
)
from rust_circuit import cast_circuit as r_cast_circuit
from rust_circuit import (
    compiler_simp,
    compiler_simp_step,
    concat_drop_size_zero,
    concat_elim_identity,
    concat_elim_split,
    concat_fuse,
    concat_merge_uniform,
    concat_pull_removable_axes,
    concat_repeat_to_rearrange,
    concat_to_scatter,
)
from rust_circuit import count_nodes as r_count_nodes
from rust_circuit import (
    deep_canonicalize,
    deep_heuristic_nest_adds,
    deep_map,
    deep_pull_concat,
    deep_pull_concat_messy,
    deep_push_down_index_raw,
    distribute,
    distribute_all,
    einsum_concat_to_add,
    einsum_elim_identity,
    einsum_elim_removable_axes_weak,
    einsum_elim_zero,
    einsum_flatten_once,
    einsum_merge_scalars,
    einsum_nest_optimize,
    einsum_of_permute_merge,
    einsum_permute_to_rearrange,
    einsum_pull_removable_axes,
    einsum_pull_scatter,
    einsum_push_down_trace,
    extract_add,
    fuse_concat_modules,
    generalfunction_evaluate_simple,
    generalfunction_merge_inverses,
    generalfunction_pull_removable_axes,
    generalfunction_special_case_simplification,
    index_concat_drop_unreached,
    index_einsum_to_scatter,
    index_elim_identity,
    index_fuse,
    index_merge_scalar,
    index_split_axes,
    optimize_and_evaluate,
    optimize_and_evaluate_many,
    optimize_to_schedule,
    permute_of_einsum_merge,
    push_down_index_once,
    rearrange_elim_identity,
    rearrange_fuse,
    rearrange_merge_scalar,
    remove_add_few_input,
    scatter_elim_identity,
    scatter_pull_removable_axes,
    scheduled_evaluate,
)
from rust_circuit import simp as basic_simp
from rust_circuit import split_to_concat, strip_names_and_tags, substitute_all_modules, symbolic_sizes
from rust_circuit.interop_rust import py_to_rust, rust_to_py

P = Parser(tensors_as_random=True)

add_settings = dict(use_weights=False, max_n_children=5, promote_dups=True)


@st.composite
def get_c_st(
    draw: st.DrawFn,
    shape: st_c.ShapeOrStrategy = st_np.array_shapes(min_dims=0, max_dims=3),
    probs_per_depth: Union[Dict[int, CP], Sequence[CP]] = [],
    max_growth_steps: int = 5,
    probs_default: CP = CP.kw(
        all=0,
        branches=10,
        Cumulant=0,  # TODO: fix bugs and make this 1
        AutoTag=0,  # TODO: fix bugs and make this 1
        ArrayConstant=5,
        One=1,
        Zero=1,
        FloatConstant=4,
        Add=(10, add_settings),
        GeneralFunction=(4, dict(allow_arbitrary_fn=False)),
        Module=5,
        Symbol=5,
        Scatter=5,
    ),
    rust: bool = False,
    from_other: bool = True,  # TODO fix bugs in tests where this is False
    probs_fill_from_py: bool = True,
):
    x = draw(
        st_Circuit(
            shape,
            must_be_constant=True,
            must_be_explicitly_computable=True,
            growth_order="breadth_first",
            probs_per_depth=probs_per_depth,
            probs_default=probs_default,
            max_growth_steps=max_growth_steps,
            possibly_different_name=True,
            rust=draw(st.booleans()) if from_other else rust,
            probs_fill_from_py=probs_fill_from_py,
        )
    )
    if from_other:
        if rust and isinstance(x, Circuit):
            x = py_to_rust(x)
        if not rust and isinstance(x, rc.Circuit):
            x = rust_to_py(x)
    return x


def noneify_rearrange(x: rRearrange):
    """adds back in nones to rust rearrange for better testing"""
    spec = x.spec
    inps = spec.input_ints
    sizes = spec.int_sizes

    for ints in inps:
        if len(ints) == 0:
            continue
        if len(ints) == 1:
            sizes[ints[0]] = None
        elif all(sizes[i] is not None for i in ints):
            i = min(ints, key=lambda i: op.unwrap(sizes[i]))
            sizes[i] = None

    return rRearrange(x.node, rRearrangeSpec(spec.input_ints, spec.output_ints, sizes), name=x.name)


def deterministic_rearrange_noneing(x: rCircuit):
    return deep_map(x, lambda y: noneify_rearrange(y) if isinstance(y, rRearrange) else y)


@hypothesis.settings(max_examples=200, deadline=None)
@hypothesis.given(
    circ=get_c_st(from_other=False),
)
def test_rust_and_back(circ: Circuit):
    circ = cast_circuit(circ, dtype=torch.float64)
    rust_circ = deterministic_rearrange_noneing(py_to_rust(circ))
    back_py = rust_circ.to_py()
    assert_close(
        MemoizedFn(evaluate_fn(dtype=torch.float64))(circ), MemoizedFn(evaluate_fn(dtype=torch.float64))(back_py)
    )

    rust_circ = py_to_rust(circ)
    back_py = rust_circ.to_py()
    assert_close(
        MemoizedFn(evaluate_fn(dtype=torch.float64))(circ), MemoizedFn(evaluate_fn(dtype=torch.float64))(back_py)
    )


@hypothesis.settings(
    phases=(
        hypothesis.Phase.explicit,
        hypothesis.Phase.reuse,
        hypothesis.Phase.generate,
    ),
    deadline=None,
)
@hypothesis.given(circ=get_c_st(from_other=False))
def test_rust_evaluate(circ: Circuit):
    circ = cast_circuit(circ, dtype=torch.float64)
    rust_circ = deep_canonicalize(py_to_rust(circ))
    assert_close(
        MemoizedFn(evaluate_fn(dtype=torch.float64))(circ),
        rust_circ.evaluate(),
    )

    rust_circ = deterministic_rearrange_noneing(py_to_rust(circ))
    assert_close(
        MemoizedFn(evaluate_fn(dtype=torch.float64))(circ),
        rust_circ.evaluate(),
    )


@st.composite
def batch_to_concat_args_st(draw):
    shape = draw(st_np.array_shapes(min_dims=1))
    c = draw(get_c_st(shape=shape, rust=True))
    dim = draw(st.integers(0, len(shape) - 1))
    bs = draw(st.integers(1, shape[dim]))
    return (c, dim, bs)


@pytest.mark.xfail
@hypothesis.given(batch_to_concat_args_st())
@mark_not_interesting_if(
    RuntimeError, message="batch_to_concat/get_axis_leaves: modules not fully supported, please substitute first"
)
def test_evaluate_batch(args):
    circ, dim, bs = args
    circ1 = batch_to_concat(circ, dim, bs)
    assert_close(circ.evaluate(), circ1.evaluate())


@pytest.mark.xfail
@hypothesis.settings(deadline=None)
@hypothesis.given(batch_to_concat_args_st())
@mark_not_interesting_if(
    RuntimeError, message="batch_to_concat/get_axis_leaves: modules not fully supported, please substitute first"
)
def test_optimize_batch(args):
    circ, dim, bs = args
    circ = r_cast_circuit(circ, TorchDeviceDtypeOp(dtype="float64"))
    circ1 = batch_to_concat(circ, dim, bs)
    print(circ.dtype, circ1.dtype)
    circ.print()
    circ1.print()
    assert_close(circ.evaluate().to(dtype=torch.float64), optimize_and_evaluate(circ1).to(dtype=torch.float64))


def raw_test_compily_py(pycirc):
    circ = py_to_rust(pycirc, TorchDeviceDtypeOp(dtype="float64", device="cpu"))
    assert_close(
        MemoizedFn(evaluate_fn(dtype=torch.float64))(pycirc),
        optimize_and_evaluate(circ),
    )


def extract_add_fixed_seed_subset(c: rAdd):
    random.seed(bytes(c.hash))
    nodes = copy(c.children)
    random.shuffle(nodes)
    subset = nodes[: random.randint(0, len(nodes) - 1)]
    return extract_add(c, rAdd(*subset))


class PartialName(functools.partial):
    def __new__(cls, func, /, *args, name: Optional[str] = None, **keywords):
        out = functools.partial.__new__(cls, func, *args, **keywords)
        new_name: str = op.unwrap_or(name, func.__name__)
        out.__name__ = new_name  # type: ignore
        return out


def get_rewrite_st(setup):
    options = setup[2] if len(setup) > 2 else {}
    probs_normalized = [[x] if not isinstance(x, list) else x for x in setup[1]]
    shape = st.sampled_from(options["shapes"]) if "shapes" in options else st_np.array_shapes(min_dims=1, max_dims=3)
    circuit_type_settings = {"Add": (1, add_settings)} | options.get("circuit_type_settings", {})
    probs_per_depth = [CP.kw(all=0, **{k: circuit_type_settings.get(k, 1) for k in x}) for x in probs_normalized]
    return get_c_st(shape, probs_per_depth, max_growth_steps=20, rust=True, from_other=False)


def deep_pull_concat_strip(c: rCircuit) -> rCircuit:
    return deep_pull_concat(strip_names_and_tags(c))


def add_deduplicate_strip(c: rCircuit):
    return add_deduplicate(cast(rAdd, strip_names_and_tags(c)))


def push_down_index_once_wrap(x: rIndex):
    try:
        return push_down_index_once(x, allow_partial_pushdown=True)
    except (
        PushDownIndexNoopOnConcatError,
        PushDownIndexNoopOnGeneralFunctionError,
        PushDownIndexEinsumNoopError,
        PushDownIndexScatterNoopError,
        PushDownIndexRearrangeNotPossibleError,
    ):
        return None


rewrite_test_setups = [
    (einsum_permute_to_rearrange, ["Einsum"], {"shapes": [(4, 4, 4)]}),
    (basic_simp, [], {"assert_different": False}),
    (rc.canonicalize_node, [], {"assert_different": False}),
    (
        einsum_elim_removable_axes_weak,
        ["Einsum", ["ArrayConstant", "FloatConstant", "UnaryRearrange"]],
        {"shapes": [(4, 4, 4)]},
    ),
    (
        add_elim_removable_axes_weak,
        ["Add", ["ArrayConstant", "FloatConstant", "UnaryRearrange"]],
        {"shapes": [(4, 4, 4)]},
    ),
    (generalfunction_evaluate_simple, ["GeneralFunction", "FloatConstant"], {"shapes": [(), (2,)]}),
    (generalfunction_special_case_simplification, ["GeneralFunction", "FloatConstant"], {"shapes": [(10,), (10, 10)]}),
    # (rc.generalfunction_gen_index_const_to_index, ["GeneralFunction"]), # TODO enable once we generate gen_index
    (
        concat_to_scatter,
        ["Concat", ["Zero", "ArrayConstant", "Scalar"]],
    ),
    (
        concat_repeat_to_rearrange,
        ["Concat", "UnaryRearrange", "ArrayConstant"],
        {"shapes": [(4, 4, 4), (1, 4, 4), (2, 2, 2, 3)]},
    ),
    (add_pull_diags, ["Add", "Einsum"], {"shapes": [(4, 4, 4), (1, 4, 4), (2, 2, 2, 3)]}),
    (einsum_push_down_trace, ["Einsum"]),
    (einsum_concat_to_add, ["Einsum"]),
    (rc.pull_concat_once, [["Add", "Einsum", "GeneralFunction"], "Concat"]),
    (rc.pull_concat_once_raw, [["Add", "Einsum", "GeneralFunction"], "Concat"]),
    (index_concat_drop_unreached, ["Index", "Concat"]),
    (concat_fuse, ["Concat", "Concat"]),
    (deep_heuristic_nest_adds, ["Add", "Add"], {"assert_different": False}),
    # # TODO fails with hypothesis.Unsatisfiable
    # (
    #     PartialName(deep_maybe_distribute, settings=OptimizationSettings()),
    #     ["Einsum", "Add"],
    #     {"assert_different": False},
    # ),
    (
        einsum_nest_optimize,
        [
            "Einsum",
        ],
        {"assert_different": False},
    ),
    (
        PartialName(add_pull_removable_axes, remove_non_common_axes=True, name="add_pull_removable_axes_remove_common"),
        ["Add", ["UnaryRearrange", "FloatConstant", "Zero", "Concat"]],
    ),
    (
        PartialName(
            add_pull_removable_axes, remove_non_common_axes=False, name="add_pull_removable_axes_no_remove_common"
        ),
        ["Add", ["UnaryRearrange", "FloatConstant", "Zero"]],
    ),
    (add_deduplicate_strip, ["Add"], {"shapes": [()]}),
    (add_flatten_once, ["Add", "Add"]),
    (add_collapse_scalar_inputs, ["Add", "FloatConstant"]),
    (concat_elim_identity, ["Concat"]),
    (extract_add_fixed_seed_subset, ["Add"]),
    (concat_merge_uniform, ["Concat", ["One", "FloatConstant"]]),
    (einsum_elim_identity, ["Einsum"]),
    (einsum_flatten_once, ["Einsum", "Einsum"]),
    (
        index_elim_identity,
        [
            "Index",
        ],
        {"shapes": [(), (2,), (1, 2)]},
    ),  # really should allow any shape, but it was never generating identities
    (index_merge_scalar, ["Index", "FloatConstant"]),
    (rearrange_elim_identity, ["UnaryRearrange"]),
    (rearrange_merge_scalar, ["UnaryRearrange", "FloatConstant"]),
    (remove_add_few_input, ["Add"]),
    (generalfunction_pull_removable_axes, ["GeneralFunction", ["FloatConstant", "UnaryRearrange"]]),
    (concat_pull_removable_axes, ["Concat", ["FloatConstant", "UnaryRearrange"]]),
    (einsum_pull_removable_axes, ["Einsum", ["FloatConstant", "UnaryRearrange"]]),
    (add_make_broadcasts_explicit, ["Add"]),
    (distribute_all, ["Einsum", "Add"]),
    (PartialName(distribute, operand_idx=0, do_broadcasts=True), ["Einsum", "Add"]),
    (einsum_of_permute_merge, ["Einsum", "UnaryRearrange"]),  # i really don't know why this needs f64
    (permute_of_einsum_merge, ["UnaryRearrange", "Einsum"]),
    (
        einsum_elim_zero,
        ["Einsum", ["UnaryRearrange", "Scalar"]],
        {"from_rust": False},
        {"circuit_type_settings": dict(Scalar=(1, dict(values=[0.0])))},
    ),
    (
        PartialName(index_split_axes, top_axes={0}),
        [
            "Index",
        ],
    ),
    (
        add_elim_zeros,
        ["Add", ["Scalar"]],
        {"assert_different": False, "shapes": [()], "circuit_type_settings": dict(Scalar=(1, dict(values=[0.0, 1.0])))},
    ),
    (rearrange_fuse, ["UnaryRearrange", "UnaryRearrange"]),
    (deep_pull_concat_strip, [["Add", "Einsum", "GeneralFunction"], "Concat"], {"assert_different": False}),
    (rc.deep_pull_concat_new, [["Add", "Einsum", "GeneralFunction"], "Concat"], {"assert_different": False}),
    (deep_pull_concat_messy, [["Add", "Einsum", "GeneralFunction"], "Concat"], {"assert_different": False}),
    (deep_canonicalize, [], {"assert_different": False}),
    (compiler_simp, [], {"assert_different": False}),
    (deep_push_down_index_raw, ["Index"], {"assert_different": False}),
    (compiler_simp_step, [], {"assert_different": False}),
    (index_fuse, ["Index", "Index"]),
    # push down index for each child type individually bc code different and need to exercise all
    (push_down_index_once_wrap, ["Index", ["UnaryRearrange"]], dict(name="push_down_index_rearrange")),
    (push_down_index_once_wrap, ["Index", ["Concat"]], dict(name="push_down_index_concat")),
    (push_down_index_once_wrap, ["Index", ["GeneralFunction"]], dict(name="push_down_index_func")),
    (push_down_index_once_wrap, ["Index", ["Einsum"]], dict(name="push_down_index_einsum")),
    (push_down_index_once_wrap, ["Index", ["Add"]], dict(name="push_down_index_add")),
    (push_down_index_once_wrap, ["Index"], dict(name="push_down_index_general")),
    (
        rc.elim_empty_module,
        ["Module"],
        {"circuit_type_settings": dict(Module=(1, dict(empty_module=True, allow_rearrange=False)))},
    ),
    (
        rc.elim_no_input_module,
        ["Module"],
        {"circuit_type_settings": dict(Module=(1, dict(empty_module=True, allow_rearrange=False)))},
    ),
    (
        rc.module_remove_unused_inputs,
        ["Module"],
        {"circuit_type_settings": dict(Module=(1, dict(empty_module=True, allow_rearrange=False, unused_inputs=True)))},
    ),
    (
        rc.deep_module_remove_unused_inputs,
        [],
        {"assert_different": False, "circuit_type_settings": dict(Module=(1, dict(unused_inputs=True)))},
    ),
]


def set_func_name(name):
    def dec(func):
        func.__name__ = name
        return func

    return dec


# Define all the `test_<rewrite_name>`
def make_test_rewrite(setup):
    if len(setup) < 3:
        setup += ({},)

    name = setup[2].get("name", setup[0].__name__)

    @hypothesis.settings(
        suppress_health_check=[hypothesis.HealthCheck.filter_too_much, hypothesis.HealthCheck.too_slow],
        deadline=timedelta(seconds=1),
        phases=(
            hypothesis.Phase.explicit,
            hypothesis.Phase.reuse,
            hypothesis.Phase.generate,
            hypothesis.Phase.shrink
            # target and shrink often dont halt
        ),
    )
    @hypothesis.given(circ=get_rewrite_st(setup), do_noneing=st.booleans())
    @mark_not_interesting_if(
        ValueError, message="einsum(): subscript in subscript list is not within the valid range [0, 52)"
    )
    @set_func_name(f"test_{name}")
    def test_rewrite(circ, do_noneing: bool):
        raw_test_rewrite(
            circ,
            setup[0],
            torch.float64,
            assert_different=setup[2].get("assert_different", True),
            do_print=False,
            do_noneing=do_noneing,
        )

    return test_rewrite


for setup in rewrite_test_setups:
    f = make_test_rewrite(setup)
    # print(setup, f.__name__)
    assert f.__name__ not in locals()
    locals()[f.__name__] = f


@hypothesis.given(
    circ=get_c_st(probs_per_depth=[CP.kw(all=0, Einsum=1), CP.kw(all=0, FloatConstant=1, UnaryRearrange=1)], rust=True),
    children=st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=5), st_Circuit(shape=(), probs_default=CP.kw(all=0, Scalar=1), rust=True)
        ),
        min_size=1,  # the rewrite returns None if only one child, but we want to test that case (and assume(False) it) sometimes
        max_size=6,
    ),
)
def test_einsum_merge_scalars(circ: rc.Einsum, children: List[Tuple[int, rc.Scalar]]):
    # Generate a bunch of scalars and insert them into the einsum
    args = circ.args
    for idx, child in children:
        args.insert(idx, (child, ()))

    new_circ = rc.Einsum(*args, out_axes=circ.out_axes, name=circ.name)
    raw_test_rewrite(new_circ, einsum_merge_scalars, dtype=torch.float64, assert_different=True, do_print=False)


def raw_test_rewrite(
    circ,
    rewrite,
    dtype=torch.float64,
    assert_different=True,
    do_print=False,
    trans: Optional[Circuit] = None,
    do_noneing: bool = True,
):
    note(circ.repr())
    # Don't canonicalize or strip names on purpose, we want circ to accurately specify what gets transformed.
    ddt = rc.TorchDeviceDtypeOp(dtype=dtype)
    circ = rc.cast_circuit(circ, ddt)
    if do_noneing:
        circ = deterministic_rearrange_noneing(circ)
    if do_print:
        PrintOptions(tensor_index_literal=True, bijection=False).print(circ)
    try:
        rewritten = rewrite(circ)
    except (
        DistributeNoopError,
        PushDownIndexNoopOnConcatError,
        PushDownIndexNoopOnGeneralFunctionError,
        PushDownIndexEinsumNoopError,
        PushDownIndexScatterNoopError,
        PushDownIndexRearrangeNotPossibleError,
        PushDownIndexUnimplementedTypeError,
        PushDownIndexThroughIndexError,
    ):
        hypothesis.assume(False, "Rewrite raised noop error")
        assert False

    if trans is not None:
        assert rewritten is not None, "expected a transformation"
    else:
        hypothesis.assume(rewritten is not None, "Rewrite returned None (noop)")
    note(rewritten.repr())
    if do_print:
        PrintOptions(tensor_index_literal=True, bijection=False).print(rewritten)
    if assert_different:
        assert rewritten != circ
    else:
        hypothesis.assume(rewritten != circ, "transform was trivial")
    if trans is not None:
        if rewritten != trans:
            raise ValueError(f"transformation is wrong, {rewritten}!={trans}")
    if dtype is not None:
        lhs = rc.cast_circuit(circ, ddt).evaluate()
        hypothesis.assume(lhs.isfinite().all())
        assert_close(lhs, rc.cast_circuit(rewritten, ddt).evaluate())
    else:
        lhs = circ.evaluate()
        hypothesis.assume(lhs.isfinite().all())
        assert_close(lhs, rewritten.evaluate())


# These are rewrites that we expect _not_ to fire. It is a place for tests of the form "this rewrite might accidentally trigger on this input
# and crash. We don't want it to trigger on this input at all. Make sure it doesn't."
def test_noop_rewrites():
    setups = [
        (
            P(
                """
0 Einsum ab,ab->
  1 Concat 0
    2 [1,3] Scalar 1
    3 [1,3] Scalar 2
  4 [2,3] Scalar 3"""
            ),
            einsum_concat_to_add,
        ),
        (
            P(
                """
0 Einsum aa,cd->
  1 Concat 0
    2 [2,4] Scalar 1
    3 [2,4] Scalar 2
  4 [7,9] Scalar 3"""
            ),
            einsum_concat_to_add,
        ),
    ]
    for circ, rewrite in setups:
        circ = strip_names_and_tags(r_cast_circuit(circ, TorchDeviceDtypeOp("cpu", "float64")))
        circ.print()
        rewritten = cast(Any, rewrite)(circ)
        assert rewritten is None


# python doesn't have scatter, so only way to test scatter is rewrite
def test_rewrite_all_rust():
    setups = [
        (
            P(
                """
0 'pgqlrflg' [4, 4] Add 
  1 '' [4, 4] Rearrange () a -> 4 (a 2)
    2 'rkysaujw' [1, 2] Scalar -10.842327468923694
  3 '' [4] Scalar 1
  4 'qbmvbyta' [4, 1] Rearrange a -> a ()
    5 '' [4] Scalar 1
  6 '' [] Einsum ->"""
            ),
            lambda x: substitute_all_modules(batch_to_concat(x, 0, 2)),
        ),
        (
            P(
                """
0 'pgqlrflg' [4, 4] Add 
  1 '' [4, 4] Rearrange () a -> 4 (a 2)
    2 'rkysaujw' [1, 2] Scalar -10.842327468923694
  3 '' [4] Scalar 1
  4 'qbmvbyta' [4, 1] Rearrange a -> a ()
    5 '' [4] Scalar 1
  6 '' [] Einsum ->"""
            ),
            lambda x: substitute_all_modules(batch_to_concat(x, 0, 2)),
        ),
        (
            P(
                """
0 'adder' Add
  1 'e' Einsum a,->a
    2 's' [2] Scalar 1.23
    3 'f2' [] Scalar 2.0
  4 'e' Einsum a,->a
    5 's' [2] Scalar 1.23
    6 'f4' [] Scalar 4.0
  7 'fother1' [] Scalar 5.0
  8 'fother2' [] Scalar 7.0
  9 'e2' Einsum a,->a
    10 's234' [2] Scalar 2.34
    11 'f1' [] Scalar 1.0"""
            ),
            add_fuse_scalar_multiples,
        ),
        (
            P(
                """
0 'pgqlrflg' [4, 4] Add
  1 '' [] Einsum ->
  2 '' [4] Array
  3 '' [4, 4] Rearrange a:1 b:2 -> (c:2 d:2 a:1) (b:2 e:2 f:1 g:1)
    4 '' [1, 2] Scalar -10.8
  5 'qbmvbyta' [4, 1] Rearrange a:4 -> a:4 b:1
    6 '' [4] Array"""
            ),
            lambda x: substitute_all_modules(batch_to_concat(x, 0, 2)),
        ),
        (
            py_to_rust(
                (
                    lambda c0: Index(
                        node=Concat(
                            circuits=(
                                Concat(
                                    circuits=(c0, c0),
                                    axis=1,
                                    name="tutmyila",
                                    shape=(1, 2, 5, 6),
                                    is_constant=True,
                                    is_explicitly_computable=True,
                                    can_be_sampled=True,
                                ),
                                Concat(
                                    circuits=(
                                        UnaryRearrange(
                                            node=FloatConstant(value=-6.103515625e-05, shape=(), name=""),
                                            op_string=" -> a b c d",
                                            axes_lengths=(("a", 1), ("b", 4), ("c", 5), ("d", 6)),
                                            name="",
                                            shape=(1, 4, 5, 6),
                                            is_constant=True,
                                            is_explicitly_computable=True,
                                            can_be_sampled=True,
                                        ),
                                    ),
                                    axis=3,
                                    name="",
                                    shape=(1, 4, 5, 6),
                                    is_constant=True,
                                    is_explicitly_computable=True,
                                    can_be_sampled=True,
                                ),
                            ),
                            axis=1,
                            name="nlkjqnym",
                            shape=(1, 6, 5, 6),
                            is_constant=True,
                            is_explicitly_computable=True,
                            can_be_sampled=True,
                        ),
                        index=(tensor([0, 0, 0, 0]), 0, slice(None, -3, None), tensor([1, 3, 1, 1, 1])),
                        name="ynffhidn",
                        hash_tensor_idx_by_value=True,
                        shape=(4, 2, 5),
                        is_constant=True,
                        is_explicitly_computable=True,
                        can_be_sampled=True,
                    )
                )(
                    UnaryRearrange(
                        node=Concat(
                            circuits=(
                                ArrayConstant(
                                    value=tensor([[-18.1877, 19.6382], [-8.6186, -12.9271], [-5.1599, 11.6911]]),
                                    shape=(3, 2),
                                    uuid=UUID("187eab5c-923e-9788-a368-5f429c09df0b"),
                                    name="",
                                ),
                            ),
                            axis=0,
                            name="dtnirxgf",
                            shape=(3, 2),
                            is_constant=True,
                            is_explicitly_computable=True,
                            can_be_sampled=True,
                        ),
                        op_string="(1 a 1 b 1) (c d 1 e) -> (f g h) (c i) j (d b a e)",
                        axes_lengths=(
                            ("a", 1),
                            ("b", 3),
                            ("c", 1),
                            ("d", 2),
                            ("e", 1),
                            ("f", 1),
                            ("g", 1),
                            ("h", 1),
                            ("i", 1),
                            ("j", 5),
                        ),
                        name="",
                        shape=(1, 1, 5, 6),
                        is_constant=True,
                        is_explicitly_computable=True,
                        can_be_sampled=True,
                    )
                )
            ),
            push_down_index_once,
        ),
        (
            P(
                """
10 Index [1,:]
  0 Concat 0
    1 [2,3] Scalar 2.2
    2 [5,3] Scalar 1.0"""
            ),
            push_down_index_once,
        ),
        (
            P(
                """
10 Index [1:5,1:]
  0 Concat 0
    1 [2,3] Scalar 2.2
    2 [5,3] Scalar 1.0"""
            ),
            push_down_index_once,
        ),
        (
            P(
                """
10 Index [3:4,1:]
  0 Concat 0
    1 [2,3] Scalar 2.2
    2 [5,3] Scalar 1.0"""
            ),
            push_down_index_once,
        ),
        (
            rGeneralFunction.new_by_name(
                rGeneralFunction.new_by_name(rScalar(2.2, ()), spec_name="reciprocal"), spec_name="reciprocal"
            ),
            generalfunction_merge_inverses,
        ),
        (
            rGeneralFunction.new_by_name(rScalar(2.2, ()), spec_name="sigmoid"),
            generalfunction_evaluate_simple,
        ),
        (
            rGeneralFunction.new_by_name(rScalar(2.2, (10, 10)), spec_name="softmax"),
            generalfunction_special_case_simplification,
        ),
        (
            rGeneralFunction.new_by_name(rScalar(2.2, (10, 10)), spec_name="log_softmax"),
            generalfunction_special_case_simplification,
        ),
        (
            rGeneralFunction.new_by_name(rScalar(2.2, (10, 10)), spec_name="last_dim_size"),
            generalfunction_special_case_simplification,
        ),
        (
            rGeneralFunction.new_by_name(rScalar(2.2, (10, 2)), spec_name="last_dim_size"),
            generalfunction_special_case_simplification,
        ),
        (
            rGeneralFunction.new_by_name(rScalar(4.2, (3, 10, 2)), spec_name="last_dim_size"),
            generalfunction_special_case_simplification,
        ),
        (
            rGeneralFunction.new_by_name(rScalar(2.8, (2,)), spec_name="last_dim_size"),
            generalfunction_special_case_simplification,
        ),
        (
            P(
                """'toiknecd' Add
  0 '' Concat 0
    'gstibtqe' [4,1] Scalar 0
    1 '' [1,1] Array
  2 '' Concat 0
    3 '' [5,1] Scalar 0
  4 '' Concat 2
    'wibqgydb' Concat 0
      5 '' [1,5,3] Array
      6 '' [2,5,3] Scalar 0
    7 '' Concat 1
      8 '' [3,2,3] Scalar 0
      9 '' Rearrange  -> 3 3 3
        10 '' [] Scalar 0"""
            ),
            deep_pull_concat,
        ),
        (
            py_to_rust(
                Index(
                    node=Index(
                        node=GeneralFunction(
                            node=Index(
                                node=Einsum(
                                    args=(
                                        (Zero(shape=(2, 5), name=""), (0, 1)),
                                        (
                                            Index(
                                                node=Zero(shape=(1, 1, 1), name=""),
                                                index=(
                                                    tensor([0, 0, 0, 0, 0, 0]),
                                                    tensor([0, 0, 0, 0, 0]),
                                                    tensor([0, 0, 0, 0]),
                                                ),
                                                name="",
                                                hash_tensor_idx_by_value=True,
                                                shape=(6, 5, 4),
                                                is_constant=True,
                                                is_explicitly_computable=True,
                                                can_be_sampled=True,
                                            ),
                                            (2, 3, 4),
                                        ),
                                    ),
                                    out_axes=(2, 3, 0, 4, 1, 3),
                                    name="",
                                    shape=(6, 5, 2, 4, 5, 5),
                                    is_constant=True,
                                    is_explicitly_computable=True,
                                    can_be_sampled=True,
                                ),
                                index=(
                                    slice(0, 1, None),
                                    tensor([0, 0, 2]),
                                    tensor([1, 0, 0]),
                                    slice(-4, None, None),
                                    slice(0, 1, None),
                                    slice(1, -3, None),
                                ),
                                name="",
                                hash_tensor_idx_by_value=True,
                                shape=(1, 3, 3, 4, 1, 1),
                                is_constant=True,
                                is_explicitly_computable=True,
                                can_be_sampled=True,
                            ),
                            function=sigmoid_fn,
                            get_jacobian=None,
                            name="",
                            allows_batching=True,
                            non_batch_dims=(),
                            shape=(1, 3, 3, 4, 1, 1),
                            is_constant=True,
                            is_explicitly_computable=True,
                            can_be_sampled=True,
                        ),
                        index=(
                            -1,
                            0,
                            tensor([0, 1, 2, 0]),
                            slice(-4, 1, None),
                            tensor([0, 0]),
                            tensor([0, 0, 0, 0]),
                        ),
                        name="",
                        hash_tensor_idx_by_value=True,
                        shape=(4, 1, 2, 4),
                        is_constant=True,
                        is_explicitly_computable=True,
                        can_be_sampled=True,
                    ),
                    index=(3, tensor([0, 0]), tensor([1, 1, 1, 0, 0, 0]), 0),
                    name="",
                    hash_tensor_idx_by_value=True,
                    shape=(2, 6),
                    is_constant=True,
                    is_explicitly_computable=True,
                    can_be_sampled=True,
                )
            ),
            deep_push_down_index_raw,
        ),
        (
            py_to_rust(
                Index(
                    node=Index(
                        node=Zero(shape=(1, 1, 1), name=""),
                        index=(slice(-1, None, None), tensor([0, 0]), tensor([0, 0])),
                        name="",
                        hash_tensor_idx_by_value=True,
                        shape=(1, 2, 2),
                        is_constant=True,
                        is_explicitly_computable=True,
                        can_be_sampled=True,
                    ),
                    index=(slice(-1, 1, None), tensor([0]), tensor([1, 1, 0])),
                    name="",
                    hash_tensor_idx_by_value=True,
                    shape=(1, 1, 3),
                    is_constant=True,
                    is_explicitly_computable=True,
                    can_be_sampled=True,
                )
            ),
            index_fuse,
        ),
        (
            rAdd(rScalar(1.0, (10, 10))),
            functools.partial(add_pull_removable_axes, remove_non_common_axes=True),
        ),
        (
            rAdd(rScalar(1.0, (10, 10))),
            functools.partial(add_pull_removable_axes, remove_non_common_axes=False),
        ),
        (rAdd(rScalar(1.0, (10,))), functools.partial(add_pull_removable_axes, remove_non_common_axes=False)),
        (rAdd(rArray.randn(3), rArray.randn(3, 1)), add_outer_product_broadcasts_on_top),
        (
            rAdd(rArray.randn(3, 3), rArray.randn(3), rArray.randn(3, 3, 1)),
            add_outer_product_broadcasts_on_top,
        ),
        ((lambda ac: rConcat(ac, ac, axis=0))(rArray.randn(10, 10)), concat_repeat_to_rearrange),
        (
            P(
                """
'Add' Add
  0 'Ein' Einsum ab->abaa
    'zero' [3,3] Scalar 0
  1 'Ein' Einsum abc->abca
    'one' [1,3,1] Scalar 1"""
            ),
            lambda x: compiler_simp(deep_canonicalize(x)),
        ),
        (
            P(
                """
'GeneralFunction' GeneralFunction sigmoid
  0 'Concat' Concat 0
    'Add' Add
      1 'zero' [2,2] Scalar 0
      'ScalarMul' Einsum a,->a
        2 'Concat' Concat 0
          3 'zero' [1] Scalar 0
        'unnamed' [] Scalar -2.941891537492132"""
            ),
            deep_pull_concat_messy,
        ),
        (
            rEinsum((rIndex(rArray.randn(3, 3, 3, 3, 3), I[0, :, :1, :, :]), (0, 1, 0, 0)), out_axes=()),
            einsum_push_down_trace,
        ),
        (
            (
                rEinsum.from_einsum_string(
                    "ab,c -> bc",
                    rConcat(rArray.randn(7, 10), rArray.randn(9, 10), axis=0),
                    rArray.randn(19),
                ),
                einsum_concat_to_add,
            )
        ),
        (rAdd(rEinsum((rScalar(1.0, (2,)), (1,)), out_axes=(1, 1))), add_pull_diags),
        (rEinsum((rConcat(rScalar(1.0, (2, 2, 2)), axis=0), (0, 0, 0)), out_axes=(0,)), einsum_push_down_trace),
        (rEinsum((rConcat(rScalar(1.0, (5, 2, 2)), axis=0), (1, 0, 0)), out_axes=(1,)), einsum_push_down_trace),
        (rEinsum((rAdd(rScalar(1.0, (5, 2, 2))), (1, 0, 0)), out_axes=(1,)), einsum_push_down_trace),
        (
            P(
                """
0 Einsum aaa->
  1 Rearrange a:10 b:10 -> a:10 b:10 10
    2 [10,10] Array rand"""
            ),
            einsum_push_down_trace,
        ),
        (
            py_to_rust(
                Einsum(
                    args=(
                        (
                            Index(
                                node=FloatConstant(
                                    value=-3.1217301188127626,
                                    shape=(4, 3, 6, 1, 3),
                                ),
                                index=(
                                    slice(1, 2, None),
                                    slice(2, 3, None),
                                    slice(2, 6, None),
                                    torch.randint(0, 1, torch.Size([5])),
                                    torch.randint(0, 3, torch.Size([4])),
                                ),
                                hash_tensor_idx_by_value=True,
                                shape=(1, 1, 4, 5, 4),
                            ),
                            (0, 1, 2, 3, 4),
                        ),
                        (
                            One(
                                shape=(4, 6, 4, 4),
                            ),
                            (5, 6, 7, 7),
                        ),
                        (
                            UnaryRearrange(
                                node=ArrayConstant(
                                    value=torch.randn(torch.Size([6, 2, 6])),
                                    shape=(6, 2, 6),
                                    uuid=UUID("f4153da5-9a41-39e6-cd0b-ce1bfa70a49e"),
                                ),
                                op_string="(d 1) c a -> a b c d",
                                axes_lengths=(("a", 6), ("b", 5), ("c", 2), ("d", 6)),
                                shape=(6, 5, 2, 6),
                            ),
                            (6, 8, 9, 6),
                        ),
                    ),
                    out_axes=(4, 6, 7),
                    shape=(4, 6, 4),
                )
            ),
            einsum_push_down_trace,
        ),
        (
            rAdd(
                rConcat(rArray.randn(10), rArray.randn(10), axis=0),
                rConcat(rArray.randn(10), rArray.randn(10), axis=0),
            ),
            deep_pull_concat_messy,
        ),
        (
            rAdd(
                rConcat(rArray.randn(10), rArray.randn(10), axis=0),
                rConcat(rArray.randn(10), rArray.randn(10), axis=0),
            ),
            deep_pull_concat,
        ),
        (rConcat(rArray.randn(10, 10), rArray.randn(0, 10), axis=0), concat_drop_size_zero),
        (rScatter(rArray.randn(10, 10), I[0:10, 0:10], (10, 10)), scatter_elim_identity),
        (
            rAdd(
                rConcat(rArray.randn(10), rArray.randn(10), axis=0),
                rConcat(rArray.randn(10), rArray.randn(10), axis=0),
            ),
            rc.pull_concat_once_raw,
        ),
        (
            rAdd(
                rConcat(rArray.randn(10), rArray.randn(10), axis=0),
                rConcat(rArray.randn(10), rArray.randn(10), axis=0),
            ),
            lambda x: compiler_simp(deep_push_down_index_raw(compiler_simp(op.unwrap(rc.pull_concat_once_raw(x))))),
        ),
        (rArray(torch.randn(10, 10)), functools.partial(split_to_concat, axis=1, sections=[2, 2, 6])),
        (
            P(
                """
'Ein' Einsum ab->b
  'Concat' Concat 1
    'one' [5,1] Scalar 1
    'zero' [5,4] Scalar 0"""
            ),
            rc.pull_concat_once_raw,
        ),
        (
            rAdd(
                rScalar(1.0, ()),
                rScalar(2.0, ()),
                rAdd(rScalar(1.0, ()), rScalar(2.0, ())),
            ),
            deep_heuristic_nest_adds,
        ),
        (
            P(
                """
0 'Concat' Concat 0
  'Index' [6] Index [1,0:6,3]
    1 'Concat' Concat 0
      'zero' [1,6,5] Scalar 0
      'unnamed' [5,6,5] Scalar 0.9169063455989984"""
            ),
            compiler_simp,
        ),
        (
            rIndex(rEinsum((rScalar(1.0, (2,)), (0,)), out_axes=(0, 0)), I[0, :]),
            index_einsum_to_scatter,
        ),  # single
        (
            rIndex(rEinsum((rScalar(1.0, (2,)), (0,)), out_axes=(0, 0)), I[0, :]),
            compiler_simp,
        ),  # single
        (
            P(
                """
'Index' [2] Index [1,0:2]
  'Ein' Einsum aa,bcda->aa
    0 'unnamed' [2,2] Scalar 0
    'Add' Add
      1 'ScalarMul' Einsum abcd,->abcd
        2 'unnamed' [3,5,5,2] Scalar 0.6158525483081693
        3 'unnamed' [] Scalar 0
      4 'ScalarMul' Einsum a,->a
        5 'unnamed' [1] Scalar 4.6430197068796994
        6 'unnamed' [] Scalar -12.139412884795673
      7 'ScalarMul' Einsum ab,->ab
        'zero' [1,1] Scalar 0
        8 'unnamed' [] Scalar -2"""
            ),
            compiler_simp_step,
        ),
        (
            (
                lambda x: rEinsum(
                    (
                        rIndex(
                            x,
                            (torch.randint(0, 4, (10,)),),
                        ),
                        (0,),
                    ),
                    (
                        rIndex(
                            x,
                            I[
                                :4,
                            ],
                        ),
                        (1,),
                    ),
                    out_axes=(0, 1),
                )
            )(
                rIndex(
                    rArray(torch.randn((10))),
                    I[
                        :4,
                    ],
                )
            ),
            compiler_simp,
        ),  # single
        (
            rScatter(
                rRearrange(
                    rArray(torch.randn((10))),
                    rRearrangeSpec(
                        [[0]],
                        [[0], [1]],
                        [10, 20],
                    ),
                ),
                I[0:10, 0:20],
                (20, 20),
            ),
            scatter_pull_removable_axes,
        ),  # single
        (
            rEinsum(
                (
                    rScatter(
                        rScalar(1.0, (1, 2)),
                        I[0:1, 0:2],
                        (2, 2),
                    ),
                    (0, 1),
                ),
                out_axes=(0, 1),
            ),
            einsum_pull_scatter,
        ),  # single
        (
            rEinsum(
                (
                    rScatter(
                        rScalar(1.0, (1, 2)),
                        I[0:1, 0:2],
                        (2, 2),
                    ),
                    (0, 1),
                ),
                (
                    rScatter(
                        rScalar(1.0, (1, 2)),
                        I[1:2, 0:2],
                        (2, 2),
                    ),
                    (0, 1),
                ),
                out_axes=(0, 1),
            ),
            einsum_pull_scatter,
        ),  # zero
        (
            rEinsum(
                (
                    rScatter(
                        rScalar(1.0, (2, 3)),
                        I[0:2, 0:3],
                        (3, 3),
                    ),
                    (0, 1),
                ),
                (
                    rScatter(
                        rScalar(1.0, (2, 3)),
                        I[1:3, 0:3],
                        (3, 3),
                    ),
                    (0, 1),
                ),
                out_axes=(0, 1),
            ),
            einsum_pull_scatter,
        ),  # intersection
        (
            rAdd(
                rScatter(
                    rScalar(
                        1.0,
                        (1, 1),
                    ),
                    I[1:2, 0:1],
                    (2, 2),
                ),
                rScatter(
                    rScalar(
                        1.0,
                        (1, 1),
                    ),
                    I[1:2, 0:1],
                    (2, 2),
                ),
            ),
            add_pull_scatter,
        ),  # multi
        (
            rAdd(
                rScatter(
                    rScalar(
                        1.0,
                        (1, 1),
                    ),
                    I[2:3, 0:1],
                    (3, 2),
                ),
                rScatter(
                    rScalar(
                        1.0,
                        (1, 1),
                    ),
                    I[1:2, 0:1],
                    (3, 2),
                ),
            ),
            add_pull_scatter,
        ),  # multi
        (rScatter(rArray(torch.randn((2, 2))), I[0:2, 0:2], (2, 2)), compiler_simp),
        (
            py_to_rust(
                Index(
                    node=Einsum(
                        args=(
                            (
                                Index(
                                    node=Zero(
                                        shape=(3, 2, 2, 2, 2),
                                    ),
                                    index=(
                                        slice(0, 1, None),
                                        slice(0, 2, None),
                                        slice(1, 2, None),
                                        1,
                                        torch.randint(0, 2, torch.Size([6])),
                                    ),
                                    hash_tensor_idx_by_value=True,
                                    shape=(1, 2, 1, 6),
                                ),
                                (0, 1, 2, 3),
                            ),
                        ),
                        out_axes=(3, 3, 3),
                        shape=(6, 6, 6),
                    ),
                    index=(3, slice(3, 5, None), 0),
                    hash_tensor_idx_by_value=True,
                    shape=(2,),
                )
            ),
            compiler_simp_step,
        ),
        (
            py_to_rust(
                Einsum(
                    args=(
                        (
                            Index(
                                node=ArrayConstant(
                                    value=torch.randn(torch.Size([3, 4, 3])),
                                    shape=(3, 4, 3),
                                    uuid=UUID("c6fc50d9-4a2f-39fc-ef28-49d239ea752f"),
                                ),
                                index=(
                                    torch.randint(0, 3, torch.Size([10])),
                                    torch.randint(0, 4, torch.Size([5])),
                                    torch.randint(0, 3, torch.Size([20])),
                                ),
                                hash_tensor_idx_by_value=True,
                                shape=(10, 5, 20),
                            ),
                            (0, 1, 2),
                        ),
                        (
                            UnaryRearrange(
                                node=ArrayConstant(
                                    value=torch.randn(torch.Size([])),
                                    shape=(),
                                    uuid=UUID("f20db4f4-55b4-3037-f60e-a84e9d79cb78"),
                                ),
                                op_string=" -> a b c d",
                                axes_lengths=(("a", 10), ("b", 2), ("c", 10), ("d", 1)),
                                shape=(10, 2, 10, 1),
                            ),
                            (0, 3, 0, 4),
                        ),
                        (
                            UnaryRearrange(
                                node=Index(
                                    node=Einsum(
                                        args=(
                                            (
                                                ArrayConstant(
                                                    value=torch.randn(torch.Size([5, 5, 1, 2])),
                                                    shape=(5, 5, 1, 2),
                                                    uuid=UUID("3731617c-190c-4893-d765-f6ee0b2f9a44"),
                                                ),
                                                (0, 1, 2, 3),
                                            ),
                                            (
                                                Einsum(
                                                    args=(
                                                        (
                                                            ArrayConstant(
                                                                value=torch.randn(torch.Size([4])),
                                                                shape=(4,),
                                                                uuid=UUID("f132760d-10d5-bdae-17c7-b9e18e288c91"),
                                                            ),
                                                            (0,),
                                                        ),
                                                        (
                                                            ArrayConstant(
                                                                value=torch.randn(torch.Size([4, 4, 4, 4])),
                                                                shape=(4, 4, 4, 4),
                                                                uuid=UUID("0b4321c8-3a82-0919-126f-9df8ec139817"),
                                                            ),
                                                            (1, 2, 3, 4),
                                                        ),
                                                        (
                                                            Einsum(
                                                                args=(
                                                                    (
                                                                        ArrayConstant(
                                                                            value=torch.randn(torch.Size([3, 2, 3, 3])),
                                                                            shape=(3, 2, 3, 3),
                                                                            uuid=UUID(
                                                                                "0fa0e225-2f70-b7a3-8471-c0761c1e51b3"
                                                                            ),
                                                                        ),
                                                                        (0, 1, 0, 0),
                                                                    ),
                                                                    (
                                                                        UnaryRearrange(
                                                                            node=ArrayConstant(
                                                                                value=torch.randn(torch.Size([3])),
                                                                                shape=(3,),
                                                                                uuid=UUID(
                                                                                    "12174f8c-87a0-6607-8b58-e3a59c0fd729"
                                                                                ),
                                                                            ),
                                                                            op_string="(a b) -> (a b)",
                                                                            axes_lengths=(("a", 3), ("b", 1)),
                                                                            shape=(3,),
                                                                        ),
                                                                        (0,),
                                                                    ),
                                                                    (
                                                                        ArrayConstant(
                                                                            value=torch.randn(torch.Size([3, 3])),
                                                                            shape=(3, 3),
                                                                            uuid=UUID(
                                                                                "73c32275-4efa-5cee-5a17-915628013216"
                                                                            ),
                                                                        ),
                                                                        (2, 0),
                                                                    ),
                                                                ),
                                                                out_axes=(0,),
                                                                shape=(3,),
                                                            ),
                                                            (5,),
                                                        ),
                                                        (
                                                            Index(
                                                                node=ArrayConstant(
                                                                    value=torch.randn(torch.Size([5, 6, 1, 5, 3, 2])),
                                                                    shape=(5, 6, 1, 5, 3, 2),
                                                                    uuid=UUID("9a0bf704-fb05-63c4-574e-f3c7d7f28da9"),
                                                                ),
                                                                index=(
                                                                    2,
                                                                    torch.randint(2, 4, torch.Size([2])),
                                                                    torch.randint(0, 1, torch.Size([3])),
                                                                    slice(
                                                                        0, 2, None
                                                                    ),  # changed from (-50, 2, None) to avoid triggering new Rust slice spec
                                                                    torch.randint(0, 2, torch.Size([3])),
                                                                    -1,
                                                                ),
                                                                hash_tensor_idx_by_value=True,
                                                                shape=(2, 3, 2, 3),
                                                            ),
                                                            (6, 7, 6, 8),
                                                        ),
                                                    ),
                                                    out_axes=(8, 6, 3, 3),
                                                    shape=(3, 2, 4, 4),
                                                ),
                                                (4, 5, 6, 7),
                                            ),
                                            (
                                                Concat(
                                                    circuits=(
                                                        UnaryRearrange(
                                                            node=ArrayConstant(
                                                                value=torch.randn(torch.Size([1])),
                                                                shape=(1,),
                                                                uuid=UUID("636f41ab-1519-2bea-56db-702ae62defa9"),
                                                            ),
                                                            op_string="a -> (b c d) (e f a g)",
                                                            axes_lengths=(
                                                                ("a", 1),
                                                                ("b", 3),
                                                                ("c", 1),
                                                                ("d", 1),
                                                                ("e", 2),
                                                                ("f", 1),
                                                                ("g", 1),
                                                            ),
                                                            shape=(3, 2),
                                                        ),
                                                    ),
                                                    axis=1,
                                                    shape=(3, 2),
                                                ),
                                                (8, 9),
                                            ),
                                            (
                                                GeneralFunction(
                                                    node=ArrayConstant(
                                                        value=torch.randn(torch.Size([2])),
                                                        shape=(2,),
                                                        uuid=UUID("48048e7c-6bc8-af1a-b834-14a1d3446885"),
                                                    ),
                                                    function=sigmoid_fn,
                                                    get_jacobian=None,
                                                    allows_batching=True,
                                                    non_batch_dims=(),
                                                    shape=(2,),
                                                ),
                                                (10,),
                                            ),
                                        ),
                                        out_axes=(),
                                        shape=(),
                                    ),
                                    index=(),
                                    hash_tensor_idx_by_value=True,
                                    shape=(),
                                ),
                                op_string=" -> ",
                                axes_lengths=(),
                                shape=(),
                            ),
                            (),
                        ),
                    ),
                    out_axes=(0, 2),
                    shape=(10, 20),
                )
            ),
            compiler_simp,
        ),
        (
            py_to_rust(
                UnaryRearrange(
                    node=UnaryRearrange(
                        node=Einsum(
                            args=(
                                (
                                    ArrayConstant(
                                        value=torch.randn(torch.Size([4])),
                                        shape=(4,),
                                        uuid=UUID("063777fa-33fb-be22-c438-ec97577c0bd1"),
                                        name="vxsxqktj",
                                    ),
                                    (0,),
                                ),
                                (One(shape=(4, 1, 1, 3), name="sjhfdjvd"), (1, 2, 3, 4)),
                                (
                                    UnaryRearrange(
                                        node=ArrayConstant(
                                            value=torch.randn(torch.Size([])),
                                            shape=(),
                                            uuid=UUID("31c481b4-b09e-cadf-986b-62f3cc55a9bb"),
                                            name="",
                                        ),
                                        op_string=" -> a b",
                                        axes_lengths=(("a", 4), ("b", 4)),
                                        name="wucuuqwf",
                                        shape=(4, 4),
                                        is_constant=True,
                                        is_explicitly_computable=True,
                                        can_be_sampled=True,
                                    ),
                                    (5, 5),
                                ),
                                (
                                    Index(
                                        node=ArrayConstant(
                                            value=torch.randn(torch.Size([3, 5, 4, 5, 3, 3])),
                                            shape=(3, 5, 4, 5, 3, 3),
                                            uuid=UUID("9955c931-80e7-b7f8-8588-a67ba6e23a2e"),
                                            name="tpeyfkue",
                                        ),
                                        index=(
                                            torch.randint(0, 3, torch.Size([4])),
                                            4,
                                            slice(-4, -1, None),
                                            -3,
                                            torch.randint(0, 3, torch.Size([4])),
                                            torch.randint(1, 3, torch.Size([4])),
                                        ),
                                        name="",
                                        hash_tensor_idx_by_value=True,
                                        shape=(4, 3, 4, 4),
                                        is_constant=True,
                                        is_explicitly_computable=True,
                                        can_be_sampled=True,
                                    ),
                                    (5, 6, 1, 1),
                                ),
                            ),
                            out_axes=(1, 5),
                            name="diigorgu",
                            shape=(4, 4),
                            is_constant=True,
                            is_explicitly_computable=True,
                            can_be_sampled=True,
                        ),
                        op_string="(1 a 1) b -> a b",
                        axes_lengths=(("a", 4), ("b", 4)),
                        name="vvyjseln",
                        shape=(4, 4),
                        is_constant=True,
                        is_explicitly_computable=True,
                        can_be_sampled=True,
                    ),
                    op_string="a b -> a b",
                    axes_lengths=(("a", 4), ("b", 4)),
                    name="",
                    shape=(4, 4),
                    is_constant=True,
                    is_explicitly_computable=True,
                    can_be_sampled=True,
                )
            ),
            lambda x: substitute_all_modules(batch_to_concat(x, 0, 1)),
        ),
        (
            rAdd(
                rAdd(
                    rScalar(1.0, ()),
                    rScalar(2.0, ()),
                    rAdd(rScalar(1.0, ()), rScalar(2.0, ())),
                ),
                rScalar(1.0, ()),
                rScalar(2.0, ()),
            ),
            deep_heuristic_nest_adds,
        ),
        (
            P(
                """0 Concat 1
  1 Index [0:1,0:1]
    2 [4,4] Array
  3 Index [0:1,1:3]
    2"""
            ),
            concat_elim_split,
        ),
        (
            P(
                """
0  [1] GeneralFunction min
  1  [1,2] Scalar -2.941891537492132
        """
            ),
            generalfunction_pull_removable_axes,
        ),
        (
            P(
                """
0  [5,9,1] GeneralFunction min
  1  [5,9,1,2] Scalar -2.941891537492132
        """
            ),
            generalfunction_pull_removable_axes,
        ),
        (
            P(
                """
0  [5,9,1,2] GeneralFunction softmax
  1  [5,9,1,2] Scalar -2.941891537492132
        """
            ),
            generalfunction_pull_removable_axes,
        ),
    ]
    for circ, rewrite in setups:
        circ = strip_names_and_tags(r_cast_circuit(circ, TorchDeviceDtypeOp("cpu", "float64")))
        assert not any("internal_expand" in x.name for x in all_children(circ))
        circ.print()
        rewritten = cast(Any, rewrite)(circ)
        assert not any("internal_expand" in x.name for x in all_children(rewritten))
        assert rewritten is not None
        rewritten.print()
        assert_close(
            circ.evaluate(),
            rewritten.evaluate(),
        )


def test_rust_evaluate_exact_value():
    circs = [
        (
            rArray.randn(10, 10, device_dtype=TorchDeviceDtypeOp.default(), seed=0),
            rArray.randn(10, 10, device_dtype=TorchDeviceDtypeOp.default(), seed=0).evaluate(),
        )
    ]
    for circ, value in circs:
        assert_close(circ.evaluate(), value)


def test_scatter_slightly():
    circ = rScatter(rScalar(1, (2, 2)), I[1:3, 1:3], (4, 4))
    result = rc.cast_circuit(circ, TorchDeviceDtypeOp(dtype="float64")).evaluate()
    assert_close(
        tensor(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        result,
    )


@hypothesis.settings(
    deadline=None
)  # This makes no sense! This just randomly started being needed in only CI?!? (works locally...)
@hypothesis.given(
    start=st.one_of(
        st.integers(min_value=-(2 ** 61), max_value=2 ** 61),
        st.integers(min_value=-(1000), max_value=1000),
        st.just(None),
    ),
    stop=st.one_of(
        st.integers(min_value=-(2 ** 61), max_value=2 ** 61),
        st.integers(min_value=-(1000), max_value=1000),
        st.just(None),
    ),
    l=st.integers(min_value=0, max_value=1000),
)
def test_rust_slice_shape(start: Optional[int], stop: Optional[int], l: int):
    sl = slice(start, stop)
    true_len = len(([None] * l)[sl])

    thunk = lambda: rIndex(rSymbol((l,), uuid4()), (sl,)).shape
    if (
        (start is not None and (start < -l or start > l))
        or (stop is not None and (stop < -l or stop > l))
        or (
            start is not None
            and stop is not None
            and (start if start >= 0 else start + l) > (stop if stop >= 0 else stop + l)
        )
    ):
        with pytest.raises(ValueError):
            thunk()
    else:
        assert thunk() == (true_len,)


def rearrange_spec_rust_to_python(rust_spec):
    assert isinstance(rust_spec, rRearrangeSpec)
    return pRearrangeSpec.from_rust(rust_spec)


def rearrange_spec_python_to_rust(py_spec):
    assert isinstance(py_spec, pRearrangeSpec)
    return py_spec.to_rust()


@hypothesis.given(st.data())
def test_fuse_rearrange(data):
    outer = data.draw(st_c.rearrange_spec())
    mid_shape, out_shape = outer.get_shapes()
    inner = data.draw(st_c.rearrange_spec(out_shape=mid_shape))
    in_shape, mid_shape2 = inner.get_shapes()
    assert mid_shape == mid_shape2

    inner_rust = rearrange_spec_python_to_rust(inner)
    outer_rust = rearrange_spec_python_to_rust(outer)
    try:
        fused = rRearrangeSpec.fuse(inner_rust, outer_rust)
    except ValueError as e:
        print("not composable", outer, inner)
        hypothesis.assume(False, "not composable")
        raise e  # type checker happy
    arr = torch.arange(math.prod(in_shape)).reshape(in_shape)
    if not fused.is_valid():
        print("ERRRRR")
        print(arr.shape)
        print(fused)
        assert False
    assert torch.equal(outer.apply(inner.apply(arr)), fused.apply(arr))


@hypothesis.given(py_spec=st_c.rearrange_spec(out_shape=(12, 6, 6)))
def test_canonicalize(py_spec: pRearrangeSpec):
    # Note: can't compare to python version because rust canonicalize has slightly different rules
    in_shape, out_shape = py_spec.get_shapes()
    arr = torch.arange(math.prod(in_shape)).reshape(in_shape)

    for special_case_ones in [True, False]:
        # use rust function (but convert back to python rearrange spec to compare)
        rust_canonicalized_spec_rust = rearrange_spec_python_to_rust(py_spec).canonicalize(special_case_ones)
        # print(rust_canonicalized_spec_rust)
        rust_canonicalized_spec = rearrange_spec_rust_to_python(rust_canonicalized_spec_rust)

        # check that canonicalizing didn't change how it works
        assert torch.equal(py_spec.apply(arr), rust_canonicalized_spec.apply(arr))

        # check that simplification rules were satisfied
        assert_spec_is_canonicalized(rust_canonicalized_spec)
        if py_spec.is_permute():
            assert all(
                len(i) == 1 for i in rust_canonicalized_spec.input_ints + rust_canonicalized_spec.output_ints
            ), "Some canon elements are not a single-element tuple, but the spec is a permutation"


def assert_spec_is_canonicalized(canon_spec: pRearrangeSpec):
    # Check every int in input appears in the output
    input_ints = set(itertools.chain.from_iterable(canon_spec.input_ints))
    output_ints = set(itertools.chain.from_iterable(canon_spec.output_ints))
    assert len(input_ints - output_ints) == 0

    # Check that all ints in the input or output are at the beginning or end of
    # tuples, and never have the same other int next to them always.
    #
    # This replicates the logic of the beginning of the `canonicalize()` function,
    # but it is still useful because `canonicalize()` does things to remove this condition.
    # So we can check it is indeed removed.
    before_ints: List[Optional[int]] = [None] * len(canon_spec.int_sizes)
    after_ints: List[Optional[int]] = [None] * len(canon_spec.int_sizes)
    ENDING = -1

    assert max(output_ints) == len(canon_spec.int_sizes) - 1

    for tup in canon_spec.input_ints + canon_spec.output_ints:
        if len(tup) == 0:
            continue
        if before_ints[tup[0]] is None:
            before_ints[tup[0]] = ENDING
        if after_ints[tup[-1]] is None:
            after_ints[tup[-1]] = ENDING

        for i in range(1, len(tup)):
            if before_ints[tup[i]] is None:
                before_ints[tup[i]] = tup[i - 1]
            if before_ints[tup[i - 1]] is None:
                after_ints[tup[i - 1]] = tup[i]

            if before_ints[tup[i]] != tup[i - 1]:
                before_ints[tup[i]] = ENDING
            if before_ints[tup[i - 1]] != tup[i]:
                before_ints[tup[i - 1]] = ENDING

    assert all(i == ENDING or j == ENDING for i, j in zip(before_ints, after_ints))


@st.composite
def reshape_shapes(draw: st.DrawFn, st_shape: st.SearchStrategy[Shape]) -> Tuple[Shape, Shape]:
    old_shape = list(draw(st_shape))
    o_el = math.prod(old_shape)
    o_i = draw(st.integers(0, len(old_shape) - 1))

    new_shape = list(draw(st_shape))
    n_el = math.prod(new_shape)
    n_i = draw(st.integers(0, len(new_shape) - 1))

    lcm = math.lcm(o_el, n_el)
    if not draw(st.booleans()):  # Force them to be mutually divisible
        old_shape.insert(o_i, lcm // o_el)
        new_shape.insert(n_i, lcm // n_el)
    return tuple(old_shape), tuple(new_shape)


@hypothesis.settings(deadline=None)
@hypothesis.given(reshape_shapes(st_np.array_shapes(max_dims=5, max_side=8)))
def test_reshape(shapes: Tuple[Shape, Shape]):
    old_shape, new_shape = shapes

    c: rCircuit
    if math.prod(old_shape) == math.prod(new_shape):
        arr = torch.arange(math.prod(old_shape), dtype=torch.float32)
        c = rArray(arr, None)
        rs = rRearrange.reshape(c, shape=new_shape)

        assert torch.equal(rs.evaluate(), arr.view(new_shape))
    else:
        c = rSymbol.new_with_random_uuid(old_shape)
        with pytest.raises(ValueError):
            rRearrange.reshape(c, shape=new_shape)


def test_einsum_string():
    string = "ab, bc,cd-> ad"
    rEinsum.from_einsum_string(string, rScalar(1, (1, 2)), rScalar(2, (2, 3)), rScalar(3, (3, 4))).print()


def test_fancy_einsum_string():
    string = "hiii persn, persn yo , yo woo  -> hiii woo"
    rEinsum.from_fancy_string(string, rScalar(1, (1, 2)), rScalar(2, (2, 3)), rScalar(3, (3, 4))).print()


def test_mixed_dtype_error():
    with pytest.raises(ValueError):
        rAdd(
            rArray(torch.tensor([1, 2, 3], dtype=torch.float32)),
            rArray(torch.tensor([4, 5, 6], dtype=torch.float64)),
        )
    # with pytest.raises(ValueError):
    #     circ = rAdd([rArray(torch.tensor([1, 2, 3], device="cuda:0")), rArray(torch.tensor([4, 5, 6], device="cpu"))])


@pytest.mark.parametrize("device", ["cpu"])  # , "cuda:0"]) for when you have cuda
def test_tensor_hashing(device):
    tensor1 = torch.randn(10, 10, device=device)
    assert rArray(tensor1) == rArray(tensor1)
    assert rArray(tensor1) == rArray(tensor1 + 0)
    assert rArray(tensor1) != rArray(torch.randn(10, 10, device=device))
    assert rArray(tensor1) != rArray(tensor1.to(dtype=torch.float16))

    # reinterpret cast makes different hash
    assert rArray(tensor1) != rArray(tensor1.view(dtype=torch.int32))
    if device == "cuda:0":
        assert rArray(tensor1) != rArray(tensor1.to(device="cpu"))
        assert rArray(tensor1) == rArray(tensor1.to(device="cpu").to(device=device))

    tensor2 = torch.randn(10000, 1000, device=device)
    assert rArray(tensor2) == rArray(tensor2 + 0)


def test_scatter_to_py():
    """rust_to_py converts scatter to concat first"""
    circs = [
        rScatter(rArray(torch.randn(10, 10)), I[0:10, 0:10], (20, 20)),
        rScatter(rArray(torch.randn(10, 10, 10)), I[0:10, 0:10, 0:10], (20, 10, 20)),
    ]
    for circ in circs:
        assert_close(circ.evaluate(), MemoizedFn(evaluate_fn())(rust_to_py(circ)))


# @hypothesis.settings(max_examples=2000)
@hypothesis.given(
    st_Circuit(
        (10, 20),
        must_be_constant=True,
        probs_default=CP.kw(all=1, Cumulant=0, Zero=0, One=0, FloatConstant=0, ArrayConstant=3),
        must_be_explicitly_computable=True,
        rust=True,
    )
)
def test_simp_nodes_low(circ):
    simped = compiler_simp(circ)
    assert r_count_nodes(simped) / 2 < r_count_nodes(circ)


@hypothesis.settings(deadline=None)
@hypothesis.given(circ=get_c_st(rust=True))
@pytest.mark.parametrize("naive_schedule", [False])
@pytest.mark.xfail
def test_scheduled_execution(circ, naive_schedule):
    raw_test_scheduled_execution(circ, naive_schedule)


def raw_test_scheduled_execution(circ, naive_schedule):
    circ_rust = rc.cast_circuit(circ, TorchDeviceDtypeOp(device="cpu", dtype="float64"))
    circ_rust.print()
    s_eval = scheduled_evaluate(
        circ_rust,
        OptimizationSettings(
            3,
            16_000_000,
            scheduling_naive=naive_schedule,
        ),
    )
    normal_eval = circ_rust.evaluate()
    assert_close(s_eval, normal_eval)
    optimized_eval = optimize_and_evaluate(
        circ_rust,
        OptimizationSettings(
            0,
            800_000,
            scheduling_naive=naive_schedule,
        ),
    )
    assert_close(normal_eval, optimized_eval, atol=1e-5, rtol=1e-5)


def test_scheduled_execution_manual():
    ac = rArray.randn(50, 50)
    circuits: List[Tuple[rCircuit, int]] = [
        (rEinsum((ac, (0, 1)), (rIndex(ac, I[0:49, :]), (2, 1)), out_axes=(0,)), 50 * 50)
    ]
    for circuit, cost in circuits:
        circuit = r_cast_circuit(circuit, TorchDeviceDtypeOp("cpu", "float64"))
        s_eval = scheduled_evaluate(circuit, OptimizationSettings(max_memory=cost * 8))
        normal_eval = circuit.evaluate()
        assert_close(normal_eval, s_eval)


@pytest.mark.xfail
@hypothesis.settings(deadline=None)
@hypothesis.given(circ=get_c_st(rust=True))
@mark_not_interesting_if(SchedulingOOMError)
def test_rust_opt_eval(circ):
    raw_test_rust_opt_eval(circ)


@pytest.mark.cuda
@hypothesis.settings(deadline=None)
@hypothesis.given(circ=get_c_st(rust=True))
@mark_not_interesting_if(SchedulingOOMError)
def test_rust_opt_eval_cuda(circ):
    raw_test_rust_opt_eval(circ, device="cuda:0")


def raw_test_rust_opt_eval(circ, device=None):
    ddto = rc.TorchDeviceDtypeOp(device=device, dtype="float64")
    circ = rc.cast_circuit(circ, ddto)
    normal_eval = circ.evaluate()
    optimized_eval = optimize_and_evaluate(circ, OptimizationSettings(0, 50_000))
    hypothesis.assume(torch.isfinite(normal_eval).all(), "evaluation not finite")
    assert_close(normal_eval, optimized_eval, rtol=1e-4, atol=1e-4)
    numerical_scale_eval = optimize_and_evaluate(
        circ,
        OptimizationSettings(0, 50_000, adjust_numerical_scale=True),
    )
    hypothesis.assume(torch.isfinite(numerical_scale_eval).all(), "evaluation not finite")
    assert_close(normal_eval, numerical_scale_eval, rtol=1e-4, atol=1e-4)


def test_adjust_scale():
    circs = [
        rEinsum(
            (
                rEinsum(
                    (
                        rEinsum(
                            (rArray(torch.full((), 1e16, dtype=torch.float32)), ()),
                            (rArray(torch.full((), 1e16, dtype=torch.float32)), ()),
                            out_axes=(),
                        ),
                        (),
                    ),
                    (
                        rAdd(
                            rArray(torch.full((), 1e16, dtype=torch.float32)),
                            rArray(torch.full((), 1e16, dtype=torch.float32)),
                        ),
                        (),
                    ),
                    out_axes=(),
                ),
                (),
            ),
            (rScalar(1e-48, ()), ()),
            out_axes=(),
        ),
        rEinsum(
            (
                rEinsum(
                    (
                        rEinsum(
                            (rArray(torch.full((), 3e6, dtype=torch.float32)), ()),
                            (rArray(torch.full((), 3e6, dtype=torch.float32)), ()),
                            (rArray(torch.full((), 3e6, dtype=torch.float32)), ()),
                            (rArray(torch.full((), 3e6, dtype=torch.float32)), ()),
                            (rArray(torch.full((), 3e6, dtype=torch.float32)), ()),
                            (rArray(torch.full((), 3e6, dtype=torch.float32)), ()),
                            (rArray(torch.full((), 3e6, dtype=torch.float32)), ()),
                            (rArray(torch.full((), 3e6, dtype=torch.float32)), ()),
                            (rArray(torch.full((), 3e6, dtype=torch.float32)), ()),
                            (rArray(torch.full((), 3e6, dtype=torch.float32)), ()),
                            out_axes=(),
                        ),
                        (),
                    ),
                    (
                        rAdd(
                            rArray(torch.full((), 3e6, dtype=torch.float32)),
                            rArray(torch.full((), 3e6, dtype=torch.float32)),
                            rArray(torch.full((), 3e6, dtype=torch.float32)),
                            rArray(torch.full((), 3e6, dtype=torch.float32)),
                            rArray(torch.full((), 3e6, dtype=torch.float32)),
                            rArray(torch.full((), 3e6, dtype=torch.float32)),
                            rArray(torch.full((), 3e6, dtype=torch.float32)),
                        ),
                        (),
                    ),
                    out_axes=(),
                ),
                (),
            ),
            (rScalar(1e-60, ()), ()),
            out_axes=(),
        ),
    ]
    for circ in circs:
        circ64 = r_cast_circuit(circ, TorchDeviceDtypeOp("cpu", "float64"))
        result = optimize_and_evaluate(
            circ64,
            OptimizationSettings(
                adjust_numerical_scale=True,
                verbose=0,
            ),
        )
        result2 = optimize_and_evaluate(circ64)
        official_result = circ64.evaluate()
        print(result, official_result, result2)
        assert_close(result, official_result)
        naive_result = optimize_and_evaluate(
            circ,
            OptimizationSettings(adjust_numerical_scale=False, verbose=0),
        )
        print(naive_result)
        if torch.isfinite(naive_result):
            assert_close(naive_result, torch.full((), 0.0, dtype=torch.float32))


def test_array_symbolic_error():
    with pytest.raises(ConstructArrayHasReservedSymbolicShapeError):
        rArray(torch.randn(10_007, 1))


def test_eval_many():
    arrconst = rArray(torch.randn(10, 10))
    circs = [rAdd(arrconst, arrconst), arrconst, rEinsum((arrconst, (0, 1)), out_axes=())]
    many_evaled = optimize_and_evaluate_many(circs, OptimizationSettings(0, 1_000_000))
    for circ, evaled in zip(circs, many_evaled):
        assert_close(circ.evaluate(), evaled)


def test_rust_generalfunction_same():
    names = ["sigmoid", "softmax", "gelu", "relu", "log_exp_p_1", "gaussian_cdf", "gaussian_pdf"]
    for name in names:
        circ = r_cast_circuit(
            rGeneralFunction.new_by_name(rArray.randn(10, 10, 10), spec_name=name),
            TorchDeviceDtypeOp("cpu", "float64"),
        )
        assert_close(
            circ.evaluate(),
            MemoizedFn(evaluate_fn(dtype=torch.float64, device="cpu"))(rust_to_py(circ)),
            rtol=1e-12,
            atol=1e-12,
        )


def test_strip_names_and_tags():
    circ = P(
        """
'tag' Tag 2184e144-2726-4326-9452-f19041a101bb
  'hi' Add
    'ho' [2,2] Scalar 1
    'weee' [2,2] Symbol
    """
    )
    assert (
        strip_names_and_tags(circ).repr()
        == """0 Add
  1 [2,2] Scalar 1
  2 Tag 00000000-0000-0000-0000-000000000000
    'weee' [2,2] Symbol"""
    )


@pytest.mark.cuda
def test_gpu_hash_not_crashing():
    for i in range(200):
        arr = rArray(torch.randn(501, 501, device="cuda:0", dtype=torch.float64))
        result = rEinsum((arr, (0, 1)), (arr, (1, 2)), out_axes=()).evaluate()
        print(result)


def test_rearrange_from_string():
    # this could get fucked up by numbering changing, but whatever
    # also doesn't check edge cases very well
    z = rRearrangeSpec.from_string("a b->b a")
    assert z.input_ints == [[0], [1]]
    assert z.output_ints == [[1], [0]]
    assert z.int_sizes == [None, None]
    z = rRearrangeSpec.from_string("a b->b a 10")
    assert z.input_ints == [[0], [1]]
    assert z.output_ints == [[1], [0], [2]]
    assert z.int_sizes == [None, None, 10]
    z = rRearrangeSpec.from_string("a b->b (11 a 10)")
    assert z.input_ints == [[0], [1]]
    assert z.output_ints == [[1], [2, 0, 3]]
    assert z.int_sizes == [None, None, 11, 10]
    z = rRearrangeSpec.from_string(" adf b ->   b (  11 adf 10)")
    assert z.input_ints == [[0], [1]]
    assert z.output_ints == [[1], [2, 0, 3]]
    assert z.int_sizes == [None, None, 11, 10]
    z = rRearrangeSpec.from_string(" adf b2 ->   b2 (  11 adf 10)")
    assert z.input_ints == [[0], [1]]
    assert z.output_ints == [[1], [2, 0, 3]]
    assert z.int_sizes == [None, None, 11, 10]
    z = rRearrangeSpec.from_string("(adf:17 other c:4) b2 -> (b2 c) adf other")
    assert z.to_string() == "(a:17 b c:4) d -> (d c:4) a:17 b"
    assert z.input_ints == [[0, 1, 2], [3]]
    assert z.output_ints == [[3, 2], [0], [1]]
    assert z.int_sizes == [17, None, 4, None]

    sym_sizes = symbolic_sizes()
    s0 = sym_sizes[0]
    s1 = sym_sizes[1]
    s6 = sym_sizes[6]
    s17 = sym_sizes[17]
    s240 = sym_sizes[240]
    z = rRearrangeSpec.from_string("(adf:0s other c:0s*1s d:14*2*6s) b2 -> (b2 c) adf other 0s (d 17s) 240s")
    assert z.to_string() == "(a:0s b c:0s*1s d:28*6s) e -> (e c:0s*1s) a:0s b 0s (d:28*6s 17s) 240s"
    assert z.input_ints == [[0, 1, 2, 3], [4]]
    assert z.output_ints == [[4, 2], [0], [1], [5], [3, 6], [7]]
    assert z.int_sizes == [s0, None, s0 * s1, 28 * s6, None, s0, s17, s240]


def test_flatten():
    x: Any = rScalar(0.3)
    torch.testing.assert_close(x.flatten().evaluate(), torch.tensor(0.3, dtype=torch.float64).flatten())
    x = rArray.randn(3, device_dtype=TorchDeviceDtypeOp(dtype="float64"))
    torch.testing.assert_close(x.flatten().evaluate(), x.value.flatten())
    x = rArray.randn(3, 4, device_dtype=TorchDeviceDtypeOp(dtype="float64"))
    torch.testing.assert_close(x.flatten().evaluate(), x.value.flatten())
    x = rArray.randn(5, 3, 4, device_dtype=TorchDeviceDtypeOp(dtype="float64"))
    torch.testing.assert_close(x.flatten().evaluate(), x.value.flatten())


def test_unflatten():
    x = rArray.randn(3)
    torch.testing.assert_close(x.unflatten((1, 1, 3)).evaluate(), x.value.reshape(1, 1, 3))
    x = rArray.randn(1)
    torch.testing.assert_close(x.unflatten((1, 1, 1)).evaluate(), x.value.reshape(1, 1, 1))
    x = rArray.randn(1)
    torch.testing.assert_close(
        x.unflatten((1,)).evaluate(),
        x.value.reshape(
            1,
        ),
    )
    x = rArray.randn(12)
    torch.testing.assert_close(x.unflatten((2, 3, 2)).evaluate(), x.value.reshape(2, 3, 2))


def test_unflatten_axis():
    x = rArray.randn(3)
    torch.testing.assert_close(x.unflatten_axis(0, (1, 1, 3)).evaluate(), x.value.reshape(1, 1, 3))
    x = rArray.randn(1)
    torch.testing.assert_close(x.unflatten_axis(0, (1, 1, 1)).evaluate(), x.value.reshape(1, 1, 1))
    x = rArray.randn(1)
    torch.testing.assert_close(
        x.unflatten_axis(0, (1,)).evaluate(),
        x.value.reshape(
            1,
        ),
    )
    x = rArray.randn(12)
    torch.testing.assert_close(x.unflatten_axis(0, (2, 3, 2)).evaluate(), x.value.reshape(2, 3, 2))

    x = rArray.randn(3, 12)
    torch.testing.assert_close(x.unflatten_axis(0, (3,)).evaluate(), x.value.reshape(3, 12))
    x = rArray.randn(3, 12)
    torch.testing.assert_close(x.unflatten_axis(0, (3, 1)).evaluate(), x.value.reshape(3, 1, 12))
    x = rArray.randn(3, 12)
    torch.testing.assert_close(x.unflatten_axis(1, (2, 3, 1, 2)).evaluate(), x.value.reshape(3, 2, 3, 1, 2))


def test_weighted_add():
    a = rArray.randn(2, 3)
    b = rArray.randn(3, 2, 1)
    assert rAdd.from_weighted_nodes((a, 1.0), (b, 1)) == rAdd(a, b)
    torch.testing.assert_close(rAdd.from_weighted_nodes((a, 2.0), (b, 1)).evaluate(), a.value * 2 + b.value * 1)
    torch.testing.assert_close(rAdd.from_weighted_nodes((a, 2.3), (b, 1.7)).evaluate(), a.value * 2.3 + b.value * 1.7)
    torch.testing.assert_close(
        rAdd.from_weighted_nodes((a, 1.0), (b, 1), use_1_weights=True).evaluate(), a.value * 1 + b.value * 1
    )
    torch.testing.assert_close(
        rAdd.from_weighted_nodes((a, 2.0), (b, 1), use_1_weights=True).evaluate(), a.value * 2 + b.value * 1
    )
    torch.testing.assert_close(
        rAdd.from_weighted_nodes((a, 2.3), (b, 1.7), use_1_weights=True).evaluate(), a.value * 2.3 + b.value * 1.7
    )


def test_minus():
    a = rArray.randn(2, 3)
    b = rArray.randn(3, 2, 1)
    torch.testing.assert_close(a.sub(b).evaluate(), a.value - b.value)
    torch.testing.assert_close(rAdd.minus(a, b).evaluate(), a.value - b.value)


# example of using
def test_schedule_replace():
    # make your circuit
    circ = Parser(tensors_as_random=True)(
        """
0 Add
  1 [2,3] Array
  2 [5,2,3] Array"""
    )
    sched = optimize_to_schedule(circ)
    print(sched.evaluate())

    # make mapping from old rust Array hashes to new tensors.
    # may need py_to_rust to convert your arrayconstants from python to rust
    tensor_replacements = {circ.children[0].hash: ArrayConstant.randn(2, 3).value}
    replaced = sched.replace_tensors(tensor_replacements)
    print(replaced.evaluate())
    mapped = sched.map_tensors(lambda x: tensor_replacements.get(x))
    print(mapped.evaluate())
    assert_close(replaced.evaluate(), mapped.evaluate())


def test_module_fuse_concat():
    circuit = Parser(tensors_as_random=True).parse_circuits(
        """
'bilinear_mlp' Einsum dh,oh->o
  1 'folded_pre_activation' Rearrange (a:2 b) -> a:2 b
    2 'pre_activation' Einsum i,hi->h
      'bilinear_mlp.input' [3] Symbol
      'bilinear_mlp.project_in' [6, 3] Symbol
  'bilinear_mlp.project_out' [11, 3] Symbol
0 Add
  'b0' Module
    'bilinear_mlp'
    3 [3] Array ! 'bilinear_mlp.input'
    4 [6,3] Array ! 'bilinear_mlp.project_in'
    5 [11,3] Array ! 'bilinear_mlp.project_out'
  'b1' Module
    'bilinear_mlp'
    3 [3] Array ! 'bilinear_mlp.input'
    4 [6,3] Array ! 'bilinear_mlp.project_in'
    5 [11,3] Array ! 'bilinear_mlp.project_out'
  'b2' Module
    'bilinear_mlp'
    3 [3] Array ! 'bilinear_mlp.input'
    4 [6,3] Array ! 'bilinear_mlp.project_in'
    5 [11,3] Array ! 'bilinear_mlp.project_out'
    """,
    )[-1]
    fused = fuse_concat_modules(circuit, [circuit.children[0].cast_module(), circuit.children[1].cast_module()])
    fused.print()
    assert_close(substitute_all_modules(fused).evaluate(), substitute_all_modules(circuit).evaluate())
    fused = fuse_concat_modules(
        circuit,
        [
            circuit.children[0].cast_module(),
            circuit.children[1].cast_module(),
            circuit.children[2].cast_module(),
        ],
    )
    fused.print()
    assert_close(substitute_all_modules(fused).evaluate(), substitute_all_modules(circuit).evaluate())


def test_index_sync():
    icirc = rArray.randn(11, 7, 5, 3, 2)
    idx = I[0, torch.randint(0, 7, (2, 3)), torch.randint(0, 5, (2, 1)), 1, 0:2]
    synced_py = Index.index_synchronized_to_start(rust_to_py(icirc), idx)
    print(synced_py)
    synced_rs = rIndex.new_synchronized_to_start(icirc, idx)
    print(synced_rs)
    assert_close(MemoizedFn(evaluate_fn())(synced_py), synced_rs.evaluate())


def test_simple_add_pull():
    circ = Parser(tensors_as_random=True)(
        """
0 Add
  1 [5, 3, 6] Array
  2 [5, 3, 1] Scalar -12.307197501736319
  3 Rearrange a b -> b a
    4 [6, 1] Array
""",
    ).cast_add()
    add_pull_removable_axes(circ, True)


def test_weak_add_elim():
    s = """
    'ejfudybd' Add
      0 '' [4,4,4] Scalar -3.572176010699677
      1 '' [4] Array rand
    """
    c = Parser()(s)
    assert add_elim_removable_axes_weak(c.cast_add()) is None

    s = """
    'ejfudybd' Add
      0 '' [4,4,4] Scalar -3.572176010699677
      1 '' [4, 4, 4] Array rand
    """
    c = Parser()(s)
    elimed = op.unwrap(add_elim_removable_axes_weak(c.cast_add()))
    assert elimed.shape == c.shape
    elimed.print()

    s = """
    'ejfudybd' Add
      0 '' [4,4,4] Scalar -3.572176010699677
      2 '' [4, 4, 4] Rearrange a -> 4 a 4
        1 '' [4] Array rand
    """
    c = Parser()(s)
    elimed = op.unwrap(add_elim_removable_axes_weak(c.cast_add()))
    assert elimed.shape == c.shape
    elimed.print()


if __name__ == "__main__":
    test_weak_add_elim()
