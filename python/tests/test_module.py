import itertools
from functools import partial
from typing import *

import hypothesis
import hypothesis.strategies as st
import pytest
import torch
from hypothesis import note
from torch.testing import assert_close

import interp.circuit.testing.strategies as st_c
import rust_circuit as rc
import rust_circuit.module_library as mod_l
from interp.circuit.print_circuit import PrintCircuit
from interp.circuit.testing.topdown_circuit import CircuitProbs as CP
from interp.circuit.testing.utils import mark_not_interesting_if, rand_matcher
from interp.tools.indexer import I
from interp.tools.rrfs import RRFS_DIR
from rust_circuit import (
    Add,
    Array,
    Circuit,
    Concat,
    ConstructConcatShapeDifferentError,
    ConstructEinsumAxisSizeDifferentError,
    DiscreteVar,
    Einsum,
    ExpandConcatAxisError,
    Expander,
    ExpandFixedIndexError,
    ExpandGeneralFunctionTriedToBatchNonBatchableInputError,
    ExtractSymbolsBatchedInputError,
    ExtractSymbolsBoundInputInconsistentError,
    ExtractSymbolsHasBindingsFromOuterModuleError,
    GeneralFunction,
    Index,
    IterativeMatcher,
    Matcher,
    MiscInputNotBroadcastableError,
    Module,
    ModuleArgSpec,
    ModulePusher,
    ModuleSpec,
    Parser,
    Rearrange,
    RearrangeSpec,
    RunDiscreteVarAllSpec,
    Sampler,
    Scalar,
    Shape,
    Symbol,
    TorchDeviceDtypeOp,
    all_children,
    any_children_with_symbolic_sizes,
    broadcast_shapes,
    cast_circuit,
    conform_all_modules,
    count_nodes,
    deep_map_preorder,
    default_nested_module_namer,
    elim_no_input_module,
    expand_node,
    extract_rewrite,
    extract_rewrite_raw,
    extract_symbols,
    extract_symbols_get,
    get_free_symbols,
    module_new_bind,
    module_remove_unused_inputs,
    new_traversal,
    replace_expand_bottom_up_dict,
    strip_names_and_tags,
    substitute_all_modules,
    symbolic_sizes,
)
from rust_circuit.interop_rust import rust_to_py
from rust_circuit.module_library import get_attention, get_bilinear_mlp, get_pointwise_mlp
from rust_circuit.py_utils import timed

from .test_rust_rewrite import get_c_st

P = Parser()


@hypothesis.given(get_c_st(rust=True))
def test_subst_all_modules(circ):
    circ1 = rc.substitute_all_modules(circ)
    assert not any("internal_expand" in x.name for x in all_children(circ1))
    assert_close(circ.evaluate(), circ1.evaluate())


@pytest.mark.xfail
@hypothesis.given(
    get_c_st(
        rust=True, probs_default=CP.kw(all=1, Module=20, Symbol=20, Cumulant=0), from_other=False, max_growth_steps=20
    ),
    st.data(),
)
@mark_not_interesting_if(rc.PushDownModulePushingPastModuleWhichOverridesSymError)
def test_push_down_modules(circ, d):
    skips = d.draw(st_c.st_subsets([x.name for x in all_children(circ) if x.is_module()]))
    cb = d.draw(st.builds(rc.ModulePusher.remove_and_elim_callback, st.booleans(), st.booleans(), st.booleans()))
    pu = rc.ModulePusher(flatten_modules=d.draw(st.booleans()), module_construct_callback=cb)
    note(circ.repr())
    circ1 = pu.push_down_modules(circ, traversal=rc.traversal, skip_module=skips)
    note(circ1.repr())
    assert_close(circ.evaluate(), circ1.evaluate())


@pytest.mark.xfail
@hypothesis.given(
    get_c_st(
        rust=True, probs_default=CP.kw(all=1, Module=20, Symbol=20, Cumulant=0), from_other=False, max_growth_steps=5
    ),
    st.data(),
)
@mark_not_interesting_if(rc.ExtractSymbolsHasBindingsFromOuterModuleError)
def test_extract_symbols(circ, d):
    note(circ.repr())
    syms = d.draw(st_c.st_subsets([x for x in all_children(circ) if x.is_symbol()]))
    conform = d.draw(st.booleans())
    try:
        circ1 = rc.extract_symbols(
            circ, syms, use_elim_no_input_modules=d.draw(st.booleans()), conform_batch_if_needed=conform
        )
    except rc.ExtractSymbolsBatchedInputError:
        hypothesis.assume(conform)
        raise
    note(circ1.repr())
    assert_close(circ.evaluate(), circ1.evaluate())


@pytest.mark.xfail
@hypothesis.given(
    get_c_st(
        rust=True, probs_default=CP.kw(all=1, Module=20, Symbol=20, Cumulant=0), from_other=False, max_growth_steps=10
    ),
    st.data(),
)
def test_upd_nested_id(circ, d):
    note(circ.repr())
    try:
        circ1 = rc.update_bindings_nested(
            circ,
            lambda x: x if d.draw(st.booleans()) else None,
            rand_matcher(d),
            run_update_on_new_spec_circuits=d.draw(st.booleans()),
            flatten_modules=d.draw(st.booleans()),
        )
    except rc.ExceptionWithRustContext as e:
        raise e.exc
    assert_close(circ.evaluate(), circ1.evaluate())


def test_expand_node():
    circ: Circuit = Rearrange(
        Symbol.new_with_none_uuid((2, 3), "0"), RearrangeSpec([[0], [1]], [[1], [0]], [None, None])
    )
    expanded = expand_node(circ, [Array.randn(5, 6)])
    assert expanded.shape == (6, 5)
    print(expanded)

    circ: Circuit = Rearrange(
        Symbol.new_with_none_uuid((2, 3), "0"), RearrangeSpec([[0], [1]], [[1], [0]], [None, None])
    )
    expanded = expand_node(circ, [Array.randn(7, 5, 6)])
    assert expanded.shape == (7, 6, 5)
    print(expanded)

    circ: Circuit = Rearrange(
        Symbol.new_with_none_uuid((2, 3), "0"), RearrangeSpec([[0], [1]], [[1], [0]], [None, None])
    )
    expanded = expand_node(circ, [Array.randn(8, 7, 5, 6)])
    assert expanded.shape == (8, 7, 6, 5)
    print(expanded)

    circ = Index(Symbol.new_with_none_uuid((2, 3), "0"), I[:, 0])
    expanded = expand_node(circ, [Array.randn(5, 3)])
    assert expanded.shape == (5,)
    print(expanded)

    circ = Index(Symbol.new_with_none_uuid((2, 3), "0"), I[:, 0])
    expanded = expand_node(circ, [Array.randn(7, 5, 3)])
    assert expanded.shape == (7, 5)
    print(expanded)

    circ = Index(Symbol.new_with_none_uuid((2, 3), "0"), I[:, 0])
    expanded = expand_node(circ, [Array.randn(8, 7, 5, 3)])
    assert expanded.shape == (8, 7, 5)
    print(expanded)

    with pytest.raises(ExpandFixedIndexError):
        circ = Index(Symbol.new_with_none_uuid((2, 3), "0"), I[0:2, 0])
        expanded = expand_node(circ, [Array.randn(5, 3)])
        print(expanded)

    circ = Add(Symbol.new_with_none_uuid((2, 3, 1), "0"), Symbol.new_with_none_uuid((2, 1, 5), "1"))
    expanded = expand_node(
        circ, [Symbol.new_with_none_uuid((7, 3, 1), "0"), Symbol.new_with_none_uuid((7, 1, 11), "1")]
    )
    assert expanded.shape == (7, 3, 11)
    print(expanded)

    circ = Add(Symbol.new_with_none_uuid((2, 3, 1), "0"), Symbol.new_with_none_uuid((2, 1, 5), "1"))
    expanded = expand_node(
        circ, [Symbol.new_with_none_uuid((13, 7, 3, 1), "0"), Symbol.new_with_none_uuid((7, 1, 11), "1")]
    )
    assert expanded.shape == (13, 7, 3, 11)
    print(expanded)

    circ = Add(Symbol.new_with_none_uuid((2, 3, 1), "0"), Symbol.new_with_none_uuid((2, 1, 5), "1"))
    with pytest.raises(MiscInputNotBroadcastableError):
        expand_node(
            circ, [Symbol.new_with_none_uuid((13, 7, 3, 1), "0"), Symbol.new_with_none_uuid((1, 7, 1, 11), "1")]
        )

    circ = Add(Symbol.new_with_none_uuid((2, 3, 1), "0"), Symbol.new_with_none_uuid((2, 1, 5), "1"))
    expanded = expand_node(
        circ, [Symbol.new_with_none_uuid((4, 13, 7, 3, 1), "0"), Symbol.new_with_none_uuid((7, 1, 11), "1")]
    )
    assert expanded.shape == (4, 13, 7, 3, 11)
    print(expanded)

    circ = Add(Symbol.new_with_none_uuid((2, 3, 1), "0"), Symbol.new_with_none_uuid((2, 1, 5), "1"))
    expanded = expand_node(
        circ, [Symbol.new_with_none_uuid((7, 3, 1), "0"), Symbol.new_with_none_uuid((4, 13, 7, 1, 11), "1")]
    )
    assert expanded.shape == (4, 13, 7, 3, 11)
    print(expanded)

    circ = Add(Symbol.new_with_none_uuid((3, 1), "0"), Symbol.new_with_none_uuid((2, 1, 5), "1"))
    expanded = expand_node(
        circ, [Symbol.new_with_none_uuid((7, 3, 1), "0"), Symbol.new_with_none_uuid((8, 7, 9, 3, 1), "1")]
    )
    assert expanded.shape == (8, 7, 9, 3, 1)
    print(expanded)

    circ = Add(Symbol.new_with_none_uuid((), "0"), Symbol.new_with_none_uuid((2, 1, 5), "1"))
    with pytest.raises(MiscInputNotBroadcastableError):
        expand_node(
            circ, [Symbol.new_with_none_uuid((7, 3, 1), "0"), Symbol.new_with_none_uuid((1, 3, 11, 9, 4, 5), "1")]
        )

    circ = Add(Symbol.new_with_none_uuid((), "0"), Symbol.new_with_none_uuid((2, 1, 5), "1"))
    with pytest.raises(MiscInputNotBroadcastableError):
        expand_node(
            circ, [Symbol.new_with_none_uuid((1, 3, 1), "0"), Symbol.new_with_none_uuid((1, 3, 11, 9, 4, 5), "1")]
        )

    circ = Add(Symbol.new_with_none_uuid((7,), "0"), Symbol.new_with_none_uuid((2, 7), "1"))
    expanded = expand_node(circ, [Symbol.new_with_none_uuid((7, 7), "0"), Symbol.new_with_none_uuid((7, 7, 7), "1")])
    assert expanded.shape == (7, 7, 7)
    print(expanded)

    circ = Add(Symbol.new_with_none_uuid((8, 7), "0"), Symbol.new_with_none_uuid((1, 1, 1, 1, 1, 7), "1"))
    expanded = expand_node(
        circ, [Symbol.new_with_none_uuid((7, 8, 9), "0"), Symbol.new_with_none_uuid((7, 1, 1, 1, 1, 1, 1), "1")]
    )
    assert expanded.shape == (7, 1, 1, 1, 1, 8, 9)
    print(expanded)

    circ = Add(Symbol.new_with_none_uuid((8, 7), "0"), Symbol.new_with_none_uuid((1, 1, 1, 1, 1, 7), "1"))
    expanded = expand_node(
        circ, [Symbol.new_with_none_uuid((2, 4, 7, 8, 9), "0"), Symbol.new_with_none_uuid((7, 1, 1, 1, 1, 1, 1), "1")]
    )
    assert expanded.shape == (2, 4, 7, 1, 1, 1, 1, 8, 9)
    print(expanded)

    circ = Add(Symbol.new_with_none_uuid((8, 7), "0"), Symbol.new_with_none_uuid((1, 1, 1, 1, 1, 7), "1"))
    expanded = expand_node(
        circ, [Symbol.new_with_none_uuid((7, 8, 9), "0"), Symbol.new_with_none_uuid((2, 4, 7, 1, 1, 1, 1, 1, 1), "1")]
    )

    circ = Einsum.from_einsum_string(
        "ab,bc->ac", Symbol.new_with_none_uuid((2, 3), "0"), Symbol.new_with_none_uuid((3, 5), "1")
    )
    expanded = expand_node(circ, [Symbol.new_with_none_uuid((5, 1, 2, 3), "0"), Symbol.new_with_none_uuid((3, 5), "1")])
    assert expanded.shape == (5, 1, *circ.shape)
    print(expanded)

    circ = Einsum.from_einsum_string(
        "ab,bc->ac", Symbol.new_with_none_uuid((2, 3), "0"), Symbol.new_with_none_uuid((3, 5), "1")
    )
    expanded = expand_node(
        circ, [Symbol.new_with_none_uuid((5, 1, 2, 3), "0"), Symbol.new_with_none_uuid((5, 1, 3, 5), "1")]
    )
    assert expanded.shape == (5, 1, *circ.shape)
    print(expanded)

    circ = Einsum.from_einsum_string(
        "ab,bc->ac", Symbol.new_with_none_uuid((2, 3), "0"), Symbol.new_with_none_uuid((3, 5), "1")
    )
    expanded = expand_node(
        circ, [Symbol.new_with_none_uuid((5, 7, 2, 3), "0"), Symbol.new_with_none_uuid((7, 3, 5), "1")]
    )
    assert expanded.shape == (5, 7, *circ.shape)
    print(expanded)

    circ = Einsum.from_einsum_string(
        "ab,bc->ac", Symbol.new_with_none_uuid((2, 3), "0"), Symbol.new_with_none_uuid((3, 5), "1")
    )
    expanded = expand_node(
        circ, [Symbol.new_with_none_uuid((5, 7, 2, 12), "0"), Symbol.new_with_none_uuid((7, 12, 11), "1")]
    )
    assert expanded.shape == (5, 7, 2, 11)
    print(expanded)

    with pytest.raises(ConstructEinsumAxisSizeDifferentError):
        circ = Einsum.from_einsum_string(
            "ab,bc->ac", Symbol.new_with_none_uuid((2, 3), "0"), Symbol.new_with_none_uuid((3, 5), "1")
        )
        expanded = expand_node(
            circ, [Symbol.new_with_none_uuid((5, 2, 2, 2), "0"), Symbol.new_with_none_uuid((5, 2, 3, 5), "1")]
        )
        print(expanded)

    with pytest.raises(MiscInputNotBroadcastableError):
        circ = Einsum.from_einsum_string(
            "ab,bc->ac", Symbol.new_with_none_uuid((2, 3), "0"), Symbol.new_with_none_uuid((3, 5), "1")
        )
        expanded = expand_node(
            circ, [Symbol.new_with_none_uuid((5, 3, 2, 3), "0"), Symbol.new_with_none_uuid((5, 2, 3, 5), "1")]
        )

    with pytest.raises(ExpandConcatAxisError):
        circ = Concat(Symbol.new_with_none_uuid((2, 3), "0"), Symbol.new_with_none_uuid((4, 3), "1"), axis=0)
        expanded = expand_node(circ, [Symbol.new_with_none_uuid((3, 3), "0"), Symbol.new_with_none_uuid((4, 3), "1")])

    with pytest.raises(MiscInputNotBroadcastableError):
        circ = Concat(Symbol.new_with_none_uuid((2, 3), "0"), Symbol.new_with_none_uuid((3, 3), "1"), axis=0)
        expanded = expand_node(
            circ, [Symbol.new_with_none_uuid((2, 2, 3), "0"), Symbol.new_with_none_uuid((3, 3, 3), "1")]
        )

    with pytest.raises(ConstructConcatShapeDifferentError):
        circ = Concat(Symbol.new_with_none_uuid((2, 3), "0"), Symbol.new_with_none_uuid((3, 3), "1"), axis=0)
        expanded = expand_node(
            circ, [Symbol.new_with_none_uuid((3, 2, 2), "0"), Symbol.new_with_none_uuid((3, 3, 3), "1")]
        )

    circ = Concat(Symbol.new_with_none_uuid((2, 3), "0"), Symbol.new_with_none_uuid((4, 3), "1"), axis=0)
    expanded = expand_node(circ, [Symbol.new_with_none_uuid((5, 2, 3), "0"), Symbol.new_with_none_uuid((5, 4, 3), "1")])
    assert expanded.shape == (5, *circ.shape)
    print(expanded)

    circ = Concat(Symbol.new_with_none_uuid((2, 3), "0"), Symbol.new_with_none_uuid((4, 3), "1"), axis=0)
    expanded = expand_node(circ, [Symbol.new_with_none_uuid((5, 2, 3), "0"), Symbol.new_with_none_uuid((4, 3), "1")])
    assert expanded.shape == (5, *circ.shape)
    print(expanded)

    circ = Concat(Symbol.new_with_none_uuid((2, 3), "0"), Symbol.new_with_none_uuid((4, 3), "1"), axis=0)
    expanded = expand_node(circ, [Symbol.new_with_none_uuid((2, 3), "0"), Symbol.new_with_none_uuid((7, 3, 4, 3), "1")])
    assert expanded.shape == (7, 3, *circ.shape)
    print(expanded)

    circ = Concat(Symbol.new_with_none_uuid((2, 3), "0"), Symbol.new_with_none_uuid((4, 3), "1"), axis=0)
    expanded = expand_node(
        circ, [Symbol.new_with_none_uuid((3, 2, 3), "0"), Symbol.new_with_none_uuid((7, 3, 4, 3), "1")]
    )
    assert expanded.shape == (7, 3, *circ.shape)
    print(expanded)

    with pytest.raises(ExpandGeneralFunctionTriedToBatchNonBatchableInputError):
        circ = GeneralFunction.gen_index(
            Symbol.new_with_none_uuid((2, 3), "0"), Symbol.new_with_none_uuid((4, 3), "1"), index_dim=-1
        )
        expanded = expand_node(
            circ, [Symbol.new_with_none_uuid((3, 2, 3), "0"), Symbol.new_with_none_uuid((4, 3), "1")]
        )

    circ = GeneralFunction.gen_index(
        Symbol.new_with_none_uuid((2, 3), "0"), Symbol.new_with_none_uuid((4, 3), "1"), index_dim=-1
    )
    expanded = expand_node(circ, [Symbol.new_with_none_uuid((2, 3), "0"), Symbol.new_with_none_uuid((7, 4, 3), "1")])
    assert expanded.shape == (7, 4, 3, 2)

    circ = GeneralFunction.gen_index(
        Symbol.new_with_none_uuid((2, 3), "0"), Symbol.new_with_none_uuid((4, 3), "1"), index_dim=-1
    )
    expanded = expand_node(
        circ, [Symbol.new_with_none_uuid((2, 3), "0"), Symbol.new_with_none_uuid((7, 11, 4, 3), "1")]
    )
    assert expanded.shape == (7, 11, 4, 3, 2)

    circ = GeneralFunction.gen_index(
        Symbol.new_with_none_uuid((4, 3, 2, 3), "0"), Symbol.new_with_none_uuid((4, 3), "1"), index_dim=-1, batch_x=True
    )
    expanded = expand_node(
        circ, [Symbol.new_with_none_uuid((4, 3, 2, 3), "0"), Symbol.new_with_none_uuid((7, 11, 4, 3), "1")]
    )
    expanded.get_unique("0 rep_for_batch")  # TODO: maybe rename me!
    assert expanded.shape == (7, 11, 4, 3, 2)

    s = """
    'rep_fst' [2s, 3s] Symbol 0bac3288-00e7-41bf-bd59-ac04de67ba47
    'rep_snd' [4s, 5s] Symbol d5f29a2c-5927-4237-a7ea-7fe7ec2c3be6
    'a' Module
      'add' Add
        'arg0' [] Symbol 7d2c48d4-6110-4759-be98-de15bfb20af2
        'arg1' [] Symbol 86b517ef-b83c-4ea0-9568-f89bb2106db9
      'fst' [0s, 1s] Symbol c41b44e7-4f04-4124-bf02-34e0133c7c19 ! 'arg0'
      'snd' [0s, 1s] Symbol 823c2e81-99a2-46b3-bec1-329109e06e20 ! 'arg1'
    """
    rep_fst, rep_snd, c = Parser().parse_circuits(s)
    c = c.cast_module()
    out = expand_node(c, [c.spec.circuit, rep_fst, rep_snd])
    assert out.shape[0] in [rep_fst.shape[0], rep_snd.shape[0]]
    assert out.shape[1] in [rep_fst.shape[1], rep_snd.shape[1]]

    s = """
    'rep_fst' [3s] Symbol 0bac3288-00e7-41bf-bd59-ac04de67ba47
    'rep_snd' [8, 5s] Symbol d5f29a2c-5927-4237-a7ea-7fe7ec2c3be6
    'a' Module
      'add' Add
        'arg0' [] Symbol 7d2c48d4-6110-4759-be98-de15bfb20af2
        'arg1' [] Symbol 86b517ef-b83c-4ea0-9568-f89bb2106db9
      'fst' [2] Symbol c41b44e7-4f04-4124-bf02-34e0133c7c19 ! 'arg0'
      'snd' [3, 2] Symbol 823c2e81-99a2-46b3-bec1-329109e06e20 ! 'arg1'
    """
    rep_fst, rep_snd, c = Parser().parse_circuits(s)
    c = c.cast_module()
    out = expand_node(c, [c.spec.circuit, rep_fst, rep_snd])
    assert out.shape[0] in [rep_snd.shape[0]]
    assert out.shape[1] in [rep_fst.shape[0], rep_snd.shape[1]]

    s = """
    'rep_fst' [7, 9, 3s] Symbol 0bac3288-00e7-41bf-bd59-ac04de67ba47
    'rep_snd' [8, 5s] Symbol d5f29a2c-5927-4237-a7ea-7fe7ec2c3be6
    'a' Module
      'add' Add
        'arg0' [0s] Symbol 7d2c48d4-6110-4759-be98-de15bfb20af2
        'arg1' [0s] Symbol 86b517ef-b83c-4ea0-9568-f89bb2106db9
      'fst' [7s] Symbol c41b44e7-4f04-4124-bf02-34e0133c7c19 ! 'arg0'
      'snd' [12s, 7s] Symbol 823c2e81-99a2-46b3-bec1-329109e06e20 ! 'arg1'
    'b' Module
      'a'
      'rep_fst' ! 'fst'
      'rep_snd' ! 'snd'
    """
    rep_fst, rep_snd, c, b_mod = Parser().parse_circuits(s)
    assert b_mod.shape[:3] == (7, 9, 8)
    assert b_mod.shape[3] in [rep_fst.shape[-1], rep_snd.shape[-1]]
    sub = b_mod.cast_module().substitute()
    assert sub.shape[:3] == (7, 9, 8)
    assert sub.shape[3] in [rep_fst.shape[-1], rep_snd.shape[-1]]

    for sym_on_top in [False, True]:
        s = f"""
        'rep_fst' [12, {"7s" if sym_on_top else 8}, 9, 7] Symbol 0bac3288-00e7-41bf-bd59-ac04de67ba47
        'rep_snd' [13, 12, 9, 10, {8 if sym_on_top else "7s"}, 7] Symbol d5f29a2c-5927-4237-a7ea-7fe7ec2c3be6
        'a' Module
          'add' Add
            'arg0' [1s, 0s] Symbol 7d2c48d4-6110-4759-be98-de15bfb20af2
            'arg1' [0s] Symbol 86b517ef-b83c-4ea0-9568-f89bb2106db9
          'fst' [21s, 24s, 20s] Symbol c41b44e7-4f04-4124-bf02-34e0133c7c19 ! 'arg0'
          'snd' [23s, 22s, 21s, 20s] Symbol 823c2e81-99a2-46b3-bec1-329109e06e20 ! 'arg1'
        'b' Module
          'a'
          'rep_fst' ! 'fst'
          'rep_snd' ! 'snd'
        """
        rep_fst, rep_snd, c, out = Parser().parse_circuits(s)
        out.print()
        out.cast_module().substitute().print()
        assert out.shape == (13, 12, 9, 10, 8, 9, 7)
        assert out.cast_module().substitute().shape == (13, 12, 9, 10, 8, 9, 7)

    s = """
    'rep_fst' [    12,      102, 7] Symbol 0bac3288-00e7-41bf-bd59-ac04de67ba47
    'rep_snd' [13, 12,  10, 102, 7] Symbol d5f29a2c-5927-4237-a7ea-7fe7ec2c3be6
    'a' Module
      'add' Add
        'arg0' [0s] Symbol 7d2c48d4-6110-4759-be98-de15bfb20af2
        'arg1' [0s] Symbol 86b517ef-b83c-4ea0-9568-f89bb2106db9
      'fst' [     21s, 20s] Symbol c41b44e7-4f04-4124-bf02-34e0133c7c19 ! 'arg0'
      'snd' [22s, 21s, 20s] Symbol 823c2e81-99a2-46b3-bec1-329109e06e20 ! 'arg1'
    'b' Module
      'a'
      'rep_fst' ! 'fst'
      'rep_snd' ! 'snd'
    """
    rep_fst, rep_snd, c, out = Parser().parse_circuits(s)
    out.print()
    out.cast_module().substitute().print()
    assert out.shape == (13, 12, 10, 102, 7)
    assert out.cast_module().substitute().shape == (13, 12, 10, 102, 7)


def test_module_shape_missing():
    s0, s1, *_ = symbolic_sizes()
    spec = ModuleSpec(
        Scalar(0.0),
        [ModuleArgSpec(Symbol.new_with_none_uuid((s0,), "a")), ModuleArgSpec(Symbol.new_with_none_uuid((s1,), "b"))],
        check_all_inputs_used=False,
    )

    c = Module(spec, a=Symbol.new_with_none_uuid((4, 5, 6, 7)), b=Symbol.new_with_none_uuid((6, 4)))

    assert c.shape == (4, 5, 6)
    assert c.substitute().shape == (4, 5, 6)
    c.substitute().print()

    c = Module(spec, a=Symbol.new_with_none_uuid((4, 5, 6, 7)), b=Symbol.new_with_none_uuid((4,)))
    assert c.shape == (4, 5, 6)
    assert c.substitute().shape == (4, 5, 6)
    c.substitute().print()

    c = Module(spec, a=Symbol.new_with_none_uuid((7,)), b=Symbol.new_with_none_uuid((4, 5, 6, 4)))
    assert c.shape == (4, 5, 6)
    assert c.substitute().shape == (4, 5, 6)
    c.substitute().print()

    spec = ModuleSpec(
        Scalar(0.0, shape=(7,)),
        [ModuleArgSpec(Symbol.new_with_none_uuid((s0,), "a")), ModuleArgSpec(Symbol.new_with_none_uuid((s1,), "b"))],
        check_all_inputs_used=False,
    )

    c = Module(spec, a=Symbol.new_with_none_uuid((7,)), b=Symbol.new_with_none_uuid((4, 5, 6, 4)))
    assert c.shape == (4, 5, 6, 7)
    assert c.substitute().shape == (4, 5, 6, 7)
    c.substitute().print()


def test_module_nodes():
    spec = ModuleSpec(
        Einsum.from_einsum_string(
            "ab,bc->ac", Symbol.new_with_none_uuid((2, 3), "0"), Symbol.new_with_none_uuid((3, 5), "1")
        ),
        [
            ModuleArgSpec(Symbol.new_with_none_uuid((2, 3), "0"), True, True),
            ModuleArgSpec(Symbol.new_with_none_uuid((3, 5), "1"), True, True),
        ],
    )
    modulenode = Module.new_flat(spec, Array.randn(2, 3), Array.randn(3, 5))
    print((modulenode))


def test_module_nodes_2():
    spec = get_pointwise_mlp().spec
    spec.circuit.print()
    modulenode = Module.new_flat(
        spec, Array.randn(256), Array.randn(256 * 4, 256), Array.randn(256 * 4), Array.randn(256, 256 * 4)
    )
    modulenode.print()

    spec = get_bilinear_mlp(output_bias=False).spec
    spec.circuit.print()
    modulenode = Module(
        spec,
        "modulename",
        **{
            "m.input": Array.randn(256, name="inputy"),
            "m.w.proj_in": Array.randn(256 * 4, 256, name="project_in_y"),
            "m.w.in_bias": Array.randn(256 * 4, name="bias_in_y"),
            "m.w.proj_out": Array.randn(256, 256 * 2, name="project_out_y"),
        },
    )
    modulenode.print()
    modulenode.substitute().print()

    spec = get_attention().spec
    spec.circuit.print()
    modulenode = Module(
        spec,
        **{
            "a.w.q": Array.randn(4, 16, 64),
            "a.w.k": Array.randn(4, 16, 64),
            "a.w.v": Array.randn(4, 16, 64),
            "a.w.o": Array.randn(4, 64, 16),
            "a.mask": Array.randn(5, 5),
            "a.input": Array.randn(5, 64),
        },
    )
    modulenode.print()
    modulenode.substitute().print()
    substitute_all_modules(modulenode).print()


def test_bind():
    proj_in, in_bias, proj_out, inp = Array.randn(64, 16), Array.randn(64), Array.randn(16, 64), Array.randn(16)
    gelu_mlp_free_body = get_pointwise_mlp().body
    m = module_new_bind(gelu_mlp_free_body, ("m.w.proj_in", proj_in))
    m = module_new_bind(m, ("m.w.proj_out", proj_out), ("m.input", inp))
    m = module_new_bind(m, ("m.w.in_bias", in_bias))
    torch.testing.assert_close(
        m.evaluate(),
        Module(
            ModuleSpec.new_free_symbols(gelu_mlp_free_body),
            name="g",
            **dict([("m.w.proj_in", proj_in), ("m.w.in_bias", in_bias), ("m.w.proj_out", proj_out), ("m.input", inp)]),
        ).evaluate(),
    )


def test_symbolic_size_constraints():
    s0, s1, *_ = symbolic_sizes()

    circ = Add(
        Symbol.new_with_random_uuid((s0, s1), name="first"), Symbol.new_with_random_uuid((s0, s1), name="second")
    )
    assert circ.symbolic_size_constraints == set()

    circ_replaced = Expander(
        ("first", lambda _: Scalar(0.5, (s0, 3), "first_input")),
    )(circ, fancy_validate=True)
    assert len(circ_replaced.symbolic_size_constraints) == 1
    constraint = circ_replaced.symbolic_size_constraints.pop()
    assert constraint.l.other_factor == 1 and constraint.l.symbolic_sizes == [1]
    assert constraint.r.other_factor == 3 and constraint.r.symbolic_sizes == []


def test_replace_deep():
    replace_expand_bottom_up_dict(
        Einsum.from_einsum_string(
            "ab,bc->ac",
            Add(
                Symbol.new_with_none_uuid((2, 3), "0"),
                Symbol.new_with_none_uuid((2, 3), "0"),
            ),
            Symbol.new_with_none_uuid((3, 5), "1"),
        ),
        {Symbol.new_with_none_uuid((2, 3), "0"): Symbol.new_with_none_uuid((100, 3), "0")},
    ).print()


def test_extract():
    ispecs = [
        (Array.randn(2, 3), ModuleArgSpec(Symbol.new_with_none_uuid((2, 3), "0"), True, True)),
        (Array.randn(3, 5), ModuleArgSpec(Symbol.new_with_none_uuid((3, 5), "1"), True, True)),
    ]
    circ = Einsum.from_einsum_string(
        "ab,bc->ac",
        Add(
            ispecs[0][0],
            ispecs[0][0],
        ),
        ispecs[1][0],
    )
    circ.print()
    module_spec = ModuleSpec.new_extract(circ, cast(Any, ispecs))
    module_spec.circuit.print()


@pytest.mark.skip("rrfs stuff lazy ci")
def test_extract_gpt():
    with timed("load"):
        orig = Parser(on_repeat_check_info_same=False)(
            open(RRFS_DIR + "/tao/circuits/gpt2_small_300_blank.circ").read()
        )
    py_version = rust_to_py(orig)
    PrintCircuit(print_html=True)(py_version)
    # with timed("print"):
    #     orig.print()
    cur = orig
    for i in range(12):
        matcher = Matcher.any(Matcher(Array), Matcher(f"a{i}.inp"))
        children = list(matcher.get(cur))
        # print(children)
        children.sort(key=lambda x: x.name)
        cur = Matcher(f"a{i}").update(
            cur,
            lambda x: extract_rewrite_raw(
                x, [(c, ModuleArgSpec.just_name_shape(c)) for c in children], f"a{i}.", f"a{i}"
            ),
            fancy_validate=True,
        )
        Matcher(f"a{i}").update(
            cur,
            lambda x: extract_rewrite(
                x,
                Matcher.any(Matcher(Array), Matcher(f"a{i}.inp")),
                f"a{i}.",
                f"a{i}",
                circuit_to_arg_spec=ModuleArgSpec.just_name_shape,
            ),
            fancy_validate=True,
        )
        # print("Traversal")
        # cur_2.print()
        # print("Raw")
        # cur.print()
        # print(cur.hash, cur_2.hash)
        # torch.testing.assert_close(cur.evaluate(), cur_2.evaluate())
        # # assert cur == cur_2

        matcher = Matcher.any(Matcher.all(Matcher(r"^m\d."), Matcher(Array)), Matcher(f"m{i}.inp"))
        children = list(matcher.get(cur))
        children.sort(key=lambda x: x.name)
        cur = Matcher(f"m{i}").update(
            cur,
            lambda x: extract_rewrite_raw(
                x,
                [(c, ModuleArgSpec.just_name_shape(c)) for c in children],
                f"m{i}.",
                f"m{i}",
            ),
            fancy_validate=True,
        )
    cur.print()
    a0s = Matcher("a0").get_unique(cur).cast_module().spec.circuit
    a0s.print()
    a1s = Matcher("a1").get_unique(cur).cast_module().spec.circuit
    a1s.print()
    print(a0s == a1s)


def raw_test_full_transformer(
    params: mod_l.TransformerBlockParams,
):
    t, _, _ = mod_l.TransformerParams(params, num_layers=7).garbage_call(
        head_size=3,
        num_heads=5,
        seq_len=11,
        batch_shape=(2, 3),
    )
    with timed("conform"):
        conformed = conform_all_modules(t)
    assert not any_children_with_symbolic_sizes(conformed)
    with timed("preorder inline"):
        deep_map_preorder(t, lambda x: x.substitute() if isinstance(x, Module) else x)
    t = strip_names_and_tags(t)
    with timed("inline"):
        t = substitute_all_modules(t)
    print(count_nodes(t))


def test_full_transformer():
    Params = mod_l.TransformerBlockParams
    raw_test_full_transformer(Params("ln", False, False, True, "gelu", False))
    raw_test_full_transformer(Params("ln", False, False, True, "relu", False))
    raw_test_full_transformer(Params("ln", False, False, True, "bilinear", False))
    raw_test_full_transformer(Params("ln", False, False, True, "relu", True))
    raw_test_full_transformer(Params("ln", False, False, True, "bilinear", True))
    raw_test_full_transformer(Params(None, False, False, True, "gelu", False))
    raw_test_full_transformer(Params("ln", True, True, True, "gelu", False))
    raw_test_full_transformer(Params("ln", False, True, True, "gelu", False))
    raw_test_full_transformer(Params("ln", False, False, False))
    raw_test_full_transformer(Params(None, False, False, False))
    raw_test_full_transformer(Params("bn", False, False, True, "gelu", False))
    raw_test_full_transformer(Params("bn", False, False, False))


def test_symbolic_prime():
    s = symbolic_sizes()[0]
    for i in range(2, s):
        assert s % i != 0


def test_module_conform():
    spec = get_attention().spec
    module = Module(
        spec,
        **{
            "a.w.q": Array.randn(4, 16, 64),
            "a.w.k": Array.randn(4, 16, 64),
            "a.w.v": Array.randn(4, 16, 64),
            "a.w.o": Array.randn(4, 64, 16),
            "a.mask": Array.randn(5, 5),
            "a.input": Array.randn(5, 64),
        },
    )
    module.print()
    new_module = module.conform_to_input_shapes()
    new_module.print()

    torch.testing.assert_close(module.evaluate(), new_module.evaluate())


@pytest.mark.parametrize("separate_mod", [False, True])
def test_complex_batching(separate_mod: bool):
    r = partial(get_complex_module_test, separate_mod=separate_mod)

    r()
    r(batch_inner_0=(3,))
    r(batch_inner_0=(3, 7))
    r(batch_inner_1=(3,))
    r(batch_inner_1=(3, 7))
    r(batch_inner_0=(3,), batch_inner_1=(3,))
    r(batch_inner_0=(7,), batch_inner_1=(7,))
    r(batch_inner_0=(3,), batch_inner_1=(7, 3))
    r(batch_inner_0=(7, 3), batch_inner_1=(7, 3))
    r(batch_inner_0=(7, 3), batch_inner_1=(3,))
    r(batch_inner_0=(1, 7, 3), batch_inner_1=(3,))

    r(batch_outer_0=(3,))
    r(batch_outer_0=(3, 7))
    r(batch_outer_1=(3,))
    r(batch_outer_1=(3, 7))
    r(batch_outer_0=(3,), batch_outer_1=(3,))
    r(batch_outer_0=(7,), batch_outer_1=(7,))
    r(batch_outer_0=(3,), batch_outer_1=(7, 3))
    r(batch_outer_0=(7, 3), batch_outer_1=(7, 3))
    r(batch_outer_0=(7, 3), batch_outer_1=(3,))
    r(batch_outer_0=(1, 7, 3), batch_outer_1=(3,))

    r(batch_inner_0=(3,), batch_outer_0=(3,))
    r(batch_inner_0=(3,), batch_outer_0=(4,))
    r(batch_inner_0=(3, 7), batch_outer_0=(4,), batch_outer_1=(6, 4))
    _, _, outer = r(batch_inner_0=(3, 7), batch_inner_1=(7,), batch_outer_0=(4,), batch_outer_1=(6, 4))
    outer.cast_module().substitute().print()
    r(batch_inner_1=(7,), batch_outer_0=(4,), batch_outer_1=(6, 4))
    r(batch_inner_1=(3,), batch_outer_0=(4,))
    r(batch_inner_1=(3,), batch_outer_1=(4,))
    r(batch_inner_0=(3,), batch_outer_1=(4,))
    r(batch_inner_0=(3,), batch_outer_1=(6, 4))


def get_complex_module_test(
    batch_inner_0: Shape = (),
    batch_inner_1: Shape = (),
    batch_outer_0: Shape = (),
    batch_outer_1: Shape = (),
    separate_mod: bool = False,
):
    alpha = [chr(i) for i in range(ord("f"), ord("z"))]
    b_inner_alpha = "".join(alpha[: len(batch_inner_0)])
    s = f"""
    'inner' Einsum az,bz,cl->bac
      'inner.sym_arg0' [0s, 4s] Symbol a409cdc3-d962-4df1-8d1b-c5c8d8b50639
      'inner.concat' Concat 0
        'inner.sym_arg1' [3, 4s] Symbol 02927290-fc4f-45b7-9822-267918e897d6
        'outer.sym_arg0' [7, 4s] Symbol 6dcfaf1e-5e35-418a-ad88-3d79ae32096c
      'outer.sym_arg1' [3s, 5s] Symbol ecf273f5-82c5-4fbf-8097-d9b1a250e035

    'inner.call' Module
      'inner'
      'inner.apply_inner' Einsum cd,{b_inner_alpha}a->{b_inner_alpha}ca ! 'inner.sym_arg0'
        'outer.sym_arg0'
        'inner.rand0' [{",".join(map(str, batch_inner_0 + (5,)))}] Array rand
      'inner.rand1' [{",".join(map(str, batch_inner_1 + (3, 5)))}] Array rand ! 'inner.sym_arg1'

    'add_inner' Add
      'inner.call'
      'inner.call'

    'outer' Module
      '{"add_inner" if separate_mod else "inner.call"}'
      'outer.rand0' [{",".join(map(str, batch_outer_0 + (7, 5)))}] Array rand ! 'outer.sym_arg0'
      'outer.rand1' [{",".join(map(str, batch_outer_1 + (9, 12)))}] Array rand ! 'outer.sym_arg1'
    """

    inner, inner_call_v, _, outer_v = [
        cast_circuit(c, TorchDeviceDtypeOp(dtype="float64")) for c in Parser().parse_circuits(s)
    ]
    inner_call = inner_call_v.cast_module()
    outer = outer_v.cast_module()
    s0 = symbolic_sizes()[0]
    s3 = symbolic_sizes()[3]
    inner_batch = broadcast_shapes(batch_inner_0, batch_inner_1)
    outer_batch = broadcast_shapes(batch_outer_0, batch_outer_1)
    assert inner.shape == (10, s0, s3)
    assert inner_call.shape == (inner_batch + (10, 7, s3))
    assert outer.shape == (outer_batch + inner_batch + (10, 7, 9))

    base_value = outer.evaluate()

    def test_close(outer: Circuit):
        def sub(x: Circuit):
            out = x.cast_module().substitute()
            assert out.name == x.name
            return out

        torch.testing.assert_close(
            substitute_all_modules(outer).evaluate(), base_value
        )  # does same as base under current impl
        torch.testing.assert_close(
            outer.update("inner.call", sub).evaluate(), base_value
        )  # does same as base under current impl
        torch.testing.assert_close(sub(outer).evaluate(), base_value)

    test_close(outer)

    def get_rand_batcher(n: str, batch_size: int):
        probs = DiscreteVar.uniform_probs_and_group(batch_size, name="probs")

        def get_discrete_var_circ(s: str):
            circ = outer.get_unique(s)
            return DiscreteVar(circ, probs, name=circ.name + " var")

        return (
            Expander(*[(s, partial(lambda s, _: get_discrete_var_circ(s), s)) for s in [f"{n}.rand0", f"{n}.rand1"]]),
            probs,
        )

    if batch_inner_0 != () or batch_inner_1 != () or batch_outer_0 != () or batch_outer_1 != ():
        no_batch_inner, no_batch_inner_call, no_batch_outer = get_complex_module_test(separate_mod=separate_mod)

        def get_manual_batcher(n: str):
            return Expander(*[(s, partial(lambda s, _: outer.get_unique(s), s)) for s in [f"{n}.rand0", f"{n}.rand1"]])

        def batch(x: Circuit):
            return get_manual_batcher("outer")(get_manual_batcher("inner")(x))

        assert batch(no_batch_inner).shape == inner.shape
        assert batch(no_batch_inner_call).shape == inner_call.shape
        assert batch(no_batch_outer).shape == outer.shape

        test_close(batch(no_batch_outer))

        if batch_inner_0 == batch_inner_1 and len(batch_inner_0) == 1 and batch_outer_0 == () and batch_outer_1 == ():
            outer_batched = get_manual_batcher("outer")(no_batch_outer)
            batcher, probs = get_rand_batcher("inner", batch_inner_0[0])
            inner_rand = batcher(outer_batched)
            sampled = Sampler(RunDiscreteVarAllSpec([probs])).sample(inner_rand)
            torch.testing.assert_close(sampled.evaluate(), base_value)

        if batch_outer_0 == batch_outer_1 and len(batch_outer_0) == 1 and batch_inner_0 == () and batch_inner_1 == ():
            inner_batched = get_manual_batcher("inner")(no_batch_outer)
            batcher, probs = get_rand_batcher("outer", batch_outer_0[0])
            outer_rand = batcher(inner_batched)
            sampled = Sampler(RunDiscreteVarAllSpec([probs])).sample(outer_rand)
            torch.testing.assert_close(sampled.evaluate(), base_value)

    return inner, inner_call, outer


def test_elim_no_input_module():
    s = """
    'inside' [] Scalar 4
    'a' Module
      'inside'
    """
    inside, circ = Parser().parse_circuits(s)
    assert elim_no_input_module(circ.cast_module()) == inside

    s = """
    'inside' [] Scalar 4
    'a' Module
      'inside'
      'a.bound' [] Scalar 3 ! 'a.inp' [] Symbol 90e7ce1a-0d52-4845-b42e-82411d6cbf56

    """
    inside, circ = Parser(module_check_all_inputs_used=False).parse_circuits(s)
    assert elim_no_input_module(circ.cast_module()) is None


def test_remove_unused_inputs():
    s = """
    'inside' [] Scalar 4
    0 'a rem_unused' Module
      'inside'
    1 'a' Module
      'inside'
      'a.bound' [] Scalar 3 ! 'a.inp' [] Symbol 90e7ce1a-0d52-4845-b42e-82411d6cbf56

    """
    inside, removed_circ, circ = Parser(module_check_all_inputs_used=False).parse_circuits(s)
    assert module_remove_unused_inputs(circ.cast_module(), use_elim_no_input_module=True) == inside
    assert module_remove_unused_inputs(circ.cast_module(), use_elim_no_input_module=False) == removed_circ

    s = """
    'inside' [] Scalar 4
    1 'a' Module
      'inside'
      'a.bound' [2, 3] Scalar 3 ! 'a.inp' [] Symbol 90e7ce1a-0d52-4845-b42e-82411d6cbf56

    """
    inside, circ = Parser(module_check_all_inputs_used=False).parse_circuits(s)
    out_removed = module_remove_unused_inputs(circ.cast_module(), use_elim_no_input_module=True)
    assert len(out_removed.get(Matcher.types(Module, Symbol))) == 0
    assert out_removed.shape == (2, 3)
    assert (out_removed.evaluate() == circ.evaluate()).all()
    out_strip_no_drop = module_remove_unused_inputs(circ.cast_module(), use_elim_no_input_module=False)
    assert len(out_strip_no_drop.get(Matcher.types(Symbol))) == 0
    assert len(out_strip_no_drop.get(Matcher.types(Module))) == 1
    assert out_removed.shape == (2, 3)
    assert (out_removed.evaluate() == circ.evaluate()).all()

    s = """
    'a' Module
      'stuff' Add
        'a.inp' [] Symbol 90e7ce1a-0d52-4845-b42e-82411d6cbf56
        'other' [3] Scalar 5
      'a.bound' [2, 3] Scalar 3 ! 'a.inp'
    """
    circ = Parser().parse_circuit(s)
    assert module_remove_unused_inputs(circ.cast_module(), use_elim_no_input_module=True) == circ
    assert module_remove_unused_inputs(circ.cast_module(), use_elim_no_input_module=False) == circ

    fancy_test_remove_unused_inputs()
    fancy_test_remove_unused_inputs(unbound0_batch=(2,))
    fancy_test_remove_unused_inputs(unbound1_batch=(2,))
    fancy_test_remove_unused_inputs(unbound0_batch=(2,), unbound1_batch=(2,))
    fancy_test_remove_unused_inputs(unbound0_batch=(3, 2), unbound1_batch=(2,))
    fancy_test_remove_unused_inputs(unbound0_batch=(2,), unbound1_batch=(3, 2))
    fancy_test_remove_unused_inputs(unbound0_batch=(4, 3, 2), unbound1_batch=(2,))
    fancy_test_remove_unused_inputs(unbound0_batch=(2,), unbound1_batch=(4, 3, 2))

    fancy_test_remove_unused_inputs(bound0_batch=(2,))
    fancy_test_remove_unused_inputs(bound1_batch=(2,))
    fancy_test_remove_unused_inputs(bound2_batch=(2,))
    fancy_test_remove_unused_inputs(bound0_batch=(2,), bound1_batch=(2,))
    fancy_test_remove_unused_inputs(bound0_batch=(3, 2), bound1_batch=(2,))
    fancy_test_remove_unused_inputs(bound0_batch=(2,), bound1_batch=(3, 2))
    fancy_test_remove_unused_inputs(bound0_batch=(4, 3, 2), bound1_batch=(2,))
    fancy_test_remove_unused_inputs(bound0_batch=(2,), bound1_batch=(4, 3, 2))

    fancy_test_remove_unused_inputs(
        bound0_batch=(2,), bound1_batch=(4, 3, 2), unbound0_batch=(2,), unbound1_batch=(4, 3, 2)
    )
    fancy_test_remove_unused_inputs(
        bound0_batch=(2,), bound1_batch=(4, 3, 2), bound2_batch=(3, 2), unbound0_batch=(2,), unbound1_batch=(4, 3, 2)
    )
    fancy_test_remove_unused_inputs(
        bound0_batch=(2,), bound1_batch=(3, 2), bound2_batch=(2,), unbound0_batch=(2,), unbound1_batch=(4, 3, 2)
    )
    fancy_test_remove_unused_inputs(
        bound0_batch=(2,), bound1_batch=(3, 2), bound2_batch=(2,), unbound0_batch=(4, 3, 2), unbound1_batch=(2,)
    )


def fancy_test_remove_unused_inputs(
    unbound0_batch: Shape = (),
    unbound1_batch: Shape = (),
    bound0_batch: Shape = (),
    bound1_batch: Shape = (),
    bound2_batch: Shape = (),
):
    s = f"""
    'a' Module
      2 Einsum i,i,,ij->i
        'a.inp_bound0' [4] Symbol 7f6cd401-b2f0-4b2d-9967-4b0c2692bdfc
        'stuff' [4] Array rand
        'a.inp_bound1' [] Symbol e4c8dcf0-e3fb-44d7-857a-60c839b216cb
        'a.inp_bound2' [4, 5] Symbol 9be09e34-f084-435e-9297-51ffce4843ca

      'a.unbound0' [{",".join(map(str, unbound0_batch))}] Array rand ! 'a.inp_unbound0' [] Symbol 90e7ce1a-0d52-4845-b42e-82411d6cbf56
      'a.bound0' [{",".join(map(str, bound0_batch + (4,)))}] Array rand ! 'a.inp_bound0' [4] Symbol 7f6cd401-b2f0-4b2d-9967-4b0c2692bdfc
      'a.unbound1' [{",".join(map(str, unbound1_batch + (6, 7)))}] Array rand ! 'a.inp_unbound1' [6, 7] Symbol a87b33e7-85bc-412c-b8df-708821df0598
      'a.bound1' [{",".join(map(str, bound1_batch ))}] Array rand ! 'a.inp_bound1' [] Symbol e4c8dcf0-e3fb-44d7-857a-60c839b216cb
      'a.bound2' [{",".join(map(str, bound2_batch + (4, 5)))}] Array rand ! 'a.inp_bound2' [4, 5] Symbol 9be09e34-f084-435e-9297-51ffce4843ca
    """
    circ = Parser(module_check_all_inputs_used=False).parse_circuit(s)
    out_strip = module_remove_unused_inputs(circ.cast_module(), use_elim_no_input_module=True)
    out_strip.print()
    assert out_strip == module_remove_unused_inputs(circ.cast_module(), use_elim_no_input_module=False)
    assert len(out_strip.get(Matcher.types(Symbol))) == 3
    torch.testing.assert_close(out_strip.evaluate(), circ.evaluate())


def run_test_push_down_mod(
    batch_inner_0: Shape = (),
    batch_inner_1: Shape = (),
    batch_outer_0: Shape = (),
    batch_outer_1: Shape = (),
    separate_mod: bool = False,
):
    _, inner_call, outer = get_complex_module_test(
        batch_inner_0=batch_inner_0,
        batch_inner_1=batch_inner_1,
        batch_outer_0=batch_outer_0,
        batch_outer_1=batch_outer_1,
        separate_mod=separate_mod,
    )
    outer.print()

    # up to some minor difference in rearrange matching, push down is equiv to all sub
    push = ModulePusher().push_down_modules(outer, traversal=new_traversal())
    push.print()
    assert push.get(Symbol) == set()
    torch.testing.assert_close(push.evaluate(), outer.evaluate())

    for flatten_modules, remove_unused_inputs, elim_no_input_modules in itertools.product(*([[False, True]] * 3)):
        # none of these args should effect this case
        equiv_push = ModulePusher(
            flatten_modules=flatten_modules,
            module_construct_callback=ModulePusher.remove_and_elim_callback(
                remove_unused_inputs=remove_unused_inputs,
                elim_no_input_modules=elim_no_input_modules,
            ),
        ).push_down_modules(outer, traversal=new_traversal())
        assert equiv_push == push

    push = ModulePusher(flatten_modules=True).push_down_modules(
        outer, traversal=new_traversal(term_early_at="inner.concat")
    )
    push.print()
    assert len(push.get(Symbol)) == 2
    assert len(push.get(Module)) == 1
    assert sum(len(m.cast_module().spec.arg_specs) for m in push.get(Module)) == 2
    torch.testing.assert_close(push.evaluate(), outer.evaluate())

    push = ModulePusher(flatten_modules=False).push_down_modules(
        outer, traversal=new_traversal(term_early_at="inner.concat")
    )
    push.print()
    assert len(push.get(Symbol)) == 2
    assert len(push.get(Module)) == 2
    assert sum(len(m.cast_module().spec.arg_specs) for m in push.get(Module)) == 2
    torch.testing.assert_close(push.evaluate(), outer.evaluate())

    push = ModulePusher(
        flatten_modules=False, module_construct_callback=ModulePusher.noop_callback()
    ).push_down_modules(outer, traversal=new_traversal(term_early_at="inner.concat"))
    push.print()
    assert len(push.get(Symbol)) == 2
    assert len(push.get(Module)) == 2
    assert sum(len(m.cast_module().spec.arg_specs) for m in push.get(Module)) == 4
    torch.testing.assert_close(push.evaluate(), outer.evaluate())

    push = ModulePusher(flatten_modules=True).push_down_modules(
        outer, traversal=new_traversal(term_early_at="inner.concat")
    )
    push.print()
    assert len(push.get(Symbol)) == 2
    assert len(push.get(Module)) == 1
    assert sum(len(m.cast_module().spec.arg_specs) for m in push.get(Module)) == 2
    torch.testing.assert_close(push.evaluate(), outer.evaluate())

    assert (
        ModulePusher().push_down_modules(outer, traversal=new_traversal(), skip_module=IterativeMatcher(True)) == outer
    )
    assert (
        ModulePusher().push_down_modules(outer, traversal=new_traversal(), skip_module={"outer", "inner.call"}) == outer
    )

    push_skip_inner = ModulePusher().push_down_modules(outer, traversal=new_traversal(), skip_module={"inner.call"})
    push_skip_inner.print()
    torch.testing.assert_close(push_skip_inner.evaluate(), outer.evaluate())

    push_skip_inner = ModulePusher().push_down_modules(
        outer, traversal=new_traversal(term_early_at="inner.concat"), skip_module={"inner.call"}
    )
    push_skip_inner.print()
    torch.testing.assert_close(push_skip_inner.evaluate(), outer.evaluate())

    push = ModulePusher().push_down_modules(outer, traversal=new_traversal(), skip_module={"outer"})
    push.print()
    assert isinstance(push, Module)
    assert push.spec.arg_specs == outer.spec.arg_specs
    assert len(push.get(Symbol)) == 2
    assert len(push.get(Module)) == 1
    torch.testing.assert_close(push.evaluate(), outer.evaluate())

    if batch_inner_1 == ():
        out = extract_symbols(
            outer, {sp.symbol for sp in inner_call.spec.arg_specs if sp.symbol.name == "inner.sym_arg1"}
        )
        out.print()
        torch.testing.assert_close(out.evaluate(), outer.evaluate())

        out_new = extract_symbols_get(outer, "inner.sym_arg1")
        assert out == out_new

    if len(batch_inner_1) > 0:
        with pytest.raises(ExtractSymbolsBatchedInputError):
            extract_symbols_get(outer, "inner.sym_arg1")

    if batch_inner_0 == ():
        with pytest.raises(ExtractSymbolsHasBindingsFromOuterModuleError):
            extract_symbols_get(outer, "inner.sym_arg0")

    if batch_outer_0 == () and batch_outer_1 == ():
        outer.print()
        pushed = ModulePusher().push_down_modules(outer, traversal=new_traversal(end_depth=3))
        pushed.print()

        # TODO: maybe extract should elim not input modules itself?
        extracted = extract_symbols_get(pushed, {"outer.sym_arg0", "outer.sym_arg1"})
        undo_push = extracted.update(
            rc.restrict(Matcher.regex("^outer( |$)"), start_depth=1),
            lambda x: x.rename("add_inner" if separate_mod else "inner.call"),
        )
        undo_push.print()
        assert undo_push == outer

    push = ModulePusher().push_down_modules(outer, traversal=new_traversal(), skip_module={"inner.call"})
    push.print()
    assert len(push.get(Module)) == 1
    assert {s.symbol.name for s in push.get_unique(Module).cast_module().spec.arg_specs} == {
        s.symbol.name for s in inner_call.spec.arg_specs
    }
    assert len(push.get(Symbol)) == 2
    torch.testing.assert_close(push.evaluate(), outer.evaluate())


@pytest.mark.parametrize("separate_mod", [False, True])
def test_complex_push_down(separate_mod: bool):
    torch.manual_seed(238)

    r = partial(run_test_push_down_mod, separate_mod=separate_mod)

    r()
    r(batch_inner_0=(3,))
    r(batch_inner_0=(3, 7))
    r(batch_inner_1=(3,))
    r(batch_inner_1=(3, 7))
    r(batch_inner_0=(3,), batch_inner_1=(3,))
    r(batch_inner_0=(7,), batch_inner_1=(7,))
    r(batch_inner_0=(3,), batch_inner_1=(7, 3))
    r(batch_inner_0=(7, 3), batch_inner_1=(7, 3))
    r(batch_inner_0=(7, 3), batch_inner_1=(3,))
    r(batch_inner_0=(1, 7, 3), batch_inner_1=(3,))

    r(batch_outer_0=(3,))
    r(batch_outer_0=(3, 7))
    r(batch_outer_1=(3,))
    r(batch_outer_1=(3, 7))
    r(batch_outer_0=(3,), batch_outer_1=(3,))
    r(batch_outer_0=(7,), batch_outer_1=(7,))
    r(batch_outer_0=(3,), batch_outer_1=(7, 3))
    r(batch_outer_0=(7, 3), batch_outer_1=(7, 3))
    r(batch_outer_0=(7, 3), batch_outer_1=(3,))
    r(batch_outer_0=(1, 7, 3), batch_outer_1=(3,))

    r(batch_inner_0=(3,), batch_outer_0=(3,))
    r(batch_inner_0=(3,), batch_outer_0=(4,))
    r(batch_inner_0=(3, 1), batch_outer_0=(4,), batch_outer_1=(6, 4))
    r(batch_inner_0=(3, 2), batch_inner_1=(2,), batch_outer_0=(4,), batch_outer_1=(6, 4))
    r(batch_inner_1=(7,), batch_outer_0=(4,), batch_outer_1=(6, 4))
    r(batch_inner_1=(3,), batch_outer_0=(4,))
    r(batch_inner_1=(3,), batch_outer_1=(4,))
    r(batch_inner_0=(3,), batch_outer_1=(4,))
    r(batch_inner_0=(3,), batch_outer_1=(6, 4))


def test_extract_traversal():
    s = """
    'add' Add
      'a' Module
        'sym' [] Symbol c9517c8e-0b5a-4ebf-94c8-6da413c9f5d1
        'b.arg' [] Scalar 73.3874 ! 'sym'
      'b' Module
        'sym'
        'a.arg' [] Scalar 28.3877 ! 'sym'
    """
    circ = P(s)
    out = extract_symbols_get(circ, "sym", traversal=new_traversal(term_early_at="a"))
    assert circ.get_unique("a") == out.get_unique("a")
    with pytest.raises(ExtractSymbolsBoundInputInconsistentError):
        extract_symbols_get(circ, "sym")


def test_push_down_module_nested_update_arg():
    s = """
    'a' Module
      'b' Module
        'c' Module
          'add' Add
            'sym_a' [] Symbol 05c93da1-ac33-47eb-b215-d1a8e8438a02
            'sym_b' [] Symbol 5ae54c16-b570-4abd-ad37-f527c25494c1
            'sym_c' [] Symbol cb22836a-b5cb-4e65-969e-563958249e81
          'arg_c' [] Scalar 3 ! 'sym_c'
        'arg_b' Einsum ij,j->i ! 'sym_b'
          'within_arg' Module
            'add_within' Add
              'sym_d' [] Symbol fbb99b36-0809-4eae-ae10-7d38a90d1386
              'w' [3] Scalar 4
            'k' [7] Scalar 7 ! 'sym_d'
          'x' [3] Scalar 8
      'arg_a' [] Scalar 82.83 ! 'sym_a'
    """
    circ = P(s)
    circ.print()

    pushed = ModulePusher().push_down_modules(circ, traversal=new_traversal())
    torch.testing.assert_close(pushed.evaluate(), circ.evaluate())

    pushed = ModulePusher(flatten_modules=False).push_down_modules(circ, traversal=new_traversal())
    torch.testing.assert_close(pushed.evaluate(), circ.evaluate())

    pushed = ModulePusher(flatten_modules=False).push_down_modules(
        circ, traversal=new_traversal(term_early_at={"sym_a", "sym_d"})
    )
    torch.testing.assert_close(pushed.evaluate(), circ.evaluate())

    pushed = ModulePusher(flatten_modules=False).push_down_modules(
        circ, traversal=new_traversal(term_early_at={"sym_a", "sym_d", "add"}), skip_module={"a", "c"}
    )
    assert isinstance(pushed.get_unique("within_arg"), Add)
    pushed.print()
    torch.testing.assert_close(pushed.evaluate(), circ.evaluate())

    pushed = ModulePusher(flatten_modules=True).push_down_modules(
        circ, traversal=new_traversal(term_early_at={"sym_a", "sym_d", "add"}), skip_module={"a", "c"}
    )
    assert isinstance(pushed.get_unique("within_arg"), Add)
    pushed.print()
    torch.testing.assert_close(pushed.evaluate(), circ.evaluate())


def test_get_free_symbols():
    s = """
    'a' [] Symbol 26051952-3d72-44dd-a5b0-3aa9ac49df60
    'a.bind' Module
      'a'
    'a.bind_unused' Module
      'a'
      'b' [] Symbol afa399cc-8db1-4670-9503-2aed00a1690d ! 'c' [] Symbol 706e3b9b-75d9-438f-b590-a7ee9c244104
    'a.bind_used' Module
      'a'
      'b' ! 'a'
    'b'
    """

    a, a_bind, a_bind_unused, a_bind_used, b = Parser(module_check_all_inputs_used=False).parse_circuits(s)
    assert get_free_symbols(a) == [a]
    assert get_free_symbols(a_bind) == [a]
    assert get_free_symbols(a_bind_unused) == [a, b]
    assert get_free_symbols(a_bind_used) == [b]


def test_complex_nesting_case():
    s = """
    'a' Module
      'b' Module
        'wrap' Add
          'inner_a' Module
            'inner_b' Module
              'add' Add
                'b_sym' [] Symbol d8d42e00-53bb-44df-8fe5-ce833805cc3e
                'b_other' [] Symbol 49ca8b31-7c1b-468c-a33e-c6a07a31e224
              'my_nested' Einsum -> ! 'b_sym'
                'a_sym' [] Symbol 43161ffc-dcf1-4163-a08b-54c6174472d3
            'a_val'  [4, 1, 1] Array rand ! 'a_sym'
          'b_sym'
        'b_other_val' [2] Array rand ! 'b_other'
        'my_nested' ! 'b_sym'
      'a_val2' [3, 1] Array rand ! 'a_sym'
    """
    c = cast_circuit(Parser()(s), TorchDeviceDtypeOp(dtype=torch.float64))

    mods = ["a", "b", "inner_a", "inner_b"]
    for perm in list(itertools.permutations(mods)):
        new = c
        for n in perm:
            # might miss some mods, whatever
            new = new.update(Matcher.regex(rf"^{n}(( )|$)") & Module, lambda x: x.cast_module().substitute())
            assert not any("internal_expand" in x.name for x in all_children(new))
        torch.testing.assert_close(new.evaluate(), c.evaluate())


def test_override_batch_sub_regression():
    s = """
    'b' Module
      'wrap' Add
        'inner_a' Module
          'add' Add
            'b_sym' [] Symbol d8d42e00-53bb-44df-8fe5-ce833805cc3e
            'b_other' [] Symbol 49ca8b31-7c1b-468c-a33e-c6a07a31e224
          'a_val' [4,1,1] Array rand ! 'b_sym'
        'b_sym'
      'b_other_val' [2] Array rand ! 'b_other'
      'a_val2' [] Scalar 23874 ! 'b_sym'
    """
    c = cast_circuit(Parser()(s), TorchDeviceDtypeOp(dtype=torch.float64))
    torch.testing.assert_close(c.cast_module().substitute().evaluate(), c.evaluate())
    assert not any("internal_expand" in x.name for x in all_children(c.cast_module().substitute()))

    s = """
    'b' Module
      'wrap' Add
        'inner_a' Module
          'add' Add
            'b_sym' [] Symbol d8d42e00-53bb-44df-8fe5-ce833805cc3e
            'b_other' [] Symbol 49ca8b31-7c1b-468c-a33e-c6a07a31e224
          'a_val' [4,1,1] Array rand ! 'b_sym'
        'b_sym'
      'b_other_val' [2] Array rand ! 'b_other'
      'a_val2' [3, 2] Scalar 23874 ! 'b_sym'
    """
    c = cast_circuit(Parser()(s), TorchDeviceDtypeOp(dtype=torch.float64))
    torch.testing.assert_close(c.cast_module().substitute().evaluate(), c.evaluate())
    print(c.cast_module().substitute())
    assert not any("internal_expand" in x.name for x in all_children(c.cast_module().substitute()))


def test_module_pusher_naming_edge_cases():
    s = """
    'random_sym' [] Symbol ded02d73-6102-4f7f-a8cf-15b1065e226e
    'b' Module
      'random_sym'
    """
    sym, c = Parser().parse_circuits(s)
    assert ModulePusher().push_down_modules(c, traversal=new_traversal()) == sym

    s = """
    'random_sym' [] Symbol ded02d73-6102-4f7f-a8cf-15b1065e226e
    'a' Module
      'b' Module
        'random_sym'
    """
    sym, c = Parser().parse_circuits(s)
    assert ModulePusher().push_down_modules(c, traversal=new_traversal()) == sym

    s = """
    'random_sym' [] Symbol ded02d73-6102-4f7f-a8cf-15b1065e226e
    'add' Add
      'random_sym'
      'random_sym'
    'b' Module
      'add'
    """
    sym, add, c = Parser().parse_circuits(s)
    assert ModulePusher().push_down_modules(c, traversal=new_traversal()) == add

    s = """
    'random_sym' [] Symbol ded02d73-6102-4f7f-a8cf-15b1065e226e
    'add' Add
      'random_sym'
      'random_sym'
    'a' Module
      'b' Module
        'add'
    """
    sym, add, c = Parser().parse_circuits(s)
    assert ModulePusher().push_down_modules(c, traversal=new_traversal()) == add

    s = """
    'random_sym' [] Symbol ded02d73-6102-4f7f-a8cf-15b1065e226e
    'add' Add
      'random_sym'
      'random_sym'
    'a' Module
      'b' Module
        'c' Module
          'add'
          'garbage' [] Scalar 3 ! 'unused0' [] Symbol 26f1c039-5491-458e-a908-1a90e258c806
        'garbage' ! 'unused0'
      'garbage' ! 'unused0'
    """
    sym, add, c = Parser(module_check_all_inputs_used=False).parse_circuits(s)
    assert ModulePusher().push_down_modules(c, traversal=new_traversal()) == add

    s = """
    'random_sym' [] Symbol ded02d73-6102-4f7f-a8cf-15b1065e226e
    'add' Add
      'random_sym'
      'random_sym'
    'c' Module
      'add'
      'garbage' [] Scalar 3 ! 'unused0' [] Symbol 26f1c039-5491-458e-a908-1a90e258c806

    'a' Module
      'b' Module
        'c'
        'garbage' ! 'unused0'
      'garbage' ! 'unused0'
    """
    sym, add, c_mod, c = Parser(module_check_all_inputs_used=False).parse_circuits(s)
    assert ModulePusher().push_down_modules(c, traversal=new_traversal(term_early_at="c")) == c_mod

    s = """
    'random_sym' [] Symbol ded02d73-6102-4f7f-a8cf-15b1065e226e
    'my_arg' [] Scalar 3874
    'b' Module
      'random_sym'
      'my_arg' ! 'random_sym'
    """
    sym, arg, c = Parser().parse_circuits(s)
    assert ModulePusher().push_down_modules(c, traversal=new_traversal()) == arg

    s = """
    'random_sym' [] Symbol ded02d73-6102-4f7f-a8cf-15b1065e226e
    'my_arg' [] Scalar 3874
    'a' Module
      'b' Module
        'random_sym'
        'my_arg' ! 'random_sym'
    """
    sym, arg, c = Parser().parse_circuits(s)
    assert ModulePusher().push_down_modules(c, traversal=new_traversal()) == arg

    s = """
    'random_sym' [] Symbol ded02d73-6102-4f7f-a8cf-15b1065e226e
    0 'b' Add
      'add bind:b' Add
        'my_arg' [] Scalar 3874
        'my_arg' [] Scalar 3874
    1 'b' Module
      'outer' Add
        'add' Add
          'random_sym'
          'random_sym'
      'my_arg' ! 'random_sym'
    """
    sym, new_add, c = Parser().parse_circuits(s)

    def namer(*args):
        print(args)
        return default_nested_module_namer()(*args)

    assert ModulePusher(namer=namer).push_down_modules(c, traversal=new_traversal()) == new_add


def test_module_sub():
    s = """
    'random_sym' [] Symbol ded02d73-6102-4f7f-a8cf-15b1065e226e
    'b' Module
      'random_sym'
    """
    sym, c = Parser().parse_circuits(s)
    assert c.cast_module().substitute() == sym

    s = """
    'random_sym' [] Symbol ded02d73-6102-4f7f-a8cf-15b1065e226e
    'a' Module
      'b' Module
        'random_sym'
    """
    sym, c = Parser().parse_circuits(s)
    assert substitute_all_modules(c) == sym
    assert substitute_all_modules(c.cast_module().substitute()) == sym

    s = """
    'random_sym' [] Symbol ded02d73-6102-4f7f-a8cf-15b1065e226e
    'add' Add
      'random_sym'
      'random_sym'
    'b' Module
      'add'
    """
    sym, add, c = Parser().parse_circuits(s)
    assert c.cast_module().substitute() == add

    s = """
    'random_sym' [] Symbol ded02d73-6102-4f7f-a8cf-15b1065e226e
    'add' Add
      'random_sym'
      'random_sym'
    'a' Module
      'b' Module
        'add'
    """
    sym, add, c = Parser().parse_circuits(s)
    assert substitute_all_modules(c) == add
    assert substitute_all_modules(c.cast_module().substitute()) == add

    s = """
    'random_sym' [] Symbol ded02d73-6102-4f7f-a8cf-15b1065e226e
    'add' Add
      'random_sym'
      'random_sym'
    'a' Module
      'b' Module
        'c' Module
          'add'
          'garbage' [] Scalar 3 ! 'unused0' [] Symbol 26f1c039-5491-458e-a908-1a90e258c806
        'garbage' ! 'unused0'
      'garbage' ! 'unused0'
    """
    sym, add, c = Parser(module_check_all_inputs_used=False).parse_circuits(s)
    assert substitute_all_modules(c) == add
    assert substitute_all_modules(c.cast_module().substitute()) == add

    s = """
    'random_sym' [] Symbol ded02d73-6102-4f7f-a8cf-15b1065e226e
    'my_arg' [] Scalar 3874
    'b' Module
      'random_sym'
      'my_arg' ! 'random_sym'
    """
    sym, arg, c = Parser().parse_circuits(s)
    assert c.cast_module().substitute() == arg

    s = """
    'random_sym' [] Symbol ded02d73-6102-4f7f-a8cf-15b1065e226e
    'my_arg' [] Scalar 3874
    'a' Module
      'b' Module
        'random_sym'
        'my_arg' ! 'random_sym'
    """
    sym, arg, c = Parser().parse_circuits(s)
    assert substitute_all_modules(c.cast_module().substitute()) == arg
    assert substitute_all_modules(c) == arg
