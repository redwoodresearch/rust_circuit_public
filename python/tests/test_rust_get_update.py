import itertools
from typing import List, Type, Union

import pytest

import rust_circuit as rc
import rust_circuit.optional as op
from interp.tools.indexer import SLICER as S
from rust_circuit import (
    FINISHED,
    Add,
    AnyFound,
    Array,
    Circuit,
    Cumulant,
    Einsum,
    Expander,
    Finished,
    GeneralFunction,
    Getter,
    IterateMatchResults,
    IterativeMatcher,
    IterativeMatcherChildNumbersOutOfBoundsError,
    Matcher,
    Module,
    PrintOptions,
    Regex,
    Scalar,
    Symbol,
    Transform,
    Updater,
    all_children,
    append_matchers_to_names,
    module_new_bind,
    new_traversal,
    print_matcher_debug,
    restrict,
    restrict_sl,
)
from rust_circuit.module_library import TransformerBlockParams, layernorm


# By default I'd recommend using the multi argument constructor for doing
# 'any'. We don't do always do this here, but that's for testing reasons.
def test_basic_matching():
    a = Scalar(1.0, shape=(1, 2), name="a_name")
    b = Scalar(2.0, shape=(2, 3), name="b_other")
    c = Einsum.from_einsum_string("ab,bc->ac", a, b)

    assert Matcher("a_name")(a)
    assert not Matcher("a_name")(b)
    assert Matcher({"a_name", "b_other"})(a)
    assert Matcher(frozenset[str]({"a_name", "b_other"}))(a)
    assert Matcher(frozenset[Type[Circuit]]({Scalar, Einsum}))(c)
    assert Matcher(frozenset[Circuit]({c, b}))(c)
    assert Matcher({"a_name", "b_other"})(b)
    assert not Matcher({"a_name", "b_other"})(c)

    assert Matcher(a)(a)
    assert not Matcher(a)(b)
    assert Matcher(set[Circuit]({a, c}))(a)
    assert Matcher(set[Circuit]({a, c}))(c)
    assert not Matcher(set[Circuit]({a, c}))(b)
    assert Matcher(Scalar)(b)
    assert not Matcher(Scalar)(c)
    assert not Matcher(Scalar).call(c)
    assert Matcher(set[Type[Circuit]]({Scalar, Einsum}))(c)
    assert Matcher(set[Type[Circuit]]({Scalar, Einsum}))(a)
    assert Matcher(set[Type[Circuit]]({Scalar, Einsum})).call(a)

    assert Matcher.circuits(a)(a)
    assert not Matcher.circuits(a)(b)
    assert Matcher.circuits(a, c)(a)
    assert Matcher.circuits(a, c)(c)
    assert not Matcher.circuits(a, c)(b)

    has_child = Matcher.match_any_child_found(a)
    assert not has_child(a)
    assert not has_child(b)
    assert has_child(c)
    assert has_child == Matcher.match_any_child_found(a)
    assert has_child == Matcher.match_any_child_found(set[Circuit]({a}))
    has_child = Matcher.match_any_child_found(set[Circuit]({a, c}))
    assert not has_child(a)
    assert not has_child(b)
    assert has_child(c)
    assert has_child == Matcher.match_any_child_found(set[Circuit]({c, a}))
    e = Scalar(3.0, shape=(1, 3), name="hi")
    d = Add(c, e)
    has_child = Matcher.match_any_child_found(a)
    assert has_child(d)
    assert has_child(c)
    assert not has_child(a)
    assert not has_child(b)
    assert not has_child(e)

    match_any_found = Matcher.match_any_found(a)
    assert match_any_found(d)
    assert match_any_found(c)
    assert match_any_found(a)
    assert not match_any_found(b)
    assert not match_any_found(e)

    assert Matcher.types(Scalar)(b)
    assert not Matcher.types(Scalar)(c)
    assert Matcher.types(Scalar, Einsum)(c)
    assert Matcher.types(Scalar, Einsum)(a)
    assert not Matcher.types(Add, Einsum)(a)

    assert Matcher("a_name") == Matcher("a_name")
    assert Matcher("a_name") != Matcher("a_name_")
    assert Matcher(Scalar) == Matcher(Scalar)
    assert Matcher.types(Scalar, Einsum) == Matcher.types(Einsum, Scalar)
    assert Matcher.types(Scalar, Einsum) == Matcher(set[Type[Circuit]]({Einsum, Scalar}))
    assert Matcher(Scalar) != Matcher(Einsum)
    assert Matcher(Scalar) != Matcher(Einsum)
    assert Matcher.true_matcher() == Matcher(True)
    assert Matcher.false_matcher() == Matcher(False)
    assert Matcher.false_matcher() != Matcher(True)
    assert hash(Matcher.false_matcher()) == hash(Matcher(False))

    assert Matcher("a_name").new_or("b_other")(b)
    assert Matcher("a_name").new_or("b_other")(a)
    assert not Matcher("a_name").new_or("b_other")(c)
    assert not Matcher("a_name").new_and("b_other")(a)
    assert not (Matcher("a_name") & "b_other")(a)
    assert (Matcher("a_name") | "b_other")(a)
    assert not ("b_other" & Matcher("a_name"))(a)
    assert ("b_other" | Matcher("a_name"))(a)
    assert not (~Matcher("a_name"))(a)
    assert (~Matcher("a_name"))(b)
    assert (~Matcher("a_name"))(c)
    assert not (Matcher("a_name").new_not())(a)
    assert (Matcher("a_name").new_not())(b)
    assert (Matcher("a_name").new_not())(c)
    assert Matcher("a_name").new_and(Scalar)(a)
    assert not Matcher("a_name").new_and(Scalar)(b)

    assert Matcher.any("a_name", "b_other")(b)
    assert Matcher.any("a_name", "b_other")(a)
    assert not Matcher.any("a_name", "b_other")(c)
    assert Matcher("a_name", "b_other") == Matcher.any("a_name", "b_other")

    # matcher short-circuiting
    empty_arr = []
    assert not (Matcher("a_name") & Matcher(lambda _: empty_arr[0]))(b)
    assert not Matcher.all("a_name", lambda _: empty_arr[0])(b)
    assert Matcher.any("b_other", lambda _: empty_arr[0])(b)
    assert (Matcher("b_other") | Matcher(lambda _: empty_arr[0]))(b)

    assert Matcher.all("a_name", Scalar)(a)
    assert not Matcher.all("a_name", Scalar)(b)
    assert not Matcher.all("a_name", Scalar)(c)

    assert Matcher.all("a_name")(a)
    assert not Matcher.all("a_name")(b)
    assert not Matcher.all("a_name")(c)

    # it's preferred to use Matcher.regex in most cases
    assert Matcher.regex("a_name") == Matcher(Regex("a_name"))
    assert Matcher.regex("a_name", escape_dot=False) == Matcher(Regex("a_name", escape_dot=False))
    assert Matcher.regex("a_name") != Matcher.regex("b_name")
    assert Matcher.regex("a_name") == Matcher.regex("a_name")
    assert Matcher.regex("a_name") != Matcher.regex("a_name", escape_dot=False)
    assert Matcher.regex("a_name", escape_dot=False) == Matcher.regex("a_name", escape_dot=False)
    assert Matcher.regex("...a_name", escape_dot=False) == Matcher.regex("...a_name", escape_dot=False)
    assert Matcher.regex("...a_name") == Matcher.regex("...a_name")

    assert Matcher.regex("a")(a)
    assert not Matcher.regex("a")(b)

    assert Matcher.regex("[ab]")(a)
    assert Matcher.regex("[ab]")(b)

    assert Matcher.regex("^my.name.has_dot$")(Scalar(2.0, shape=(2, 3), name="my.name.has_dot"))
    assert not Matcher.regex("^my.name...s_dot$")(Scalar(2.0, shape=(2, 3), name="my.name.has_dot"))
    assert Matcher.regex(r"^my.name.\.\.s_dot$")(Scalar(2.0, shape=(2, 3), name="my.name.has_dot"))
    assert Matcher.regex(r"^\.*$")(c)
    assert not Matcher.regex(r"^\.*$")(c.rename(None))

    for n in [a, b, c]:
        for v in [False, True]:
            assert Matcher(v)(n) == v

    with pytest.raises(TypeError, match="coerced"):
        Matcher("a_name") and Matcher("b_other")


def test_matcher_get():
    a = Scalar(1.0, shape=(1, 2), name="a_name")
    b = Scalar(2.0, shape=(2, 3), name="b_other")
    c = Einsum.from_einsum_string("ab,bc->ac", a, b)

    assert Matcher("a_name").get(c) == {a}
    assert c.get("a_name") == {a}
    assert Matcher("b_other").get(c) == {b}
    assert Matcher("b_other").get_first(c) == b
    assert Matcher("a_name", "b_other").get_first(c) == b


def test_iterative_just_matcher():
    assert Matcher("hi").to_iterative_matcher() == IterativeMatcher("hi")
    assert Matcher("hi").to_iterative_matcher() != IterativeMatcher("hij")
    assert Matcher(False).to_iterative_matcher() == IterativeMatcher(False)
    assert Matcher(False).to_iterative_matcher() != IterativeMatcher(True)
    assert Matcher.regex("a").to_iterative_matcher() == IterativeMatcher(Matcher.regex("a"))
    assert Matcher.regex("a").to_iterative_matcher() != IterativeMatcher(Matcher.regex("ab"))
    assert Matcher(Matcher.regex("a")).to_iterative_matcher() != IterativeMatcher(Matcher.regex("ab"))

    a = Scalar(1.0, shape=(1, 2), name="a_name")
    b = Scalar(2.0, shape=(2, 3), name="b_other")
    c = Einsum.from_einsum_string("ab,bc->ac", a, b)

    def simple_match_check(res: IterateMatchResults, found: bool):
        assert res.updated is None
        assert res.found == found

    simple_match_check(IterativeMatcher("a_name").match_iterate(a), True)
    simple_match_check(IterativeMatcher("a_name").match_iterate(b), False)
    simple_match_check(IterativeMatcher({"a_name", "b_other"}).match_iterate(a), True)
    simple_match_check(IterativeMatcher({"a_name", "b_other"}).match_iterate(b), True)
    simple_match_check(IterativeMatcher({"a_name", "b_other"}).match_iterate(c), False)

    simple_match_check(IterativeMatcher(a).match_iterate(a), True)
    simple_match_check(IterativeMatcher(a).match_iterate(b), False)
    simple_match_check(IterativeMatcher(set[Circuit]({a, c})).match_iterate(a), True)
    simple_match_check(IterativeMatcher(set[Circuit]({a, c})).match_iterate(c), True)
    simple_match_check(IterativeMatcher(set[Circuit]({a, c})).match_iterate(b), False)
    simple_match_check(IterativeMatcher(Scalar).match_iterate(b), True)
    simple_match_check(IterativeMatcher(Scalar).match_iterate(c), False)
    simple_match_check(IterativeMatcher(Scalar).match_iterate(c), False)
    simple_match_check(IterativeMatcher(set[Type[Circuit]]({Scalar, Einsum})).match_iterate(c), True)
    simple_match_check(IterativeMatcher(set[Type[Circuit]]({Scalar, Einsum})).match_iterate(a), True)
    simple_match_check(IterativeMatcher(set[Type[Circuit]]({Scalar, Einsum})).match_iterate(a), True)


def test_iterative_restrict_constructor():
    all_items = list(
        itertools.product(
            ["hi", "x", True],
            [False, True],
            [None, 0, 2],
            [None, 0, 2],
            ["hi", False],
        )
    )

    for base, term_if_matches, start_depth, end_depth, term_early_at in all_items:
        restrict_args = term_if_matches, start_depth, end_depth, term_early_at
        restrict_sl_args = term_if_matches, slice(start_depth, end_depth), term_early_at
        item = restrict(base, *restrict_args)
        assert restrict(Matcher(base).to_iterative_matcher(), *restrict_args) == item
        assert restrict(IterativeMatcher(base), *restrict_args) == item
        assert restrict_sl(base, *restrict_sl_args) == item
        assert restrict_sl(IterativeMatcher(base), *restrict_sl_args) == item

        for (
            base_other,
            term_if_matches_other,
            start_depth_other,
            end_depth_other,
            term_early_at_other,
        ) in all_items:
            restrict_args_other = term_if_matches_other, start_depth_other, end_depth_other, term_early_at_other
            if base_other != base or restrict_args_other != restrict_args:
                other_item = restrict(Matcher(base_other), *restrict_args_other)
                assert other_item != item


def test_finished_eq():
    assert IterativeMatcher.term().match_iterate(Scalar(0.3)).updated == FINISHED


def test_iterative_restrict():
    m = restrict_sl("a_name", depth_slice=S[2:])
    a = Scalar(1.0, shape=(1, 2), name="a_name")
    b = Scalar(2.0, shape=(2, 3), name="b_other")

    def check_valid(m: IterativeMatcher, a: Circuit, reached: bool = True) -> IterativeMatcher:
        x = m.match_iterate(a)
        assert x.updated is None or isinstance(x.updated, IterativeMatcher)
        assert (x.updated is None) == reached, (x.updated, reached)
        assert x.found == reached

        return op.unwrap_or(x.updated, m)

    m = check_valid(m, a, reached=False)
    m = check_valid(m, a, reached=False)
    m = check_valid(m, a, reached=True)
    m = check_valid(m, a, reached=True)

    def check_valid_both(m: IterativeMatcher, a: Circuit, found: bool = True, last: bool = False):
        x = m.match_iterate(a)
        assert (x.updated == FINISHED) == last
        assert x.found == found
        if x.updated == FINISHED:
            # hack for typing
            return m

        assert x.updated is None or isinstance(x.updated, IterativeMatcher)
        return op.unwrap_or(x.updated, m)

    m = restrict_sl("a_name", depth_slice=S[3:5])

    for _ in range(3):
        m = check_valid_both(m, a, found=False)
    m = check_valid_both(m, a, found=True)
    m = check_valid_both(m, a, found=True, last=True)

    m = restrict_sl("a_name", depth_slice=S[:0])
    check_valid_both(m, a, found=False, last=True)
    m = restrict_sl("a_name", depth_slice=S[0:0])
    check_valid_both(m, a, found=False, last=True)
    m = restrict_sl("a_name", depth_slice=S[1:0])
    check_valid_both(m, a, found=False, last=True)

    m = restrict_sl("a_name", term_if_matches=True, depth_slice=S[3:5])
    for _ in range(3):
        m = check_valid_both(m, a, found=False)
    m = check_valid_both(m, a, found=True, last=True)

    m = restrict_sl("a_name", term_if_matches=True, depth_slice=S[3:7])
    for _ in range(3):
        m = check_valid_both(m, a, found=False)
    m = check_valid_both(m, b, found=False, last=False)
    m = check_valid_both(m, b, found=False, last=False)
    m = check_valid_both(m, a, found=True, last=True)

    def correct_no_update(m: IterativeMatcher, a: Circuit, found: bool = True, last: bool = False):
        x = m.match_iterate(a)
        if last:
            assert x.updated == FINISHED
            return m  # typing hack
        else:
            assert x.updated is None
        assert x.found == found
        assert x.updated is None or isinstance(x.updated, IterativeMatcher)

        return op.unwrap_or(x.updated, m)

    m = restrict("a_name", term_early_at="b_other")
    m = correct_no_update(m, a, found=True)
    m = correct_no_update(m, a, found=True)
    m = correct_no_update(m, b, found=False, last=True)


def test_start_depth_behavior():
    b = rc.Add(name="B")
    c = rc.Add(name="C")
    a = rc.Add(b, c, name="A")

    assert new_traversal(term_early_at=rc.Add, start_depth=0).get(a) == {a}
    for d in range(1, 4):
        assert new_traversal(term_early_at=rc.Add, start_depth=d).get(a) == set()

    assert new_traversal(start_depth=1, end_depth=0).match_iterate(a).updated == FINISHED


def test_chain_constructor():
    all_items = list(
        itertools.product(
            ["hi", "x", True],
            [("other",), ("other", "hi"), (False,), (False, False), (False, True)],
        )
    )

    for must_be_sub in [False, True]:
        for first, rest in all_items:
            for first_other, rest_other in all_items:
                item = Matcher(first).chain(*rest, must_be_sub=must_be_sub)
                assert Matcher(first).to_iterative_matcher().chain(*rest, must_be_sub=must_be_sub) == item
                assert IterativeMatcher.new_chain(first, *rest, must_be_sub=must_be_sub) == item

                if first_other != first or rest != rest_other:
                    other_item = Matcher(first_other).chain(*rest_other, must_be_sub=must_be_sub)
                    assert other_item != item

                assert Matcher(first).chain_many(rest, must_be_sub=must_be_sub) == item
                assert Matcher(first).to_iterative_matcher().chain_many(rest, must_be_sub=must_be_sub) == item
                assert IterativeMatcher.new_chain_many([first, *rest], must_be_sub=must_be_sub) == item


@pytest.mark.parametrize("must_be_sub", [True, False])
def test_chain_matching(must_be_sub: bool):
    # must_be_sub should make no difference for these tests
    m = IterativeMatcher.new_chain("top", "leaf", must_be_sub=must_be_sub)

    def check_update(m: IterativeMatcher, a: Circuit, found: bool = True, change: bool = True, finish: bool = False):
        x = m.match_iterate(a)
        if finish:
            assert x.updated == FINISHED
            return m  # hack for typing
        assert (x.updated is not None) == (change and not finish)
        assert x.found == found
        assert x.updated is None or isinstance(x.updated, IterativeMatcher)

        return op.unwrap_or(x.updated, m)

    a = Scalar(1.0, shape=(1, 2), name="top")
    b = Scalar(2.0, shape=(2, 3), name="leaf")
    c = Scalar(2.0, shape=(2, 3), name="other")
    d = Scalar(2.0, shape=(2, 3), name="dog")

    m = check_update(m, a, found=False)
    m = check_update(m, a, found=False, change=False)
    m = check_update(m, a, found=False, change=False)
    m = check_update(m, a, found=False, change=False)
    m = check_update(m, b, found=True, change=False)
    m = check_update(m, b, found=True, change=False)

    m = IterativeMatcher.new_chain(restrict("top", end_depth=2), restrict("leaf", end_depth=3), must_be_sub=must_be_sub)

    m = check_update(m, a, found=False)
    m = check_update(m, a, found=False)
    m = check_update(m, a, found=False)
    m = check_update(m, a, found=False, finish=True)

    m = IterativeMatcher.new_chain(restrict("top", end_depth=2), restrict("leaf", end_depth=3), must_be_sub=must_be_sub)

    m = check_update(m, b, found=False)
    m = check_update(m, b, found=False, finish=True)

    m = IterativeMatcher.new_chain(restrict("top", end_depth=2), restrict("leaf", end_depth=4), must_be_sub=must_be_sub)

    m = check_update(m, a, found=False)
    m = check_update(m, c, found=False)
    m = check_update(m, b, found=True)
    m = check_update(m, b, found=True, finish=True)

    m = IterativeMatcher.new_chain(
        restrict("top", end_depth=2),
        restrict("leaf", end_depth=4),
        Matcher("other"),
        must_be_sub=must_be_sub,
    )

    m = check_update(m, a, found=False)
    m = check_update(m, c, found=False)
    m = check_update(m, c, found=False)
    m = check_update(m, b, found=False)
    m = check_update(m, c, found=True, change=False)
    m = check_update(m, c, found=True, change=False)

    m = IterativeMatcher.new_chain_many(
        (restrict("top", end_depth=2), restrict("leaf", end_depth=4), Matcher("other")),
        (Matcher("dog"),),
        must_be_sub=must_be_sub,
    )

    m = check_update(m, a, found=False)
    m = check_update(m, d, found=True)
    m = check_update(m, c, found=False)
    m = check_update(m, b, found=False)
    m = check_update(m, c, found=True, change=False)
    m = check_update(m, c, found=True, change=False)
    m = check_update(m, d, found=True, change=False)

    m = IterativeMatcher.any(restrict("top", end_depth=2), Matcher("dog"))

    m = check_update(m, a, found=True)
    m = check_update(m, a, found=True)
    m = check_update(m, a, found=False, change=False)
    m = check_update(m, d, found=True, change=False)


def test_chain_must_be_sub():
    a = Scalar(1.0, shape=(1, 2), name="top")
    b = Scalar(2.0, shape=(2, 2), name="leaf")
    c = Scalar(2.0, shape=(2, 3), name="other")
    d = Scalar(2.0, shape=(2, 3), name="dog")

    def check_update(m: IterativeMatcher, a: Circuit, found: bool = True, change: bool = True, finish: bool = False):
        x = m.match_iterate(a)
        if finish:
            assert x.updated == FINISHED
            return m  # hack for typing
        assert (x.updated is not None) == (change and not finish)
        assert x.found == found
        assert x.updated is None or isinstance(x.updated, IterativeMatcher)

        return op.unwrap_or(x.updated, m)

    m = IterativeMatcher(Scalar).chain(Scalar, must_be_sub=False)
    m = check_update(m, a)
    m = check_update(m, b, change=False)
    m = check_update(m, c, change=False)

    m = IterativeMatcher(Scalar).chain(Scalar, must_be_sub=True)
    m = check_update(m, a, found=False)
    m = check_update(m, b, change=False)
    m = check_update(m, c, change=False)

    m = IterativeMatcher("top").chain(restrict(lambda x: x.shape[-1] == 2, end_depth=2), "dog", must_be_sub=True)
    m = check_update(m, b, found=False, change=False)
    m = check_update(m, d, found=False, change=False)
    m = check_update(m, a, found=False, change=True)
    m = check_update(m, a, found=False)
    m = check_update(m, d, found=True, change=True)

    m = IterativeMatcher("top").chain(restrict(lambda x: x.shape[-1] == 2, end_depth=2), "dog", must_be_sub=True)
    m = check_update(m, b, found=False, change=False)
    m = check_update(m, d, found=False, change=False)
    m = check_update(m, a, found=False, change=True)
    m = check_update(m, b, found=False)
    m = check_update(m, b, found=False, change=False)
    m = check_update(m, b, found=False, change=False)
    m = check_update(m, b, found=False, change=False)
    m = check_update(m, d, found=True, change=False)

    m = IterativeMatcher("top").chain(restrict(lambda x: x.shape[-1] == 2, end_depth=2), "dog", must_be_sub=True)
    m = check_update(m, b, found=False, change=False)
    m = check_update(m, d, found=False, change=False)
    m = check_update(m, a, found=False, change=True)
    m = check_update(m, c, found=False, change=True)
    m = check_update(m, c, found=False, change=False)
    m = check_update(m, d, found=False, change=False)

    m = IterativeMatcher("top").chain(restrict(lambda x: x.shape[-1] == 2, end_depth=2), "dog", must_be_sub=False)
    m = check_update(m, b, found=False, change=False)
    m = check_update(m, d, found=False, change=False)
    m = check_update(m, a, found=False, change=True)
    m = check_update(m, c, found=False, change=True)
    m = check_update(m, c, found=False, change=False)
    m = check_update(m, d, found=True, change=False)


def test_update_branching():
    a = Scalar(1.0, shape=(1, 2), name="a")
    b = Scalar(2.0, shape=(2, 3), name="b")
    ein = Einsum.from_einsum_string("ij,jk->ik", a, b, name="ein")
    soft = GeneralFunction.new_by_name(ein, spec_name="softmax", name="soft")
    added = Add(ein, soft, name="fin")

    b_prime = Scalar(1.7, shape=(2, 3), name="b_prime")

    def check(matcher: Union[IterativeMatcher, Matcher], expected: Circuit, *updater_args):
        assert matcher.update(added, *updater_args) == expected
        assert added.update(matcher, *updater_args) == expected
        assert Updater(*updater_args)(added, matcher) == expected
        assert Updater(*updater_args).bind(matcher)(added) == expected
        assert matcher.updater(*updater_args)(added) == expected

    a = Scalar(1.0, shape=(1, 2), name="a")
    ein_full_p = Einsum.from_einsum_string("ij,jk->ik", a, b_prime, name="ein")
    soft_full_p = GeneralFunction.new_by_name(ein_full_p, spec_name="softmax", name="soft")
    added_full_p = Add(ein_full_p, soft_full_p, name="fin")

    check(Matcher(b), added_full_p, lambda _: b_prime)

    ein_prime = Einsum.from_einsum_string("ij,jk->ik", a, b_prime, name="ein")
    soft_prime = GeneralFunction.new_by_name(ein_prime, spec_name="softmax", name="soft")
    added_prime = Add(ein, soft_prime, name="fin")

    check(IterativeMatcher.new_chain("soft", "b"), added_prime, lambda _: b_prime)


def test_transform_ident():
    a = Scalar(1.0, shape=(1, 2), name="a")
    assert Transform.ident()(a) == a


def test_get():
    a = Scalar(1.0, shape=(1, 2), name="a")
    b = Scalar(2.0, shape=(2, 3), name="b")
    ein = Einsum.from_einsum_string("ij,jk->ik", a, b, name="ein")
    soft = GeneralFunction.new_by_name(ein, spec_name="softmax", name="soft")
    added = Add(ein, soft, name="fin")
    other_circ = Symbol.new_with_random_uuid((3,))

    assert Getter()(added, Scalar, fancy_validate=True) == {a, b}
    assert Getter()(added, ein, fancy_validate=True) == {ein}
    assert Getter().get_unique_op(added, ein, fancy_validate=True) == ein
    assert Getter().get_unique(added, ein, fancy_validate=True) == ein
    assert Getter().get_unique_op(added, "sdkfj") is None
    assert Getter()(added, "sdkfj") == set()

    with pytest.raises(RuntimeError, match="found no"):
        Getter().get_unique(added, "sdkfj")

    assert Getter()(added, ~Matcher(Scalar), fancy_validate=True) == {ein, soft, added}
    assert (~Matcher(Scalar)).get(added) == {ein, soft, added}
    assert Matcher("a").get_unique(added) == a
    assert Matcher("a").get_unique_op(added) == a
    assert Matcher("sdkfj").get_unique_op(added) is None
    assert Matcher("sdkfj").get(added) == set()

    assert added.get(~Matcher(Scalar), fancy_validate=True) == {ein, soft, added}
    assert added.get(~Matcher(Scalar)) == {ein, soft, added}
    assert added.get_unique("a") == a
    assert added.get_unique(Matcher("a")) == a
    assert added.get_unique(IterativeMatcher(Matcher("a"))) == a
    assert added.get_unique_op("a") == a
    assert added.get_unique_op("sdkfj") is None
    assert added.get("sdkfj") == set()

    assert (~Matcher(Scalar)).getter()(added) == {ein, soft, added}
    assert Matcher("a").getter().get_unique(added) == a
    assert Matcher("a").getter().get_unique_op(added) == a

    with pytest.raises(RuntimeError, match="Didn't match all names"):
        Getter()(added, "sdkfj", fancy_validate=True)
    with pytest.raises(RuntimeError, match="Didn't match all names"):
        Getter()(added, {"sdkfj", "ein"}, fancy_validate=True)
    with pytest.raises(RuntimeError, match="Didn't match all types"):
        Getter()(added, Cumulant, fancy_validate=True)
    with pytest.raises(RuntimeError, match="Didn't match all types"):
        Getter()(added, Matcher.types(Einsum, Cumulant), fancy_validate=True)
    with pytest.raises(RuntimeError, match="Didn't match all types"):
        Matcher.types(Einsum, Cumulant).validate(added)
    with pytest.raises(RuntimeError, match="Didn't match all types"):
        Matcher.types(Einsum, Cumulant).get(added, fancy_validate=True)
    with pytest.raises(RuntimeError, match="Didn't match all types"):
        Matcher.types(Einsum, Cumulant).get_unique_op(added, fancy_validate=True)
    with pytest.raises(RuntimeError, match="Didn't match all types"):
        Matcher.types(Einsum, Cumulant).get_unique(added, fancy_validate=True)
    with pytest.raises(RuntimeError, match="Didn't match all circuits"):
        Getter()(added, other_circ, fancy_validate=True)
    with pytest.raises(RuntimeError, match="Didn't match all circuits"):
        Matcher.circuits(other_circ, a).get(added, fancy_validate=True)
    with pytest.raises(RuntimeError, match="Didn't match all circuit"):
        Matcher.any(Matcher.circuits(other_circ, a), "dif").get(added, fancy_validate=True)
    with pytest.raises(RuntimeError, match="Didn't match all names"):
        Matcher.any("dif", Matcher.circuits(other_circ, a)).get(added, fancy_validate=True)
    with pytest.raises(RuntimeError, match="Didn't match all circuit"):
        Matcher.all(Matcher.circuits(other_circ, a), "dif").get(added, fancy_validate=True)
    with pytest.raises(RuntimeError, match="Didn't match all names"):
        Matcher.all("dif", Matcher.circuits(other_circ, a)).get(added, fancy_validate=True)
    with pytest.raises(RuntimeError, match="Didn't match all names"):
        Matcher.all("dif", Matcher.circuits(other_circ, a)).getter(default_fancy_validate=True).get(
            added, fancy_validate=None
        )
    with pytest.raises(RuntimeError, match="Didn't match all names"):
        Matcher.all("dif", Matcher.circuits(other_circ, a)).getter(default_fancy_validate=False).get(
            added, fancy_validate=True
        )

    a = Scalar(1.0, shape=(1, 2), name="a")
    b = Scalar(2.0, shape=(2, 3), name="b")
    b_dup = Scalar(2.0, shape=(3,), name="b")
    ein = Einsum.from_einsum_string("ij,jk->ik", a, b, name="ein")
    soft = GeneralFunction.new_by_name(ein, spec_name="softmax", name="soft")
    added = Add(ein, soft, b_dup, name="fin")

    assert Matcher("ein").chain(Scalar).get(added, fancy_validate=True) == {a, b}
    assert Matcher("ein").chain("b").get(added, fancy_validate=True) == {b}
    assert Matcher("b").get(added, fancy_validate=True) == {b, b_dup}
    assert Matcher("ein").chain(b_dup).get(added, fancy_validate=False) == set()
    with pytest.raises(RuntimeError, match="Didn't match all circuits"):
        Matcher("ein").chain(b_dup).get(added, fancy_validate=True)
    assert Matcher("ein").chain(a).new_or(Matcher("fin").chain(b_dup)).get(added, fancy_validate=True) == {a, b_dup}

    with pytest.raises(RuntimeError, match="Didn't match all circuits"):
        Matcher("ein").chain(other_circ).new_or(Matcher("fin").chain(b_dup)).get(added, fancy_validate=True)
    with pytest.raises(RuntimeError, match="Didn't match all circuits"):
        Matcher("ein").chain(Matcher.circuits(other_circ, a)).new_or(Matcher("fin").chain(b_dup)).get(
            added, fancy_validate=True
        )
    with pytest.raises(RuntimeError, match="Didn't match all circuits"):
        Matcher("ein").chain(a).new_or(Matcher("fin").chain(Matcher.circuits(other_circ, b_dup))).get(
            added, fancy_validate=True
        )
    with pytest.raises(RuntimeError, match="Didn't match all circuits"):
        Matcher("ein").chain_many([a], [b_dup]).get(added, fancy_validate=True)


def test_any_found():
    a = Scalar(1.0, shape=(1, 2), name="a")
    b = Scalar(2.0, shape=(2, 3), name="b")
    ein = Einsum.from_einsum_string("ij,jk->ik", a, b, name="ein")
    soft = GeneralFunction.new_by_name(ein, spec_name="softmax", name="soft")
    added = Add(ein, soft, name="fin")
    other_circ = Symbol.new_with_random_uuid((3,))

    assert AnyFound()(added, Scalar)
    assert not AnyFound()(added, other_circ)
    assert AnyFound()(added, Einsum)
    assert not AnyFound()(b, Einsum)
    assert not IterativeMatcher(Einsum).any_found()(b)
    assert IterativeMatcher(Einsum).any_found()(added)
    assert not IterativeMatcher(Einsum).are_any_found(b)
    assert IterativeMatcher(Einsum).are_any_found(added)

    # we could add more tests I guess


def test_get_per_child():
    a = Scalar(1.0, shape=(1, 2), name="a")
    b = Scalar(2.0, shape=(2, 3), name="b")
    ein = Einsum.from_einsum_string("ij,jk->ik", a, b, name="ein")

    def run_func(x: Circuit):
        assert x == ein
        return IterateMatchResults([FINISHED, True], False)

    matcher = IterativeMatcher.new_func(run_func)
    res = matcher.match_iterate(ein)
    assert not res.found
    assert isinstance(res.updated, list)
    assert res.updated[0] == FINISHED

    assert Getter()(ein, matcher, fancy_validate=True) == {b}

    def run_func_both(x: Circuit):
        assert x == ein
        return IterateMatchResults([True, True], False)

    matcher = IterativeMatcher.new_func(run_func_both)
    assert Getter()(ein, matcher, fancy_validate=True) == {a, b}

    def matcher_for_both(_: Circuit):
        return IterateMatchResults(["b", "a"], False)

    matcher = IterativeMatcher.new_func(matcher_for_both)
    assert Getter()(ein, matcher, fancy_validate=True) == set()

    ein_new = Einsum.from_einsum_string(
        "ij,jk->ik", Add(b, a.sum()), Scalar(34.0, shape=(3, 7), name="!@*(#&$"), name="ein"
    )
    assert Getter()(ein_new, matcher, fancy_validate=True) == {b}

    ein_new = Einsum.from_einsum_string(
        "jk,ij->ik", Scalar(34.0, shape=(3, 7), name="!@*(#&$"), Add(b, a.sum(name="sum")), name="ein"
    )
    assert Getter()(ein_new, matcher, fancy_validate=True) == {a}


def test_any_found_per_child():
    # literally just copy paste + minor changes to above test
    # could do property based testing with 'get', but I'm not bothering for now
    a = Scalar(1.0, shape=(1, 2), name="a")
    b = Scalar(2.0, shape=(2, 3), name="b")
    ein = Einsum.from_einsum_string("ij,jk->ik", a, b, name="ein")

    def run_func(x: Circuit):
        assert x == ein
        return IterateMatchResults([FINISHED, True], False)

    matcher = IterativeMatcher.new_func(run_func)
    assert AnyFound()(ein, matcher)

    def run_func_both(x: Circuit):
        assert x == ein
        return IterateMatchResults([True, True], False)

    matcher = IterativeMatcher.new_func(run_func_both)
    assert AnyFound()(ein, matcher)

    def matcher_for_both(_: Circuit):
        return IterateMatchResults(["b", "a"], False)

    matcher = IterativeMatcher.new_func(matcher_for_both)
    assert not AnyFound()(ein, matcher)

    ein_new = Einsum.from_einsum_string(
        "ij,jk->ik", Add(b, a.sum()), Scalar(34.0, shape=(3, 7), name="!@*(#&$"), name="ein"
    )
    assert AnyFound()(ein_new, matcher)

    ein_new = Einsum.from_einsum_string(
        "jk,ij->ik", Scalar(34.0, shape=(3, 7), name="!@*(#&$"), Add(b, a.sum(name="sum")), name="ein"
    )
    assert AnyFound()(ein_new, matcher)


def test_update_per_child():
    a = Scalar(1.0, shape=(1, 2), name="a")
    b = Scalar(2.0, shape=(2, 3), name="b")

    def matcher_for_both(_: Circuit):
        return IterateMatchResults(["b", "a"], False)

    matcher = IterativeMatcher.new_func(matcher_for_both)

    ein_new = Einsum.from_einsum_string(
        "ij,jk->ik", Add(b, a.sum(), name="add@#$"), Scalar(34.0, shape=(3, 7), name="!@*(#&$"), name="ein"
    )

    assert Updater(lambda x: Scalar(0.0, shape=x.shape, name="zero"))(
        ein_new, matcher, fancy_validate=True
    ) == Einsum.from_einsum_string(
        "ij,jk->ik",
        Add(Scalar(0.0, b.shape, name="zero"), a.sum(), name="add@#$"),
        Scalar(34.0, shape=(3, 7), name="!@*(#&$"),
        name="ein",
    )

    ein_new = Einsum.from_einsum_string(
        "jk,ij->ik", Scalar(34.0, shape=(3, 7), name="!@*(#&$"), Add(b, a.sum(name="sum"), name="hi"), name="ein"
    )
    assert Updater(lambda x: Scalar(0.0, shape=x.shape, name="zero"))(
        ein_new, matcher, fancy_validate=True
    ) == Einsum.from_einsum_string(
        "jk,ij->ik",
        Scalar(34.0, shape=(3, 7), name="!@*(#&$"),
        Add(b, Scalar(0.0, shape=a.shape, name="zero").sum(name="sum"), name="hi"),
        name="ein",
    )


def test_updater_assert_any_found():
    a = Scalar(1.0, shape=(1, 2), name="a")

    to_zero = lambda x: Scalar(0.0, shape=x.shape, name="zero")

    assert a.update("a", to_zero, assert_any_found=True) == to_zero(a)
    assert a.update({"a", "b"}, to_zero, assert_any_found=True) == to_zero(a)
    assert a.update("b", to_zero) == a
    with pytest.raises(RuntimeError, match="No matches found"):
        a.update("b", to_zero, assert_any_found=True)

    assert IterativeMatcher("a").update(a, to_zero, assert_any_found=True) == to_zero(a)
    assert IterativeMatcher({"a", "b"}).update(a, to_zero, assert_any_found=True) == to_zero(a)
    assert IterativeMatcher("b").update(a, to_zero) == a
    with pytest.raises(RuntimeError, match="No matches found"):
        IterativeMatcher("b").update(a, to_zero, assert_any_found=True)

    assert IterativeMatcher("a").updater(to_zero, default_assert_any_found=True)(a) == to_zero(a)
    assert IterativeMatcher({"a", "b"}).updater(to_zero, default_assert_any_found=True)(a) == to_zero(a)
    assert IterativeMatcher("b").updater(to_zero)(a) == a
    with pytest.raises(RuntimeError, match="No matches found"):
        IterativeMatcher("b").updater(to_zero, default_assert_any_found=True)(a)

    assert IterativeMatcher("a").updater(to_zero)(a, assert_any_found=True) == to_zero(a)
    assert IterativeMatcher({"a", "b"}).updater(to_zero)(a, assert_any_found=True) == to_zero(a)
    assert IterativeMatcher("b").updater(to_zero)(a) == a
    with pytest.raises(RuntimeError, match="No matches found"):
        IterativeMatcher("b").updater(to_zero)(a, assert_any_found=True)

    assert IterativeMatcher("a").updater(to_zero, default_assert_any_found=True).update(a) == to_zero(a)
    assert IterativeMatcher({"a", "b"}).updater(to_zero, default_assert_any_found=True).update(a) == to_zero(a)
    assert IterativeMatcher("b").updater(to_zero).update(a) == a
    with pytest.raises(RuntimeError, match="No matches found"):
        IterativeMatcher("b").updater(to_zero, default_assert_any_found=True).update(a)

    assert IterativeMatcher("a").updater(to_zero).update(a, assert_any_found=True) == to_zero(a)
    assert IterativeMatcher({"a", "b"}).updater(to_zero).update(a, assert_any_found=True) == to_zero(a)
    assert IterativeMatcher("b").updater(to_zero).update(a) == a
    with pytest.raises(RuntimeError, match="No matches found"):
        IterativeMatcher("b").updater(to_zero).update(a, assert_any_found=True)

    lax_updater = Updater(to_zero)
    # loop to test caching behavior
    for _ in range(2):
        assert lax_updater(a, "a", assert_any_found=True) == to_zero(a)
        assert lax_updater(a, {"a", "b"}, assert_any_found=True) == to_zero(a)
        assert lax_updater(a, "b") == a
        with pytest.raises(RuntimeError, match="No matches found"):
            lax_updater(a, "b", assert_any_found=True)
        with pytest.raises(RuntimeError, match="No matches found"):
            lax_updater.update(a, "b", assert_any_found=True)

    strict_updater = Updater(to_zero, default_assert_any_found=True)
    for _ in range(2):
        assert strict_updater(a, "a") == to_zero(a)
        assert strict_updater(a, {"a", "b"}) == to_zero(a)
        assert strict_updater(a, "b", assert_any_found=False) == a
        with pytest.raises(RuntimeError, match="No matches found"):
            strict_updater(a, "b")
        with pytest.raises(RuntimeError, match="No matches found"):
            strict_updater.update(a, "b")


def test_expander_assert_any_found():
    a = Scalar(1.0, shape=(3, 2), name="a")

    to_zero_expand = lambda x: Scalar(0.0, shape=(5, 3, 2), name="zero")

    e_a = Expander(("a", to_zero_expand), default_assert_any_found=True)
    e_a_b_1 = Expander(({"a", "b"}, to_zero_expand), default_assert_any_found=True)
    e_a_b_2 = Expander(("a", to_zero_expand), ("b", to_zero_expand), default_assert_any_found=True)
    e_b_lax = Expander(("b", to_zero_expand))
    e_b_strict = Expander(("b", to_zero_expand), default_assert_any_found=True)

    # loop to test caching behavior
    for _ in range(2):
        assert e_a(a) == to_zero_expand(a)
        assert e_a_b_1(a) == to_zero_expand(a)
        assert e_a_b_2(a) == to_zero_expand(a)
        assert e_b_lax(a) == a
        with pytest.raises(RuntimeError, match="No matches found"):
            e_b_strict(a)

    e_a = Expander(("a", to_zero_expand))
    e_a_b_1 = Expander(({"a", "b"}, to_zero_expand))
    e_a_b_2 = Expander(("a", to_zero_expand), ("b", to_zero_expand))
    e_b = Expander(("b", to_zero_expand))

    # loop to test caching behavior
    for _ in range(2):
        assert e_a(a, assert_any_found=True) == to_zero_expand(a)
        assert e_a_b_1(a, assert_any_found=True) == to_zero_expand(a)
        assert e_a_b_2(a, assert_any_found=True) == to_zero_expand(a)
        assert e_b(a) == a
        with pytest.raises(RuntimeError, match="No matches found"):
            e_b(a, assert_any_found=True)


def test_batch():
    a = Scalar(1.0, shape=(1, 2), name="a")
    b = Scalar(2.0, shape=(2, 3), name="b")
    ein = Einsum.from_einsum_string("ij,jk->ik", a, b, name="ein")
    soft = GeneralFunction.new_by_name(ein, spec_name="softmax", name="soft")
    added = Add(ein, soft, name="fin")

    scalar_batch = lambda x: x.cast_scalar().evolve_shape((5, *x.shape))

    batched = Expander((a, scalar_batch))(added)
    # batched.print()
    assert batched.shape == (5,) + added.shape
    assert batched.name == added.name

    batched = Expander((Matcher.circuits(a, b), scalar_batch))(added)
    # batched.print()
    assert batched.shape == (5,) + added.shape
    assert batched.name == added.name
    canon_ein = Matcher(Einsum).get_unique(batched).cast_einsum().normalize_ints()
    assert canon_ein.all_input_axes() == [(0, 1, 2), (0, 2, 3)]
    assert canon_ein.out_axes == (0, 1, 3)
    assert canon_ein.name == ein.name

    batched = Expander((a, scalar_batch), (b, scalar_batch))(added)
    # batched.print()
    assert batched.shape == (5,) + added.shape
    assert batched.name == added.name
    canon_ein = Matcher(Einsum).get_unique(batched).cast_einsum().normalize_ints()
    assert canon_ein.all_input_axes() == [(0, 1, 2), (0, 2, 3)]
    assert canon_ein.out_axes == (0, 1, 3)
    assert canon_ein.name == ein.name

    batched = Expander((a, scalar_batch), (b, scalar_batch), suffix="batch_suffix")(added)
    # batched.print()
    assert batched.shape == (5,) + added.shape
    assert batched.name == added.name + "_batch_suffix"
    canon_ein = Matcher(Einsum).get_unique(batched).cast_einsum().normalize_ints()
    assert canon_ein.all_input_axes() == [(0, 1, 2), (0, 2, 3)]
    assert canon_ein.out_axes == (0, 1, 3)
    assert canon_ein.name == ein.name + "_batch_suffix"


def test_restrict_module_spec():
    args = {"ln.w.bias": Array.randn(17), "ln.w.scale": Array.randn(17), "ln.input": Array.randn(17)}
    mod = Module(layernorm.spec, "ln_out", **args)
    assert new_traversal(start_depth=1).filter_module_spec().get(mod) == set(args.values())

    mod = module_new_bind(
        TransformerBlockParams().get().body,
        ("a.ln.w.bias", Array.randn(17)),
        ("a.ln.w.scale", Array.randn(17)),
        ("b.input", Array.randn(5, 17)),
    )
    all_a_inputs = (
        Matcher("a.norm")
        .chain(new_traversal(start_depth=1).filter_module_spec())
        .get(TransformerBlockParams().get().body)
    )
    assert all_a_inputs == Matcher({"a.norm.input", "a.ln.w.scale", "a.ln.w.bias"}).get(mod)

    all_mlp_sub_inputs = (
        Matcher("m.norm_call")
        .chain(restrict_sl(Matcher(Module).new_not(), depth_slice=S[1:]).filter_module_spec())
        .get(TransformerBlockParams().get().body)
    )
    assert all_mlp_sub_inputs == Matcher({"m.norm.input", "m.ln.w.bias", "m.ln.w.scale"}).get(mod)

    all_mlp_sub_inputs = (
        Matcher("m.norm_call")
        .chain(restrict_sl(Matcher(Module).new_not(), depth_slice=S[1:]).filter_module_spec())
        .get(mod)
    )
    assert all_mlp_sub_inputs == Matcher({"m.norm.input", "m.ln.w.bias", "m.ln.w.scale"}).get(mod)

    scale_val = Array.randn(17, name="arr_scale").rearrange_str("a->a", name="perm")
    mod_expand = module_new_bind(
        TransformerBlockParams().get().body,
        ("m.ln.w.scale", scale_val),
    ).substitute()

    all_mlp_sub_inputs = (
        Matcher("m.norm_call")
        .chain(restrict_sl(Matcher(Module).new_not(), depth_slice=S[1:]).filter_module_spec())
        .get(mod_expand)
    )
    assert all_mlp_sub_inputs == Matcher({"m.norm.input", "m.ln.w.bias"}).get(mod).union(all_children(scale_val))


def test_spec_circuit_matcher():
    mod = module_new_bind(
        TransformerBlockParams().get().body,
        ("a.ln.w.bias", Array.randn(17)),
        ("a.ln.w.scale", Array.randn(17)),
        ("b.input", Array.randn(5, 17)),
    )

    print([x.name for x in IterativeMatcher(Module).spec_circuit_matcher().get(mod)])
    assert IterativeMatcher(Module).spec_circuit_matcher().get(mod) == Matcher(
        {"b", "m.norm_call", "m", "ln", "a.norm_call", "not_mask", "a", "a.head.on_inp", "a.head"}
    ).get(mod, fancy_validate=True)

    assert IterativeMatcher("m.norm").spec_circuit_matcher().get(mod) == Matcher({"ln"}).get(mod)


def test_children_matcher():
    mod = module_new_bind(
        TransformerBlockParams().get().body,
        ("a.ln.w.bias", Array.randn(17)),
        ("a.ln.w.scale", Array.randn(17)),
        ("b.input", Array.randn(5, 17)),
    )

    print(mod.get_unique("m.norm").children)

    assert IterativeMatcher("m.norm").children_matcher({0}).get(mod) == Matcher({"ln"}).get(mod)

    assert IterativeMatcher("m.norm").children_matcher({0, 1}).get(mod) == Matcher({"ln", "m.norm.input"}).get(mod)
    assert IterativeMatcher("m.norm").children_matcher({0, 3}).get(mod) == Matcher({"ln", "m.ln.w.scale"}).get(mod)
    assert IterativeMatcher("m.norm").children_matcher({2, 1}).get(mod) == Matcher({"m.norm.input", "m.ln.w.bias"}).get(
        mod
    )

    with pytest.raises(IterativeMatcherChildNumbersOutOfBoundsError):
        IterativeMatcher("m.norm").children_matcher({4}).get(mod)
    with pytest.raises(IterativeMatcherChildNumbersOutOfBoundsError):
        IterativeMatcher("m.norm").children_matcher({7}).get(mod)


def test_module_arg_matcher():
    mod = module_new_bind(
        TransformerBlockParams().get().body,
        ("a.ln.w.bias", Array.randn(17)),
        ("a.ln.w.scale", Array.randn(17)),
        ("b.input", Array.randn(5, 17)),
    )

    PrintOptions().print(mod)

    assert IterativeMatcher("m.norm").module_arg_matcher("ln.input").get(mod) == Matcher({"m.norm.input"}).get(mod)
    assert IterativeMatcher("m.norm").module_arg_matcher("ln.w.bias").get(mod) == Matcher({"m.ln.w.bias"}).get(mod)
    assert IterativeMatcher("m.norm").module_arg_matcher("ln.w.scale").get(mod) == Matcher({"m.ln.w.scale"}).get(mod)
    assert IterativeMatcher("m.norm").module_arg_matcher({"ln.input", "ln.w.bias", "ln.w.scale"}).get(mod) == Matcher(
        {"m.norm.input", "m.ln.w.bias", "m.ln.w.scale"}
    ).get(mod)

    # module arg matchers match by argument *symbol*, not argument *content*, and hence cannot match
    # the module spec (which does not have an argument symbol)
    assert IterativeMatcher("m.norm").module_arg_matcher("ln").get(mod) == set()

    assert IterativeMatcher("a.norm_call").module_arg_matcher("a.input").get(mod) == Matcher("a.norm").get(mod)

    # Check that we only search for arguments under modules which match the first matcher
    assert IterativeMatcher("a.norm_call").module_arg_matcher("ln.input").get(mod) == set()
    assert IterativeMatcher("m.norm").module_arg_matcher("a.input").get(mod) == set()


def test_print_matcher():
    setups = [
        (Matcher("a_name"), '''"a_name"'''),
        (Matcher({"a_name", "b_other"}), """{"a_name", "b_other"}"""),
        (Matcher(set[Type[Circuit]]({Scalar, Einsum})), """{Einsum, Scalar}"""),
        (
            Matcher.all(Matcher(set[Type[Circuit]]({Scalar, Einsum})), Matcher({"a_name", "b_other"})),
            """All({Einsum, Scalar}, {"a_name", "b_other"})""",
        ),
        (
            Matcher.any(Matcher(set[Type[Circuit]]({Scalar, Einsum})), Matcher.new_not(Matcher({"a_name", "b_other"}))),
            """Any({Einsum, Scalar}, Not {"a_name", "b_other"})""",
        ),
        (Matcher.true_matcher(), """Always"""),
        (Matcher.false_matcher(), """Never"""),
        (Matcher(Scalar(1.0, shape=(1, 2), name="a_name")), """'a_name' Scalar 80114e"""),
        (
            Matcher(
                set[Circuit]({Scalar(1.0, shape=(1, 2), name="a_name"), Scalar(2.0, shape=(2, 3), name="b_other")})
            ),
            """{'a_name' Scalar 80114e, 'b_other' Scalar c85657}""",
        ),
        (Matcher.regex("^.$"), """re-escdot'^.$'"""),
        (Matcher.regex("^.$", escape_dot=False), """re'^.$'"""),
    ]
    for matcher, target in setups:
        print(matcher)
        assert repr(matcher) == target
    print(Matcher(lambda x: x.name.startswith("hi")))


def test_print_iterative_matcher():
    setups = [
        (IterativeMatcher.new_chain("a_name", "b_name"), """["a_name" -> "b_name"]"""),
        (
            IterativeMatcher.any(
                IterativeMatcher.new_chain("a_name", "b_name"), IterativeMatcher.new_chain("b_name", "a_name")
            ),
            """Any(["a_name" -> "b_name"], ["b_name" -> "a_name"])""",
        ),
        (IterativeMatcher("a_name"), '''"a_name"'''),
        (IterativeMatcher.new_children_matcher("a_name", frozenset([0, 1])), """Children("a_name", [0,1])"""),
        (IterativeMatcher.new_module_arg_matcher("a_name", "b_name"), """ModuleArg("a_name", "b_name")"""),
        (IterativeMatcher.new_spec_circuit_matcher("a_name"), """SpecCircuit("a_name")"""),
        (IterativeMatcher("a_name").filter_module_spec(), """NoModuleSpec("a_name")"""),
        (
            new_traversal(start_depth=1, end_depth=3),
            """Restrict{depth_range: 1:3}""",
        ),
        (
            restrict(
                restrict(
                    restrict(
                        IterativeMatcher.new_chain(
                            "n;asdkjfn;lasknfl;wkenf;alknklk;lasdknfl;an", "b", "c", "d", "efghijk"
                        ),
                        start_depth=1,
                        end_depth=10,
                    ),
                    term_early_at={"asdfasdfasdf", "1234567890", "12345678912345678901234567890", "qwertyuiop"},
                ),
                term_if_matches=True,
            ),
            """Restrict{
    matcher: Restrict{    
        matcher: Restrict{        
            depth_range: 1:10        
            matcher: ["n;asdkjfn;lasknfl;wkenf;alknklk;lasdknfl;an" -> "b" -> "c" -> "d" -> "efghijk"]        
        }    
        term_at: {        
            "1234567890"        
            "12345678912345678901234567890"        
            "asdfasdfasdf"        
            "qwertyuiop"        
        }    
    }
    term_if_matches: true
}""",
        ),
    ]
    for matcher, target in setups:
        print(matcher)
        assert repr(matcher) == target
    print(IterativeMatcher.any("a_name", lambda x: False))


def test_iterative_matcher_name_append():
    TransformerBlockParams().get().body.print()
    matcher = IterativeMatcher.new_chain("b.m", "m.norm_call", GeneralFunction)
    append_matchers_to_names(TransformerBlockParams().get().body, matcher).print()


def test_matcher_debug():
    matcher = IterativeMatcher.new_chain("b.mlp", "gelu_mlp", GeneralFunction)
    print_matcher_debug(TransformerBlockParams().get().body, matcher, PrintOptions())
    print_matcher_debug(TransformerBlockParams().get().body, matcher, PrintOptions(arrows=True))
    print_matcher_debug(
        rc.Parser()(
            """
'b.mlp' Einsum ,,->
  'gelu_mlp' Einsum ,->
    's0' [] Symbol
    's1' [] Symbol
  'gelu_mlp'
  's0'
    """
        ),
        matcher,
        PrintOptions(arrows=True),
    )


def test_iterative_matcher_and():
    a = Scalar(1.0, shape=(1, 2), name="a_name")
    b = Scalar(2.0, shape=(2, 3), name="b_other")
    c = Einsum.from_einsum_string("ab,bc->ac", a, b, name="c_name")

    def check_for_same(m: IterativeMatcher, c: Circuit, found: bool):
        res = m.match_iterate(c)
        assert res.updated is None
        assert res.found == found

    check_for_same(IterativeMatcher("a_name") & "b_other", a, False)
    check_for_same(IterativeMatcher("a_name") & "b_other", a, False)
    check_for_same(IterativeMatcher("a_name") | "b_other", a, True)
    check_for_same("b_other" & IterativeMatcher("a_name"), a, False)
    check_for_same("b_other" | IterativeMatcher("a_name"), a, True)
    check_for_same(IterativeMatcher.all("a_name", Scalar), a, True)
    check_for_same(IterativeMatcher.all("a_name", Scalar), b, False)

    check_for_same(IterativeMatcher.all("a_name", Scalar), a, True)
    check_for_same(IterativeMatcher.all("a_name", Scalar), b, False)
    check_for_same(IterativeMatcher.all("a_name", Scalar), c, False)

    check_for_same(IterativeMatcher.all("a_name"), a, True)
    check_for_same(IterativeMatcher.all("a_name"), b, False)
    check_for_same(IterativeMatcher.all("a_name"), c, False)

    out = IterativeMatcher.all("hi", restrict("a_name", end_depth=3)).match_iterate(c)
    assert not out.found
    up = restrict("a_name", end_depth=3).match_iterate(c).updated
    assert up is not None
    assert not isinstance(up, list)
    assert not isinstance(up, Finished)
    assert out.updated == IterativeMatcher.all("hi", up)

    out = IterativeMatcher.all(Scalar, restrict("a_name", end_depth=3)).match_iterate(a)
    assert out.found
    up = restrict("a_name", end_depth=3).match_iterate(c).updated
    assert up is not None
    assert not isinstance(up, list)
    assert not isinstance(up, Finished)
    assert out.updated == IterativeMatcher.all(Scalar, up)

    out = IterativeMatcher.all(Einsum, restrict("a_name", end_depth=3)).match_iterate(a)
    assert not out.found
    up = restrict("a_name", end_depth=3).match_iterate(c).updated
    assert up is not None
    assert not isinstance(up, list)
    assert not isinstance(up, Finished)
    assert out.updated == IterativeMatcher.all(Einsum, up)

    out = IterativeMatcher.all(Einsum, IterativeMatcher.term()).match_iterate(a)
    assert not out.found
    assert out.updated == FINISHED

    out = IterativeMatcher.all(Scalar, IterativeMatcher.term()).match_iterate(a)
    assert not out.found
    assert out.updated == FINISHED

    out = IterativeMatcher.all(IterativeMatcher.term(), Scalar).match_iterate(a)
    assert not out.found
    assert out.updated == FINISHED

    out = IterativeMatcher.all(Scalar, IterativeMatcher.term(match_next=True)).match_iterate(a)
    assert out.found
    assert out.updated == FINISHED

    out = IterativeMatcher.all(Scalar, restrict("c_name", end_depth=3).children_matcher({1})).match_iterate(c)
    assert not out.found
    up = restrict("c_name", end_depth=3).children_matcher({1}).match_iterate(c).updated
    assert up is not None
    assert isinstance(up, list)
    new_up: List[IterativeMatcher] = []
    for u in up:
        assert not isinstance(u, Finished)
        new_up.append(u)

    assert out.updated == [IterativeMatcher.all(Scalar, new_up[0]), IterativeMatcher.all(Scalar, new_up[1])]

    out = IterativeMatcher.all(
        Scalar, IterativeMatcher.new_func(lambda _: IterateMatchResults([FINISHED, IterativeMatcher(True)], found=True))
    ).match_iterate(c)
    assert not out.found

    assert out.updated == [FINISHED, IterativeMatcher.all(Scalar, IterativeMatcher(True))]

    with pytest.raises(TypeError, match="coerced"):
        IterativeMatcher("a_name") and IterativeMatcher("b_other")


def test_get_paths():
    a = Scalar(1.0, name="match")
    a_bis = Add(name="other")
    b = Add(a, a_bis, name="parent")
    c = Add(b, name="grandparent")

    expected = {a: [a, b, c]}
    assert Getter().get_paths(c, Matcher(Scalar)) == expected
    assert Matcher(Scalar).get_paths(c) == expected
    assert c.get_paths(Matcher(Scalar)) == expected
    assert Getter().bind(Scalar).get_paths(c) == expected

    assert Getter().get_paths(c, Matcher(Array)) == {}

    assert Getter().get_paths(c, {a, c, a_bis}) == {a: [a, b, c], a_bis: [a_bis, b, c], c: [c]}


def test_get_all_paths():
    a = Scalar(1.0, name="match")
    a_bis = Add(name="other")
    b = Add(a, a_bis, name="parent")
    c = Add(b, name="grandparent")

    expected = {a: [[a, b, c]]}
    assert Getter().get_all_paths(c, Matcher(Scalar)) == expected
    assert Matcher(Scalar).get_all_paths(c) == expected
    assert c.get_all_paths(Matcher(Scalar)) == expected
    assert Getter().bind(Scalar).get_all_paths(c) == expected

    assert Getter().get_all_paths(c, Matcher(Array)) == {}

    assert Getter().get_all_paths(c, {a, c, a_bis}) == {a: [[a, b, c]], a_bis: [[a_bis, b, c]], c: [[c]]}

    d = Add(a, a)
    assert Getter().get_all_paths(d, Matcher(Scalar)) == {a: [[a, d], [a, d]]}

    e = Add(c, b)
    r = Getter().get_all_paths(e, Matcher(Scalar))
    assert r == {a: [[a, b, c, e], [a, b, e]]} or r == {a: [[a, b, e], [a, b, c, e]]}


def test_non_callable_errors_on_construct():
    class RandomObj:
        ...

    obj = RandomObj()

    with pytest.raises(TypeError):
        Matcher(obj)  # type: ignore
    with pytest.raises(TypeError):
        IterativeMatcher(obj)  # type: ignore
    with pytest.raises(TypeError):
        IterativeMatcher.new_func(obj)  # type: ignore
