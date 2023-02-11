from __future__ import annotations

import functools
import hashlib
import itertools
import math
from collections import Counter
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TypeVar, Union, cast, overload

import torch
from typing_extensions import Concatenate, Never, ParamSpec

import rust_circuit.optional as op
from rust_circuit.py_utils import assert_never, check_cast

from ._rust import (
    Add,
    Circuit,
    Concat,
    ConstructError,
    Einsum,
    Index,
    IterativeMatcherIn,
    Matcher,
    MatcherIn,
    PushDownIndexEinsumNoopError,
    Rearrange,
    RearrangeSpec,
    Scalar,
    TorchAxisIndex,
    add_elim_zeros,
    add_flatten_once,
    concat_elim_identity,
    distribute,
    einsum_elim_identity,
    einsum_flatten,
    einsum_flatten_once,
    einsum_of_permute_merge,
    einsum_permute_to_rearrange,
)
from ._rust import extract_add as extract_add_rust
from ._rust import (
    index_elim_identity,
    make_broadcast,
    new_traversal,
    permute_of_einsum_merge,
    pull_concat_once_raw,
    push_down_index_once,
    rearrange_elim_identity,
    rearrange_fuse,
    traverse_until_depth,
)
from .py_utils import I

T = TypeVar("T")


def dedup_with_order(seq: Iterable[T]) -> List[T]:
    seen = set()
    out: List[T] = []
    for x in seq:
        if x in seen:
            continue
        else:
            seen.add(x)
            out.append(x)

    return out


def normalize_index(i: int, count: int):
    assert count >= 0, count
    if i >= 0:
        assert i < count, (i, count)
    else:
        assert i >= -count, (i, count)
    return i % count


Rewrite = Callable[[Circuit], Circuit]


class CircuitTypeError(TypeError):
    ...


class RewriteError(Exception):
    ...


class ImpossibleRewriteError(RewriteError):
    ...


class NoopRewriteError(RewriteError):
    ...


CircuitT = TypeVar("CircuitT", bound=Circuit)
CircuitT2 = TypeVar("CircuitT2", bound=Circuit)
P = ParamSpec("P")


def try_transform_wrapper(
    fn: Callable[Concatenate[CircuitT, P], CircuitT2],
    name: Optional[str] = None,
):
    """
    Wraps rewrites so that they noop rather than :class:`CircuitTypeError` or :class:`ImpossibleRewriteError`
    when called on inappropriate functions
    """

    @overload
    def wrapped_func(c: CircuitT, *args: P.args, **kwargs: P.kwargs) -> CircuitT2:
        ...

    @overload
    def wrapped_func(c: Circuit, *args: P.args, **kwargs: P.kwargs) -> Circuit:
        ...

    # This signature will also appear in the overloaded type definition, but it will never match the passed type so it
    # won't matter.
    def wrapped_func(c: Never, *args, **kwargs) -> Never:  # type: ignore
        try:
            return fn(c, *args, **kwargs)  # type: ignore
        except RewriteError:
            return c  # type: ignore
        except ConstructError:
            return c  # type: ignore
        except CircuitTypeError:
            return c  # type: ignore
        except TypeError:  # TODO: I don't currently see how to make thie exception more specific : (
            return c  # type: ignore
        except NotImplementedError:
            return c  # type: ignore

    if name is not None:
        wrapped_func.__name__ = name
    else:
        wrapped_func.__name__ = f"try_{fn.__name__}"

    return wrapped_func


def try_transform_wrapper_no_params(
    fn: Callable[[CircuitT], CircuitT2],
    name: Optional[str] = None,
):
    "like `try_transform_wrapper` but typechecks for overloaded functions"
    return try_transform_wrapper(fn, name=name)


@overload
def eliminate_zeros(circuit: Einsum) -> Scalar:
    ...


@overload
def eliminate_zeros(circuit: Rearrange) -> Scalar:
    ...


@overload
def eliminate_zeros(circuit: Index) -> Scalar:
    ...


@overload
def eliminate_zeros(circuit: Add) -> Union[Add, Scalar]:
    ...


def eliminate_zeros(circuit):
    """
    Applies the identities `x + 0 = x` and `x * 0 = 0` to the top level of the circuit.

    Also `reshape(zeros(...)) = zeros(new_shape)`, `index(zeros(...)) = zeros(new_shape)`.

    Raises error for types this transform doesn't apply to, or if it would noop
    """

    def is_zero(c: Circuit):
        return op.map(c.maybe_scalar(), lambda x: x.is_zero())

    circuit = check_cast((Einsum, Rearrange, Index, Add), circuit, exception=CircuitTypeError)
    if isinstance(circuit, (Einsum, Rearrange, Index)):
        if any(is_zero(c) for c in circuit.children):
            return Scalar(0.0, shape=circuit.shape, name=circuit.name)
        else:
            raise ImpossibleRewriteError
    elif isinstance(circuit, Add):
        zeros: Set[Circuit] = {k for k in circuit.children if is_zero(k)}
        if not zeros:
            raise ImpossibleRewriteError("No zeros to drop")
        return drop_from_add(circuit, zeros)
    else:
        assert_never(circuit)


# Needed because `eliminate_zeros` is overloaded
try_eliminate_zeros = try_transform_wrapper_no_params(eliminate_zeros)


def get_einsum_shape_d(circuit: Einsum):
    shape_d: Dict[int, int] = {}
    for c, axes in circuit.args:
        shape_d.update(dict(zip(axes, c.shape)))
    return shape_d


def rearrange_traced(
    node: Circuit,
    inp: List[List[int]],
    out: List[List[int]],
    sizes: List[Optional[int]],
    name_rearrange: Optional[str] = None,
    name: Optional[str] = None,
):
    """trace over repeated outputs, primarily for internal usage"""

    deduped_out = [list(t) for t in dedup_with_order([tuple(l) for l in out])]
    new = Rearrange(
        node,
        RearrangeSpec(inp, deduped_out, sizes),
        name=op.unwrap_or(name_rearrange, f"{node.name}_rearrange_for_trace"),
    )
    full_name = op.unwrap_or(name, f"{node.name}_rearrange_trace")
    if deduped_out == out:
        return new.rename(full_name).cast_rearrange()
    deduped_nums = tuple(range(len(deduped_out)))
    out_item_to_num = dict(zip([tuple(l) for l in deduped_out], deduped_nums))
    return Einsum(
        (new, deduped_nums),
        out_axes=tuple(out_item_to_num[tuple(x)] for x in out),
        name=full_name,
    )


def drop_mul_ones(circuit: Einsum, idxs: Optional[Iterable[int]] = None) -> Union[Einsum, Rearrange]:
    """
    Applies the identity `x*1 = x` to the top level of an Einsum circuit

    Raises error for other types
    """

    circuit = circuit.cast_einsum()

    def is_one(c: Circuit):
        return op.map(c.maybe_scalar(), lambda x: x.is_one())

    if len(circuit.args) <= 1:
        raise ImpossibleRewriteError("only should be applied to multiple argument einsum")

    to_remove = op.unwrap_or(
        op.map(idxs, lambda idxs_v: set(normalize_index(i, len(circuit.args)) for i in idxs_v)),
        {i for i, c in enumerate(circuit.children) if is_one(c)},
    )

    if len(to_remove) == 0:
        raise ImpossibleRewriteError("no Ones to remove")

    for i in to_remove:
        assert is_one(
            circuit.children[i]
        ), "maybe you passed idxs into `drop_mul_ones` which didn't correspond to ones?"

    removed_nums = set[int]().union(*(circuit.all_input_axes()[i] for i in to_remove))
    retained_nums = set[int]().union(*(axes for j, axes in enumerate(circuit.all_input_axes()) if j not in to_remove))
    no_longer_present = removed_nums - retained_nums

    new_args = [arg for i, arg in enumerate(circuit.args) if i not in to_remove]

    new_out_axes = list(filter(lambda x: x not in no_longer_present, circuit.out_axes))

    if len(no_longer_present) == 0:
        return Einsum(*new_args, out_axes=tuple(new_out_axes), name=circuit.name)

    new_out_axes = dedup_with_order(new_out_axes)

    # separate evolve to avoid renumbering
    new = Einsum(*new_args, out_axes=tuple(new_out_axes), name=f"{circuit.name}_drop_ones")

    shape_d = get_einsum_shape_d(circuit)

    # special broadcast using axes
    out = rearrange_traced(
        new,
        [[a] for a in new_out_axes],
        [[a] for a in circuit.out_axes],
        [(shape_d[a] if a in no_longer_present else None) for a in sorted(shape_d.keys())],
        name=circuit.name,
    )

    multiplier = 1.0
    for num in no_longer_present:
        if num not in circuit.out_axes:
            multiplier *= shape_d[num]

    if multiplier == 1.0:
        return out

    return Einsum.scalar_mul(
        out.rename(f"{circuit.name}_unscale"),
        scalar=multiplier,
        name=circuit.name,
    )


try_drop_mul_ones = try_transform_wrapper(drop_mul_ones)


def rearrange_of_const_to_const(circuit: Rearrange):
    """converts repeated constant to a constant"""

    circuit = circuit.cast_rearrange()
    node = circuit.node.cast_scalar()
    return Scalar(node.value, shape=circuit.shape, name=circuit.name)


try_rearrange_of_const_to_const = try_transform_wrapper(rearrange_of_const_to_const)


def flatten_adds_matching(c: Add, match: MatcherIn, allow_noop: bool = False) -> Add:
    """
    Applies the identity `x + (y + z) = x + y + z` to the top level of an Add circuit (when `(y + z)` matches the matcher function).

    Raises error for other types
    """
    c = c.cast_add()

    matcher = Matcher(match)

    out = []
    found_match = False
    for child in c.children:
        if isinstance(child, Add) and matcher(child):
            for grandchild in child.children:
                out.append(grandchild)
            found_match = True
        else:
            out.append(child)
    if not found_match and not allow_noop:
        raise ImpossibleRewriteError

    return Add(*out, name=c.name)


def flatten_adds(c: Add) -> Add:
    """
    Same as flatten_adds_matching, but uses all matchings it can
    """
    out = add_flatten_once(c.cast_add())
    if out is None:
        raise ImpossibleRewriteError
    return out


try_flatten_adds = try_transform_wrapper(flatten_adds)


def is_rearrange(c: Einsum) -> bool:
    """
    If this einsum has one argument and no repeated axes, it's a rearrange.
    """
    return len(c.args) == 1 and len(c.args[0][1]) == len(set(c.args[0][1])) and len(c.out_axes) == len(set(c.out_axes))


def einsum_to_rearrange(c: Einsum) -> Rearrange:
    """
    Replaces Einsum with equivalent Rearrange.
    Raises:
        ImpossibleRewriteError: If input Einsum cannot be converted to rearrange.
    """
    out = einsum_permute_to_rearrange(c.cast_einsum())

    if out is None:
        raise ImpossibleRewriteError
    return out.cast_rearrange()


try_einsum_to_rearrange = try_transform_wrapper(einsum_to_rearrange)

T = TypeVar("T")


def flatten(l: List[List[T]]) -> List[T]:
    return [x for sl in l for x in sl]


def explicit_squeeze(c: Rearrange, suffix: str = "_no_squeeze") -> Einsum:
    spec = c.spec

    # Squeeed are empty lists
    nb_squeezed_dims = len([l for l in spec.input_ints if not l])
    if nb_squeezed_dims == 0:
        raise ImpossibleRewriteError("No squeezed dimensions")

    new_spec = RearrangeSpec(spec.input_ints, spec.output_ints + [[] for _ in range(nb_squeezed_dims)], spec.int_sizes)

    all_axes = tuple(range(len(new_spec.output_ints)))

    return Einsum(
        (Rearrange(c.node, new_spec, name=f"{c.name}{suffix}"), all_axes),
        out_axes=all_axes[: c.rank],
        name=c.name,
    )


try_explicit_squeeze = try_transform_wrapper(explicit_squeeze)


def remove_noop_rearrange(c: Union[Einsum, Rearrange], keep_name: bool = False) -> Circuit:
    """
    Attempts to strip rearranges that noop at the top level of the circuit for trivial Einsums and Rearranges

    Errors for other types, ImpossibleRewriteErrors where there isn't a noop rearrange

    Note: for einsum with no args, returns 1, as in remove_empty_einsum
    Note: for einsum with with times 1, remove the times 1
    """
    c = check_cast((Einsum, Rearrange), c, exception=CircuitTypeError)

    if isinstance(c, Einsum):
        out = einsum_elim_identity(c)
        if out is not None:
            return out
        else:
            raise ImpossibleRewriteError(f"Einsum {c} is not identity")

    out = rearrange_elim_identity(c.cast_rearrange())
    if out is not None:
        if keep_name:
            return out.rename(c.name)
        else:
            return out
    else:
        raise ImpossibleRewriteError(f"Rearrange {c} is not a noop")


try_remove_noop_rearrange = try_transform_wrapper(remove_noop_rearrange)


# remove_add_times_one -> remove_add_single_summand
# maybe keep remove_add_times_one for backcompatibility?
# maybe make it deal with add of einsum.scalar_mul ?
# maybe use remove_add_few_input?
def remove_add_single_summand(c: Add, keep_name: bool = False) -> Circuit:
    """
    Strips add with only one summand at the top level of the circuit.

    Raises error for other circuit types, and :class:`ImpossibleRewriteError` for :class:`Add` s
    """
    c = c.cast_add()

    if c.num_children == 1:
        (out,) = c.children
        if keep_name:
            return out.rename(c.name)
        else:
            return out
    else:
        raise ImpossibleRewriteError("Input has more than 1 element")


try_remove_add_single_summand = try_transform_wrapper(remove_add_single_summand)


def remove_single_concat(c: Concat, keep_name: bool = False) -> Circuit:
    """
    Strips concat with only one argument the top level of the circuit.

    Raises error for other circuit types, and ImpossibleRewriteError for :class:`Add` s
    """
    c = c.cast_concat()

    out = concat_elim_identity(c)

    if out is not None:
        if keep_name:
            return out.rename(c.name)
        else:
            return out
    else:
        raise ImpossibleRewriteError("Input has more than 1 element")


try_remove_single_concat = try_transform_wrapper(remove_single_concat)

# TODO: add remove_inclusive_specified_slice back?
def remove_trivial_index(c: Index) -> Circuit:
    """
    Strips index which has only slice(None) at top level of circuit

    Raises error for other circuit types or an ImpossibleRewriteError when types are correct
    """
    c = c.cast_index()

    out = index_elim_identity(c)
    if out is not None:
        return out
    else:
        raise ImpossibleRewriteError


try_remove_trivial_index = try_transform_wrapper(remove_trivial_index)

# maybe remove? (already covered by remove_noop_rearrange)
def remove_empty_einsum(c: Einsum) -> Scalar:
    """
    Replaces empty einsum with One (or noop for other einsums)

    Raises error for other circuit types, ImpossibleRewriteError is not an empty Einsum
    """
    c = c.cast_einsum()
    if len(c.args) == 0:
        return Scalar(1.0, name=c.name)
    else:
        raise ImpossibleRewriteError


try_remove_empty_einsum = try_transform_wrapper(remove_empty_einsum)

# maybe revert to nodes: Set[Circuit]?
def drop_from_add(add: Add, drop_match: MatcherIn, noop_error: bool = False) -> Union[Add, Rearrange, Scalar]:
    """
    Not a rewrite! Remove specified summands.

    See drop_zero_add below
    """
    add = add.cast_add()
    drop_matcher = Matcher(drop_match)
    new_nodes = [node for node in add.children if not drop_matcher(node)]

    if noop_error:
        if add.num_children == 0:
            raise NoopRewriteError("`add` is empty")
        if len(new_nodes) == add.num_children:
            raise NoopRewriteError("No nodes removed")

    if len(new_nodes) == 0:
        return Scalar(0.0, add.shape, name=add.name)
    out: Circuit = Add(*new_nodes, name=add.name)

    if out.shape != add.shape:
        # TODO: fix name duplicate?
        braocasted = make_broadcast(out, add.shape)
        if braocasted is None:
            raise ImpossibleRewriteError
        out = braocasted.rename(out.name)
    return check_cast((Add, Rearrange, Scalar), out, exception=CircuitTypeError)


try_drop_from_add = try_transform_wrapper(drop_from_add)

# warning! the meaning changed! here it's node zero, not weight zero. Maybe change the name?
def drop_zero_add(c: Add) -> Add:
    """
    Applies the identity `x + 0 = x` at the top level of the circuit.

    Raises error for other types, NoopRewriteError if nothing to drop
    """
    c = c.cast_add()

    out = add_elim_zeros(c)
    if out is None:
        raise NoopRewriteError
    return out


try_drop_zero_add = try_transform_wrapper(drop_zero_add)

# maybe use remove_add_few_input?
def zero_empty_add(c: Add) -> Scalar:
    """
    Replaces add-with-no-summands with zero.

    Errors if wrong type, ImpossibleRewriteError if not applicable.
    """

    c = c.cast_add()

    if c.num_children > 0:
        raise ImpossibleRewriteError(f"{c.num_children} must be zero")
    return Scalar(0.0, shape=c.shape, name=c.name)


try_zero_empty_add = try_transform_wrapper(zero_empty_add)


def factor_add_of_mul_to_mul_of_add(c: Add, is_lhs: bool = True) -> Einsum:
    """
    Applies the identity `a*b + a*c = a(b + c)` (or `b*a + c*a = (b + c)*a`) at the top level of the circuit.

    Errors when type is wrong. Raises ImpossibleRewriteError where not applicable
    """
    c = c.cast_add()

    items = [k.cast_einsum() for k in c.children]

    dim_vals = {tuple((c.shape, axes) for c, axes in x.args) for x in items}
    out_axes_vals = {x.out_axes for x in items}

    if len(dim_vals) != 1:
        raise ImpossibleRewriteError(f"Summands have {len(dim_vals)} != 2 different axis specifications")

    out_axes = list(out_axes_vals)[0]
    dim_items = list(dim_vals)[0]

    if len(dim_items) != 2:
        raise ImpossibleRewriteError(f"Each summand contains the product of {len(dim_items)} != 2 things")

    args_vals = [{x.children[i] for x in items} for i in range(len(dim_items))]

    count_is_1 = [len(a) == 1 for a in args_vals]
    count_is_1_total = sum(count_is_1)

    if count_is_1_total < len(args_vals) - 1 or len(args_vals) <= 1:
        raise ImpossibleRewriteError  # TODO this is redundant

    if not count_is_1[int(not is_lhs)]:
        raise ImpossibleRewriteError("There isn't a common factor that can be factored out")

    if is_lhs:
        assert len(list(args_vals[0])) == 1
        lhs: Circuit = list(args_vals[0])[0]
        rhs: Circuit = Add(*[k.children[1] for k in items], name=f"{c.name}_new_add_from_factor")
    else:
        lhs = Add(*[k.children[0] for k in items], name=f"{c.name}_new_add_from_factor")
        assert len(list(args_vals[1])) == 1
        rhs = list(args_vals[1])[0]

    (_, lhs_axes), (_, rhs_axes) = list(dim_vals)[0]

    return Einsum(
        (lhs, lhs_axes),
        (rhs, rhs_axes),
        out_axes=out_axes,
        name=c.name,
    )


try_factor_add_of_mul_to_mul_of_add = try_transform_wrapper(factor_add_of_mul_to_mul_of_add)

# maybe add back more precise typing at the end
# TODO: check names are not broken
def pull_through_add_concat(c: Add) -> Circuit:
    """
    Replaces sum of concats with concat of sums.

    All concats must have children of the same shapes and be along the same axis.

    Errors for wrong types, ImpossibleRewriteError for things that are Adds but fail
    """

    out = pull_concat_once_raw(c.cast_add())
    if out is None:
        raise ImpossibleRewriteError
    return out


try_pull_through_add_concat = try_transform_wrapper(pull_through_add_concat)


def hash_index(index: List[TorchAxisIndex]) -> bytes:
    """Like index hash, but does not support hash_tensor_idx_by_value"""
    index_hash = hashlib.blake2b(b"<INDEX>")
    from interp.circuit.eq_by_big_hash import hash_add, hash_add_by_id

    for i in index:
        if isinstance(i, torch.Tensor):
            hash_add(index_hash, i.shape)
            hash_add_by_id(index_hash, i)
        elif isinstance(i, (int, slice)):
            hash_add(index_hash, i)
        else:
            assert_never(i)
        index_hash.update(b"<ITEM>")
    index_hash.update(b"<END>")
    return index_hash.digest()


def pull_through_add_index(c: Add) -> Index:
    """
    Replaces sum of indexes with index of sums.

    All indexes must have the same output shape and the "same" (by hash) index attr.

    Errors for other types, ImpossibleRewriteError for things that are Adds but fail
    """

    c = c.cast_add()
    items: List[Index] = []
    for k in c.children:
        if not isinstance(k, Index):
            raise ImpossibleRewriteError(f"Child {k} is not an Index")
        items.append(k)

    shapes_index_hash = {(k.node.shape, hash_index(k.idx)) for k in items}
    if len(shapes_index_hash) != 1:
        raise ImpossibleRewriteError

    return Index(
        Add(*[k.node for k in items], name=f"{c.name}_new_add_for_index"),
        items[0].idx,
        name=c.name,
    )


try_pull_through_add_index = try_transform_wrapper(pull_through_add_index)


def pull_through_add_rearrange(c: Add) -> Union[Add, Rearrange]:
    """
    Replaces sum of rearranges with rearrange of sum.

    All rearranges must have the same output shape and the same spec.

    Errors for other types, ImpossibleRewriteError for things that are Adds but fail
    """

    items: List[Rearrange] = []
    for k in c.children:
        if not isinstance(k, Rearrange):
            raise ImpossibleRewriteError(f"Child {k} is not a Rearrange")
        items.append(k)

    shapes_spec = set((k.node.shape, k.spec) for k in items)
    if len(shapes_spec) != 1:
        raise ImpossibleRewriteError(f"Children Rearranges have different specs, {shapes_spec}")

    ((_, spec),) = shapes_spec

    return Rearrange(
        Add(*[k.node for k in items], name=f"{c.name}_new_add_for_rearrange"),
        spec,
        name=c.name,
    )


try_pull_through_add_rearrange = try_transform_wrapper(pull_through_add_rearrange)


def apply_permute_axes_ints(
    inp: Tuple[int, ...], out: Tuple[int, ...], axes: Tuple[T, ...], is_on_output: bool
) -> Tuple[T, ...]:
    assert len(out) == len(axes)
    if is_on_output:
        # Swap the variables so we apply the permutation to output
        inp, out = out, inp
    return tuple(axes[out.index(inp)] for inp in inp)


def apply_permute_axes(spec: RearrangeSpec, axes: Tuple[T, ...], is_on_output: bool) -> Tuple[T, ...]:
    assert spec.is_permute()
    flat_inp = tuple(i for (i,) in spec.input_ints)
    flat_out = tuple(i for (i,) in spec.output_ints)
    return apply_permute_axes_ints(flat_inp, flat_out, axes, is_on_output=is_on_output)


def is_permute(c: Circuit):
    return rearrange_no_splits_or_joins(c, just_permute=True)


def has_no_splits_or_joins(spec: RearrangeSpec):
    return all([len(x) == 1 for x in spec.input_ints]) and all([len(x) == 1 for x in spec.output_ints])


def rearrange_no_splits_or_joins(c: Circuit, just_permute: bool = False):
    if not isinstance(c, Rearrange):
        return False

    spec = c.spec.canonicalize()
    if just_permute:
        return spec.is_permute()
    else:
        return has_no_splits_or_joins(spec)


def unpack_n(*args: Iterable[T], n: int) -> Tuple[Tuple[T, ...], ...]:
    out = tuple(zip(*args))
    if len(args) == 0:
        return ((),) * n
    assert len(out) == n
    return out


def einsum_end_num(c: Einsum) -> int:
    return max(itertools.chain(*c.all_input_axes(), c.out_axes), default=-1) + 1


def fuse_einsum_rearrange(
    c: Einsum, just_permute: bool = False, index: Optional[int] = None, force_despite_diaged_repeat: bool = False
) -> Union[Einsum, Rearrange]:
    """
    Replaces einsum of non splitting/joining rearrange with just einsum.

    Errors if wrong type, ImpossibleRewriteError otherwise

    By default, doesn't fuse if there is a output diag of a repeated dim (can't really be simplified, but we can put in different form)
    """

    def rearrange_info(rearrange: Rearrange):
        spec = rearrange.spec.canonicalize()

        flat_inp = tuple(i for (i,) in spec.input_ints)
        flat_out = tuple(i for (i,) in spec.output_ints)

        return spec, flat_inp, flat_out

    def rearrange_repeat_output_dims(rearrange: Rearrange):
        _, flat_inp, flat_out = rearrange_info(rearrange)

        return [i for i, o in enumerate(flat_out) if o not in flat_inp]

    c = c.cast_einsum()
    first_rearrange = next(
        iter(
            [
                i
                for i, (x, axes) in enumerate(c.args)
                if rearrange_no_splits_or_joins(x, just_permute=just_permute)
                and isinstance(x, Rearrange)
                and (
                    (not any(c.out_axes.count(axes[i]) > 1 for i in rearrange_repeat_output_dims(x)))
                    or force_despite_diaged_repeat
                )
            ]
        ),
        None,
    )
    re_idx = op.unwrap_or(index, first_rearrange)
    if re_idx is None:
        raise ImpossibleRewriteError

    re_idx = normalize_index(re_idx, len(c.args))
    assert re_idx is not None  # mypy strikes again

    (rearrange_v, rearrange_axes), other_args = get_info(c, re_idx)
    rearrange = rearrange_v.cast_rearrange()

    if not rearrange_no_splits_or_joins(rearrange):
        raise ImpossibleRewriteError

    spec, flat_inp, flat_out = rearrange_info(rearrange)
    assert len(rearrange_axes) == len(flat_out)

    assert len(set(flat_inp)) == len(flat_inp)
    assert len(set(flat_out)) == len(flat_out)

    extra_num = einsum_end_num(c)

    for j, inp in enumerate(flat_inp):
        if inp not in flat_out:
            assert spec.int_sizes[j] == 1
            flat_out += (inp,)
            rearrange_axes += (extra_num,)
            extra_num += 1

    (non_repeat_axes, non_repeat_out), (repeat_axes, repeat_out) = [
        unpack_n(*filter(lambda x: (x[1] in flat_inp) == is_in_inp, zip(rearrange_axes, flat_out)), n=2)
        for is_in_inp in [True, False]
    ]

    assert set(non_repeat_out) == set(flat_inp)

    new_axes = apply_permute_axes_ints(flat_inp, non_repeat_out, non_repeat_axes, is_on_output=False)

    out_other_args = other_args
    if len(repeat_out) != 0:
        out_other_args += (
            (
                Scalar(1.0, tuple(spec.int_sizes[i] for i in repeat_out), name=f"{c.name}_ones_from_repeat"),
                repeat_axes,
            ),
        )

    return reconstruct_by(c, re_idx, rearrange.node, new_axes, out_other_args)


try_fuse_einsum_rearrange = try_transform_wrapper(fuse_einsum_rearrange)

# maybe add back index?
def fuse_einsum_permute(c: Einsum) -> Einsum:
    """
    Replaces einsum of permute with just einsum.

    Errors if wrong type, ImpossibleRewriteError otherwise
    """
    out = einsum_of_permute_merge(c)
    if out is None:
        raise ImpossibleRewriteError
    return out


try_fuse_einsum_permute = try_transform_wrapper(fuse_einsum_permute)


def fuse_permute_einsum(c: Rearrange) -> Einsum:
    """
    Replaces permute of einsum with just einsum.
    (we could generalize to repeating also via ones, but we'll avoid this)

    Errors if wrong type, ImpossibleRewriteError otherwise
    """
    out = permute_of_einsum_merge(c)
    if out is None:
        raise ImpossibleRewriteError
    return out


try_fuse_permute_einsum = try_transform_wrapper(fuse_permute_einsum)


def fuse_rearrange(c: Rearrange) -> Rearrange:
    """
    Fuses two rearranges into a single rearrange.

    Errors if both of the rearranges contain splits or joins (fuse is not always possible in this case).
    """
    out = rearrange_fuse(c.cast_rearrange())
    if out is None:
        raise ImpossibleRewriteError(
            "Both rearranges contain a split/join"
        )  # TODO: check that rearrange fuse is still as general
    return out


try_fuse_rearrange = try_transform_wrapper(fuse_rearrange)


def fuse_single_einsum(c: Einsum) -> Einsum:
    """
    Replaces single einsum of einsum with just 1 einsum.

    Errors if wrong type, ImpossibleRewriteError otherwise
    """
    c = c.cast_einsum()
    if len(c.args) == 1 and isinstance(c.children[0], Einsum):
        out = einsum_flatten_once(c)
        if out is None:
            raise ImpossibleRewriteError
        return out
    else:
        raise ImpossibleRewriteError


try_fuse_single_einsum = try_transform_wrapper(fuse_single_einsum)


def fuse_einsum_single(circ: Einsum) -> Einsum:
    """
    Replaces einsum of single einsum with just 1 einsum.

    Errors if type is wrong and ImpossibleRewriteError if not applicable.
    """
    circ = circ.cast_einsum()
    to_not_fuse = [c for c in circ.children if not (isinstance(c, Einsum) and len(c.args) == 1)]
    if len(to_not_fuse) == len(circ.children):
        raise ImpossibleRewriteError("There are no Einsums with 1 child under this one")

    new_c = einsum_flatten(circ, new_traversal(end_depth=3, term_early_at=set(to_not_fuse)))
    return new_c


try_fuse_einsum_single = try_transform_wrapper(fuse_einsum_single)


def get_info(einsum: Einsum, idx: int):
    return einsum.args[idx], tuple(einsum.args[:idx] + einsum.args[idx + 1 :])


def reconstruct_by(
    einsum: Einsum,
    idx: int,
    x: Circuit,
    axes: Tuple[int, ...],
    other_args: Tuple[Tuple[Circuit, Tuple[int, ...]], ...],
    new_name: Optional[str] = None,
    new_out_axes: Optional[Tuple[int, ...]] = None,
):
    new_args = other_args[:idx] + ((x, axes),) + other_args[idx:]
    out = einsum.evolve(args=new_args)
    if new_out_axes is not None:
        out = out.evolve(out_axes=new_out_axes)
    if new_name is not None:
        out = cast(Einsum, out.rename(new_name))
    return out


def split_einsum_concat(
    c: Einsum,
    element_index: int,
    index_groups: List[List[int]],
    group_names: Optional[Sequence[str]] = None,
    axis_name: Optional[str] = None,
    use_axis_name: bool = True,
    push_down_through_concat: bool = True,
    apply_try_remove_single_concat: bool = True,
) -> Union[Add, Concat]:
    """
    Replaces an einsum where one input is a concat with a concat (or add) of einsums (each with smaller concats).

    Every group in index_groups must be a contiguous increasing list of indices. The groups must be ordered in
    increasing order by their smallest element. e.g. the following are valid index groups:
    ::

        [[2]]
        [[1, 2]]
        [[0], [2]]
        [[0, 1], [2]]

    If any indices are not in a group, they are automatically grouped into contiguous stretches. e.g. in the first
    example above, ``[0, 1]`` would also be a group; in the second, ``[0]``; and in the third, ``[1]``.

    Args:
        c: The einsum to rewrite.
        is_lhs: If True, splits up the left argument to the einsum (must be a concat). Otherwise splits up the right.
        index_groups: List of lists of indexes to group into concats. These refer to the order circuits appear in the
            original concat. Must be nonempty and each element must be nonempty.

    Returns:
        Concat (or sum) of einsums, one per index group. Each einsum has the original rhs (lhs) argument if is_lhs
        (not is_lhs), and its lhs (rhs) is a concat of the children of the original lhs (rhs) specified by the indices
        in the group (or just the child if the group is of size 1). Concat order is preserved.

        The output is a concat if the original concat dim was in the output of the einsum; and a sum otherwise.
    """
    c = c.cast_einsum()

    assert len(index_groups) > 0, "you must have at least one index group"
    assert all(len(g) > 0 for g in index_groups), "groups must be non-empty"

    node_v, _ = c.args[element_index]
    node = node_v.cast_concat()

    index_groups = list(sorted((list(sorted(g)) for g in index_groups), key=lambda x: x[0]))

    new = []
    running = 0
    for group in index_groups:
        assert len(group) > 0, group
        assert len(group) == len(set(group)), group
        if group[0] != running:
            assert group[0] > running, (group[0], running)
            new.append(list(range(running, group[0])))
        new.append(group)
        if len(group) != group[-1] - group[0] + 1:
            raise NotImplementedError("only supports contiguous groups atm")
        running = len(group) + group[0]

    assert running <= len(node.children), (running, node)
    if running < len(node.children):
        new.append(list(range(running, len(node.children))))

    index_groups = new
    axis = node.axis

    new_slice: List[slice] = []

    for g in index_groups:
        assert len(g) > 0, g

        before = sum(node.children[j].shape[axis] for j in range(0, g[0]))
        end = before + sum(node.children[j].shape[axis] for j in range(g[0], g[-1] + 1))

        new_slice.append(I[before:end])

    def run_on_index(new_idx: Index, _, name: str):
        if not push_down_through_concat:
            return new_idx

        try:
            pushed_down = push_down_index_once(new_idx, suffix="_" + name).cast_einsum()
        except PushDownIndexEinsumNoopError:
            # Detects identity einsum (even fancy ones not detected by elim_identity_einsum, like aa->a when shape is (1,1))
            raise ImpossibleRewriteError("Can't split_einsum_concat on identity einsum")

        (this_node_v, axes), other = get_info(pushed_down, element_index)
        if isinstance(this_node_v, Index):
            assert this_node_v.node == node_v
            new_node_v = push_down_index_once(this_node_v, suffix="").map_children(
                lambda x: remove_trivial_index(x.cast_index())
            )
        else:
            assert this_node_v == node_v
            new_node_v = this_node_v
        assert isinstance(new_node_v, Concat)
        if apply_try_remove_single_concat:
            new_node_v = try_remove_single_concat(new_node_v)

        return reconstruct_by(pushed_down, element_index, new_node_v, axes, other)

    return explicit_reduce(
        c,
        axis=axis,
        element_index=element_index,
        partitioning_idxs=new_slice,
        partitioning_idx_names=group_names,
        axis_name=axis_name,
        use_axis_name=use_axis_name,
        run_on_index=run_on_index,
    )


try_split_einsum_concat = try_transform_wrapper(split_einsum_concat)


def distribute_old(c: Einsum, element_index: int, suffix: Optional[str] = None, iters: int = 1) -> Add:
    """
    Recursively applies the identity `a*(b + c) = a*b + a*c`.

    Args:
        c: The einsum to distribute.
        element_index: The value of ``i`` in ``a_1 * a_2 * ... * (b_i + c_i) * ...``, if the Einsum has the latter
            expression.
        suffix: For naming the new terms. If None, new nodes won't be named.
        iters: Number of times to recurse into children. Must be at least 1.
    """

    c = c.cast_einsum()

    if iters <= 0:
        raise ImpossibleRewriteError("iters must be >= 0")
    if (
        element_index >= len(c.args)
        or c.args[element_index][0].maybe_add() is None
        or len(c.args[element_index][0].cast_add().children) == 0
    ):
        # Make sure at least one level of distribute if possible
        raise ImpossibleRewriteError(f"child of {c} at {element_index} must be an none empty Add")

    out = distribute(
        c,
        element_index,
        traverse_until_depth(iters),
        suffix,
        allow_partial_distribute=True,
        do_broadcasts=True,
    ).cast_add()

    return out


try_distribute_old = try_transform_wrapper(distribute_old)


def permute_to_einsum(c: Rearrange) -> Einsum:
    spec = c.cast_rearrange().spec
    if not spec.is_permute():
        raise ImpossibleRewriteError
    flat_inp = tuple(itertools.chain(*spec.input_ints))
    flat_out = tuple(itertools.chain(*spec.output_ints))

    return Einsum((c.node, flat_inp), out_axes=flat_out, name=c.name)


try_permute_to_einsum = try_transform_wrapper(permute_to_einsum)


def push_down_permute_via_einsum(c: Rearrange, suffix: Optional[str] = "perm", iters=1) -> Add:
    """similar to distribute for einsum"""
    c = c.cast_rearrange()
    return distribute_old(permute_to_einsum(c), element_index=0, suffix=suffix, iters=iters)


try_push_down_permute_via_einsum = try_transform_wrapper(push_down_permute_via_einsum)

# TODO: instead of equiv partition, we should support nest_adds!

# https://stackoverflow.com/questions/38924421/is-there-a-standard-way-to-partition-an-interable-into-equivalence-classes-given
def iterable_gen_equivalence_partition(
    iterable: Iterable[T], relation: Callable[[T, T], bool]
) -> Tuple[List[List[T]], Dict[T, List[T]]]:
    """Partitions a set of objects into equivalence classes

    Args:
        iterable: collection of objects to be partitioned
        relation: equivalence relation. I.e. relation(o1,o2) evaluates to True
            if and only if o1 and o2 are equivalent

    Returns: classes, partitions
        classes: A sequence of sets. Each one is an equivalence class
        partitions: A dictionary mapping objects to equivalence classes
    """
    classes: List[List[T]] = []
    partitions: Dict[T, List[T]] = {}
    for o in iterable:  # for each object
        # find the class it is in
        found = False
        for c in classes:
            if relation(c[0], o):  # is it equivalent to this class?
                c.append(o)
                partitions[o] = c
                found = True
                break
        if not found:  # it is in a new class
            classes.append([o])
            partitions[o] = classes[-1]
    return classes, partitions


def equivalence_partition_list(
    c: Add,
    get_group_name: Callable[[Circuit], str],
    is_eq: Callable[[Circuit, Circuit], bool],
) -> List[Add]:
    """
    Rearranges sum to a list of sums of groups of items (grouped by is_eq).

    Returned list can be summed to recover an equivalent sum to the original.
    """
    classes, _ = iterable_gen_equivalence_partition(c.children, lambda a, b: is_eq(a, b))
    return [Add(*c, name=get_group_name(c[0])) for c in classes]


def equivalence_partition(
    c: Add,
    get_group_name: Callable[[Circuit], str],
    is_eq: Callable[[Circuit, Circuit], bool],
    noop_error: bool = False,
) -> Add:
    """Rearranges sum to first sum groups of items (grouped by is_eq), then sum the groups."""
    if noop_error and len(c.children) == 0:
        raise NoopRewriteError("Add node `c` has no elements")
    return Add(*equivalence_partition_list(c, get_group_name, is_eq), name=c.name)


def make_single_explicit_reduce(i: int):
    return (I[:i] + (i,) + I[i + 1 :]), (f"before_{i}", f"{i}", f"after_{i}")


def make_single_explicit_reduce_idxed(i: int, size: int, device: Optional[Union[torch.device, str]] = None):
    return ((torch.cat([torch.arange(i), torch.arange(i + 1, size)]).to(device=device),) + (i,)), (f"not_{i}", f"{i}")


@functools.cache
def make_single_explicit_reduce_idxed_cached(i: int, size: int, device: Optional[Union[torch.device, str]] = None):
    """use at your peril: this function will fill up the RAM with non-garbage-collectable objects"""
    return make_single_explicit_reduce_idxed(i, size, device)


def explicit_reduce(
    c: Einsum,
    axis: int,
    element_index: int,
    partitioning_idxs: Sequence[TorchAxisIndex],
    partitioning_idx_names: Optional[Sequence[str]] = None,
    axis_name: Optional[str] = None,
    use_axis_name: bool = True,
    run_on_index: Callable[[Index, TorchAxisIndex, str], Circuit] = lambda x, *_: x,
    check: bool = True,
    use_dot: bool = False,
) -> Union[Add, Concat]:
    """explicitly concat with separate nodes (and then reduce) instead of implicitly via einsum

    Can be used for things like having a different circuit for each attention head.
    """
    c = c.cast_einsum()

    if len(c.args) == 0:
        raise ImpossibleRewriteError("c is an empty Einsum")

    _, node_axes = c.args[element_index]
    if len(node_axes) == 0:
        raise ImpossibleRewriteError("The asked-for child of Einsum is a scalar")

    num = node_axes[axis]

    if c.out_axes.count(num) > 1:
        raise NotImplementedError("trace case not implemented")

    is_concat = num in c.out_axes

    if use_axis_name:
        axis_name_prefix = "_" + op.unwrap_or(axis_name, "at")
    else:
        axis_name_prefix = ""

    if is_concat:
        new_out_axes = c.out_axes
        new_c = c
    else:
        new_out_axes = (num,) + c.out_axes
        # TODO improve naming maybe
        new_c = Einsum(*c.args, out_axes=new_out_axes, name=f"{c.name}{'.' if use_dot else '_'}keep{axis_name_prefix}")

    axis_loc = new_out_axes.index(num)

    vals = split_reduce_concat_impl(
        new_c,
        axis_loc=axis_loc,
        is_concat=is_concat,
        partitioning_idxs=partitioning_idxs,
        partitioning_idx_names=partitioning_idx_names,
        axis_name=axis_name,
        use_axis_name=use_axis_name,
        run_on_index=run_on_index,
        check=check,
        use_dot=use_dot,
    )

    if is_concat:
        return Concat(*vals, axis=axis_loc, name=c.name)
    else:
        assert len(partitioning_idxs) > 0
        nums = tuple(range(new_c.rank))
        return Add(
            *[
                v if isinstance(idx, int)
                # nums[1:] is right because new_out_axes = (num,) + c.out_axes
                else Einsum((v, nums), out_axes=nums[1:], name=f"{v.name}_sum")
                for v, idx in zip(vals, partitioning_idxs)
            ],
            name=c.name,
        )


def split_reduce_concat_impl(
    c: Circuit,
    axis_loc: int,
    is_concat: bool,
    partitioning_idxs: Sequence[TorchAxisIndex],
    partitioning_idx_names: Optional[Sequence[str]] = None,
    axis_name: Optional[str] = None,
    use_axis_name: bool = True,
    run_on_index: Callable[[Index, TorchAxisIndex, str], Circuit] = lambda x, *_: x,
    check: bool = True,
    use_dot: bool = False,
):
    assert axis_loc < c.rank

    if use_axis_name:
        axis_name_suffix = op.unwrap_or(axis_name, "at") + "_"
    else:
        axis_name_suffix = ""

    if partitioning_idx_names is None:
        partitioning_idx_names = [f"idx_{i}" for i in range(len(partitioning_idxs))]
    assert len(partitioning_idx_names) == len(partitioning_idxs)

    partitioning_idxs = [normalize_index(i, c.shape[axis_loc]) if isinstance(i, int) else i for i in partitioning_idxs]
    if is_concat:
        partitioning_idxs = [slice(i, i + 1) if isinstance(i, int) else i for i in partitioning_idxs]
    for idx in partitioning_idxs:
        if isinstance(idx, torch.Tensor):
            assert idx.ndim == 1

    if check:
        device = None
        for idx in partitioning_idxs:
            if isinstance(idx, torch.Tensor):
                device = idx.device

        to_set = torch.zeros((c.shape[axis_loc],), dtype=torch.bool, device=device)
        for idx in partitioning_idxs:
            assert (~to_set[idx]).any() or to_set[idx].nelement() == 0
            to_set[idx] = True
            if is_concat:
                assert (torch.sort(to_set, descending=True)[0] == to_set).all()
        assert to_set.all(), (to_set, to_set.shape)
    from interp.circuit.computational_node import make_index_at

    return [
        run_on_index(
            Index(
                c,
                make_index_at(p_idx, axis_loc),
                name=f"{c.name}{'.' if use_dot else '_'}{axis_name_suffix}{p_idx_name}",
            ),
            p_idx,
            f"{axis_name_suffix}{p_idx_name}",
        )
        for p_idx, p_idx_name in zip(partitioning_idxs, partitioning_idx_names)
    ]


def split_to_concat(
    c: Circuit,
    axis: int,
    partitioning_idxs: Sequence[TorchAxisIndex],
    partitioning_idx_names: Optional[Sequence[str]] = None,
    axis_name: Optional[str] = None,
    use_axis_name: bool = True,
    run_on_index: Callable[[Index, TorchAxisIndex, str], Circuit] = lambda x, *_: x,
    check: bool = True,
    use_dot: bool = False,
) -> Concat:
    """
    Replaces the circuit `c` with a concatenation of slices of `c`.
    Slices along `axis`, where the i-th slice will include the indicies `partition_idxs[i]`.

    Naming options:
    By default each Index will be named `{c.name}_at_idx_{i}`
    `use_dot` replaces the first underscore with a period
    `axis_name` replaces "at_". `use_axis_name=False` removes this part entirely.
    `partitioning_idx_names` replaces `idx_{i}`

    Other options:
    `run_on_index` is an optional transformation to run on every created index. For instance, a call to `push_down_index` could be provided.
    `check` ensures it is an algebric rewrite.

    See also: rc.split_to_concat which has less options but is in rust (and thus probably faster?).
    """
    # this exists for fancy naming options
    vals = split_reduce_concat_impl(
        c,
        axis_loc=axis,
        is_concat=True,
        partitioning_idxs=partitioning_idxs,
        partitioning_idx_names=partitioning_idx_names,
        axis_name=axis_name,
        use_axis_name=use_axis_name,
        run_on_index=run_on_index,
        check=check,
        use_dot=use_dot,
    )
    return Concat(*vals, axis=axis, name=c.name)


# TODO: port to rust completly (I assume some code is already doing sth like that?)
def extract_output_diags(c: Einsum):
    """separate einsum for output diags"""

    c = c.cast_einsum()

    if len(c.out_axes) == len(set(c.out_axes)):
        raise ImpossibleRewriteError("no output diags to extract")

    deduped_axes = tuple(dedup_with_order(c.out_axes))
    evolved = Einsum(*c.args, out_axes=deduped_axes, name=f"{c.name}_no_diag")

    return Einsum((evolved, deduped_axes), out_axes=c.out_axes, name=c.name)


try_extract_output_diags = try_transform_wrapper(extract_output_diags)


def extract_input_diags(c: Einsum, item_index: int):
    """separate einsum for input diags, nice for distributing and similar"""

    c = c.cast_einsum()

    (node, node_axes), other = get_info(c, item_index)

    deduped_axes = tuple(dedup_with_order(node_axes))

    if len(deduped_axes) == len(node_axes):
        raise ImpossibleRewriteError()

    new_node = Einsum((node, node_axes), out_axes=deduped_axes, name=f"{node.name}_traced")

    return reconstruct_by(c, item_index, new_node, deduped_axes, other)


try_extract_input_diags = try_transform_wrapper(extract_input_diags)


def einsum_flatten_bans_noop(c: Einsum, traversal: IterativeMatcherIn = new_traversal(), allow_noop: bool = False):
    """
    equivalent to einsum_flatten, but bans noop (raises exception)
    """
    out = einsum_flatten(c, traversal)
    if out == c and not allow_noop:
        raise NoopRewriteError
    return out


try_einsum_flatten = try_transform_wrapper(einsum_flatten_bans_noop)


def residual_rewrite(c: Circuit, estim: Circuit, running_name: str, flip_naming: bool = False):
    """
    Rewrites c as `(estim + (c - estim))`. Then returns the tuple:
      - estim + (c - estim),    # named `{c.name}`
      - estim,
      - c - estim               # named `{c.name}_{running_name}_residual`

    The inner copy of c is renamed to `{c.name}_orig`.

    If flip_naming is true then the entire circuit is named `{c.name}_{running_name}_out` and the inner copy of c is named `{c.name}`.
    """
    residual = c.rename(c.name if flip_naming else f"{c.name}_orig").sub(estim, f"{c.name}_{running_name}_residual")

    return (estim.add(residual, name=f"{c.name}_{running_name}_out" if flip_naming else c.name), estim, residual)


# TODO: maybe implement extract add in terms of nest_adds (which hasn't yet been implemented)


def extract_add(c: Add, sub: Add) -> Add:
    """Extract sub from c: a+b+c+d, c+d -> a+b+(c+d)"""
    out = extract_add_rust(c.cast_add(), sub.cast_add())
    if out is None:
        raise ImpossibleRewriteError
    return out


def extract_add_by_match(c: Add, match: Matcher, sub_name: Optional[str] = None):
    """Extract matching node from c: a+b+c+d -> b+c+(a+d) (where a & d match)"""
    c = c.cast_add()
    matcher = Matcher(match)
    sub = Add(*[node for node in c.children if matcher(node)], name=op.unwrap_or(sub_name, c.name + "_sub"))

    return extract_add(c, sub)


def get_dups(*elems: T) -> Set[T]:
    counts = Counter(elems)
    return set([x for x in elems if counts[x] > 1])


def check_permutation(perm: List[int], count: int, allow_rest: bool = False):
    perm_set = set(perm)
    assert len(perm) == len(perm_set), (perm, get_dups(perm))
    assert perm_set.issubset(range(count)), (perm, count, perm_set - set(range(count)))
    rest = set(range(count)) - perm_set
    if not allow_rest:
        assert len(rest) == 0, rest
    return rest


def nested_einsum_axes_permute(c: Einsum, element_index: int, permutation: Sequence[int], keep_name: bool = True):
    """TODO: maybe reimplement in terms of rearrange muls"""
    c = c.cast_einsum()

    (node_v, node_axes), other = get_info(c, element_index)
    node = node_v.cast_einsum()

    perm = list(permutation)
    check_permutation(perm, len(node_axes), allow_rest=False)

    def get_permuted(vals: Tuple[int, ...]):
        new_axes = torch.zeros((len(node_axes),), dtype=torch.long)
        new_axes[torch.tensor(perm, dtype=torch.long)] = torch.tensor(vals, dtype=torch.long)
        return tuple(int(i.item()) for i in new_axes)

    new_out_axes = get_permuted(node.out_axes)
    new_node_axes = get_permuted(node_axes)

    new_node = cast(Einsum, node.evolve(out_axes=new_out_axes).rename(node.name if keep_name else f"{node.name}_perm"))
    result = reconstruct_by(c, element_index, new_node, new_node_axes, other)
    if result == c:
        raise NoopRewriteError("nested_einsum_axes_permute noop")
    return result


def nested_einsum_permute_dups_to_eq(c: Einsum):
    """TODO: maybe reimplement in terms of rearrange muls"""
    # untested, annoying to test...
    c = c.cast_einsum()

    def norm_out(x: Einsum):
        return x.evolve(out_axes=tuple(sorted(x.out_axes)))

    to_class: Dict[Tuple[int, Einsum], List[Tuple[int, Einsum]]] = iterable_gen_equivalence_partition(
        [(i, x) for i, x in enumerate(c.children) if isinstance(x, Einsum)],
        relation=lambda x, y: norm_out(x[1]) == norm_out(y[1]),
    )[1]

    for (idx, x), ((_, first), *_) in to_class.items():
        if x != first:
            c = nested_einsum_axes_permute(
                c, idx, permutation=[x.out_axes.index(i) for i in first.out_axes], keep_name=True
            )

    return c


try_nested_einsum_permute_dups_to_eq = try_transform_wrapper(nested_einsum_permute_dups_to_eq)


def multiply_axes_by_identity(
    c: Circuit,
    axes: Sequence[int],
    ident_prefix_name: Optional[str] = None,
    keep_sub_name: bool = False,
    batch_axes: Sequence[int] = [],
):
    """note that this will be simplified away!"""
    axes = [normalize_index(a, c.rank) for a in axes]
    shape = tuple(c.shape[a] for a in batch_axes) + tuple(c.shape[a] for a in axes)
    assert len(set(batch_axes).intersection(axes)) == 0
    ident_prefix_name_v = op.unwrap_or(ident_prefix_name, c.name + "_ident")
    batch_nums = tuple(range(len(batch_axes)))
    nums = tuple(range(len(batch_nums), len(batch_nums) + len(axes)))
    ident = Einsum(
        (Scalar(1.0, shape, f"{ident_prefix_name_v}_one"), batch_nums + nums),
        out_axes=batch_nums + nums + nums,
        name=ident_prefix_name_v,
    )

    all_nums = tuple(range(c.rank))
    out_axes = tuple(range(c.rank, c.rank + len(axes)))

    all_nums_out_replaced = list(all_nums)
    for i, a in enumerate(axes):
        all_nums_out_replaced[a] = out_axes[i]

    return Einsum(
        (c.rename(c.name + ("" if keep_sub_name else "_no_ident")), all_nums),
        (ident, tuple(batch_axes) + tuple(axes) + out_axes),
        out_axes=tuple(all_nums_out_replaced),
        name=c.name + ("_ident" if keep_sub_name else ""),
    )


# TODO: test me!!!
def make_diag_mask_pair(n: int, ndim: int, name_n: str):
    diag = Einsum(
        (Scalar(1.0, (n,), name=f"{name_n}_size_one"), (0,)), out_axes=(0,) * ndim, name=f"{name_n}_diag_{ndim}"
    )
    non_diag = Scalar(1.0, (n,) * ndim, name=f"{name_n}_{ndim}_size_one").sub(diag, name=f"{name_n}_non_diag_{ndim}")

    return Add(diag, non_diag, name=f"{name_n}_one_mask_pair")


# TODO: test me!!!
def insert_diag_mul_mask(
    c: Einsum, item_idx: int, n: int, dims: Sequence[int], name_n: str, use_distribute: bool = True
):
    return insert_diag_mul_mask_many(c, [item_idx], n, [dims], name_n, use_distribute=use_distribute)


# TODO: test me!!!
def insert_diag_mul_mask_many(
    c: Einsum, item_idxs: Sequence[int], n: int, dims: Sequence[Sequence[int]], name_n: str, use_distribute: bool = True
):
    c = c.cast_einsum()
    all_nums = [c.all_input_axes()[i] for i in item_idxs]
    item_nums = tuple(itertools.chain(*([nums[i] for i in dims_per] for dims_per, nums in zip(dims, all_nums))))
    mask_pair = make_diag_mask_pair(n, len(item_nums), name_n=name_n)

    out = c.evolve(args=tuple(c.args) + ((mask_pair, item_nums),))

    if use_distribute:
        out_add = distribute_old(out, element_index=len(c.args))
        assert len(out_add.children) == 2
        return out_add.map_children(lambda x: x.rename(c.name + ("_non_diag" if "non_diag" in x.name else "_diag")))

    return out


# maybe use split_to_concat
def split_to_concat_for_batch(circ: Circuit, batch_size: int, axis: int = 0):
    assert circ.rank >= 1
    from interp.circuit.computational_node import make_index_at

    return Concat(
        *[
            Index(circ, make_index_at(I[i * batch_size : min((i + 1) * batch_size, circ.shape[axis])], axis))
            for i in range(math.ceil(circ.shape[axis] / batch_size))
        ],
        axis=axis,
    )
