import re
from enum import Enum
from typing import Literal, Optional, Union

import rust_circuit.algebric_rewrite as algrw
import rust_circuit.optional as op

from . import _rust as rc

# from functools import total_ordering


def push_down_transformer_namer(circ: rc.Circuit, _, mods: list[rc.Module], _num):
    if re.match(r"([amb]\.)|([am]$)|(ln$)", circ.name):
        if circ.name == "ln":
            name = mods[0].name
        else:
            name = circ.name
        split_first, _, split_rest = name.partition(".")
        suffix = f".{split_rest}" if "." in name else ""
        num: Optional[int] = None
        for x in mods:
            get_name = re.match(r"b(\d+)$", x.name)
            if get_name is None:
                continue
            num = int(get_name.group(1))
            break

        # this is need to handle final.ln case correctly
        num_str = str(num) if isinstance(num, int) else ""

        return f"{split_first}{num_str}{suffix}"

    return circ.name


# total ordering doesn't type check, sad
# @total_ordering
class PushDownTransformerTo(Enum):
    BLOCK = 0
    ATTN_MLP = 1
    ATTN_MLP_NORM = 2
    ATTN_HEAD_MLP_NORM = 3

    # def __lt__(self, other):
    #     if self.__class__ is other.__class__:
    #         return self.value < other.value
    #     return NotImplemented


To = PushDownTransformerTo


def strip_arr(c: rc.Circuit):
    """strip _arr suffix from array names"""
    return c.update(
        rc.Array & rc.Matcher(lambda x: x.name.endswith("_arr")), lambda x: x.rename(x.name.removesuffix("_arr"))
    )


def pull_up_bias(c: rc.Circuit):
    """
    pull up attention/mlp bias passed module binding sites

    useful because the bias isn't really strongly associated with the attention/mlp

    Might be nice for flattening out the residual stream
    """

    def namer(x: rc.Circuit, _, mods: list[rc.Module], _num):
        if len(mods) == 0:
            return x.name
        norm_call = ["a.norm_call", "m.norm_call"]
        block_call = ["b.a", "b.m"]
        if x.name in ["a", "m"]:
            if mods[-1].name in block_call:
                assert len(mods) in [1, 2]
                if len(mods) == 2:
                    assert mods[0].name in norm_call
                return f"b.{x.name}"
            elif len(mods) == 1:
                assert mods[0].name in norm_call
                return f"{x.name}.norm_call"
            else:
                assert False
        return x.name

    out = rc.ModulePusher(namer=namer, flatten_modules=False).push_down_modules(
        c,
        traversal=rc.new_traversal(term_early_at=rc.Matcher.regex(r"^[am]$")),
        skip_module=~(rc.Matcher.regex(r"^[am].norm_call$") | rc.Matcher.regex(r"^b.[am]$")),
    )
    return out.update(rc.Matcher.regex(r"^b.[am]$") & rc.Add, lambda x: x.rename(x.name + ".p_bias"))


def push_down_transformer_weights(
    c: rc.Circuit,
    to: PushDownTransformerTo,
    use_strip_arr: bool = True,
):
    """
    to: How to push down weights (to more granular modules) if at all. The enum is ordered with lower values
    meaning 'less pushing' and higher values meaning 'more pushing'. None means
    keep the weights bound at the top level.

    use_strip_arr: now that weights have been pushed down, names no longer
    confict with symbols so we can remove _arr as a suffix (if desired).
    """
    assert c.name == "t.bind_w"

    skip_on_fuse = ~rc.Matcher(rc.Matcher.regex(r"^b\d+$"), "final.norm", "t.bind_w", rc.Matcher.regex(r"^[am].norm$"))

    term_early_at: list[rc.MatcherIn] = ["ln", "bn"]
    if to == To.BLOCK:
        term_early_at.append("b")
    elif to == To.ATTN_MLP:
        term_early_at.append(rc.Matcher.regex(r"^[am].norm_call$"))
    if to.value <= To.ATTN_MLP_NORM.value:
        term_early_at.append(rc.Matcher.regex(r"^[am](.p_bias)?$"))
    if to.value <= To.ATTN_HEAD_MLP_NORM.value:
        term_early_at.append(rc.Matcher.regex(r"^m(.p_bias)?$"))
        term_early_at.append("a.head.on_inp")

    # TODO: add support for full substitute?

    # we're not doing anything too interesting here - just pushing down some select modules
    out = rc.ModulePusher(namer=push_down_transformer_namer).push_down_modules(
        c,
        traversal=rc.new_traversal(term_early_at=rc.Matcher.any(*term_early_at)),
        skip_module=skip_on_fuse,
    )

    # not currently needed
    # if to >= To.ATTN_MLP:
    #     out = out.update(rc.Matcher.regex(r"^b\d+.[am].set$"), lambda x: x.rename(x.name.removesuffix(".set")))

    if use_strip_arr:
        out = strip_arr(out)

    return out


SplitConfigDict = dict[int, list[tuple[rc.TorchAxisIndex, str]]]
SplitConfigSetDict = Union[set[int], SplitConfigDict]
SplitConfig = Union[Literal["full"], SplitConfigSetDict]


def split_by_head(c: rc.Circuit, split: SplitConfig = "full"):
    """
    requires that weights have been pushed down all the way to heads

    full: split heads on all layers
    set[int]: split heads on these layers
    dict[int, list[...]]: split heads on these layers into this list of groups
        where each group has an index and name
    """

    heads = c.get(rc.Matcher.regex(r"^a\d+.heads$"))
    num_layers = len(heads)
    split_v: SplitConfigSetDict
    if split == "full":
        split_v = set(range(num_layers))
    else:
        split_v = split

    if num_layers == 0:
        return c

    num_heads = list(heads)[0].shape[0]
    split_vv: SplitConfigDict
    if isinstance(split_v, set):
        split_vv = {l: [(h, f"h{h}") for h in range(num_heads)] for l in split_v}
    else:
        split_vv = split_v

    idxed = c
    for i, items in split_vv.items():
        idxs = [idx for idx, _ in items]
        names = [name for _, name in items]
        assert idxed.are_any_found(f"a{i}")
        from .algebric_rewrite import explicit_reduce

        idxed = idxed.update(
            f"a{i}",
            lambda x: explicit_reduce(
                x.cast_einsum(),
                axis=0,
                element_index=0,
                partitioning_idxs=idxs,
                use_axis_name=False,
                partitioning_idx_names=names,
                run_on_index=lambda x, _, s: rc.push_down_index(x, suffix="." + s),
                use_dot=True,
            ),
        )
    return rc.simp(idxed).update(rc.Matcher.regex(r"^a\d+.heads."), lambda x: x.rename(x.name.replace(".heads.", ".")))


def pull_up_head_split(c: rc.Circuit, to_pull_up: Optional[set[int]] = None):
    """
    pulls up split heads past modules: this results in the layernorm and input
    being bound for each head individually.

    This might be nice for flatten_res.
    """

    def namer(x: rc.Circuit, _, mods: list[rc.Module], _num):
        block_call = r"b\d+\.a$"
        norm_call = r"a\d+\.norm_call$"
        if len(mods) > 0 and re.match(rf"{norm_call}|{block_call}", mods[0].name):
            assert re.match(r"^a\d+", x.name), x.name
            if re.match(block_call, mods[-1].name):
                assert len(mods) in [1, 2]
                if len(mods) == 2:
                    assert re.match(norm_call, mods[0].name)
                attn, _, head_rest = x.name.partition(".")
                num = int(attn[1:])
                return f"b{num}.a.{head_rest}"
            elif len(mods) == 1:
                assert re.match(norm_call, mods[0].name)
                return f"{x.name}.norm_call"
            assert False
        return x.name

    num_layers = len(c.get(rc.Matcher.regex(r"^a\d+$")))
    to_pull_up_v = op.unwrap_or(to_pull_up, set(range(num_layers)))
    out = c
    for i in to_pull_up_v:
        out = rc.ModulePusher(namer=namer, flatten_modules=False).push_down_modules(
            out,
            traversal=rc.new_traversal(term_early_at=~rc.Matcher.match_any_found(f"a{i}")).chain(
                rc.IterativeMatcher.term()
            ),
            skip_module=~(rc.Matcher(f"a{i}.norm_call") | f"b{i}.a"),
        )
    return out


def clear_block_module(c: rc.Circuit):
    """
    substitute out calling blocks on arguments.

    This is only nice to do if you've already pushed weights past blocks.

    This ensures that adding into the residual stream isn't inside a module.
    """
    return c.update(
        rc.Matcher.regex(r"^b\d+.call$"), lambda x: x.cast_module().substitute().rename(x.name.partition(".")[0])
    )


def flatten_res(c: rc.Circuit, flatten_components: bool = False):
    """
    flatten the residual stream

    flatten_components makes flatten operate within mlp/attn components. This
    only matters if you have a bias or attn is separated into heads.
    """
    if flatten_components:
        traversal = rc.new_traversal()
    else:
        traversal = rc.new_traversal(
            term_early_at=rc.Matcher.regex(r"^b\d+.[am](.p_bias)?$") | rc.Matcher.regex("^[am].(.p_bias)?")
        )

    return c.update(
        rc.Matcher.regex(r"^b\d+(.resid.\w)?$"), lambda x: rc.add_flatten(x.cast_add(), traversal=traversal)
    )


def configure_transformer(
    c: rc.Circuit,
    to: Optional[To] = To.ATTN_MLP_NORM,
    use_pull_up_bias: bool = True,
    use_strip_arr: bool = True,
    split_by_head_config: Optional[SplitConfig] = None,
    use_pull_up_head_split: bool = False,
    use_clear_block_module: bool = True,
    use_flatten_res: bool = False,
    flatten_components: bool = False,
    check_valid: bool = True,
):
    """
    See docs for pull_up_bias, push_down_transformer_weights, split_by_head,
    pull_up_head_split, clear_block_module, and flatten_res.

    This function just subsequently runs these operations (if they are enabled
    via a flag or by having their configuration set to non-None)
    """
    assert c.name == "t.bind_w"

    if check_valid:
        if use_strip_arr:
            assert to is not None, "stripping arr without pushing down results in name collision"
        if split_by_head_config is not None:
            assert to == To.ATTN_HEAD_MLP_NORM, "You can't split by head without pushing down weights to head"
        if use_pull_up_head_split:
            assert split_by_head_config is not None, "You can't pull up split heads if heads aren't split!"
        if use_clear_block_module:
            assert (
                to is not None and to.value >= To.ATTN_MLP.value
            ), "clearing block module not supported without pushing down to at least ATTN_MLP"
        if use_flatten_res:
            assert (
                use_clear_block_module
            ), "if you want to flatten the residual stream, you'll need to use_clear_block_module"
        if flatten_components:
            assert use_flatten_res, "setting flatten_components has no effect without use_flatten_res"

    if use_pull_up_bias:
        c = pull_up_bias(c)
    if to is not None:
        c = push_down_transformer_weights(c, to, use_strip_arr=use_strip_arr)
    if split_by_head_config is not None:
        c = split_by_head(c, split_by_head_config)
    if use_pull_up_head_split:
        to_pull_up: Optional[set[int]] = None
        if isinstance(split_by_head_config, set):
            to_pull_up = split_by_head_config
        elif isinstance(split_by_head_config, dict):
            to_pull_up = set(split_by_head_config.keys())
        c = pull_up_head_split(c, to_pull_up=to_pull_up)
    if use_clear_block_module:
        c = clear_block_module(c)
    if use_flatten_res:
        c = flatten_res(c, flatten_components=flatten_components)

    # pull this out into function maybe
    c = c.update(rc.SetSymbolicShape, lambda x: x.rename(x.get_autoname()))

    return c


def batch_mul(
    x: rc.Einsum,
    operand_idx: int,
    retained_axes: tuple[int, ...],
    insert_at_idx: Optional[int] = 0,  # None: end because -0 doesn't exist
    assert_not_present: bool = True,
    assert_not_diag: bool = True,
):
    einsum_axes = [x.all_input_axes()[operand_idx][i] for i in retained_axes]
    if assert_not_diag:
        assert len(einsum_axes) == len(set(einsum_axes))

    if assert_not_present:
        assert len(set(einsum_axes).intersection(x.out_axes)) == 0
    insert_at_idx_v = op.unwrap_or(insert_at_idx, len(x.out_axes))
    return x.evolve(out_axes=x.out_axes[:insert_at_idx_v] + tuple(einsum_axes) + x.out_axes[insert_at_idx_v:])
