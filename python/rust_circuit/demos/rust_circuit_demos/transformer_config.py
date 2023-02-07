# %%

from typing import List

import pytest

import rust_circuit as rc
import rust_circuit.module_library as mod_l
from interp.tools.indexer import SLICER as S
from rust_circuit.model_rewrites import (
    To,
    clear_block_module,
    configure_transformer,
    flatten_res,
    pull_up_bias,
    pull_up_head_split,
    push_down_transformer_weights,
    split_by_head,
    strip_arr,
)

# %%

# we'll start by getting a model
# we'll use biases for demoing for now
params = mod_l.TransformerParams(mod_l.TransformerBlockParams(attn_bias=True, mlp_output_bias=True), num_layers=2)
c, _, _ = params.garbage_call(num_heads=2)  # 2 heads to avoid too much clutter

# note that we bind here - all of the below config funcs are intended to be
# applied to this t.bind_w circuit.
# You might want to use `c.update("t.bind_w", ...)` in various cases
c = c.get_unique("t.bind_w")

# %%

# and some printers
basic_term_early: List[rc.MatcherIn] = ["a", "m", "ln", "a.head.on_inp"]
printer = rc.PrintOptions(
    bijection=False,
    traversal=rc.new_traversal(term_early_at=rc.Matcher(*basic_term_early)),
    colorer=rc.PrintOptions.type_colorer(),
)
less_term_early: List[rc.MatcherIn] = [
    *basic_term_early,
    rc.Matcher.regex(r"^a\d+((.h\d+)|(.norm))$"),
    rc.Matcher.regex(r"^m\d+(.norm)?(.p_bias)?$"),
    "final.norm",
]
less_printer = printer.evolve(
    traversal=rc.new_traversal(term_early_at=rc.Matcher(*less_term_early, rc.Matcher.regex(r"^a\d+(.p_bias)?$")))
)
less_printer_split = printer.evolve(traversal=rc.new_traversal(term_early_at=rc.Matcher(*less_term_early)))

# %%

# note that the model will be loaded with the weights all bound in one big
# module - this is nice for loading and some other stuff, but not typically
# nice for interp
printer.print(c)

# %%

# now we can configure our model in a variety of ways!

# here are the default args
# see docs for more details
out = configure_transformer(
    c,
    to=To.ATTN_MLP_NORM,  # push down weights to bind at attn/mlp/norm
    use_pull_up_bias=True,
    use_strip_arr=True,
    split_by_head_config=None,
    use_pull_up_head_split=False,
    use_clear_block_module=True,
    use_flatten_res=False,
    flatten_components=False,
    check_valid=True,
)
printer.print(out)

# %%

# quite pretty IMO
less_printer.print(out)

# %%

# now some examples/tests

# %%


no_pull = configure_transformer(c, use_pull_up_bias=False)
printer.print(no_pull)

# %%

# side by side
less_printer.print(out)
less_printer.print(no_pull)  # biases are inside

# %%

# be warned, some args require specific settings of other args!
with pytest.raises(AssertionError) as exc0:
    configure_transformer(c, split_by_head_config="full")
print(exc0.exconly())

# %%

split = configure_transformer(c, to=To.ATTN_HEAD_MLP_NORM, split_by_head_config="full")
printer.print(split)

# %%

less_printer_split.print(split)

# %%

# this moves the norm to be per head - this can be nice if you want to think of
# each head + norm as a separate function
split_pull = configure_transformer(
    c, to=To.ATTN_HEAD_MLP_NORM, split_by_head_config="full", use_pull_up_head_split=True
)
printer.print(split_pull)

# %%

less_printer_split.print(split)
less_printer_split.print(split_pull)

# %%

# I don't typically recommend using flat, but you can if you want
flat = configure_transformer(c, use_flatten_res=True)
less_printer.print(flat)

# %%

# if we pull up biases, by default they still aren't flattened in
# (same as above, use_pull_up_bias is default)
flat = configure_transformer(c, use_flatten_res=True, use_pull_up_bias=True)
# printer.print(flat)
less_printer.print(flat)

# %%

# but we can do that by using flatten_components
flat = configure_transformer(c, use_flatten_res=True, use_pull_up_bias=True, flatten_components=True)
# printer.print(flat)
less_printer.print(flat)

# %%

# attn heads also aren't flattened in even if pulled up
split_pull_some_flat = configure_transformer(
    c, to=To.ATTN_HEAD_MLP_NORM, split_by_head_config="full", use_pull_up_head_split=True, use_flatten_res=True
)
# printer.print(split_pull_some_flat)
less_printer_split.print(split_pull_some_flat)

# %%

# but we can flatten_components
split_pull_flat = configure_transformer(
    c,
    to=To.ATTN_HEAD_MLP_NORM,
    split_by_head_config="full",
    use_pull_up_head_split=True,
    use_flatten_res=True,
    flatten_components=True,
)
# printer.print(split_pull_flat)
less_printer_split.print(split_pull_flat)

# %%

# the head splitting also supports some more detailed splitting.
split_first = configure_transformer(
    c,
    to=To.ATTN_HEAD_MLP_NORM,
    split_by_head_config={0},  # just split layer 0
)
# printer.print(split_first)
less_printer_split.print(split_first)

# %%

# you can also pass a dict of a list of tuples (args for explicit_reduce)
split_complex = configure_transformer(
    c,
    to=To.ATTN_HEAD_MLP_NORM,
    split_by_head_config={
        0: [(1, "my_group"), (0, "my_other_group")],
        1: [(S[:1], "first_part"), (1, "snd")],
    },
)
# printer.print(split_complex)
less_printer_split.print(split_complex)

# %%

# if you'd like, you can also apply head splitting afterward

to_split = configure_transformer(c, to=To.ATTN_HEAD_MLP_NORM)
split_after = split_by_head(to_split, split="full")
# printer.print(split_after)
less_printer_split.print(split_after)

# %% [markdown]

# ## The end
#
# this is end of the tutorial part of this notebook. Below here we just have misc testing.

# %%

pushed = push_down_transformer_weights(c, To.BLOCK)
printer.print(pushed)

# %%

pushed = strip_arr(push_down_transformer_weights(c, To.ATTN_MLP))
printer.print(pushed)

# %%

pushed = push_down_transformer_weights(c, To.ATTN_MLP_NORM)
printer.print(pushed)

# %%

pushed = push_down_transformer_weights(c, To.ATTN_HEAD_MLP_NORM)
printer.print(pushed)

# %%

pushed = push_down_transformer_weights(c, To.ATTN_HEAD_MLP_NORM)
# printer.print(pushed)
split = split_by_head(pushed)
printer.print(split)
# pushed.print()

# %%

pushed = push_down_transformer_weights(c, To.ATTN_MLP_NORM)
pushed = clear_block_module(pushed)
# printer.print(pushed)
flat = flatten_res(pushed)
printer.print(flat)

# rc.PrintOptions.add_nest_default().print(pushed.get_unique("b1"))

# printer_few = rc.PrintOptions(
#     bijection=False, traversal=rc.new_traversal(term_early_at=rc.Matcher("a", "m", "ln", "a.head.on_inp", rc.Matcher.regex("^[am].ln_call$")))
# )

# printer_few.print(pushed)

# %%

pushed = push_down_transformer_weights(pull_up_bias(c), To.ATTN_HEAD_MLP_NORM)
pushed = split_by_head(pushed)
pushed = clear_block_module(pushed)
# pushed = pull_up_head_split(pushed)
printer.print(pushed)
# printer.print(c)

# %%

pushed = push_down_transformer_weights(pull_up_bias(c), To.ATTN_HEAD_MLP_NORM)
pushed = split_by_head(pushed)
pushed = clear_block_module(pushed)
pushed = pull_up_head_split(pushed)
flat = flatten_res(pushed, flatten_components=True)
less_printer_split.print(flat)


# %%

with pytest.raises(AssertionError):
    configure_transformer(c, to=None, use_clear_block_module=True, use_strip_arr=False)

# %%

params_nb = mod_l.TransformerParams(mod_l.TransformerBlockParams(attn_bias=False, mlp_output_bias=False), num_layers=2)
c_nb, _, _ = params_nb.garbage_call(num_heads=3)
c_nb = c_nb.get_unique("t.bind_w")

# %%

yes_pull = configure_transformer(c_nb, use_pull_up_bias=True)
# printer.print(default)
less_printer.print(yes_pull)

# %%

# same as above, no bias
no_pull = configure_transformer(c_nb, use_pull_up_bias=False)
# printer.print(default)
less_printer.print(no_pull)

# %%

x = configure_transformer(
    c_nb,
    to=To.ATTN_HEAD_MLP_NORM,
    split_by_head_config="full",
    use_pull_up_head_split=True,
    use_flatten_res=True,
    flatten_components=True,
)
less_printer_split.print(x)

# %%

params_nb_nn = mod_l.TransformerParams(
    mod_l.TransformerBlockParams(attn_bias=False, mlp_output_bias=False, norm_type=None), num_layers=2
)
c_nb_nn, _, _ = params_nb_nn.garbage_call(num_heads=3)
c_nb_nn = c_nb_nn.get_unique("t.bind_w")

# %%

yes_pull = configure_transformer(c_nb_nn, use_pull_up_bias=True)
# printer.print(default)
less_printer.print(yes_pull)

# %%

# same as above, no bias
no_pull = configure_transformer(c_nb_nn, use_pull_up_bias=False)
# printer.print(default)
less_printer.print(no_pull)


# %%

x = configure_transformer(
    c_nb_nn,
    to=To.ATTN_HEAD_MLP_NORM,
    split_by_head_config="full",
    use_pull_up_head_split=True,
    use_flatten_res=True,
    flatten_components=True,
)
less_printer_split.print(x)

# %%


x = configure_transformer(
    c_nb_nn,
    to=To.ATTN_HEAD_MLP_NORM,
    split_by_head_config="full",
    use_pull_up_head_split=False,
    use_flatten_res=True,
    flatten_components=True,
)
less_printer_split.print(x)

# %%

x = configure_transformer(
    c_nb_nn,
    to=To.ATTN_HEAD_MLP_NORM,
    split_by_head_config="full",
    use_pull_up_head_split=False,
    use_flatten_res=True,
    flatten_components=True,
)
# printer.print(x)
less_printer_split.print(x)
