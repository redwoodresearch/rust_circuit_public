# %%
import uuid

import pytest
import torch

import rust_circuit as rc
import rust_circuit.module_library as mod_l
from rust_circuit.module_utils import drop_module_inputs

# %% [markdown]

# # Overview

# this notebook will go through:
# - Symbols
# - Expanding
# - Symbolic sizes
# - And how all of these things combing to yield Modules

# For demonstration purposes, we'll do various things with Expander which you
# should probably use Modules for in general. So, don't assume the stuff we do
# with 'Expander' is how you should do things. Similarly, we'll use some more
# verbose module construction approaches when you'll often want to use more
# consise approaches.

# We assume your're familiar with the basics of circuits + printing/parsing
# (see interp/demos/rust_circuit/printing_parsing.py for more on that)

# %% [markdown]

# # Symbols

# %%

# symbol nodes can be used as place holders in various cases
sym = rc.Symbol.new_with_random_uuid((2, 3), name="sym")
sym

# %%

# symbols have uuids. If the uuid is randomly generated, then the symbol is universally unique
sym.uuid

# %%

# symbols aren't explicitly computable
assert sym.is_explicitly_computable is False
with pytest.raises(rc.TensorEvalNotExplicitlyComputableError) as exc:
    sym.evaluate()
print(exc.exconly())

# %%

P = rc.Parser()

# if you parse a symbol without a uuid present, the null uuid will be used
null_uuid = P("'sym_with_random_uuid' [7] Symbol").cast_symbol().uuid
assert null_uuid == uuid.UUID("00000000-0000-0000-0000-000000000000")

# Ofc, you can specify a specific uuid. I recommed having some convenient way to generate new random uuids
# (For instance, I have a vim snippet which expands 'uuid' to a new uuid).

rand_uuid = P("'sym_with_random_uuid' [7] Symbol dc87d8f0-8d11-43d6-86ca-84bc097fae82").cast_symbol().uuid
rand_uuid

# %%

# you can also use 'rand'. But be warned, this will generate a new uuid each time the string is parsed.
rand_uuid = P("'sym_with_random_uuid' [7] Symbol rand").cast_symbol().uuid
rand_uuid

# %% [markdown]

# # Expanding
#
# Now let's talk about replacing symbols inside of circuits

# %%

# we can construct a circuit with symbols as placeholders
circ = P(
    """
'symbolic_mul' Einsum ij,jk->ik
  'my_sym' [7, 2] Symbol 0dc30ed9-239f-459d-869a-78c735e17b36
  'mat' [2, 5] Array rand
"""
)

printer = rc.PrintOptions(shape_only_when_necessary=False, colorer=rc.PrintOptions.type_colorer())

# this isn't possible to compute, but we can substitute in a value for sym and then compute
circ_comp = rc.Updater(lambda _: rc.Scalar(0.7, (7, 2), "rep"))(circ, "my_sym", fancy_validate=True)
print()
printer.print(circ_comp)
assert circ_comp.is_explicitly_computable

# %%

# often, when doing substitutions like this, we'd like to be able to add extra
# dimensions (which are batched over)
circ_batched = rc.Expander(("my_sym", lambda _: rc.Scalar(0.7, (9, 10, 11, 7, 2), "rep")))(circ, fancy_validate=True)
assert circ_batched.shape == (9, 10, 11, 7, 5)
printer.print(circ_batched)

# %%

# expanding also supports changing non-batched dims which can be changed
# (at least in most/many cases when this can be supported it is supported)

# we can change this 7 dim to a 12
circ_batched = rc.Expander(("my_sym", lambda _: rc.Scalar(0.7, (11, 12, 2), "rep")))(circ, fancy_validate=True)
assert circ_batched.shape == (11, 12, 5)
printer.print(circ_batched)


# but we can't change the 2 dim because it needs to correspond to 'mat'
with pytest.raises(rc.ConstructEinsumAxisSizeDifferentError) as exc0:
    rc.Expander(("my_sym", lambda _: rc.Scalar(0.7, (11, 12, 3), "rep")))(circ, fancy_validate=True)
print(exc0.exconly())

# %%

# for a more concrete example, let's consider a symbolic bilinear_mlp
# We'll need to set a few specific sizes

HIDDEN = 256
OUT_HIDDEN = 256
MLP_PROJ = 512
mlp = P(
    f"""
'bilinear_mlp' Einsum h,h,oh->o
  'bilinear_mlp.pre0' Index [0,:]
    'bilinear_mlp.fold_pre' Rearrange (a:2 b) -> a:2 b
      'bilinear_mlp.pre' Einsum i,hi->h
        'bilinear_mlp.input' [{HIDDEN}] Symbol 873a937e-2bb9-4f7f-b55e-a100db3dde52
        'bilinear_mlp.w.proj_in' [{MLP_PROJ*2}, {HIDDEN}] Symbol c171d519-8793-4a8b-ac5e-d550347f30a6
  'bilinear_mlp.pre1' Index [1,:]
    'bilinear_mlp.fold_pre'
  'bilinear_mlp.w.proj_out' [{OUT_HIDDEN}, {MLP_PROJ}] Symbol e61637eb-9f17-4325-b2c2-5eb2518026cf
"""
)

printer.print(mlp)

# %%

BATCH_SHAPE = (3, 5, 7)
NEW_HIDDEN = 17
NEW_OUT_HIDDEN = 13
NEW_MLP_PROJ = 11
mlp_expanded = rc.Expander(
    ("bilinear_mlp.input", lambda _: rc.Scalar(0.7, (*BATCH_SHAPE, NEW_HIDDEN), "rep_input")),
    ("bilinear_mlp.w.proj_in", lambda _: rc.Scalar(0.7, (NEW_MLP_PROJ * 2, NEW_HIDDEN), "rep_proj_in")),
    ("bilinear_mlp.w.proj_out", lambda _: rc.Scalar(0.7, (NEW_OUT_HIDDEN, NEW_MLP_PROJ), "rep_proj_out")),
)(mlp, fancy_validate=True)

print()
printer.print(mlp_expanded)

# %%

# Here's something which is a bit sad: we can't partially replace symbols with different sizes

with pytest.raises(rc.ConstructEinsumAxisSizeDifferentError) as excinfo:
    rc.Expander(
        ("bilinear_mlp.input", lambda _: rc.Scalar(0.7, (*BATCH_SHAPE, NEW_HIDDEN), "rep_input")),
    )(mlp, fancy_validate=True)
print(excinfo.exconly())

# to resolve this, we'll move onto the next section, symbolic sizes

# %% [markdown]

# # Symbolic sizes
#
# There's support for 'symbolic sizes' which can be resolved to different sizes without an error.
# If an impossible set of size assignments is detected, this will error.
# These are global in some important sense - they must be resolvable to the same size throughout a circuit.

# %%

s0, s1, s2, *_ = rc.symbolic_sizes()
sym_size_mlp = P(
    f"""
'bilinear_mlp' Einsum h,h,oh->o
  'bilinear_mlp.pre0' Index [0,:]
    'bilinear_mlp.fold_pre' Rearrange (a:2 b) -> a:2 b
      'bilinear_mlp.pre' Einsum i,hi->h
                            # symbolic sizes can be indicated with <NUM>s
        'bilinear_mlp.input' [0s] Symbol 873a937e-2bb9-4f7f-b55e-a100db3dde52
                                 # we can also indicate products (including products of symbolic sizes)
        'bilinear_mlp.w.proj_in' [2*1s, 0s] Symbol c171d519-8793-4a8b-ac5e-d550347f30a6
  'bilinear_mlp.pre1' Index [1,:]
    'bilinear_mlp.fold_pre'
                                 # you can also use `rc.symbolic_sizes()` which returns the full list
  'bilinear_mlp.w.proj_out' [2s, {s1}] Symbol e61637eb-9f17-4325-b2c2-5eb2518026cf
"""
)

printer.print(sym_size_mlp)

# %%

# now we can partially replace!
sym_size = rc.Expander(
    ("bilinear_mlp.input", lambda _: rc.Scalar(0.7, (*BATCH_SHAPE, NEW_HIDDEN), "rep_input")),
)(sym_size_mlp, fancy_validate=True)
printer.print(sym_size)

# note the 'SetSymbolicShape' node. this is used to introduce contraints on symbolic sizes
# (you shouldn't typically need to do anything by hand with these nodes, they should just be automatically managed in expand)

# %%

# you can access a list of constraints if you so wish!
print(sym_size.symbolic_size_constraints)

# %%

# quick test of printing/parsing
parsed = P(
    """
'bilinear_mlp' Einsum pqrh,pqrh,oh->pqro
  'bilinear_mlp.pre0' Index [:,:,:,0,:]
    'bilinear_mlp.fold_pre' Rearrange c d e (a:2 b) -> c d e a:2 b
      'bilinear_mlp.pre' Einsum jkli,hi->jklh
        'rep_input' [3,5,7,17] Scalar 0.7
        'bilinear_mlp.w.proj_in set_shape' [2*1s,17] SetSymbolicShape
          'bilinear_mlp.w.proj_in' [2*1s,0s] Symbol c171d519-8793-4a8b-ac5e-d550347f30a6
  'bilinear_mlp.pre1' Index [:,:,:,1,:]
    'bilinear_mlp.fold_pre'
  'bilinear_mlp.w.proj_out' [2s,1s] Symbol e61637eb-9f17-4325-b2c2-5eb2518026cf
"""
)
assert rc.deep_normalize(parsed) == rc.deep_normalize(sym_size)

# %%

sym_size_out = rc.Expander(
    ("bilinear_mlp.w.proj_out", lambda _: rc.Scalar(0.7, (NEW_OUT_HIDDEN, NEW_MLP_PROJ), "rep_proj_out"))
)(sym_size, fancy_validate=True)
printer.print(sym_size_out)

# %%

# we can also do the other one
sym_size_in = rc.Expander(
    ("bilinear_mlp.w.proj_in", lambda _: rc.Scalar(0.7, (NEW_MLP_PROJ * 2, NEW_HIDDEN), "rep_proj_in")),
)(sym_size, fancy_validate=True)
printer.print(sym_size_in)


# %%

# or finish substituting
sym_size_in_out = rc.Expander(
    ("bilinear_mlp.w.proj_out", lambda _: rc.Scalar(0.7, (NEW_OUT_HIDDEN, NEW_MLP_PROJ), "rep_proj_out"))
)(sym_size_in, fancy_validate=True)
printer.print(sym_size_in_out)
assert sym_size_in_out.is_explicitly_computable
assert mlp_expanded == sym_size_in_out

# now note that the 'SetSymbolicShape' is gone and we're explicitly computable.
# We've resolved all of the shapes!

# %%

# we'll get an error if we make shapes inconsistent (in this case an einsum error)
MIS_MATCH_HIDDEN = 298
with pytest.raises(rc.ConstructError) as exc1:
    rc.Expander(
        ("bilinear_mlp.w.proj_in", lambda _: rc.Scalar(0.7, (NEW_MLP_PROJ * 2, MIS_MATCH_HIDDEN), "rep_proj_in")),
    )(sym_size, fancy_validate=True)
print(exc1.exconly())

# %%

# you can also get symbolic constraint errors like this example
scl0, scl1, v, new0, new1 = P.parse_circuits(
    """
# Here we are parsing a file with multiple circuits into a list of circuits
1 [7] Scalar 1.3
3 [7,3] Scalar 1.5
0 Einsum i,i,ij,j->i
  1
  2 [7] Scalar 1.5
  3
  4 [3] Scalar 1.7
'new0' [0s] Symbol
'new1' [0s] Symbol
"""
)

with pytest.raises(rc.SymbolicSizeSetFailedToSatisfyContraintsError) as exc2:
    # expand node just expands 1 level on replaced inputs (like expander)
    rc.expand_node(v, [scl0, new0, scl1, new1])
print(exc2.exconly())

# %% [markdown]

# # Modules
#
# TODO: improve this explanation as needed
#
# Modules are a circuit type which themselves encode a symbolic substitution (where substitution is done via expand as shown above).
# We'll go through examples in a second, but let's do a quick explanation first.
#
# If we imagine a circuit with symbols as having 'free variables', then Modules
# are a call site which binds (some of) those free variables to specific
# inputs.
#
# Note that modules don't bind symbols which are bound by a child module - we'll go through some examples of this.
#
# Modules are built of two parts:
# - a ModuleSpec
# - and inputs
#
# ModuleSpecs can be thought of as 'lambdas' analogously to Modules being call sites. They consist of:
# - A spec circuit (which has symbols/'free variables', aka the function body)
# - And a list of 'ModuleArgSpec' which specifies which symbol to substitute out as well as any restrictions on expansion.
#
# Now let's build some modules!

# %%

input_sym, proj_in_sym, proj_out_sym, mlp = P.parse_circuits(
    """
'bilinear_mlp.input' [0s] Symbol 873a937e-2bb9-4f7f-b55e-a100db3dde52
'bilinear_mlp.w.proj_in' [2*1s, 0s] Symbol c171d519-8793-4a8b-ac5e-d550347f30a6
'bilinear_mlp.w.proj_out' [2s, 1s] Symbol e61637eb-9f17-4325-b2c2-5eb2518026cf
'bilinear_mlp' Einsum h,h,oh->o
  'bilinear_mlp.pre0' Index [0,:]
    'bilinear_mlp.fold_pre' Rearrange (a:2 b) -> a:2 b
      'bilinear_mlp.pre' Einsum i,hi->h
                            # symbolic sizes can be indicated with <NUM>s
        'bilinear_mlp.input'
        'bilinear_mlp.w.proj_in'
  'bilinear_mlp.pre1' Index [1,:]
    'bilinear_mlp.fold_pre'
  'bilinear_mlp.w.proj_out'
"""
)
input_sym, proj_in_sym, proj_out_sym = [x.cast_symbol() for x in [input_sym, proj_in_sym, proj_out_sym]]

# we'll start with the straightforward, but verbose construction strategy
# see docs for ModuleArgSpec as needed
input_arg = rc.ModuleArgSpec(input_sym, batchable=True, expandable=True, ban_non_symbolic_size_expand=True)
proj_in_arg = rc.ModuleArgSpec(proj_in_sym, batchable=False, expandable=True, ban_non_symbolic_size_expand=True)
proj_out_arg = rc.ModuleArgSpec(proj_out_sym, batchable=False, expandable=True, ban_non_symbolic_size_expand=True)

mlp_spec = rc.ModuleSpec(
    mlp, [input_arg, proj_in_arg, proj_out_arg], check_all_inputs_used=True, check_unique_arg_names=True
)

input_arr = rc.Array.randn(*BATCH_SHAPE, NEW_HIDDEN, name="b.input")
proj_in_arr = rc.Array.randn(2 * NEW_MLP_PROJ, NEW_HIDDEN, name="b.proj_in")
proj_out_arr = rc.Array.randn(NEW_OUT_HIDDEN, NEW_MLP_PROJ, name="b.proj_out")

BATCH_SHAPE = (3, 5, 7)
NEW_HIDDEN = 17
NEW_OUT_HIDDEN = 13
NEW_MLP_PROJ = 11

# Now this module binds the inputs (input_arr, proj_in_arr, proj_out_arr)
# to the arg spec symbols (input_sym, proj_in_sym, proj_out_sym)
mlp_call = rc.Module(
    mlp_spec,
    "b.bilinear_mlp",
    **{
        "bilinear_mlp.input": input_arr,
        "bilinear_mlp.w.proj_in": proj_in_arr,
        "bilinear_mlp.w.proj_out": proj_out_arr,
    },
)

# modules have children: spec_circuit, input_0 ! input_sym_0, input_1 ! input_sym_1,...
# where input_0 is bound to input_sym_0
#
# If ModuleArgSpec flags aren't the default, then the are printed like input_0 ! input_sym_0 ftt
# where t/f correspond to true and false
print()
printer.print(mlp_call)

# %%

# as per usual, parsing works on this representation
assert mlp_call == P(mlp_call.repr())

# %%

# this module is extentionally equal to the following substitution (done with expand)
printer.print(mlp_call.substitute())

# %%

# we could have also used the 'new_flat' factory (which uses the argument order)
flat_mlp_call = rc.Module.new_flat(
    mlp_spec,
    input_arr,
    proj_in_arr,
    proj_out_arr,
    name="b.bilinear_mlp",
)
assert flat_mlp_call == mlp_call

# %%

# it's ofc possible to define modules/spec which only partially bind.

mlp_weights_spec = rc.ModuleSpec(mlp, [proj_in_arg, proj_out_arg])
mlp_weights = rc.Module(
    mlp_weights_spec,
    "bw.bilinear_mlp",
    **{
        "bilinear_mlp.w.proj_in": proj_in_arr,
        "bilinear_mlp.w.proj_out": proj_out_arr,
    },
)
printer.print(mlp_weights)

# %%

# and then we can nest this module as well.
mlp_w_inp_spec = rc.ModuleSpec(mlp_weights, [input_arg])
mlp_nested_mod = rc.Module(
    mlp_w_inp_spec,
    "b.bilinear_mlp",
    **{
        "bilinear_mlp.input": input_arr,
    },
)
printer.print(mlp_nested_mod)

# %%

substitutor = rc.Updater(lambda x: x.cast_module().substitute())
sub_inner_only = substitutor(mlp_nested_mod, "bw.bilinear_mlp")
printer.print(sub_inner_only)

# %%

sub_outer_only = substitutor(mlp_nested_mod, "b.bilinear_mlp")
printer.print(sub_outer_only)

# %%

# we can also sub all
assert mlp_call.substitute() == rc.substitute_all_modules(mlp_nested_mod)

# %%

# NOTE: in some cases, substituting an inner module may require Expander due to symbolic sizes!

# %%

# this is typically the most consise way to construct a module from a circuit with free symbols. See docs for details.
mlp_call_bind = rc.module_new_bind(
    mlp,
    ("bilinear_mlp.input", input_arr),
    rc.BindItem("bilinear_mlp.w.proj_in", proj_in_arr, batchable=False),
    rc.BindItem("bilinear_mlp.w.proj_out", proj_out_arr, batchable=False),
    name="b.bilinear_mlp",
)
assert mlp_call_bind == mlp_call

# %%

# there are also some utils for getting free symbols and constructing like this
assert rc.get_free_symbols(mlp_call) == []
assert rc.get_free_symbols(mlp) == [input_sym, proj_in_sym, proj_out_sym]
assert rc.get_free_symbols(mlp_weights) == [input_sym]

# this will use the default values for ModuleArgSpec (all true)
with_all_free = rc.ModuleSpec.new_free_symbols(mlp)
flat_mod = rc.Module.new_flat(with_all_free, input_arr, proj_in_arr, proj_out_arr, name="b.bilinear_mlp")

# there's printing/parsing shorthand for the all true case for Modules: all_t (see top line in print)
print()
printer.print(flat_mod)

# %% [markdown]

# modules are nice because they allow you to have fixed copy of the function
# body instead of needing to duplicate it with different shapes.
# this can allow for stuff like:
# - module specific rewrites for a given spec (in the same way we have einsum or rearrange specfic rewrites)
# - reusing a spec circuit multiple times with different shapes in the same circuit

# %%

RUNNING_HIDDEN = 20
RUNNING_MLP_PROJ = 11

new_proj_in_arr = rc.Array.randn(2 * RUNNING_MLP_PROJ, NEW_OUT_HIDDEN, name="b_new.proj_in")
new_proj_out_arr = rc.Array.randn(RUNNING_HIDDEN, RUNNING_MLP_PROJ, name="b_new.proj_out")

mlp_2_layer = rc.module_new_bind(
    mlp,
    ("bilinear_mlp.input", mlp_call),  # using this circuit as input
    rc.BindItem("bilinear_mlp.w.proj_in", new_proj_in_arr, batchable=False),
    rc.BindItem("bilinear_mlp.w.proj_out", new_proj_out_arr, batchable=False),
    name="outer_b.bilinear_mlp",
)

# note how this has the same spec circuit operating on different shapes!
# and this spec circuit is only printed once
print()
printer.print(mlp_2_layer)

# %%

# we can do rewrites on specs (but note that indexing into symbolic sizes might not work as desired!)
perm_einsum = rc.Updater(lambda x: rc.nest_einsums(x.cast_einsum(), (2, 1, 0)))(mlp_2_layer, "bilinear_mlp")
printer.print(perm_einsum)

# %%

# we can also do rewrites on just one of the specs
outer_spec_matcher = rc.IterativeMatcher("outer_b.bilinear_mlp").spec_circuit_matcher()
perm_einsum_diff = outer_spec_matcher.update(
    mlp_2_layer, lambda x: rc.nest_einsums(x.cast_einsum(), (2, 1, 0)).rename("bilinear_mlp_permute_args")
)
printer.print(perm_einsum_diff)

# %%


# you might find it useful to rename all circuits on the path to a given circuit within a module spec

# within the spec, match all parents of bilinear_mlp.pre and that circuit itself
within_spec_path_to_matcher = outer_spec_matcher.chain(rc.Matcher.match_any_found("bilinear_mlp.pre"))

perm_inside_einsum = within_spec_path_to_matcher.update(mlp_2_layer, lambda x: x.rename(x.name + " perm"))  # do rename
perm_inside_einsum = outer_spec_matcher.chain("bilinear_mlp.pre perm").update(
    perm_inside_einsum,
    lambda x: rc.nest_einsums(x.cast_einsum(), (1, 0)),  # and then finally update the circuit inside the spec
)
printer.print(perm_inside_einsum)

# %%

# it's sometimes useful to conform all symbolic shapes to their actual shape
# this results in the module specs no longer being equal because they were called on different shapes
printer.print(rc.conform_all_modules(mlp_2_layer))

# %%

# another useful thing we can do is get bound sub circuits
# (by pushing down modules)
# NOTE: ModulePusher has quite complex handling around batching dims. I've
# tried to test this well, but I currently put reasonably high probability on
# this being buggy in complex batching cases! (it would likely panic if buggy)
outer_bound_pre = rc.ModulePusher().get_unique_push_down_modules(
    mlp_2_layer, rc.IterativeMatcher("outer_b.bilinear_mlp").spec_circuit_matcher().chain("bilinear_mlp.pre")
)
printer.print(outer_bound_pre)
assert outer_bound_pre.is_explicitly_computable
actual = rc.substitute_all_modules(mlp_2_layer).get_unique(
    rc.restrict(rc.IterativeMatcher("bilinear_mlp.pre"), end_depth=5)
)
torch.testing.assert_close(outer_bound_pre.evaluate(), actual.evaluate())

# %%

# or for the inner module
inner_bound_pre = rc.ModulePusher().get_unique_push_down_modules(
    mlp_2_layer, rc.IterativeMatcher("b.bilinear_mlp").spec_circuit_matcher().chain("bilinear_mlp.pre")
)
printer.print(inner_bound_pre)
assert inner_bound_pre.is_explicitly_computable
actual = rc.substitute_all_modules(mlp_2_layer).get_unique(
    rc.restrict(rc.IterativeMatcher("bilinear_mlp.pre"), start_depth=5)
)
torch.testing.assert_close(inner_bound_pre.evaluate(), actual.evaluate())

# TODO: better demo of fancy push down cases + namer?

# %%

# another convenient util is drop/filter_module_inputs
dropped = drop_module_inputs(mlp_2_layer, "bilinear_mlp.input")
assert not dropped.is_explicitly_computable  # no longer computable because we don't have all symbols bound
printer.print(dropped)

# %%

# we can also extract bound symbols to a wrapping module
# note that this requires that the symbols are always bound to the same input -
# this is necessary, otherwise substitution becomes incorrect.
extracted = rc.extract_symbols_get(mlp_call, {"bilinear_mlp.w.proj_in"})
torch.testing.assert_close(extracted.evaluate(), mlp_call.evaluate())
printer.print(extracted)

# %%

# this allows for partial substitution:
printer.print(extracted.substitute())

# %%

# extracting batched inputs is a bit tricky. By default this is disallowed
with pytest.raises(rc.ExtractSymbolsBatchedInputError) as exc3:
    rc.extract_symbols_get(mlp_call, {"bilinear_mlp.input"})
print(exc3.exconly())

# %%

# but, as mentioned in the error message, we can use conform_batch_if_needed to
# handle this case (which requires conforming).

x = rc.extract_symbols_get(mlp_call, {"bilinear_mlp.input"}, conform_batch_if_needed=True)
printer.print(x)
torch.testing.assert_close(x.evaluate(), mlp_call.evaluate())

# %% [markdown]

# ### Nested batching
#
# TLDR: module batching works the way you would expect - it always adds new
# batch dims which are batched together.
#
# Modules add new batch dims as needed and can batch over each other in nested
# ways. For a complex example of this, see test_complex_batching in
# interp/circuit/interop_rust/test_module.py (I might add a single example at
# some point).
#
# This sometimes requires adding rearranges, so be aware of that.

# %% [markdown]

# ### Module rules
#
# We guarantee that substitution in any order is extentionally equal (but
# different orders may result in different rearranges and different names for
# these rearranges). Various transformation require reshaping symbols, so we
# also want validity guarantees wrt to this.
#
# First, we have to resolve the case where a parent module binds a symbol and a child
# module binds that same symbol. In this case, the inner most binding always
# takes priority. NOTE: this differs from how 'Expander' works (intentionally)!
#
# We also have to enforce the following requirements:
#
# - No higher order functions: If a module substitutes node x into the spec of a nested module, then that nested
#   module can't substitute inside of node x (this ensures we avoid higher order
#   functions, which cause all kinds of problems as well as being turing
#   complete!)
# - Spec circuits can't have 'near misses': free symbols which have equivalent
#   identification (name and uuid), but are overall not equal (typically due to
#   shape). This ensures that reshaping symbols is always fine.

# %%


# same symbol, different value - the inner value is used
s = """
'b.arg' [] Scalar 1.2
'a' Module
  'b' Module
    'sym' [] Symbol c9517c8e-0b5a-4ebf-94c8-6da413c9f5d1
    'b.arg' ! 'sym'
  'a.arg' [] Scalar 1.3 ! 'sym'
"""
# the outer arg is unused!
b_arg, circ_override = rc.Parser(module_check_all_inputs_used=False).parse_circuits(s)
assert circ_override.cast_module().spec.are_args_used() == [False]
assert rc.substitute_all_modules(circ_override) == b_arg
assert rc.substitute_all_modules(circ_override.cast_module().substitute()) == b_arg

# %%

# within the same module, later children are considered 'more inner' and substituted first
s = """
'b.arg' [] Scalar 1.2
'a' Module
  'sym' [] Symbol c9517c8e-0b5a-4ebf-94c8-6da413c9f5d1
  'a.arg' [] Scalar 1.3 ! 'sym'
  'b.arg' ! 'sym'
"""
b_arg, circ_same = rc.Parser(module_check_all_inputs_used=False).parse_circuits(s)
assert circ_same.cast_module().spec.are_args_used() == [False, True]
assert rc.substitute_all_modules(circ_same) == b_arg

# %%

# extract gets the outer most binding
orig = rc.Add(circ_override, rc.Scalar(5.0), name="new_outer")
ex = rc.extract_symbols_get(orig, "sym")
assert isinstance(ex, rc.Module)
torch.testing.assert_close(ex.evaluate(), orig.evaluate())
ex.print()

# %%

# the same applies for more/less inner children
orig = rc.Add(circ_same, rc.Scalar(5.0), name="new_outer")
ex = rc.extract_symbols_get(orig, "sym")
ex.print()
assert isinstance(ex, rc.Module)
torch.testing.assert_close(ex.evaluate(), orig.evaluate())
ex.print()

# %%

# if you push down through an override then the flattened module with have two
# args, identically to the circ_same case. Then, by default this unused arg would be
# removed (but we can disable removing it to see what that looks like).
# Removal is done using a callback. We can change this callback to a noop.
#
# See the docs for ModuleConstructCallback and ModulePusher.
double_push_down = rc.ModulePusher(flatten_modules=True, module_construct_callback=rc.ModulePusher.noop_callback())(
    circ_override, traversal=rc.new_traversal(term_early_at="sym")
)
assert circ_same == double_push_down

# %%

# higher order function case
s = """
'a' Module
  'b' Module
    'a.sym' [] Symbol c9517c8e-0b5a-4ebf-94c8-6da413c9f5d1
    'same.arg' [] Scalar 1.3 ! 'b.sym' [] Symbol 33ef7e97-434e-4831-a292-d10818df4752
  'b.sym' ! 'a.sym'
"""
with pytest.raises(rc.SubstitutionCircuitHasFreeSymsBoundByNestedModuleError) as exc6:
    # by default, this would also error on 'not all args present'
    rc.Parser(module_check_all_inputs_used=False)(s)
print(exc6.exconly())

# %%

# fancier higher order function case
s = """
'a' Module
  'x' Add
    'b' Module
      'mul' Einsum ,a->a
        'a.sym' [] Symbol c9517c8e-0b5a-4ebf-94c8-6da413c9f5d1
        'sub' [7] Scalar 7.3
      'same.arg' [] Scalar 1.3 ! 'b.sym' [] Symbol 33ef7e97-434e-4831-a292-d10818df4752
    'a.sym'
  'b.sym' ! 'a.sym'
"""
with pytest.raises(rc.SubstitutionCircuitHasFreeSymsBoundByNestedModuleError) as exc7:
    # by default, this would also error on 'not all args present'
    rc.Parser(module_check_all_inputs_used=False)(s)
print(exc7.exconly())

# %%

# 'near miss' case
s = """
'b' Module
  'sum' Add
    0 'sym' [] Symbol c9517c8e-0b5a-4ebf-94c8-6da413c9f5d1
    # note the different shape!
    1 'sym' [3] Symbol c9517c8e-0b5a-4ebf-94c8-6da413c9f5d1
  'b.arg' [] Scalar 73.3874 ! 0 'sym'
"""
# the outer arg is unused!
with pytest.raises(rc.SubstitutionFoundNEQFreeSymbolWithSameIdentificationError) as exc11:
    P(s)
print(exc11.exconly())

# %% [markdown]

# ### Renaming/replace path utils
#
# it can also be useful to substitute as module with renaming

# %%

# by default we get a bunch of repeated items
printer.print(mlp_2_layer.substitute())

# but we can avoid this by using a name prefix
printer.print(mlp_2_layer.substitute(name_prefix="outer."))

# %%

# note that substituting only renames circuits which are in the replaced path
# here a more complex example so you can see what's going on
s = """
'a.norm' Module
  'ln' Add
    'ln.w.bias' [0s] Symbol 621c7792-0177-45ab-87c5-7ff1c3bec487
    'ln.y_scaled' Einsum h,h->h
      'ln.y' Einsum h,->h
        'ln.mean_subbed' Add
          'ln.input' [0s] Symbol 981b4d2a-711b-4a9d-a11c-d859c311e80c
          'ln.neg_mean' Einsum h,z,->z
            'ln.input'
            'ln.neg' [1] Scalar -1
            'ln.c.recip_hidden_size' GeneralFunction reciprocal
              'ln.c.hidden_size' GeneralFunction last_dim_size
                'ln.input'
        'ln.rsqrt' GeneralFunction rsqrt
          'ln.var_p_eps' Add
            'ln.c.eps' [] Scalar 0.00001
            'ln.var' Einsum h,h,->
              'ln.mean_subbed'
              'ln.mean_subbed'
              'ln.c.recip_hidden_size'
      'ln.w.scale' [0s] Symbol 0fa341c3-34b3-4699-847f-08674808b28a
  'a.norm.input' [1s,0s] Symbol 6a622698-fd68-4d25-aeee-e8d38e68049e ! 'ln.input'
  'a.ln.w.bias' [0s] Symbol 2c737289-2702-404c-a22e-ad37c2652620 ! 'ln.w.bias'
  'a.ln.w.scale' [0s] Symbol c564149d-e226-4e3d-8e47-7d6e2ceea99e ! 'ln.w.scale'
"""
a_norm = P(s)
# (see below for extract symbols explanation)
extracted_a_norm = rc.extract_symbols_get(a_norm, {"ln.w.bias", "ln.w.scale"})
extracted_sub = extracted_a_norm.substitute(name_prefix="w.a.")
expected_extracted = """
'a.norm' Module
  'w.a.ln' Add
    'a.ln.w.bias' [0s] Symbol 2c737289-2702-404c-a22e-ad37c2652620
    'w.a.ln.y_scaled' Einsum h,h->h
      'ln.y' Einsum h,->h
        'ln.mean_subbed' Add
          'ln.input' [0s] Symbol 981b4d2a-711b-4a9d-a11c-d859c311e80c
          'ln.neg_mean' Einsum h,z,->z
            'ln.input'
            'ln.neg' [1] Scalar -1
            'ln.c.recip_hidden_size' GeneralFunction reciprocal
              'ln.c.hidden_size' GeneralFunction last_dim_size
                'ln.input'
        'ln.rsqrt' GeneralFunction rsqrt
          'ln.var_p_eps' Add
            'ln.c.eps' [] Scalar 0.00001
            'ln.var' Einsum h,h,->
              'ln.mean_subbed'
              'ln.mean_subbed'
              'ln.c.recip_hidden_size'
      'a.ln.w.scale' [0s] Symbol c564149d-e226-4e3d-8e47-7d6e2ceea99e
  'a.norm.input' [1s,0s] Symbol 6a622698-fd68-4d25-aeee-e8d38e68049e ! 'ln.input'
"""
assert extracted_sub == P(expected_extracted)

printer.print(extracted_sub)

# %%

# we can also do map_on_replaced_path and rename on replaced path without subbing
extracted_a_norm.rename_on_replaced_path(prefix="hi").print()
extracted_a_norm.map_on_replaced_path(lambda x: x.rename(x.name.replace(".", "_"))).print()

# %%

# and this does the right thing  when the symbol is overridden by an inner module (this is just a test)
s = """
'a' Module
  'add' Add
    'b' Module
      'sym' [] Symbol c9517c8e-0b5a-4ebf-94c8-6da413c9f5d1
      'b.arg' [] Scalar 1.2 ! 'sym'
    'sym'
  'a.arg' [] Scalar 1.3 ! 'sym'
"""
circ_override = rc.Parser()(s).cast_module()

expected = """
'a' Module
  'hiadd' Add
    'b' Module
      'sym' [] Symbol c9517c8e-0b5a-4ebf-94c8-6da413c9f5d1
      'b.arg' [] Scalar 1.2 ! 'sym'
    'sym'
  'a.arg' [] Scalar 1.3 ! 'sym'
"""
assert circ_override.rename_on_replaced_path(prefix="hi") == rc.Parser()(expected)

s = """
'a' Module
  'add' Add
    'b' Module
      'sym' [] Symbol c9517c8e-0b5a-4ebf-94c8-6da413c9f5d1
      'add_inner' Add ! 'sym'
        'new' [] Scalar 1.7
        'sym'
    'sym'
  'a.arg' [] Scalar 1.3 ! 'sym'
"""
circ_override = rc.Parser()(s).cast_module()

expected = """
'a' Module
  'hiadd' Add
    'hib' Module
      'sym' [] Symbol c9517c8e-0b5a-4ebf-94c8-6da413c9f5d1
      'hiadd_inner' Add ! 'sym'
        'new' [] Scalar 1.7
        'sym'
    'sym'
  'a.arg' [] Scalar 1.3 ! 'sym'
"""
assert circ_override.rename_on_replaced_path(prefix="hi") == rc.Parser()(expected)

# %%

# TODO: add nice utility for renaming symbols (e.g., on conform?)

# %% [markdown]

# ### Module errors
#
# You can get really long errors due to nested substitution resulting in incompatible dtypes.
# But, the root cause should typically make it clear what went wrong!

# %%


t, _, _ = mod_l.TransformerParams(mod_l.TransformerBlockParams(), num_layers=7).garbage_call(
    hidden_size=15,
    head_size=3,
    num_heads=5,
    seq_len=11,
    batch_shape=(2, 3),
)
t = t.get_unique("t.bind_w")

# %%

print(t.device_dtype)

# %%

with pytest.raises(rc.MiscInputChildrenMultipleDtypesError) as exc8:
    rc.module_new_bind(
        t, ("t.input", rc.Array.randn(2, 3, 11, 15, device_dtype=rc.TorchDeviceDtypeOp(dtype="float64")))
    )
# this error is *huge*, but it should be pretty clear what went wrong just by the root cause
# we might improve the situation somewhat later.
print(exc8.exconly())


# %%

# random aside:
# if you call the optimizer on something which still has some symbolic sizes
# (which aren't removed by conform_all_modules), then you'll get a warning
s_with_sym = """
0 Einsum ij,jk->ik
  1 [1s,2s] Symbol 6594dafb-81f8-49a3-8381-6ffc4746b474
  2 [2s,3s] Symbol bc646cfe-a299-4580-9a54-d8d5001a566b
"""

with pytest.warns(rc.OptimizingSymbolicSizeWarning) as w:
    rc.optimize_circuit(rc.Parser()(s_with_sym))
print(w.list[0])

# %%

# another random aside: modules do various global caching. This caching needs to
# retain circuits which results in keeping those circuits around forever.
# This can result in Arrays not being freed
#
# You can clear this cache via
rc.clear_module_circuit_caches()
# to ensure that circuits are freed.
# This is somewhat of a hack.
