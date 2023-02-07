import pytest
import torch
from torch.testing import assert_close

from rust_circuit import FINISHED, Add, Array, Einsum, IterateMatchResults, IterativeMatcher, Matcher
from rust_circuit import NestAddsSpecInfo as AddInfo
from rust_circuit import NestEinsumsSpec
from rust_circuit import NestEinsumsSpecInfo as EinInfo
from rust_circuit import (
    NestMatcher,
    NestMatcherMatchedMultipleAndMustBeUniqueError,
    NestPermutationMissesIdxsAndNoRestInSpecError,
)
from rust_circuit import NestRest as Rest
from rust_circuit import (
    PermIntsNotUniqueError,
    PrintOptions,
    einsum_flatten,
    nest_adds,
    nest_einsums,
    new_traversal,
    restrict,
)

# os.environ["RUST_BACKTRACE"] = "1"


# %% [markdown]

# # Basic idea

# In this demo, we'll look at nesting rewrites.
# These are powerful and convenient rewrites for circuit types
# which can be manipulated into trees. Specifically, we can nest einsums and adds.
#
# We'll be looking at nest_einsums and nest_adds

# (This is the same as the original rearrange_muls rewrite in python, but with a decent number of additional features)

# %%

# first, we'll create a flat einsum we want to rearrange
a, b, c, d, f, e = 2, 3, 4, 5, 6, 7
x = Einsum.from_einsum_string(
    "a b c, b c d, a e, b d, f c, a c d, e a f -> a c d f",
    *(
        Array(torch.randn(*shape, dtype=torch.float64), name=name)
        for (shape, name) in [
            ((a, b, c), "abc"),
            ((b, c, d), "bcd"),
            ((a, e), "ae"),
            ((b, d), "bd"),
            ((f, c), "ec"),
            ((a, c, d), "acd"),
            ((e, a, f), "eaf"),
        ]
    )
).normalize_ints()

x.print()

# %%

# nest einsums take a nested sequence structure.
# Each integer refers to an index into the flattened einsum.

# So here, we group the first 3 arguments in one subset and the last 4 arguments into another subset
z = nest_einsums(x, ((0, 1, 2), (3, 4, 5, 6)))
assert_close(x.evaluate(), z.evaluate())
z.print()

# %%


# Here's a more complex setup
z = nest_einsums(x, ((5, (1, 3)), 2, ((4,), 0), 6))
assert_close(x.evaluate(), z.evaluate())
z.print()

# %%

# ints must be a permutation of all leaf locations
with pytest.raises(NestPermutationMissesIdxsAndNoRestInSpecError):
    nest_einsums(x, (0, 2))
with pytest.raises(PermIntsNotUniqueError):
    nest_einsums(x, (0, 1, 1, 2, 3, 4, 5, 6))
with pytest.raises(PermIntsNotUniqueError):
    nest_einsums(x, (0, 1, 2, 3, 4, (5, 2), 6))

# %%

# this also works for adds!

x_add = Add(
    *(
        Array(torch.randn(*shape, dtype=torch.float64), name=name)
        for (shape, name) in [
            ((a, b, c), "abc"),
            ((a, 1, 1), "a11"),
            ((1, b, 1), "1b1"),
            ((b, c), "bc"),
            ((c,), "c"),
            ((1, c), "1c"),
        ]
    )
)
x_add.print()

z_add = nest_adds(x_add, ((0, 4, 2), (3, 1, 5)))
assert_close(x_add.evaluate(), z_add.evaluate())
z_add.print()

# we'll proceed using einsums, but all of this applies to adds also!

# %%

# you can print with the corresponding numbers
my_printer = PrintOptions(number_leaves=True, traversal=new_traversal(term_early_at=~Matcher(Einsum)), bijection=False)
# the 'number_leaves' print option implies printing as a tree (not a dag), so
# be carefully, this might be very big if you don't term_early_at
my_printer.print(x)

# you can also use
PrintOptions().einsum_nest_default().print(x)

# there's also
PrintOptions().add_nest_default().print(x_add)

# going forward we'll use this printer
opt = PrintOptions().einsum_nest_default()


# %% [markdown]

# # Additional Feats

# That's pretty much the entire basic idea, but there are a bunch of additional features which can be quite useful.

# %% [markdown]

# ## 'Rest'

# %%


# You can specify a 'rest' argument which groups all arguments not otherwise handled.
z = nest_einsums(x, ((5, (1, 3)), 2, Rest()))
assert_close(x.evaluate(), z.evaluate())
opt.print(z)

# %%

# you can also make this 'flatten' one level inline (imagine this as 'splatting' the rest (splat is the '*' python operator))
z_rest = nest_einsums(x, ((5, (1, 3)), 2, Rest(flat=True)))
assert_close(x.evaluate(), z_rest.evaluate())
opt.print(z_rest)

# %%

# If Rest matches nothing, it will be flattened in
z = nest_einsums(x, (0, 1, 2, 3, 4, 5, 6, Rest()))
assert_close(x.evaluate(), z.evaluate())
opt.print(z)


# %% [markdown]

# ## Matching instead of numbering

# %%

# numbering can some times be annoying, so we can instead pass a matcher
z_new = nest_einsums(x, (("acd", ("bcd", Matcher.regex("^bd$"))), 2, Rest(flat=True)))
# (this is any IterativeMatcherIn, the iterative matcher is run starting at the root einsum)
# warning: in python, 'Regex' matched starts, but in rust it can match anywhere. So we anchor with '^'!
assert z_new == z_rest  # exactly the same as previous example

# %%

# this matcher has to match exactly one thing!!!
with pytest.raises(NestMatcherMatchedMultipleAndMustBeUniqueError):
    nest_einsums(x, (("acd", Matcher.regex("b")), 2, Rest(flat=True)))

# %% [markdown]

# ## Matching multiple with a matcher

# %%

# we can also explicitly pass in a matcher which can match multiple via the `NestMatcher` class.
# this has roughly the same semantics as `Rest` including the ability to specify `flat=True`
z = nest_einsums(x, (NestMatcher(Matcher.regex("b")), Rest(flat=True)))
assert_close(x.evaluate(), z.evaluate())
opt.print(z)
# (we could change things so that a matcher which isn't wrapped (like previous
# section) can match multiple, but this doesn't seem like right default IMO)

# %%

z = nest_einsums(x, ((NestMatcher(Matcher.regex("b"), flat=True), 2), Rest(flat=True)))
assert_close(x.evaluate(), z.evaluate())
opt.print(z)

# %%

# there are some validation args also
z = nest_einsums(
    x,
    (
        (
            NestMatcher({"bd", "ae"}, fancy_validate=True),  # checks that all things in the set indeed match something
            NestMatcher("abc", assert_unique=True),
            NestMatcher("347387", assert_exists=False),  # matches nothing, flattened in
        ),
        Rest(flat=True),
    ),
)
assert_close(x.evaluate(), z.evaluate())
opt.print(z)


# %% [markdown]

# ## Partial traversals

# %%

# sometimes we have deep einsum trees and we only want to traverse down part of
# the way and rearrange up to the that point.

# for example:
opt.print(z_rest)
# here, let's say we don't want to rearrange bcd * bd


# a traversal is just an iterative matcher where we *only* care about whether or not it has terminated.
print_traversal = restrict(opt.traversal, term_early_at="bcd * bd")

# note that the numbers here are the ones obtained when partially traversing! you might want to use matchers
# here, or print with the same filter as you are traversing with.
opt.evolve(traversal=print_traversal).print(z_rest)

z_filter = nest_einsums(z_rest, (1, 2, 3, 0, 4, 5), traversal=print_traversal)
assert_close(x.evaluate(), z_filter.evaluate())
opt.evolve(traversal=print_traversal).print(z_filter)

# %%

# the previous example can also be done via setting an end_depth
depth_traverse = restrict(opt.traversal, end_depth=3)

opt.evolve(traversal=depth_traverse).print(z_rest)
z_filter_new = nest_einsums(z_rest, (1, 2, 3, 0, 4, 5), traversal=depth_traverse)
assert z_filter_new == z_filter

# %% [markdown]

# ## Setting node info

# it can often be nice to set names etc on a given group.
# There's a nice API for this!

# %%

# we can specify names inside the spec with EinInfo
z_named = nest_einsums(x, ((5, EinInfo((1, 3), name="my_new_name")), 2, EinInfo(Rest(), name="named_rest")))
assert_close(x.evaluate(), z_named.evaluate())
_ = Matcher("my_new_name").get_unique(z_named)
_ = Matcher("named_rest").get_unique(z_named)
opt.print(z_named)

# %%

# We can also permute the output axes of sub einsums
z_perm = nest_einsums(
    x, ((5, EinInfo((1, 3), name="my_new_name", out_axes_perm=[2, 0, 1])), 2, EinInfo(Rest(), name="named_rest"))
)
assert_close(x.evaluate(), z_perm.evaluate())
assert_close(
    Matcher("my_new_name").get_unique(z_perm).evaluate(),
    Matcher("my_new_name").get_unique(z_named).rearrange_str("a b c -> c a b").evaluate(),
)
_ = Matcher("named_rest").get_unique(z_perm)
opt.print(z_perm)

# %%

# nest_adds just support setting the name (because they don't have output axes to permute)
z_add = nest_adds(x_add, ((5, AddInfo((1, 3), name="my_new_name")), 2, AddInfo(Rest(), name="named_rest")))
assert_close(x_add.evaluate(), z_add.evaluate())
_ = Matcher("my_new_name").get_unique(z_add)
_ = Matcher("named_rest").get_unique(z_add)
opt.print(z_add)


# %% [markdown]

# ## Retaining node names

# generally when doing rewrites, it's useful to keep names and other node properties when possible.
# nest_{einsums/adds} does this!

# %%

# let's first make an einsum with a bunch of names
z_with_names = nest_einsums(x, (4, EinInfo((3, EinInfo((2, 1), "other"), EinInfo(((0, 5), 6), "here")), "outer")))
assert_close(x.evaluate(), z_with_names.evaluate())
print()
opt.print(z_with_names)

z_here, z_other, z_outer = [Matcher(n).get_unique(z_with_names) for n in ["here", "other", "outer"]]

# %%


# if we nest things such that some subnode could be made extensionally equal (if its children are the same numbers),
# the name and output axis ordering will be retained!
z_keep_here = nest_einsums(z_with_names, (0, (1, 2, 3, (6, 4, 5))))
assert_close(x.evaluate(), z_keep_here.evaluate())
print()
opt.print(z_keep_here)
assert_close(Matcher("here").get_unique(z_keep_here).evaluate(), z_here.evaluate())
assert_close(Matcher("outer").get_unique(z_keep_here).evaluate(), z_outer.evaluate())
assert Matcher("other").get_unique_op(z_keep_here) is None

# %%

# We can also do this with rest and matchers (this example is silly, but you get the idea
z_keep_via_match = nest_einsums(
    z_with_names, (0, (NestMatcher({"abc", "acd", "eaf"}, flat=True), NestMatcher({"ae", "bcd"}, flat=True), 1))
)
assert_close(x.evaluate(), z_keep_via_match.evaluate())
print()
opt.print(z_keep_via_match)
assert_close(Matcher("here").get_unique(z_keep_via_match).evaluate(), z_here.evaluate())
assert_close(Matcher("outer").get_unique(z_keep_via_match).evaluate(), z_outer.evaluate())
assert_close(Matcher("other").get_unique(z_keep_via_match).evaluate(), z_other.evaluate())

# %%

z_keep_via_rest = nest_einsums(z_with_names, (0, (Rest(flat=True), NestMatcher({"ae", "bcd"}, flat=True), 1)))
assert z_keep_via_rest == z_keep_via_match

# %% [markdown]

# ## Shrinking nodes with extra retained dims (not very important)

# %%


# this retaining will (by default) retain extra axes which could be reduced
sub_ein = Einsum.from_einsum_string(
    "eab,ae->ab", Array.randn(e, a, b, name="eab"), Array.randn(a, e, name="ae"), name="sub"
)
with_sub_ein = Einsum.from_einsum_string(
    "ab,ac->c",
    sub_ein,
    Array.randn(a, c, name="ac"),
)
# here, we could sum out the 'b' dim in the above eab,ae->ab, but we retain it for equivalence with original node
unneeded_retain = nest_einsums(
    with_sub_ein,
    (2, (1, 0)),
)
assert_close(Matcher("sub").get_unique(unneeded_retain).evaluate(), sub_ein.evaluate())
opt.print(unneeded_retain)

# %%

# but if you'd like you can manually force these dims to be 'shrunk'
unneeded_retain_shrink = nest_einsums(
    with_sub_ein,
    (
        2,
        EinInfo((1, 0), shrink_out_axes=True, name="sub_new"),
    ),  # note that we had to specify the name again. The name would have been autorenamed if we didn't set this.
)
assert_close(Matcher("sub_new").get_unique(unneeded_retain_shrink).evaluate(), sub_ein.evaluate().sum(dim=1))
opt.print(unneeded_retain_shrink)

# %% [markdown]

# ## More tests

# From here on out, there are some more tests/demos. Feel free to ignore!


# %%


# now we can verify various changes have no effect
y = einsum_flatten(x)  # flattening has no effect
assert x == y
y = nest_einsums(x, (0, 1, 2, 3, 4, 5, 6)).normalize_ints()  # maybe this should normalize ints by default?
assert x == y

# %%


def print_multiline_escape(s: str):
    uuid = "6c6805a0-3aaf-492c-93a3-00851eacaacf"
    print(s.replace("\n", uuid).encode("unicode_escape").decode("ASCII").replace(uuid, "\n"))


z = nest_einsums(x, ((5, (1, 3)), 2, ((4,), 0), 6))
assert_close(x.evaluate(), z.evaluate())
print()
opt.print(z)
print()
print_multiline_escape(opt.repr(z))

s = """
abc * bcd * ae * bd * ec * acd * eaf Einsum acef,ad,abce,dab->acfb
  acd * bcd * bd Einsum acf,cef->acef
    acd [2,4,5] Array # \x1b[35m0\x1b[0m
    bcd * bd Einsum ecf,ef->cef
      bcd [3,4,5] Array # \x1b[35m1\x1b[0m
      bd [3,5] Array # \x1b[35m2\x1b[0m
  ae [2,7] Array # \x1b[35m3\x1b[0m
  ec * abc Einsum bc,aec->abce
    0 ec Einsum bc->bc
      1 ec [6,4] Array # \x1b[35m4\x1b[0m
    abc [2,3,4] Array # \x1b[35m5\x1b[0m
  eaf [7,2,6] Array # \x1b[35m6\x1b[0m""".strip(
    "\n"
)
assert opt.repr(z) == s


# %%

z = nest_einsums(x, (4, Rest()))
assert_close(x.evaluate(), z.evaluate())
print()
z.print()


# %%

z = nest_einsums(x, (4, (1, 2, Rest())))
assert_close(x.evaluate(), z.evaluate())
print()
z.print()


# %%

z = nest_einsums(x, Rest())
assert_close(x.evaluate(), z.evaluate())
print()
z.print()

# %%

z = nest_einsums(x, Rest(flat=True))
assert_close(x.evaluate(), z.evaluate())
print()
z.print()


# %%

z_no_flat = nest_einsums(x, (4, (1, 2, Rest(flat=True))))
assert_close(x.evaluate(), z_no_flat.evaluate())
print()
z_no_flat.print()

# %%

z = nest_einsums(x, (4, (1, 2, Rest(flat=True))))
assert_close(x.evaluate(), z.evaluate())
print()
z.print()

# %%

z_prev = nest_einsums(x, (4, (3, 2, 1, ((0, 5), 6))))
assert_close(x.evaluate(), z_prev.evaluate())
print()
z_prev.print()

# %%

z = nest_einsums(x, (4, (3, 2, 1, (EinInfo((0, 5), out_axes_perm=(2, 3, 1, 0)), 6))))
assert_close(x.evaluate(), z.evaluate())
print()
z.print()

# %%

sub_new = Einsum.from_einsum_string(
    "a b c, b c d, a e -> e",
    *(
        Array(torch.randn(*shape, dtype=torch.float64), name=name)
        for (shape, name) in [
            ((a, b, c), "abc"),
            ((b, c, d), "bcd"),
            ((a, e), "ae"),
        ]
    )
).normalize_ints()
sub_new.print()
sub = nest_einsums(sub_new, (0, (1, 2))).normalize_ints()
sub_fin = nest_einsums(sub, ((1, 2), 0))

# %%


# %%

z_with_names = nest_einsums(
    x, (4, EinInfo((3, EinInfo((2, 1), "other"), EinInfo(((0, 5), 6), "here")), "outer"))
).normalize_ints()
assert_close(x.evaluate(), z_with_names.evaluate())
print()
z_with_names.print()

z_here, z_other, z_outer = [Matcher(n).get_unique(z_with_names) for n in ["here", "other", "outer"]]

# %%


z_with_names.print()
einsum_flatten(z_with_names).all_input_axes()
z_keep_here = nest_einsums(z_with_names, (0, (1, 2, 3, (6, 4, 5))))
assert_close(x.evaluate(), z_keep_here.evaluate())
print()
z_keep_here.print()
assert_close(Matcher("here").get_unique(z_keep_here).evaluate(), z_here.evaluate())
assert_close(Matcher("outer").get_unique(z_keep_here).evaluate(), z_outer.evaluate())
assert Matcher("other").get_unique_op(z_keep_here) is None

# %%

z_keep_other = nest_einsums(z_with_names, (0, ((2, 3), (6, 4, 1, 5))))
assert_close(x.evaluate(), z_keep_other.evaluate())
print()
z_keep_other.print()
assert_close(Matcher("other").get_unique(z_keep_other).evaluate(), z_other.evaluate())
assert_close(Matcher("outer").get_unique(z_keep_other).evaluate(), z_outer.evaluate())
assert Matcher("here").get_unique_op(z_keep_other) is None

# %%

z_keep_just_here = nest_einsums(z_with_names, (0, (2, 1, 3), (6, 4, 5)))
assert_close(x.evaluate(), z_keep_just_here.evaluate())
print()
z_keep_just_here.print()
assert_close(Matcher("here").get_unique(z_keep_here).evaluate(), z_here.evaluate())
assert Matcher("other").get_unique_op(z_keep_just_here) is None
assert Matcher("outer").get_unique_op(z_keep_just_here) is None

# %%

NEW: NestEinsumsSpec = EinInfo((EinInfo((3, 2), "bd,ae"), 1), "bd,ae,bcd")

z = nest_einsums(x, (4, NEW, (EinInfo((0, 5), "abc,acd"), 6)))
assert_close(x.evaluate(), z_keep_just_here.evaluate())
print()
z_keep_just_here.print()

# %%

z_very_named = nest_einsums(
    x, (4, EinInfo((EinInfo((3, 2), "bd,ae"), 1), "bd,ae,bcd"), (EinInfo((0, 5), "abc,acd"), 6))
)
assert_close(x.evaluate(), z_very_named.evaluate())
print()
z_very_named.print()

# %%

z_very_named_rearrange = nest_einsums(z_very_named, (6, Rest(flat=True)))
assert_close(x.evaluate(), z_very_named_rearrange.evaluate())
print()
z_very_named_rearrange.print()

# %%

print()
term_early_out = nest_einsums(
    z_very_named,
    (3, (1, 4), Rest(flat=False)),
    traversal=new_traversal(term_early_at=Matcher.regex("^bd")),
)
assert_close(z_very_named.evaluate(), term_early_out.evaluate())
print()
term_early_out.print()

# %%

by_name = nest_einsums(x, ("eaf", ("ec", 2), Rest()))
assert_close(x.evaluate(), by_name.evaluate())
print()
by_name.print()

# %%

ignores_empty = nest_einsums(x, ("eaf", ("ec", 2), (), (), Rest()))
assert_close(x.evaluate(), ignores_empty.evaluate())
assert Matcher(lambda x: isinstance(x, Einsum) and len(x.args) == 0).get_unique_op(ignores_empty) is None
print()
ignores_empty.print()

# %%


fancy_permute = nest_einsums(x, ("eaf", ("ec", 2), EinInfo((3, 0), "hi"), Rest()))
assert_close(x.evaluate(), fancy_permute.evaluate())
print()
fancy_permute.print()

fancy_permute_2 = nest_einsums(x, ("eaf", ("ec", 2), EinInfo((3, 0), "hi", (1, 2, 0, 3)), Rest()))


assert_close(
    Matcher("hi").get_unique(fancy_permute).rearrange_str("a b c d -> b c a d").evaluate(),
    Matcher("hi").get_unique(fancy_permute_2).evaluate(),
)

# %%

empty_with_children = nest_einsums(
    Einsum.from_einsum_string(
        ",->",
        Einsum.from_einsum_string(
            "->",
        ),
        Einsum.from_einsum_string(
            "->",
        ),
    ),
    (),
)
empty_with_children.print()

# %%


sub_ein = Einsum.from_einsum_string(
    "eab,ae->ab", Array.randn(e, a, b, name="eab"), Array.randn(a, e, name="ae"), name="sub"
)
with_sub_ein = Einsum.from_einsum_string(
    "ab,ac->c",
    sub_ein,
    Array.randn(a, c, name="ac"),
)
unneeded_retain = nest_einsums(
    with_sub_ein,
    (2, (1, 0)),
)
assert_close(Matcher("sub").get_unique(unneeded_retain).evaluate(), sub_ein.evaluate())
unneeded_retain.print()

# %%

unneeded_retain_shrink = nest_einsums(
    with_sub_ein,
    (2, EinInfo((1, 0), shrink_out_axes=True, name="sub")),
)
assert_close(Matcher("sub").get_unique(unneeded_retain_shrink).evaluate(), sub_ein.evaluate().sum(dim=1))
unneeded_retain_shrink.print()

# %%

unneeded_retain_shrink_2 = nest_einsums(
    with_sub_ein,
    (2, EinInfo((1, 0), shrink_out_axes=True, name="sub")),
)
assert_close(Matcher("sub").get_unique(unneeded_retain_shrink_2).evaluate(), sub_ein.evaluate().sum(dim=1))
unneeded_retain_shrink_2.print()

# %%

z_new = nest_einsums(x, (("acd", ("bcd", Matcher.regex("^bd$"))), ("ae", "ec"), Rest()))
z_new.print()
sub_match_nest = nest_einsums(
    z_new,
    (4, 3, 0, 1, 2),
    IterativeMatcher.new_func(
        lambda _: IterateMatchResults(
            [new_traversal(term_early_at=Matcher.regex(r"^\w+ \* \w+$")), FINISHED, True],
            True,
        )
    ),
)
assert sub_match_nest == nest_einsums(
    z_new, ("eaf", "abc", "acd", NestMatcher({"bcd", "bd"}, flat=True), NestMatcher({"ae", "ec"}, flat=True))
)
