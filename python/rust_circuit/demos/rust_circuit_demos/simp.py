from rust_circuit import Array, Einsum, Rearrange, SimpFnSubset, simp

# %%

fin = Rearrange.from_string(Einsum.from_einsum_string("abc->cab", Array.randn(2, 3, 4, name="hi")), "c a b -> a c b")
print()
fin.print()

# %%

print()
simp(fin).print()

# %%

print()
SimpFnSubset.compiler_default().simp(fin).print()

# %%

print()
# same as just `simp`
SimpFnSubset.default().simp(fin).print()

# %%

print()
# disable a function
SimpFnSubset.default().set(rearrange_fuse=False).simp(fin).print()

# %%

# fancy repr
assert str(SimpFnSubset.default()) == "SimpFnSubset.default()"
print(str(SimpFnSubset.default().set(rearrange_fuse=False)))
assert (
    str(SimpFnSubset.default().set(rearrange_fuse=False))
    == """SimpFnSubset.default().set(
    # Rearrange
    rearrange_fuse = False,
)"""
)
print(str(SimpFnSubset.default().set(rearrange_fuse=False, einsum_concat_to_add=True)))
assert (
    str(SimpFnSubset.default().set(rearrange_fuse=False, einsum_concat_to_add=True))
    == """SimpFnSubset.default().set(
    # Einsum
    einsum_concat_to_add = True,
    # Rearrange
    rearrange_fuse = False,
)"""
)

# %%

# TODO: add support for partial simp with traversal and demo this!
