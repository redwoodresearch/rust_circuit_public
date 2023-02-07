# %%
import torch

import rust_circuit as rc

# %%

# We can print circuits, like so:

circuit = rc.Einsum.from_einsum_string(
    "ab,bc->aa",
    rc.Scalar(1.0, (2, 3), "scalar1"),
    rc.Rearrange.from_string(rc.Scalar(2.0, (4, 3), "scalar2"), "a b -> b a"),
    name="einy",
)
circuit.print()

# repr is same as printing, just to string
assert (
    circuit.repr()
    == """
'einy' Einsum ab,bc->aa
  'scalar1' [2,3] Scalar 1
  'scalar2 rearrange' Rearrange a b -> b a
    'scalar2' [4,3] Scalar 2
""".strip()
)

# python's print(circuit) by default only prints depth 2 (root and children, no grandchildren), and wraps it in <> so you it works well with python datastructures
print((circuit, 1))
# %%

# Or use options

rc.PrintOptions(bijection=True).print(circuit)

# if a subcircuit appears multiple times,
# subsequent appearances will be replaced with just the name or serial number
assert (
    rc.Add(rc.Scalar(1.0, (2,)), rc.Scalar(1.0, (2,))).repr()
    == """
0 Add
  1 [2] Scalar 1
  1 Scalar""".strip()
)
assert (
    rc.Add(rc.Scalar(1.0, (2,), "scaley"), rc.Scalar(1.0, (2,), "scaley"), name="addey").repr()
    == """
\'addey' Add
  'scaley' [2] Scalar 1
  'scaley'""".strip()
)

# %%

# if bijection=True (which is the default!), you can parse printed circuits back into circuits again! This is lossless, except for custom GeneralFunctions

P = rc.Parser()
parsed = P(rc.PrintOptions(bijection=True).repr(circuit))
assert parsed == circuit

# You can also write circuits yourself, and store them in files called .circ
P(
    """
'mycircuit' Einsum a,ab->b
  'sym1' [4] Symbol 
  'sym2' [4,5] Symbol 
"""
)

# (indentation width is 2 spaces, not configurable right now)

# you can use parse_circuits to parse a file containing multiple circuits
# you can use this with names/serial numbers to factor out parts of your circuit
# parsing also supports comments with #


_, factored_circuit = P.parse_circuits(
    """
# the submodule, declared first
'submod1' [2] Einsum a,a->a
  'inp' [2] Scalar 1
  'inp'
'main_circuit' Einsum a,aa->
  'submod1' # the reference to the submodule
  'traced' [2,2] Scalar 2
"""
)
factored_circuit.print()

# %%

# rust_circuit hashes all tensors (contents and type info) on intake
# Tensors are stored by hash, locally (by default in ~/tensors_by_hash_cache automatically) and in rrfs/tensor_db
# in a fixed file trie.

# when you print a circuit with bijection=True, if any of its tensors that aren't saved locally will be saved
# later you can sync your local tensors to rrfs with rc.sync_all_unsynced_tensors
# or you can use rc.PrintOptions(sync_tensors=True) to print and push tensors to rrfs immediately (slow).
# There's also sync_specific_tensors(...) which takes a list of hash prefix strings

demo_array = rc.Array(torch.zeros(2, 3))
demo_array.print()
circuit_string_to_share = rc.PrintOptions(bijection=True, sync_tensors=True).repr(
    P(
        """
0 Add
  1 [2,3] Array 2f97b0bb06d4adba46cd8756
  1
  """
    )
)
print(circuit_string_to_share)

# Here are the formats for all the node info types
# shapes are only provided when needed. Only Scalar, Array, Symbol,  SetSymbolicShape, Scatter need shapes
P.parse_circuits(
    """
'_' [] Scalar 1 # any number
'_1' [2] Symbol 93aa0db1-2435-4fa6-8675-711543cab705 # no uuid means null uuid
'_2' [2,3] Array 2f97b0bb06d4adba46cd8756 # prefix of hash of tensor
'einsum' Einsum a,ab,->b
  '_1'
  '_2'
  '_'
'einsum fancy string' Einsum fancy: ay, ay bee,->bee
  '_1'
  '_2'
  '_'
'add' Add
  '_1'
  '_'
'rearrange' Rearrange a b -> b a ()
  '_2'
'index' Index [0:1,1]
  '_2'
'cumulant' Cumulant
  '_'
  '_1'
"""
)

# %%
# Options rundown:

rc.PrintOptions(arrows=True).print(circuit)
# Parsing treats all characters used in arrows as whitespace, so you can parse with arrows=True (but not control nesting with arrows, that's indentation)

# colorers! you can supply a function, or use a built in one
rc.PrintOptions(colorer=rc.PrintOptions.hash_colorer()).print(circuit)
rc.PrintOptions(colorer=rc.PrintOptions.type_colorer()).print(circuit)

# commenters is a list of functions that generate comment strings, put after the node in the same line
rc.PrintOptions(commenters=[lambda x: str(rc.count_nodes(x))]).print(circuit)

# this prints child argnames (such as einsum axes) as comments on children
rc.PrintOptions(comment_arg_names=True).print(
    P(
        """
0 DiscreteVar
  1 Einsum ab,b->ab
    's1' [2,3] Symbol
    's2' [3] Symbol
  's3' [2] Symbol
"""
    )
)


# %%

# You can terminate printing early (avoid printing children) with a traversal. (this is incompatible with bijeciton=True)

deepcirc = P(
    """
0 Rearrange ->
  1 Rearrange ->
    2 Rearrange ->
      'sym' [] Symbol
"""
)
rc.PrintOptions(bijection=False, traversal=rc.new_traversal(start_depth=0, end_depth=1)).print(deepcirc)
rc.PrintOptions(bijection=False, traversal=rc.new_traversal(start_depth=0, end_depth=2)).print(deepcirc)

# Diff print

orig_circuit = P(
    """
'a0' Add
  'a1' Add
    'sym1' [2] Symbol
    'sym2' [2] Symbol
  'a2' Add
    'sym1'
    'sym2'
    'sym2'
"""
)
rewritten_circuit = orig_circuit.update("sym2", lambda _: P("'sym3' [2] Symbol"))
rewritten_circuit.print()
print(rc.diff_circuits(rewritten_circuit, orig_circuit, rc.PrintOptions()))

rewritten_circuit = orig_circuit.update("sym2", lambda _: P("'sc0' [2] Scalar 1"))
rewritten_circuit.print()
print(rc.diff_circuits(rewritten_circuit, orig_circuit, rc.PrintOptions()))

# %%
