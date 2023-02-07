# %%
from pdb import post_mortem

import torch

from rust_circuit import (
    Add,
    Array,
    Einsum,
    Index,
    Rearrange,
    RearrangeSpec,
    Scalar,
    Symbol,
    add_collapse_scalar_inputs,
    add_deduplicate,
    add_flatten_once,
    add_pull_removable_axes,
    distribute_all,
    einsum_flatten_once,
    einsum_pull_removable_axes,
    remove_add_few_input,
)

# %%

base_add = Add(
    Scalar(0.2, name="hi"),
    Scalar(0.2, name="hi"),
)

nested_add = Add(base_add, base_add)
nested_add.print()
flattened = add_flatten_once(nested_add)
assert flattened is not None
flattened.print()

deduped = add_deduplicate(flattened)
assert deduped is not None
deduped.print()
deduped_2 = add_deduplicate(base_add)
assert deduped_2 is not None
deduped_2.print()

elimed = remove_add_few_input(deduped)
assert elimed is not None
elimed.print()

scalar_merged = add_collapse_scalar_inputs(
    Add(
        Scalar(171),
        Scalar(9),
    )
)
assert scalar_merged is not None
scalar_merged.print()

scalar_merged_call = add_collapse_scalar_inputs(
    Add(
        Scalar(171),
        Scalar(9),
    )
)
assert scalar_merged_call is not None
scalar_merged_call.print()

add_with_rearrange = Add(
    Rearrange(Scalar(2), RearrangeSpec([], [[0], [1]], [2, 3])),
    Rearrange(Scalar(2, (2,)), RearrangeSpec([[0]], [[0], [1]], [2, 3])),
)
post_rearrange = add_pull_removable_axes(add_with_rearrange, True)
assert post_rearrange is not None
post_rearrange.print()
# %%

ein = Einsum((Scalar(2, (2, 3)), (0, 1)), (Scalar(3, (3, 4)), (1, 2)), out_axes=(0, 2))
ein.print()
ein_deep = Einsum((ein, (0, 1)), (ein, (0, 1)), out_axes=(0,))
ein_deep.print()
ein_flat = einsum_flatten_once(ein_deep)
assert ein_flat is not None
ein_flat.print()

# %%

for_distribute = Einsum((base_add, ()), (base_add, ()), out_axes=())

distributed = distribute_all(for_distribute)
assert distributed is not None
distributed.print()

for_pull = Einsum(
    (Rearrange(Scalar(2), RearrangeSpec([], [[0], [1]], [2, 3])), (0, 1)),
    (Rearrange(Scalar(2, (2,)), RearrangeSpec([[0]], [[0], [1]], [2, 3])), (0, 1)),
    out_axes=(0, 1),
)
pulled = einsum_pull_removable_axes(for_pull)
assert pulled is not None
pulled.print()

rearrange_identity = Rearrange(Scalar(2, (2,)), RearrangeSpec([[0]], [[0]], [2]))
rearrange_identity_2 = Rearrange(Scalar(2, (1, 1)), RearrangeSpec([[0], [1]], [[1], [0]], [1, 1]))
print(rearrange_identity.spec.is_identity())
print(rearrange_identity.spec.is_identity())
print(rearrange_identity_2.spec.is_identity())
print(rearrange_identity_2.spec.is_identity())

# %%

r1 = Rearrange(Array(torch.randn(2, 15)), RearrangeSpec([[0], [1]], [[0, 1]], [2, 15]))
r2 = Rearrange(r1, RearrangeSpec([[0, 1]], [[0], [1]], [10, 3]))
r1.node

# %%
print(RearrangeSpec([[0], [1]], [[1], [0]], [1, 1]).canonicalize(True))

print(Einsum(out_axes=(), name=None) == Add(name=None))
