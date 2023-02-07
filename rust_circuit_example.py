# %%
import uuid

import torch

from rust_circuit import Add, Array, Concat, Einsum, Index, Rearrange, RearrangeSpec, Scalar, Symbol

# %%
e = Einsum.from_einsum_string("a->", Scalar(0.2, (2,)))
e.print()

x = Einsum(
    (Scalar(0.2), ()),
    (Symbol((3, 4, 5), uuid.uuid4()), (0, 1, 2)),
    (Array(torch.randn(3, 4, 5)), (0, 1, 2)),
    out_axes=(2, 0),
    name="hi",
)
print(x.out_axes)

assert x == x
x

# %%

x.args

# %%

assert isinstance(x.children[0], Scalar)
assert isinstance(x.children[1], Symbol)
assert isinstance(x.children[2], Array)

# %%

assert Scalar(0.2) == Scalar(0.2)
assert hash(Scalar(0.2)) == hash(Scalar(0.2))
assert Scalar(0.2) != Scalar(0.20001)
assert hash(Scalar(0.2)) != hash(Scalar(0.20001))
assert x == Einsum(*x.args, out_axes=x.out_axes, name=x.name)
assert x != Einsum(*x.args, out_axes=x.out_axes, name="different")

# %%

Array(torch.randn(3, 4, 5)).value

# %%

try:
    Array(8)  # type: ignore
except TypeError as e:
    print(e)

# %%

Symbol((3,), uuid.uuid4(), name="hi").name

# %%

try:
    Symbol((3,), uuid.uuid4().bytes, name="hi")  # type: ignore
except TypeError as e:
    print(e)

# %%

y = Add(
    Scalar(0.2),
    Symbol((3, 1, 5), uuid.uuid4()),
    Array(torch.randn(3, 4, 1)),
)
print(y.children)
print(y)

# %%

Scalar(0.5, (3, 5), name="new_scale").shape

# %%

Scalar(0.5, (3, 5), name="new_scale").is_one()

# %%

Scalar(1.0, (3, 5), name="new_scale").is_one()

# %%
assert (
    RearrangeSpec.fuse(RearrangeSpec([[0], [1]], [[0, 1]], [2, 15]), RearrangeSpec([[0, 1]], [[0], [1]], [10, 3]))
    is not None
)
RearrangeSpec.fuse(
    RearrangeSpec([[0], [1]], [[0], [1]], [2, 1]),
    RearrangeSpec(
        [[0], [1]],
        [[1], [0]],
        [2, 1],
    ),
)

RearrangeSpec.canonicalize(RearrangeSpec([[0, 1], [2, 3]], [[0, 1, 2, 3]], [3, 1, 1, 1]))


# %%
i = Index(Array(torch.randn(2, 3)), (0, 1))
assert i != Index(Array(torch.randn(2, 3)), (0, 1))
assert i == Index(i.node, (0, 1))
assert Index(i.node, (1, 0)) != Index(i.node, (0, 1))
idx = (torch.randint(0, 10, (20,)), 0)
tensor_indexed = Index(i.node, idx)
assert Index(i.node, idx) != Index(i.node, (0, 1))
assert Index(i.node, (slice(0, 1),)) == Index(i.node, (slice(0, 1),))
assert Index(i.node, (slice(None, 1),)) == Index(i.node, (slice(None, 1),))
assert Index(i.node, (slice(1, None),)) == Index(i.node, (slice(1, None),))

# assert Index(tensor_indexed.node,tensor_indexed.index)==tensor_indexed # this is failing atm bc PyObject isnt stable reference to tensor?
print(i)
print(i.index)

# %%

v1 = Concat(Array(torch.randn(2, 3)), Array(torch.randn(3, 3)), axis=0)
v2 = Concat(*v1.children, axis=0)
assert v1.shape == (5, 3)
assert v1 == v2

v1 = Concat(Array(torch.randn(3, 3)), Array(torch.randn(3, 3)), axis=0)
v2 = Concat(*v1.children, axis=1)
assert v1 != v2


# %%
