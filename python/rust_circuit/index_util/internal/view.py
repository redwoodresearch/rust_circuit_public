from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Sequence, Tuple, Optional, Union

from dataclasses import dataclass

import torch

from rust_circuit.index_util.internal.resolve import IntLitIdx, OutAxisIdx

if TYPE_CHECKING:
    ViewIdxExpr = Union[OutAxisIdx, IntLitIdx]

#
# Planning
#


@dataclass
class ConstantIndexPlan:
    constants: list[Optional[int]]


def plan_constant_index(indices: Sequence[ViewIdxExpr]) -> Tuple[ConstantIndexPlan, list[OutAxisIdx]]:
    constants: list[Optional[int]] = []
    new_indices: list[OutAxisIdx] = []
    for idx in indices:
        if isinstance(idx, IntLitIdx):
            constants.append(idx.value)
        else:
            constants.append(None)
            new_indices.append(idx)
    return ConstantIndexPlan(constants), new_indices


@dataclass
class DiagonalPlan:
    """
    Each tuple in 'diagonals' has the semantics of replacing the first axis in the tuple with the diagonal along both
    axes.

    Note that this is not the same as the semantics of 'torch.diagonal', which places the diagonal at the end.

    We implement the semantics here in terms of PyTorch's primitives by using 'torch.diagonal' to get the diagonal, and
    then using 'torch.transpose' to move the diagonal to the position of the first axis.
    """

    diagonals: list[Tuple[int, int]]


def plan_diagonals(indices: Sequence[OutAxisIdx]) -> Tuple[DiagonalPlan, list[OutAxisIdx]]:
    diagonals: list[Tuple[int, int]] = []
    first_seen: dict[int, int] = {}
    new_indices: list[OutAxisIdx] = []
    for i, idx in enumerate(indices):
        prev = first_seen.get(idx.axis)
        if prev is not None:
            diagonals.append((prev, i))
        else:
            first_seen[idx.axis] = i
            new_indices.append(idx)
    return DiagonalPlan(diagonals), new_indices


@dataclass
class TransposePlan:
    perm: list[int]


def plan_transpose(indices: Sequence[OutAxisIdx]) -> Tuple[TransposePlan, list[OutAxisIdx]]:
    perm = sorted(range(len(indices)), key=lambda i: indices[i].axis)
    return TransposePlan(perm), [indices[i] for i in perm]


@dataclass
class BroadcastPlan:
    """
    If 'broadcast_axes[i]' is not 'None', then a new axis should be inserted at position 'i', and the new axis should
    be broadcast to the same size as the axis 'broadcast_axes[i]' in the output.
    """

    broadcast_axes: list[Optional[int]]


def plan_broadcast(out_rank: int, indices: Sequence[OutAxisIdx]) -> Tuple[BroadcastPlan, list[OutAxisIdx]]:
    broadcast_axes: list[Optional[int]] = []
    new_indices: list[OutAxisIdx] = []
    j = 0
    for i in range(out_rank):
        while j < len(indices):
            idx = indices[j]
            if idx.axis == i:
                broadcast_axes.append(None)
                new_indices.append(idx)
                j += 1
                break
            else:
                assert idx.axis > i, "indices must be sorted by prior transpose step"
                broadcast_axes.append(i)
                new_indices.append(OutAxisIdx(i))
                break
        else:
            broadcast_axes.append(i)
            new_indices.append(OutAxisIdx(i))

    return BroadcastPlan(broadcast_axes), new_indices


@dataclass
class ViewPlan:
    input_rank: int
    constant_index: ConstantIndexPlan
    diagonal: DiagonalPlan
    transpose: TransposePlan
    broadcast: BroadcastPlan


def plan_view(out_rank: int, indices: Sequence[ViewIdxExpr]) -> ViewPlan:
    """
    Given an indexing expression

        arr[e_0, e_1, ..., e_{k-1}]

    expressed in terms of iteration variables ("out axes")

        i_0, i_1, ..., i_{n-1}

    returns a "plan" representing a function 'f' such that

        f(arr)[i_0, i_1, ..., i_{n-1}] == arr[e_0, e_1, ..., e_{k-1}]
    """

    rank = len(indices)
    constant_index, indices = plan_constant_index(indices)
    diagonal, indices = plan_diagonals(indices)
    transpose, indices = plan_transpose(indices)
    broadcast, indices = plan_broadcast(out_rank, indices)
    assert indices == [OutAxisIdx(i) for i in range(out_rank)]
    return ViewPlan(rank, constant_index, diagonal, transpose, broadcast)


#
# Execution
#


def execute_plan(plan: ViewPlan, out_shape: Sequence[Optional[int]], arr: torch.Tensor) -> torch.Tensor:
    batch_rank = arr.ndim - plan.input_rank
    assert batch_rank >= 0

    # Constant index
    constant_indices = [slice(None)] * batch_rank + [
        slice(None) if value is None else value for value in plan.constant_index.constants
    ]
    arr = arr[constant_indices]

    # Diagonals
    for axis1, axis2 in plan.diagonal.diagonals:
        arr = arr.diagonal(dim1=batch_rank + axis1, dim2=batch_rank + axis2)
        new_rank = arr.ndim
        perm = list(range(new_rank - 1))
        perm.insert(batch_rank + axis1, new_rank - 1)
        arr = arr.permute(perm)

    # Transpose
    arr = arr.permute(list(range(batch_rank)) + [batch_rank + i for i in plan.transpose.perm])

    # Broadcast
    assert len(out_shape) == len(plan.broadcast.broadcast_axes), "out_shape should not have batch dimensions"
    full_out_shape: list[int] = []
    for i, axis in enumerate(plan.broadcast.broadcast_axes):
        if axis is None:
            size = out_shape[i]
            assert size is None or size == arr.shape[batch_rank + i]
            full_out_shape.append(arr.shape[batch_rank + i])
        else:
            arr = arr.unsqueeze(batch_rank + i)
            size = out_shape[i]
            assert size is not None
            full_out_shape.append(size)
    arr = arr.expand(list(arr.shape[: arr.ndim - len(full_out_shape)]) + full_out_shape)

    return arr
