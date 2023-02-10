from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional, Tuple

from dataclasses import dataclass

import torch

import rust_circuit.index_util.internal.resolve as resolve

#
# Planning
#


@dataclass
class SizeVar:
    arr_number: int
    axis: int


@dataclass
class SizeEqConstraint:
    lhs: SizeVar
    rhs: SizeVar


@dataclass
class ShapesPlan:
    arr_ranks: list[int]
    axis_sizes: list[Optional[SizeVar]]
    constraints: list[SizeEqConstraint]


def plan_shapes(num_axes: int, array_indices: list[list[resolve.IdxExpr]]) -> ShapesPlan:
    arr_ranks = [len(indices) for indices in array_indices]

    axis_sizes: list[Optional[SizeVar]] = [None] * num_axes
    constraints: list[SizeEqConstraint] = []

    for arr_number, indices in enumerate(array_indices):
        for i, idx in enumerate(indices):
            if not isinstance(idx, resolve.OutAxisIdx):
                continue
            axis = idx.axis
            prev = axis_sizes[axis]
            if prev is None:
                axis_sizes[axis] = SizeVar(arr_number, i)
            else:
                constraints.append(SizeEqConstraint(prev, SizeVar(arr_number, i)))

    return ShapesPlan(arr_ranks=arr_ranks, axis_sizes=axis_sizes, constraints=constraints)


#
# Execution
#


@dataclass
class RuntimeShapes:
    batch_shape: list[int]
    axis_sizes: list[int]


def execute_shapes_plan(
    plan: ShapesPlan, arrays: list[Tuple[str, list[int]]], user_axis_sizes: Optional[list[Optional[int]]] = None
) -> RuntimeShapes:
    assert len(arrays) == len(plan.arr_ranks)

    opt_axis_sizes: list[Optional[int]]
    if user_axis_sizes is None:
        opt_axis_sizes = [None] * len(plan.axis_sizes)
    else:
        opt_axis_sizes = user_axis_sizes

    if len(opt_axis_sizes) < len(plan.axis_sizes):
        raise ValueError(f"Expected at least {len(plan.axis_sizes)} axis sizes, got {len(opt_axis_sizes)}")

    axis_sizes_batch_rank = len(opt_axis_sizes) - len(plan.axis_sizes)

    batch_shapes: list[list[int]] = []
    batch_shapes.append([size if size is not None else 1 for size in opt_axis_sizes[:axis_sizes_batch_rank]])
    for arr_rank, (name, shape) in zip(plan.arr_ranks, arrays):
        if len(shape) < arr_rank:
            raise ValueError(f"Expected array {name!r} to have rank at least {arr_rank}; got shape {shape}")
        batch_shapes.append(shape[: len(shape) - arr_rank])

    try:
        batch_shape = list(torch.broadcast_shapes(*batch_shapes))
        assert all(x == batch_shape for x in batch_shapes[1:])
    except RuntimeError as e:
        raise ValueError(f"Batch dimensions are incompatible")

    def get_size_var(size_var: SizeVar) -> int:
        arr_shape = arrays[size_var.arr_number][1]
        arr_batch_rank = len(arr_shape) - plan.arr_ranks[size_var.arr_number]
        return arr_shape[arr_batch_rank + size_var.axis]

    axis_sizes: list[int] = []
    for i, size_var in enumerate(plan.axis_sizes):
        if size_var is None:
            size = opt_axis_sizes[axis_sizes_batch_rank + i]
            if size is None:
                raise ValueError(f"Could not infer size for axis {i - len(plan.axis_sizes)}")
            axis_sizes.append(size)
        else:
            axis_sizes.append(get_size_var(size_var))

    for constraint in plan.constraints:
        lhs_size = get_size_var(constraint.lhs)
        rhs_size = get_size_var(constraint.rhs)
        if lhs_size != rhs_size:
            lhs_num = constraint.lhs.arr_number
            rhs_num = constraint.rhs.arr_number
            lhs_name = arrays[lhs_num][0]
            rhs_name = arrays[rhs_num][0]
            lhs_rank = plan.arr_ranks[lhs_num]
            rhs_rank = plan.arr_ranks[rhs_num]
            lhs_expr = f"{lhs_name}.shape[{constraint.lhs.axis - lhs_rank}]"
            rhs_expr = f"{rhs_name}.shape[{constraint.rhs.axis - rhs_rank}]"
            raise ValueError(
                f"Incompatible shapes: "
                + f"expected {lhs_expr} == {rhs_expr}; "
                + f"got {lhs_expr} = {lhs_size}, {rhs_expr} = {rhs_size}"
            )

    return RuntimeShapes(batch_shape=batch_shape, axis_sizes=axis_sizes)
