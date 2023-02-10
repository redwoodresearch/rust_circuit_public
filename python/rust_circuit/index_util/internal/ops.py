from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal, Optional, Tuple, Sequence, TypeVar, Callable
    from _typeshed import SupportsRichComparisonT
    from rust_circuit.index_util.internal.resolve import IdxExpr

import dataclasses
from dataclasses import dataclass

import torch

import rust_circuit.index_util.internal.infer_shapes as infer_shapes
import rust_circuit.index_util.internal.resolve as resolve
import rust_circuit.index_util.internal.view as view

#
# Planning
#


@dataclass
class OpAxisPlan:
    add_synthetic_axis: bool
    op_axis: int


def plan_op_axis(
    out_rank: int, op_arr_indices: list[IdxExpr], pos_arr_indices: list[IdxExpr]
) -> Tuple[OpAxisPlan, int, list[IdxExpr]]:
    op_idx_set = {idx.axis for idx in op_arr_indices if isinstance(idx, resolve.OutAxisIdx)}
    pos_idx_set = {idx.axis for idx in pos_arr_indices if isinstance(idx, resolve.OutAxisIdx)}

    candidates = pos_idx_set - op_idx_set
    if len(candidates) > 0:
        op_axis = min(candidates)
        return OpAxisPlan(add_synthetic_axis=False, op_axis=op_axis), out_rank, pos_arr_indices

    new_axis = out_rank

    return (
        OpAxisPlan(add_synthetic_axis=True, op_axis=new_axis),
        out_rank + 1,
        pos_arr_indices + [resolve.OutAxisIdx(new_axis)],
    )


@dataclass
class GatherPlan:
    shapes: infer_shapes.ShapesPlan
    axis: OpAxisPlan
    src_view: view.ViewPlan
    pos_view: view.ViewPlan


def plan_gather(spec: resolve.GatherSpec) -> GatherPlan:
    shapes = infer_shapes.plan_shapes(spec.out_rank, [spec.src_indices, spec.pos_indices])

    axis, inner_out_rank, inner_pos_indices = plan_op_axis(spec.out_rank, spec.src_indices, spec.pos_indices)

    src_view_indices: list[view.ViewIdxExpr] = []
    for idx in spec.src_indices:
        if isinstance(idx, resolve.PosDataIdx):
            src_view_indices.append(resolve.OutAxisIdx(axis.op_axis))
        else:
            src_view_indices.append(idx)

    pos_view_indices: list[view.ViewIdxExpr] = []
    for idx in inner_pos_indices:
        assert not isinstance(idx, resolve.PosDataIdx)
        pos_view_indices.append(idx)

    src_view = view.plan_view(inner_out_rank, src_view_indices)
    pos_view = view.plan_view(inner_out_rank, pos_view_indices)

    return GatherPlan(shapes, axis, src_view, pos_view)


@dataclass
class ScatterPlan:
    shapes: infer_shapes.ShapesPlan
    axis: OpAxisPlan
    dst_view: view.ViewPlan
    src_view: view.ViewPlan
    pos_view: view.ViewPlan
    reduce: Optional[Literal["add", "multiply"]]


def plan_scatter(spec: resolve.ScatterSpec) -> ScatterPlan:
    shapes = infer_shapes.plan_shapes(
        len(spec.axis_names),
        [spec.dst_indices, spec.src_indices, spec.pos_indices],
    )

    axis, inner_out_rank, inner_pos_indices = plan_op_axis(len(spec.axis_names), spec.dst_indices, spec.pos_indices)

    dst_view_indices: list[view.ViewIdxExpr] = []
    for idx in spec.dst_indices:
        if isinstance(idx, resolve.PosDataIdx):
            dst_view_indices.append(resolve.OutAxisIdx(axis.op_axis))
        else:
            dst_view_indices.append(idx)

    src_view_indices: list[view.ViewIdxExpr] = []
    for idx in spec.src_indices:
        assert not isinstance(idx, resolve.PosDataIdx)
        src_view_indices.append(idx)
    if axis.add_synthetic_axis:
        src_view_indices.append(resolve.OutAxisIdx(axis.op_axis))

    pos_view_indices: list[view.ViewIdxExpr] = []
    for idx in inner_pos_indices:
        assert not isinstance(idx, resolve.PosDataIdx)
        pos_view_indices.append(idx)

    dst_view = view.plan_view(inner_out_rank, dst_view_indices)
    src_view = view.plan_view(inner_out_rank, src_view_indices)
    pos_view = view.plan_view(inner_out_rank, pos_view_indices)

    return ScatterPlan(shapes, axis, dst_view, src_view, pos_view, spec.reduce)


#
# Execution
#


def infer_gather_shapes(
    plan: GatherPlan,
    out_shape: Optional[list[Optional[int]]],
    src_shape: list[int],
    pos_shape: list[int],
    src_name: str,
    pos_name: str,
) -> infer_shapes.RuntimeShapes:
    return infer_shapes.execute_shapes_plan(
        plan.shapes,
        [(src_name, src_shape), (pos_name, pos_shape)],
        out_shape,
    )


def execute_gather(
    plan: GatherPlan,
    out_shape: Optional[list[Optional[int]]],
    src: torch.Tensor,
    pos: torch.Tensor,
    src_name: str,
    pos_name: str,
) -> torch.Tensor:
    shapes = infer_gather_shapes(plan, out_shape, list(src.shape), list(pos.shape), src_name, pos_name)

    if plan.axis.add_synthetic_axis:
        inner_axis_sizes = shapes.axis_sizes + [1]
        inner_pos = pos.unsqueeze(-1)
    else:
        inner_axis_sizes = shapes.axis_sizes
        inner_pos = pos

    src_view_shape: list[Optional[int]] = list(inner_axis_sizes)
    src_view_shape[plan.axis.op_axis] = None
    src_view = view.execute_plan(plan.src_view, src_view_shape, src)
    pos_view = view.execute_plan(plan.pos_view, inner_axis_sizes, inner_pos)
    src_view = src_view.expand(shapes.batch_shape + list(src_view.shape[src_view.ndim - len(inner_axis_sizes) :]))
    pos_view = pos_view.expand(shapes.batch_shape + inner_axis_sizes)

    inner_gather = torch.gather(src_view, len(shapes.batch_shape) + plan.axis.op_axis, pos_view)

    if plan.axis.add_synthetic_axis:
        return inner_gather.squeeze(-1)
    else:
        return inner_gather


def infer_scatter_shapes(
    plan: ScatterPlan,
    dst_shape: list[int],
    src_shape: list[int],
    pos_shape: list[int],
    dst_name: str,
    src_name: str,
    pos_name: str,
) -> Tuple[infer_shapes.RuntimeShapes, list[int]]:
    shapes = infer_shapes.execute_shapes_plan(
        plan.shapes,
        [(dst_name, dst_shape), (src_name, src_shape), (pos_name, pos_shape)],
    )

    out_shape = torch.broadcast_shapes(shapes.batch_shape + [1] * plan.shapes.arr_ranks[0], dst_shape)

    return shapes, list(out_shape)


def execute_scatter(
    plan: ScatterPlan,
    dst: torch.Tensor,
    src: torch.Tensor,
    pos: torch.Tensor,
    dst_name: str,
    src_name: str,
    pos_name: str,
) -> torch.Tensor:
    shapes = infer_scatter_shapes(
        plan, list(dst.shape), list(src.shape), list(pos.shape), dst_name, src_name, pos_name
    )[0]

    if plan.axis.add_synthetic_axis:
        inner_axis_sizes = shapes.axis_sizes + [1]
        inner_src = src.unsqueeze(-1)
        inner_pos = pos.unsqueeze(-1)
    else:
        inner_axis_sizes = shapes.axis_sizes
        inner_src = src
        inner_pos = pos

    dst = dst.clone()
    dst_view_shape: list[Optional[int]] = list(inner_axis_sizes)
    dst_view_shape[plan.axis.op_axis] = None
    dst_view = view.execute_plan(plan.dst_view, dst_view_shape, dst)
    src_view = view.execute_plan(plan.src_view, inner_axis_sizes, inner_src)
    pos_view = view.execute_plan(plan.pos_view, inner_axis_sizes, inner_pos)
    dst_view = dst_view.expand(shapes.batch_shape + list(dst_view.shape[dst_view.ndim - len(inner_axis_sizes) :]))
    src_view = src_view.expand(shapes.batch_shape + inner_axis_sizes)
    pos_view = pos_view.expand(shapes.batch_shape + inner_axis_sizes)

    dst_view.scatter_(
        len(shapes.batch_shape) + plan.axis.op_axis,
        pos_view,
        src_view,
        **({"reduce": plan.reduce} if plan.reduce is not None else {}),
    )

    return dst
