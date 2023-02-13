from __future__ import annotations

import math
import os
from copy import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, TypeVar, Union
from warnings import warn

import torch

import rust_circuit.optional as op
from rust_circuit.algebric_rewrite import check_permutation
from rust_circuit.py_utils import assert_never, make_index_at
from rust_circuit.ui.very_named_tensor import VeryNamedTensor, ViewSpec, ViewSpecIdx

from .._rust import Circuit, Index, Shape, TorchAxisIndex

EvalCallback = Callable[[Circuit, ViewSpec], Union[Tuple[torch.Tensor, Set[int], Set[int]], torch.Tensor]]
""" returns evaluated tensor, handled view items, removed dim items """

T = TypeVar("T")


@dataclass
class CircuitsVeryNamedTensor:
    def __init__(
        self,
        circuit: Circuit,
        eval_callback: EvalCallback,
        dim_names: Sequence[str],
        dim_types: Sequence[str],
        dim_idx_names: Sequence[Sequence[str]],
        units: str = "units",
        title: str = "untitled_tensor",
        aux_info: Any = {},
        selected_options=None,
        default_axis_view: ViewSpecIdx = "axis",
        default_view_spec: Optional[ViewSpec] = None,  # overrides axis
        extra_shape: Shape = (),
        skip_last_dims: int = 0,
        index_device: Union[str, torch.device] = "cpu",
        caching: bool = True,
        caching_numel: int = 1024 ** 3 // 16,
        caching_perm: Optional[List[Union[int, str]]] = None,
        custom_can_be_canon_view: Optional[Callable[[Tuple[int, ...], ViewSpec, ViewSpec], bool]] = None,
        always_allow_cache_on_numel: bool = True,
        prefer_index_axis_caching: bool = True,
    ):
        self.circuit = circuit

        self.eval_callback = eval_callback

        self.dim_names = dim_names
        self.dim_types = dim_types
        self.dim_idx_names = [[str(i) for i in x] for x in dim_idx_names]
        self.units: str = units
        self.title = title

        self.aux_info = aux_info
        self.selected_options = selected_options
        via_axis: ViewSpec = [default_axis_view] * len(self.dim_names)
        self.default_view_spec: ViewSpec = op.unwrap_or(default_view_spec, via_axis)

        self.extra_shape = extra_shape
        self.skip_last_dims = skip_last_dims
        self.index_device = index_device

        # TODO: should caches be on cpu???
        self.caching = caching
        self.caching_numel = caching_numel
        self.caching_perm = caching_perm
        self.custom_can_be_canon_view = custom_can_be_canon_view
        self.prefer_index_axis_caching = prefer_index_axis_caching
        self.always_allow_cache_on_numel = always_allow_cache_on_numel

        self.cache: Dict[Tuple[Union[Tuple[ViewSpecIdx, ...], ViewSpecIdx], ...], VeryNamedTensor] = {}

        _ = self.caching_perm_val  # call to trigger asserts
        assert self.skip_last_dims >= 0, self.skip_last_dims
        assert self.skip_last_dims <= self.circuit.rank
        self.avoid_count = len(self.extra_shape) + self.skip_last_dims
        self.non_avoid_count = self.ndim - self.avoid_count

        assert len(dim_names) == len(
            self.shape
        ), f"{len(dim_names)} dim_names given ({dim_names}), but tensor is of rank {len(self.shape)} ({self.shape})"
        assert len(dim_types) == len(
            self.shape
        ), f"{len(dim_types)} dim_types given ({dim_types}), but tensor is of rank {len(self.shape)} ({self.shape})"
        idx_shape = tuple([len(g) for g in self.dim_idx_names])
        assert (
            idx_shape == self.shape
        ), f"dim_idx_names of shape {idx_shape} does not match tensor of shape {self.shape}"

    @property
    def caching_perm_val(self):
        if self.caching_perm is None:
            out = list(range(len(self.dim_names)))
        else:
            out = [(i if isinstance(i, int) else self.dim_names.index(i)) for i in self.caching_perm]
        check_permutation(out, len(self.dim_names))

        return out

    @property
    def shape(self):
        return self.circuit.shape + self.extra_shape

    @property
    def ndim(self):
        return len(self.shape)

    @staticmethod
    def try_idx_view(
        view_el: Union[ViewSpecIdx, List[ViewSpecIdx]], index_device: Union[str, torch.device] = "cpu"
    ) -> Optional[Union[int, torch.Tensor]]:
        if isinstance(view_el, int):
            return view_el
        elif isinstance(view_el, list):
            all_idxs: List[int] = []
            for i in view_el:
                if not isinstance(i, int):
                    return None
                all_idxs.append(i)
            return torch.tensor(all_idxs, device=index_device)

        return None

    @classmethod
    def apply_view_to_circ(
        cls,
        circ: Circuit,
        view_el: Union[ViewSpecIdx, List[ViewSpecIdx]],
        dim: int,
        index_device: Union[str, torch.device] = "cpu",
    ) -> Optional[Tuple[Circuit, bool]]:
        """returns new circuit and whether or not the dim was removed"""

        if (idx := cls.try_idx_view(view_el, index_device=index_device)) is not None:
            return Index(circ, make_index_at(idx, dim)), len(cls.get_view_shape_handled(view_el, circ.shape[dim])) == 0
        elif view_el == "mean":
            return circ.mean(axis=dim), True
        elif view_el == "sum":
            return circ.sum(axis=dim), True

        return None

    @classmethod
    def get_view_shape_handled(cls, view_el: Union[ViewSpecIdx, List[ViewSpecIdx]], size: int):
        return cls.get_view_shape_handled_status(view_el, size)[0]

    @classmethod
    def get_view_shape_handled_status(cls, view_el: Union[ViewSpecIdx, List[ViewSpecIdx]], size: int):
        if (idx := cls.try_idx_view(view_el)) is not None:
            if isinstance(idx, int):
                out_shape = ()
            elif isinstance(idx, torch.Tensor):
                out_shape = (len(idx),)
            else:
                assert_never(idx)

            assert len(out_shape) in [0, 1]
            return out_shape, True
        elif view_el == "mean":
            return (), True
        elif view_el == "sum":
            return (), True
        return (size,), False

    @staticmethod
    def apply_idx_to_list(idx: Union[slice, torch.Tensor], lst: List[T]) -> List[T]:
        if isinstance(idx, slice):
            return lst[idx]
        assert idx.ndim == 1
        return [lst[int(i)] for i in idx]

    def default_can_be_canon_view(self, sizes: Tuple[int, ...]):
        return math.prod(sizes) < self.caching_numel

    def can_be_canon_view(self, sizes: Tuple[int, ...], view: ViewSpec, orig_view: ViewSpec) -> bool:
        assert len(sizes) == len(view)
        if self.custom_can_be_canon_view is not None:
            return self.custom_can_be_canon_view(sizes, view, orig_view)
        else:
            return self.default_can_be_canon_view(sizes)

    @classmethod
    def strip_leftmost_index(cls, view: ViewSpec):
        view = copy(view)
        idx = min((i for i, v in enumerate(view) if cls.try_idx_view(v) is not None), default=None)
        if idx is None:
            return None
        view[idx] = "axis"
        return view

    def get_canonical_cache_suffix_view(
        self,
        perm_view: ViewSpec,
        perm_shape: Tuple[int, ...],
        perm_back: List[int],
        orig_view: ViewSpec,
        is_base: bool = True,
    ) -> Optional[ViewSpec]:
        # this caching logic is kinda gross and sad... : /
        index_stripped = self.strip_leftmost_index(perm_view)
        if self.prefer_index_axis_caching and index_stripped is not None:
            if (
                rec_out := self.get_canonical_cache_suffix_view(
                    index_stripped, perm_shape, perm_back, orig_view, is_base=False
                )
            ) is not None:
                return rec_out

        vals = [self.get_view_shape_handled_status(view_el, size) for view_el, size in zip(perm_view, perm_shape)]
        sizes = tuple(math.prod(s) for s, _ in vals)
        for i in reversed(range(self.ndim + 1)):
            target_view_start: ViewSpec = ["axis"] * i
            target_view = target_view_start + [("sum" if x == "mean" else x) for x in perm_view[i:]]
            new_p_shape = perm_shape[:i] + sizes[i:]
            un_perm_shape = tuple(new_p_shape[p] for p in perm_back)
            un_perm_view = [target_view[p] for p in perm_back]
            if self.can_be_canon_view(un_perm_shape, un_perm_view, orig_view) or (
                i == 0
                and is_base
                and self.default_can_be_canon_view(un_perm_shape)
                and self.always_allow_cache_on_numel
            ):
                return target_view

        return None

    def get_canonical_cache_suffix_view_perm(self, view: ViewSpec) -> Optional[ViewSpec]:
        perm_view = [view[p] for p in self.caching_perm_val]
        perm_shape = tuple(self.shape[p] for p in self.caching_perm_val)
        perm_back_d = {p: i for i, p in enumerate(self.caching_perm_val)}
        perm_back = [perm_back_d[i] for i in range(len(view))]

        out = self.get_canonical_cache_suffix_view(perm_view, perm_shape, perm_back, view)
        return op.map(out, lambda o: [o[p] for p in perm_back])

    def getView(self, view: ViewSpec, truncate_nan_rows: bool = False, allow_cache: bool = True):
        """
        view is list, where every element is either:
            - 'axis' or 'facet', representing no change
            - int, representing index for that dim
            - str representing reduction fn
        """
        print(f"{view=}")
        print(f"{os.getpid()=}")

        if self.caching and allow_cache and (canon_view := self.get_canonical_cache_suffix_view_perm(view)) is not None:
            cachable_canon_view = tuple((tuple(x) if isinstance(x, list) else x) for x in canon_view)
            if cachable_canon_view in self.cache:
                print("hit cache!!!", f"{canon_view=}")
                canon_vnt = self.cache[cachable_canon_view]
            else:
                print("fresh computation", f"{canon_view=}")
                canon_vnt = self.getView(canon_view, truncate_nan_rows=False, allow_cache=False)
                self.cache[cachable_canon_view] = canon_vnt

            ret_vnt = copy(canon_vnt)
            new_tensor = ret_vnt.tensor.clone()

            for i, (c_v_el, v_el) in enumerate(zip(canon_view, view)):
                if v_el == "mean" and c_v_el == "sum":
                    new_tensor /= self.circuit.shape[i]

            ret_vnt.tensor = new_tensor

            reduced_away = lambda x: x in ["mean", "sum"] or isinstance(x, int)
            # if the dim wasn't reduced and wasn't canonicalized, it must have been handled which implies that we should forward "axis"
            final_view: ViewSpec = [
                (v_el if c_v_el == "axis" else "axis")
                for c_v_el, v_el in zip(canon_view, view)
                if not reduced_away(c_v_el)
            ]

            return ret_vnt.getView(final_view, truncate_nan_rows=truncate_nan_rows)

        out_circ = self.circuit

        handled_dims = set[int]()
        removed_dims = set[int]()

        assert len(view) == self.ndim

        for view_el in view:
            if isinstance(view_el, str) and view_el not in ["mean", "sum", "axis"]:
                warn(
                    f"{view_el=}\nFor the circuit VNT, non mean/sum reduction operations are run last!\nThese "
                    "ops aren't commutative over ordering, so this isn't consistent with other VNT types!"
                )

        for i, view_el in reversed(list(enumerate(view))[: self.non_avoid_count]):
            maybe_new_circ_removed = self.apply_view_to_circ(out_circ, view_el, i)
            if maybe_new_circ_removed is not None:
                handled_dims.add(i)
                out_circ, is_removed = maybe_new_circ_removed
                if is_removed:
                    removed_dims.add(i)

        eval_out = self.eval_callback(out_circ, view[self.non_avoid_count :])

        handled: Set[int]
        removed: Set[int]
        if isinstance(eval_out, torch.Tensor):
            tensor = eval_out
            handled, removed = set(), set()
        else:
            tensor, handled, removed = eval_out
        assert handled.issubset(range(len(view[self.non_avoid_count :])))
        assert removed.issubset(range(len(view[self.non_avoid_count :])))
        for was_handled in handled:
            handled_dims.add(was_handled + self.non_avoid_count)
        for was_removed in removed:
            removed_dims.add(was_removed + self.non_avoid_count)

        assert handled_dims.issuperset(removed_dims)

        new_dim_idxs: List[List[str]] = []
        for dim_idxs, view_el in zip(self.dim_idx_names, view):
            if (idx := self.try_idx_view(view_el)) is not None and not isinstance(idx, int):
                dim_idxs = self.apply_idx_to_list(idx, dim_idxs)
            new_dim_idxs.append(list(dim_idxs))

        TypeVar("T")

        def filter_by_removed(item: Sequence[T]) -> List[T]:
            assert len(item) == self.ndim
            return [x for i, x in enumerate(item) if i not in removed_dims]

        return VeryNamedTensor(
            tensor,
            filter_by_removed(self.dim_names),
            filter_by_removed(self.dim_types),
            filter_by_removed(new_dim_idxs),
            units=self.units,
            title=self.title,
            aux_info=self.aux_info,
            selected_options=self.selected_options,
            default_view_spec=self.default_view_spec,
        ).getView(
            [x if i not in handled_dims else "axis" for i, x in enumerate(view) if i not in removed_dims],
            truncate_nan_rows=truncate_nan_rows,
        )

    def to_dict(self):
        print("sending circ vnt", self.title)
        return {
            "_getView": self.getView,
            "_getSparseView": lambda x: x,
            "dim_names": self.dim_names,
            "dim_idx_names": self.dim_idx_names,
            "dim_types": self.dim_types,
            "units": self.units,
            "title": self.title,
            "optionsSpec": None,
            "aux_info": self.aux_info,
            "default_view_spec": self.default_view_spec,
        }
