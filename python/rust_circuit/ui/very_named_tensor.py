import functools
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union, cast

import torch

import rust_circuit.optional as op

RRFS_DIR = "/home/ubuntu/rrfs"
# from interp.tools.rrfs import RRFS_DIR

# tokenizer loading copied from the old interpretability_tools.py to avoid jax dependency
RRFS_INTERP_MODELS_DIR = f"{RRFS_DIR}/interpretability_models_jax/"
MODELS_DIR = os.environ.get("INTERPRETABILITY_MODELS_DIR", RRFS_INTERP_MODELS_DIR)


def get_gpt_tokenizer_with_end_tok():
    import transformers

    tokenizer = transformers.GPT2TokenizerFast.from_pretrained(f"{MODELS_DIR}/gpt2/tokenizer")
    tokenizer._add_tokens(["[END]"])
    tokenizer.pad_token = "[END]"
    return tokenizer


@functools.cache
def get_interp_tokenizer():
    tokenizer = get_gpt_tokenizer_with_end_tok()
    tokenizer._add_tokens(["[BEGIN]"])
    tokenizer.eos_token = "[END]"
    return tokenizer


def with_dim_i_in_front(tensor, func: Callable[[Any], Any], axis):
    dim_permutation = [axis] + [i for i, x in enumerate(tensor.shape) if i != axis]
    inv_permutation = [x for x in dim_permutation]
    for i, x in enumerate(dim_permutation):
        inv_permutation[x] = i

    tensor = torch.permute(tensor, dim_permutation)
    tensor = func(tensor)
    tensor = torch.permute(tensor, inv_permutation)
    return tensor


def index_to_string(index):
    fixed_list = []
    for x in index:
        if isinstance(x, slice):
            if x.start is None and x.stop is None:
                fixed_list.append(":")
            else:
                fixed_list.append(
                    ("" if x.start is None else str(x.start)) + ":" + ("" if x.stop is None else str(x.stop))
                )
        elif isinstance(x, str):
            fixed_list.append(f'"{x}"')
        else:
            fixed_list.append(str(x))
    return f"[{', '.join(fixed_list)}]"


ViewSpecIdx = Union[int, str]
ViewSpec = List[Union[List[ViewSpecIdx], ViewSpecIdx]]


def array_getitem_separate(array, axes):
    """normal np getitem treats multiple array dimensions as matched, but this doesn't"""
    axes = axes + (slice(None, None, None),) * (len(array.shape) - len(axes))
    for i, ax in reversed(list(enumerate(axes))):
        array = array.__getitem__(tuple((slice(None, None),) * i + (ax,)))
    return array


def entropy_fn(tensor, dim):
    if torch.sum(tensor < 0) > 0:
        return torch.full_like(torch.sum(tensor, dim=dim), torch.nan)
    tensor = tensor / torch.nansum(tensor, dim=dim, keepdim=True)
    return -torch.nansum(tensor * torch.log(tensor))


def torch_take(tensor: torch.Tensor, indices, axis: int):
    return tensor[(slice(None, None),) * axis + (indices,)]


@dataclass
class VeryNamedTensor:
    reduction_fns = {
        "mean": torch.nanmean,
        "sum": torch.nansum,
        "norm": torch.linalg.norm,
        "max": torch.amax,
        "min": torch.amin,
        "entropy": entropy_fn,
    }

    def __init__(
        self,
        tensor,
        dim_names,
        dim_types,
        dim_idx_names,
        units="units",
        title="untitled_tensor",
        aux_info: Any = {},
        selected_options=None,
        default_axis_view: Union[List[ViewSpecIdx], ViewSpecIdx] = "axis",
        default_view_spec: Optional[ViewSpec] = None,  # overrides axis
    ):
        self.tensor: torch.Tensor = tensor
        self.selected_options = selected_options
        self.dim_names: List[str] = dim_names
        self.dim_types: List[str] = dim_types
        self.dim_idx_names: List[List[str]] = [[str(i) for i in x] for x in dim_idx_names]
        self.units: str = units
        self.title = title
        self.shape = self.tensor.shape
        self.aux_info = aux_info
        via_axis: ViewSpec = [default_axis_view] * len(self.dim_names)
        self.default_view_spec: ViewSpec = op.unwrap_or(default_view_spec, via_axis)

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

    # can index with slices of numbers, or names
    # doesn't support
    def __getitem__(self, axes):
        return self.getitem(axes, False)

    def getitem(self, axes, no_title_change=False):
        if not isinstance(axes, tuple):
            axes = (axes,)

        if len(axes) < len(self.dim_types):
            axes = axes + (slice(None, None),) * (len(self.dim_types) - len(axes))

        axes = tuple([self.dim_idx_names[i].index(x) if isinstance(x, str) else x for i, x in enumerate(axes)])
        used_axes = [i for i, x in enumerate(axes) if (isinstance(x, slice) or isinstance(x, torch.Tensor))]
        dim_idx_names = []
        for i, (ax, names) in enumerate(zip(axes, self.dim_idx_names)):
            if i in used_axes:
                if isinstance(ax, torch.Tensor):
                    dim_idx_names.append([names[j] for j in ax])
                else:
                    dim_idx_names.append(names.__getitem__(ax))
        return VeryNamedTensor(
            tensor=array_getitem_separate(self.tensor, axes),
            dim_names=[name for i, name in enumerate(self.dim_names) if i in used_axes],
            dim_types=[ty for i, ty in enumerate(self.dim_types) if i in used_axes],
            dim_idx_names=dim_idx_names,
            units=self.units,
            title=self.title if no_title_change else f"{self.title}{index_to_string(axes)}",
            selected_options=self.selected_options,
            default_view_spec=self.default_view_spec,
        )

    def getView(self, view: ViewSpec, truncate_nan_rows=False):
        """
        view is list, where every element is either:
            - 'axis' or 'facet', representing no change
            - int, representing index for that dim
            - str in self.reduction_fns.keys(), representing reduction fn
        """
        result_tensor = self.tensor
        used_axes = [i for i, x in enumerate(view) if x in ["axis", "facet"] or isinstance(x, list)]
        for i, view_el in reversed(list(enumerate(view))):
            if view_el in ["axis", "facet"]:
                continue
            elif isinstance(view_el, str):
                assert (
                    view_el in self.reduction_fns.keys()
                ), f"element {view_el} in view {view} not supported, options are {list(self.reduction_fns.keys())}"
                reduction_fn: Callable = op.unwrap(self.reduction_fns.get(view_el))  # type: ignore
                result_tensor = reduction_fn(result_tensor, dim=i)
            elif isinstance(view_el, int):
                result_tensor = torch_take(result_tensor, view_el, i)
            elif isinstance(view_el, list):
                string_idxs = [i for i, x in enumerate(view_el) if isinstance(x, str)]
                idxs_zeros_for_strings = [-1 if isinstance(x, str) else x for x in view_el]
                view_el_v = view_el

                def myfunc(tensor):
                    tensor = tensor.clone()[torch.tensor(idxs_zeros_for_strings, dtype=torch.long)]
                    for string_idx in string_idxs:
                        view_item = view_el_v[string_idx]
                        assert isinstance(view_item, str)
                        tensor[string_idx] = self.reduction_fns[view_item](tensor, dim=0)
                    return tensor

                result_tensor = with_dim_i_in_front(result_tensor, myfunc, i)
                assert isinstance(result_tensor, torch.Tensor)
            else:
                raise ValueError(f"Invalid view: {view}")
        result_vnt = self.__getitem__(
            tuple(
                [
                    slice(None, None)
                    if dim in ["axis", "facet"]
                    else (
                        torch.tensor(
                            [inner_idx if isinstance(inner_idx, int) else 0 for inner_idx in dim], dtype=torch.long
                        )
                        if isinstance(dim, list)
                        else 0
                    )
                    for dim in view
                ]
            )
        )
        for used_i, base_i in enumerate(used_axes):
            listy_view_el = view[base_i]
            if isinstance(listy_view_el, list):
                result_vnt.dim_idx_names[used_i] = [
                    self.dim_idx_names[base_i][inner_idx] if isinstance(inner_idx, int) else inner_idx
                    for inner_idx in listy_view_el
                ]

        assert isinstance(result_tensor, torch.Tensor), type(result_tensor)

        result_vnt.tensor = result_tensor
        return result_vnt.truncate_nan_rows() if truncate_nan_rows else result_vnt

    def truncate_nan_rows(self):
        is_nan = torch.isnan(self.tensor)
        all_nan_by_dim_row = [
            torch.stack([torch_take(is_nan, torch.tensor([i]), dim).all() for i in range(self.shape[dim])], dim=0)
            for dim in range(len(self.shape))
        ]
        any_row_all_nan = torch.stack(all_nan_by_dim_row).any()
        if any_row_all_nan:
            new_tensor = self.tensor
            new_dim_idx_names = self.dim_idx_names
            for dim, rows_all_nan in enumerate(all_nan_by_dim_row):
                if rows_all_nan.any():
                    not_nan_idxs = torch.arange(len(rows_all_nan))[~rows_all_nan]
                    new_tensor = torch_take(new_tensor, not_nan_idxs, dim)
                    new_dim_idx_names[dim] = [self.dim_idx_names[dim][i] for i in not_nan_idxs]
            return VeryNamedTensor(
                tensor=new_tensor,
                dim_names=self.dim_names,
                dim_types=self.dim_types,
                dim_idx_names=new_dim_idx_names,
                units=self.units,
                title=self.title,
                selected_options=self.selected_options,
                default_view_spec=self.default_view_spec,
            )
        else:
            return self

    def to_dict(self):
        return {
            "tensor": self.tensor.cpu().numpy(),
            "dim_names": self.dim_names,
            "dim_idx_names": self.dim_idx_names,
            "dim_types": self.dim_types,
            "units": self.units,
            "title": self.title,
            "selected_options": self.selected_options,
            "default_view_spec": self.default_view_spec,
        }

    def to_lvnt(self):
        return LazyVeryNamedTensor(
            lambda: self.tensor,
            dim_names=self.dim_names,
            dim_types=self.dim_types,
            dim_idx_names=self.dim_idx_names,
            units=self.units,
            title=self.title,
            default_view_spec=self.default_view_spec,
        )

    def __repr__(self):
        indent = " " * 4  # len("VeryNamedTensor(")
        str = "VeryNamedTensor(\n"
        str += f"{indent}shape={self.shape}\n"
        str += f"{indent}dim_names={self.dim_names}\n"
        str += f"{indent}dim_types={self.dim_types}\n"
        str += f"{indent}dim_idx_names={self.dim_idx_names}\n"
        str += f"{indent}units={self.units}\n"
        str += f"{indent}tensor={self.tensor}\n"
        str += f"{indent}default_view_spec={self.default_view_spec}\n"
        str += ")"
        return str


class LazyVeryNamedTensor:
    def __init__(
        self,
        tensor_thunk,
        dim_names,
        dim_types,
        dim_idx_names,
        units="units",
        title="untitled_tensor",
        aux_info: Any = {},
        options_spec: Any = None,
        default_axis_view: Union[List[ViewSpecIdx], ViewSpecIdx] = "axis",
        default_view_spec: Optional[ViewSpec] = None,  # overrides axis
    ):
        self.options_spec = options_spec
        self.tensor_thunk: Union[Callable[[], Any], Callable[[Any], Any]] = tensor_thunk
        self.dim_names: List[str] = dim_names
        self.dim_types: List[str] = dim_types
        self.dim_idx_names: List[List[str]] = [[str(i) for i in x] for x in dim_idx_names]
        self.units: str = units
        self.title = title
        self.shape = tuple([len(x) for x in dim_idx_names])
        self.aux_info = aux_info
        self.realized_vnt: Optional[VeryNamedTensor] = None
        via_axis: ViewSpec = [default_axis_view] * len(self.dim_names)
        self.default_view_spec: ViewSpec = op.unwrap_or(default_view_spec, via_axis)

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

    def __getitem__(self, axes):
        return self.getitem(axes, False)

    def getitem(self, axes: Tuple, no_title_change: bool = False):
        if not isinstance(axes, tuple):
            axes = (axes,) + (slice(None, None),) * (len(self.dim_types) - 1)

        if len(axes) < len(self.dim_types):
            axes = axes + (slice(None, None),) * (len(self.dim_types) - len(axes))

        axes = tuple([self.dim_idx_names[i].index(x) if isinstance(x, str) else x for i, x in enumerate(axes)])

        if self.realized_vnt is not None:
            thunk = cast(
                Union[Callable[[], Any], Callable[[Any], Any]],
                lambda: op.unwrap(self.realized_vnt).tensor.__getitem__(axes),
            )
        else:
            if self.options_spec is not None:
                # using type ignores bc mypy doesn't know what "function that takes 1 or 0 arguments" is
                thunk = lambda options: cast(Any, self.tensor_thunk)(fd.freeze(options)).__getitem__(axes)  # type: ignore
            else:
                thunk = lambda: cast(Any, self.tensor_thunk)().__getitem__(axes)  # type: ignore
        return LazyVeryNamedTensor(
            tensor_thunk=thunk,
            dim_names=[x for i, x in zip(axes, self.dim_names) if isinstance(i, slice)],
            dim_types=[x for i, x in zip(axes, self.dim_types) if isinstance(i, slice)],
            dim_idx_names=[x.__getitem__(i) for i, x in zip(axes, self.dim_idx_names) if isinstance(i, slice)],
            units=self.units,
            title=self.title if no_title_change else f"{self.title}{index_to_string(axes)}",
            aux_info=self.aux_info,
            options_spec=self.options_spec,
            default_view_spec=self.default_view_spec,
        )

    def getView(
        self,
        view,
        options=None,
        truncate_nan_rows=False,
    ):
        realized_tensor = None
        if self.realized_vnt is None or (
            self.options_spec is not None and json.dumps(options) != json.dumps(self.realized_vnt.selected_options)
        ):
            realized_tensor = self.tensor_thunk(*([options] if options is not None else []))
        if realized_tensor is not None:
            self.realized_vnt = VeryNamedTensor(
                tensor=realized_tensor,
                dim_names=self.dim_names,
                dim_types=self.dim_types,
                dim_idx_names=self.dim_idx_names,
                units=self.units,
                title=self.title + (" options:" + json.dumps(options) if options is not None else ""),
                selected_options=options,
            )
        return op.unwrap(self.realized_vnt).getView(view, truncate_nan_rows=truncate_nan_rows)

    def to_dict(self):
        return {
            "_getView": self.getView,
            "_getSparseView": lambda x: x,  # TODO reimplement this
            "dim_names": self.dim_names,
            "dim_idx_names": self.dim_idx_names,
            "dim_types": self.dim_types,
            "units": self.units,
            "title": self.title,
            "optionsSpec": self.options_spec,
            "aux_info": self.aux_info,
            "default_view_spec": self.default_view_spec,
        }


def vnt_guessing_shit_model_tokens(tensor, model, tokens, title="title", permissive=True, potential_dims=[]):
    return vnt_guessing_shit(
        tensor,
        [
            {"type": "seq", "name": "seq", "idx_names": [get_interp_tokenizer().decode([x]) for x in tokens]},
            {"type": "layer", "name": "layer", "idx_names": [str(i) for i in range(model.num_layers)]},
            {
                "type": "layerWithIO",
                "name": "layerWithIO",
                "idx_names": (["embeds"] + [str(i) for i in range(model.num_layers)] + ["outputs"]),
            },
            {"type": "heads", "name": "heads", "idx_names": [str(i) for i in range(model.num_heads)]},
            {"type": "neurons", "name": "neurons", "idx_names": [str(i) for i in range(model.hidden_size * 4)]},
            {"type": "hidden", "name": "hidden", "idx_names": [str(i) for i in range(model.hidden_size)]},
            *potential_dims,
        ],
        title=title,
        permissive=permissive,
    )


def vnt_guessing_shit(tensor, potential_dims, title="title", units="units", permissive=True):
    dim_types = []
    dim_idx_names = []
    dim_names = []
    for dim_length in tensor.shape:
        matching_potentials = [x for x in potential_dims if len(x["idx_names"]) == dim_length]
        if len(matching_potentials) == 0:
            assert permissive is True, "no matching length dims"
            dim_idx_names.append([str(i) for i in range(dim_length)])
            dim_types.append("unknown")
            name = "unknown"
            while name in dim_names:
                name += "again"
            dim_names.append(name)
        else:
            matching = matching_potentials[0]
            dim_types.append(matching["type"])
            dim_idx_names.append(matching["idx_names"])
            name = matching["name"]
            while name in dim_names:
                name += "again"
            dim_names.append(name)
    return VeryNamedTensor(tensor, dim_names, dim_types, dim_idx_names, units, title)


def get_shapes_spec(tokens, model):
    return {
        "tokens": tokens,
        "numLayers": model.num_layers,
        "numHeads": model.num_heads,
        "hiddenSize": model.hidden_size,
    }
