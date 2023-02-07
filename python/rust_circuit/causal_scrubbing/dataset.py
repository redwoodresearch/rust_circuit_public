from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Set, Type, TypeVar, Union

import attrs
import torch
from attr import frozen

import rust_circuit as rc

from ..py_utils import FrozenDict


def all_same_len(self: Dataset, attribute, arrs: Mapping[str, rc.Array]):
    assert len(arrs) >= 1
    l = None
    for a in arrs.values():
        if l is None:
            l = a.shape[0]
        elif not a.shape[0] == l:
            raise ValueError(
                f"All Arrays in a Dataset must have the same `len`, found instead at least two: {l}, {a.shape[0]}"
            )


def arrs_to_map(arrs: Union[Iterable[rc.Array], Mapping[str, rc.Array]]) -> FrozenDict[str, rc.Array]:
    if isinstance(arrs, Mapping):
        for name, a in arrs.items():
            assert name == a.name
        return FrozenDict(arrs)
    else:
        arrs_map: Dict[str, rc.Array] = {}
        for a in arrs:
            if a.name in arrs_map:
                if not a == arrs_map[a.name]:
                    raise ValueError(f"We found two different Arrays with the same name: {a}, {arrs_map[a.name]}")
            arrs_map[a.name] = a
        return FrozenDict(arrs_map)


def names_in_arrs(self: Dataset, attribute, input_names: Set[str]):
    for name in input_names:
        assert name in self.arrs, name


def frozenset_converter(xs: Set[str]) -> frozenset[str]:  # mypy has issues with just using frozenset as converter
    return frozenset(xs)


TDataset = TypeVar("TDataset", bound="Dataset")


@frozen(hash=True)
class Dataset:
    # Holds arrays look-up-able by name; you can provide any iterable and it will be turned into a map.
    # In particular no two can have the same name. Also, they can be accessed as `ds.arr_name`, unless
    # you name your array something silly like __len__
    arrs: Mapping[str, rc.Array] = attrs.field(validator=all_same_len, converter=arrs_to_map)
    # Which arrays (by name) are inputs to the circuit that should be replaced when doing causal scrubbing.
    # By default this is all of them.
    input_names: Set[str] = attrs.field(
        validator=names_in_arrs, factory=set, converter=frozenset_converter, kw_only=True
    )

    def __attrs_post_init__(self):
        if not len(self.input_names):
            object.__setattr__(self, "input_names", frozenset(self.arrs.keys()))

    def __getattr__(self, __name: str) -> rc.Array:
        try:
            return self.arrs[__name]
        except KeyError:
            raise AttributeError(__name)

    def __len__(self) -> int:
        return list(self.arrs.values())[0].shape[0]

    def __getitem__(self, idxs: rc.TorchAxisIndex):
        if isinstance(idxs, int):
            idxs = slice(idxs, idxs + 1)
        return attrs.evolve(self, arrs={name: rc.Array(inp.value[idxs], name) for name, inp in self.arrs.items()})

    def __str__(self) -> str:
        # Probably you want to overwrite this when subclassing so you get pretty prints!
        return str([f"<{a.name} {a.shape}>" for a in self.arrs.values()])

    def sample(self, count, rng: Optional[torch.Generator] = None):
        idxs = torch.multinomial(
            torch.ones(size=(len(self),), dtype=torch.float32, device=(rng.device if rng else None)),
            num_samples=count,
            replacement=True,
            generator=rng,
        )
        return self[idxs]

    @classmethod
    def unwrap(cls: Type[TDataset], d: Dataset) -> TDataset:
        assert isinstance(d, cls), (type(d), cls)
        return d


def color_dataset(ds, html=False):
    def color_string_from_int(i: int):
        hue = (i % (255 / 10)) * 10
        return f"hsl({hue}, 90%, 60%)"

    if html:
        return "darkgrey" if ds is None else color_string_from_int(hash(ds))
    else:
        # pretty made up, would be nice if printer had a "color by feature"
        return 6 if ds is None else 2 + hash(ds) % 6
