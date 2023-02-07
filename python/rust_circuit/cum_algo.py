from __future__ import annotations

import itertools
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar, cast

from interp.circuit.partitioning import partition
from rust_circuit import optional as op

from ._rust import (
    Add,
    Circuit,
    Cumulant,
    Einsum,
    Module,
    ModuleArgSpec,
    ModuleSpec,
    Scalar,
    Symbol,
    extract_rewrite_raw,
    kappa_term,
    symbolic_sizes,
)


def cum_name_to_eps_name(cum_name: str):
    """Replace the "k" at the beginning by an "eps" """
    return "eps" + cum_name[1:] if cum_name[0] == "k" else cum_name + "_eps"


def kappa_hat(cumulant: Cumulant):
    """this is an mathematical object we define for eps attrib/decomposition"""
    assert cumulant.num_children > 0
    if cumulant.num_children == 1:
        return Add.minus(cumulant.children[0], cumulant, name=f"{cum_name_to_eps_name(cumulant.name)}")
    return cumulant


def eps_term(args: List[List[Tuple[int, Circuit]]]):
    new_out, _ = kappa_term(args, on_sub_cumulant_fn=kappa_hat)
    # Note: autonaming + rename in kappa_hat will produce a nice name
    return new_out


def get_multiplier_default(p: List[List[Tuple[int, Circuit]]]) -> float:
    return (-1) ** sum(len(b) > 1 for b in p)


T = TypeVar("T")


def split_in_buckets(seq: Sequence[T], bucket_sizes: Sequence[int]) -> List[Sequence[T]]:
    res: List[Sequence[T]] = []
    runnning_count = 0
    for bsize in bucket_sizes:
        res.append(seq[runnning_count : runnning_count + bsize])
        runnning_count += bsize
    return res


def eps_attrib_module(
    node_ranks: Sequence[int],
    non_centered: bool = False,
    name: str = "epsilon",
    extra_filter: Callable[[List[List[Tuple[int, Circuit]]]], bool] = lambda _: True,
    get_multiplier: Callable[[List[List[Tuple[int, Circuit]]]], float] = get_multiplier_default,
    symbolic_size_start: int = 50,
) -> ModuleSpec:
    """returns epsilion(*circuits)
    non_centered means we don't subtract out the cumulant
    """

    shapes = split_in_buckets(symbolic_sizes()[symbolic_size_start:], node_ranks)

    circuits: List[Circuit] = list[Circuit](
        [Symbol.new_with_random_uuid(tuple(shape), f"input_{i}") for i, shape in enumerate(shapes)]
    )
    input_specs = [(c, ModuleArgSpec(cast(Symbol, c))) for c in circuits]

    out: List[Einsum] = []

    if len(circuits) == 0:
        eps: Circuit = Scalar(1.0, name=name)
        return extract_rewrite_raw(eps, input_specs, prefix_to_strip=None, module_name=name).spec

    if non_centered and len(circuits) == 1:
        eps = circuits[0]
        return extract_rewrite_raw(eps, input_specs, prefix_to_strip=None, module_name=name).spec

    for p in partition(list(enumerate(circuits))):
        if (non_centered and len(p) == 1) or not extra_filter(p):
            continue

        new_out = eps_term(p)
        assert new_out.shape == tuple(itertools.chain.from_iterable(c.shape for c in circuits))
        out.append(Einsum.scalar_mul(new_out, get_multiplier(p)))
    eps = Add(*out, name=f"{name}_sum")

    return extract_rewrite_raw(eps, input_specs, prefix_to_strip=None, module_name=name).spec


def eps_attrib(
    cumulant: Cumulant,
    non_centered: bool = False,
    name: Optional[str] = None,
    extra_filter: Callable[[List[List[Tuple[int, Circuit]]]], bool] = lambda _: True,
    get_multiplier: Callable[[List[List[Tuple[int, Circuit]]]], float] = get_multiplier_default,
):
    """returns epsilion(*cumulant.circuits)
    non_centered means we don't subtract out the cumulant
    """

    name_ = op.unwrap_or(name, cum_name_to_eps_name(cumulant.name))
    cum_circuits = cumulant.children
    return Module.new_flat(
        eps_attrib_module(
            [c.rank for c in cum_circuits],
            non_centered=non_centered,
            extra_filter=extra_filter,
            get_multiplier=get_multiplier,
        ),
        *cum_circuits,
        name=name_,
    )
