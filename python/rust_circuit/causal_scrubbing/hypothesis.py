from __future__ import annotations

import functools
import uuid
from collections import Counter
from typing import Callable, Dict, Hashable, List, Optional, Protocol, Set, Tuple, Union
from warnings import warn

import attrs
import torch

from rust_circuit import (
    Circuit,
    GeneralFunction,
    GeneralFunctionShapeInfo,
    GeneralFunctionSpecBase,
    IterativeMatcher,
    IterativeMatcherIn,
    MatcherIn,
    PrintHtmlOptions,
    PrintOptions,
    hash_tensor,
    new_traversal,
    restrict,
)

from .dataset import Dataset, color_dataset


@attrs.define(eq=True, hash=True, init=False)
class PoolAnnotation:
    """
    Pool annotation for a cond sampler. See pool.py demo for more info.

    size: size of the pool, i.e. how many different datasets will be sampled. A pool size of 1 has semantic meaning
        (you want to enforce things being sampled together). A pool of size, say, 4-10 is useful for performance.
        Intermediate sizes seem confusing, it's not clear to me why you would want them.
    id: to define which annotations, and therefore which cond samplers, are sharing a pool.
    """

    size: int
    id: uuid.UUID

    def __init__(self, size: int = 4, id: Optional[uuid.UUID] = None):
        if size <= 0:
            raise ValueError(size)
        self.size = size
        self.id = uuid.uuid4() if id is None else id


@attrs.define(hash=True, eq=True)
class PoolNotPresent:
    ...


PNP = PoolNotPresent()
MaybePoolAnnotation = Union[PoolAnnotation, PoolNotPresent]

DsEqClassT = Hashable
Pool = List[Optional[Dataset]]
PoolPerDsEqClass = dict[DsEqClassT, Pool]
SharedPoolKey = Tuple["CondSampler", Dataset]
PoolsPerSampler = dict[SharedPoolKey, PoolPerDsEqClass]


@attrs.define(hash=True, eq=True)
class CondSampler(Protocol):
    """
    Responsible for sampling a new dataset given a source dataset to draw from, and a reference dataset.

    Parameters:
    'pool_annot`: See pool.py demo for more info. If PoolNotPresent (the default), no sample reuse will
        happen. Using pools is recommended if your treeified model is too big (e.g for deep networks with
        non-trivial samplers).
    """

    pool_annot: MaybePoolAnnotation = PNP

    def __call__(self, ref: Dataset, ds: Dataset, rng=None) -> Dataset:
        ...

    def ds_eq_class(self, ds: Dataset) -> Hashable:
        """
        Defines what reference datasets share a pool. By default, no sharing is allowed.

        This should return some value which will be the same for datasets that "are the same, according to this sampler."

        Note: not guaranteed to actually agree with __call__, that's up to the author.
        """
        return ds

    def sample_and_update_pool(
        self, ref: Dataset, ds: Dataset, rng: torch.Generator, pools_per_sampler: PoolsPerSampler
    ) -> Dataset:
        """
        Reuses a dataset from the pool or samples a new one and updates the pool.
        """
        if isinstance(self.pool_annot, PoolNotPresent):
            return self(ref, ds, rng)

        # Get pools for this sampler. Based on hash, which depends on its PoolAnnotation as well as other attrs.
        if (self, ds) not in pools_per_sampler:
            pools_per_sampler[(self, ds)] = PoolPerDsEqClass()
        pools = pools_per_sampler[(self, ds)]

        # Get the pool for the ref ds. Depends on what ref ds's eq class is, according to this sampler.
        ref_eq_class = self.ds_eq_class(ref)
        if ref_eq_class not in pools:
            pools[ref_eq_class] = [None for _ in range(self.pool_annot.size)]
        pool = pools[ref_eq_class]

        # Pick a ds from the pool, lazily sampling a new one if needed.
        picked = torch.randint(low=0, high=len(pool), size=(), generator=rng)
        picked_ds = pool[picked]
        if picked_ds is None:
            picked_ds = self(ref, ds, rng)
            pool[picked] = picked_ds
        return picked_ds

    def pretty_str(self, data: Optional[Dataset], datum_idx=0) -> str:
        # Not __str__ because it takes a dataset
        if data is None:
            extra = ""
        else:
            datum = data[datum_idx]
            extra = f"d={datum}"
            cond_extra = self.str_cond(datum)
            if cond_extra:
                extra += ", " + cond_extra
            extra = f"({extra})"
        return f"{self.class_str()}{extra}"

    def class_str(self):
        return f"{self.__class__.__name__}" + (f"#{hash(self.pool_annot) & 1023}" if self.pool_annot != PNP else "")

    def str_cond(self, datum: Dataset) -> str:
        return ""


@attrs.define(hash=True, eq=True, init=False)
class FuncSampler(CondSampler):
    """
    Samples each datum uniformly from the subset of ds that agrees on the value of func. That is, for x in ref_ds we sample a corresponding x' uniformly from `[y for y in ds if func(y) == func(x)]`.

    This impliments the standard causal scrubbing sampling operation as described in the writeup, if func returns the feature(s) computed by the interp node.

    `func: Dataset -> Tensor` must return a 1 or 2 dimensional tensor, where the first dimesion is the same length as the dataset the function is called on.
    """

    func: Callable[[Dataset], torch.Tensor] = attrs.field(
        kw_only=True
    )  # irrelevant since we are writing our own init method, but attrs complains otherwise

    def __init__(
        self,
        func: Callable[[Dataset], torch.Tensor],
        pool_annot: MaybePoolAnnotation = PNP,
    ):
        self.func = func
        self.pool_annot = pool_annot

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def get_matching_idxs(
        func: Callable[[Dataset], torch.Tensor], ref: Dataset, ds: Dataset
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        r_vals = func(ref)
        d_vals = func(ds)
        r_vals = r_vals[:, None] if r_vals.ndim == 1 else r_vals
        d_vals = d_vals[:, None] if d_vals.ndim == 1 else d_vals
        result = []
        for elem in torch.unique(r_vals, dim=0):
            matching_idxs_r = torch.nonzero(torch.all(r_vals == elem, dim=1).ravel(), as_tuple=True)[0]
            matching_idxs_d = torch.nonzero(torch.all(d_vals == elem, dim=1).ravel(), as_tuple=True)[0]
            r_count = len(matching_idxs_r)
            d_count = len(matching_idxs_d)

            assert d_count > 0, f"no matching idxs found for value {elem} of condition {func}"
            if (
                d_count < r_count
            ):  # Unfortunately happens a lot when func is injective, because r may contain duplicates but d won't
                warn(f"not enough unique dataset samples for value {elem} of condition {func}, {d_count} < {r_count}")
            result.append((matching_idxs_r, matching_idxs_d))
        return result

    def __call__(self, ref: Dataset, ds: Dataset, rng=None) -> Dataset:
        r_d_pairs = FuncSampler.get_matching_idxs(self.func, ref, ds)
        idxs = torch.full((len(ref),), -1, dtype=torch.int64, device=r_d_pairs[0][0].device)
        for matching_idxs_r, matching_idxs_d in r_d_pairs:
            idxs[matching_idxs_r] = matching_idxs_d[
                torch.multinomial(
                    torch.ones(len(matching_idxs_d), device=matching_idxs_d.device, dtype=torch.float32),
                    num_samples=len(matching_idxs_r),
                    replacement=True,
                    generator=rng,
                )
            ]

        assert (idxs != -1).all(), "this should never happen!"
        return ds[idxs]

    def ds_eq_class(self, ds):
        return hash_tensor(self.func(ds))

    def str_cond(self, datum: Dataset) -> str:
        out = self.func(datum)[0]
        if isinstance(out, torch.Tensor) and out.numel() == 1:
            out = out.item()  # type: ignore
        return f"f(d)={str(out)}"


@attrs.define(hash=True, eq=True)
class UncondSampler(CondSampler):
    """
    Samples randomly without conditioning on the reference dataset.
    """

    def __call__(self, ref: Dataset, ds: Dataset, rng=None) -> Dataset:
        return ds.sample(len(ref), rng)

    def ds_eq_class(self, ds):
        return ()


@attrs.define(hash=True, eq=True)
class UncondTogetherSampler(UncondSampler):
    """
    Samples randomly without conditioning on the reference dataset,
    and has a pool of size 1 so a single sample will be reused.

    By default all instances of this class are equal, so all nodes with an
    UncondTogetherSampler will get the same ds, but you can pass a different
    uuid at init if you want to create groups of nodes.
    """

    def __init__(self, id=uuid.UUID("566ae005-4a34-4e37-9383-731e1a722ef2")):
        self.pool_annot = PoolAnnotation(1, id=id)


@attrs.define(hash=True, eq=True)
class ExactSampler(CondSampler):
    def __call__(self, ref: Dataset, ds: Dataset, rng=None) -> Dataset:
        return ref  # should maybe check this is valid draw from ds


@attrs.define(hash=True, eq=True)
class FixedOtherSampler(UncondSampler):
    other: Dataset = attrs.field(
        kw_only=True
    )  # irrelevant since we are writing our own init method, but attrs complains otherwise

    def __init__(self, other: Dataset):
        super().__init__()
        self.other = other

    def __call__(self, ref: Dataset, ds: Dataset, rng=None) -> Dataset:
        assert len(self.other) == len(ref), (len(self.other), len(ref))
        return self.other  # should maybe check this is a valid draw from ds


def chain_excluding(parent: IterativeMatcher, child: IterativeMatcherIn, term_early_at: MatcherIn = False):
    """Matches `child` from `parent`, excluding any intermediate nodes matched by `term_early_at`."""
    if term_early_at is False:
        return parent.chain(child)
    return restrict(parent.chain(IterativeMatcher(child)), term_early_at=term_early_at)


@attrs.define(frozen=True)
class SampledInputs:
    """Holds the sampled inputs at each node of an interpretation graph."""

    datasets: dict[InterpNode, Dataset] = attrs.field(factory=dict)
    other_inputs_datasets: dict[InterpNode, Dataset] = attrs.field(factory=dict)
    sampler_pools: PoolsPerSampler = attrs.field(factory=dict)

    def __getitem__(self, node: InterpNode) -> Dataset:
        if node.is_leaf():
            return self.datasets[node]
        else:
            return self.other_inputs_datasets[node]

    def get(self, node: InterpNode, default: Optional[Dataset] = None) -> Optional[Dataset]:
        try:
            return self[node]
        except KeyError:
            return default


class InterpNodeGeneralFunctionSpec(GeneralFunctionSpecBase):
    def __init__(self, function, name):
        self.function = function
        self.name = name

    def name(self):
        return self.name

    def function(self, *tensors):
        return self.function(*tensors)

    def get_shape_info(self, *shapes):
        # empty shape
        return GeneralFunctionShapeInfo((), 0, [])


class InterpNode:
    """
    Interpretation graph (tree!) node.

    It doesn't have to be a tree in general, but we are lazy and haven't written treeification code for it.

    cond_sampler: samples data that this node is "indifferent between", e.g.
        data agreeing on a feature with a reference datum if this node only cares about that feature
    name: should be unique within an interpretation. it's nice to be able to uniquely identify things
        by name rather than by more opaque id.
    chidren: interpretation graph nodes that are children of this one. this is used for recursive sampling
        when causal scrubbing, but not much else. it's also kind of nice to think of your interpretation
        as being a graph.
    other_inputs_sampler: formally, causal scrubbing hypotheses should have graph isomorphisms between
        the interpretation and the graph you are trying to explain. for convenience, you can instead specify
        how inputs into the image of this node should be sampled, if not otherwise stated. this is equivalent
        to adding children to this node to make it surjective, and using this sampler as the cond_sampler for
        all of those. not applicable for leaves. (for example, you could have your interpretation just consist
        of the circuit you think is important, and say that other inputs should be sampled randomly).
    """

    def __init__(self, cond_sampler: CondSampler, name: str, other_inputs_sampler: CondSampler = UncondSampler()):
        self._name = name
        self.cond_sampler = cond_sampler
        self.other_inputs_sampler = other_inputs_sampler
        self._children: Tuple["InterpNode", ...] = tuple()

    @property
    def name(self) -> str:
        return self._name

    @property
    def children(self) -> Tuple["InterpNode", ...]:
        return self._children

    def make_descendant(
        self,
        cond_sampler: CondSampler,
        name: str,
        other_inputs_sampler: CondSampler = UncondSampler(),
    ) -> "InterpNode":
        """Create a child node and add it to this node's children."""
        d = self.__class__(name=name, cond_sampler=cond_sampler, other_inputs_sampler=other_inputs_sampler)
        self._children = self.children + (d,)
        return d

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_tree_size(self) -> int:
        """Gets the number of nodes in the interp graph tree rooted at this node."""
        return 1 + sum([c.get_tree_size() for c in self.children])

    def get_nodes(self: "InterpNode") -> List["InterpNode"]:
        return [
            self,
        ] + [n for c in self.children for n in c.get_nodes()]

    def get_descendants_with_parents(self: "InterpNode") -> List[Tuple["InterpNode", "InterpNode"]]:
        return [(self, c) for c in self.children] + [
            pair for c in self.children for pair in c.get_descendants_with_parents()
        ]

    def print(
        self,
        options: Optional[Union[PrintOptions, PrintHtmlOptions]] = None,
        sampled_inputs: SampledInputs = SampledInputs(),
        print_data=True,
        color_by_data=True,
        repr=False,
        datum_idx=0,
    ):
        """
        Parameters:
            print_data: if True, prints a sample datum from cond_sampler as a comment
            color_by_data: if True, colors nodes by ds
            repr: if True, returns string instead of printing it (used for testing)
        """

        def to_circuit(i: InterpNode) -> Tuple[Circuit, Dict[Circuit, "InterpNode"]]:
            node_circuit_map = {}
            child_circuits = []
            for child in i.children:
                child_circuit, child_map = to_circuit(child)
                for k, v in child_map.items():
                    node_circuit_map[k] = v
                child_circuits.append(child_circuit)

            spec = InterpNodeGeneralFunctionSpec(lambda x: x, "")
            this_circuit = GeneralFunction(*child_circuits, spec=spec, name=i.name)
            node_circuit_map[this_circuit] = i
            return this_circuit, node_circuit_map

        c, map = to_circuit(self)

        if options is None:
            new_options: Union[PrintOptions, PrintHtmlOptions] = PrintHtmlOptions(traversal=new_traversal())
        else:
            new_options = options.evolve()
        if print_data:
            new_options.commenters += [
                lambda c: map[c].str_samplers(sampled_inputs=sampled_inputs, datum_idx=datum_idx)
            ]
        if isinstance(new_options, PrintOptions):
            new_options.bijection = False
        if color_by_data:
            new_options.colorer = lambda c: color_dataset(  # type: ignore
                sampled_inputs.datasets.get(map[c]),
                html=isinstance(new_options, PrintHtmlOptions),
            )

        if repr:
            assert isinstance(new_options, PrintOptions)
            return c.repr(new_options)
        else:
            c.print(new_options)

    def _sample_into(
        self,
        rng,
        source_ds,
        parent_ds,
        into: SampledInputs,
        recursive: bool = True,
    ):
        # The one place that mutating SampledInputs is allowed
        ds = self.cond_sampler.sample_and_update_pool(parent_ds, source_ds, rng, into.sampler_pools)
        if not self.is_leaf():
            into.other_inputs_datasets[self] = self.other_inputs_sampler.sample_and_update_pool(
                ds, source_ds, rng, into.sampler_pools
            )
        into.datasets[self] = ds
        if recursive:
            for child in self.children:
                child._sample_into(rng, source_ds, ds, into)

    def sample(
        self,
        rng,
        source_ds,
        parent_ds,
        recursive=True,
    ) -> SampledInputs:
        into = SampledInputs()
        self._sample_into(rng, source_ds, parent_ds, into, recursive=recursive)
        return into

    def str_samplers(self, sampled_inputs: SampledInputs, datum_idx=0) -> str:
        cond_sampler_str = self.cond_sampler.pretty_str(sampled_inputs.datasets.get(self), datum_idx=datum_idx)
        samplers_str = f"cond_sampler={cond_sampler_str}"
        if not self.is_leaf():
            other_sampler_str = self.other_inputs_sampler.pretty_str(
                sampled_inputs.other_inputs_datasets.get(self), datum_idx=datum_idx
            )
            samplers_str += f", other_inputs_sampler={other_sampler_str}"
        return samplers_str

    def pretty_str(self, sampled_inputs: SampledInputs) -> str:
        return f"InterpNode(name={self.name}, {self.str_samplers(sampled_inputs)}, children={[c.name for c in self.children]})"

    def __str__(self):
        return self.pretty_str(SampledInputs())

    def __repr__(self):
        # Not the best repr; but this makes errors nicer for pyo stuff (which will repr rather than str)
        return str(self)


corr_root_matcher = restrict(new_traversal(), term_if_matches=True)


class IllegalCorrespondenceError(Exception):
    ...


DatasetOrInputNames = Union[Dataset, Set[str]]


def get_input_names(ds: DatasetOrInputNames) -> Set[str]:
    return ds.input_names if isinstance(ds, Dataset) else ds


def to_inputs(matcher: IterativeMatcher, ds: DatasetOrInputNames) -> IterativeMatcher:
    return matcher.chain(get_input_names(ds))


class Correspondence:
    """
    Holds correspondences between interp graph and model.

    The parts of the model are pointed at by IterativeMatchers picking out subgraphs of the model (with a single source and sink).
    In theory land we imagine that our correspondences are surjective, but in practice this is painful to implement: you either
    need drastic circuit rewrites, or very many InterpNodes and entries in your correspondence. To make life easier, we allow
    correspondences to model "branches" picked out by matchers.

    corr: mapping from interpretation to branches in the model
    """

    corr: Dict[InterpNode, IterativeMatcher]
    i_names: Dict[str, InterpNode]

    def __init__(self):
        self.corr = {}
        self.i_names = {}

    def copy(self):
        """Shallow copy of a Correspondence. The InterpNodes and matchers are attached to a new Correspondence, but
        not themselves copied."""
        new = Correspondence()
        new.corr = self.corr.copy()  # type: ignore
        new.i_names = self.i_names.copy()
        return new

    def __len__(self):
        return len(self.corr)

    def __getitem__(self, i: InterpNode) -> IterativeMatcher:
        return self.corr[i]

    def get_by_name(self, s: str) -> InterpNode:
        return self.i_names[s]

    def add(self, i_node: InterpNode, m_node: IterativeMatcher):
        if i_node in self.corr:
            raise ValueError("Node already in correspondence!", i_node, m_node)
        if i_node.name in self.i_names:
            raise ValueError(
                "Different node with same name already in correspondence!", self.i_names[i_node.name], i_node, m_node
            )
        self.corr[i_node] = m_node
        self.i_names[i_node.name] = i_node

    def replace(self, i_node: InterpNode, m_node: IterativeMatcher):
        if i_node not in self.corr:
            raise ValueError("Node not found in correspondence!", i_node, m_node)
        assert i_node.name in self.i_names, (
            "Node in corr but its name is not in names map, should never happen!",
            i_node,
            m_node,
        )
        self.corr[i_node] = m_node
        self.i_names[i_node.name] = i_node

    def get_root(self) -> InterpNode:
        if len(self) == 0:
            raise IllegalCorrespondenceError("Empty correspondence has no root!")

        i_nodes = list(self.corr.keys())

        # store parent counts and check that exactly one is 0
        parent_counts = {i: 0 for i in i_nodes}  # this doesn't implement hash or eq but we can just id it
        for i_node in i_nodes:
            for child in i_node.children:
                parent_counts[child] += 1
        roots = [i for i, count in parent_counts.items() if count == 0]
        if len(roots) > 1:
            raise IllegalCorrespondenceError("Found multiple roots!", roots)
        if len(roots) == 0:
            raise IllegalCorrespondenceError("Found no root, not a tree! (No node is an ancestor of all other nodes)")

        return roots[0]

    def in_dfs_order(self) -> List[Tuple[InterpNode, IterativeMatcher]]:
        if len(self) == 0:
            return []
        return [(i_node, self[i_node]) for i_node in self.get_root().get_nodes()]

    def in_dfs_order_with_parents(self) -> List[Tuple[Optional[InterpNode], InterpNode, IterativeMatcher]]:
        if len(self) == 0:
            return []
        root = self.get_root()
        root_w_parent: List[Tuple[Optional[InterpNode], InterpNode, IterativeMatcher]] = [(None, root, self[root])]
        return root_w_parent + [
            (i_node, i_child, self[i_child]) for (i_node, i_child) in root.get_descendants_with_parents()
        ]

    def check_complete(self):
        for i_node in self.corr.keys():
            children = (
                i_node.children
            )  # this used to be i_node.get_nodes (all descendents) but just .children is enough
            for d in children:
                if d not in self.corr:
                    raise IllegalCorrespondenceError(
                        "Node is a descendent of a node in the corr but is not in the corr itself",
                        d,
                        i_node,
                    )

    def check_well_defined(self, circuit: Circuit, ds: Optional[DatasetOrInputNames] = None):
        """
        Checks that each interp graph node matches exactly one circuit node.
        This may fail if e.g. your circuit has some "basically the same" nodes, like one with an extra 1 dim, but generally should pass.
        Includes checking the implicit input nodes exist (by extending matchers to the inputs).
        """
        for m_node in self.corr.values():
            try:
                m_node.get_unique(circuit)
            except RuntimeError:
                raise IllegalCorrespondenceError(
                    f"matcher {m_node} failed well defined check. This check is done in dfs order so all parents must have passed."
                )
            if ds is not None:
                m_to_inputs_matcher = to_inputs(m_node, ds)
                if not m_to_inputs_matcher.are_any_found(circuit):
                    raise IllegalCorrespondenceError(
                        f"matcher extended to inputs {m_to_inputs_matcher} failed existence check. This check is done in dfs order so all parents must have passed."
                    )

    def check_injective_on_treeified(self, circuit: Circuit):
        """
        Checks that every circuit node is matched by at most one interp graph node.
        We expect this to be true after the circuit is tree-ified but not before
        (and maybe not after we replace the treeified inputs, as some of them may
        turn out to be identical).
        """
        if len(self) == 0:
            return
        all_matched = self.get_all_matched(circuit)
        multiple_matches = [(c, i) for (c, i) in Counter(all_matched).items() if i > 1]
        if len(multiple_matches) != 0:
            raise IllegalCorrespondenceError(multiple_matches)

    def check_root_is_model(self):
        root_branch = self.corr[self.get_root()]
        if root_branch != corr_root_matcher:
            raise IllegalCorrespondenceError("the root of the model branch should be empty")

    def get_all_matched(self, circuit: Circuit) -> List[Circuit]:
        """Gets all circuit nodes matched by interp graph nodes."""
        return [c for m_node in self.corr.values() for c in m_node.get(circuit)]

    def check(self, circuit: Circuit, circuit_treeified=True):
        self.check_complete()
        self.check_root_is_model()
        if circuit_treeified:
            self.check_injective_on_treeified(circuit)

    def sample(self, source_ds, rng, ref_ds) -> SampledInputs:
        return self.get_root().sample(source_ds=source_ds, parent_ds=ref_ds, rng=rng)

    def __repr__(self):
        return f"{self.__class__.__name__}(corr={repr(self.corr)})"
