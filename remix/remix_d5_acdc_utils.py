import typing
import wandb
import networkx as nx
import warnings
from collections import OrderedDict, deque
from copy import deepcopy
from typing import (
    Literal,
    Any,
    Callable,
    Deque,
    Dict,
    Iterable,
    List,
    Optional,
    OrderedDict,
    Set,
    Type,
    Union,
    Tuple,
)

import graphviz  # type: ignore
import IPython
import numpy as np
import rust_circuit as rc
import torch
from interp.circuit.causal_scrubbing.dataset import Dataset
from interp.circuit.causal_scrubbing.experiment import (
    Experiment,
    ExperimentCheck,
    ExperimentEvalSettings,
    ScrubbedExperiment,
)
from interp.circuit.causal_scrubbing.hypothesis import (
    CondSampler,
    Correspondence,
    ExactSampler,
    FixedToDatasetSampler,
    InterpNode,
    SampledInputs,
    UncondSampler,
    chain_excluding,
    corr_root_matcher,
)
from interp.circuit.circuit_model_rewrites import MLPHeadAndPosSpec
from interp.circuit.circuit_models import circuit_gpt_and_tok_embeds
from interp.circuit.get_update_node import (
    FunctionIterativeNodeMatcher,
    NameMatcher,
    NodeUpdater,
)
from interp.circuit.interop_rust.interop_rust import py_to_rust
from interp.circuit.projects.gpt2_gen_induction.counting_dataset import (
    incrementing_numbers,
    logit_diff,
)
from interp.circuit.projects.gpt2_gen_induction.rust_path_patching import (
    any_prefix_matcher,
    make_arr,
)
from interp.model.gpt_model import Gpt
from matplotlib import pyplot as plt
from rust_circuit import Add, Array, Circuit, IterativeMatcher
from tqdm import tqdm
from transformers import AutoTokenizer


class ACDCInterpNode(InterpNode):
    """In ACDC, we'll need to add and remove multiple parents and children.

    Tidbit: also, by default both samplers are FixedToDataset, because leaves and nodes through the tree will
    need to be sampled on the 'correct' distribution; in ACDCCorrespondence.expand_node, we
    will change the other_inputs_sampler

    PARAMS:
    - `name`: *unique* name for this node in the correspondence
    """

    def __init__(
        self,
        name: str,
        cond_sampler: CondSampler = ExactSampler(),
        other_inputs_sampler: CondSampler = FixedToDatasetSampler(),
        is_root: bool = False,
    ):
        super().__init__(cond_sampler, name, other_inputs_sampler)
        self.parents: Set[ACDCInterpNode] = set()
        self.is_root = is_root

        # these track the effect sizes of patching edges as we do ACDC
        self.outgoing_effect_sizes: Dict[str, float] = {}
        # incoming effect sizes are Optional, and so that we can track
        # which nodes we are yet to process (incoming_effect_sizes is None)
        # and which nodes we have processed (incoming_effect_sizes is a dict)
        self.incoming_effect_sizes: Optional[Dict[str, float]] = None

        # sometimes we end up removing all inputs from a node; this is a flag to track this
        self.is_redundant = False

    def pretty_str(self, sampled_inputs: SampledInputs) -> str:
        return f"ACDCInterpNode(name={self.name}, {self.str_samplers(sampled_inputs)}, children={[c.name for c in self.children]})"

    def add_parent(self, parent: "ACDCInterpNode"):
        assert parent not in self.parents, (parent, "you shouldn't add parents twice")
        self.parents.add(parent)

    def add_child(self, child: "ACDCInterpNode"):
        if child not in self._children:
            self._children += (child,)

    def get_nodes(self):
        """Since ACDCInterpNodes may not form a tree, we need a new implementation"""
        nodes = [self]
        idx = 0
        while idx < len(nodes):
            children = nodes[idx].children
            for child in children:
                if child not in nodes:
                    nodes.append(child)
            idx += 1
        return nodes


def show_node_name(node: ACDCInterpNode) -> str:
    return node.name + ("_but_patched" if node.is_redundant else "")

def match_nodes_except(
    excepted_names: Iterable[str], all_the_names: Set[str]
) -> rc.Matcher:
    """Used to ensure that we only match nodes on the paths in the DAG that we have defined"""
    assert set(excepted_names).issubset(
        all_the_names
    ), f"{set(excepted_names) - all_the_names}"
    these_names = all_the_names - set(excepted_names)
    return any_prefix_matcher(these_names, dot_after_name=False)


class ACDCCorrespondence(Correspondence):
    """Represents the ACDC Hypothesis we are currently working with.
    You should probably not call the private methods (prefixed by underscore)
    but use the methods of ACDCExperiment to do this
    (because probably you want to edit the ACDCExperiment._base_circuit's state, too)

    When adding new attributes, ensure you add these to the custom deepcopy method"""

    corr: OrderedDict[InterpNode, IterativeMatcher]
    # OrderedDict so that we can topologically sort
    i_names: Dict[str, InterpNode]
    all_names: Set[str]
    topologically_sorted: bool

    def __init__(
        self, all_names: Set[str], topologically_sorted: bool = False
    ):
        self.corr = OrderedDict() 
        self.i_names = {} # super.__init__() just makes these two things (and we use them with different types) so don't call it
        self.all_names = all_names
        self.topologically_sorted = topologically_sorted
        self.connection_strengths = {}
        self.path_lengths = []
        self.connection_strengths_for_path_lengths = []

    def get_root(self):
        return typing.cast(ACDCInterpNode, list(self.corr.keys())[0])

    def remove_node(self, node: ACDCInterpNode):
        self.corr.pop(node)
        self.i_names.pop(node.name)

    def _add_connection(self, parent_node: ACDCInterpNode, child_node: ACDCInterpNode):
        assert parent_node in self.corr, f"{parent_node} not in {self.corr.keys()}"
        assert child_node in self.corr
        child_node.add_parent(parent_node)
        parent_node.add_child(child_node)
        self.replace(parent_node, self.get_matcher(parent_node))
        self.replace(child_node, self.get_matcher(child_node))

    def _remove_connection(
        self, parent_node: ACDCInterpNode, child_node: ACDCInterpNode
    ):
        child_node.parents.remove(parent_node)

        def remove_from_tuple(t, x):
            assert x in t
            ans = tuple([y for y in t if y != x])
            return ans

        parent_node._children = remove_from_tuple(parent_node._children, child_node)

    def get_matcher(
        self,
        node: ACDCInterpNode,
        check_in_corr=True,
        parent_subset: Optional[Set[ACDCInterpNode]] = None,
        recursive: bool = True,
        update: bool = True,
    ) -> IterativeMatcher:
        """Get the matcher for a node,
        and update it in the process.
        if `recursive` is False, this assumes the parents' matchers are correct"""
        if check_in_corr:
            assert node in self.corr, f"{node} not in {self.corr.keys()}"
        if node.is_root:
            return corr_root_matcher
        if parent_subset is None:
            parent_subset = node.parents
        new_matcher = rc.IterativeMatcher.any(
            *[
                chain_excluding(
                    (self.get_matcher(parent) if recursive else self.corr[parent]),
                    child=node.name,
                    term_early_at=self.all_names
                    - {
                        parent.name
                    },  # TODO it seems things work fine if we also take - {node.name} here too. Work out why these are equivalent
                )
                for parent in parent_subset
            ],
        )
        if update:
            self.corr[node] = new_matcher
        return new_matcher

    def add_with_auto_matcher(self, i_node: ACDCInterpNode):
        warnings.warn(
            "add_with_auto_matcher is deprecated and will be deleted, use .add instead"
        )
        self.add(i_node)

    def add(self, i_node: ACDCInterpNode, m_node: Optional[rc.IterativeMatcher] = None):
        """Since in ACDC all correspondences are computational subgraphs,
        matchers should be inferrable from the parents' matchers (and all_names)
        alone"""
        if m_node is not None:
            warnings.warn(
                "You shouldn't use ACDCCorrespondence.add with a matcher argument, it should handle all defaults"
            )
            super().add(i_node, m_node)
            return

        if len(self.corr) == 0:
            assert i_node.is_root, "First node must be root"
        assert i_node not in self.corr, f"{i_node} already in {self.corr.keys()}"
        assert (
            i_node.name not in self.i_names
        ), f"{i_node.name} already in {self.i_names.keys()}"
        assert i_node.name in self.all_names, f"{i_node.name} not in {self.all_names}"
        matcher = self.get_matcher(
            i_node, check_in_corr=False, recursive=False, update=False
        )
        super().add(i_node, matcher)

    def generate_random_color(self, colorscheme: str) -> str:
        """
        https://stackoverflow.com/questions/28999287/generate-random-colors-rgb
        """
        import cmapy
        import random

        def rgb2hex(rgb):
            """
            https://stackoverflow.com/questions/3380726/converting-an-rgb-color-tuple-to-a-hexidecimal-string
            """
            return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

        return rgb2hex(cmapy.color("Pastel2", random.randrange(0, 256), rgb_order=True))

    def build_colorscheme(self, colorscheme: str) -> Dict[str, str]:
        colors = {}
        for child in self.corr:
            colors[str(child.name)] = self.generate_random_color(colorscheme)
        return colors

    def show(
        self, fname=None, colorscheme: str = "Pastel2", minimum_penwidth: float = 0.3
    ):
        """
        takes matplotlib colormaps
        """
        g = graphviz.Digraph(format="png")

        colors = self.build_colorscheme(colorscheme)

        # create all nodes
        for index, child in enumerate(self.corr):
            parent: ACDCInterpNode
            child = typing.cast(ACDCInterpNode, child)
            if len(child.parents) > 0 or index == 0:
                g.node(
                    show_node_name(child),
                    fillcolor=colors[child.name],
                    style="filled, rounded",
                    shape="box",
                    fontname="Helvetica",
                )

        # connect the nodes
        for child in self.corr:
            parent: ACDCInterpNode
            child = typing.cast(ACDCInterpNode, child)
            for parent in child.parents:
                penwidth = self.get_connection_strengths(
                    parent, child, minimum_penwidth
                )
                g.edge(
                    show_node_name(child),
                    show_node_name(parent),
                    penwidth=penwidth,
                    color=colors[child.name],
                )

        if fname is not None:
            assert fname.endswith(
                ".png"
            ), "Must save as png (... or you can take this g object and read the graphviz docs)"
            g.render(outfile=fname, format="png")
        return g

    def get_connection_strengths(
        self,
        parent: ACDCInterpNode,
        child: ACDCInterpNode,
        minimum_penwidth: float = 0.1,
    ) -> str:
        potential_key = str(parent.name) + "->" + str(child.name)
        if potential_key in self.connection_strengths.keys():
            penwidth = self.connection_strengths[potential_key]
            list_of_connection_strengths = [
                i for i in self.connection_strengths.values()
            ]
            penwidth = (
                10
                * (penwidth - min(list_of_connection_strengths))
                / (
                    1e-5
                    + (
                        max(list_of_connection_strengths)
                        - min(list_of_connection_strengths)
                    )
                )
            )
            if penwidth < minimum_penwidth:
                penwidth = minimum_penwidth
        else:
            warnings.warn("A potential key is not in connection strength keys") # f"{potential_key} not in f{self.connection_strengths.keys()}")
            penwidth = 1
        return str(float(penwidth))

    def build_networkx_graph(self):
        # create all nodes
        import networkx as nx

        g = nx.DiGraph()

        for index, child in enumerate(self.corr):
            parent: ACDCInterpNode
            comp_type: str
            child = typing.cast(ACDCInterpNode, child)
            if len(child.parents) > 0 or index == 0:
                g.add_node(child.name)

        # connect the nodes
        for child in self.corr:
            parent: ACDCInterpNode
            child = typing.cast(ACDCInterpNode, child)
            for parent in child.parents:
                g.add_edge(
                    child.name, parent.name,
                )
        return g

    def extract_connection_strengths(self, a_node):
        connection_strengths = []
        for a_key in self.connection_strengths.keys():
            if a_node in a_key.split("->")[1]:
                connection_strengths.append(self.connection_strengths[a_key])
        return connection_strengths

    def compute_path_length_stats(
        self,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        networkx_graph = self.build_networkx_graph()
        output_node = self.get_root().name
        for a_node in networkx_graph:
            if a_node == output_node:
                continue
            if len(self.connection_strengths.keys()) == 0:
                continue
            distance = len(
                nx.shortest_path(networkx_graph, source=a_node, target=output_node)
            )
            extracted_connection_strengths = self.extract_connection_strengths(a_node)
            for i in extracted_connection_strengths:
                self.connection_strengths_for_path_lengths.append(i)
            self.path_lengths += [distance] * len(extracted_connection_strengths)
        return self.get_path_length_stats()

    def get_path_length_stats(
        self,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        log_dict = {}
        for i, a_distance in enumerate(self.path_lengths):
            if a_distance not in log_dict.keys():
                log_dict[a_distance] = [self.connection_strengths_for_path_lengths[i]]
            else:
                log_dict[a_distance].append(
                    self.connection_strengths_for_path_lengths[i]
                )
        x = [i for i in log_dict.keys()]
        average_y = [np.mean(log_dict[i]) for i in x]
        min_y = [np.min(log_dict[i]) for i in x]
        max_y = [np.max(log_dict[i]) for i in x]
        return (x, average_y, min_y, max_y)

    def topologically_sort_corr(self):
        """Sorts the nodes in topologically sorted order"""
        if self.topologically_sorted:
            return
        # validate that the graph is a DAG
        network_by_names = nx.DiGraph()
        nodes = list(self.corr.keys())
        for node in nodes:
            for child in node.children:
                network_by_names.add_edge(node.name, child.name)
        assert (
            len(nx.descendants(network_by_names, self.get_root().name))
            == len(self.corr) - 1
        ), "Graph is not a DAG"
        sorted = nx.topological_sort(network_by_names)
        sorted_nodes = [self.i_names[node] for node in sorted]

        # make new corr
        new_corr = OrderedDict()
        for node in sorted_nodes:
            new_corr[node] = self.corr[node]
        self.corr = new_corr
        self.topologically_sorted = True

    def __deepcopy__(self, memo={}):
        """
        Deep copy of an ACDCCorrespondence
        See ACDCExperiment.__deepcopy__ for explanation of why
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        if len(self.__dict__) != 7:
            warnings.warn(
                f"ACDCCorrespondence has more than 7 attributes. __deepcopy__ is probably wrong {self.__dict__.keys()}"
            )

        setattr(result, "corr", {})
        for k, v in self.corr.items():
            result.corr[deepcopy(k, memo)] = v
        setattr(result, "i_names", deepcopy(self.i_names, memo))
        setattr(result, "all_names", deepcopy(self.all_names, memo))
        setattr(result, "topologically_sorted", self.topologically_sorted)
        setattr(result, "path_lengths", deepcopy(self.path_lengths, memo))
        setattr(
            result, "connection_strengths", deepcopy(self.connection_strengths, memo)
        )
        setattr(
            result,
            "connection_strengths_for_path_lengths",
            deepcopy(self.connection_strengths_for_path_lengths, memo),
        )
        return result


class ACDCTemplateCorrespondence(ACDCCorrespondence):
    """Use this to define the MAXIMAL subgraph of the circuit that you want to match (acdc.py)
    It has features removed to hopefully stop this from being used like a normal correspondence."""

    def __init__(self, all_names):
        super().__init__(
            all_names=all_names, topologically_sorted=False
        )

    def _add_connection(self):
        raise NotImplementedError()

    def replace(self, i_node, matcher):
        raise NotImplementedError()

    def sample(self, num_samples):
        raise NotImplementedError()

    def check_all_names(self):
        assert set([node.name for node in self.corr]) == set(
            self.all_names
        ), f"all_names: {self.all_names} not equal to {set([node.name for node in self.corr])}"



class ACDCExperiment(Experiment):
    """Manages runs of ACDC.
    Note that this passes the *patch* dataset as the `dataset` argument to Experiment.__init__.
    See acdc.py for explanation.

    `ds` is the proper dataset, `ref_ds` is the patched dataset.

    Set `parallel_hypotheses` to >1 to enable parallel circuit evaluation."""

    def __init__(
        self,
        circuit: Circuit,
        ds: Dataset,
        ref_ds: Dataset,
        template_corr: ACDCTemplateCorrespondence,
        metric: Callable[[Dataset, torch.Tensor], float],
        num_examples: int = 1,
        random_seed: int = 42,
        check: ExperimentCheck = True,
        group: Optional[Circuit] = None,
        threshold: float = 0.0,
        verbose: bool = False,
        parallel_hypotheses: int = 1,
        corr: Optional[ACDCCorrespondence] = None,
        expand_by_importance: bool = True,
        using_wandb: bool = False,
        remove_redundant: bool = False,
        monotone_metric: Literal["off", "maximize", "minimize"] = "off", # if this is set to "maximize" or "minimize", then the metric will be maximized or minimized, respectively instead of us trying to keep the metric roughly the same 
    ):
        if corr is not None:
            warnings.warn(
                "Passing the correspondence to ACDC is deprecated, this has not effect at the moment"
            )

        self.template_corr = template_corr
        template_corr.topologically_sort_corr()
        if using_wandb:
            self.store_no_upstream_edges() # this tracks
        corr = ACDCCorrespondence(
            all_names=template_corr.all_names,
            topologically_sorted=True,
        )
        for (
            node
        ) in template_corr.corr:  # add all the nodes, but only attach the root node
            corr.add(ACDCInterpNode(name=node.name, is_root=node.is_root),)
        self.current_node = corr.get_root()
        super().__init__(
            circuit=circuit,
            dataset=ref_ds,  # read docstring for this choice
            corr=corr,
            num_examples=num_examples,
            random_seed=random_seed,
            check=check,
            group=group,
        )

        self._default_ds = ds
        self.setup_datasets()
        self._sampled_inputs = SampledInputs(
            datasets={node: self._default_ds for node in self._nodes.corr},
            other_inputs_datasets={node: self._dataset for node in self._nodes.corr},
        )

        self._original_circuit = self._base_circuit
        self.metric = metric
        self.update_cur_metric()

        self.threshold = threshold
        self.verbose = verbose

        self.expand_by_importance = expand_by_importance
        self.remove_redundant = remove_redundant
        self.monotone_metric = monotone_metric

        self.parallel_hypotheses = parallel_hypotheses
        self._circuits: List[rc.Circuit] = []  # backlog for efficient parallelization
        self.using_wandb = using_wandb

        if self.using_wandb:
            self.metrics_to_plot = {}
            self.metrics_to_plot["new_metrics"] = []
            self.metrics_to_plot["list_of_parents_evaluated"] = []
            self.metrics_to_plot["list_of_children_evaluated"] = []
            self.metrics_to_plot["list_of_nodes_evaluated"] = []
            self.metrics_to_plot["evaluated_metrics"] = []
            self.metrics_to_plot["current_metrics"] = []
            self.metrics_to_plot["results"] = []
            self.metrics_to_plot["acdc_step"] = 0
            self.metrics_to_plot["num_edges"] = []

    def setup_datasets(self):
        """It's essential that patch dataset is the same as default dataset, except the objects are tagged"""
        assert set(self._dataset.input_names) == set(self._default_ds.input_names)
        # assert all(["Tag" in str(type(arr)) for arr in self._dataset.arrs.values()])

        patch_ds_keys = set(self._dataset.arrs.keys())
        for key in patch_ds_keys:
            if "Tag" not in str(type(self._dataset.arrs[key])):
                self._dataset.arrs[key] = rc.Tag.new_with_random_uuid(
                    self._dataset.arrs[key], name=key
                )

        assert all(
            ["Tag" not in str(type(arr)) for arr in self._default_ds.arrs.values()]
        )

    def set_current_circuit(self):
        # self.set_sampled_inputs()
        self._base_circuit = self.wrap_in_var(self._base_circuit, self._dataset)
        self._base_circuit = self.replace_inputs(
            self._base_circuit, self._sampled_inputs
        )

    def update_cur_metric(self, sanity_check=False) -> rc.Circuit:
        """Perform a causal scrub (from scratch) to update the metric"""
        self._base_circuit = self._original_circuit
        scrub = self.scrub(ref_ds=self._default_ds, treeify=False)
        self.cur_logits = scrub.evaluate(
            eval_settings=ExperimentEvalSettings(
                optimize=True,
                optim_settings=rc.OptimizationSettings(scheduling_naive=True),
            )
        )
        self.cur_metric = self.metric(self._default_ds, self.cur_logits)
        self.set_current_circuit()

        if sanity_check:
            # update the metric
            sanity_logits = (
                ExperimentEvalSettings(optimize=True)
                .get_sampler(len(self._dataset), self._group)
                .sample(self._base_circuit)
                .evaluate()
            )
            sanity_metric = self.metric(self._default_ds, sanity_logits)
            assert (
                abs(sanity_metric - self.cur_metric) < 1e-5
            ), f"Sanity check failed: {sanity_metric=} vs {self.cur_metric=}"

        return scrub.circuit

    def increment_current_node(self):
        """ACDCCorrespondences have OrderedDicts, so we can just increment the index"""
        assert (
            typing.cast(ACDCInterpNode, self.current_node).incoming_effect_sizes
            is not None
        ), "We've already processed the current node-experiment may be finished"
        node_names = [
            node.name
            for node in self._nodes.corr
            if node.incoming_effect_sizes is None
            and len(node.parents) > 0
            and len(self.template_corr.i_names[node.name].children)
            > 0  # checks: i) this node is in the existing correspondence, ii) this node has not yet been processed, iii) this node has actually has children, so we can refine hypothesis more after this
        ]
        if len(node_names) == 0:
            self.current_node = None
            print("\nNo more nodes to process!")
            return

        if self.expand_by_importance:
            # sort by effect size
            sorted_node_names = sorted(
                node_names,
                key=lambda node_name: sum(
                    abs(effect_size)
                    for effect_size in self._nodes.i_names[
                        node_name
                    ].outgoing_effect_sizes.values()
                ),
                reverse=True,
            )
            self.current_node = self._nodes.i_names[sorted_node_names[0]]

        else:
            self.current_node = self._nodes.i_names[node_names[0]]

        if self.verbose:
            print(
                "Current node has moved onto",
                self.current_node.name if self.current_node is not None else "finished",
            )

    def expand_current_node(self):
        """Expand the current node, adding connections to all the children"""
        assert (
            self.current_node is not None
        ), "Can't expand-are you done with the experiment?"
        assert self.current_node.is_leaf()

        for node in self.template_corr.i_names[self.current_node.name].children:
            self.add_connection(
                parent_node=self.current_node,
                child_node=self._nodes.i_names[node.name],
            )

    def run_eval(self):
        answers = rc.optimize_and_evaluate_many(
            self._circuits, rc.OptimizationSettings(scheduling_naive=True)
        )
        self._circuits = []
        return answers

    def remove_connection(
        self, parent_node: ACDCInterpNode, child_node: ACDCInterpNode
    ):
        """Remove a connection between two nodes (and reflect this in the sampled_inputs and current circuit)"""

        # make a matcher that's specifically on the parent->child path
        parent_to_child_matcher = typing.cast(
            ACDCCorrespondence, self._nodes
        ).get_matcher(
            typing.cast(ACDCInterpNode, child_node),
            parent_subset={
                typing.cast(ACDCInterpNode, parent_node)
            },  # only match through the path *from this parent*
        )
        # then chain this to the default inputs (which are not `rc.Tag`s)
        matcher = parent_to_child_matcher.chain(
            rc.restrict(
                (rc.Matcher(set(self._default_ds.arrs.keys())) & ~rc.Matcher(rc.Tag)),
                term_early_at=rc.Tag,
            )
        )

        # efficiently update `_default_ds` things to `._dataset` (patched) things
        if len(matcher.get(self._base_circuit)) == 0:
            warnings.warn(
                str(
                    (
                        "Removing this connection would have no effect-this is probably a bug",
                        parent_node,
                        child_node,
                        matcher,
                    )
                )
            )
        self._base_circuit = matcher.update(
            self._base_circuit, lambda c: self._dataset.arrs[c.name],
        )
        typing.cast(ACDCCorrespondence, self._nodes)._remove_connection(
            typing.cast(ACDCInterpNode, parent_node),
            typing.cast(ACDCInterpNode, child_node),
        )
        child_descendants = child_node.get_nodes()
        for descendant in child_descendants:
            descendant_matcher = self._nodes.get_matcher(descendant, recursive=False)
            self._nodes.corr[descendant] = descendant_matcher

    def add_connection(self, parent_node: ACDCInterpNode, child_node: ACDCInterpNode):
        """Add a connection between two nodes (and reflect this in the sampled_inputs and current circuit)"""

        if len(parent_node.children) == 0:
            # leaf becomes non-leaf, so patch all inputs
            matcher = self._nodes.get_matcher(parent_node).chain(
                rc.restrict(
                    (
                        rc.Matcher(set(self._default_ds.arrs.keys()))
                        & ~rc.Matcher(rc.Tag)
                    ),
                    term_early_at=rc.Tag,
                )
            )

            def safe_update(c: rc.Circuit):
                assert "Tag" not in str(type(c)), type(c)
                return self._dataset.arrs[c.name]

            self._base_circuit = matcher.update(self._base_circuit, safe_update)

        typing.cast(ACDCCorrespondence, self._nodes)._add_connection(
            parent_node=typing.cast(ACDCInterpNode, parent_node),
            child_node=typing.cast(ACDCInterpNode, child_node),
        )

        child_descendants = set(child_node.get_nodes())
        for descendant in child_descendants:
            descendant_matcher = self._nodes.get_matcher(descendant, recursive=True,)
            self._nodes.corr[descendant] = descendant_matcher
            to_inputs_matcher = descendant_matcher.chain(
                rc.Matcher(set(self._default_ds.input_names)) & rc.Matcher(rc.Tag)
            )
            soft_to_inputs_matcher = descendant_matcher.chain(
                rc.Matcher(set(self._default_ds.input_names))
            )
            if descendant.is_leaf():

                def safe_update(c: rc.Circuit):
                    assert "Tag" in str(type(c)), type(c)
                    return self._default_ds.arrs[c.name]

                assert len(soft_to_inputs_matcher.get(self._base_circuit)) > 0, (
                    parent_node,
                    child_node,
                    descendant,
                    soft_to_inputs_matcher,
                )
                if len(to_inputs_matcher.get(self._base_circuit)) == 0:
                    self.bad_matcher = to_inputs_matcher
                    print(
                        parent_node, child_node, descendant, to_inputs_matcher,
                    )
                    assert False
                self._base_circuit = to_inputs_matcher.update(
                    self._base_circuit, safe_update,
                )

    def check_circuit_conforms(self):
        """Ensure that the `self._base_circuit` conforms to the current correspondence `self._nodes`
        Does this by checking that we reach all the self._default_ds inputs
        through the matchers from leafs, and that all other inputs are self._dataset inputs, (recall these are `rc.Tag`s)"""

        copy_circuit = self._base_circuit

        def rename_defaults(c: rc.Circuit):
            assert "Tag" not in str(type(c)), type(c)
            return c.rename(c.name + "_renamed")

        for node in self._nodes.corr:
            if node.is_leaf():
                matcher = self._nodes.get_matcher(node, recursive=False).chain(
                    self._default_ds.input_names
                )
                copy_circuit = matcher.update(copy_circuit, rename_defaults)
        rc.Matcher(self._default_ds.input_names).get(copy_circuit)
        matched = rc.restrict(
            rc.Matcher(self._default_ds.input_names & ~rc.Matcher(rc.Tag)),
            term_early_at=rc.Tag,
        ).get(copy_circuit)
        assert all(["Tag" in str(type(c)) for c in matched]), matched
        print("Circuit conforms")

    def step(self):
        """Main two steps of ACDC:
        1. Expand the current node (add all its children to the correspondence)
        2. Remove all the unnecessary children"""
        assert (
            self.current_node is not None
        ), "Can't step-are you done with the experiment?"
        if self.verbose:
            print(
                "Working on node", typing.cast(ACDCInterpNode, self.current_node).name
            )
        typing.cast(ACDCInterpNode, self.current_node).incoming_effect_sizes = {}

        self.expand_current_node()
        self.update_cur_metric()
        if self.verbose:
            print("New metric:", self.cur_metric)

        current_node_children = list(self.current_node.children)
        for node_idx, node in tqdm(enumerate(current_node_children)):

            self.remove_connection(
                typing.cast(ACDCInterpNode, self.current_node),
                typing.cast(ACDCInterpNode, node),
            )
            self._circuits.append(
                ExperimentEvalSettings(optimize=True)
                .get_sampler(len(self._dataset), self._group)
                .sample(self._base_circuit)
            )
            self.add_connection(
                typing.cast(ACDCInterpNode, self.current_node),
                typing.cast(ACDCInterpNode, node),
            )
            if (
                len(self._circuits) == self.parallel_hypotheses
                or node_idx == len(current_node_children) - 1
            ):
                # evaluate the circuits we've collected!
                # then remove the connections to unimportant nodes, and keep the other connections
                no_circuits = len(self._circuits)
                evaluated_logits = self.run_eval()
                for evaluating_idx in range(1, no_circuits + 1):
                    evaluated_node = current_node_children[
                        node_idx - no_circuits + evaluating_idx
                    ]
                    evaluated_metric = self.metric(
                        self._default_ds, evaluated_logits[evaluating_idx - 1]
                    )

                    if self.verbose:
                        print(
                            "Metric after removing connection to",
                            evaluated_node.name,
                            "is",
                            evaluated_metric,
                            "(and current metric " + str(self.cur_metric) + ")",
                        )

                    if self.monotone_metric == "off":
                        result = abs(self.cur_metric - evaluated_metric)
                    if self.monotone_metric == "maximize":
                        result = self.cur_metric - evaluated_metric
                    if self.monotone_metric == "minimize":
                        result = evaluated_metric - self.cur_metric

                    if self.verbose:
                        print("Result is", result)
                    self.current_node.incoming_effect_sizes[evaluated_node] = evaluated_node.outgoing_effect_sizes[
                        self.current_node
                    ] = result

                    if result < self.threshold:
                        self.remove_connection(
                            parent_node=typing.cast(ACDCInterpNode, self.current_node),
                            child_node=typing.cast(ACDCInterpNode, evaluated_node),
                        )

                    else:
                        if self.verbose:
                            print("...so keeping connection")
                        self._nodes.connection_strengths[
                            str(self.current_node.name)
                            + "->"
                            + str(evaluated_node.name)
                        ] = abs(result)

                    if self.using_wandb:
                        self.log_metrics_to_wandb(
                            current_metric=self.cur_metric,
                            parent=self.current_node,
                            child=evaluated_node,
                            evaluated_metric=evaluated_metric,
                            result=abs(result),
                        )

                self.update_cur_metric(sanity_check=True)

        if len(self.current_node.children) == 0:
            print(
                f"Warning: we added {self.current_node.name} earlier, but we just removed all its child connections. So we are {('' if self.remove_redundant else 'not')} removing it and its redundant descendants now (remove_redundant={self.remove_redundant})"
            )
            self.current_node.is_redundant = True
            if self.remove_redundant:
                queue = [self.current_node]
                idx = 0
                for idx in range(0, int(1e9)):
                    if idx >= len(queue):
                        break
                    parents = list(queue[idx].parents)
                    for parent in parents:
                        self.remove_connection(
                            parent_node=typing.cast(ACDCInterpNode, parent),
                            child_node=typing.cast(ACDCInterpNode, queue[idx]),
                        )
                        if len(parent.children) == 0 and parent not in queue:
                            parent.is_redundant = True
                            queue.append(parent)

        self.increment_current_node()

    def store_no_upstream_edges(self):
        """Store the number of edges behind each node in the *treeified* template correspondence graph,
        if none of these upstream edges were removed"""

        self.no_upstream_edges: Dict[str, int] = {}
        if not self.template_corr.topologically_sorted:
            self.template_corr.topological_sort()

        # reverse topological sort means this is an easy loop...

        for node in reversed(self.template_corr.corr):
            name = node.name
            self.no_upstream_edges[name] = 0
            for child in node.children:
                self.no_upstream_edges[name] += self.no_upstream_edges[child.name] + 1
        
    def get_no_edges(self):
        """Return the number of edges in the graph
        (INCLUDING all the edges 'hidden' behind the leaf nodes)"""

        cnt = 0
        number_of_paths = {
            self._nodes.get_root().name: 1
        }
        for node in self._nodes.corr:
            if len(node.parents) == 0 and not node.is_root:
                continue # not in the current graph
            if len(node.children) == 0:
                cnt += number_of_paths[node.name] * self.no_upstream_edges[node.name]
            for child in node.children:
                cnt += number_of_paths[node.name]
                if child.name not in number_of_paths:
                    number_of_paths[child.name] = 0
                number_of_paths[child.name] += number_of_paths[node.name]
        return cnt


    def log_metrics_to_wandb(
        self,
        current_metric: float,
        parent: ACDCInterpNode,
        child: ACDCInterpNode,
        evaluated_metric: float,
        result: float,
    ) -> None:

        self.metrics_to_plot["new_metrics"].append(self.cur_metric)
        self.metrics_to_plot["list_of_nodes_evaluated"].append(
            typing.cast(ACDCInterpNode, self.current_node).name
        )
        self.metrics_to_plot["list_of_parents_evaluated"].append(parent.name)
        self.metrics_to_plot["list_of_children_evaluated"].append(child.name)
        self.metrics_to_plot["evaluated_metrics"].append(evaluated_metric)
        self.metrics_to_plot["current_metrics"].append(evaluated_metric)
        self.metrics_to_plot["results"].append(result)
        self.metrics_to_plot["num_edges"].append(self.get_no_edges())

        self.metrics_to_plot["acdc_step"] += 1
        list_of_timesteps = [i + 1 for i in range(self.metrics_to_plot["acdc_step"])]
        if self.metrics_to_plot["acdc_step"] > 1:
            self.do_plotly_plot_and_log(
                x=list_of_timesteps,
                y=self.metrics_to_plot["results"],
                metadata=[f"{parent_string} to {child_string}" for parent_string, child_string in zip(self.metrics_to_plot["list_of_parents_evaluated"], self.metrics_to_plot["list_of_children_evaluated"])],
                plot_name="results",
            )
            self.do_plotly_plot_and_log(
                x=list_of_timesteps,
                y=self.metrics_to_plot["evaluated_metrics"],
                metadata=self.metrics_to_plot["list_of_nodes_evaluated"],
                plot_name="evaluated_metrics",
            )
            self.do_plotly_plot_and_log(
                x=list_of_timesteps,
                y=self.metrics_to_plot["current_metrics"],
                metadata=self.metrics_to_plot["list_of_nodes_evaluated"],
                plot_name="evaluated_metrics",
            )

            # this usually looks best with an x scale logged, so plot this using normal wandb charting stuff
            wandb.log({"num_edges_total": self.metrics_to_plot["num_edges"][-1]})

            # some path related stuff, too
            (
                path_lengths,
                average_strengths,
                min_strengths,
                max_strengths,
            ) = self._nodes.compute_path_length_stats()
            self.do_plotly_plot_and_log(
                x=path_lengths,
                y=average_strengths,
                plot_name="path_vs_average_connection_strengths",
            )
            self.do_plotly_plot_and_log(
                x=path_lengths,
                y=min_strengths,
                plot_name="path_vs_min_connection_strengths",
            )
            self.do_plotly_plot_and_log(
                x=path_lengths,
                y=max_strengths,
                plot_name="path_vs_max_connection_strengths",
            )

    def do_plotly_plot_and_log(
        self, x: List[int], y: List[float], plot_name: str, metadata: List[str] = None,
    ) -> None:
        import wandb
        import plotly.graph_objects as go

        # Create a plotly plot with metadata
        fig = go.Figure(
            data=[go.Scatter(x=x, y=y, mode="lines+markers", text=metadata)]
        )
        wandb.log({plot_name: fig})

    def __deepcopy__(self, memo, object_to_copy=None):
        """
        Deepcopy the ACDCExperiment object (useful for checkpointing experiments).
        We can't deepcopy rust_circuit objects, so we just use the same object.
        This works as everything in rust_circuit is immutable.
        """

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        d = self.__dict__

        if object_to_copy is not None:
            d = object_to_copy.__dict__

        print("Deepcopying all attributes", end=" ")
        for k, v in d.items():
            print(", ", k, end="")
            if k in [
                "_base_circuit",
                "_group",
                "_dataset",
                "_default_ds",
                "_rng",
                "_old_circuit",
                "eval_settings",
                "sampler",
                "_original_circuit",
            ]:
                setattr(result, k, v)
            elif k == "_circuits":
                setattr(result, k, [c for c in v])
            elif k == "_sampled_inputs":
                setattr(result, k, SampledInputs({}, {}))
                print("(doing sampled inputs correctly...", end="")
                for k, v in d["_sampled_inputs"].datasets.items():
                    relevant_nodes = [
                        node for node in result._nodes.corr if node.name == k.name
                    ]
                    assert len(relevant_nodes) == 1, relevant_nodes
                    node = relevant_nodes[0]
                    result._sampled_inputs.datasets[node] = d["_default_ds"]
                    result._sampled_inputs.other_inputs_datasets[node] = d["_dataset"]
                print(" done)", end="")
            else:
                setattr(result, k, deepcopy(v, memo))

        print(": done.")
        return result
