from __future__ import annotations

from typing import Dict, Literal, Optional, Union
from warnings import warn

import attrs
import torch

import interp.tools.optional as op
import rust_circuit as rc

from .dataset import Dataset, color_dataset
from .hypothesis import Correspondence, IllegalCorrespondenceError, InterpNode, SampledInputs, to_inputs

ExperimentCheck = Union[bool, Literal["fast"]]


class ExperimentEvalSettings:
    """
    Settings for ScrubbedExperiment.evaluate.

    optimize: whether to optimize before evaluating. If true, evaluate uses less memory and more time.
    optim_settings: settings for optimizer. Only used if optimize=True.
    batch_size: if set, reshapes circuit into batches before evaluating, to take less memory and more time.
    run_on_all: if True, runs the scrubbed model on the whole batch of scrubbed inputs in order; see
        rc.DiscreteVarAllSpec for more info. likely you want this to be True. if True, num_samples_should not be set.
    num_samples: number of samples to sample when using rc.RandomSampleSpec, so should not be set if run_on_all=True.
        if run_on_all=False and this is not set, will default to the size of the group passed to get_sampler. likely
        you don't want to set this: it only affects *evaluation of the scrubbed circuit*, and not the scrubbing process.
    device_dtype: device and dtype to evaluate on. if only a str is supplied, this is assumed to be the device.
    """

    optimize: bool
    optim_settings: rc.OptimizationSettings
    batch_size: Optional[int]
    run_on_all: bool  # set to False to sample from scrubbed inputs, else will run on all of them in order
    num_samples: Optional[int]
    device_dtype: rc.TorchDeviceDtypeOp

    def __init__(
        self,
        optimize=True,
        optim_settings=rc.OptimizationSettings(),
        batch_size=None,
        run_on_all=True,
        num_samples=None,
        device_dtype: Union[rc.TorchDeviceDtypeOp, str] = rc.TorchDeviceDtypeOp("cuda:0"),
    ):
        if run_on_all and (num_samples is not None):
            raise ValueError(f"num_samples should be None when run_on_all is True, but was {num_samples}")
        self.optimize = optimize
        self.run_on_all = run_on_all
        self.num_samples = num_samples
        self.batch_size = batch_size
        if isinstance(device_dtype, str):
            device_dtype = rc.TorchDeviceDtypeOp(device_dtype, None)
        self.device_dtype = device_dtype
        if self.optimize:
            self.optim_settings = optim_settings

    def get_sampler(self, group, seed):
        """
        First, creates a sample spec: if run_on_all, this is just a RunDiscreteVarAllSpec targeting the specified
        group; otherwise, a RandomSampleSpec that sets a seed for this group. Note seeds are not set for other groups:
        we don't explicitly support having more than one group.

        Then, creates a sampler that will substitute modules, batch, and cast the circuit after sampling.
        """
        sample_spec: rc.SampleSpecIn
        if self.run_on_all:
            sample_spec = rc.RunDiscreteVarAllSpec([group])
        else:
            sample_spec = rc.RandomSampleSpec(
                seeder=lambda g: seed if g == group else None,
                shape=(op.unwrap_or(self.num_samples, len(group)),),
                probs_and_group_evaluation_settings=rc.OptimizationSettings(),
            )

        maybe_batch = (
            lambda x: x
            if self.batch_size is None
            else rc.batch_to_concat(
                x,
                axis=0,
                batch_size=self.batch_size,
            )
        )

        return rc.Sampler(
            sample_spec,
            run_on_sampled=lambda x: rc.cast_circuit(maybe_batch(rc.substitute_all_modules(x)), self.device_dtype),
        )


class Experiment:
    """
    A causal scrubbing experiment.

    The circuit you pass in can be the *complete* circuit to scrub--that is, computing the full behavior function you want
    the scrubbed expectation of, or it can be a subset. For example, this can be convenient if you want to pass in a circuit
    representing a neural network, and compute its scrubbed loss yourself.

    circuit: circuit to be scrubbed. assumes it has inputs matching dataset.input_names with no batch dimension.
    dataset: dataset to both use as reference and sample from for scrubbing. that is, we run causal scrubbing on a batch.
    corr: hypothesized correspondence.
    random_seed: seed for RNG for replicability. will be used 1) to set up a torch generator that is consumed every time we
        need to sample inputs, and 2) for eval if you use ExperimentEvalSettings.run_on_all = False.
    check: whether to check the correspondence for validity. recommend setting it to True, switching to "fast" if your code
        is frustratingly slow; and back to True if you are having confusing errors or results.
    """

    def __init__(
        self,
        circuit: rc.Circuit,
        dataset: Dataset,
        corr: Correspondence,
        random_seed=42,
        check: ExperimentCheck = True,
    ):
        self._base_circuit: rc.Circuit = circuit
        self._dataset = dataset
        self._check = check

        if self._check is not True:
            warn(f"You're using check = {self._check}, your correspondence may have issues that won't be caught!")
        if self._check:
            corr.check(self._base_circuit, circuit_treeified=False)

        self._rng = torch.Generator(device=circuit.device)
        self._rng.manual_seed(random_seed)
        self._seed = random_seed
        self._nodes = corr

    @property
    def base_circuit(self) -> rc.Circuit:
        return self._base_circuit

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    @property
    def nodes(self) -> Correspondence:
        return self._nodes

    def make_ref_ds(self, num_examples: int):
        return self._dataset.sample(num_examples, self._rng)

    def scrub(
        self,
        ref_ds_or_num: Union[Dataset, int],
        group: Optional[rc.Circuit] = None,
        treeify: bool = True,
        check: ExperimentCheck = True,
    ) -> ScrubbedExperiment:
        """
        Performs the _project_ operation of causal scrubbing, except for the actual evaluation.
        Call .evaluate() on the returned ScrubbedExperiment to do that.

        ref_ds: can be used to set the reference dataset to a particular value. If an int is specified, randomly samples
            that size reference dataset from self.dataset, which is normally what you should do when causal scrubbing.
            You might set ref_ds if you want to examine a particular subset of your dataset, however.
        group: the probs_and_group to use for wrapping your inputs in var, and for sampling. Note that the probs
            are ignored if you end up evaluating with run_on_all = True. Must match len(ref_ds) aka num_examples.
        check: whether to check the correspondence for validity
        treeify: will perform an explicit treeification and extra checks
        """
        check = bool(self._check if check is None else check)
        if isinstance(ref_ds_or_num, int):
            num_examples = ref_ds_or_num
            ref_ds = self.make_ref_ds(num_examples)
        else:
            ref_ds = ref_ds_or_num
            num_examples = len(ref_ds)

        if group is None:
            group = rc.DiscreteVar.uniform_probs_and_group(num_examples)
        else:
            if group.shape != (num_examples,):
                raise ValueError(f"group has shape {group.shape=}, should have shape={(num_examples,)}")

        sampled_inputs = self.sample(ref_ds=ref_ds)
        if treeify:
            circuit = self.treeified(check)
            if check:
                self._nodes.check_injective_on_treeified(circuit)
        else:
            circuit = self._base_circuit
        circuit = self.wrap_in_var(circuit, ref_ds, group)
        circuit = self.replace_inputs(circuit, sampled_inputs)
        return ScrubbedExperiment(
            circuit, ref_ds=ref_ds, sampled_inputs=sampled_inputs, group=group, nodes=self._nodes, seed=self._seed
        )

    def sample(self, ref_ds: Dataset) -> SampledInputs:
        """Samples from the interpretation with the given reference dataset"""
        return self._nodes.sample(self._dataset, self._rng, ref_ds)

    def treeified(self, check: ExperimentCheck = True):
        """
        Treeifies the circuit as much as is required by the correspondence, and checks corresponcence
        validity along the way.

        Note this does not depend at all on the interpretation, only on what parts of the circuit
        are mapped! and may be moved into ModelBranch in the future.

        This step is optional: it can be nice to examine your treeified model without setting
        scrubbed inputs, if you skip it then setting scrubbed inputs will treeify the model
        as a side effect.

        This is an algrebraic rewrite!
        """

        def replacement(i_node: InterpNode, a: rc.Circuit) -> rc.Tag:
            input_name = a.name

            if i_node != self._nodes.get_root():
                # Not the root, so inputs upstream of this should have already been tagged
                a = a.cast_tag().node
                if check:  # maybe don't run if check is "fast"? unsure whether this is slow.
                    assert f"{input_name}_scrub_" in a.name, (
                        "We've already seen the root but haven't replaced all inputs, should never happen!",
                        i_node.name,
                        a.name,
                        input_name,
                    )
                    # What node tagged it? Should be a parent of this node
                    prev_tagger_name = a.name.replace(f"{input_name}_scrub_", "")
                    prev_tagger = self._nodes.get_by_name(prev_tagger_name)
                    if i_node not in prev_tagger.children:
                        raise IllegalCorrespondenceError(i_node, prev_tagger)
                    # Check that this parent-child pair respects digraph structure: we know now that the child matches
                    # only paths matched by the parent, but this could also happen if the parent matches nodes that
                    # are *only children* of the nodes matched by the child.
                    parent_circs = self._nodes[prev_tagger].get(circuit)
                    this_circs = self._nodes[i_node].get(circuit)
                    for this_c in this_circs:
                        if not any(parent_c.are_any_found(rc.Matcher(this_c)) for parent_c in parent_circs):
                            raise IllegalCorrespondenceError(
                                "Expected child to be a descendant of at least one parent but was not!",
                                prev_tagger,
                                i_node,
                                this_c,
                            )

            # Note the *input* is being given a unique name, while the outer tag keeps the input_name.
            # This way, when we match on input_name, we'll actually see that there are multiple matches
            # after treeification!
            new_a = a.rename(f"{input_name}_scrub_{i_node.name}")
            return rc.Tag.new_with_random_uuid(new_a, name=input_name)

        circuit = self._base_circuit
        for i_node, m_node in self._nodes.in_dfs_order():
            # we use dataset here, but all that matters is the names
            # (which must match names in the circuit, but don't all have to exist).
            input_matcher = to_inputs(m_node, self._dataset)
            circuit = input_matcher.update(circuit, lambda a: replacement(i_node, a))
        return circuit

    def replace_inputs(self, circuit: rc.Circuit, sampled_inputs: SampledInputs) -> rc.Circuit:
        """
        Replaces inputs to model branches according to the correspondence:
        - conditional sampled dataset at each leaf
        - other-children dataset at internal nodes (relevant iff correspondence is not surjective)

        Works regardless of whether model has already been treeified.

        N.B. This is not an algebraic rewrite!
        """
        for parent_node, interp_node, m in self._nodes.in_dfs_order_with_parents():
            circuit = self._replace_one_input(
                circuit, interp_node, matcher=m, sampled_inputs=sampled_inputs, parent_node=parent_node
            )
        return circuit

    def _replace_one_input(
        self,
        circuit: rc.Circuit,
        interp_node: InterpNode,
        matcher: rc.IterativeMatcher,
        sampled_inputs: SampledInputs,
        parent_node: Optional[InterpNode] = None,
    ) -> rc.Circuit:
        """Internal method"""
        if parent_node is not None and sampled_inputs[parent_node] == sampled_inputs[interp_node]:
            return circuit
        input_matcher = to_inputs(matcher, self._dataset)
        if not input_matcher.are_any_found(circuit):
            raise IllegalCorrespondenceError("No matches found in circuit", input_matcher)

        circuit = input_matcher.update(circuit, lambda a: sampled_inputs[interp_node].arrs[a.name])
        return circuit

    def wrap_in_var(self, circuit: rc.Circuit, ref_ds: Dataset, group: rc.Circuit):
        # TODO: add tests / better support for wrap_in_var = False. Also, add better errors or warnings for input
        # of wrong shape, redundant vars, etc.

        # we use ref_ds here, but we'll overwrite the values later.
        # All that matter are the shape (must match future datasets) and the names
        # (which must match names in the circuit, but don't all have to exist; thus fancy_validate=False).
        def arr_to_var(arr: rc.Circuit):
            arr = ref_ds.arrs[arr.name]
            return rc.DiscreteVar(
                values=arr,
                probs_and_group=group,
                name=f"{arr.name}_var",
            )

        circuit = rc.IterativeMatcher(set(ref_ds.input_names)).update(circuit, arr_to_var, fancy_validate=False)
        return circuit


@attrs.define(frozen=True)
class ScrubbedExperiment:
    """
    The result of calling `Experiment.scrub()`. Keeps track of the scrubbed circuit, the reference dataset, and the
    sampled inputs for all the interpretation nodes.
    """

    circuit: rc.Circuit
    ref_ds: Dataset
    sampled_inputs: SampledInputs
    group: rc.Circuit
    seed: int

    # only needed for printing
    nodes: Correspondence

    def evaluate(self, eval_settings: ExperimentEvalSettings = ExperimentEvalSettings()) -> torch.Tensor:
        """
        Returns the output of the scrubbed circuit, evaluated on the reference dataset.

        If the circuit is a subset of a bigger circuit, you can use .ref_ds to evaluate the other parts of the
        circuit as well.
        """
        sampler = eval_settings.get_sampler(self.group, self.seed)
        circ = sampler.sample(self.circuit)
        return (
            rc.optimize_and_evaluate(
                circ,
                eval_settings.optim_settings,
            )
            if eval_settings.optimize
            else circ.evaluate()
        )

    def print(
        self,
        options: Optional[Union[rc.PrintOptions, rc.PrintHtmlOptions]] = None,
        print_data=True,
        color_by_data=True,
        repr=False,
    ):
        """
        Print the scrubbed circuit, terminating at leaf mapped nodes and at nodes that are not ancestors of any mapped node.
        Optionally annotate all the nodes with the data sampled for them.

        Parameters:
            print_data: if True, prints a sample datum from cond_sampler as a comment
            color_by_data: if True, colors mapped nodes by the data they are sampled from and greys out nodes that are not ancestors of a mapped node
            repr: if True, returns string instead of printing it (used for testing)
        """
        new_options: Union[rc.PrintOptions, rc.PrintHtmlOptions]
        if options is None:
            new_options = rc.PrintHtmlOptions(traversal=rc.new_traversal())
        else:
            new_options = options.evolve()

        # Find all circuits which are mapped to by correspondence and record what dataset they're getting
        scrubbed_nodes: Dict[rc.Circuit, InterpNode] = {
            c: i for i in self.nodes.corr.keys() for c in list(self.nodes[i].get(self.circuit))
        }
        scrubbed_inputs: Dict[rc.Circuit, InterpNode] = {
            c_input: i for c, i in scrubbed_nodes.items() for c_input in c.get(self.ref_ds.input_names)
        }
        scrubbed_matcher = rc.Matcher(set(scrubbed_nodes.keys()))

        # Is it outside the image of the correspondence, i.e. is it not in any path matched by the corr's matchers?
        # note we're not using matcher's get_all_paths because it results in a giant list of nodes; since
        # the model has been treeified, it's sufficient to check whether a node has scrubbed nodes downstream of it.
        not_in_scrubbed_path = rc.Matcher(lambda c: not scrubbed_matcher.are_any_found(c))
        # Expand the children by default?
        scrubbed_leaf_matcher = rc.Matcher(lambda c: scrubbed_matcher(c) and scrubbed_nodes[c].is_leaf())
        term_at_matcher = not_in_scrubbed_path | scrubbed_leaf_matcher

        # Colors!
        def get_color(c: rc.Circuit, html: bool):
            # non html colors are pretty made up, would be nice if printer had a "color by feature"
            if scrubbed_matcher(c):
                return color_dataset(self.sampled_inputs[scrubbed_nodes[c]], html)
            elif c in scrubbed_inputs:
                return color_dataset(self.sampled_inputs[scrubbed_inputs[c]], html)
            elif not_in_scrubbed_path(c):
                return "darkgray" if html else 6  # todo light mode
            else:
                return "lightgray" if html else 13

        if color_by_data:
            new_options.colorer = lambda c: get_color(c, isinstance(new_options, rc.PrintHtmlOptions))  # type: ignore

        # Annotate mapped nodes with an example datum
        def get_data_comment(c: rc.Circuit) -> str:
            if scrubbed_matcher(c):
                return scrubbed_nodes[c].str_samplers(sampled_inputs=self.sampled_inputs)
            elif c in scrubbed_inputs:
                return str(self.sampled_inputs[scrubbed_inputs[c]])
            else:
                return ""

        new_options.commenters += [get_data_comment] if print_data else []

        # traversal
        if isinstance(new_options, rc.PrintOptions):
            new_options.bijection = False
        new_options.traversal = rc.restrict(new_options.traversal, term_early_at=term_at_matcher)

        if repr:
            assert isinstance(new_options, rc.PrintOptions)
            return self.circuit.repr(new_options)
        else:
            self.circuit.print(new_options)
