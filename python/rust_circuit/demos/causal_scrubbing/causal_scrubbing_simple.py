# %% [markdown]
# # Causal scrubbing
# %% [markdown]
# ## A simple example

import uuid

# We'll walk through how the code implements the causal scrubbing algorithm in the case of a simple model (it's the one we discuss in
# our [writeup](https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-redwood-research#3_1_An_informal_description__What_activation_replacements_does_a_hypothesis_imply_are_valid_);
# though to run the code we have to make it a bit more specific).
# %% [markdown]
### G to explain and proposed correspondence
# %%
from typing import Dict, List, Tuple

import torch

import rust_circuit as rc
from interp.circuit.testing.notebook import NotebookInTesting
from interp.tools.indexer import TORCH_INDEXER as I
from rust_circuit.causal_scrubbing.dataset import color_dataset
from rust_circuit.causal_scrubbing.experiment import Experiment, ExperimentEvalSettings, ScrubbedExperiment
from rust_circuit.causal_scrubbing.hypothesis import (
    Correspondence,
    FuncSampler,
    InterpNode,
    corr_root_matcher,
    to_inputs,
)
from rust_circuit.causal_scrubbing.testing_utils import IntDataset

# %%
# Make a dataset of xs and labels
data_generator = torch.Generator()
data_generator.manual_seed(33)
data = torch.randint(high=10, size=(100, 3), generator=data_generator)
ds = IntDataset(
    (rc.Array(data, "xs"), rc.Array(torch.logical_or(data[:, 0] > 2, data[:, 1] > 4).to(torch.int64), "labels"))
)
print(repr(ds))
print(ds[0])

# Placeholder inputs; we'll feed in real inputs later
xs = rc.Array(torch.zeros((3,)), name="xs")
labels = rc.Array(torch.zeros(()), name="labels")

# Construct the computational graph (I just made this one up; in practice this might be some neural network).
# You might find it helpful to draw this graph on paper.
x0 = rc.Index(xs, I[0], name="x0")
x1 = rc.Index(xs, I[1], name="x1")
x2 = rc.Index(xs, I[2], name="x2")
A = rc.sigmoid(rc.Add(x0, rc.Einsum.scalar_mul(x1, 0), rc.Einsum.scalar_mul(x2, 0), rc.Scalar(-3)), name="A")
B = rc.sigmoid(rc.Add(x0, rc.Einsum.scalar_mul(x1, 0), rc.Einsum.scalar_mul(x2, 0), rc.Scalar(-3)), name="B")
C = rc.Add(x0, x1, x2)
D = rc.Add(rc.Einsum.elementwise_broadcasted(A, B), rc.Einsum.scalar_mul(C, 0), name="D")
loss = rc.Einsum.from_einsum_string(",  -> ", D, labels, name="loss")

loss.print()
# %%
# Defining our interpretation graph: we're going to hypothesize that e.g. the node "A" in our model
# only cares about whether the the data at index 0 is > 3 or not. Again, you might find it helpful
# to draw this out!
# First we define some helper functions:
def x0_f(d: IntDataset):
    return (d.xs.value)[:, 0]


def x1_f(d: IntDataset):
    return (d.xs.value)[:, 1]


def x0_gt_3(d: IntDataset):
    return x0_f(d) > 3


def x1_gt_3(d: IntDataset):
    return x1_f(d) > 3


def x0_or_x1_gt_3(d: IntDataset):
    return torch.logical_or(x0_gt_3(d), x1_gt_3(d))


def x0_or_x1_gt_3_agrees_with_label(d: IntDataset):
    return x0_or_x1_gt_3(d) == d.labels.value


def label_f(d: IntDataset):
    return d.labels.value


# Then we actually create a graph of InterpNodes
# FuncSampler of a function means that our sampled inputs must agree on this function with the reference;
# we'll talk more about CondSamplers lower down, for now you just need to know that you need to wrap
# your function in it.
out = InterpNode(cond_sampler=FuncSampler(x0_or_x1_gt_3_agrees_with_label), name="out")  # type: ignore
D_prime = out.make_descendant(FuncSampler(x0_or_x1_gt_3), name="D'")  # type: ignore
A_prime = D_prime.make_descendant(FuncSampler(x0_gt_3), name="A'")  # type: ignore
B_prime = D_prime.make_descendant(FuncSampler(x1_gt_3), name="B'")  # type: ignore
x0_prime = A_prime.make_descendant(FuncSampler(x0_f), name="x0'")  # type: ignore
x1_prime = B_prime.make_descendant(FuncSampler(x1_f), name="x1'")  # type: ignore
y_prime = out.make_descendant(FuncSampler(label_f), name="y'")  # type: ignore
# We can print out the graph we've defined!
out.print()
# %%
# And, finally, writing out the correspondence between the interpretation I (rooted at the InterpNode `out`)
# and the computational graph G (rooted at the Circuit `loss`).
# In particular, we associate every interp node with an IterativeMatcher which points at a part of the treeified circuit.
corr = Correspondence()
corr.add(out, corr_root_matcher)
corr.add(D_prime, rc.IterativeMatcher("D"))
corr.add(B_prime, rc.IterativeMatcher("B"))
corr.add(A_prime, rc.IterativeMatcher("A"))
corr.add(x1_prime, rc.IterativeMatcher("B").chain("x1"))
corr.add(x0_prime, rc.IterativeMatcher("A").chain("x0"))
corr.add(y_prime, rc.IterativeMatcher("labels"))
# %%

# %% [markdown]

# ### Doing causal scrubbing: recursively sampling data
#
# Terminology note: in circuits land, including here, we think of our circuit as being a dependency graph
# with arrows from nodes to nodes they depend on. This means that the output is an *ancestor* of the
# input. So, any time you see "parent", this means something closer to the output, and vice versa for "child".
# In the causal scrubbing algorithm, we traverse the interpretation graph from parents to children.
#
# The first thing we do in a causal scrubbing experiment is recursively (starting from the root, i.e. the output)
# sample two new data for each node of our interpretation graph. At each node n_I we have two inputs:
# 1. A source dataset that we can sample from
# 2. A parent datum that constrains our next sample
# And are sampling:
# 1. The new datum, which agrees with the parent datum on the function at n_I
# 2. An "other datum" that will be used for any inputs to this node that are not specified by the correspondence
#
# Normally this would happen behind the scenes in an Experiment, but here we'll walk through it step by step.

# After each recursive sampling step, we'll print out the interpretation. The nodes will be colored
# by the hash of their dataset, and they'll be annotated with the two data we've sampled for them.
# As you run the sampling, verify for yourself that the sampled datum at each node does in fact agree
# on the value of the function at that node with the datum of its parent.
# %%
generator = torch.Generator()  # rng generation
generator.manual_seed(22)

# the reference datum is the parent datum for the root; see section towards the end for why we do this
ref_datum = ds.sample(1, generator)
print(f"ref_datum: {str(ref_datum)}")
sampled_inputs = out.sample(generator, ds, ref_datum, recursive=False)
out.print(sampled_inputs=sampled_inputs)
# %%
def get_parents_and_children(i: InterpNode):
    return [(i, c) for c in i.children] + [pair for c in i.children for pair in get_parents_and_children(c)]


def step_sampler(parents_and_children: List[Tuple[InterpNode, InterpNode]]):
    for (p, c) in parents_and_children:
        c._sample_into(generator, ds, sampled_inputs[p], into=sampled_inputs, recursive=False)
        yield


step_sample = step_sampler(get_parents_and_children(out))
# %%
# we sample for D' a datum that agrees on if x0 > 3 OR x1 > 3
next(step_sample)
out.print()
# %%
# we sample for y' a datum that agrees on the label
next(step_sample)
out.print()
# %%
# we sample for A' a datum that agrees on if x0 > 3
next(step_sample)
out.print()
# %%
# we sample for B' a datum that agrees on if x1 > 3
next(step_sample)
out.print()
# %%
# we sample for x0' a datum that agrees on x0
next(step_sample)
out.print()
# %%
# we sample for x1' a datum that agrees on x1
next(step_sample)
out.print()
# %%
try:
    next(step_sample)
    assert False, "should be done sampling at this point!"
except StopIteration:
    pass

assert (  # To make sure this notebook doesn't get broken
    out.print(
        rc.PrintOptions(), sampled_inputs, color_by_data=False, repr=True
    )  # ignoring colors, which are sadly nondeterministic atm
    == """out GeneralFunction # cond_sampler=FuncSampler(d=IntDatum(xs=[9, 1, 0], label=1), f(d)=True), other_inputs_sampler=UncondSampler(d=IntDatum(xs=[0, 6, 8], label=1))
  D' GeneralFunction # cond_sampler=FuncSampler(d=IntDatum(xs=[0, 9, 8], label=1), f(d)=True), other_inputs_sampler=UncondSampler(d=IntDatum(xs=[3, 2, 9], label=1))
    A' GeneralFunction # cond_sampler=FuncSampler(d=IntDatum(xs=[0, 8, 5], label=1), f(d)=False), other_inputs_sampler=UncondSampler(d=IntDatum(xs=[0, 4, 6], label=0))
      x0' GeneralFunction # cond_sampler=FuncSampler(d=IntDatum(xs=[0, 5, 5], label=1), f(d)=0)
    B' GeneralFunction # cond_sampler=FuncSampler(d=IntDatum(xs=[0, 2, 8], label=0), f(d)=False), other_inputs_sampler=UncondSampler(d=IntDatum(xs=[0, 9, 4], label=1))
      x1' GeneralFunction # cond_sampler=FuncSampler(d=IntDatum(xs=[3, 9, 8], label=1), f(d)=9)
  y' GeneralFunction # cond_sampler=FuncSampler(d=IntDatum(xs=[6, 0, 6], label=1), f(d)=1)"""
)
# %% [markdown]

# ### Doing causal scrubbing: replacing inputs to the treeified model

# Recall that in causal scrubbing, we need to treeify our model; but we don't need to treeify it entirely,
# only as much as our hypothesis requires. That is, if our hypothesis says that some nodes need to be sampled
# together, they can stay together.

# In the context of an Experiment, we can call treeified() to perform this rewrite. This finds the paths
# that are picked out by this hypothesis, and wraps the inputs to each with a random tag to uniqueify them.
# This is an algebraic rewrite -- the resulting circuit is extensionally equal -- because we're not actually
# changing any of the inputs.
#
# Treeifying also allows us to run some checks that your correspondence is doing what it says on the tin.
# These checks assert various invariances that aren't otherwise enforced when constructing a Correspondence or
# Experiment, and catch various misleading correspondences that will run but give surprising results.
# We recommend running with treeify=True and check=True to validate your correspondence when you first create it.
#
# Before running the below, try drawing out what the treeified graph will look like on paper.
# %%
ex = Experiment(loss, ds, corr, random_seed=11)
treeified_circuit = ex.treeified()
corr.check(treeified_circuit, circuit_treeified=True)

treeified_circuit.print_html()
# %% [markdown]
# After that we can replace the inputs. Normally this would happen behind the scenes in an experiment, but we'll step through it here.
# %%
circuit = ex.wrap_in_var(treeified_circuit, ref_datum, rc.DiscreteVar.uniform_probs_and_group(1))

already_scrubbed: Dict[rc.IterativeMatcher, InterpNode] = {}
already_scrubbed_inputs: Dict[rc.IterativeMatcher, InterpNode] = {}


def colorer(c: rc.Circuit, super: rc.Circuit) -> str:
    color = "darkgrey"
    for m, i in already_scrubbed.items():
        m_endpoints = m.get(super)
        if c in m_endpoints:
            color = color_dataset(sampled_inputs.datasets[i], html=True)
        elif color == "darkgrey" and i.is_leaf() and c.are_any_found(m_endpoints):
            color = "lightgrey"
    for m, i in already_scrubbed_inputs.items():
        m_endpoints = m.get(super)
        if c in m_endpoints:
            color = color_dataset(sampled_inputs.datasets[i], html=True)
    return color  # type: ignore


def commenter(c: rc.Circuit, super: rc.Circuit) -> str:
    comment = ""
    for m, i in already_scrubbed.items():
        if c in m.get(super):
            comment = i.str_samplers(sampled_inputs)
    for m, i in already_scrubbed_inputs.items():
        if c in m.get(super):
            comment = f"ds set by '{i.name}': {sampled_inputs[i]}"
    return comment


po = rc.PrintHtmlOptions(
    colorer=lambda c: colorer(c, circuit),
    traversal=rc.IterativeMatcher.noop_traversal(),
    commenters=[lambda c: commenter(c, circuit)],
)


def step_scrubber(ex: Experiment, circuit: rc.Circuit):
    for interp_node, m in ex.nodes.in_dfs_order():
        print(interp_node)
        input_matcher = to_inputs(m, ex.dataset)
        already_scrubbed[m] = interp_node
        already_scrubbed_inputs[input_matcher] = interp_node
        # Internal method, called here just for demonstration
        circuit = ex._replace_one_input(circuit, interp_node, m, sampled_inputs)
        yield circuit


# %%
# Notice we're not using ScrubbedExperiment.print() here: the logic there assumes we have finished replacing all the
# inputs, so in order to color and annotate nodes as we step through we write the printing logic ourselves. As in
# ScrubbedExperiment.print(), nodes here are being colored to match their corresponding interp nodes. Dark grey nodes
# are outside the image of the correspondence (or we haven't gotten to replacing them yet), and light grey nodes are in
# matched paths but not directly mapped to. The annotation shows the data we sampled at that node, and the value of the
# interp node's function on that datum.
#
# These are exactly the data that we sampled earlier--we are now using those samples to replace the inputs to our
# circuit.
#
# You might find it useful to expand the nodes (click a > to turn it into a v) so you can see the entire treeified circuit, including
# the colors and annotations at the inputs. By default repeated nodes will be collapsed (including sometimes other copies of the input)

step_scrub = step_scrubber(ex, circuit)

circuit.print(po)
# %%
# We set _all_ inputs downstream of 'out' with out's 'other datum'
circuit = next(step_scrub)
circuit.print(po)
# %%
# We set all inputs downstream of 'D' with D's 'other datum'
circuit = next(step_scrub)
circuit.print(po)
# %%
# We set all inputs downstream of 'A' with A's 'other datum'
circuit = next(step_scrub)
circuit.print(po)
# %%
# We set all inputs downstream of 'x0'. As x0 is a leaf node we get to use it's main datset!
circuit = next(step_scrub)
circuit.print(po)
# %%
# Now into B, again with the 'other datum'
circuit = next(step_scrub)
circuit.print(po)
# %%
# B has one child -- x1. x1 is a leaf, so we use it's main data set to set inputs
circuit = next(step_scrub)
circuit.print(po)
# %%
# And finally set the inputs upstream of labels (also a leaf interp node)
circuit = next(step_scrub)
circuit.print(po)
# %%
try:
    next(step_scrub)
    assert False, "should be done scrubbing at this point!"
except StopIteration:
    pass

treeify_first = circuit
# %%
# Or, we can replace inputs without explicitly treeifying-- the treeification will happen automatically
# as inputs to different paths are replaced!
circuit = ex.base_circuit
group = rc.DiscreteVar.uniform_probs_and_group(1)
circuit = ex.wrap_in_var(circuit, ref_datum, group)
circuit = ex.replace_inputs(circuit, sampled_inputs)
circuit.print()

# %%
# Tags have random uuids, and treeified-first has some extra tags, but otherwise the two circuits are identical.
# We'll remove the extra tags, and replace all tags with the same uuid.
clean_tags = lambda c: c.update(
    lambda x: x.is_tag(), lambda x: rc.Tag(x.node, uuid.UUID("00000000-0000-0000-0000-000000000000"))
)
treeify_first_clean = clean_tags(treeify_first).update(lambda x: x.is_tag() and x.node.is_var(), lambda x: x.node)
assert treeify_first_clean == clean_tags(circuit)
# %% [markdown]

# ### Doing causal scrubbing: computing the behavior function on the scrubbed model
#
# Now we just run the scrubbed model forward!
# %%
device = "cpu" if NotebookInTesting.currently_in_notebook_test else "cuda:0"  # for CI
eval_settings = ExperimentEvalSettings(device_dtype=device)
scrubbed_out = ScrubbedExperiment(circuit, ref_datum, sampled_inputs, group, 11, ex._nodes).evaluate(
    eval_settings=eval_settings
)
print(scrubbed_out)
# %% [markdown]

# ### Doing causal scrubbing: expectation of the behavior function
# In practice, we run causal scrubbing on a batch and take the expectation.
# %%
out_batch = Experiment(loss, ds, corr, random_seed=11).scrub(20).evaluate(eval_settings)
print(out_batch.mean())
# %%
# Recall that our goal is to get as close as possible to the value of the behavior function on the original inputs:
loss_with_data = rc.Expander(("xs", lambda _: ds.xs), ("labels", lambda _: ds.labels))(loss)
print(loss_with_data.evaluate().mean())
# %%
# %% [markdown]

# ## Conveniences
#
# The causal scrubbing code provides a number of conveniences to make expressing your hypotheses easier, rather than forcing you
# to manually specify a graph isomorphism between your interpretation and the model.
# %% [markdown]

# ### IterativeMatcher

# Suppose you had the diamond graph (where A is the input and D is the output)
# ```
#   A
#  / \
# B   C
#  \ /
#   D
# ```
# and you wanted to hypothesize that A does some computation and later D uses that information to do something. You might want to
# draw the interpretation `A -> D`; but there is no such edge in the model, so you can't write down a homomorphism c from I to G!
#
# Or, suppose you were analyzing a neural network. You had analyzed the attention heads and thought 0.1 and 2.1 were important while
# no heads in layer 1 were, but you had not yet looked at the MLPs. How would you express the hypothesis that paths from 0.1 to 2.1,
# potentially passing through MLPs but not through layer 1 heads, were important?
#
# In both of these cases, you could rewrite your interpretation with more nodes -- in the first example, adding nodes to your
# interpretation corresponding to B and C. Or, you could do a circuit rewrite to group together some of the nodes in your model.
# Both of these are pretty annoying to do, so instead we allow you to correspond an interpretation node to an IterativeMatcher picking
# out some paths in the model. That is, you can say things like:
# ```
# correspondence.add(a_I, rc.IterativeMatcher("D").chain("A"))
# ```

# %% [markdown]

# ### Implicit extension to inputs

# If you have not explicitly specified an interpretation for some of the paths between c(I) and the inputs, this is filled
# in for you. After all, we have to have inputs for every path through the model! In other words, for every matcher in your
# interp graph, all the inputs upstream of nodes that matcher matches will be replaced. They will be replaced by the interp
# node's `ds` if they are leaves, and its `other_inputs_ds` if they are not (see the next section); some inputs will then be
# replaced again by child interp nodes if they exist.
#
# To do this we have to know what the "inputs" are. The names of the inputs are defined in the dataset you supply to the
# experiment, and are expected to be present in the circuit you pass in.

# %% [markdown]

# ### More on: nodes not in any matched paths
#
# We mentioned briefly earlier that each interpretation node n_I samples a other_inputs_ds. If c(n_I) has children that are not in
# the paths defined by c(children(n_I)), they will be run on this dataset. For instance, in the example we looked at, the input to
# x1 and x2 going into A was drawn from A_primes other_inputs_ds. This allows you to very easily say that "the other
# children don't matter" -- by default, other_inputs_ds is sampled randomly from the source dataset.
#
# But you can also use this in other ways: for example, you can set n_I's cond_sampler to an UncondSampler, and n_I's other_inputs_sampler
# to an ExactSampler, which represents "c(children(n_I)) don't matter, and the other children do matter and can't be resampled at all".
# For example, you can use this to ablate a single path in the model.
#
# ```
# n_I = InterpNode(cond_sampler=UncondSampler(), other_inputs_sampler=ExactSampler(), name="n")
# ```
#
# Sometimes we might call the set `children(c(n_I)) \ c(children(n_I))` the "unimportant" children as a shorthand, even though they
# are not necessarily unimportant. We also sometimes refer to them as "not in the image of c".

# %% [markdown]

# ### Ref ds for extending the graph
#
# When you .evaluate a scrubbed experiment, you only get the output on the circuit that you specified. If the scrubbed
# circuit computes your entire behavior function, you are done. But, if you want to feed the result of your scrubbed circuit into
# a continued behavior function, you should use the reference ds for any computations there, which you can get by reading the
# ScrubbedExperiment.ref_ds field. For example, if the circuit computes logits and you want to compute a loss, you should use the
# reference dataset to get the labels.

# %% [markdown]

# ## Checks
#
# The code performs a number of checks to make sure your correspondence induces well-defined scrubs. For examples, take a look
# at test_hypothesis.py and test_experiment.py. But we'll briefly list out the checks here as well:
#
# - out maps to out
# - in maps to in
# - corr must map every node in the interp graph to something
# - must respect model tree structure: an interp node I is a child of I' iff each path in c(I) is an extension of a path in c(I')
# - each matcher must match something
# - no two matchers can match the same path
# - bonus: after treeifying, no two matchers can match the same node. this should be equivalent to the above.
#
# Most of these won't run if you disable checks on your experiment; but some are mandatory (like the fact that each matcher must
# match something).
# %% [markdown]

# ## Generalization
#
# Our code is also more general than the algorithm we describe in our writeup in a couple of useful ways.
#
# 1. Each interp node, rather than computing a function and sampling data that agree with the parent datum on that function,
# is associated with a sampler. This is convenient for lots of reasons, like generating data on the fly, but is also fundamentally
# more general: you do not have to sample uniformly from a subset of the data!
# 2. The interpretation is not a computational graph. Again, some things are simply more convenient to express as conditional on the
# dataset rather than the outputs of a computational graph--for example, sampling some things together, enforcing agreement on a feature
# the samples for different nodes, phi rewrites. But, conditions are more powerful than agreement on a function; we won't elaborate on
# that here.

# %% [markdown]

# ## Code base components
#
# The causal scrubbing code relies on the following components:
# 1. An `Experiment`, which takes a `Circuit`, a `Dataset`, and a `Correspondence`.
# 2. A `Circuit`, the model we want to interpret
# 3. A `Correspondence` object, which associates `InterpNodes` with `IterativeMatchers`
# 4. `InterpNodes` are associated in a tree structure, where children are created by calling
# `InterpNode().make_decendant()`. Each `InterpNode` has a name (str) and two `CondSamplers`, one
# that is used for it's 'main datum' and one used for the 'other datum'
# 5. `CondSamplers` are objects that implement `sample(ref_datum, dataset) -> new_datum`.
# For instance a `FuncSampler` will sample new_datum from dataset that agrees on a specified function.
# In reality, they do this in a vectorized way from `(ref_dataset, dataset) -> new_dataset`
# but it can be helpful to think one datapoint at a time.
# 6. `IterativeMatcher` which points at a single subcircuit of the model, potentially on a restricted set of paths.
# 7. `Dataset` objects are pretty simple, mostly wrappers around several arrays with some
# convenience functions (e.g. printing). We require that they frozen for hashing.

# %%
