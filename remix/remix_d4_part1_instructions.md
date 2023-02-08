
# Causal scrubbing with a simple example

Redwood has a code base for causal scrubbing, and you won't need to implement the algorithm yourself when you use it in your research projects. However, it is still valuable to understand how the algorithm is implemented, and practice using the code base on a simple problem. In this notebook, we'll walk through how the our code implements the causal scrubbing algorithm in the case of a simple example, the one we discuss in our [writeup](https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-redwood-research#3_1_An_informal_description__What_activation_replacements_does_a_hypothesis_imply_are_valid_); though to run the code we have to make it a bit more specific.

I recommend having the writeup open as you work through today's content.

We'll use some of the notation from the writeup, e.g. $c: I \to G$ is the correspondence mapping nodes $n_I$ of the interpretation graph $I$ to nodes $n_G$ of the computational graph of interest $G$.

## Table of Contents

- [Learning Objectives](#learning-objectives)
- [Code base components](#code-base-components)
- [Make dataset](#make-dataset)
- [Construct computational graph](#construct-computational-graph)
- [Construct interpretation graph](#construct-interpretation-graph)
- [Doing causal scrubbing](#doing-causal-scrubbing)
    - [Recursively sampling data](#recursively-sampling-data)
    - [Replacing inputs to the treeified model](#replacing-inputs-to-the-treeified-model)
    - [Computing the behavior function on the scrubbed model](#computing-the-behavior-function-on-the-scrubbed-model)
- [Exercises](#exercises)
    - [Exercise 1](#exercise-)
    - [Exercise 2](#exercise-)
    - [Exercise 3](#exercise-)
    - [Other hypotheses](#other-hypotheses)
- [Further comments](#further-comments)
    - [Conveniences](#conveniences)
    - [IterativeMatcher](#iterativematcher)
    - [Implicit extension to inputs](#implicit-extension-to-inputs)
    - [More on: nodes not in any matched paths](#more-on-nodes-not-in-any-matched-paths)
    - [Reference dataset for extending the graph](#reference-dataset-for-extending-the-graph)
    - [Checks](#checks)
    - [Generalization](#generalization)

## Learning Objectives

After today's material, you should be able to:

- Use classes in the `causal_scrubbing` package.

## Code base components

The causal scrubbing code base relies on the following components. You have encountered some of them before, and some of them will be new to you.

1. An `Experiment`, which takes a `Circuit`, a `Dataset`, and a `Correspondence`.
2. A `Circuit`, the model we want to interpret.
3. `DataSet` objects are pretty simple, mostly wrappers around several arrays with some convenience functions (e.g. printing). We require that they are frozen for hashing.
4. A `Correspondence` object, which specifies the correspondence between the interpretation and the model. In practice, it associates `InterpNodes` with `IterativeMatchers`.
5. `InterpNodes` are associated in a tree structure, where children are created by calling `interp_node.make_descendant()`. Each `InterpNode` has a name (`str`) and two `CondSamplers`.
6. `CondSamplers` are objects that implement `sample(ref_datum, dataset) -> new_datum`. For instance a `FuncSampler` will sample new_datum from dataset that agrees on a specified function with the ref_datum. This is a little bit different from the writeup, where we just talked about sampling a new_datum that agrees on a specified function; we talk a bit about this generalization in the Comments section below. In practice, CondSamplers sample in a vectorized way from `(ref_dataset, dataset) -> new_dataset`, but it can be helpful to think one datapoint at a time.
7. `IterativeMatcher` which picks out a set of paths through the model (this is mathematically equivalent to picking out a set of nodes in the treeified model). This lets us say things like "head 6.5, going to the output via the MLPs but not via any other attention layers, contributes the tense of the sentence."

Why does an InterpNode have *two* CondSamplers? One is needed for running the sampling step of the causal scrubbing algorithm (make sure this makes sense to you!). The other is purely for convenience: there might be many paths in the model that you think are unimportant and don't want to have to write out explicitly in your correspondence, so the other_inputs_sampler at an InterpNode takes care of paths that you haven't specified. Take a look at the hypothesis diagram in the link above: we have an interpretation for $z_1 \to A'$, but haven't explicitly said anything about $z_2 \to A'$ or $z_3 \to A'$. The other_inputs_sampler at $A'$ is responsible for sampling the input to these unspecified paths: you can think of it as, if we *had* written out a more complete interpretation *I* that included $z_2 \to A'$ and $z_3 \to A'$, this is the sampler that we would have associated with those two nodes. We talk a bit more about this second CondSampler in the Comments section below; but for now you can just assume it's always going to be sampling a random datum for the paths we haven't said anything explicit about.



```python
from typing import Dict, List, Optional, Tuple, cast
import uuid
from interp.circuit.causal_scrubbing.dataset import color_dataset, Dataset
from interp.circuit.causal_scrubbing.experiment import Experiment, ExperimentEvalSettings, ScrubbedExperiment
from interp.circuit.causal_scrubbing.testing_utils import IntDataset
from interp.circuit.causal_scrubbing.hypothesis import (
    Correspondence,
    FuncSampler,
    ExactSampler,
    UncondSampler,
    InterpNode,
    corr_root_matcher,
    to_inputs,
)
import rust_circuit as rc
import torch as t
import torch
from interp.circuit.testing.notebook import NotebookInTesting
from interp.tools.indexer import TORCH_INDEXER as I

MAIN = __name__ == "__main__"
if MAIN:
    from remix_extra_utils import check_rust_circuit_version

    check_rust_circuit_version()

```

## Make dataset

Recall that the data in this example consist of `(x0, x1, x2, labels)`. We didn't say in our writeup what the label is: let's say it's 1 when either `x0` or `x1` is greater than 3, and 0 otherwise (this is in agreement with the interpretation we propose in the writeup, but in general this doesn't have to be the case!). Here we will assume that the inputs are all integers. `IntDataset` is a subclass of `Dataset`, and supports integer data. Here, we generate some random `(x0, x1, x2)`, and compute the labels.


To set a seed for torch randint we need to use a generator.


```python
data_generator = torch.Generator()
data_generator.manual_seed(33)
data = torch.randint(high=10, size=(10000, 3), generator=data_generator)
ds = IntDataset(
    (rc.Array(data, "xs"), rc.Array(torch.logical_or(data[:, 0] > 3, data[:, 1] > 3).to(torch.int64), "labels"))
)
print(repr(ds))
print(ds[0])

```

## Construct computational graph

Next, we need to construct the computational graph. In practice, this would be the neural network that you want to interpret. For the purpose this example, we just made up a computational graph. You might find it helpful to draw this graph on a whiteboard or piece of paper.

We use placeholder inputs `xs` and `labels`, which will be replaced later.



```python
xs = rc.Array(torch.zeros((3,)), name="xs")
labels = rc.Array(torch.zeros(()), name="labels")
x0 = rc.Index(xs, I[0], name="x0")
x1 = rc.Index(xs, I[1], name="x1")
x2 = rc.Index(xs, I[2], name="x2")
A = rc.sigmoid(rc.Add(x0, rc.Einsum.scalar_mul(x1, 0.1), rc.Einsum.scalar_mul(x2, -0.05), rc.Scalar(-3)), name="A")
B = rc.sigmoid(rc.Add(rc.Einsum.scalar_mul(x0, -0.01), x1, rc.Einsum.scalar_mul(x2, 0.2), rc.Scalar(-3)), name="B")
C = rc.Add(x0, x1, x2, name="C")
D = rc.Add(rc.Add(A, B), rc.Einsum.scalar_mul(C, 0.1), name="D")
diff = rc.Add.minus(D, labels, name="diff")
loss = rc.Einsum.from_einsum_string(",  -> ", diff, diff, name="loss")
loss.print()

```

Let's start by quickly computing a couple of baselines: the loss of the original circuit, and the loss when we randomly sample the labels. Recall that in causal scrubbing, we want to recover a loss as close to the original as possible; and the loss on shuffled labels provides a useful intuition for what "really far from the original loss" is for this particular function and dataset.


```python
loss_with_data = rc.Expander(("xs", lambda _: ds.xs), ("labels", lambda _: ds.labels))(loss)
print(f"Original: {loss_with_data.evaluate().mean()}")
loss_with_shuffled = rc.Expander(
    ("xs", lambda _: ds.xs), ("labels", lambda _: ds.sample(len(ds), data_generator).labels)
)(loss)
print(f"Shuffled labels: {loss_with_shuffled.evaluate().mean()}")

```

## Construct interpretation graph

Interpretation graphs are constructed out of `InterpNode`s and have a tree structure. Each `InterpNode` has a name and two `CondSampler`s, which specify how to sample the 'new datum' and 'other inputs datum' respectively. (The 'other inputs datum' is sampled randomly by default, and we won't need to modify it for our purposes here.)

We're going to construct the whole interpretation graph with calls of `.make_descendant()` starting from the output. `.make_descendant()` takes in as argument a `CondSampler`, which will be used to sample the new datum. Here we will be using `FuncSampler`, which is a type of `CondSampler`. `FuncSampler(fn)` implements `sample(ref_datum, dataset) -> new_datum` and ensures that `new_datum` agrees with `ref_datum` on the output of `fn`.

For our interpretation, we're going to hypothesize that the node "A" in our model only cares about whether the data at index 0 is greater than 3 or not, and so on, as described in the writeup. First we will need to make some helper functions to pass to `FuncSampler`. Each one should be a single line. Then, construct the interpretation graph. You should find that you will use each helper function once. To be consistent with the solution file, name the node corresponding to `D` as `D'`, and so on. Again, you might find it helpful to draw it on a whiteboard.

Finally, since we already have all the nice circuit printing functionalities, it is designed so that you can also call `.print()` on an interpretation graph. When printing, interpretation graphs are converted into `Circuit`s (that are not actually computable; this is somewhat of a hack to make use of the nice circuit printing), and `InterpNode`s are rendered as `GeneralFunction`s.

<details>
<summary>Why does my helper function take `Dataset` and not `IntDataset`?</summary>

The helper function will eventually be called by `FuncSampler`, which is a general class that only knows about the base class `Dataset`. This shouldn't cause any problems, but it's worth noting that you can do `IntDataset.unwrap(ds)` to cast to `IntDataset` in the same way as you've encountered before.

</details>


```python
def x0_val(d: Dataset) -> t.Tensor:
    return d.xs.value[:, 0]


def x1_val(d: Dataset) -> t.Tensor:
    """TODO: YOUR CODE HERE"""
    pass


def x0_gt_3(d: Dataset) -> t.Tensor:
    """TODO: YOUR CODE HERE"""
    pass


def x1_gt_3(d: Dataset) -> t.Tensor:
    """TODO: YOUR CODE HERE"""
    pass


def x0_or_x1_gt_3(d: Dataset) -> t.Tensor:
    """TODO: YOUR CODE HERE"""
    pass


def x0_or_x1_gt_3_agrees_with_label(d: Dataset) -> t.Tensor:
    """TODO: YOUR CODE HERE"""
    pass


def label_val(d: Dataset) -> t.Tensor:
    """TODO: YOUR CODE HERE"""
    pass


out = InterpNode(cond_sampler=FuncSampler(x0_or_x1_gt_3_agrees_with_label), name="out")
"TODO: YOUR CODE HERE"
out.print()

```

You might notice above that the leaves of the interpretation graph don't have an other_inputs_sampler printed. It's not needed there--there are no unspecified paths from the inputs to `x0'`--so we avoid showing it.

Finally, with both the computational graph and interpretation graph ready, we can write out the correspondence between the interpretation $I$ (rooted at the InterpNode `out`) and the computational graph $G$ (rooted at the Circuit `loss`).
In particular, we associate every `InterpNode` with an `IterativeMatcher` which points at a part of the treeified circuit. Note that since `A` and `B` and `C` take `xs` as input, but the interpretation says that `A` actually only depends on `x0`, we need a way to get only the `x0` node input to `A` to put in our interpretation.

<details>
<summary>Hint - How to match only the x0 input to A?</summary>
You will want to first find the node A, and then find x0 in its children. In other words, you want to chain together two `IterativeMatcher`s.
</details>

<details>
<summary>Spoiler - How to match only the x0 input to A?</summary>
`rc.IterativeMatcher("A").chain("x0")` matches the `x0` node input to `A` only.
</details>



```python
corr = Correspondence()
corr.add(out, corr_root_matcher)
corr.add(D_prime, rc.IterativeMatcher("D"))
"TODO: YOUR CODE HERE"

```

## Doing causal scrubbing

### Recursively sampling data

Terminology note: in Circuits land, we think of our circuit as being a dependency graph with arrows from nodes to nodes they depend on. This means that the output is an *ancestor* of the input. So, any time you see "parent", this means something closer to the output, and "child" means closer to the input. In the causal scrubbing algorithm, we traverse the interpretation graph from parents (output) to children (inputs).

The first thing we do in a causal scrubbing experiment is recursively (starting from the root, i.e. the output) sample two new data for each node of our interpretation graph. At each node $n_I$ we start with two inputs:
1. A source dataset that we can sample from
2. A parent datum that constrains what we can sample

And need to sample:

1. The new datum, which agrees with the parent datum on the function at $n_I$. This is the datum we discuss in the causal scrubbing algorithm.
2. An "other inputs datum" that will be used for any inputs to this node that are not specified by the correspondence.

Normally this would happen behind the scenes in an `Experiment`, but here we'll walk through it step by step.


After each recursive sampling step, we'll print out the interpretation. The nodes will be colored by the hash of their dataset, and they'll be annotated with the two data we've sampled for them. As you run the sampling, verify for yourself that the sampled datum at each node does in fact agree on the value of the function at that node with the datum of its parent.

Note on `ref_datum`: Since the root (the output) has no parent, we create a dummy datum to use as the parent. This is what `ref_datum` is.


```python
generator = torch.Generator()
generator.manual_seed(11)
ref_datum = ds.sample(1, generator)
print(f"ref_datum: {str(ref_datum)}")
sampled_inputs = out.sample(generator, ds, ref_datum, recursive=False)
out.print()


def get_parents_and_children(i: InterpNode):
    return [(i, c) for c in i.children] + [pair for c in i.children for pair in get_parents_and_children(c)]


def step_sampler(parents_and_children: List[Tuple[InterpNode, InterpNode]]):
    for (p, c) in parents_and_children:
        c._sample_into(generator, ds, sampled_inputs.datasets[p], into=sampled_inputs, recursive=False)
        yield


step_sample = step_sampler(get_parents_and_children(out))
print("We sample for D' a datum that agrees on if x0 > 3 OR x1 > 3.")
next(step_sample)
out.print()
print("We sample for y' a datum that agrees on the label.")
next(step_sample)
out.print()
print("We sample for A' a datum that agrees on if x0 > 3.")
next(step_sample)
out.print()
print("We sample for B' a datum that agrees on if x1 > 3.")
next(step_sample)
out.print()
print("We sample for x0' a datum that agrees on x0.")
next(step_sample)
out.print()
print("We sample for x1' a datum that agrees on x1.")
next(step_sample)
out.print()

```

After 6 function calls to `step_sample()`, we should be done sampling. Run the cell below to check that your implementation is working.


```python
try:
    next(step_sample)
    assert False, "Should be done sampling at this point!"
except StopIteration:
    pass
assert (
    out.print(rc.PrintOptions(), color_by_data=False, repr=True, sampled_inputs=sampled_inputs)
    == "out GeneralFunction # cond_sampler=FuncSampler(d=IntDatum(xs=[8, 0, 4], label=1), f(d)=True), other_inputs_sampler=UncondSampler(d=IntDatum(xs=[3, 8, 3], label=1))\n  D' GeneralFunction # cond_sampler=FuncSampler(d=IntDatum(xs=[8, 0, 3], label=1), f(d)=True), other_inputs_sampler=UncondSampler(d=IntDatum(xs=[4, 5, 8], label=1))\n    A' GeneralFunction # cond_sampler=FuncSampler(d=IntDatum(xs=[6, 0, 8], label=1), f(d)=True), other_inputs_sampler=UncondSampler(d=IntDatum(xs=[4, 3, 4], label=1))\n      x0' GeneralFunction # cond_sampler=FuncSampler(d=IntDatum(xs=[6, 1, 9], label=1), f(d)=6)\n    B' GeneralFunction # cond_sampler=FuncSampler(d=IntDatum(xs=[8, 3, 2], label=1), f(d)=False), other_inputs_sampler=UncondSampler(d=IntDatum(xs=[3, 5, 3], label=1))\n      x1' GeneralFunction # cond_sampler=FuncSampler(d=IntDatum(xs=[9, 3, 0], label=1), f(d)=3)\n  y' GeneralFunction # cond_sampler=FuncSampler(d=IntDatum(xs=[0, 7, 7], label=1), f(d)=1)"
)

```

### Replacing inputs to the treeified model

Recall that in causal scrubbing, we treeify our model; but we don't need to treeify it entirely, only as much as our hypothesis requires. That is, if our hypothesis says that some nodes need to be sampled together, they can stay together.

In the context of an `Experiment`, we can call `.treeified()` to perform this rewrite. This finds the paths that are picked out by this hypothesis, and wraps the inputs to each with a random tag to uniqueify them. This is an algebraic rewrite — the resulting circuit is extensionally equal — because we're not actually changing any of the inputs.

Treeification also allows us to run some checks that your correspondence is doing what it says on the tin. These checks assert various invariances that aren't otherwise enforced when constructing a `Correspondence` or `Experiment`, and catch various misleading correspondences that will run but give surprising results. We recommend running `.check()` with the flag `circuit_treeified=True` to validate your correspondence when you first create it.

Before running the code below, try drawing out what the treeified graph will look like on a whiteboard and check it against the writeup.


```python
ex = Experiment(loss, ds, corr, random_seed=11)
treeified_circuit = ex.treeified()
assert ex.nodes == corr
corr.check(treeified_circuit, circuit_treeified=True)
treeified_circuit.print_html()

```

Now compare that with the result of calling `loss.print_html()` (before treeifying). Note that you get a computational graph that looks essentially the same as the one above. `xs` appears nine times, but the copies of it except for the child of A say "(repeat)" . That is, before calling `.treeified()`, the `xs`'s are all the same actual node.

Now with the treeified model, we can replace the inputs. Normally this would happen behind the scenes in an experiment, but we'll step through it here. In order to show what's happening as we step through in a nice way, we wrote some custom printing logic in the cell below. When you call `ex.circuit.print(po)`, nodes are colored to match their corresponding `InterpNode`s. Dark grey nodes are outside the image of the correspondence (or we haven't gotten to replacing them yet), and light grey nodes are in matched paths but not directly mapped to. The annotation shows the data we sampled at that node, and the value of the `InterpNode`'s function on that datum.


```python
circuit = ex.wrap_in_var(treeified_circuit, ref_datum)
already_scrubbed: Dict[rc.IterativeMatcher, InterpNode] = {}
already_scrubbed_inputs: Dict[rc.IterativeMatcher, InterpNode] = {}


def colorer(c: rc.Circuit, super: rc.Circuit) -> str:
    color = "darkgrey"
    for (m, i) in already_scrubbed.items():
        m_endpoints = m.get(super)
        if c in m_endpoints:
            color = color_dataset(sampled_inputs.datasets[i], html=True)
        elif color == "darkgrey" and i.is_leaf() and c.are_any_found(m_endpoints):
            color = "lightgrey"
    for (m, i) in already_scrubbed_inputs.items():
        m_endpoints = m.get(super)
        if c in m_endpoints:
            color = color_dataset(sampled_inputs.datasets[i], html=True)
    return color


def commenter(c: rc.Circuit, super: rc.Circuit) -> str:
    comment = ""
    for (m, i) in already_scrubbed.items():
        if c in m.get(super):
            comment = i.str_samplers(sampled_inputs)
    for (m, i) in already_scrubbed_inputs.items():
        if c in m.get(super):
            comment = f"ds set by '{i.name}': {sampled_inputs[i]}"
    return comment


po = rc.PrintHtmlOptions(
    colorer=lambda c: colorer(c, circuit),
    traversal=rc.IterativeMatcher.noop_traversal(),
    commenters=[lambda c: commenter(c, circuit)],
)


def step_scrubber(ex: Experiment, circuit: rc.Circuit):
    for (interp_node, m) in ex.nodes.in_dfs_order():
        print(interp_node)
        input_matcher = to_inputs(m, ex.dataset)
        already_scrubbed[m] = interp_node
        already_scrubbed_inputs[input_matcher] = interp_node
        circuit = ex._replace_one_input(circuit, interp_node, m, sampled_inputs)
        yield circuit


step_scrub = step_scrubber(ex, circuit)
circuit.print(po)

```

As you run the cell below, you should find that these are exactly the data that we sampled earlier--we are now using those samples to replace the inputs to our circuit. We're replacing the inputs with the corresponding interp node's `ds` if it is a leaf, and its `other_inputs_ds` if it's not (see the Comments section below for more info on this); some inputs will then be replaced again by child interp nodes if they exist.

You might find it useful to expand the nodes (click a > to turn it into a v) so you can see the entire treeified circuit, including the colors and annotations at the inputs. By default repeated nodes will be collapsed (including sometimes other copies of the input).

Note: In this example, there is no input downsteam of `D'` that is not also downstream of `A'` or `B'`, so eventually no input will be run on `D'`'s 'other inputs datum'. But conceptually the step is still taken.


```python
print("We set _all_ inputs downstream of 'out' with out's 'other inputs datum'.")
circuit = next(step_scrub)
circuit.print(po)
print("We set all inputs downstream of 'D' with D's 'other inputs datum'.")
circuit = next(step_scrub)
circuit.print(po)
print("We set all inputs downstream of 'A' with A's 'other inputs datum'.")
circuit = next(step_scrub)
circuit.print(po)
print("We set all inputs downstream of 'x0'. As x0 is a leaf node we get to use its main datum.")
circuit = next(step_scrub)
circuit.print(po)
print("Now onto B, again with the 'other inputs datum'.")
circuit = next(step_scrub)
circuit.print(po)
print("B has one child -- x1. x1 is a leaf, so we use its main datum to set inputs")
circuit = next(step_scrub)
circuit.print(po)
print("And finally set the inputs upstream of labels (also a leaf interp node)")
circuit = next(step_scrub)
circuit.print(po)
try:
    next(step_scrub)
    assert False, "should be done scrubbing at this point!"
except StopIteration:
    pass
treeify_first = circuit

```

Now that you have seen how the recursion works, the good news is you won't have to do it by hand again. In practice, we can replace inputs without explicitly treeifying--the treeification will happen automatically as inputs to different paths are replaced!


```python
circuit = ex.base_circuit
circuit = ex.wrap_in_var(circuit, ref_datum)
circuit = ex.replace_inputs(circuit, sampled_inputs)
circuit.print()

```

Let us make sure that the `.replace_inputs()` call indeed did what we want it to. The two resulted circuits should be identical, except that the circuit that we treeified by hand has some tags, which have random uuids. We'll remove the extra tags, and replace all tags with the same uuid.


```python
clean_tags = lambda c: c.update(
    lambda x: x.is_tag(), lambda x: rc.Tag(x.node, uuid.UUID("00000000-0000-0000-0000-000000000000"))
)
treeify_first_clean = clean_tags(treeify_first).update(lambda x: x.is_tag() and x.node.is_var(), lambda x: x.node)
assert treeify_first_clean == clean_tags(circuit)

```


### Computing the behavior function on the scrubbed model

Now we just run the scrubbed model forward!


```python
device = "cpu" if NotebookInTesting.currently_in_notebook_test else "cuda:0"
eval_settings = ExperimentEvalSettings(device_dtype=device)
scrubbed_out = ScrubbedExperiment(circuit, ref_datum, sampled_inputs, ex._group, ex._nodes).evaluate(
    eval_settings=eval_settings
)
print(scrubbed_out)

```

In practice, we run causal scrubbing on a batch and take the expectation. Recall that our goal is to get as close as possible to the value of the behavior function on the original inputs. How did our hypothesis perform, and why?


```python
out_batch = Experiment(loss, ds, corr, num_examples=5000, random_seed=11).scrub().evaluate(eval_settings)
print(f"Scrubbed: {out_batch.mean()}")
print(f"Original: {loss_with_data.evaluate().mean()}")

```

## Exercises

Now, let's out some different hypotheses and see how they do.

### Exercise 1
Suppose in the example above, you figured out that the output of the model is checking if either `x0` or `x1` is greater than 3, but you haven't figured out how the model is doing this computation. That is, you haven't formed any hypothesis about what `A`, `B`, and `C` are doing. How would you express this hypothesis? How does this hypothesis perform?


```python
"TODO: Your code here"

```

### Exercise 2
You suspect that `x0` directly influences `A` a lot, and you want to test this. How would you do this?


```python
"TODO: Your code here"

```

### Exercise 3
You have pinned down the flow of information as
```
x0  x1  y
|   |   |
A   B  /
 \ /  /
  D  /
   \/
  out
```
but don't know what values the nodes represent. How do you write down this hypothesis?


```python
"TODO: Your code here"

```

### Other hypotheses

You can also try out some other hypotheses yourself.


```python
"TODO: Your code here"

```

## Further comments
### Conveniences

The causal scrubbing code provides a number of conveniences to make expressing your hypotheses easier, rather than forcing you to manually specify a graph isomorphism between your interpretation and the model.

### IterativeMatcher

Recall that in our writeup, we talk about the correspondence mapping to single nodes in the treeified model. But, in practice this is quite cumbersome.

Suppose you had the diamond graph (where A is the input and D is the output)
```
  A
 / \
B   C
 \ /
  D
```
and you wanted to hypothesize that A does some computation and later D uses that information to do something. You might want to
draw the interpretation $A \to D$; but there is no such edge in the model, so you can't write down a homomorphism c from $I$ to $G$!

Or, suppose you were analyzing a neural network. You had analyzed the attention heads and thought 0.1 (layer.head) and 2.1 were important while
no heads in layer 1 were, but you had not yet looked at the MLPs. How would you express the hypothesis that paths from 0.1 to 2.1,
potentially passing through MLPs but not through layer 1 heads, were important?

In both of these cases, you could rewrite your interpretation with more nodes -- in the first example, adding nodes to your
interpretation corresponding to B and C. Or, you could do a circuit rewrite to group together some of the nodes in your model.
Both of these are pretty annoying to do, so instead we allow you to correspond an interpretation node to an IterativeMatcher picking
out some paths in the model. That is, you can say things like:
```
correspondence.add(a_I, rc.IterativeMatcher("D").chain("A"))
```

You can also write IterativeMatchers that match more than one node, like `rc.IterativeMatcher("D").chain({"B", "C"})`, and use these in your correspondence. I think these can be harder to reason about, especially when you are matching nodes where one is downstream of another, but can save you the trouble of adding nodes to your correspondence or doing a circuit rewrite.

### Implicit extension to inputs

If you have not explicitly specified an interpretation for some of the paths between $c(I)$ (that is, that the set of nodes mapped to from the interpretation graph via the correspondance) and the inputs, this is filled
in for you. After all, we must have inputs for every path through the model! In other words, for every matcher in your
interp graph, all the inputs upstream of nodes that matcher matches will be replaced. They will be replaced by the interp
node's `ds` if they are leaves, and its `other_inputs_ds` if they are not (see the next section); some inputs will then be
replaced again by child interp nodes if they exist.

The conditional behavior here can be confusing. Remember, in the causal scrubbing algorithm we ultimately don't care about the intermediate datasets we sampled--those were just a tool for helping us express our hypothesis--and we just need the datasets for the treeified inputs to pass into our scrubbed model. Also, remember that the other_inputs_ds is a default for what to do with paths we haven't explicitly included in our hypothesis.

To do this extension we have to know what the "inputs" are. The names of the inputs are defined in the dataset you supply to the
experiment, and are expected to be present in the circuit you pass in.


### More on: nodes not in any matched paths

We mentioned earlier that each interpretation node $n_I$ samples a `other_inputs_ds`. If c($n_I$) has children that are not in
the paths defined by c(children($n_I$)), they will be run on this dataset. For instance, in the example we looked at, the input to
`x1` and `x2` going into `A` was drawn from `A_prime`'s `other_inputs_ds`. This allows you to very easily say that "the other
children don't matter" -- by default, `other_inputs_ds` is sampled randomly from the source dataset.

But you can also use this in other ways: for example, you can set $n_I$'s other_inputs_sampler to an `ExactSampler`, which represents
"c(children($n_I$)) matter in some way that I'm claiming; and the other children I don't want to touch at this point, they can't be resampled at all".

```
n_I = InterpNode(cond_sampler=FuncSample(f), other_inputs_sampler=ExactSampler(), name="n")
```

Sometimes we might call the set `children(c(n_I)) \ c(children(n_I))` the "unimportant" children as a shorthand, even though they
are not necessarily unimportant. We also sometimes refer to them as "not in the image of c".

### Reference dataset for extending the graph

When you call `.scrub()` on an `Experiment`, it returns a `ScrubbedExperiment` object, which has the attribute `ref_ds`, the reference dataset. If the scrubbed circuit computes your entire behavior function, then you don't need this dataset. But, if you want to feed the result of your scrubbed circuit into a continued behavior function, you should use the `ref_ds` for any computations there. For example, if the circuit computes logits and you want to compute a loss, you should use the reference dataset to get the labels.

### Checks

The code performs a number of checks to make sure your correspondence induces well-defined scrubs. For examples, take a look at `test_hypothesis.py` and `test_experiment.py`. But we'll briefly list out the checks here as well:

- Out maps to out
- In maps to in
- Corrrespondence must map every node in the interp graph to something
- Must respect model tree structure: An interp node $I$ is a child of $I'$ iff each path in $c(I)$ is an extension of a path in $c(I')$
- Each matcher must match something
- No two matchers can match the same path
- Bonus: After treeifying, no two matchers can match the same node. this should be equivalent to the above.

Most of these won't run if you disable checks on your experiment; but some are mandatory (like the fact that each matcher must match something).

### Generalization

Our code is also more general than the algorithm we describe in our writeup in a couple of useful ways.

1. Each interp node, rather than computing a function and sampling data that agree with the parent datum on that function, is associated with a sampler. This is convenient for lots of reasons, like generating data on the fly, but is also fundamentally more general: you do not have to sample uniformly from a subset of the data!
2. The interpretation is not a computational graph. Again, some things are simply more convenient to express as conditional on the dataset rather than the outputs of a computational graph--for example, sampling some things together, enforcing agreement on a feature the samples for different nodes, phi rewrites. But, conditions are more powerful than agreement on a function; we won't elaborate on that here.
