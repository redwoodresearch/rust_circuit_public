# %%
"""
# REMIX Day 5, Part 3 - creating your tools

This notebook gives you a tour of the classic tools to use to explore language model internals. We'll build the tools ourselves and run our first experiments. We'll also learn to interpret (sometimes confusing) experimental results.

<!-- toc -->

## Learning Objectives

After today's material, you should be able to:

- Understand the motivation and the implementation of path patching
- Use `iterative_path_patching` to implement your own experiments
- Understand the motivation behind "moving pieces experiments" and their implementation
- Gather attention patterns and visualize them with the CUI
- Use helper functions to handle grouping of nodes and creation of causal scrubbing hypothesis and matchers

## Readings

- [Exploratory interp exercises presentation](https://docs.google.com/document/d/1qyHT4W9TtVL77AMKN514SjXT9fyNS70DJH9FFQ7YiDg/edit?usp=sharing)
- [Introduction to path patching](https://docs.google.com/document/d/1FWJUwnD50-IMrr92K6w3LIaAjhJp-HBN7ixvXBJM70o/edit?usp=sharing)
* The [slides from the lecture](https://docs.google.com/presentation/d/13Bvmo8E6N5qhgj1yCXq5O7zNRzNNXZLzexlgdzdgZ_E/edit?usp=sharing) for the terminology.
"""
import os
import sys
from interp.circuit.causal_scrubbing.dataset import Dataset
from interp.circuit.causal_scrubbing.experiment import (
    Experiment,
    ExperimentEvalSettings,
)
from interp import cui
from interp.ui.very_named_tensor import VeryNamedTensor


# %%
from remix_d5_utils import (
    IOIDataset,
    load_and_split_gpt2,
    load_logit_diff_model,
)

MAIN = __name__ == "__main__"  # this notebook will be imported. We don't want to run long experiment during import

if MAIN:
    from remix_extra_utils import check_rust_circuit_version
    check_rust_circuit_version()

if "SKIP":
    # Skip CI for now - avoids downloading GPT2
    IS_CI = os.getenv("IS_CI")
    if IS_CI:
        sys.exit(0)

# TBD: remove this as it breaks CI?
if "SKIP":
    if MAIN:
        get_ipython().run_line_magic("load_ext", "autoreload")
        get_ipython().run_line_magic("autoreload", "2")

import time
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union, cast

import plotly.express as px
import rust_circuit as rc
import torch
import torch as t
from interp.circuit.causal_scrubbing.hypothesis import (
    CondSampler,
    Correspondence,
    ExactSampler,
    InterpNode,
    UncondSampler,
    chain_excluding,
    corr_root_matcher,
)
from interp.circuit.circuit_model_rewrites import (
    AttnSuffixForGpt,
    HeadOrMlpType,
    MLPHeadAndPosSpec,
)
import remix_utils


# %%
"""

## Claim 1

We take as a running example the claim: "Attention heads directly influencing the logits are either not influencing IO and S logits, or are increasing the IO logits more than the S.". At the end of this notebook, you should be able to have a nuanced view of what this claim means and to what extent it is correct.

### Setup

Before that, we'll need to import the model and the dataset using the code we wrote in the first two notebooks.

### Creating the dataset. 

Despite our object being able to support multiple templates that are not aligned (e.g. position of IO varies from sequence to sequence), we will only use one template for this demo. 

Thus, the name position is the same for all sequences. Because sentences are aligned, we can define global variables for the position of the tokens.
"""

# %%

ioi_dataset = IOIDataset(prompt_type="BABA", N=50, seed=42, nb_templates=1)

MAX_LEN = ioi_dataset.prompts_toks.shape[1]

for k, idx in ioi_dataset.word_idx.items():  # check that all the sentences are aligned
    assert (idx == idx[0]).all()

END_POS = int(ioi_dataset.word_idx["END"][0].item())
IO_POS = int(ioi_dataset.word_idx["IO"][0].item())
S1_POS = int(ioi_dataset.word_idx["S1"][0].item())
S2_POS = int(ioi_dataset.word_idx["S2"][0].item())


# %%

"""
### Defining Dataset Variation

We use the `gen_flipped_prompts` method to create datasets we will use in this notebook. We defined them at the start of the notebook, so we make sure that they are the same for all experiments (to avoid the case where running a cell twice leads to different results).

Exercise: print the first 5 sentences of the `flipped_IO_dataset` and `flipped_S_dataset` and make sure you understand what information they hold. What are the families of all these datasets?

"""
# %%
# a dataset where the IO token is fipped
flipped_IO_dataset = ioi_dataset.gen_flipped_prompts("IO")
flipped_S_dataset = ioi_dataset.gen_flipped_prompts("S")

flipped_IO_S_dataset = ioi_dataset.gen_flipped_prompts("IO").gen_flipped_prompts("S")

flipped_IO_S1_order = ioi_dataset.gen_flipped_prompts("order")
# %%

"""
### Model Loading

We will then load the model using the steps described previously. First the main circuit, then the logit diff (ld for short) circuit for path patching experiments. `group` will be used to keep the labels in sync with the inputs.
"""
circuit = load_and_split_gpt2(MAX_LEN)

io_s_labels = torch.cat([ioi_dataset.io_tokenIDs.unsqueeze(1), ioi_dataset.s_tokenIDs.unsqueeze(1)], dim=1)
ld_circuit, group = load_logit_diff_model(circuit, io_s_labels)

# %%
"""
It's always important to check that our model is working as expected before running any experiments.
"""

c = ld_circuit.update(
    "tokens",
    lambda _: rc.DiscreteVar(rc.Array(ioi_dataset.prompts_toks, name="tokens"), probs_and_group=group),
)

if MAIN:
    transform = rc.Sampler(rc.RunDiscreteVarAllSpec([group]))
    results = transform.sample(c).evaluate()

    print(f"Logit difference for the first 5 prompts: {results[:5]}")
    print(f"Average logit difference: {results.mean()} +/- {results.std()}")

    ref_ld = results.mean()

    assert ref_ld > 2.5 and ref_ld < 4  # usual range
# %%
"""
## Experiments

Now that we're all set, let's think about experiments!

We want to prove (or disprove) the claim: "Attention heads directly influencing the logits are either not influencing IO and S logits, or are increasing the IO logits more than the S."

Pause for a moment to think about what would be the first step to test this claim.

* How can you divide this claim into smaller claims?
* What is the first experiment you want to run?

<details>

<summary>Solution</summary>

* First, we have to find heads that directly influence the logits. We're looking at their effect by only considering the final layer norm and unembeddings as intermediate computations. 
* Then, we need to identify the *direction* of the effect of these heads on the IO and S logits. A way to summarize this is to look at the logit difference. 

Experimentally, we can implement this using the following techniques:

* Path patching to the logits
* Simple causal scrubbing hypothesis
* Projecting the output of the head using the unembedding matrix (the logit lens).

Here we'll focus on path patching. This technique is the most well-suited for this kind of experiment. Go through the docs (including the exercises!) introducing [path patching](https://docs.google.com/document/d/1FWJUwnD50-IMrr92K6w3LIaAjhJp-HBN7ixvXBJM70o/edit?usp=sharing) and then come back here.
</details>


"""
# %%

"""
## Build your tools

### Path patching

Let's implement path patching. From the description in the doc above, you can have the feeling that it looks like causal scrubbing. However, the implementation will be much simpler because there is no complicated dataset computation: only two are necessary. The similarity is the operation "change the input to one branch, but not to another".

Note: in this document, we'll use "matcher" to sometimes refer to `Matcher` and sometimes to `IterativeMatcher` - hopefully it will be clear from the context.

In this section we will create two main functions:

* `path_patching` takes as input a matcher (specifying the path to patch) and returns the patched circuit where inputs through the path are replaced by the patched input and all other inputs are set to a baseline input.

Once we can perform path patching, we are often interested in answering questions of the form "Given that I found the influence A->B, what is directly influencing A?". To answer this question, we will iterate over each node N that comes before A and run the path patching `N->A->B` (note that we are filtering downstream effect because B appears). Then we select the N that leads to the greater effect size: they are the ones directly influencing A. This is the motivation behind `iterative_path_patching`: make it easy to iterate over candidate steps like "connect N to A" extending already known paths.

* Concretely, the idea of `iterative_path_patching` is to expand hypothesis H by making some nodes grow "by one step". For instance, "grow by one step" can mean "starting on node A, add the direct path to head 5.2", where A is a parameter. This "grow by one step" operation is implemented using _matcher extenders_. A matcher extender is a function of type signature `IterativeMatcher -> IterativeMatcher`. `iterative_path_patching` takes as arguments a list of matcher extenders. In practice, a single matcher extender consists of a `.chain(node)` operation where `node` is a fixed node.

The loop implemented by `iterative_path_patching` is:

```
For each matcher extender E:
    patching_matcher = empty matcher
    For each node to connect in H:
        patching_matcher = patching_matcher U E(node)
    path_patching(patching_matcher)
```

Technical detail: there is no straightforward way to define an empty matcher, you might want to initialize `patching_matcher` with the first matcher extender.

In the code, we use causal scrubbing hypotheses for convenience, but they are never run. We use them as an easy way to store matchers and to specify which node we want to connect.

If `hypothesis` is a `Correspondence`, you can access the matcher of the `InterpNode` `node` using `hypothesis.corr[node]`.

Exercise: Implement `path_patching` and `iterative_path_patching` using the following skeleton. We provide you with the function `replace_inputs` that will replace the input starting from a given matcher by a `DiscreteVar` sampling from a given `Tensor`.

* For `path_patching` you can use a different `array_suffix` in `replace_inputs` when you replace by the baseline data and patched data. Then by inspecting the graph you can see if the correct inputs are replaced.

* For `iterative_path_patching` you have to implement the inner loop. You also have to increment the variable `nb_not_found` to count the number of times the inputs are not found after applying the matcher extender. This can happen if the extender tries to connect a node N that is at a later layer than the node to A. In this case, the path `N->A` does not exist.

In the next cell, you can find the definition of matcher extender to debug your implementation.
"""
# %%


def replace_inputs(
    c: rc.Circuit,
    x: torch.Tensor,
    input_name: str,
    m: rc.IterativeMatcher,
    group: rc.Circuit,
    array_suffix: str = "_array",
):
    """
    Replace the input on the model branch define by the matcher `m` with a DiscreteVar.
    The input in the circuit `c` are expected non batched.
    """
    assert x.ndim >= 1
    c = c.update(
        m.chain(input_name),
        lambda _: rc.DiscreteVar(
            rc.Array(x, name=input_name + array_suffix),
            name=input_name,
            probs_and_group=group,
        ),
    )
    return c


def path_patching(
    circuit: rc.Circuit,
    baseline_data: torch.Tensor,
    patch_data: torch.Tensor,
    matcher: rc.IterativeMatcher,
    group: rc.Circuit,
    input_name: str,
) -> rc.Circuit:
    """Replace the input connected to the paths matched by `matcher` with `patch_data`. All the other inputs are replaced with `baseline_data`.
    Return the patched circuit where inputs are DiscreteVar using the sampling group `group`."""
    "SOLUTION"
    baseline_circuit = replace_inputs(
        circuit,
        baseline_data,
        input_name,
        corr_root_matcher,
        group,
        array_suffix="_baseline",
    )
    if len(matcher.get(circuit)) == 0:
        return baseline_circuit
    patched_circuit = replace_inputs(
        baseline_circuit,
        patch_data,
        input_name,
        matcher,
        group,
        array_suffix="_patched",
    )
    return patched_circuit


def iterative_path_patching(
    circuit: rc.Circuit,
    hypothesis: Correspondence,
    nodes_to_connect: List[InterpNode],
    baseline_data: torch.Tensor,
    patch_data: torch.Tensor,
    group: rc.Circuit,
    matcher_extenders: List[Callable[[rc.IterativeMatcher], rc.IterativeMatcher]],
    input_name: str,
    output_shape: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    """
    This function apply a set of `matcher_extenders` the matchers from `nodes_to_connect` in the `hypothesis`. The result is the concatenation of the circuit outputs after the application of each matcher_extender.

    * circuit - the circuit to patch
    * hypothesis - a causal scrubbing hypothesis. No causal scrubbing method is used in the code, it's just a convenient way to store matchers linked to nodes and limit the number of object to keep track.
    * nodes_to_connect - the InterpNode from the hypothesis where we want to expand
    * baseline_data - the baseline data to use for the replacement
    * patch_data - the patch data to use for the replacement
    * group - the group for the DiscreteVar
    * matcher_extenders - a list of function that take a matcher and return a new matcher. This define in which way we want to make the hypothesis grow. FOr example, one element of this list is a `.chain` operation that matches one specific attention head by the most direct path.
    * input_name - the name of the input in the circuit
    * output_shape - Optionnal reshaping of the result. If None, the output shape is `(len(matcher_extenders)) + the shape of the circuit ouput (can be different from `circuit.shape` if there is a batch dimension added).
    """

    t1 = time.time()
    circuits = []
    sampler = rc.Sampler(rc.RunDiscreteVarAllSpec([group]))
    nb_not_found = 0
    for matcher_extender in matcher_extenders:

        if "SOLUTION":
            matchers_to_h = []
            for node in nodes_to_connect:
                matchers_to_h.append(matcher_extender(hypothesis.corr[node]))
            union_matcher = matchers_to_h[0]

            for matcher in matchers_to_h[1:]:
                union_matcher = union_matcher | matcher

            if len(union_matcher.get(circuit)) == 0:
                nb_not_found += 1
            patched_circuit = path_patching(circuit, baseline_data, patch_data, union_matcher, group, input_name)
        else:
            raise NotImplementedError("Inner loop not implemented!")
        patched_circuit = sampler(patched_circuit)  # we replace discrete vars by the real arrays
        circuits.append(patched_circuit)

    if nb_not_found > 0:
        print(f"Warning: No match found for {nb_not_found} matcher extenders")

    # a fancy function to evaluate fast many circuit that share tensors in common
    results = rc.optimize_and_evaluate_many(
        circuits,
        rc.OptimizationSettings(scheduling_simplify=False, scheduling_naive=True),
    )
    t2 = time.time()
    print(f"Time for path patching :{t2 - t1:.2f} s")
    if output_shape is None:
        return torch.cat([x.unsqueeze(0) for x in results], dim=0)

    return torch.cat(results).reshape(output_shape)


# %%

matcher = rc.Matcher("final.input").chain(
    rc.restrict(
        rc.Matcher("a2.p_bias"),  # arbitrary target
        start_depth=1,
        end_depth=2,
    )
)

# a matcher that match all the paths that are not taken by `matcher`.
complement_matcher = rc.Matcher("final.input").chain(
    rc.restrict(
        ~rc.Matcher("a2.p_bias"),  # the complement operation can only be made on Matcher
        start_depth=1,
        end_depth=2,
    )
)

if MAIN:
    patched_circuit = path_patching(
        ld_circuit,
        baseline_data=ioi_dataset.prompts_toks,
        patch_data=flipped_IO_dataset.prompts_toks,
        group=group,
        matcher=matcher,
        input_name="tokens",
    )

    patched_circuit.print_html()

    ## Test your implementation

    patched_array = matcher.chain("tokens").chain(rc.Array).get(patched_circuit)
    non_patched_array = complement_matcher.chain("tokens").chain(rc.Array).get(patched_circuit)

    assert len(patched_array) == 1 and len(non_patched_array) == 1

    assert list(patched_array)[0].name == "tokens_patched"
    assert list(non_patched_array)[0].name == "tokens_baseline"


# %% path patching debug example


def extender1(m: rc.IterativeMatcher) -> rc.IterativeMatcher:
    return m.chain(
        rc.restrict(
            rc.Matcher("m8.p_bias"),
            end_depth=9,  # we use the end_depth to select only the most direct path
            term_if_matches=True,
        )
    )


def extender2(m: rc.IterativeMatcher) -> rc.IterativeMatcher:
    return m.chain(
        rc.restrict(
            rc.Matcher("a2.p_bias"),
            end_depth=9,
            term_if_matches=True,
        )
    )


if MAIN:
    # we only have our logit as a root
    corr = Correspondence()
    i_root = InterpNode(ExactSampler(), name="logits")
    m_root = corr_root_matcher
    corr.add(i_root, m_root)

    results = iterative_path_patching(
        circuit=ld_circuit,
        hypothesis=corr,
        nodes_to_connect=[i_root],
        baseline_data=ioi_dataset.prompts_toks,
        patch_data=flipped_IO_dataset.prompts_toks,
        group=group,
        matcher_extenders=[extender1, extender2],
        input_name="tokens",
    )

    print(results.shape, results.mean(dim=-1))

    # checking the results. Make sure you run the dataset definition cell only once!
    assert torch.isclose(results[:, 0], torch.tensor([1.6240, 1.6022]), atol=1e-3).all()
# %%
"""
### Our first path patching experiment

Instead of defining matcher extenders by hand, we often define an extender factory that takes a parameter (e.g. the layer and head number) and returns an extender that reaches this targeted head from an arbitrary starting point.

If our target head is H and we investigate its _direct effect_ on the logit, we don't allow paths of the form H->A->logits where A is an arbitrary node. To translate this into matchers, we'll use the `term_early_at` argument of `restrict`. This argument allows us to stop the traversal when we reach a certain node. In our case, we want to stop the traversal when we reach a node that is _not_ one we specify.

To this end, we define `ALL_NODES_NAMES` to be the set of all the MLP and attention heads at a particular position. We can then recover the set of names of all but the target: each time we come across a name in this set, we should stop.

To define `ALL_NODES_NAMES`, we use the class `MLPHeadAndPosSpec` that handles attention head and MLP nodes. It includes helpful methods such as `to_name` that return the name of the node given a prefix.

Advanced details:

If you look at the code below, you can spot two additional details in addition to this story. What are they?

<details>
<summary>Click here to see the answer</summary>

* We add a `qkv` parameter to the extender factory. This parameter is used to restrict the extender to a particular qkv head. This is useful to allow paths that go through only Q, K or V of heads.
* We add `rc.new_traversal(start_depth=1, end_depth=2)` before specifying the direct path. This is because the starting point of the matcher is part of `ALL_NODES_NAMES`, we don't want to stop at the root! So we force the matcher to go one level deeper.

</details>

Note: we heavily rely on `end_depth` when defining matchers. This makes them easier to understand, but they are also much more brittle! A single rewrite of the circuit can mess up the depth of the nodes we are interested in. Beware when copy-pasting such definitions in your project, and always print your circuit to be sure what you're matching.
"""

# %%
ALL_NODES_NAMES = set(
    [
        MLPHeadAndPosSpec(l, cast(HeadOrMlpType, h), pos).to_name("")
        for l in range(12)
        for h in (list(range(12)) + ["mlp"])  # type: ignore
        for pos in range(MAX_LEN)
    ]
)


def extender_factory(node: MLPHeadAndPosSpec, qkv: Optional[str] = None):
    """
    `qkv` define the input of the attention block we want to reach.
    """
    assert qkv in ["q", "k", "v", None]

    node_name = node.to_name("")
    nodes_to_ban = ALL_NODES_NAMES.difference(set([node_name]))

    if qkv is None:
        attn_block_input = rc.new_traversal(start_depth=0, end_depth=1)
    else:
        attn_block_input = rc.restrict(f"a.{qkv}", term_if_matches=True, end_depth=8)

    def matcher_extender(m: rc.IterativeMatcher):
        return m.chain(attn_block_input).chain(
            rc.new_traversal(start_depth=1, end_depth=2).chain(
                rc.restrict(
                    rc.Matcher(node_name),
                    term_early_at=rc.Matcher(nodes_to_ban),
                    term_if_matches=True,
                )
            )
        )

    return matcher_extender


matcher_extenders = [
    extender_factory(MLPHeadAndPosSpec(l, cast(HeadOrMlpType, h), END_POS), qkv=None)
    for l in range(12)
    for h in list(range(12)) + ["mlp"]  # type: ignore
]

"""

### Let's run our first experiment!

Question: Question: Does the experiment affect the output of heads and MLPs that come after the patched connection? (i.e. do we filter for downstream effects?)

<details>
<summary>Click here to see the answer</summary>
    Because we're path patching N->logits, any potential downstream effect would come _after_ the logits. However, the logits are the output of our model, so there is nothing to filter here. Filtering downstream effects doesn't mean anything for this particular experiment.
</details>
"""
# %%
if MAIN:
    # we only have our logit as a root
    corr = Correspondence()
    i_root = InterpNode(ExactSampler(), name="logits")
    m_root = corr_root_matcher
    corr.add(i_root, m_root)

    results_IO = iterative_path_patching(
        circuit=ld_circuit,
        hypothesis=corr,
        nodes_to_connect=[i_root],
        baseline_data=ioi_dataset.prompts_toks,
        patch_data=flipped_IO_dataset.prompts_toks,
        group=group,
        matcher_extenders=matcher_extenders,
        input_name="tokens",
        output_shape=(12, 13, -1),
    )

# %%
"""
#### Visualizing the results

We use [plotly](https://plotly.com/python/) to plot the results. It produces interactive graphs made from HTML. You can hover over the graph to see the exact values of each entry. 

Some tricks that make results easier to visualize:

Format the results:

* We reshape the results to have a 12x13 matrix, where the 13th column is the mlp
* We compute the mean of the results over the 3 heads

Plotly tricks:

* Center the color map so white is zero
* Add labels to the axis

Feel free to reuse `show_mtx` for your experiments. The default value of the "title" variable is a nudge to encourage you to always define it ;)
"""


def show_mtx(mtx, title="NO TITLE :(", color_map_label="Logit diff variation"):
    """Show a plotly matrix with a centered color map. Designed to display results of path patching experiments."""
    # we center the color scale on zero by defining the range (-max_abs, +max_abs)
    max_val = float(max(abs(mtx.min()), abs(mtx.max())))
    x_labels = [f"h{i}" for i in range(12)] + ["mlp"]
    fig = px.imshow(
        mtx,
        title=title,
        labels=dict(x="Head", y="Layer", color=color_map_label),
        color_continuous_scale="RdBu",
        range_color=(-max_val, max_val),
        x=x_labels,
        y=[str(i) for i in range(mtx.shape[0])],
        aspect="equal",
    )
    fig.show()


if MAIN:
    variation_ld_flipped_IO = results_IO.mean(dim=-1) - ref_ld

    show_mtx(variation_ld_flipped_IO, title="Logit diff variation (flipped IO)")

# %%

"""
Question: How do you interpret this plot? (you can try to answer by dividing two parts: "Observation" and "Interpretation"). What can you conclude about the claim just by looking at this plot?

Reminder: we're trying to investigate the claim "Attention heads directly influencing the logits are either not influencing IO and S logits, or are increasing the IO logits more than the S."

<details>
<summary>Click here to see the answer</summary>

### Observation

First, we observe that most of the heads in layers earlier than 7 don't influence the logit diff (the difference in logit diff is < 1% of `ref_ld`).  

We can identify heads that are directly influencing logits. After path patching, some lead to a decrease in logit diff (e.g. 9.9, 9.6, and 10.0) while for others, we observe a higher logit diff after patching them (e.g. 10.7 and 11.10). 

### Interpretation

For the heads causing a decrease: they are run on an input unrelated IO (let's call it A), so they are pushing to increase logit A - logit S instead of logit IO - logit S. The intervention leads to a decrease in total logit diff.

For the heads causing an increase in logit diff: when they are run on unrelated input, the final logit diff is higher. This means that on a fixed sentence for a random A, their contribution to logit A - logit S is greater than logit IO - logit S. This suggests that in normal conditions, they are pushing against the IO logit.

However, we cannot conclude the claim yet. We only compared the contribution to logit IO with a logit A, for a random name. We need to intervene the S logit to compare the relative contribution to IO _and_ S.
</details>

"""
# %%
"""

### Exercise!

We used `flipped_IO_dataset = ioi_dataset.gen_flipped_prompts("IO")` to generate the dataset to patch from. 

Question: What would happen if we use the `flipped_S_dataset` defined by `flipped_S_dataset = ioi_dataset.gen_flipped_prompts("S")`? (i.e. the dataset is still from the IOI family but both occurrences of S are replaced by the same random name)

(It's not an easy exercise, it's unlikely you will be able to predict the result. However, it's a good way to practice thinking ahead about what would surprise you before running an experiment.)

Question: Run the experiment and show the results on the `flipped_S_dataset`. How to conclude the claim given these new pieces of evidence?

Hint: We introduce notation to help you formalize the intuition developed in the previous explanation and apply it to this new case.

We call $H(T, x)$ the "contribution" of head $H$ to the logit of token $T$ when run on $x$. This can be seen as the projection along the unembedding vector of $T$ if we neglect the role of the layer norm.
We call $x$ the baseline input $z$ the patched input. If a head H is pushing to increase the logit diff, then we have $H(IO_x, x)$ >> $H(S_x, x)$.

"""
# %%
variation_ld_flipped_S: t.Tensor
if MAIN:
    variation_ld_flipped_S = (
        iterative_path_patching(
            circuit=ld_circuit,
            hypothesis=corr,
            nodes_to_connect=[i_root],
            baseline_data=ioi_dataset.prompts_toks,
            patch_data=flipped_S_dataset.prompts_toks,
            group=group,
            matcher_extenders=matcher_extenders,
            input_name="tokens",
            output_shape=(12, 13, -1),
        ).mean(dim=-1)
        - ref_ld
    )
if MAIN:
    if "SOLUTION":
        show_mtx(variation_ld_flipped_S, title="Logit diff variation (flipped S)")
    else:
        if True:
            show_mtx(variation_ld_flipped_S, title="Logit diff variation (flipped S)")
        else:
            print("Think about what you expect before showing this plot!")
# %%
r"""
**Disclaimer: I tried to make a description as precise as possible. If at some point you feel like they are overly detailed because you understood the intuition, feel free to skip them.**

<details>
<summary>Answer</summary>
We observe a really different plot! 

### Observation 

First, the effect size is about half as big as the previous one. Most heads lead to a positive variation in logit diff after patching. Moreover, there are new heads that appear that were not present in the previous experiment. The plot is less sparse than the first one. 

The heads appearing in the first plot have an opposite effect size in the second plot. The change in absolute effect size varies from being 50% smaller in the second (e.g. 10.0) to being 7x smaller in the second plot (e.g. 9.9).

### Interpretation

Let $H$ be one of the heads that leads to a strong positive variation in logit diff after patching H->logits. 

The logit diff is measuring $H(IO_x, z)-H(S_x, z)$. 
* $S_x$ is a random name from the point of view of $H$ run on $z$. 
* $IO_x$ is the same in both $x$ and $z$. And $x$ and $z$ are similar up to the value of $S$, so we expect $H(IO_x, z) \simeq H(IO_x, x)$
* H lead to a positive variation in logit diff. We isolated the direct path H->logits such that we can consider that the variation in the global logit diff is in fact the variation in the contribution of H to the logit diff. We thus have $H(IO_x, x)-H(S_x, x)$ << $H(IO_x, z)-H(S_x, z)$ (pre patching << post patching) 
* Finally, we have $H(S_x, x)$ >> $H(S_x, z)$ 

The heads we observe in the plot are the heads writing in the direction of $S$. Those are the ones responsible for the fact that the proba of $S$ is much higher that a random name.

### Conclusion

The same argument can be made to formalize the first plot. The only difference is that the effect is reversed (due to the negative sign in the logit diff). The first plot shows heads writing IO more than a random name. 

The heads that appear in the two plots with a flipped sign are writing both S and IO more than random names.

By comparing the two plots, we can conclude the relative importance between $H(IO_x, x)$ and $H(S_x, x)$ (by looking at a single plot, the only reference was a random name). The effect size is much higher for the first plot: there is a sparse set of heads (10.0, 9.6, 9.9) specialized in identifying IO and boosting it much more than a random name. In the second plot, the large set of heads seems to implement a mechanism like "push tokens from the context", with a smaller effect size than in the case of IO.

To find the heads that have higher $H(IO_x, x)$ than $H(S_x, x)$, we can visualize the sum of the two plots (no need to take the difference). Positive values are heads that are pushing S more than IO. Negative values are heads that are pushing IO more than S.

### Back to the claim

Plotting this is the most accurate way to answer the claim as we control for the baseline effect of "writing all names in context more than a random name". In practice, the plots we obtain are really close to the ones we identified in the first plot. Still, we can see for instance that 10.0 is not pushing IO much more than S compared to what the first plot shows.

To conclude, we can clearly observe attention heads significantly contributing to the logit, and pushing for S more than IO (e.g. 10.7 and 11.10). The claim is thus **false**.

</details>

"""
# %%
if MAIN:
    show_mtx(
        variation_ld_flipped_S + variation_ld_flipped_IO,
        title="Sum of the two logit diff variation (flipped IO + flipped S)",
    )

# %%

r"""
That was quite a convoluted explanation! Can you think of a way to show the same thing with only one iterative path patching experiment while still having input in the IOI family?


<details>
<summary>Answer</summary>

If we consider the final logit diff to be the sum of the head contribution, we have:

$logit diff = \sum{H(IO_x, x) - H(S_x, x)}$.

For that, we could just kill one term of the sum we're interested in by patching a head $H$ on input with a different S _and_ a different IO. The difference in logit diff before and after patching would be $H(IO_x, x) - H(S_x, x) - [H(IO_x, z) - H(S_x, z)]$. Both $H(S_x, z)$ and $H(IO_x, z)$ are about the same value: they are the average logit for names unrelated to the sentence $z$. We're left with $H(IO_x, x) - H(S_x, x)$, that's what we care about.

If you run this experiment, you can see that we have results that a close to the previous cell (with the sum of the two first experiments). Actually, by computing `variation_ld_flipped_S+variation_ld_flipped_IO-variation_ld_flipped_S_IO` you can evaluate the error from our simple model where we decompose the logit diff in the sum of head contribution: if the model was perfect, the error should be zero.

</details>

Exercise: write an experiment that addresses the claim using a single run of `iterative_path_patching`
"""
# %%


if "SOLUTION":
    if MAIN:
        variation_ld_flipped_S_IO = (
            iterative_path_patching(
                circuit=ld_circuit,
                hypothesis=corr,
                nodes_to_connect=[i_root],
                baseline_data=ioi_dataset.prompts_toks,
                patch_data=flipped_IO_S_dataset.prompts_toks,
                group=group,
                matcher_extenders=matcher_extenders,
                input_name="tokens",
                output_shape=(12, 13, -1),
            ).mean(dim=-1)
            - ref_ld
        )

# %%
"Plot the difference between your experiment and the previous best guess we had (variation_ld_flipped_S+variation_ld_flipped_IO)"

if "SOLUTION":
    if MAIN:
        show_mtx(
            variation_ld_flipped_S_IO - (variation_ld_flipped_S + variation_ld_flipped_IO),
            title="Residual error from linear model (variation_ld_flipped_S_IO-(variation_ld_flipped_S+variation_ld_flipped_IO) ).",
        )

# %%
"""
### Takeaway from the previous experiments

We went through a convoluted path to answer the claim, and we thoroughly detailed every experimental result. It is supposed to simulate a realistic chain of thoughts of research:

* Start with an experiment (the flipped IO)
* Realize that it did not show what you expected it to show
* Think about a way to make additional experiments to show your point (the flipped S)
* Realize that you could have done things more directly (the flipped S_IO)

In practice, you're of course encouraged to think about the more direct experiment first. Moreover, you might want to detail your thoughts less than what we did here at the risk of not being able to address only a few of the claims. This section shows a standard of "if you think carefully, this is how far you can get by interpreting the results of a single experiment".

### Checkpoint!

Read section 3.1 of the [IOI paper](https://arxiv.org/pdf/2211.00593.pdf). (Don't read section 3.2 to avoid spoilers) 

Notice how the discovery of name movers is different (here we did not use the ABC dataset). It should give you more context on the heads you just identified.

## Advanced Tooling

We could stop here, however, to demonstrate more tools, we'll push further the investigation of these newly identified heads. We'll demonstrate:

* Getting activations of the heads
* Visualizing attention patterns with CUI
* Simple causal scrubbing experiments
* Example of moving pieces experiments

### Helper functions

Before delving into more advanced tools, we'll need to define a few helper functions. We will make our hypothesis grow step by step by adding new nodes that connect to previously discovered nodes. For instance here, we'd like to connect the name movers to the logit nodes.

`extend_corr` enables us to handle hypotheses more easily. It adds an `InterpNode` connected to a node and creates a new matcher by applying a matcher extender.
"""
# %%


def extend_corr(
    corr: Correspondence,
    from_name: str,
    to_name: str,
    matcher_extender: Callable[[rc.IterativeMatcher], rc.IterativeMatcher],
    cond_sampler: CondSampler,
    other_inputs_sampler: CondSampler = UncondSampler(),
):
    prev_node = corr.get_by_name(from_name)
    prev_matcher = corr.corr[prev_node]
    new_node = prev_node.make_descendant(
        name=to_name,
        cond_sampler=cond_sampler,
        other_inputs_sampler=other_inputs_sampler,
    )
    new_matcher = matcher_extender(prev_matcher)
    corr.add(new_node, new_matcher)


# %%
"""
To specify its counterpart in the computation graph, we will define a matcher extender that matches the new head through a direct path.

This matcher extender is slightly different than the one we described above: we will reuse part of the code from `extender_factory`, however, instead of reaching a single target (the `MLPHeadAndPosSpec` object), we want to reach a group of nodes given by their names.

Exercise: inspired by the first matcher extender (see the definition of `extender_factory`), write the body of `add_path_to_group`.

Be careful: here we want to match a set of nodes given by their names.

<details>
<summary>Hint</summary>

You need to use the `term_if_matches` parameter of `restrict`. If we want to reach a set of heads `{H1, H2}` that are not at the same layer, sometimes we want to avoid paths of the form `logits -> heads H1 -> heads H2`. To do that, we'll use the flag `term_if_matches`: once a node is matched, we stop the traversal.
This restriction was unnecessary for `extender_factory` we defined earlier: as we are targeting a node with a name appearing once in the circuit, there is no risk of composition.

</details>

"""

# TBD adding  exercises here?


def add_path_to_group(
    m: rc.IterativeMatcher,
    nodes_group: List[str],
    term_if_matches=True,
    qkv: Optional[str] = None,
):
    """Add the path from a matcher to a group of nodes using chain operation. Different filtering parameters.
    If `term_if_matches=False` and `qkv` is not `None`, the `qkv` restrition will only be applied on the path to the first nodes found starting from `m`, indirect effect will not be restricted by `qkv`."""

    assert qkv in ["q", "k", "v", None]

    nodes_to_ban = ALL_NODES_NAMES.difference(set(nodes_group))
    if qkv is None:
        attn_block_input = rc.new_traversal(start_depth=0, end_depth=1)
    else:
        attn_block_input = rc.restrict(f"a.{qkv}", term_if_matches=True, end_depth=8)

    if "SOLUTION":
        return m.chain(attn_block_input).chain(
            rc.new_traversal(start_depth=1, end_depth=2).chain(
                rc.restrict(
                    rc.Matcher(*nodes_group),
                    term_early_at=rc.Matcher(nodes_to_ban),
                    term_if_matches=term_if_matches,
                )
            )
        )
    else:
        raise NotImplementedError("You need to implement this function")


def extend_matcher(
    match_nodes: List[str],
    term_if_matches=True,
    restrict=True,
    qkv: Optional[str] = None,
):
    def match_func(m: rc.IterativeMatcher):
        if restrict:
            return add_path_to_group(m, match_nodes, term_if_matches=term_if_matches, qkv=qkv)
        else:
            return m.chain(rc.restrict(rc.Matcher(*match_nodes), term_if_matches=term_if_matches))

    return match_func


"""
### Matcher debugging: show path by groups of nodes

Before putting those helper functions in action, we'll create a debugger tool that can show succinctly the path our matcher is taking. Let's begin by giving a name to the nodes we just discovered. For consistency, we'll stick to the old names (name movers). You'll not have the chance to practice creative naming this time!

#### Debugger info

`print_all_heads_paths` print paths matched by an `IterativeMatcher`. Because paths are often long and involve nodes we're not interested in (e.g. layer norms), it applies various filtering:

* Only show nodes parts of `ALL_NODES_NAMES`
* If show_qkv is `True`, also show qkv nodes
* If a node name is a key of the dict `short_names`, print the value in this dict
* Never print the same string twice

Exercise: write the body of the function `keep_nodes_on_path` to filter the nodes we want to keep. You can run the next three cells to debug your implementation.

We'll see how it works in action in a second.
"""

qkv_names = [f"a{i}.q" for i in range(12)] + [f"a{i}.k" for i in range(12)] + [f"a{i}.v" for i in range(12)]


def keep_nodes_on_path(path: list[rc.Circuit], nodes_to_keep: set[str]) -> list[str]:
    """
    Given a path as a list of nodes, create the list of the names of the nodes present in `nodes_to_keep`, in the order they appear in the path."""
    filtered_path = []
    if "SOLUTION":
        for x in path:
            if x.name in nodes_to_keep:  # we keep only the interesting nodes: mlp and attn heads
                filtered_path.append(x.name)
    return filtered_path


def print_all_heads_paths(
    matcher: rc.IterativeMatcher,
    show_qkv=False,
    short_names: Union[Dict[str, str], None] = None,
):

    print_by_class = short_names is not None
    if show_qkv:
        nodes_to_include = set(list(ALL_NODES_NAMES) + ["a.q", "a.k", "a.v"])
    else:
        nodes_to_include = set(ALL_NODES_NAMES)
    nodes_to_include.add("logits")

    all_paths = matcher.get_all_paths(circuit)
    for target, paths in all_paths.items():
        print()
        print(f"--- paths to {target.name} ---")
        already_printed = set()
        for i, path in enumerate(paths):
            if print_by_class:
                nodes_to_print = keep_nodes_on_path(path, nodes_to_include)
                class_to_print = []
                for n in nodes_to_print:
                    if n in short_names:
                        class_to_print.append(short_names[n])  # type: ignore
                    else:
                        class_to_print.append(n)

                p = "->".join(class_to_print)
            else:
                p = "->".join(keep_nodes_on_path(path, nodes_to_include))
            if p in already_printed:
                continue
            already_printed.add(p)
            print(f"Path {i} : {p}")


# %%
"""
Let's add names to our nodes! We all call them "POS_NM" and "NEG_NM" for Positive / Negative Name Movers (short names are better to be printed along paths).
We will keep this list of heads up to date each time we find new nodes to add. It's useful to keep up to date with various variables that store the information about the found nodes. We create a function `add_node_to_pokedex` to handle bookkeeping for us. 
"""

# %%
short_names = {}

grouped_nodes_name: dict[str, list[str]] = {}
grouped_nodes_spec: dict[str, list[MLPHeadAndPosSpec]] = {}


def add_node_to_pokedex(nodes: list[Tuple[MLPHeadAndPosSpec, str]]):
    global short_names, grouped_nodes_name, grouped_nodes_spec
    for node, name in nodes:
        if node not in short_names:
            short_names[node.to_name("")] = name
        if name not in grouped_nodes_name:
            grouped_nodes_name[name] = []
            grouped_nodes_spec[name] = []
        grouped_nodes_name[name].append(node.to_name(""))
        grouped_nodes_spec[name].append(node)


# %%
"""
To avoid duplicate entries in the global variable above, don't run cells like these twice!
"""
add_node_to_pokedex(
    [
        (MLPHeadAndPosSpec(10, 0, END_POS), "POS_NM"),
        (MLPHeadAndPosSpec(9, 6, END_POS), "POS_NM"),
        (MLPHeadAndPosSpec(9, 9, END_POS), "POS_NM"),
    ]
)

add_node_to_pokedex(
    [
        (MLPHeadAndPosSpec(10, 7, END_POS), "NEG_NM"),
        (MLPHeadAndPosSpec(11, 10, END_POS), "NEG_NM"),
    ]
)


# %%
"""
We'll extend the root matcher to include the direct path to the three nodes we found. We'll use the `extend_matcher` function we defined earlier.
"""
# %%
NM_corr = Correspondence()
i_root = InterpNode(ExactSampler(), name="logits")
m_root = corr_root_matcher
NM_corr.add(i_root, m_root)

extend_corr(
    NM_corr,
    "logits",
    "POS_NM",
    extend_matcher(grouped_nodes_name["POS_NM"], term_if_matches=True),
    ExactSampler(),
)

extend_corr(
    NM_corr,
    "logits",
    "NEG_NM",
    extend_matcher(grouped_nodes_name["NEG_NM"], term_if_matches=True),
    ExactSampler(),
)


# %%
"""
We debug the new matcher to see if it's doing what we expect.
Note: be careful with the use of this debugger: when nodes are early in the graph, the number of paths grows exponentially, so the `.get_all_paths` method can take a long time to run. If you want to stop it, then you don't have any choice but to restart the kernel as `rust_circuit` doesn't support interruption.
"""
# %%
POS_interp = NM_corr.i_names["POS_NM"]
POS_matcher = NM_corr.corr[POS_interp]
if MAIN:
    print_all_heads_paths(POS_matcher, short_names=short_names, show_qkv=True)

# %%

"""
Exercise: What would `print_all_heads_paths` show if we changed the flag `term_if_matches=False`?

<details>

<summary>Solution</summary>

The interaction between positive name movers would be allowed. Positive name movers are spread on two layers, so we would see paths like "POS_NM->POS_NM->logits" in addition to the direct path "POS_NM->logits".

This is sometimes what we want.
</details>

Exercise: what is the difference between using `term_if_matches=False` and `restrict=False`?

<details>

<summary>Solution</summary>

* If we use `restrict=False` and `term_if_matches=False`, we allow _any_ node on the way from the starting point and the set of target nodes.
* If we use `restrict=True` and `term_if_matches=False` we allow _only_ node from the set of targets, but they can appear an arbitrary number of times on the path.
* If you use `restrict=False` and `term_if_matches=True`, you match all node between source and target _except_ the interaction between the target node. (This is rarely what we want)
</details>
"""
# %%
"""
Exercise: How can you extend the correspondence to include the affirmation "Positive NM can Q-compose with positive NM from a latter layer? There is no third order effect (e.g. NM1 -> NM2.Q -> NM2 -> NM3.Q -> NM3)"?

Hint: you can add a new interp node called "POS_NM_second_order".

<details>
<summary>Answer</summary>
`print_all_heads_paths` should show the second-order effect for the earliest NM:

```
--- paths to a9_h6_t15 ---
Path 0 : POS_NM->a.q->POS_NM->logits

--- paths to a9_h9_t15 ---
Path 0 : POS_NM->a.q->POS_NM->logits
```

</details>

"""
# %%
if "SOLUTION":
    if MAIN:
        extend_corr(
            NM_corr,
            "POS_NM",
            "POS_NM_second_order",
            extend_matcher(grouped_nodes_name["POS_NM"], term_if_matches=True, qkv="q"),
            ExactSampler(),
        )
if MAIN:
    print_all_heads_paths(
        NM_corr.corr[NM_corr.i_names["POS_NM_second_order"]],
        short_names=short_names,
        show_qkv=True,
    )


# %%
r"""
### Moving Pieces Experiments

Now that we have the tools to build and debug complex correspondences, we'll put them into practice.

Until now, we only considered heads in isolation. What if we patch them together? We'll select the group of heads writing negatively and see what happens when we patch them together.

This is an example of "Moving Pieces Experiments": we changed the input of particular nodes to make the model behave in a predictable way despite the activations being out of distribution.

Exercise: 
* Predict the result of the code before running it (i.e. discuss with your partner an explanation similar to the interpretation of path patching results given above.)
* Change the group of nodes from "NEG_NM" to "POS_NM" and predict what will be the result.
* Run the two experiments above by allowing all downstream effects after the negative or positive heads. What do you observe? (you can go back to the introduction to path patching doc for the distinction between with/without downstream effects)

<details>
<summary>Hint: how to allow downstream effect</summary>
 you can use the `restrict=False` flag in `extend_matcher` to allow all downstream effects.
</details>

<details>
<summary>Negative heads patching</summary>
The result is highly positive. This is not a surprise: each node individually is writing -(IO-S) when run on the correct IO. When we patch their input with IO', they write -(IO'-S), so their negative contribution is no more, and the final logit diff is greater than the reference.
If we patch several of them, this effect is even stronger because we kill more negative terms in the final sum.

This reasoning assumes a naive model where the total logit diff is the sum of the effect of individual heads. In this case, it is close to reality as we are filtering all downstream effects: the only path patched are NEG_NM->logits. So all the heads that are not NEG_NM, are seeing the output of NEG_NM that are the same as in the baseline forward pass. 

Because of the filtering, we can _really_ write $logits = \sum_{h \in NEG_NM}{h(x_{IO'})} + \sum_{n \in OTHERS}{n(x_{IO})} $ ($OTHERS$ is all the nodes (heads and mlp) not in $NEG_NM$)

</details>

<details>
<summary>Positive heads patching</summary>
This is the opposite of negative heads. In the sum: $logits = \sum_{h \in POS_NM}{h(x_{IO'})} + \sum_{n \in OTHERS}{n(x_{IO})} $, positive heads are positive contribution. Patching them kills those terms and thus makes the logits more negative.
</details>

<details>
<summary>With all downstream effects</summary>
The simple decomposition in a sum doesn't hold anymore! if $n \in OTHERS$ is in a latter layer as $h \in POS_NM$, $n$ sees the output of $h$ that is different from the baseline. 

What we observe in practice is that the sign of the effect is the same as when filtering for downstream effect, but the magnitude is much lower. This suggests the existence of complex interactions between heads that are "dampening" the perturbation from dysfunctioning (patched) heads.

However, this "dampening" doesn't exist when patching negative heads. This can be explained by the fact that they are at the latter layers: no head can see the output of heads at layer 11 and adjust their output accordingly.

It's important to keep in mind these complicated interactions when doing patching experiments (be it causal scrubbing, path patching, or other).

</details>
"""
# %%

# global parameters so you don't have to change all the occurence in the cell and the results can be printed with the parameters of the experiment
term_if_matches = True
restrict = True
node_name_to_connect = "NEG_NM"

# redefinition of the correspondance to have clean objects. In general err on the side of redefining too many objects to be sure you know what they contain.
NM_corr = Correspondence()
i_root = InterpNode(ExactSampler(), name="logits")
m_root = corr_root_matcher
NM_corr.add(i_root, m_root)

extend_corr(
    NM_corr,
    "logits",
    "POS_NM",
    extend_matcher(grouped_nodes_name["POS_NM"], term_if_matches=term_if_matches, restrict=restrict),
    ExactSampler(),
)

extend_corr(
    NM_corr,
    "logits",
    "NEG_NM",
    extend_matcher(grouped_nodes_name["NEG_NM"], term_if_matches=term_if_matches, restrict=restrict),
    ExactSampler(),
)



def se(c):
    """Short function for Sample and Evaluate along the global variable `group`"""
    transform = rc.Sampler(rc.RunDiscreteVarAllSpec([group]))
    return transform.sample(c).evaluate()


if MAIN:
    interp_node = NM_corr.i_names[node_name_to_connect]
    matcher = NM_corr.corr[interp_node]

    result = se(
        path_patching(
            ld_circuit,
            ioi_dataset.prompts_toks,
            flipped_IO_S_dataset.prompts_toks,
            matcher,
            group,
            "tokens",
        )
    ).mean()


if MAIN:
    if "SOLUTION":
        print(
            f"Logit diff={result} after path patching {node_name_to_connect} with `restrict={restrict}`, term_if_matches={term_if_matches}"
        )
    else:
        if False:
            print(
                f"Logit diff={result} after path patching {node_name_to_connect} with `restrict={restrict}`, term_if_matches={term_if_matches}"
            )
        else:
            print(f"Predict the result before printing it!")


# %%

"""
### Advanced Iterative Path Patching

# More second order patching

Exploring composition: let's go one step further. We want to find the heads influencing the Name Movers through their keys at the IO position. No more templates this time - it's your turn to write the full experiment.

Exercise: Write an experiment investigating the K-composition of the name movers at the IO token. You can try with positive and negative name movers. In isolation, or as a group.
(The solution only considers positive name movers.)

`results_K_comp_IO_NM` is the results of the iterative path patching experiment after averaging on the batch dimension, reshaping to shape (12,13) and removing the `ref_ld` as done in the previous iterative path patching experiments. The `patch_data` argument is the prompts from `flipped_IO_dataset` (try to think about what would change if we change this dataset).

"""
# %%

if "SOLUTION":
    if MAIN:
        matcher_extenders = [
            extender_factory(MLPHeadAndPosSpec(l, cast(HeadOrMlpType, h), IO_POS), qkv="k")
            for l in range(12)
            for h in list(range(12)) + ["mlp"]
        ]

        NM_corr = Correspondence()
        i_root = InterpNode(ExactSampler(), name="logits")
        m_root = corr_root_matcher
        NM_corr.add(i_root, m_root)

        extend_corr(
            NM_corr,
            "logits",
            "NM",
            extend_matcher(
                grouped_nodes_name["POS_NM"],
                term_if_matches=True,
                restrict=True,
            ),
            ExactSampler(),
        )

        results_K_comp_IO_NM = (
            iterative_path_patching(
                circuit=ld_circuit,
                hypothesis=NM_corr,
                nodes_to_connect=[NM_corr.i_names["NM"]],
                baseline_data=ioi_dataset.prompts_toks,
                patch_data=flipped_IO_dataset.prompts_toks,
                group=group,
                matcher_extenders=matcher_extenders,
                input_name="tokens",
                output_shape=(12, 13, -1),
            ).mean(dim=-1)
            - ref_ld
        )

        show_mtx(
            results_K_comp_IO_NM,
            title="N->Name Mover K->logits path patching at IO token (flipped IO)",
        )

assert torch.isclose(results_K_comp_IO_NM[:3, 5], torch.tensor([ 0.0247, -0.0080,  0.0115]), atol=1e-4).all()

#%%
"""
# Automatic higher order patching ("ACDC")

At this point, hopefully it seems clear that there are a lot of similar experiments that elicit GPT-2 small's behavior on the IOI task, by iteratively applying path patching. ACDC (Automatic Circuit Discovery) is one algorithm that automates these experiments, and creates diagrams of circuits which may i) describe model behavior well, or ii) provide hints for us to interpret what model components are doing.

ACDC is a greedy algorithm, finding the heads that matter directly for the output, then the heads that have largest effect on those heads, and so on (algorithm described in detail in the notebook `remix_d5_acdc.py`).

Exercise 1: run the ACDC code from `remix_d5_acdc.py`. Make sure you `git pull` the latest `remix-stable` unity branch and `remix` branch in the REMIX repo.

You should be able to create an image like `remix_d5_acdc_initial_graph.png` from this repo.

Exercise 2 (optional): in LeelaZero (which the Go project people are working on), instead of a continual residual stream as in transformers, the output of block $i$ ($b_i$) is computed from the output $b_{i-1}$ of block $i-1$: $b_i = ReLU( b_{i-1} + f_i(b_{i-1}) )$ (where $f_i$ is a convolution, here). Naively, we can't build a computational graph with connections from *all* earlier blocks to the final output. However, there is a way to rewrite the computation so we can make a computational graph with edges between distant blocks. How can we do this?

<details>
<summary>Solution</summary>
We can write $\\tilde{b}_i = \\text{max}(-\\tilde{b}_{i-1}, f_i(\\tilde{b}_{i-1}))$. Then $b_i = \\tilde{b}_i + \\tilde{b}_{i-2} + ... + \\tilde{b}_0$, and so we can build a computation graph on all the $\\tilde{b}_i$.
</details>

Exercise 3 (optional): find a "counterexample" of a network for which ACDC will find a circuit that does not explain model behavior well.

Exercise 4: the ACDC code only finds which heads and MLPs are important for the IOI task. Modify the notebook to find out which of the Q and K and V connections are important

<details>
<summary>Hint 1</summary>
This can be done without editing any internal files (just the objects in the notebook). 

Be careful about the nodes you match however: there is a lot of duplication in the circuit.
</details>

<details>
<summary>Hint 2</summary>
This can be done by editing the `model` to separate its Q and K and V nodes. Try and use the `rc.Updater` to do this.
</details>

You should be able to produce an image like `remix_d5_acdc_solution_graph.png` (this only steps some layers back). 
<details>
<summary>Solution</summary>

Using the following code for two cells, instead of what currently exists in the `remix_d5_acdc.py` file, you can get information on the Q and K and V use of different heads.

```python
attention_head_name = "a{layer}.h{head}.{let}_p_bias"

for layer in tqdm(range(12)):
    for head in range(12):
        for let in ["q", "k", "v"]:
            matcher = rc.Matcher(f"a{layer}.h{head}").chain(rc.restrict(f"a.{let}_p_bias", end_depth=6))
            assert len(matcher.get(model)) == 1, (layer, head, let)
            model = matcher.update(model, lambda c: c.rename(attention_head_name.format(layer=layer, head=head, let=let)))

mlp_name = "m{layer}"
embed_name = "tok_embeds"
root_name = "final.inp"
no_layers = 12
no_heads = 12
all_names = (
    [embed_name, root_name]
    + [mlp_name.format(layer=layer) for layer in range(no_layers)]
    + [attention_head_name.format(layer=layer, head=head, let=let) for let in ["q", "k", "v"] for layer in range(no_layers) for head in range(no_heads)]
)
all_names = set(all_names)
template_corr = ACDCCorrespondence(all_names=all_names)
root = ACDCInterpNode(root_name, is_root=True)
template_corr.add_with_auto_matcher(root)

all_nodes: List[ACDCInterpNode] = []
all_nodes.append(root)

print("Constructing big Correspondence...")
for layer in tqdm(range(no_layers - 1, -1, -1)):
    # add MLP
    mlp_node = ACDCInterpNode(mlp_name.format(layer=layer))
    for node in all_nodes:
        node.add_child(mlp_node)
        mlp_node.add_parent(node)
    template_corr.add_with_auto_matcher(mlp_node)

    all_nodes.append(mlp_node)

    # add heads
    head_nodes = []
    for head in range(no_heads):
        for letter in ["q", "k", "v"]:
            head_node = ACDCInterpNode(attention_head_name.format(layer=layer, head=head, let=letter))
            head_nodes.append(head_node)
            for node in all_nodes:
                node.add_child(head_node)
                head_node.add_parent(node)
    for node in head_nodes:
        template_corr.add_with_auto_matcher(node)

    for i, head_node in enumerate(head_nodes):
        all_nodes.append(head_node)
print("...done")

embed_node = ACDCInterpNode(embed_name)

for node in tqdm(all_nodes):
    node.add_child(embed_node)
    embed_node.add_parent(node)
template_corr.add_with_auto_matcher(embed_node)
```
</details>

If you leave ACDC running overnight, you should be able to make graphs like `remix_d5_acdc_big.png`, which is annotated with how well it corresponds to the IOI circuit. We want this to be faster and less noisy, however! Think about your counterexample, and how the algorithm could be redesigned to have fewer issues.
"""

# %%

"""
### Running causal scrubbing experiment

All the correspondences we created are not just a fancy way to store `IterativeMatcher`. We can easily build and run causal scrubbing experiments from them. Here is the definition of a helper function to do so. 

There is no exercise in this section, it's a demonstration you can reuse when you want to run your own causal scrubbing experiments in part 4.
"""
# %%


def run_experiment(ld_circuit: rc.Circuit, ioi_dataset: IOIDataset, corr: Correspondence):
    """Run a causal scrubbing experiment on the logit diff circuit using the data from the ioi_dataset and the given correspondance. Return the vector of srubbed logit diff"""

    ## define the dataset: it uses the labels and token as input.
    io_s_labels = torch.cat(
        [ioi_dataset.io_tokenIDs.unsqueeze(1), ioi_dataset.s_tokenIDs.unsqueeze(1)],
        dim=1,
    )

    cs_ioi_ds = Dataset(
        arrs={
            "tokens": rc.Array(ioi_dataset.prompts_toks.to("cpu"), name="tokens"),
            "labels var": rc.Array(io_s_labels, name="labels var"),
        },
        input_names={"tokens", "labels var"},
    )
    # create the experiment and tweak evaluation param.
    exp = Experiment(ld_circuit, cs_ioi_ds, corr, num_examples=len(ioi_dataset))

    MAX_MEMORY = 20_000_000_000

    eval_settings = ExperimentEvalSettings(
        optim_settings=rc.OptimizationSettings(
            max_memory=MAX_MEMORY,
        ),
        device_dtype="cuda:0",
        optimize=False,
        batch_size=len(ioi_dataset),
    )

    scrubbed = exp.scrub()
    result = scrubbed.evaluate(eval_settings)
    return result


# %%
"""
Exercise: Use the `run_experiment` function to run the causal scrubbing experience where we claim that "Only the direct link from Name Movers to logits matters. All the paths from input to Name Movers' matter." 
"""

if "SOLUTION":
    NM_corr = Correspondence()
    i_root = InterpNode(ExactSampler(), name="logits")
    m_root = corr_root_matcher
    NM_corr.add(i_root, m_root)

    extend_corr(
        NM_corr,
        "logits",
        "NM",
        extend_matcher(
            grouped_nodes_name["POS_NM"] + grouped_nodes_name["NEG_NM"],
            term_if_matches=True,
            restrict=True,
        ),
        ExactSampler(),
        other_inputs_sampler=UncondSampler(),
    )

scrubbed_results = run_experiment(ld_circuit, ioi_dataset, NM_corr)
print(f"Averaged scrubbed logit diff: {scrubbed_results.mean().item():.3f} +- {scrubbed_results.std().item():.3f}")

# %%
"""
Results: the scrubbed logit diff is around 50% of the original logit diff. This means that we are missing around half of the direct effect of the logits. It's likely that this effect comes from the heads we neglected to include when discovering the Name Mover to keep our group of nodes sparse. 

This is a common tradeoff to make: the more nodes we include, the more we recover the behavior but the less we understand the hypothesis. Here, we'll lean on the side of thoroughly understanding the discovered mechanism even if we are far from recovering the full effect.
"""

# %%

"""
### Attention pattern visualisation

Exercise: write a function that gets the attention pattern of a given head. You can use the `eval_on_toks` function defined earlier.
Begin by exploring the computational graph to know where to look for the attention pattern of a given head.

We provide you the `eval_on_toks` function that evaluates a circuit (in your case, this will be a subcircuit of `circuit`) on given tokens.

BONUS: it's often useful to weigh the attention probabilities by the norm of the values at the key positions. I.e. for the key axis i, multiply all the entries` a_ij` by `||v_i||`, where `v_i` is the value vector at position `i`. If `add_value_weighted` is `True`, compute the weighted attention pattern.

<details>
<summary>Hint</summary>

To get the attention probabilities of a head, you need to get the node `a.attn_probs` that is just after the node `a{l}.h{h}`. All the heads have a node called `a.attn_probs`. So to ensure you match the right one, you can:
* use `end_depth=N` to stop the search after N steps in the graph
* use `term_if_matches=True` to stop the search as soon as you find a match (but be careful, there might be cases like H -> A,B, and later B-> A' if you wanted to match only A, you could also end up with A' if you only restrict using `term_if_matches`)
* `term_early_at` to prune off the branches you don't want to explore (e.g. the B in the example above).

Try constructing the right matcher before adding it to the function. Don't worry about the optimization of the evaluate function.

</details>

"""
# %%


def eval_on_toks(c: rc.Circuit, toks: torch.Tensor):
    group = rc.DiscreteVar.uniform_probs_and_group(len(toks))
    c = c.update(
        "tokens",
        lambda _: rc.DiscreteVar(rc.Array(toks, name="tokens"), probs_and_group=group),
    )
    transform = rc.Sampler(rc.RunDiscreteVarAllSpec([group]))
    results = transform.sample(c).evaluate()
    return results


def get_attention_pattern(
    c: rc.Circuit,
    heads: List[Tuple[int, int]],
    toks: torch.Tensor,
    add_value_weighted=False,
):
    assert toks.ndim == 2
    seq_len = toks.shape[1]
    attn_patterns = torch.zeros((len(heads), len(toks), seq_len, seq_len))

    for i, (l, h) in enumerate(heads):
        if "SOLUTION":
            a = rc.Matcher(f"a{l}.h{h}").chain(rc.restrict("a.attn_probs", term_if_matches=True, end_depth=3))
            pattern_circ = a.get_unique(c)
            attn = eval_on_toks(pattern_circ, toks)

            if add_value_weighted:
                v = rc.Matcher(f"a{l}.h{h}").chain(rc.restrict("a.v_p_bias", term_if_matches=True, end_depth=3))
                values = v.get_unique(c)
                vals = eval_on_toks(values, toks)
                vals = torch.linalg.norm(vals, dim=-1)
                attn_patterns[i] = torch.einsum("bKQ,bK->bKQ", attn, vals)
            else:
                attn_patterns[i] = attn

    return attn_patterns


heads = [(10, 7), (11, 10)]
nb_seqs = 5

if MAIN:
    attn_patterns = get_attention_pattern(circuit, heads, ioi_dataset.prompts_toks[:nb_seqs])

    val_weighted_attn_patterns = get_attention_pattern(
        circuit, heads, ioi_dataset.prompts_toks[:nb_seqs], add_value_weighted=True
    )

    # The dataset is seeded, so the attention patterns should be the same.
    assert torch.allclose(attn_patterns[0][3][15][15], torch.tensor(0.0041), atol=1e-4)
    assert torch.allclose(val_weighted_attn_patterns[0][3][15][15], torch.tensor(0.0232), atol=1e-4)

# %%

"""
Exercise: create a `VeryNamedTensor` to give annotation on the attention patterns, and the Composable UI to visualize them (introduced in day 1)
"""
# %%
if MAIN:
    remix_utils.await_without_await(lambda: cui.init(port=6789))

    attn_pattern_vnt = VeryNamedTensor(
        attn_patterns,
        dim_names="head sentence queries keys".split(),
        dim_types="example example axis axis".split(),
        dim_idx_names=[
            heads,
            [f"seq {i}" for i in range(nb_seqs)],
            ioi_dataset.prompts_text_toks[0],
            ioi_dataset.prompts_text_toks[0],
        ],
        title="Attention patterns",
    )

    remix_utils.await_without_await(lambda: cui.show_tensors(attn_pattern_vnt))


# %%
"""
### Time to lead your research!

Now that you have the tool, it's your turn to put them into practice. The following exercises are open-ended and don't include code solutions. You are in charge of leading the research! You still have access to "Possible reasoning steps" to help you check you're on the right track, or to help you get started.

Don't hesitate to write your own tools if you have experiments in mind that we did not cover here. You can also try to combine the tools we wrote so far. For example, it's sometimes useful to visualize attention patterns after path patching.
"""
