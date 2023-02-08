# %%
"""
# REMIX Day 5, Part 4 - Let's make claim about GPT-2 small

In this notebook, it's your turn to write the experiments!

Instructions are here: [Exploratory interp exercises presentation](https://docs.google.com/document/d/1qyHT4W9TtVL77AMKN514SjXT9fyNS70DJH9FFQ7YiDg/edit?usp=sharing)

We included an indicative time estimate of how much time you should spend on each claim. This should help you decide whether to extend a claim after having addressed it, or stop reading at the checkpoint and move on to the next one.

"""
import os
import sys

# %%
MAIN = __name__ == "__main__"

if MAIN:
    from remix_extra_utils import check_rust_circuit_version

    check_rust_circuit_version()

if "SKIP":
    # Skip CI for now - avoids downloading GPT2
    IS_CI = os.getenv("IS_CI")
    if IS_CI:
        sys.exit(0)

from remix_d5_utils import (
    IOIDataset,
    load_and_split_gpt2,
    load_logit_diff_model,
)

if "SKIP":
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import plotly.express as px
import rust_circuit as rc
import torch
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
    split_heads_and_positions,
)
from interp.circuit.interop_rust.algebric_rewrite import split_to_concat
from interp.circuit.interop_rust.model_rewrites import To, configure_transformer
from interp.circuit.interop_rust.module_library import load_model_id
from interp.circuit.projects.gpt2_gen_induction.rust_path_patching import (
    CopyDsSampler,
    direct_path_patching,
    direct_path_patching_up_to,
    logprob_on_labels,
    make_arr,
    match_nodes_except,
)
from interp.tools.indexer import TORCH_INDEXER as I
from interp.tools.interpretability_tools import print_max_min_by_tok_k_torch


"""
We import the tools from the part 3 solution notebook. 

"""
# path patching utils
from remix_d5_part3_solution import (
    path_patching,
    iterative_path_patching,
    extender_factory,
    show_mtx,
)

# Helper functions
from remix_d5_part3_solution import (
    extend_corr,
    add_path_to_group,
    extend_matcher,
    print_all_heads_paths,
)
from remix_d5_part3_solution import se  # sample and evaluate

from remix_d5_part3_solution import get_attention_pattern

"""
 Same datasets and model loading as part 3.
"""
# %%

ioi_dataset = IOIDataset(prompt_type="BABA", N=50, seed=42, nb_templates=1)

MAX_LEN = ioi_dataset.prompts_toks.shape[1]  # maximal length

for k, idx in ioi_dataset.word_idx.items():  # check that all the sentences are aligned
    assert (idx == idx[0]).all()

# Because sentences are aligned, we can define global variables for the position of the tokens
END_POS = int(ioi_dataset.word_idx["END"][0].item())
IO_POS = int(ioi_dataset.word_idx["IO"][0].item())
S1_POS = int(ioi_dataset.word_idx["S1"][0].item())
S2_POS = int(ioi_dataset.word_idx["S2"][0].item())

# %%


flipped_IO_dataset = ioi_dataset.gen_flipped_prompts("IO")
flipped_S_dataset = ioi_dataset.gen_flipped_prompts("S")

flipped_IO_S_dataset = ioi_dataset.gen_flipped_prompts("IO").gen_flipped_prompts("S")

flipped_IO_S1_order = ioi_dataset.gen_flipped_prompts("order")

# %% model loading
circuit = load_and_split_gpt2(MAX_LEN)

io_s_labels = torch.cat([ioi_dataset.io_tokenIDs.unsqueeze(1), ioi_dataset.s_tokenIDs.unsqueeze(1)], dim=1)
ld_circuit, group = load_logit_diff_model(circuit, io_s_labels)

# %%

"""
Management of the discovered nodes: redefinition of `add_node_to_pokedex` to handle the new nodes.
"""

short_names = {}

grouped_nodes_name = {}  # type: ignore
grouped_nodes_spec = {}  # type: ignore


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
## Claim 2
~ 30 min

Name Movers have high attention scores for the IO token. However, they don't rely on the key values of the IO token to compute this score.

Step 1 addresses the claim. Step 2 is an extension of the claim. Before looking at Step 2, it's worth spending some time thinking about how you would extend the claim yourself.

Possible reasoning:

<details>
<summary>Step 1</summary>
Clarifying what "don't rely" means. Surely, you cannot put random noise instead of the K values. As in the causal scrubbing framework, we will use resampling ablation to formalize "don't rely". This can be interpreted: "Inasmuch as the K values are computed on a name token at the same position, the attention score will be high regardless of the value of the name"

### Experiments
Run path patching where all the inputs that are connected to the name movers' keys are patched with a random name at position IO. Measuring the attention score to the IO token.

Alternatively, we can also directly run a causal scrubbing experiment using the `run_experiment` we defined in the previous notebook to scrub the input of the name mover's keys and nothing else.

In fact, this is a manual way to do a causal scrubbing experiment, as you did during the induction head day.
</details>

<details>
<summary>Step 2: extension of the claim</summary>
Extension of the claim: so what do Name Movers use to compute their attention score? 

If they rely on K only to the extent it's a name, it seems likely that most of the information to differentiate IO from S comes from Q instead.

We will thus address the question: What heads are used for Q composition?

### Experiments
Iterative path patching experiments to discover heads where they Q-compose.
Experiment details: choosing to consider Name Movers as a group, or studying each NM individually. In the previous demonstration, we always considered them as a group, this makes the results less noisy but we also don't differentiate individual head specifically. It can be worth running both experiments.

</details>

"""

"""

<details>
<summary>Expected results</summary>
Expected experiment results:
* The claim is true if we interpret "don't rely" in a narrow way ("to the extent that the IO token is a name").
* Discovery of S-inhibition heads by investigating the queries.
</details>

### Checkpoint
Description of the S-Inhibition heads discovery (Sec 3.2) in the [IOI paper](https://arxiv.org/pdf/2211.00593.pdf).

"""

"""
## Claim 3
~ 1h

"S-inhibition heads are Q-composing with Name Movers by communicating the value of the IO token. This is why Name Movers are able to pay attention to the IO and not the S token."

Step 1 addresses the claim. Step 2 is an extension of the claim. Step 3 is a question that should be addressed after Step 2.

Possible reasoning:

<details>
<summary>Step 1: Hint</summary>
The claim can be addressed without running any experiments.
</details>

<details>
<summary>Step 1: Answer</summary>
This claim can be disproven without any experiment: you showed earlier that Name Movers don't rely on the value of the name to compute their keys at the IO token position (under the condition it's still a name). If the claim was true, Name Mover should use a match between the query and the value of the name to compute their attention score, such that the value of the IO token would matter in their keys.

### Experiments
We can confirm this argument using path patching where we replace the input to Name Movers' queries with sequences where the name at the IO position was randomly flipped.

Again, this can also be done by directly running the corresponding causal scrubbing experiment using the `run_experiment` function we defined in the previous notebook.
</details>


<details>
<summary>Step 2: claim extension</summary>

If S-Inhibition heads are not communicating the value of the IO tokens, what information, independent of the token value, is transmitted?

The crucial information to get pay attention to the right position is not the value of the token, but the position of the token. S-Inhibition heads could directly share the position of the IO token with Name Movers.

However, "position" is a fuzzy term here. We can make it more concrete (i.e. concrete enough that we can design experiments) by interpreting it as:
* Absolute position (S-Inhibition heads at the END position are communicating "IO is in position 2")
* Relative position ("IO is 7 tokens before the current token")
* Template information ("IO is the second name appearing in the sequence", this is equivalent to sharing the template type ABB or BAB)


### Experiments

For each of these interpretations of "position" the workflow is the same:
* Design datasets where the feature you investigate is decorrelated from the alternative hypothesis, e.g. relative/absolute position, by adding a random length prefix.
* Apply path patching of SIN -> NM with SIN run on the dataset with the randomized feature.
* Observe if the attention of the Name Movers to the IO token is reduced. Observe if the logit diff is reduced.

This is a crude way to identify which feature matters.

To have a more fine-grained understanding, you could also run Moving Pieces Experiments.

Once you have identified a minimal feature that matters, you can modify it in an arbitrary way and observe if the Name Movers' attention behaves as expected.

Eg. if you identified that the relative distance is what matters:
* Patch SIN -> NM with 
    * S-Inhibition heads run on a dataset where IO tokens are 8 tokens before the END.
    * The default dataset contains sequences where IO tokens are 5 tokens before the END.
* If S-Inhibition heads are communicating relative position, you should expect the Name Movers to pay attention to the token 8 tokens before the END. This result should hold even if you randomized the absolute position of END and IO tokens by adding a random length prefix.

### Results

A minimal feature that matters and successful Moving Pieces Experiments where you isolated this feature (i.e. you applied the maximum amount of random variation introduced that keeps the feature intact like the random prefix in the example above).

</details>

<details>
<summary>Step 3: Question</summary>
Try to think about possible ways to decorrelate between "the S-inhibition heads are sharing the S position" and "the S-inhibition heads are sharing the IO position"?
</details>


<details>
<summary>Step 3: Answer</summary>

If the position is encoded as absolute or relative position, it's possible to change the S position while keeping the IO position constant (and the opposite). Path patching from such a dataset can allow disentangling of the two hypotheses.

If the position is encoding "position among names in context" (third bullet point in step 2) this is really hard as the two hypotheses allow the same causal scrubbing swaps. 

In general, I don't have any good experiment to propose here, but I think it's a useful exercise to think about this. It's unclear if the question means anything in this context.
</details>

"""

"""
<details>
<summary>Expected results</summary>
* Token value matters a bit (scrubbing it causes a drop of logit diff by ~ 30%)
* But the bulk of the effect is position-wise. The position is invariant to adding prefixes or changing the distance between END and S2. The feature that matters is the position among names in the context (i.e. the template type).
</details>

### Checkpoint
* Reading "what are S-Inhibition heads writing?" Appendix A of the [IOI paper](https://arxiv.org/pdf/2211.00593.pdf). We use "token signal" and "position signal" to limit the number of hypotheses we make on them, as it's tricky to know exactly what they are about.

"""


"""
## Claim 4
~ 1h

S-inhibition heads are reading the position of the S token at the S2 token position. They are using OV-composition with earlier heads that detect duplication of the S token.

Possible reasoning:

<details>
<summary>Step 1: Early experiment to make sense of the claim</summary>

It's worth making sense of the question by visualizing the (value-weighted) attention pattern of the S-inhibition heads: they are attending to the S2 token, and that's a good sign that it makes sense to look for what they are reading there.


Then, we need to see if such "earlier heads" exist at all. We are searching for heads:
* At the S token position
* That are influencing the output of S-Inhibition heads through the values of S-Inhibition heads.

### Experiments

Searching for such heads can be done with an iterative path patching experiment. 

I'll call H the set of heads found here. From there, you can either consider each head in H individually or group them. The easiest thing to do is to consider them grouped until you have reason to split them.

</details>

<details>
<summary>Step 2: Addressing the "reading position"</summary>

We now have to check the claim that H is communicating the S token position. Again, "position" is ill-defined here. As noted in the previous claim, it'll be hard to differentiate the hypothesis "communicating the S position" and "communicating the IO position". 

The same path patching experiment can be reused as in the previous claim. Instead of path patching SIN->NM, we now use H->SIN->NM.

Similarly, the Moving Pieces Experiment can be reused. 

If both holds, this is great evidence that S-Inhibition heads are transmitting the information from H at position S2 to END.
</details>

<details>
<summary>Step 3: Addressing the "detecting duplication"</summary>


### Preliminary experiments
A quick look at the attention pattern should divide the newly found head into two groups: heads attending to S1 and heads attending to S1+1.
* We'll call the first group D.
* You can recognize in the second group the signature of induction heads introduced in [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html). We'll call this group I.


### Formalization

The "detecting duplication" is a fuzzy sub-claim and must be formalized. It also naturally extends beyond the context of IOI.

One natural interpretation is "As far as IOI behavior is concerned, the output of these heads are the same under the condition that the S token is duplicated and the position of the first occurrence is the same". This leads to a natural path patching / CS experiment to run.

### Extension outside IOI

The "detecting duplication" can also be naturally extended to cases outside IOI. Some ideas of how you could approach the problem:
* Attention pattern analysis of those heads: do they keep the same pattern as in IOI? Easy to run, but don't tell much about their interaction with the rest of the network.
* Look at the output of the heads on a duplicated vs non-duplicated word on OWT sentences / random sequences of tokens. Dimensionality reduction on the output: are there clusters depending on the "is duplicated" feature? 
* Exploring of potential limitation of the "detecting duplication" behavior:
    * Do they detect duplication of common words like "the" and "to"?
    * Do the behavior depends on the distance between the two occurrences?
* For group I, you can use the definition introduced in [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) using the prefix matching and copying criteria.


</details>

"""


"""
<details>
<summary>Expected experimental results</summary>
* Discovery of the Induction and Duplicate Token heads acting at the S2 position.
* Division in two such groups.
* Some heads have i) crisp behavior (i.e. crisp attention patterns), ii) a large influence on S-Inhibition queries (i.e. large effect size on path patching) and iii) generalized outside IOI (e.g. OWT or random tokens). Other heads score lower on these axes. Those three axes are highly correlated.
* Maybe some confusing results when trying the moving pieces experiments to check if those heads are writing the position of S1.
</details>

### Checkpoint
From the [IOI paper](https://arxiv.org/pdf/2211.00593.pdf), we have:
* Description of the discovery of the induction and duplicate token heads (Sec 3.3). It's possible that the experiment you run to test the claim about "H is writing position" are more precise than what's done in the paper (only patching is performed in the paper).
* Appendix C (effect on S-Inhibition heads keys. Not included in the claim, but one natural question to ask if we aim at exhaustiveness).
* Appendix H & I for validation of duplicate and induction tokens outside IOI. 
* Discussion about the confusing part: some heads are straightforwardly generalized to outside IOI, while others are less understandable.
"""


"""

Congrats on completing the exploratory experiment!

From here, you can either:
* Explore how the node discovery process can be automated using Automatic Circuit Discovery Code (ACDC). You can see a demo notebook called `interp/circuit/projects/acdc/acdc.py` on the `remix-acdc` branch of the `unity` repository.
* Read the `remix_d5_on_the_fly_CondSampler_demo.py` notebook to discover how to build conditional samplers using on-the-fly sampling.

## Additional claims

Some bonus claims that you can try to test if you have time. 

### Claim 5
Duplicate Token Heads Q-compose with later Induction Heads.

### Claim 6
The Name Movers are the only heads that can make the logit difference positive.
"""
