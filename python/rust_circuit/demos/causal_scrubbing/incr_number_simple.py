# %%

from IPython import get_ipython

get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")


import gc

from interp.circuit.circuit_model_rewrites import MLPHeadAndPosSpec, split_heads_and_positions
from rust_circuit.causal_scrubbing.dataset import Dataset

# added by tao, the notebook empirically runs faster and doesnt oom with this
gc.disable()
import os
from typing import *

import jax
import torch
from transformers import AutoTokenizer

import rust_circuit as rc
from interp.circuit.circuit_models import circuit_gpt_and_tok_embeds
from interp.circuit.projects.gpt2_gen_induction.rust_path_patching import match_nodes_except
from interp.model.model_loading import load_model
from rust_circuit.causal_scrubbing.experiment import Experiment, ExperimentEvalSettings
from rust_circuit.causal_scrubbing.hypothesis import (
    Correspondence,
    FuncSampler,
    InterpNode,
    UncondSampler,
    corr_root_matcher,
)
from rust_circuit.interop_rust import py_to_rust

jax.config.update("jax_platform_name", "cpu")

# %% [markdown]

# This notebook is intended as a basic introduction to the causal scrubbing API. It is operating on the circuit that
# predicts incrementing numbers once it's seen a sequence of numbers. This experiment doesn't use the full functionality
# of causal scrubbing - it just checks whether these nodes are used in the circuit for predicting next numbers.
#
# Before running, make sure you're using the latest rust_circuit version - run `pip install rust_circuit --upgrade`
#
# This is still a work in progress. Here are some things I want to do to improve it:
# - [ ] New model loader once splitting exists for that model loader
# - [ ] Use a real hypothesis that actually talks about what the circuit does rather than just "is it important"
# - [ ] This circuit isn't exhaustive - its depth is only 2. Keep expanding it

# %% [markdown]

# First, load a tokenizer and the data we'll use for this experiment
#
# I load a token for every number because, later on, we'll be using this to determine incorrect results
#
# The sequences are just '<|endoftext|> 4 5 6' (for example) with random starting points between 1 and 100.
#
# We'll be scrubbing out unimportant nodes with other number sequences so we can isolate the nodes that are involved
# in incrementing the numbers directly - not in checking if the sequence is incrementing, for example, or in checking
# whether the tokens are numbers (even though these heuristics are likely useful for implementing the overall behavior).

# %%

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

all_numbers = [f" {i}" for i in range(1, 104)]
all_number_tokens = tokenizer(all_numbers, padding=True, return_tensors="pt").input_ids.squeeze(-1)

start_points = list(range(1, 101))
sequences = [
    tokenizer.bos_token + "".join(f" {i}" for i in range(start_point, start_point + 3)) for start_point in start_points
]

tokens = tokenizer(sequences, padding=True, return_tensors="pt").input_ids

correct_values = "".join(f" {start_point + 3}" for start_point in start_points)

correct_tokens = tokenizer(correct_values, padding=True, return_tensors="pt").input_ids.squeeze(0)

# %% [markdown]

# Some utilities we'll be using later on

# %%


def logit_diff(logits: torch.Tensor, labels: rc.Array) -> float:
    """
    Returns difference between probability of correct number and all numbers [1, 103]. Will always be negative, since
    the correct number is one of those numbers. Higher values (closer to zero) are more correct.
    """
    correct_probs = logits[:, 3].gather(-1, labels.value.cpu().long().unsqueeze(-1)).logsumexp(dim=-1)
    all_number_probs = logits[:, 3].gather(-1, all_number_tokens.unsqueeze(0)).logsumexp(dim=-1)
    return (correct_probs - all_number_probs).mean().item()


def loss_fn(logits: torch.Tensor, labels: rc.Array) -> float:
    return torch.nn.functional.cross_entropy(logits[:, 3], labels.value.cpu().long()).item()


def print_top_k(tokenizer: AutoTokenizer, model_res: torch.Tensor):
    """
    Useful for debugging what the model has chosen to output
    """
    print(model_res[-1:].softmax(-1).topk(10))
    print(tokenizer.batch_decode(model_res[-1:].topk(10).indices.T))  # type: ignore


def node_loc_to_name(
    layer: int,
    head: Union[int, Literal["mlp"]],
    position: int,
) -> str:
    """
    Find node name from location info
    """
    if head == "mlp":
        return f"m{layer}_t{position}"
    else:
        return f"a{layer}.out_h{head}_t{position}"


SameLabelSampler = lambda: FuncSampler(lambda ds: ds.labels.value.long())


# %% [markdown]

# # Load GPT model
#
# For historical reasons, we load the GPT-2 model from Jax before loading it into circuits
#
# TODO: rewrite when new loaders exist

# %%

# load GPT model
models_dir_local = os.path.expanduser("~/rrfs/interpretability_models_jax")

gpt2, gpt2_vars, _ = load_model("gpt2", models_dir=models_dir_local)
gpt2_b = gpt2.bind(gpt2_vars)


# %% [markdown]

# # Next, convert the model to a circuit
#
# First, we need to call a model with `circuit_gpt_and_tok_embeds` to convert it to a full circuit. This will create a
# full computation graph for the model call with this input.
#
# Since we're going to be scrubbing the inputs later, we can put in arbitrary data for now. It just needs to be the
# same shape as the data you'll scrub in later. There's also no need to create different circuits for models called on
# different inputs.

# %%

# convert model to circuit

c_logprobs, tok_inp = circuit_gpt_and_tok_embeds(
    gpt2_b,
    torch.ones_like(tokens[:1]),  # arbitrary data
    use_batch=True,
    output_type="log_probs",
)


# %% [markdown]

# # Split the model by head
#
# Because attention heads are implemented as one matrix multiplication in the model, by default they don't have
# different nodes in the computation graph. Similarly,
#
# The split heads will have names that look like `m{layer}_at_h{head}` for MLPs or `a{layer}_at_h{head}_at_t{position}`
# for heads.

# %%

# split model by head
nodes_to_split = [
    MLPHeadAndPosSpec(layer, head_or_mlp, pos)  # type: ignore
    for layer in range(12)
    for head_or_mlp in list(range(12)) + ["mlp"]
    for pos in range(1, 4)
]

split_circuit = split_heads_and_positions(c_logprobs, nodes_to_split)

# %% [markdown]

# Next, move the model to rust circuits. TODO: remove this once we move splitting to rust

# %%

circuit = py_to_rust(split_circuit, rc.TorchDeviceDtypeOp("cuda:0", "float32"))


# %%

# This is an alternate way to load the model, pre-split for us. It also replaces the inputs with DiscreteVars. The code
# below expects this model, so you should run this cell or things won't work.

with open(os.path.expanduser("~/rrfs/adria/gpt2small/seq_len_4.txt")) as f:
    circuit = rc.Parser()(f.read())

# %% [markdown]

# # Next, make a Dataset
#
# A vanilla Dataset takes (tensor, tensor) (or numpy array). It can sample from and shuffle that dataset.
#
# The commented-out version is used for replacing the tok_embeds node instead, because GPT-2 doesn't have a tokens node
# by default (but adria's version does)

# %%

dataset = Dataset(
    (
        rc.Array(tokens.float().to("cuda:0"), "tokens"),
        rc.Array(correct_tokens.squeeze(-1).float().to("cuda:0"), "labels"),
    ),
    input_names={"tokens"},
)

# Alternate dataset for handmade model

# dataset = Dataset(
#     np.array(gpt2_b.embedding.token_embedding(incr_dataset.tokens.long().numpy())),
#     incr_dataset.correct_tokens.squeeze(-1).long(),
# )

# %% [markdown]

# Let's test a simple hypothesis: everything in this graph matters. To do this, we're going to start by creating several
# objects:
#
# - Correspondence maps from nodes in the model to an interpretation of the model
# - ModelBranch is a single edge from one node to another that we think is relevant for the model (plus there's an extra
#   one for the root node)
# - InterpNode represents a leaf node. The name is only for printing, so I'm using the same names as the circuit
#   nodes themselves. The important feature here is the cond_sampler, which is used to choose how to sample the
#   activations for the model node associated with this InterpNode. In this case, we enforce that the activations must
#   be the same as another input with the same label (SameLabelSampler).
# - Experiment takes a Circuit, a Dataset, and a Correspondance, and automatically re-samples according to the causal
#   scrubbing algorithm.
#
# You create matching ModelBranches and InterpNodes (making sure they descend from corresponding MB/INs)
#
# You must start with a root node. This will match the root of the circuit.
#
# Notice that we don't hook any other branches/nodes up to this node in this case. This is because, unless you
# explicitly connect every node to the input node (in this case "tokens"), Experiment.run will automatically include
# every path from every leaf node to the inputs in your hypothesis. In this case, this will include every path from the
# root node to the inputs, so it includes the entire circuit. This is just saying that the entire circuit should be
# included.
#
# First, let's print the circuit.
# %%

# everything matters hypothesis

corr = Correspondence()

m_log_probs = corr_root_matcher
log_probs = InterpNode(name="log_probs", cond_sampler=SameLabelSampler())
corr.add(log_probs, m_log_probs)

everything_experiment = Experiment(
    circuit,  # type: ignore
    dataset,  # type: ignore
    corr,
)
everything_scrubbed = everything_experiment.scrub(100)
everything_scrubbed.print()

# %%
eval_settings = ExperimentEvalSettings(optimize=False)
everything_output = everything_scrubbed.evaluate(eval_settings)

# TODO: explain why we're using the labels from CS
everything_logit_diff = logit_diff(everything_output.cpu(), everything_scrubbed.ref_ds.labels)
print("logit diff (less negative is better):", everything_logit_diff)

everything_loss = loss_fn(everything_output.cpu(), everything_scrubbed.ref_ds.labels)
print("loss (lower is better):", everything_loss)

# %% [markdown]

# Now let's do the same thing, but with `cond_sampler=UncondSampler()`. This will make the sampler sample randomly from
# all inputs regardless of the label, so it'll scramble all our activations.

# Our goal will be to recover something close to the above value, starting at this value.

# %%

# nothing matters hypothesis

corr = Correspondence()  # maybe different input name?

m_log_probs = corr_root_matcher
logits = InterpNode(name="log_probs", cond_sampler=UncondSampler())
corr.add(logits, m_log_probs)

uncond_experiment = Experiment(
    circuit,  # type: ignore
    dataset,  # type: ignore
    corr,
)

uncond_scrubbed = uncond_experiment.scrub(100)
uncond_scrubbed.print()

# %%

uncond_outputs = uncond_scrubbed.evaluate(eval_settings)
uncond_logit_diff = logit_diff(uncond_outputs.cpu(), uncond_scrubbed.ref_ds.labels)
print("logit diff (less negative is better):", uncond_logit_diff)

uncond_loss = loss_fn(uncond_outputs.cpu(), uncond_scrubbed.ref_ds.labels)
print("loss (lower is better):", uncond_loss)

# When recovering loss do we use


# %% [markdown]

# For pedagogical purposes, let's compare this to running the model with a set of inputs. You'll see that the results
# are similar but not the same, because the input values are different. CS will sample from our input tokens, while this
# code will use exactly our selected input tokens.

# To swap out the inputs, we need to replace the input DiscreteVars that are there by default with our own.

# Why do we need to do this? Because DiscreteVars represent distributions over values, not a specific value.

# We can replace the original DiscreteVar with one that uses the values we care about, and then we can sample from that
# distribution to get a Circuit we can evaluate.

# %%

# compare to original model

# new input var

input_var = rc.DiscreteVar.new_uniform(rc.Array(tokens.to("cuda:0").float()))

# replace tokens

orig_toks_circuit = rc.Updater(lambda c: input_var)(circuit, "tokens")

# sample using a spec that samples every value once

orig_results = rc.Sampler(rc.RunDiscreteVarAllSpec([input_var.probs_and_group])).sample(orig_toks_circuit).evaluate()

orig_logit_diff = logit_diff(orig_results.cpu(), rc.Array(correct_tokens))
print("logit diff (less negative is better):", orig_logit_diff)

orig_loss = loss_fn(orig_results.cpu(), rc.Array(correct_tokens))
print("loss (lower is better):", orig_loss)

# %% [markdown]

# Now let's try a real graph that implements incrementing numbers. We will add branches and nodes to the correspondence
# to make sure we represent our fullest hypothesis. Again, any leaf nodes include all paths to the tokens, so this
# hypothesis is actually much more generous to us than it should be.
#
# So you don't have to run a lot of cells, I've just added comments inline to walk through what we're doing.
#
# One important new feature: When adding a new branch off an existing one, if you use `to`, it'll go through all nodes
# in the circuit by default. This will make your hypothesis a lot less specific. If you don't want this, yo uneed to use
# match_nodes_except as we do here.
#
# Note that the printed experiment is much more interesting now - the relevant nodes we've selected are highlighted.
#
# Also note that we recover most of the loss!


# %%

# small hypothesis

# List heads/mlps we think are involved

heads: list[MLPHeadAndPosSpec] = [
    MLPHeadAndPosSpec(10, 2, 3),
    MLPHeadAndPosSpec(9, 1, 3),
    MLPHeadAndPosSpec(8, 11, 3),
    MLPHeadAndPosSpec(7, 10, 3),
]
mlps: list[MLPHeadAndPosSpec] = [
    MLPHeadAndPosSpec(10, "mlp", 3),
    MLPHeadAndPosSpec(9, "mlp", 3),
    MLPHeadAndPosSpec(8, "mlp", 3),
]

# Create basic correspondence

corr = Correspondence()

m_log_probs = corr_root_matcher
log_probs = InterpNode(name="log_probs", cond_sampler=SameLabelSampler())
corr.add(log_probs, m_log_probs)

# add log_probs -> heads/MLPs

for layer, head, position in [*heads, *mlps]:
    name = node_loc_to_name(layer, head, position)
    m_head = rc.restrict(
        m_log_probs.chain(name), term_early_at=match_nodes_except([node_loc_to_name(layer, head, position)])
    )
    i_head = log_probs.make_descendant(name=name, cond_sampler=SameLabelSampler())
    corr.add(i_head, m_head)

# create experiment

small_experiment = Experiment(
    circuit,  # type: ignore
    dataset,  # type: ignore
    corr,
)

small_scrubbed = small_experiment.scrub(100)
small_scrubbed.print()

# %%
small_outputs = small_scrubbed.evaluate(eval_settings)

# TODO: explain why we're using the labels from CS
small_logit_diff = logit_diff(small_outputs.cpu(), small_scrubbed.ref_ds.labels)
print("logit diff (less negative is better):", small_logit_diff)

small_loss = loss_fn(small_outputs.cpu(), small_scrubbed.ref_ds.labels)
print("loss (lower is better):", small_loss)

# %% [markdown]

# Now let's calculate the loss recovered so far.

# %%

print((small_loss - uncond_loss) / (everything_loss - uncond_loss))

# %% [markdown]

# This is a more complete graph. It won't recover as much loss, unfortunately, because it's actually checking a smaller
# subset of the model because it doesn't include all paths from the relevant heads and MLPs to the origin. There's often
# a tradeoff between specficity and accuracy.

# %%

# real hypothesis

# More heads we know about

inhibition: list[MLPHeadAndPosSpec] = [MLPHeadAndPosSpec(10, 7, 3)]

# Create basic correspondence

corr = Correspondence()

m_log_probs = corr_root_matcher
log_probs = InterpNode(name="log_probs", cond_sampler=SameLabelSampler())
corr.add(log_probs, m_log_probs)

# add log_probs -> heads (which tend not to look at other heads)

for layer, head, position in heads:
    name = node_loc_to_name(layer, head, position)
    m_head = rc.restrict(
        m_log_probs.chain(name), term_early_at=match_nodes_except([node_loc_to_name(layer, head, position)])
    )
    i_head = log_probs.make_descendant(name=name, cond_sampler=SameLabelSampler())
    corr.add(i_head, m_head)

# add log_probs -> mlps -> other heads and mlps

for layer, _, position in mlps:
    name = node_loc_to_name(layer, "mlp", position)
    m_mlp = rc.restrict(
        m_log_probs.chain(name), term_early_at=match_nodes_except([node_loc_to_name(layer, "mlp", position)])
    )
    i_mlp = log_probs.make_descendant(name=name, cond_sampler=SameLabelSampler())
    corr.add(i_mlp, m_mlp)

    for h_layer, h_head, h_position in heads + mlps:
        if h_layer > layer or h_layer == layer and h_head == "mlp":
            continue
        h_name = node_loc_to_name(h_layer, h_head, h_position)
        m_to_head = rc.restrict(
            m_mlp.chain(h_name),
            term_early_at=match_nodes_except(
                [node_loc_to_name(layer, "mlp", position), node_loc_to_name(h_layer, h_head, h_position)]
            ),
        )
        i_to_head = i_mlp.make_descendant(name=h_name, cond_sampler=SameLabelSampler())
        corr.add(i_to_head, m_to_head)

# add inhibition heads

for layer, head, position in inhibition:
    name = node_loc_to_name(layer, head, position)
    m_mlp = rc.restrict(
        m_log_probs.chain(name), term_early_at=match_nodes_except([node_loc_to_name(layer, head, position)])
    )
    i_mlp = log_probs.make_descendant(name=name, cond_sampler=SameLabelSampler())
    corr.add(i_mlp, m_mlp)

tree_experiment = Experiment(
    circuit,  # type: ignore
    dataset,  # type: ignore
    corr,
)
tree_scrubbed = tree_experiment.scrub(100)
# tree_scrubbed.print()

# %%

tree_outputs = tree_scrubbed.evaluate(eval_settings)

tree_logit_diff = logit_diff(tree_outputs.cpu(), tree_scrubbed.ref_ds.labels)
print("logit diff (less negative is better):", tree_logit_diff)

tree_loss = loss_fn(tree_outputs.cpu(), tree_scrubbed.ref_ds.labels)
print("loss (lower is better):", tree_loss)

# %% [markdown]

# Here's the percentage of loss recovered - 89%, still pretty good!

# %%

print((tree_loss - uncond_loss) / (everything_loss - uncond_loss))

# %%
