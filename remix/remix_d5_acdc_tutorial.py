
# %%
import os
import time
os.chdir(os.path.expanduser("~/mlab2/"))
import sys
import uuid
from typing import Optional, Tuple
import interp.tools.optional as op
import numpy as np
import rust_circuit as rc
import torch
from interp.circuit.causal_scrubbing.experiment import (
    Experiment,
    ExperimentCheck,
    ExperimentEvalSettings,
    ScrubbedExperiment,
)
from interp.circuit.causal_scrubbing.hypothesis import (
    Correspondence,
    CondSampler,
    ExactSampler,
    FuncSampler,
    InterpNode,
    UncondSampler,
    chain_excluding,
    corr_root_matcher,
)
from interp.circuit.interop_rust.algebric_rewrite import (
    residual_rewrite,
    split_to_concat,
)
from interp.circuit.interop_rust.model_rewrites import To, configure_transformer
from interp.circuit.interop_rust.module_library import load_model_id
from interp.tools.indexer import TORCH_INDEXER as I
from torch.nn.functional import binary_cross_entropy_with_logits
import remix_d4_part2_test as tests
from remix_d4_part2_setup import ParenDataset, ParenTokenizer, get_h00_open_vector
import wandb

MAIN = __name__ == "__main__"
if MAIN:
    from remix_extra_utils import check_rust_circuit_version

    check_rust_circuit_version()
SEQ_LEN = 42
NUM_EXAMPLES = 4000
MODEL_ID = "jun9_paren_balancer"
PRINT_CIRCUITS = True
ACTUALLY_RUN = True
SLOW_EXPERIMENTS = True
DEFAULT_CHECKS: ExperimentCheck = True
EVAL_DEVICE = "cuda:0"
MAX_MEMORY = 20000000000
BATCH_SIZE = 2000

import IPython

if IPython.get_ipython() is not None:
    IPython.get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore
    IPython.get_ipython().run_line_magic("autoreload", "2")  # type: ignore
from copy import deepcopy
from typing import List
from tqdm import tqdm

from interp.circuit.causal_scrubbing.dataset import Dataset
from interp.circuit.causal_scrubbing.hypothesis import corr_root_matcher
from interp.circuit.interop_rust.model_rewrites import To, configure_transformer
from interp.circuit.interop_rust.module_library import load_model_id
from remix_d5_acdc_utils import (
    ACDCCorrespondence,
    ACDCExperiment,
    ACDCInterpNode,
)
from interp.circuit.projects.gpt2_gen_induction.rust_path_patching import make_arr

#%% [markdown]
# In this notebook we'll setup and run a basic version of Automatic Circuit Discovery Code (ACDC) on the parenbalancer task on a three layer transformer. ACDC is a method for automatically finding subgraphs of your model that explain model behaviour well. The structure of this notebook is as follows:
# <ul>
#     <li>Set up a model and dataset</li>
#     <li>Define a DAG of all nodes of interest in the model and run an ACDC experiment</li>
#     <li>Introduce of more nodes into the DAG and run another ACDC experiment</li>
#     <li>Use hierarchical structure to speed up ACDC</li>
# </ul>
#
# Consider the notation in <a href="https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing#2_Setup">the casual scrubbing post</a>. <b>In ACDC, we only ever consider interpretations I that are subgraphs of the computational graph G</b>. ACDC is a greedy algorithm for heuristically finding a subgraph that describes model behavior well. There are two steps to the algorithm:
# <ol>
#     <li>Expanding nodes: algebraically rewriting a leaf node to add its inputs. `as`</li>
#     <li>Removing an edge in I.</li>
# </ol>
#
# So it is similar to a Breadth-First Search. Here's pseudocode for ACDC:
#
# Note: for a fast implementation, we batch the child_node loop

print(
    """
# sort G so the a node is always processed before its inputs (so the OUTPUT node is G[0])
G.reverse_topological_sort()

# initialize I to be the graph with only the output node
I = Correspondence(G[0], G[0].matcher)
# this means we don't scrub anything, initially

metric = compute_metric(G, I)

# set some ACDC threshold
threshold = 0.1
# (larger = more nodes, but slower runtime)

node_queue = [G.root()]

while node_queue is not empty:

    node = node_queue.pop()
    I.expand(node) # add all of node's inputs to I

    for child_node in node.children:
        I.remove(parent_node=node, child_node=child_node)
        new_metric = compute_metric(G, I)

        if abs(new_metric - metric) < threshold:
            # child_node->node wasn't important
            metric = new_metric
        else:
            # child_node->node was important
            I.add(parent_node=node, child_node=child_node)

            if child_node not in node_queue:
                node_queue.append(child_node)

print(I) # print the final ACDC graph!
"""
)

# %% [markdown]
# Let's get to it. We'll start by
# <ul>
#     <li> Loading the model and building the circuit</li>
#     <li> Loading the dataset (parentheses strings)</li>
#     <li> Creating the metric for the parenbalancer task (cross entropy with logits) </li>
# </ul>
# %%
"""
Load model and build circuit
"""
(circ_dict, _, model_info) = load_model_id(MODEL_ID)
circuit = circ_dict["t.bind_w"]
toks_uuid = uuid.UUID("ce34280e-169f-40bd-b78e-8adeb4274aba")
tokens_arr = rc.Symbol(
    (SEQ_LEN, ParenTokenizer.vocab_size), uuid=toks_uuid, name="tokens"
)
tok_embeds = rc.Einsum.from_fancy_string(
    "seqlen vocab_size, vocab_size hidden -> seqlen hidden",
    tokens_arr,
    circ_dict["t.w.tok_embeds"],
    name=f"tok_embeds",
)
attn_mask = rc.Add.minus(
    rc.Scalar(1), rc.Index(tokens_arr, I[:, ParenTokenizer.PAD_TOKEN]), name="pos_mask"
)
circuit = model_info.bind_to_input(
    circuit, tok_embeds, circ_dict["t.w.pos_embeds"], attn_mask
)
circuit = circuit.update(
    "t.bind_w",
    lambda c: configure_transformer(
        c,
        To.ATTN_HEAD_MLP_NORM,
        split_by_head_config="full",
        use_pull_up_head_split=True,
        use_flatten_res=True,
        flatten_components=True,
    ),
)
circuit = circuit.cast_module().substitute()
circuit = rc.Index(circuit, I[0]).rename("logits_pos0")
circuit = rc.conform_all_modules(circuit)
circuit = circuit.update("t.call", lambda c: c.rename("logits"))
circuit = circuit.update("t.call", lambda c: c.rename("logits_with_bias"))
circuit = circuit.update(
    rc.Regex("[am]\\d(.h\\d)?$"), lambda c: c.rename(c.name + ".inner")
)
circuit = circuit.update("t.inp_tok_pos", lambda c: c.rename("embeds"))
circuit = circuit.update("t.a.mask", lambda c: c.rename("padding_mask"))
for l in range(model_info.params.num_layers):
    circuit = circuit.update(f"b{l}.m", lambda c: c.rename(f"m{l}"))
    circuit = circuit.update(f"b{l}.a.h0", lambda c: c.rename(f"a{l}.h0"))
    circuit = circuit.update(f"b{l}.a.h1", lambda c: c.rename(f"a{l}.h1"))
    next = "final" if l == model_info.params.num_layers - 1 else f"a{l + 1}"
    circuit = circuit.update(f"b{l}", lambda c: c.rename(f"{next}.input"))
printer = rc.PrintHtmlOptions(
    shape_only_when_necessary=False,
    traversal=rc.restrict(
        rc.IterativeMatcher(
            "embeds", "padding_mask", "final.norm", rc.Regex("^[am]\\d(.h\\d)?$")
        ),
        term_if_matches=True,
    ),
)
circuit = rc.substitute_all_modules(circuit)
circuit.print_html()

"""
Load dataset
"""
ds = ParenDataset.load()

"""
Create metric
"""


def bce_with_logits_loss(logits: torch.Tensor, labels: torch.Tensor):
    """
    Computes the binary cross entropy loss for the provided labels.
    logits: [batch, 2]. Class 0 is unbalanced logit, class 1 is balanced logit.
    labels: [batch]. True if balanced.
    """
    targets = labels.to(dtype=logits.dtype, device=logits.device)
    logit_diff = logits[..., 1] - logits[..., 0]
    correct = (logit_diff > 0) == targets
    loss = binary_cross_entropy_with_logits(logit_diff, targets, reduction="none")
    return (loss, correct)


def cross_entropy_metric(dataset: Dataset, logits: torch.Tensor, mean=True,) -> float:
    labels = dataset.arrs["is_balanced"].evaluate()
    logit_diff = bce_with_logits_loss(logits, labels)[0]
    if mean:
        return logit_diff.mean().item()
    else:
        return float(logit_diff)


# %% [markdown]
# We'll run ACDC only paying attention to the MLPs and the attention heads at first. We'll define the set of names of nodes that our interpretation graph can contain in the all_names1 object. Then we'll instantiate the correspondence in template_corr1 and add the root node.
#
# Time to define the DAG. This DAG represents the **maximal hypothesis graph**: any node found in this DAG can later be added to our hypothesis graph, while nodes we do not add to this DAG will never be added to our hypothesis graph. For example, if we never specify MLPs in the DAG, then ACDC will not directly check for the influence of any MLP, and will never add an MLP to the hypothesis graph.
# For ACDC, each node in the DAG is named after the corresponding node in the rc.Circuit. **Importantly, ACDC requires that the nodes are named uniquely**. This means 'qkv nodes' like 'a.q' found on layer 3 and at head 2 will need to be renamed to a string which can uniquely identify it from other qkv nodes nodes, like 'a3.h2.q'.
#
# We first specify the `all_names1` object which stores the names of every node we want to add to our maximal hypothesis graph. Then, we define our correspondence in `template_corr1`, where the edges are implemented as .parents and .children of the ACDCInterpNode objects. Be careful to avoid specifying cycles here.
# %%
attention_head_name = "a{layer}.h{head}"
mlp_name = "m{layer}"
embed_name = "tok_embeds"
root_name = "final.inp"
no_layers = 3
no_heads = 2

all_names1 = (
    [embed_name, root_name]
    + [mlp_name.format(layer=layer) for layer in range(no_layers)]
    + [
        attention_head_name.format(layer=layer, head=head)
        for layer in range(no_layers)
        for head in range(no_heads)
    ]
)
all_names1 = set(all_names1)
template_corr1 = ACDCCorrespondence(all_names=all_names1)
root = ACDCInterpNode(root_name, is_root=True)
template_corr1.add(root)

# %% [markdown]
# Now that we have instantiated the correspondence, we'll create the DAG of the maximal interpretation graph (the maximum allowable hypothesis graph that ACDC can find). Running the below cell creates the maximal intepretation graph and visualizes it.
# %%
# Running list of nodes that have children in the residual stream
all_residual_stream_parents: List[ACDCInterpNode] = []
all_residual_stream_parents.append(root)
print("Constructing big Correspondence...")
for layer in tqdm(range(no_layers - 1, -1, -1)):

    # add attention heads
    head_nodes_list = []
    for head in range(no_heads):
        head_node = ACDCInterpNode(attention_head_name.format(layer=layer, head=head))
        for node in all_residual_stream_parents:
            node.add_child(head_node)
            head_node.add_parent(node)
        template_corr1.add(head_node)
        head_nodes_list.append(head_node)
    all_residual_stream_parents += head_nodes_list

# add embedding node
embed_node = ACDCInterpNode(embed_name)
for node in tqdm(all_residual_stream_parents):
    node.add_child(embed_node)
    embed_node.add_parent(node)
template_corr1.add(embed_node)

template_corr1.show()
# %% [markdown]
# The next cell checks that there are no cycles in the DAG
# %%
template_corr1.topologically_sort_corr()
# %% [markdown]
# Now let's define an ACDC experiment.
# %%
num_examples = 100
exp1 = ACDCExperiment(
    circuit=circuit,
    ds=ds[:num_examples],
    ref_ds=ds[num_examples : 2 * num_examples],
    template_corr=template_corr1,
    metric=cross_entropy_metric,
    random_seed=1234,
    num_examples=num_examples,
    check="fast",
    threshold=0.03,
    verbose=False,
    parallel_hypotheses=10,
    using_wandb=False,
)
exp1._nodes.show()

print(exp1.cur_metric)
es = [deepcopy(exp1)]  # to checkpoint experiments. This can be ignored
#%%
# New check! This makes sure exp._nodes is reflecting the same things as exp._base_circuit (which is actually used for internal fast computations)
# if you keyboard interrupt or do cursed things, this is good to check
exp1.check_circuit_conforms()
#%% [markdown]
# This cell should produce an image of the first step of ACDC
exp1.step()
es.append(deepcopy(exp1))
exp1._nodes.show()
#%% [markdown]
# An example of using the ACDC hypothesis graph as a causal scrubbing hypothesis:
# note that the values here are NOT the same as the values in the ACDCExperiment; this is normal causal scrubbing
# and so involves randomisation rather than patching to the same dataset.


def test(corr: Correspondence) -> float:
    experiment = Experiment(
        circuit=circuit, dataset=ds[:num_examples], corr=corr, num_examples=100,
    )
    scrubbed_experiment = experiment.scrub()
    logits = scrubbed_experiment.evaluate()
    return cross_entropy_metric(scrubbed_experiment.ref_ds, logits)


print(test(es[0]._nodes))
print(test(es[1]._nodes))
# %% [markdown]
# Remove the commenting below to run ACDC. The algorithm stops when it cannot iterate any further.
#
# Estimated run time: 8 seconds
# %%
# TODO: remove commenting below to run ACDC

# time_before = time.time()
# terminate = False
# while terminate == False:
#     try:
#         exp1.step()
#     except:
#         total_time = time.time() - time_before
#         print(f"\nTotal time: {total_time:.2f} seconds")
#         terminate = True
# %% [markdown]
# Run the next cell to visualize the final interpretation graph
# %%
exp1._nodes.show()
# %% [markdown]
# Now that you are familiar with ACDC, let's examine the motivation for using hierarchy in our interpretation graphs.
#
# In this section, we will
# <ul>
#     <li> Run ACDC on a simple example without hierarchy </li>
#     <li> Run ACDC on the same example with hierarchy </li>
# </ul>
# Let's investigate the impact of using query key and value nodes in order to expand our search for a hypothesis. We'll do this by creating a new correspondence that has query, key, and value nodes instead of the attention heads, while keeping the other nodes the same. We'll then run ACDC on this new correspondence.
#
# We'll start by creating a new circuit that has unqiue node names for the query, key, and value nodes. We'll need to do this because the names of the nodes in the circuit must match the names of the nodes in the correspondence, and the names of the nodes in the correspondence must be unique.
# %%
def create_path_matcher(
    start_node: rc.MatcherIn, path: list[str], max_distance=4
) -> rc.IterativeMatcher:
    """
    Creates a matcher that matches a path of nodes, given in a list of names, where the
    maximum distance between each node on the path is max_distance
    """

    initial_matcher = rc.IterativeMatcher(start_node)
    max_dis_path_matcher = lambda name: rc.restrict(
        rc.Matcher(name), end_depth=max_distance
    )
    chain_matcher = initial_matcher.chain(max_dis_path_matcher(path[0]))
    for i in range(1, len(path)):
        chain_matcher = chain_matcher.chain(max_dis_path_matcher(path[i]))
    return chain_matcher


q_path = [
    "a.comb_v",
    "a.attn_probs",
    "a.attn_scores",
    "a.attn_scores_raw",
    "a.q_p_bias",
    "a.q",
]
k_path = [
    "a.comb_v",
    "a.attn_probs",
    "a.attn_scores",
    "a.attn_scores_raw",
    "a.k_p_bias",
    "a.k",
]
v_path = ["a.comb_v", "a.v_p_bias", "a.v"]
qkv_paths = {"q": q_path, "k": k_path, "v": v_path}
num_layers = 3
num_heads = 2
qkv_name = "a{layer}.h{head}.{qkv}"
new_circuit = circuit
for l in range(num_layers):
    for h in range(num_heads):
        for qkv in ["q", "k", "v"]:
            qkv_matcher = create_path_matcher(f"a{l}.h{h}", qkv_paths[qkv])
            new_circuit = new_circuit.update(
                qkv_matcher, lambda c: c.rename(f"a{l}.h{h}.{qkv}")
            )
print(
    f"\nPlay around with the circuits below: the old circuits has generic names for the query key and value nodes, \nwhile the new circuit has unique names for each query key and value node"
)
print(f"\nOld circuit at a2.h0")
circuit.get_unique("a2.h0").print_html()
print(f"\nNew circuit at a2.h0")
new_circuit.get_unique("a2.h0").print_html()

# %% [markdown]
# Now let's create a new correspondence that has query, key, and value nodes instead of the attention heads.
# %%
all_names2 = (
    [embed_name, root_name]
    + [mlp_name.format(layer=layer) for layer in range(no_layers)]
    + [
        qkv_name.format(layer=layer, head=head, qkv=qkv)
        for layer in range(no_layers)
        for head in range(no_heads)
        for qkv in ["q", "k", "v"]
    ]
)
all_names2 = set(all_names2)
template_corr2 = ACDCCorrespondence(all_names=all_names2)
root = ACDCInterpNode(root_name, is_root=True)
template_corr2.add(root)

all_residual_stream_parents: List[ACDCInterpNode] = []
all_residual_stream_parents.append(root)
print("Constructing big Correspondence...")
for layer in tqdm(range(no_layers - 1, -1, -1)):

    # add MLP
    mlp_node = ACDCInterpNode(mlp_name.format(layer=layer))
    for node in all_residual_stream_parents:
        node.add_child(mlp_node)
        mlp_node.add_parent(node)
    template_corr2.add(mlp_node)
    all_residual_stream_parents.append(mlp_node)

    # add qkv nodes
    qkv_nodes_list = []
    for head in range(no_heads):
        for qkv in ["q", "k", "v"]:
            qkv_node = ACDCInterpNode(qkv_name.format(layer=layer, head=head, qkv=qkv))
            for node in all_residual_stream_parents:
                node.add_child(qkv_node)
                qkv_node.add_parent(node)
            template_corr2.add(qkv_node)
            qkv_nodes_list.append(qkv_node)
    all_residual_stream_parents += qkv_nodes_list

# add embedding node
embed_node = ACDCInterpNode(embed_name)
for node in tqdm(all_residual_stream_parents):
    node.add_child(embed_node)
    embed_node.add_parent(node)
template_corr2.add(embed_node)

template_corr2.show()
# %% [markdown]
# Check for no cycles
# %%
template_corr2.topologically_sort_corr()
# %% [markdown]
# Now let's run ACDC on this new correspondence.
#
# Approx runtime: 165 seconds
# %%
num_examples = 100
exp2 = ACDCExperiment(
    circuit=new_circuit,
    ds=ds[:num_examples],
    ref_ds=ds[num_examples : 2 * num_examples],
    template_corr=template_corr2,
    metric=cross_entropy_metric,
    random_seed=1234,
    num_examples=num_examples,
    check="fast",
    threshold=0.03,
    verbose=False,  # Gives live updates on the progress of the algorithm if True
    parallel_hypotheses=10,
    using_wandb=False,
)
exp2._nodes.show()

# %%
# TODO: remove commenting below to run ACDC

# time_before = time.time()
# terminate = False
# while terminate == False:
#     try:
#         exp2.step()
#     except:
#         total_time = time.time() - time_before
#         print(f"\nTotal time: {total_time:.2f} seconds")
#         terminate = True
# %% [markdown]
# Run the next cell to visualize the final interpretation graph
# %%
exp2._nodes.show()
# %% [markdown]
# Not bad, but we can do better. Let's try to add more structure to the correspondence.
#
# We'll now define the all_names3 object which will contain both attention heads and query, key, and value nodes. We'll construct the DAG such that the attention heads are parents of the query, key, and value nodes, and the query, key, and value nodes are parents of earlier layers in the residual stream.

# PS. an explainer on how the nodes are arranged in hierarchical structure is given in the next cell.
# %%
all_names3 = (
    [embed_name, root_name]
    + [mlp_name.format(layer=layer) for layer in range(no_layers)]
    + [
        attention_head_name.format(layer=layer, head=head)
        for layer in range(no_layers)
        for head in range(no_heads)
    ]
    + [
        qkv_name.format(layer=layer, head=head, qkv=qkv)
        for layer in range(no_layers)
        for head in range(no_heads)
        for qkv in ["q", "k", "v"]
    ]
)
all_names3 = set(all_names3)
template_corr3 = ACDCCorrespondence(all_names=all_names3)
root = ACDCInterpNode(root_name, is_root=True)
template_corr3.add(root)

all_residual_stream_parents: List[ACDCInterpNode] = []
all_residual_stream_parents.append(root)
print("Constructing big Correspondence...")
for layer in tqdm(range(no_layers - 1, -1, -1)):

    # add MLP
    mlp_node = ACDCInterpNode(mlp_name.format(layer=layer))
    for node in all_residual_stream_parents:
        node.add_child(mlp_node)
        mlp_node.add_parent(node)
    template_corr3.add(mlp_node)
    all_residual_stream_parents.append(mlp_node)

    """
    Add the heads and qkv nodes
        - loop over heads
            - instantiate head node as ACDCInterpNode
            - add child-parent edge to all_residual_stream_parents
            - add head node to template_corr
            - loop over qkv nodes
                - instantiate qkv node as ACDCInterpNode
                - add child-parent edge to head node
                - add qkv node to template_corr
                - add qkv nodes to qkv_nodes_list
        - add all qkv nodes to all_residual_stream_parents
    """
    # add heads and qkv's
    head_nodes_list = []
    qkv_nodes_list = []
    for head in range(no_heads):
        head_node = ACDCInterpNode(attention_head_name.format(layer=layer, head=head))
        head_nodes_list.append(head_node)
        for node in all_residual_stream_parents:
            node.add_child(head_node)
            head_node.add_parent(node)
        template_corr3.add(head_node)
        for qkv in ["q", "k", "v"]:
            qkv_node = ACDCInterpNode(f"a{layer}.h{head}.{qkv}")
            head_node.add_child(qkv_node)
            qkv_node.add_parent(head_node)
            template_corr3.add(qkv_node)
            qkv_nodes_list.append(qkv_node)
    all_residual_stream_parents += qkv_nodes_list


# add embedding node
embed_node = ACDCInterpNode(embed_name)
for node in tqdm(all_residual_stream_parents):
    node.add_child(embed_node)
    embed_node.add_parent(node)
template_corr3.add(embed_node)

template_corr3.show()
# %% [markdown]
# Check for no cycles
# %%
template_corr3.topologically_sort_corr()
# %% [markdown]
# Now let's run ACDC on this new correspondence.
#
# Approx runtime: 18 seconds
# %%
num_examples = 100
exp3 = ACDCExperiment(
    circuit=new_circuit,
    ds=ds[:num_examples],
    ref_ds=ds[num_examples : 2 * num_examples],
    template_corr=template_corr3,
    metric=cross_entropy_metric,
    random_seed=1234,
    num_examples=num_examples,
    check="fast",
    threshold=0.03,
    verbose=False,  # Gives live updates on the progress of the algorithm if True
    parallel_hypotheses=10,
    using_wandb=False,
    remove_redundant=True,
)
exp3._nodes.show()

#%% [markdown]
# Run the next cell to visualize the final interpretation graph
# %%
# run until error

import time

time_before = time.time()
# %% [markdown]
# Remove the commenting below to run ACDC.
# %%
# TODO: remove the commenting below
idx = 0
for idx in range(10000):
    exp3.step()
    exp3._nodes.show(fname="acdc_plot_" + str(idx) + ".png")
    if exp3.current_node is None:
        print("Done")
        break
exp3._nodes.show()
#%%

exp3._nodes.show()

#%%