
# REMIX Day 5, Part 2 - Model Loading

In this notebook, we'll go step by step through the process of loading GPT2-small. We'll apply various modifications to make it easier to write our experiments.

## Table of Contents

- [Learning Objectives](#learning-objectives)
- [Readings](#readings)
- [Model Loading](#model-loading)
    - [Token Embeddings](#token-embeddings)
- [configure_transformer](#configuretransformer)
- [conform_all_modules](#conformallmodules)
- [t.call substitution](#tcall-substitution)
- [Removing all modules but layer norm](#removing-all-modules-but-layer-norm)
- [Renaming of blocks](#renaming-of-blocks)
- [Split By Position](#split-by-position)
- [More Renames](#more-renames)
- [Running the Circuit!](#running-the-circuit)
- [Logit Differences](#logit-differences)
- [Labels](#labels)
- [Sanity Check](#sanity-check)

## Learning Objectives

After today's material, you should be able to:

- Rewrite and rename the model to make it easier to write experiments

## Readings

None


```python
import os
import sys
import rust_circuit as rc
import torch
import torch as t
import os
from interp.circuit.interop_rust.model_rewrites import To, configure_transformer
from interp.circuit.interop_rust.module_library import load_model_id
from interp.tools.interpretability_tools import print_max_min_by_tok_k_torch
from interp.tools.indexer import TORCH_INDEXER as I
from remix_d5_utils import IOIDataset
from typing import Optional

MAIN = __name__ == "__main__"
if MAIN:
    from remix_extra_utils import check_rust_circuit_version

    check_rust_circuit_version()

```

## Model Loading

We want our circuit to be split both by attention head and by token position, and as a metric we are only interested in the logit difference between IO and S. The basic plan is:

- Split by heads using `configure_transformer`
- Split by token position using `split_to_concat`

We can't split by token position until we know how many token positions there. We'll make an `IOIDataset` for demo purposes that will have a specific sequence length.

You've probably loaded this model in Day 2, but if not then this will take some time to download the weights from RRFS.

The `t.bind_w` circuit is short for "transformer bind weights" - it's a `Module` with the pretrained weights included, but no token or positional embeddings yet.


```python
ioi_dataset = IOIDataset(prompt_type="mixed", N=100)
MAX_LEN = ioi_dataset.prompts_toks.shape[1]
MODEL_ID = "gelu_12_tied"
(circ_dict, tokenizer, model_info) = load_model_id(MODEL_ID)
unbound_circuit = circ_dict["t.bind_w"]

```

### Token Embeddings

To get our token embeddings, we'll use a placeholder for the tokens themselves of the appropriate length. Recall that we don't include a batch dimension.

`bind_to_input` computes two Circuits: the attention mask and an `Add` of the positional and token embeddings. This is a plain Python function so you can look into it if you like.


```python
tokens_sym = rc.Symbol.new_with_random_uuid((MAX_LEN,), name="tokens")
token_embeds = rc.GeneralFunction.gen_index(circ_dict["t.w.tok_embeds"], tokens_sym, 0, name="tok_embeds")
bound_circuit = model_info.bind_to_input(unbound_circuit, token_embeds, circ_dict["t.w.pos_embeds"])
print(bound_circuit)

```

## configure_transformer

Here, `split_by_head_config="full"` means that each head will have its own set of parameters.

`use_flatten_res=True` means that the input to each module is the sum of all previous modules. Instead of having nested `Add`s for each block, we have one `Add` with a long list of all the components (MLPs and attention layers) coming before.

Exercise: in `transformed_circuit`, examine the first block's attention layer 'b0.a' and make sure you understand how the heads are split.

Exercise: in `transformed_circuit`, examine the third block `b2` and make sure you understand how the flattening works. Compare to the previous structure in `bound_circuit`.


```python
transformed_circuit = bound_circuit.update(
    "t.bind_w",
    lambda c: configure_transformer(
        c, To.ATTN_HEAD_MLP_NORM, split_by_head_config="full", use_pull_up_head_split=True, use_flatten_res=True
    ),
)
"TODO: YOUR CODE HERE"

```

## conform_all_modules

Our network still has some symbolic sizes in it, but we're now using concrete inputs with known sizes, so there's no need for symbolic sizes anymore. In fact, we won't be able to split by position if the sequence dimension is still symbolic.

Calling `rc.conform_all_modules` walks the tree and for each `Module`, replaces symbolic sizes with known ones wherever possible.


```python
print("Before conforming: ", transformed_circuit.get_unique("b0.a"))
transformed_circuit = rc.conform_all_modules(transformed_circuit)
print("After conforming: ", transformed_circuit.get_unique("b0.a"))

```

## t.call substitution

The outer `t.call` `Module`'s only purpose is to have placeholders for the attention mask and the input embeddings. We don't need it anymore, so we can substitute it out to make our circuit a bit more readable.


```python
subbed_circuit = transformed_circuit.cast_module().substitute()
subbed_circuit = subbed_circuit.rename("logits")
subbed_circuit.print_html()

```

## Removing all modules but layer norm

`Module`s are helpful when building the model to avoid copy pasting code, but they are not very helpful when we want to specify precise paths through the model: we cannot chain through `Symbol` instances!

Exercise: make a new `subbed_circuit` where you substitute away all `Module`s except for layer norms.



```python
"TODO: update subbed_circuit so the test passes"
expected = [
    "a0.norm",
    "a1.norm",
    "a10.norm",
    "a11.norm",
    "a2.norm",
    "a3.norm",
    "a4.norm",
    "a5.norm",
    "a6.norm",
    "a7.norm",
    "a8.norm",
    "a9.norm",
    "final.call",
    "final.norm",
    "m0.norm",
    "m1.norm",
    "m10.norm",
    "m11.norm",
    "m2.norm",
    "m3.norm",
    "m4.norm",
    "m5.norm",
    "m6.norm",
    "m7.norm",
    "m8.norm",
    "m9.norm",
]
actual = sorted([c.name for c in subbed_circuit.get(rc.Module)])
assert actual == expected

```

## Renaming of blocks

We will use these names a lot in our future experiment. To make it easier, we shorten them to remove useless information (such as the 'b' for 'block' that is not interesting for us. We consider attention heads and mlps, not blocks as a whole.)


```python
renamed_circuit = subbed_circuit.update(rc.Regex("[am]\\d(.h\\d)?$"), lambda c: c.rename(c.name + ".inner"))
renamed_circuit = renamed_circuit.update("t.inp_tok_pos", lambda c: c.rename("embeds"))
for l in range(model_info.params.num_layers):
    "b0 -> a1.input, ... b11 -> final.input"
    next = "final" if l == model_info.params.num_layers - 1 else f"a{l + 1}"
    renamed_circuit = renamed_circuit.update(f"b{l}", lambda c: c.rename(f"{next}.input"))
    "b0.m -> m0, etc."
    renamed_circuit = renamed_circuit.update(f"b{l}.m", lambda c: c.rename(f"m{l}"))
    renamed_circuit = renamed_circuit.update(f"b{l}.m.p_bias", lambda c: c.rename(f"m{l}.p_bias"))
    renamed_circuit = renamed_circuit.update(f"b{l}.a", lambda c: c.rename(f"a{l}"))
    renamed_circuit = renamed_circuit.update(f"b{l}.a.p_bias", lambda c: c.rename(f"a{l}.p_bias"))
    for h in range(model_info.params.num_layers):
        "b0.a.h0 -> a0.h0, etc."
        renamed_circuit = renamed_circuit.update(f"b{l}.a.h{h}", lambda c: c.rename(f"a{l}.h{h}"))
renamed_circuit.print_html()

```

## Split By Position

We'll want to have the ability to target only specific sequence positions for interventions such as "the output of head 0.0 at sequence position 5". To do this, we'll create an intermediate `Index` node named "a0.h0_at_idx_5", and then concatenate a bunch of these back together to get a "a0.h0_by_pos" that is the same as the original "a0.h0".

Exercise: implement `split_to_concat_axis_0`.


```python
def split_to_concat_axis_0(c: rc.Circuit) -> rc.Concat:
    """Turns `c` into `Concat(c[0:1], c[1:2], ...)`.

    Each index should be named {c.name}_at_idx_{i}.
    The output name should be {c.name}_by_pos.

    Simplified version of rc.split_to_concat.
    """
    "TODO: YOUR CODE HERE"
    pass


head_and_mlp_matcher = rc.IterativeMatcher(rc.Regex("^(a\\d+.h\\d+?|m\\d+)$"))
split_circuit = renamed_circuit.update(head_and_mlp_matcher, split_to_concat_axis_0)
a0h0 = split_circuit.get_unique("a0.h0_by_pos")
for i in range(16):
    idx = a0h0.children[i]
    assert isinstance(idx, rc.Index)
    assert idx.name == f"a0.h0_at_idx_{i}"

```

## More Renames

Again, we rename some names to make the circuit easier to read. We use a trick to make renaming faster: we create a dictionary of old names to new names, and then use the `update` method to rename all the nodes at once.


```python
new_names_dict = {}
for l in range(model_info.params.num_layers):
    for i in range(MAX_LEN):
        for h in range(model_info.params.num_layers):
            new_names_dict[f"a{l}.h{h}_at_idx_{i}"] = f"a{l}_h{h}_t{i}"
        new_names_dict[f"m{l}_at_idx_{i}"] = f"m{l}_t{i}"
split_circuit = split_circuit.update(
    rc.Matcher(*list(new_names_dict.keys())), lambda c: c.rename(new_names_dict[c.name])
)
split_circuit.print_html()

```

## Running the Circuit!

You may also be wondering about how to actually run your circuit!

Here we are expanding the model to have a batch dimension using `rc.Sampler`.

We also replace the tokens with a `DiscreteVar`, but then tell our `Sampler` to run on all datums in the `DiscreteVar`'s input dataset in order (so there is no actual randomness).

We print the top 5 tokens with max and min logits. In the top 5 logits, IO appears first, but you can also find S. IO is put a probability much stronger than S, but the proba of S is still much higher than a random name.



```python
def evaluate_on_dataset(c: rc.Circuit, tokens: torch.Tensor, group: Optional[rc.Circuit] = None):
    """Run the circuit on all elements of tokens. Assumes the 'tokens' module exists in the circuit."""
    arr = rc.Array(tokens, name="tokens")
    var = rc.DiscreteVar(arr)
    c2 = c.update("tokens", lambda _: var)
    transform = rc.Sampler(rc.RunDiscreteVarAllSpec([var.probs_and_group]))
    return transform.sample(c2).evaluate()


all_logits = evaluate_on_dataset(split_circuit, ioi_dataset.prompts_toks[:5, :])
next_token_logits = all_logits[torch.arange(5), MAX_LEN - 1]
for i in range(5):
    print(f'\n\nExpected completions for prompt: "{ioi_dataset.prompts_text[i]}"')
    print_max_min_by_tok_k_torch(next_token_logits[i], k=5)

```

## Logit Differences

We'll then create a new circuit that is only computing the logit difference betwee IO and S (the metric we're interested in). This circuit will also contain the labels of our dataset. We will use it to run path patching experiments where we are only interested in changing the inputs (and so don't have to deal with the labels after that).


```python
io_s_labels = torch.cat([ioi_dataset.io_tokenIDs.unsqueeze(1), ioi_dataset.s_tokenIDs.unsqueeze(1)], dim=1)
device_dtype = rc.TorchDeviceDtype(dtype="float32", device="cpu")
tokens_device_dtype = rc.TorchDeviceDtype(device_dtype.device, "int64")
labels = rc.cast_circuit(rc.Array(torch.zeros(2), name="labels"), tokens_device_dtype.op()).cast_array()
labels1 = rc.Index(labels, I[0], name="labels1")
labels2 = rc.Index(labels, I[1], name="labels2")
logit1 = rc.GeneralFunction.gen_index(split_circuit.index((-1,)), labels1, index_dim=0, batch_x=True, name="logit1")
logit2 = rc.GeneralFunction.gen_index(split_circuit.index((-1,)), labels2, index_dim=0, batch_x=True, name="logit2")
logit_diff_circuit = rc.Add.minus(logit1, logit2)

```

## Labels

Let's add the labels to our circuit. The labels are `DiscreteVar`s inserted in the circuit and the `group` variable stores the order they are sampled from. As long as the same `group` is used between sentences and labels, they'll be kept in the same order.


```python
def add_labels_to_circuit(c: rc.Circuit, tokens: torch.Tensor, labels: torch.Tensor):
    """Run the circuit on all elements of tokens. Assumes the 'tokens' module exists in the circuit."""
    assert tokens.ndim == 2 and tokens.shape[1] == MAX_LEN
    batch_size = tokens.shape[0]
    print(batch_size)
    group = rc.DiscreteVar.uniform_probs_and_group(batch_size)
    c = c.update("labels", lambda _: rc.DiscreteVar(rc.Array(labels, name="labels"), probs_and_group=group))
    return (c, group)


(logit_diff_circuit, group) = add_labels_to_circuit(logit_diff_circuit, ioi_dataset.prompts_toks, io_s_labels)

```

## Sanity Check

As a sanity check, let's run the logit diff circuit on our dataset


```python
c = logit_diff_circuit.update(
    "tokens", lambda _: rc.DiscreteVar(rc.Array(ioi_dataset.prompts_toks, name="tokens"), probs_and_group=group)
)
transform = rc.Sampler(rc.RunDiscreteVarAllSpec([group]))
results = transform.sample(c).evaluate()
print(f"Logit difference for the first 5 prompts: {results[:5]}")
print(f"Average logit difference: {results.mean()} +/- {results.std()}")
print("Testing that results are in a usual range: ")
assert results.mean() > 2.5 and results.mean() < 4

```