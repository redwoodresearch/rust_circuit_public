
# REMIX Day 3 - Replicating Results on Induction Heads

Today you'll be replicating the results on induction heads from our [writeup](https://www.lesswrong.com/posts/j6s9H9SHrEhEfuJnq/causal-scrubbing-results-on-induction-heads). By the end, you'll have a more nuanced understanding of induction and be equipped to formulate and test your own hypotheses!

This second half of this notebook closely follows the writeup and I recommend having the writeup open to the corresponding section (where applicable) to look at the diagrams.

## Table of Contents

- [Learning Objectives](#learning-objectives)
- [Readings](#readings)
- [Setup](#setup)
- [The dataset](#the-dataset)
    - [Data loading](#data-loading)
    - [Data Inspection](#data-inspection)
    - [Tokens Where Induction is Likely](#tokens-where-induction-is-likely)
- [The Model](#the-model)
    - [Working on GPU](#working-on-gpu)
    - [Examining the model](#examining-the-model)
- [Random variables and sampling](#random-variables-and-sampling)
    - [Discrete variables](#discrete-variables)
    - [Sampling discrete variables](#sampling-discrete-variables)
    - [Sampling the input dataset](#sampling-the-input-dataset)
- [Revisiting some `rust_circuit` concepts](#revisiting-some-rustcircuit-concepts)
    - [Slicing and indexing](#slicing-and-indexing)
    - [Modules](#modules)
- [Evaluating model performance: constructing the loss](#evaluating-model-performance-constructing-the-loss)
    - [Inputs and targets](#inputs-and-targets)
    - [Model Binding](#model-binding)
    - [Cumulants](#cumulants)
- [Causal scrubbing](#causal-scrubbing)
    - [Setting up sampler](#setting-up-sampler)
    - [Custom printing](#custom-printing)
- [Establishing a Baseline](#establishing-a-baseline)
    - [Scrubbing all inputs](#scrubbing-all-inputs)
    - [Rewriting the model to split up the heads](#rewriting-the-model-to-split-up-the-heads)
    - [Substitution](#substitution)
    - [Scrubbing Induction Heads](#scrubbing-induction-heads)
- [Initial Naive Hypothesis](#initial-naive-hypothesis)
    - [The embeddings --> value hypothesis](#the-embeddings----value-hypothesis)
        - [Debugging tip - circuit diffing](#debugging-tip---circuit-diffing)
    - [The embeddings --> query hypothesis](#the-embeddings----query-hypothesis)
    - [The previous-token head --> key hypothesis](#the-previous-token-head----key-hypothesis)
    - [Scrubbing them all together](#scrubbing-them-all-together)
- [Bonus: Showing previous token head only depends on the previous token](#bonus-showing-previous-token-head-only-depends-on-the-previous-token)
    - [Splitting up the previous token head into two parts](#splitting-up-the-previous-token-head-into-two-parts)
    - [Using ModulePusher so we can scrub only one part](#using-modulepusher-so-we-can-scrub-only-one-part)
    - [Actually scrubbing the non mask_prev tokens.](#actually-scrubbing-the-non-maskprev-tokens)
- [Scrubbing them all together](#scrubbing-them-all-together-)

## Learning Objectives

After today's material, you should be able to:

- Sample from `Circuit`s containing random variables represented by `DiscreteVar`
- Customize `PrintOptions` to control color and expansion of appropriate nodes
- Write scrubbing code manually to test hypotheses

## Readings

- [Induction Head Writeup on Less Wrong](https://www.lesswrong.com/posts/j6s9H9SHrEhEfuJnq/causal-scrubbing-on-induction-heads-part-4-of-5)

## Setup


```python
import os
import sys
import uuid
from pprint import pprint
import rust_circuit as rc
import torch
from interp.circuit.interop_rust.model_rewrites import To, configure_transformer
from interp.circuit.interop_rust.module_library import load_model_id, negative_log_likelyhood
from interp.tools.data_loading import get_val_seqs
from interp.tools.indexer import SLICER as S
from interp.tools.indexer import TORCH_INDEXER as I
from interp.tools.rrfs import RRFS_DIR
from torch.testing import assert_close
import remix_d3_test as tests

MAIN = __name__ == "__main__"
DEVICE = "cuda:0"
if MAIN:
    from remix_extra_utils import check_rust_circuit_version

    check_rust_circuit_version()

```

## The dataset

Before we can actually run the experiments, we have a lot of preparation to get through.

We'll start by loading and examining the dataset, which is text from the validation set of OpenWebText.

### Data loading

As discussed in Day 2, when we parse a string representation of a `Circuit`, `rc.Parser` will automatically download referenced tensors from RRFS. The first run of the below cell might take a few seconds, but later runs should be nearly instant.


```python
seq_len = 300
n_files = 12
reload_dataset = False
toks_int_values: rc.Array
if reload_dataset:
    dataset_toks = torch.tensor(get_val_seqs(n_files=n_files, files_start=0, max_size=seq_len + 1)).cuda()
    (n_samples, _) = dataset_toks.shape
    toks_int_values = rc.Array(dataset_toks.float(), name="toks_int_vals")
    print(f'new dataset "{toks_int_values.repr()}"')
else:
    P = rc.Parser()
    toks_int_values = P("'toks_int_vals' [104091,301] Array 3f36c4ca661798003df14994").cast_array()

```

### Data Inspection

Machine learning is "garbage in, garbage out" so it's important to understand your data.

Here the data consists of 104091 examples, each consisting of 301 tokens.

To inspect the data we need the `tokenizer` that was used to convert text to tokens. This is typically stored with the model. We'll examine the model in a bit, but we need to load it here to get access to the tokenizer.

Again, this will take some time on the first run.


```python
model_id = "attention_only_2"
(loaded, tokenizer, extra_args) = load_model_id(model_id)

```

Exercise: convert the first two training examples back to text using [`tokenizer.batch_decode`](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.batch_decode) and manually inspect the data. Can you spot opportunities for induction or induction-like behaviors to help with predictions?


```python
"TODO: YOUR CODE HERE"

```

### Tokens Where Induction is Likely

Out of around 50,000 tokens in the vocabulary, we're narrowing our investigation to a smaller list of around 10,000 tokens. We tried to choose tokens "A" where "hard" induction ("AB...AB") is particularly helpful. (Hard induction is induction with exactly repeated tokens, as contrasted with "soft" induction which may copy over sentence structure or some other feature of the previous token). For each token in the vocabulary, `good_induction_candidate[token_id]` is 1 if the token is part of the short list.

In this work, we're not interested in explaining everything the induction heads do, only what they do on these examples (where we can reasonably expect them to be largely doing induction rather than other things).

See the appendix of the writeup for information on how these were chosen.

Exercise: Convert all the tokens in the short list back to text (again using [`tokenizer.batch_decode`](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.batch_decode)) and print them out to get a sense of what sorts of tokens exist and are likely to benefit from induction.

Exercise: What is the longest token in this set of good induction candidates?

Optional exercise: Find a non-English token. What does it mean? Does it kinda make sense that this token would benefit from induction?


```python
CACHE_DIR = f"{RRFS_DIR}/ryan/induction_scrub/cached_vals"
good_induction_candidate = torch.load(f"{CACHE_DIR}/induction_candidates_2022-10-15 04:48:29.970735.pt").to(
    device=DEVICE, dtype=torch.float32
)
"TODO: YOUR CODE HERE"

```

## The Model

### Working on GPU

`rc.cast_circuit` is a general function to cast an entire `Circuit` to a different device and/or dtype. In this case of arrays like `toks_int_values`, it's just the same as doing `Array(toks_int_values.to(DEVICE, dtype="int64"))`, but it's good to know this function for more complicated cases.

Note that unlike the PyTorch method `nn.Module.to` which modifies the module in-place, `rc.cast_circuit` returns a new `Circuit` and doesn't modify the input `Circuit`. This aligns with the general philosophy in the circuits library to never modify in-place.

Move the dataset and model to the GPU:


```python
toks_int_values = rc.cast_circuit(toks_int_values, rc.TorchDeviceDtypeOp(device="cuda", dtype="int64")).cast_array()
loaded = {s: rc.cast_circuit(c, rc.TorchDeviceDtypeOp(device="cuda")) for (s, c) in loaded.items()}

```

### Examining the model

At this stage, print out a representation of the model using the code below and take a minute to appreciate the mighty two-layer attention only model in all its glory. Other than some chunks being missing, there's only one substantial difference in architecture from the GPT-2-small model you implemented yesterday.


```python
orig_circuit = loaded["t.bind_w"]
tok_embeds = loaded["t.w.tok_embeds"]
pos_embeds = loaded["t.w.pos_embeds"]
if MAIN:
    orig_circuit.print_html()

```

Exercise: inspect the circuit. When compared to GPT-2, this model is missing blocks (e.g., the MLP layers). In terms of the blocks that it does contain, how do they differ from the corresponding blocks in GPT-2?

We suggest getting practice looking at the printout. Note that in the printout, `orig_circuit.print()` with default options will omit repeated modules, so you only see the attention module expanded under `'b1'` and not under `'b0'`. You can also try `orig_circuit.print_html()`.

<details>
<summary>Can you give me a super specific hint about where to look?</summary>

Focus on the embeddings. Try this: `rc.Matcher("a.qk_input").get_unique(orig_circuit).print()`

</details>

<details>

<summary>Solution</summary>

The way positional embeddings work is different - the "shortformer" encoding is used. It's important to understand that in GPT, the position embedding is added to the token embedding at the very beginning, meaning positional information is mixed in with token information within the residual stream.

In shortformer, instead of multiplying the Q weight with the attention input, we multiplying the Q weight with (attention input + positional embedding). The same thing goes for K weight (but not V), and this applies at every layer. For more, see [the Shortformer paper](https://aclanthology.org/2021.acl-long.427.pdf).

</details>



## Random variables and sampling

The circuits that we built yesterday were able to represent deterministic computational graphs. The circuits library also needs ways to deal with sampling from random distributions (and especially sampling over the dataset or some subset of it). In this context, we only need to deal with sampling from finite sets, so we will specialize to discrete random variables.

Here we take a brief look at how the circuits library deals with discrete random variables. In terms of what you need for the research phase, you need familiarity with how things are structured, but most of the code dealing with random variables and sampling will be provided as a library (which we'll see in tomorrow's material).

### Discrete variables

Discrete random variables are created in `rust_circuit` using the `rc.DiscreteVar` class. If you hover over `rc.DiscreteVar`, you'll see that it takes two arguments: a Circuit `values` specifying the values the variable can take, and another Circuit called `probs_and_group`. To specify a random variable we need to specify the values it can take and the probabilities of each of those outcomes. These correspond to the two arguments of `rc.DiscreteVar`, except that `probs_and_group` serves a second purpose, which is to track which random variables should be sampled together, as we'll see below.

Here are some simple examples. Follow along with this code:


```python
dice_values = rc.Array(torch.arange(1, 7).to(dtype=torch.float))
dice1 = rc.DiscreteVar(dice_values)
probs = dice1.probs_and_group.cast_tag().node
if MAIN:
    print(f"The values the dice can take are: {dice_values.value}")
    print("Not specifying group_and_probs gives you the uniform distribution.")
    print(
        f"The default probs_and_group object is wrapped in a Tag (to add a UUID) but we can unwrap that to see the probabilities, which are {probs}."
    )

```

We mentioned above that `probs_and_group` served a second purpose, which is to track which random variables should be sampled together. The idea here is that if you set the `probs_and_group` attribute to the same object in two different random variables, those variables will be sampled together, i.e., the samples will be perfectly correlated.


```python
dice2 = rc.DiscreteVar(dice_values, dice1.probs_and_group)
dice3 = rc.DiscreteVar(dice_values)
if MAIN:
    print(f"dice2 will be perfectly correlated with dice")
    print(
        "By default, not specifying probs_and_group will give you a new uniform distribution, so dice and dice3 are uncorrelated."
    )

```

### Sampling discrete variables

To test these claims about correlation, we need to be able to sample these random variables.

To sample a random variable, the `rust_circuit` library uses an `rc.Sampler` object, which needs to be initialized with an `rc.SampleSpec` object.

Today you'll need `rc.RandomSampleSpec`, for when we want to sample randomly, and `rc.RunDiscreteVarAllSpec`, which ignores the probabilities and samples every  input (which can be useful when trying to figure out what's going on with a circuit).

Follow along again:


```python
random_sampler = rc.Sampler(rc.RandomSampleSpec((10,)))
all_values_spec = rc.RunDiscreteVarAllSpec.create_full_from_circuits(dice1)
all_values_sampler = rc.Sampler(all_values_spec)
if MAIN:
    for i in range(1, 4):
        print(f"Dice {i}: ", random_sampler.sample(locals()[f"dice{i}"]).evaluate())
    print("All values: ", all_values_sampler.sample(dice1.add(dice1)).evaluate())

```

Optional exercise: using these classes, estimate the expectation of (i) the value of `dice1` multiplied by the value of `dice2` and (ii) the value of `dice1` multiplied by the value of `dice3`.


```python
"TODO: YOUR CODE HERE"

```

### Sampling the input dataset

Let's now apply all this to our dataset.

Run the code below and note that `toks_int_var` has a shape that represents a single sampled datum. It is not, however, explicitly computable. Intuitively it represents the random discrete variable over their input dataset.

We can however, sample from them using a `Sampler` object! This will add in a batch dimension we can evaluate them over.

You can also 'sample' every possible value of the discrete variable. The `group` argument tells it which set of random variables to sample (as mentioned above, all variables that share an identical `group` attribute are sampled together).


```python
toks_int_var = rc.DiscreteVar(toks_int_values, name="toks_int_var")
if MAIN:
    print("Variable")
    print("  Shape: ", toks_int_var.shape)
    print("  Computable: ", toks_int_var.is_explicitly_computable)
sampled_var = rc.Sampler(rc.RandomSampleSpec((200,))).sample(toks_int_var)
if MAIN:
    print("\nRandom samples:")
    print("  Shape: ", sampled_var.shape)
    print("  Computable: ", sampled_var.is_explicitly_computable)
group = toks_int_var.probs_and_group
on_all_var = rc.Sampler(rc.RunDiscreteVarAllSpec([group])).sample(toks_int_var)
if MAIN:
    print("\n All samples:")
    print("  Shape: ", on_all_var.shape)
    print("  Computable: ", on_all_var.is_explicitly_computable)

```

## Revisiting some `rust_circuit` concepts

### Slicing and indexing

What about if we want to get at individual tokens? Well for that we need `Indexer`s. Let's review.

In Day 1 we introduced the `slice` object as a way to represent indexing into a single dimension. Recall that you can use the slice constructor with two integers like `slice(start, stop)`. What does it do when you pass just one argument?

One might guess that you get `slice(start, None, None)`, but this is wrong - actually it is the same as `slice(None, stop, None)`. This is consistent with the way that the `range()` builtin works, but is still easy to trip over. If you want to avoid remembering this and also save a couple keystrokes, we have a helper object usually imported as `S`. You can use it like so:


```python
assert S[1:] == slice(1, None, None)
assert S[:1] == slice(None, 1, None)
assert S[2:3:-1] == slice(2, 3, -1)
if MAIN:
    try:
        S[2]
    except Exception as e:
        print("It is an error to pass a plain int to SLICER.")
    try:
        S[2:3, 4:5]
    except Exception as e:
        print("It is an error to pass multiple slices to SLICER.")

```

Analogously, a helper object usually imported as `I` provides an equivalent and succint way to generate representations of (possibly multi-dimensional) indexing.

These objects are commonly used with the `rc.Index` class: in the last line of the following block we show how to use an Indexer object with `rc.Index` to  specialize `toks_int_var` to the first position.


```python
assert I[5] == (5,)
assert I[5, 6] == (5, 6)
assert I[5:6] == (slice(5, 6, None),)
assert I[5:6, 7:8] == (slice(5, 6, None), slice(7, 8, None))
first_tok_int_var = rc.Index(toks_int_var, I[0], name="first_tok_int_var")

```

### Modules

We saw Modules yesterday but they can be quite confusing because they are close in concept-space to a bunch of similar things, so it's worth recapping them. Modules are in the category of concepts that once you grok what's going on, it all becomes obvious, so if you notice that you are confused later in the day, please ask a TA to explain.

Yesterday we explained modules in the context of neural networks. Today we'll go a bit more abstract. Let's start with an analogy: When programming in Python, we can do things like add two tensors `t1` and `t2`, for which the syntax in Python is (obviously) `t_1 + t_2`. You know how to represent the same computation in `rust_circuit` land: we wrap the tensors in `rc.Array` and then combine them using an `rc.Add` circuit, i.e., `rc.Add(rc.Array(t1, name="t1"), rc.Array(t2, name="t2"), name="t1 + t2")`.

When programming in Python, it is also useful to use functions to encapsulate behaviour, e.g., `def plus(a, b): return a + b` (which you can also write in Python as `plus = lambda a, b: a + b`). The `rc.ModuleSpec` class in `rust_circuit` is a pretty direct mapping of the concept of a function in a programming language.

Let's get comfortable with this by creating a `rc.ModuleSpec` for the plus function defined in the previous paragraph.

We first need to make `rc.Symbol`s for the arguments. To keep things super-simple here, we'll make symbols with shape `(2,)`, i.e., they are vectors in 2-dimensions.


```python
a = rc.Symbol.new_with_random_uuid(shape=(2,), name="a")
b = rc.Symbol.new_with_random_uuid(shape=(2,), name="b")

```

Then we need a `rc.Circuit` to represent the function body. For the `plus` function this is easy:


```python
spec_circuit = rc.Add(a, b, name="a + b")

```

Finally, the `rust_circuit` representation of our plus function `plus = lambda a, b: a + b` is


```python
spec = rc.ModuleSpec(circuit=spec_circuit, arg_specs=[rc.ModuleArgSpec(a), rc.ModuleArgSpec(b)])

```

Note that we needed to wrap the `Symbol`s in `rc.ModuleArgSpec`. Hover over that in VS Code to see the docstring and you'll see that this lets us specify whether we can run the function on batched inputs (we needed this yesterday), and whether we can pass arguments with different sizes (also needed yesterday).

So that's the abstract function. But when a function appears in a computational tree, it needs specific values for the arguments and that's essentially what an `rc.Module` is for: it's a function, i.e., an `rc.ModuleSpec`, together with a list of `rc.Circuits` that are bound to the arguments.

(Binding variables is a concept from logic. You might want to have a quick look at [wikipedia](https://en.wikipedia.org/wiki/Free_variables_and_bound_variables#Examples) if you haven't seen it before.)

Anyway, to see how this works, let's create some specific vectors and then bind them to the arguments.


```python
t1 = rc.Array(torch.Tensor([1, 0]), name="t1")
t2 = rc.Array(torch.Tensor([0, 1]), name="t2")
module = rc.Module(spec=spec, name="t1 + t2", a=t1, b=t2)
module.print_html()

```

Study the printout and make sure you understand it **completely**, as you'll be looking at bunch of more complicated versions later today.

You'll see the `rc.Module` at top, which has three children: (i) its `rc.ModuleSpec`, (ii) the argument binding for `a`, and (iii) the argument binding for `b`.

An argument binding looks like `'t1' [2] Array ! 'a'`. You read this as: the value to the left of the '!' is bound to the symbol with the name to the right of the '!'.

Optional exercise: what happens if multiple Symbols have the same name `a`? What happens if there's a non-symbol with the name `a`?

<details>
<summary>
Solution
</summary>
The name on the right of the '!' refers to the argument with that name in the current module, i.e., the symbols with that name in the `rc.ModuleSpec` for that module. It's fine if there are symbols with the same name elsewhere, or non-symbols with that name.

The `rust_circuit` library requires that the names of arguments in an `rc.ModuleSpec` be unique, i.e., running the following will result in an exception:

```python
try:
    a1 = rc.Symbol.new_with_random_uuid(shape=(2,), name="a")
    a2 = rc.Symbol.new_with_random_uuid(shape=(2,), name="a")
    spec_circuit = rc.Add(a1, a2, name="a + a")
    spec = rc.ModuleSpec(circuit=spec_circuit, arg_specs=[rc.ModuleArgSpec(a1), rc.ModuleArgSpec(a2)])
except rc.ConstructModuleArgsDupNamesError as e:
    print("Exception raised: ", e)
```

The bindings are also always listed in the same order as the symbols appear in the list `arg_specs` that we used to create the `rc.ModuleSpec`, a fact we'll need to exploit later.
</details>

<br />
Notice that our function is stored in unevaluated form. We can evaluate it, like any circuit:


```python
module.evaluate()

```

There's another operation we'll need extensively today: `rc.Module.substitute`. Calling `substitute()` will 'dissolve' the module, substituting the symbols with the objects bound to them. This is the equivalent of inlining our Python `plus` function, i.e., replacing `plus(t1, t2)` with `t1 + t2`.

Try that now and see how the printout changes.


```python
module.substitute().print_html()

```

Like functions in Python, modules can be nested, and so let's get experience looking at the printouts of those, as you'll be doing that a bunch later today.


```python
plus1 = rc.Module(spec=spec, name="plus1", a=t1, b=t2)
plus2 = rc.Module(spec=spec, name="plus2", a=plus1, b=t2)
plus3 = rc.Module(spec=spec, name="plus3", a=plus1, b=plus2)
plus3.print_html()

```

Again, study the printout. Some things to note:

* The copies of the spec are marked "(repeat)" and are not expanded by default.
* The second child of 'plus3' is printed as `'plus1' Module ! 'a'`, which means the output of the `plus1` module is bound
  to the argument corresponding to the the `rc.Symbol 'a'`, just as we set it up.

Now let's call substitute on the `plus2` module, and look at the printout. What changed? Does it look as you expect?


```python
plus3_with_partial_substitution = rc.Module(spec=spec, name="plus3", a=plus1, b=plus2.substitute())
plus3_with_partial_substitution.print_html()

```

## Evaluating model performance: constructing the loss

In this section, we'll construct the loss that our network was trained to minimize. Of course, we want to construct this as a circuit as well!

### Inputs and targets

Our model was trained to do next-token prediction.

Define `input_toks` to be all tokens except the last position, and `true_tokens` to be the corresponding ground truth next tokens (for every sequence position, not just for the last position). This is a good time to practice using `I`.


```python
input_toks: rc.Index
true_toks: rc.Index
"TODO: YOUR CODE HERE"
assert input_toks.shape == (seq_len,)
assert true_toks.shape == (seq_len,)

```

### Model Binding

The method `rc.get_free_symbols` helpfully tells us which `Symbol`s haven't been bound yet. This is useful for debugging purposes.


```python
if MAIN:
    print("Free symbols: ")
    pprint(rc.get_free_symbols(orig_circuit))

```

It's time to bind these free symbols. The function `rc.module_new_bind` is just a more succint way to create a `Module` instance then calling the `Module` constructor. You pass tuples containing symbol names and the values to bind and away you go! Note that this doesn't modify `orig_circuit`.

The node "t.input" represents the embedded tokens just like GPT, "a.mask" is the causal mask just like GPT, and "a.pos_input" is computed the same way as in GPT, but again in shortformer it will be used differently by the model.

Exercise: explain to your partner in your own words how "a.pos_input" will be used.


```python
idxed_embeds = rc.GeneralFunction.gen_index(tok_embeds, input_toks, index_dim=0, name="idxed_embeds")
assert extra_args.causal_mask, "Should not apply causal mask if the transformer doesn't expect it!"
causal_mask = rc.Array(
    (torch.arange(seq_len)[:, None] >= torch.arange(seq_len)[None, :]).to(tok_embeds.cast_array().value),
    f"t.a.c.causal_mask",
)
assert extra_args.pos_enc_type == "shortformer"
pos_embeds = pos_embeds.index(I[:seq_len], name="t.w.pos_embeds_idxed")
model = rc.module_new_bind(
    orig_circuit, ("t.input", idxed_embeds), ("a.mask", causal_mask), ("a.pos_input", pos_embeds), name="t.call"
)
assert model.are_any_found(orig_circuit)
assert not rc.get_free_symbols(model)
loss = rc.Module(negative_log_likelyhood.spec, **{"ll.input": model, "ll.label": true_toks}, name="t.loss")

```
For today's work, we only want to compute loss on the good induction candidates:


```python
is_good_induction_candidate = rc.GeneralFunction.gen_index(
    x=rc.Array(good_induction_candidate, name="tok_is_induct_candidate"),
    index=input_toks,
    index_dim=0,
    name="induct_candidate",
)
loss = rc.Einsum((loss, (0,)), (is_good_induction_candidate, (0,)), out_axes=(0,), name="loss_on_candidates")

```

### Cumulants

A cumulant is a concept in probability theory, but you don't need to know anything about cumulants right now. The one relevant fact for today is that the "first cumulant" of a distribution is just the regular old mean of a distribution that you already know about. (Higher order cumulants come up in [other research(https://arxiv.org/abs/2210.01892) done at Redwood.)

Right now, our `loss` node depends on the input `DiscreteVar`s. Since these are random variables, our loss will also being a random variable. By wrapping `loss` in an `rc.Cumulant`, we're saying that we will be interested in the mean loss over the input distribution.

This cumulant will have shape `(seq_len,)` since we're computing the loss at every position. We then take the mean to get the average loss per model prediction (just like regular LM loss).


```python
expected_loss_by_seq = rc.Cumulant(loss, name="t.expected_loss_by_seq")
expected_loss = expected_loss_by_seq.mean(name="t.expected_loss", scalar_name="recip_seq")
printer = rc.PrintHtmlOptions(
    shape_only_when_necessary=False,
    traversal=rc.new_traversal(
        term_early_at=rc.Regex("a\\.*.w.\\.*ind")
        | rc.Matcher(
            {"b", "final.norm", "idxed_embeds", "nll", "t.w.pos_embeds_idxed", "true_toks_int", "induct_candidate"}
        )
    ),
    comment_arg_names=True,
)
if MAIN:
    expected_loss.print(printer)

```

## Causal scrubbing

Congratulations! You made it through the prepatory work. It's finally time to do causal scrubbing!

Recall from the writeup that we'll be running our model on two inputs. One is the original input, and we'll use its next tokens to compute the loss. The other is the random other input, and we'll run the parts of the model we claim don't matter on this one.

Exercise: make another `DiscreteVar`, `toks_int_var_other` that will be uncorrelated with `toks_int_var`.


```python
"TODO: YOUR CODE HERE"
if MAIN:
    print("Your names should match these to make later validation much easier:")
assert toks_int_var.name == "toks_int_var"
assert toks_int_var_other.name == "toks_int_var_other"

```

Quick test:


```python
def seeder(c: rc.Circuit) -> int:
    """
    Just a silly way to get two fixed seeds.
    Setting seeds for consistent results between runs of this notebook.
    """
    if c == toks_int_var.probs_and_group:
        return 11
    elif c == toks_int_var_other.probs_and_group:
        return 22
    else:
        raise ValueError("Expected one of the probs_and_group we constructed earlier, but got something else!", c)


sampler = rc.Sampler(rc.RandomSampleSpec((200,), seeder=seeder))
assert (
    torch.corrcoef(
        torch.stack(
            (sampler.sample(toks_int_var).evaluate()[:, 10], sampler.sample(toks_int_var_other).evaluate()[:, 10])
        )
    )[0, 1]
    < 0.1
)

```

### Setting up sampler


```python
def sample_and_evaluate(c: rc.Circuit, num_samples: int = 16 * 128, batch_size=32) -> float:
    """
    More samples is better! It'll just take (linearly) longer to run.

    (In this notebook we aren't calculating error bars, but you're welcome to do so.)
    """

    def run_on_sampled(c: rc.Circuit) -> rc.Circuit:
        """
        Function for sampler to run after sampling (before we evaluate the resulting circuit).

        batch_to_concat breaks up num_samples dim into batches of batch_size so they can be evaluated separately (and not run out of memory).

        substitute_all_modules gets rid of Module nodes; compiler complains if you don't do this today, sorry.
        """
        return rc.batch_to_concat(rc.substitute_all_modules(c), axis=0, batch_size=batch_size)

    sampler = rc.Sampler(
        rc.RandomSampleSpec((num_samples,), seeder=seeder), run_on_sampled=run_on_sampled, device_dtype=c.device_dtype
    )
    return rc.optimize_and_evaluate(sampler.estimate(c)).item()

```

### Custom printing

Below is a helpful printer: it will color things getting the random input red, things getting the original input blue, things getting both purple, and things getting neither grey. It's good practice to play around with printing until you can clearly see what's going on in your Circuit.


```python
scrubbed = lambda c: c.are_any_found(toks_int_var_other)
not_scrubbed = lambda c: c.are_any_found(toks_int_var)


def scrub_colorer(c):
    getting_scrubbed = c.are_any_found(toks_int_var_other)
    getting_unscrubbed = c.are_any_found(toks_int_var)
    if getting_scrubbed and getting_unscrubbed:
        return "purple"
    elif getting_scrubbed:
        return "red"
    elif getting_unscrubbed:
        return "cyan"
    else:
        return "lightgrey"


scrubbed_printer = printer.evolve(
    colorer=scrub_colorer,
    traversal=rc.restrict(printer.traversal, term_early_at=lambda c: not c.are_any_found(toks_int_var_other)),
)
unscrubbed_out = sample_and_evaluate(expected_loss)
if MAIN:
    print(f"Loss with no scrubbing: {unscrubbed_out:.3f}")
assert_close(unscrubbed_out, 0.17, atol=0.01, rtol=0.001)

```

## Establishing a Baseline

### Scrubbing all inputs

When scrubbing, we want to compute our "percent loss recovered" as a metric. While this is generally sensible, it isn't completely satisfactory for various reasons. The metric can go over 100%, and it feels like researchers can [Goodhart](https://en.wikipedia.org/wiki/Goodhart%27s_law) the metric. We're thinking about ways to make this more valid involving having an adversary, but for now we'll just take the metric as an indicator that provides some evidence where higher (up to 100%) is better.

In the [Causal Scrubbing Appendix](https://www.lesswrong.com/posts/kcZZAsEjwrbczxN2i/causal-scrubbing-appendix#2_1__Percentage_of_loss_recovered__as_a_measure_of_hypothesis_quality), we gave a formula for this using a baseline where the inputs are scrubbed.

Concretely, we run our model on random inputs, while computing the loss w.r.t. the original labels. This isn't actually the baseline we will use (we'll explain more later) but it's a good warm-up.

Exercise: implement `scrub_input` and use it to replace the inputs to the model with uncorrelated inputs. You need to define the `rc.IterativeMatcher` `unused_baseline_path` so that `scrub_input` replaces the inputs but not the labels.


```python
def scrub_input(c: rc.Circuit, in_path: rc.IterativeMatcher) -> rc.Circuit:
    """Replace all instances of `toks_int_var` descended from in_path with `toks_int_var_other`"""
    "TODO: YOUR CODE HERE"
    pass


unused_baseline_path: rc.IterativeMatcher
"TODO: YOUR CODE HERE"
if MAIN:
    tests.test_all_inputs_matcher(unused_baseline_path, expected_loss)
unused_baseline = scrub_input(expected_loss, unused_baseline_path)
if MAIN:
    expected_loss.print()

```

Take a look at this print: does it look like what you expected?


```python
if MAIN:
    scrubbed_printer.print(unused_baseline)
unused_baseline_out = sample_and_evaluate(unused_baseline)
if MAIN:
    print(f"Loss with scrubbing the whole model: {unused_baseline_out:.3f}")
assert_close(unused_baseline_out, 0.81, atol=0.01, rtol=0.001)

```

### Rewriting the model to split up the heads

The actual baseline we want to use isn't random inputs to everything, but only random inputs to the induction heads. This represents (very roughly) a model that is working normally except that induction is "disabled".

We want to be able to pass different inputs into the induction heads and the other heads. To do this, we'll rewrite our transformer so that there's a node named "a1.ind" consisting of just heads 1.5 and 1.6, and a node "a1.not_ind" (called "a1 other" in the writeup) consisting of the other layer 1 heads.

For future experiments, we'll also want to separate the "previous token head" 0.0 into its own node named "a0.prev", and call the other layer 0 heads "a0.not_prev".

Exercise: read through the source code for `configure_transformer` and figure out how to call it so that these heads are split up. We're expecting you to use `use_pull_up_head_split=True`; the other arguments you should be able to figure out. Verify the printed circuit looks reasonable.

Warning: the tests here can be very finicky -- in particular, there are several ways to write the split_by_head config that are equivilant in meaning but the test will reject for silly reasons (e.g. S[0] != S[:1]).


```python
split_by_head_config = "TODO: YOUR CODE HERE"
by_head = configure_transformer(
    expected_loss.get_unique("t.bind_w"),
    to=To.ATTN_HEAD_MLP_NORM,
    split_by_head_config=split_by_head_config,
    use_pull_up_head_split=True,
    check_valid=True,
)

```
Sanity checks


```python
assert by_head.get({"a1.ind", "a1.not_ind", "a0.prev", "a0.not_prev"}, fancy_validate=True)

```
Bit of tidying: renames, and replacing symbolic shapes with their numeric values


```python
by_head = by_head.update(lambda c: ".keep." in c.name, lambda c: c.rename(c.name.replace(".keep.", ".")))
by_head = rc.conform_all_modules(by_head)
printer = printer.evolve(
    traversal=rc.restrict(
        printer.traversal, term_early_at=rc.Matcher({"b0", "a1.norm", "a.head"}) | rc.Regex("\\.*not_ind\\.*")
    )
)
if MAIN:
    printer.print(by_head)
    tests.test_by_head(by_head)
    print("Updating expected_loss to use the by_head version")
expected_loss = expected_loss.update("t.bind_w", lambda _: by_head)

```
replace symbolic shapes with their real values


```python
expected_loss = rc.conform_all_modules(expected_loss)

```

### Substitution
Note: this section can be pretty confusing! Take it slowly, talk it through with your partner, and be willing to ask call a TA to help explain things!

Let us focus on one particular part of the circuit, a module called "a.head.on_inp" inside of our induction heads:



```python
ind_head_on_inp_subcircuit = expected_loss.get_unique("a1.ind").get_unique("a.head.on_inp")
if MAIN:
    ind_head_on_inp_subcircuit.print(printer)

```

Recall that within a module `circ_a Add ! sym_a Symbol` means the module binds the value `circ_a` (an Add node) to the symbol `sym_a`, which is then required to appears in that module's spec.

Some things to notice:
 - This module is representing our induction heads as a function that takes three inputs: `a.q.input`, `a.k.input`, and `a.v.input`. These inputs are used to form the queries, keys, and values respectfully.
 - Normally when we run an attention head we use the same `[seq_len, hidden_size]` matrix as inputs to all three of these. However, it is possible to run the attention head on three different inputs! In fact this is necessary to replicate the causal scrubbing experiments.
- This `a.head.on_inp` module is responsible for binding these three inputs. It binds both the query and key inputs to a simple circuit which adds two symbols: `a.input` (representing the input to the attention head) and `pos_input` (represeting the positional embeddings). The value input is bound to the same `a.input`.

We want to be able to replicate an experiment where we change some of the tokens that are upstream of the value-input to the induction heads, but not change either the query-input or key-input. This would require writing an Iterative Macher that can match paths through the value-input but not the other two.

Unfortunately there is no way to do this with the current circuit. While we could write a matcher that matches only one copy of the `a.input` symbol, there's no way to chain that matcher to upstream to the embeddings. It's just a symbol, there are no embeddings upstream!

In this section we will rewrite the model so that writing these sorts of matchers is possible.


So what is this `a.input` symbol? What is it doing here? And where are the embeddings?

To answer that we need to zoom out from this particular a.head.on_inp module.

A more complete sub-circuit representing the induction head would look something like this:
```
mystery_module Module
  spec of mystery_module
      ...
      a.head.on_inp Module
        a.head Einsum
          ...
        a.qk_input Add ! a.q.input
          a.input
          pos_embeds
        a.qk_input Add ! a.k.input
          a.input
          pos_embeds
        a.input ! a.v.input
  output_of_a1_ln ! a.input
```

Here `mystery_module` is binding the `a.input` symbol to the `output_of_a1_ln` circuit.



So now how can we fix the problem described above (that we can't write an iterative matcher to perform the update we want)? Well, it should be possible to rewrite the circuit shown above into this form instead:

```
spec of mystery_module
  ...
    a.head.on_inp Module
      a.head Einsum
        ...
      a.qk_input Add ! a.q.input
        output_of_a1_ln
        pos_embeds
      a.qk_input Add ! a.k.input
        output_of_a1_ln
        pos_embeds
      output_of_a1_ln ! a.v.input
```
Then we could perform the above update on the copy of `a1.norm` that is bound to `a.v.input`!

This is analogous to the transformation between
```
polynomial_function = lambda x: x**2 + 3*x + 4
polynomial_function(23)
```
into `23**2 + 3*23 + 4`: we transform a function call (`polynomial_function(23)`) into the body of the function (`x**2 + 3*x + 4`), with the input symbol (`x`) replaced by it's bound value (23).

This transformation can be achieved by calling `.substitute()` on the `'a1.ind_sum.norm_call'` module!. This eliminates the module and performs the transformation above!

(you may need to use `cast_module()` to convince your typechecker that `a1.ind_sum.norm_call` really is a Module!)


Okay, enough talking! Time to peform this substitution.

Exercise: First, figure out which module bind through the circuit to find the identity of `a.input`.

Do this by examining the below print and looking for a line that ends in `! 'a.input'` as this is the symbol we are trying to substitute.

The print below only focuses on the sub-circuit representing the attention head.


```python
if MAIN:
    expected_loss.get_unique("b1.a.ind_sum").print(printer)

```

<details>
<summary>Solution</summary>
The module `a1.ind_sum.norm_call` binds `a1.norm` to `a.input`.
</details>


Now use `.update` to call substitute on this module!


```python
with_a1_ind_inputs_v1 = expected_loss
"TODO: YOUR CODE HERE"
ind_head_on_inp_subcircuit_v1 = with_a1_ind_inputs_v1.get_unique("a1.ind").get_unique("a.head.on_inp")
ind_head_on_inp_subcircuit_v1.print(printer.evolve(traversal=rc.new_traversal(term_early_at={"a.head", "ln"})))
assert "a.input" not in [symb.name for symb in rc.get_free_symbols(ind_head_on_inp_subcircuit_v1)]

```

Unfortunately we aren't done yet. There is still a single symbol that forces all three inputs to be the same. What is that symbol? What module binds it? Once again, figure out what module is binding it and substitute that module.


```python
if MAIN:
    with_a1_ind_inputs_v1.get_unique("b1.a.ind_sum").print(printer)
with_a1_ind_inputs_v2 = with_a1_ind_inputs_v1
"TODO: YOUR CODE HERE"
if MAIN:
    ind_head_on_inp_subcircuit_v2 = with_a1_ind_inputs_v2.get_unique("a1.ind").get_unique(
        rc.restrict("a.head.on_inp", end_depth=2)
    )
    printer.print(ind_head_on_inp_subcircuit_v2)
    assert set((symb.name for symb in rc.get_free_symbols(ind_head_on_inp_subcircuit_v2))) == {
        "a.w.q_h",
        "a.w.k_h",
        "a.mask",
        "a.w.v_h",
        "a.w.o_h",
        "a.pos_input",
        "t.input",
    }

```

We're very close! There's still one symbol that is shared across all three inputs that prevents us from changing the token embeddings through one path but not the other.


```python
if MAIN:
    with_a1_ind_inputs_v2.get_unique("b1.a.ind_sum").print(printer)
with_a1_ind_inputs_v3 = with_a1_ind_inputs_v2
"TODO: YOUR CODE HERE"
if MAIN:
    ind_head_on_inp_subcircuit_v3 = with_a1_ind_inputs_v3.get_unique("a1.ind").get_unique(
        rc.restrict("a.head.on_inp", end_depth=2)
    )
    printer.print(ind_head_on_inp_subcircuit_v3)
    print("Checking various things...")
    assert set((symb.name for symb in rc.get_free_symbols(ind_head_on_inp_subcircuit_v3))) == {
        "a.w.q_h",
        "a.w.k_h",
        "a.w.v_h",
        "a.w.o_h",
    }
    a1_ind = with_a1_ind_inputs_v3.get_unique("a1.ind")
    assert not rc.get_free_symbols(a1_ind), "there should be no free symbols in a1!"
    assert a1_ind.are_any_found(
        toks_int_var
    ), "toks_int_var should appear at least one in the subcircuit rooted at a1.ind"
    tests.test_with_a1_ind_inputs(with_a1_ind_inputs_v3)
    print("Checks passed! Well done!!")
with_a1_ind_inputs = with_a1_ind_inputs_v3

```

We can now progress on to replicating experiments.


### Scrubbing Induction Heads

To complete the baseline section, we need to run the model where the induction heads are "scrubbed" - all inputs to the induction heads are replaced with inputs chosen randomly.


```python
scrubbed_ind = scrub_input(with_a1_ind_inputs, rc.IterativeMatcher("a1.ind"))
if MAIN:
    scrubbed_printer.print(scrubbed_ind)
baseline_out = sample_and_evaluate(scrubbed_ind)
if MAIN:
    print(f"Loss with induction heads scrubbed: {baseline_out:.3f}")


def loss_recovered(l):
    return (l - baseline_out) / (unscrubbed_out - baseline_out)

```

## Initial Naive Hypothesis

We'll test each of the three claims in our hypothesis separately, and then combine them.

### The embeddings --> value hypothesis


Exercise: We'll want to replace the (embeddings --> induction heads' V input) path, and then later do the same for Q and K. To do this we'll write three iterative matchers upfront, one for each of these.

To do this we want to match the circuit that a.head.on_inp binds to a particular symbol. Unfortunately rust_circuits has no good utility for this yet. It is easiest to do this with `my_matcher.children_matcher({i})` which extends `my_matcher` to match the ith child of whatever it matched before.


```python
q_ind_input_matcher: rc.IterativeMatcher
k_ind_input_matcher: rc.IterativeMatcher
v_ind_input_matcher: rc.IterativeMatcher
"TODO: YOUR CODE HERE (define a1_ind_inputs_matcher)"
if MAIN:
    tests.test_qkv_ind_input_matchers(q_ind_input_matcher, k_ind_input_matcher, v_ind_input_matcher, with_a1_ind_inputs)

```

Now we can replicate the the [first experiment](https://www.lesswrong.com/posts/j6s9H9SHrEhEfuJnq/causal-scrubbing-on-induction-heads-part-4-of-5#The_embeddings___value_hypothesis) in the writeup!

Now is a good time to go back and read the corresponding section of the writeup about the emeddings -> value hypothesis.

Exercise: replicate the hypothesis in the writeup.


```python
text_printer = rc.PrintOptions(traversal=rc.new_traversal(end_depth=2), bijection=False)
if MAIN:
    with_a1_ind_inputs.get_unique(q_ind_input_matcher).print(text_printer)
v_matcher: rc.IterativeMatcher
"TODO: YOUR CODE HERE (define v_matcher so the scrub does the right thing)"
if MAIN:
    tests.test_v_matcher(v_matcher, with_a1_ind_inputs)

```

If you're having trouble understanding what's going on with your iterative matcher, looking at the scrubbed printer can help:


```python
scrubbed_v = scrub_input(with_a1_ind_inputs, v_matcher)
if MAIN:
    scrubbed_printer.print(scrubbed_v)

```

#### Debugging tip - circuit diffing

Recall that you can diff two circuits - this is sometimes useful for telling if you've actually updated the right nodes.


```python
if MAIN:
    print(rc.diff_circuits(scrubbed_v, with_a1_ind_inputs))
v_out = sample_and_evaluate(scrubbed_v)
v_loss_rec = loss_recovered(v_out)
if MAIN:
    print(f"Loss recovered by embeddings --> value hypothesis: {v_loss_rec:.3f}")
assert_close(v_loss_rec, 0.9, atol=0.01, rtol=0.01)

```

### The embeddings --> query hypothesis

Exercise: test this hypothesis - the code is similar to the previous one.


```python
q_matcher: rc.IterativeMatcher
"TODO: YOUR CODE HERE (define q_matcher)"
scrubbed_q = scrub_input(with_a1_ind_inputs, q_matcher)
q_out = sample_and_evaluate(scrubbed_q)
q_loss_rec = loss_recovered(q_out)
if MAIN:
    print(f"Loss recovered by embeddings --> query hypothesis: {q_out:.3f}")
assert_close(q_loss_rec, 0.51, atol=0.01, rtol=0.01)

```

### The previous-token head --> key hypothesis

Now we'll make use of the separation of the previous token head from the other heads ("not_prev").

Exercise: test this hypothesis.


```python
k_matcher: rc.IterativeMatcher
"TODO: YOUR CODE HERE (define k_matcher)"
scrubbed_k = scrub_input(with_a1_ind_inputs, k_matcher)
if MAIN:
    scrubbed_printer.print(scrubbed_k)
k_out = sample_and_evaluate(scrubbed_k)
k_loss_rec = loss_recovered(k_out)
if MAIN:
    print(f"Loss recovered by previous-token head --> key hypothesis: {k_loss_rec:.3f}")
assert_close(k_loss_rec, 0.83, atol=0.01, rtol=0.01)

```

### Scrubbing them all together

Exercise: scrub all three things at the same time. Tip: thanks to operator overloading, you can use the pipe character "|" on two `IterativeMatchers` to make a new one that matches if either matches. This is equivalent to passing the matchers to the `IterativeMatcher` constructor, but a little more succint.


```python
combined_matcher: rc.IterativeMatcher
"TODO: YOUR CODE HERE"
scrubbed_all = scrub_input(with_a1_ind_inputs, combined_matcher)
all_out = sample_and_evaluate(scrubbed_all)
all_loss_rec = loss_recovered(all_out)
if MAIN:
    print(f"Loss recovered by whole hypothesis: {all_loss_rec:.3f}")
assert_close(all_loss_rec, 0.37, atol=0.01, rtol=0.01)

```

Congrats! This concludes the main part of today's content.

This scrub does not exactly correspond to the hypothesis tested in the writeup, however. This is because while we tested that only the previous token head is important for the k input, we didn't test that the previous token head is actually a previous token head! That is we didn't test that only the previous token is important for this head.

In the section below we separate out the previous token head into two parts, one of which only cares about the previous token. This involves some reasonably complicated rewrites. While we do think it would be valuable practice, if you our short on time it is not necessary to understand the content of future days.


## Bonus: Showing previous token head only depends on the previous token
### Splitting up the previous token head into two parts

We want to rewrite a0.prev comb_v as sum of the contribution from the previous token, and the contribution from the remaining tokens.

Exercise: this term doesn't have a k dimension. How would you break it up by k?

<details>
<summary>Click on this to show hint</summary>

It's an Einsum, one of whose inputs does have this dimension.

</details>

<details>
<summary>Click on this to show answer</summary>

We can break up a.attn_probs first and then distribute.

</details>



Exercise: how would you break up a term that does have a k dimension into two terms: one the contribution from the previous token, and one from the rest?
<details>
<summary>Solution</summary>
We can make a mask array that picks out the previous position, and then write the attn_probs as attn_probs*mask + attn_probs*(1-mask).
</details>

Exercise: rewrite a0.prev comb_v as specified above. If you're feeling really brave, you can try to do this yourself now---otherwise, work through the substeps that follow.

<details>
<summary>Click on this to show substep (spoiler for earlier exercise!)</summary>

Exercise: write mask_prev and mask_not_prev. Be sure to give them names for later use!

</details>



```python
"TODO: YOUR CODE HERE"

```

<details>
<summary>Click on this to show substep (spoiler for earlier exercise!)</summary>

Exercise: find a0 prev attn_probs, and rewrite it as a sum of a prev term and a not_prev term.

</details>

<details>
<summary>Click on this to show hint</summary>

Use `rc.Einsum.from_einsum_string` to multiply attn_probs by a mask to get the masked attn_probs.

</details>



```python
mask_a0: rc.Circuit
"TODO: YOUR CODE HERE"

```


<details>
<summary>Click on this to show substep (spoiler for earlier exercise!)</summary>

Exercise: use rc.distribute with `suffix=".a.comb_v"` (the default naming is a bit ugly) to break up a0 comb_v into a sum of two terms, one corresponding to prev and one to not_prev.

</details>



```python
"TODO: YOUR CODE HERE"
k_to_a0_printer = printer.evolve(
    traversal=rc.new_traversal(
        term_early_at=rc.Matcher(
            {"ln", "final.norm", "idxed_embeds", "nll", "t.w.pos_embeds_idxed", "true_toks_int", "induct_candidate"}
        )
        | rc.Regex("\\.*not_ind\\.*")
        | (lambda c: not c.are_any_found("mask_prev.a.comb_v"))
    )
)

```

Does this print look like what you expected?


```python
if MAIN:
    k_to_a0_printer.print(mask_a0)

```

In the rewrite you did above, you have a lot of freedom for how to do the rewrite. Ultimately, if you get the correct k_loss_rec below, you did this right!

Here we test that you did the rewrite exactly the way we did it (including names, etc). If this fails, feel free to push on and see if you get the correct loss recovered, or just copy mask_a0 from the solutions.


```python
if MAIN:
    tests.test_mask_a0(mask_a0)

```

### Using ModulePusher so we can scrub only one part

There is a slight problem with our circuit now. Even though we have split up the spec of the module for the previous token head into the mask and not-mask portion, both portions are still bound to a single input by the head's module. We want to be able to scrub the input to one part of this spec and not the other.

In order to do this we can _push down_ this module, so that the input binding occurs separately within the previous-token mask and outside of the mask. This is analagous to `push_down_index` in many ways.

This is implimented in rust_circuits by the ModulePusher class. The class can be a little difficult to use, as it's performing a pretty complicated operation (and the docstring leaves something to be desired). This section isn't essential content for the future days, so if you are short on time you can read the solution file for this exercise and the next one: it is a higher priority to know that ModulePusher exists and roughly what it does than to be proficient in using it.

The first exercise is to use ModulePusher on a small toy circuit. Then, we'll use it on your circuit to fix the problem described above.

Exercise: Use `ModulePusher` to rewrite this circuit to be a sum of modules `b` and `c`.

Hint: By default ModulePusher will push modules down forever. You'll want to pass it a traversal to prevent this.




```python
s = "\n'a' Module\n  'add' Add\n    'b' Module\n      'add_b' Add\n        'sym_a' [] Symbol 05c93da1-ac33-47eb-b215-d1a8e8438a02\n        'sym_b' [] Symbol 5ae54c16-b570-4abd-ad37-f527c25494c1\n    'c' Module\n      'add_c' Add\n        'sym_a' [] Symbol 05c93da1-ac33-47eb-b215-d1a8e8438a02\n        'sym_c' [] Symbol cb22836a-b5cb-4e65-969e-563958249e81\n      'arg_c' [] Scalar 3 ! 'sym_c'\n  'arg_b' Einsum ij,j->i ! 'sym_b'\n    'within_arg' Module\n      'add_within' Add\n        'sym_d' [] Symbol fbb99b36-0809-4eae-ae10-7d38a90d1386\n        'w' [3] Scalar 4\n      'k' [7] Scalar 7 ! 'sym_d'\n    'x' [3] Scalar 8\n  'arg_a' [] Scalar 82.83 ! 'sym_a'\n"
module_of_add = rc.Parser()(s)
add_of_module = module_of_add
if MAIN:
    add_of_module.print_html()

```
Quick checks


```python
add_of_module.cast_add()
add_of_module.get_unique("b").cast_module()
add_of_module.get_unique("c").cast_module()
if MAIN:
    tests.test_add_of_module(add_of_module)

```

Exercise: Rewrite the `a1_ind_inputs_matcher` circuit so that the mask_prev branch has inputs separate from the mask_not_prev branch.

same_namer is a simple function for you to pass to ModulePusher in order to avoid it's default naming (which we don't want in this case).


```python
same_namer = lambda c, d, ms, n: c.name
a0_by_pos: rc.Circuit
"TODO: YOUR CODE HERE"
if MAIN:
    k_to_a0_printer.print(a0_by_pos)

```
Quick checks


```python
a0_prev_module = a0_by_pos.get_unique(rc.Matcher(rc.Module) & rc.Regex("mask_prev."))
assert a0_prev_module.are_any_found(toks_int_var)
assert not rc.get_free_symbols(rc.substitute_all_modules(a0_prev_module))

```

### Actually scrubbing the non mask_prev tokens.
Exercise: write a matcher that matches all toks -> a1.ind k paths except those going through a0.prev mask_prev


```python
fancy_k_matcher: rc.IterativeMatcher
"TODO: YOUR CODE HERE"
scrubbed_k = scrub_input(a0_by_pos, fancy_k_matcher)
if MAIN:
    scrubbed_printer.print(scrubbed_k)
    tests.test_fancy_k_matcher(fancy_k_matcher, a0_by_pos)
k_out = sample_and_evaluate(scrubbed_k)
k_loss_rec = loss_recovered(k_out)
if MAIN:
    print(f"Loss recovered by previous position --> previous-token head value --> key hypothesis: {k_loss_rec:.3f}")
assert_close(k_loss_rec, 0.82, atol=0.01, rtol=0.01)

```

## Scrubbing them all together

Exercise: once again, scrub all three things at the same time. It should be the same as above, but with `fancy_k_matcher`.


```python
combined_matcher: rc.IterativeMatcher
"TODO: YOUR CODE HERE"
scrubbed_all = scrub_input(a0_by_pos, combined_matcher)
all_out = sample_and_evaluate(scrubbed_all)
all_loss_rec = loss_recovered(all_out)
if MAIN:
    print(f"Loss recovered by whole hypothesis: {all_loss_rec:.3f}")
assert_close(all_loss_rec, 0.37, atol=0.01, rtol=0.01)

```

This replicates the initial naive hypothesis in the writeup!

Congratulations on completing all of day's content!

# Extra-bonus exercises

- Test some of the hypotheses in the writeup that we didn't cover here.
- Play with the dataset and see how your results change.
    - How would you expect a shorter or longer sequence length to affect the results? Check your prediction.
    - Can you think of a different way to filter/subset the data?
- Think up your own hypotheses and test them!
