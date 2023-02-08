# %%
"""
# REMIX Day 2 - Build Your Own GPT in Circuits

Today, you'll learn more features of the Circuits library and use them to write a GPT-style transformer from scratch. By the end, you'll be able to load pretrained weights and do inference on GPT-2.

<!-- toc -->

## Learning Objectives

After today's material, you should be able to:

- Explain all the individual operations in a transformer and how they combine
- Explain how batch dimensions are handled in Circuits
- Write your own `Module`s as necessary

## Readings

- [Language Modelling with Transformers](https://docs.google.com/document/d/1XJQT8PJYzvL0CLacctWcT0T5NfL7dwlCiIqRtdTcIqA/edit#) - note that we're going to completely ignore dropout, since it isn't used at inference time.
"""
# %%
from __future__ import annotations
import os
import sys
import torch
import torch as t
import torch.nn as nn
from tqdm.notebook import tqdm
import torch.nn.functional as F
from typing import (
    Optional,
    Union,
    cast,
    Callable,
    Iterable,
    Any,
    Type,
    Literal,
    Sequence,
)
import pandas as pd
from einops import rearrange, repeat
import rust_circuit as rc
from rust_circuit import (
    Circuit,
    GeneralFunction,
    Add,
    Scalar,
    Einsum,
    Index,
    Symbol,
    Array,
    Rearrange,
    Module,
    ModuleSpec,
    ModuleArgSpec,
)
from dataclasses import dataclass
import remix_d2_utils
from remix_d2_utils import LayerWeights, GPT2Weights, get_weights

pd.set_option("display.precision", 3)
MAIN = __name__ == "__main__"

if MAIN:
    from remix_extra_utils import check_rust_circuit_version

    check_rust_circuit_version()

if "SKIP":
    IS_CI = os.getenv("IS_CI")
    if IS_CI:
        sys.exit(0)


@dataclass(frozen=True)
class GPTConfig:
    """Constants used throughout the GPT2 model."""

    activation_function: str = "gelu"
    num_layers: int = 12
    num_heads: int = 12
    vocab_size: int = 50257
    hidden_size: int = 768
    max_position_embeddings: int = 1024
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5


config = GPTConfig()

# %%
"""

## The Symbol class

In Day 1, our `Circuit` had specific sizes like 28*28 "baked in", and it would've been a pain to reuse the same `Circuit` for a situation where the structure of the network was the same, but the sizes of some things were different.

It would be nice if we had a `Circuit` that represents the network architecture in the abstract. This `Circuit` wouldn't reference any weight `Array`s, and wouldn't reference specific sizes like the hidden size that can vary between models using this architecture. We could write this abstract thing once and then "substitute in" weights to obtain a `Circuit` representing a specific model we can evaluate.

While we're at it, it would be nice to solve the problem of batching. In Day 1, we manually modified our `Circuit` to have batching support. While most things "just work" with a batch dimension prepended to the front (thanks to broadcasting rules), some things didn't work without modification like our `Einsum` string.

Both difficulties are addressed with a new subclass of `Circuit` called `Symbol`. It has a name and a shape, but no data. Trying to evaluate a `Symbol` will throw an exception. `Symbol` is just a placeholder that needs to be replaced with something else before evaluation happens. It's safer than an `Array` as a placeholder - if we forgot to replace an `Array`, then we would still be able to call `evaluate` using the wrong data.

### UUIDs

A `Symbol` also has a UUID. If you've never heard of a UUID, all you need to know is that that they are random numbers big enough that when you generate one, it's almost certainly true that no other Circuits user has ever generated the same one. Refer to the [Python standard library](https://docs.python.org/3/library/uuid.html) if you want to know more.

The point of the UUID is to distinguish between two `Symbols` with identical names and shapes. Equality on `Symbol` requires that the UUID is equal.

<details>

<summary>Can't we just use the memory address of the Symbol object for uniqueness?</summary>

We're going to be saving Circuits to disk, and they can't be guaranteed to be in the same memory location after reloading the Circuit. The UUID can be saved with the Symbol and thus stay the same between sessions.

</details>

### Symbolic Sizes

To solve the issue with specific sizes being hardcoded, one option is to just use a special value that a size can't have, like -1, so a single input to a transformer would have size (-1, -1) for the sequence and hidden dimensions, for example.

One issue is we lose out on some consistency checking in this way - it's no longer possible to check that the sequence dimension is consistent everywhere in the network, for example.

The galaxy brain solution is to have a bunch of pre-determined special values, where one represents "hidden dimension", one represents "sequence dimension" and so on. If we use them consistently throughout the architecture, we'll be able to enforce consistency. In the current implementation, these are just some primes that are larger than all integers that are used for actual sizes.

This particular implementation is pretty unsafe and subject to change. Its main benefit is that all the regular shape inference code you wrote in Day 1 supports this feature without any modifications (it doesn't care whether the integer represents a real size or a symbolic size).
"""
# %%
(
    HIDDEN,
    SEQ,
    HEADS,
    MLP_PROJ,
    HEAD_SIZE,
    LOG_LIKELIHOOD_CLASSES,
    *__unused,
) = rc.symbolic_sizes()

print("The hidden dimension (dimension of the residual stream", HIDDEN)
print(
    "The number of classes (for our language model, the vocabulary size)",
    LOG_LIKELIHOOD_CLASSES,
)
print("The sequence dimension (equal to the length of the context window)", SEQ)


def sym(shape: rc.Shape, name: str):
    """Create a new symbol."""
    return Symbol.new_with_random_uuid(shape, name)


ln_input = sym((HIDDEN,), "ln.input")
ln_weight = sym((HIDDEN,), "ln.w.scale")
ln_bias = sym((HIDDEN,), "ln.w.bias")


# %%
"""
### Printing of Symbolic Sizes

It would be really confusing to show the primes when printing. Instead, the printing and serialization code knows about the primes and renders them as "0s", "1s", "2s", etc.

"""
# %%
print("This is a placeholder of symbolic shape (HIDDEN,).", ln_input)
print("When serialized or printed, these are rendered specially:")
ln_input.print()

print(
    "\nSymbolic sizes are propagated as normal: ",
)
x = Einsum.from_einsum_string("a,b->ab", ln_weight, ln_bias)
x.print(rc.PrintOptions(shape_only_when_necessary=False))

print(
    "\nSymbolic sizes can also be products of symbolic sizes (this works because they are primes and have a unique factorization)"
)
x.rearrange_str("a b -> (a b)").print(rc.PrintOptions(shape_only_when_necessary=False))


print("\nSymbolic sizes can be products of regular and symbolic sizes")
cat = rc.Concat(ln_input, ln_input, ln_input, axis=0)
cat.print(rc.PrintOptions(shape_only_when_necessary=False))

# %%
"""
## Useful Helper Functions

### reciprocal

We'll need to compute `1/x` for a `Circuit` `x` today. We could use operator overloading to make this syntax work, but then we'd have no way of giving the result a `name` field.

Instead, we have functions like `rc.reciprocal`, which evaluate to the same as `torch.reciprocal` but allow specifying the name. `rc.reciprocal` also has metadata describing algebraic properties of this function. For example, `(1/x)[i] == `1/(x[i])`.
"""
# %%
x_data = t.tensor([0.1, 0.2, 0.3])
x_arr = Array(x_data, name="x")
out = rc.reciprocal(x_arr, name="x_inv")
t.testing.assert_close(out.evaluate(), t.reciprocal(x_data))
t.testing.assert_close(out.evaluate(), 1 / x_data)
out.print()
# %%
"""
### rsqrt

Another one you'll need is `rc.rsqrt`, which stands for reciprocal square root. Compared to computing `sqrt(x)` and then doing a reciprocal, `rsqrt` is one operation so it could be faster and/or more accurate depending on the implementation.
"""
out = rc.rsqrt(Array(x_data), name="x_rsqrt")
print(out.evaluate())
diff = out.evaluate() - ((1 / x_data) ** 0.5)
print(diff.max().item())
# %%
"""
### last_dim_size

Finally, `rc.last_dim_size(x)` evaluates to `torch.full(x.shape[:-1], x.shape[-1])`.

This is kinda weird and subject to change, but the point of using this is that symbolic sizes work properly with it. 

Exercise: discuss with your partner why `mean_broken` below computes the wrong output. This is an important point, so call a TA if you don't feel confident. Write a new `mean` that works correctly.
"""
# %%
def mean_broken(x: Circuit) -> Circuit:
    """Compute the mean of x. This one doesn't work properly.

    x: shape (n,)
    Output: shape ()
    """
    denominator = rc.reciprocal(Scalar(x.shape[-1], name="last_dim_size"))
    return Einsum.from_einsum_string("a,->", x, denominator, name=f"mean({x.name})")


def mean(x: Circuit) -> Circuit:
    """Compute the mean of x.

    x: shape (n,)
    Output: shape ()
    """
    "SOLUTION"
    denominator = rc.reciprocal(rc.last_dim_size(x))
    return Einsum.from_einsum_string("a,->", x, denominator, name=f"mean({x.name})")


mean_broken_circuit = mean_broken(ln_input)
replaced = mean_broken_circuit.update("ln.input", lambda _: Array(x_data))
print("Expected 0.2, got: ", replaced.evaluate().item())


mean_circuit = mean(ln_input)
replaced = mean_circuit.update("ln.input", lambda _: Array(x_data))
print("Expected 0.2, got: ", replaced.evaluate().item())

# %%
"""
### Perils of Automatic Naming

Exercise: explain to your partner why `sum_last_broken` doesn't work properly. Fix the issue.
"""
# %%
def sum_last_broken(x: Circuit) -> Circuit:
    """Compute the sum of x along the last dimension. This one doesn't work properly.

    x: shape (n,)
    Output: shape ()
    """
    return Einsum.from_einsum_string("a->", x)


sum_broken_circuit = sum_last_broken(ln_input)
replaced = sum_broken_circuit.update("ln.input", lambda _: Array(x_data))
print("Expected 0.6, got: ", replaced.evaluate())


# %%
r"""
## LayerNorm in Circuits

A PyTorch implementation of LayerNorm is provided for reference. Additionally, in the `remix_images` folder there are some PDF schematics you can refer to. To view these, either copy them to your local machine or install the [vscode-pdf](https://marketplace.visualstudio.com/items?itemName=tomoki1207.pdf) extension.

Your implementation can be less general than the PyTorch version. In particular, you should assume:

- The parameter `elementwise_affine` is always `True`;
- The shapes of `input`, `weight`, and `bias` are all `(HIDDEN,)` (we'll get to batching later);
- Normalization is computed over the last (and only) dimension; and
- You don't need to worry about resetting the `weight` and `bias` as we'll only be using the Circuit for inference today.

This is sufficient for doing inference in transformers, and specializing in this way makes our job easier.

It's not important today to have an intuition about what LayerNorm is doing other than "it's surprisingly complicated". The article [Re-Examining LayerNorm](https://www.lesswrong.com/posts/jfG6vdJZCwTQmG7kb/re-examining-layernorm) is worth a read at some future time.

Exercise: Implement LayerNorm as a `Circuit`. The names of the internal nodes are up to you; it can be helpful to give unique names to things for debugging purposes but it's also fine to ignore the names.

In addition to `Einsum`, `Add`, and `Scalar` from before, use the below functions. You could write your own lambdas for these, but using these ones from the API ensures that your Circuit will be serializable (more on this later).

Tip: Circuit provides some helpful methods like `Circuit.add(other)` as a shorthand for `Add(self, other)`. The method `Circuit.mul` is also useful shorthand for elementwise multiplication.

<details>
<summary>Hint: help me get started!</summary>

Let's start by computing the mean with an Einsum. The syntax is the same as yesterday: `mean = rc.Einsum.from_einsum_string("h->", input)` or `rc.Einsum.from_fancy_string("hidden -> ", input)`.
</details>

<details>
<summary>I have a weird Scalar 0.000099930048965724 in my Circuit and I can't figure out why!</summary>

This value is equal to `1/HIDDEN` and it probably means you accessed the input shape directly when building the Circuit. You have to use `rc.last_dim_size` to delay evaluation until after real shapes have been substituted for the symbolic shapes. This is a subtle point - ask a TA if this isn't clear!

</details>
"""


# %%
class TorchLayerNorm(nn.Module):
    """A PyTorch implementation of nn.LayerNorm, provided for reference."""

    weight: nn.Parameter
    bias: nn.Parameter

    def __init__(
        self,
        normalized_shape: Union[int, tuple, t.Size],
        eps=1e-5,
        elementwise_affine=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.normalize_dims = tuple(range(-1, -1 - len(self.normalized_shape), -1))
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(t.empty(self.normalized_shape, device=device, dtype=dtype))  # type: ignore
            self.bias = nn.Parameter(t.empty(self.normalized_shape, device=device, dtype=dtype))  # type: ignore
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the weight and bias, if applicable."""
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """x and the output should both have shape (batch, *)."""
        # Chris: MLAB1 repo solution had .detach() here but I think that is wrong
        mean = x.mean(dim=self.normalize_dims, keepdim=True)
        var = x.var(dim=self.normalize_dims, keepdim=True, unbiased=False)

        x = x - mean
        x = x / ((var + self.eps) ** 0.5)
        if self.elementwise_affine:
            x = x * self.weight
            x = x + self.bias
        return x


def layernorm(input: Circuit, weight: Circuit, bias: Circuit, eps=1e-5) -> Circuit:
    """Circuit computing the same thing as TorchLayerNorm, subject to the simplifications detailed in the instructions."""
    "SOLUTION"
    recip_h = rc.reciprocal(rc.last_dim_size(input))
    centered = input.add(
        Einsum.from_einsum_string("h,z,->z", input, Scalar(-1, (1,)), recip_h),
        name="centered",
    )
    var = Einsum.from_einsum_string("h,h,->", centered, centered, recip_h, name="var")
    scale_factor = rc.rsqrt(var.add(Scalar(eps)), name="scale")
    y = centered.mul(scale_factor)
    return y.mul(weight).add(bias)


ln = layernorm(ln_input, ln_weight, ln_bias)
ln.print()

# %%
"""
## Substitutions and rc.Expander

Above, we didn't have to specify the shape of `ln` - it was automatically inferred to be `(HIDDEN,)` via the normal rules you implemented yesterday.

If we try to call `ln.evaluate()`, we'll see an error about it not being explicitly computable. This is telling us that we have `Symbol` instances remaining somewhere in the tree.

We could use an `Updater` like in Day 1 to replace the `Symbol`s with `Array`, but then we'd have to update our `Einsum` again, and we'd need a separate `Updater` for each thing we want to replace.

`rc.Expander` is a class that does all the updates in one shot and also takes care of the batching. Here's how to use it. (If your output doesn't match the reference output, continue reading for troubleshooting tips).

<details>

<summary>Help - It's not working!</summary>

Some steps to start with:

- Check the shapes in the printout and see if they all make sense. Unintended broadcasting is one bad thing that can happen.
- Give your nodes unique names - this helps with reading the printout, and also prevents bugs related to multiple nodes having the same name.

If it's still not working, don't spend much time stuck here - call a TA!

</details>
"""
# %%
t.manual_seed(0)
actual_inputs = [
    ("single dim", t.randn((8,))),
    ("batch and seq dims prepended", t.randn((5, 7, 8))),
]
actual_weight = t.randn((8,))
actual_bias = t.randn((8,))
torch_ln = TorchLayerNorm((8,))
torch_ln.weight.data = actual_weight
torch_ln.bias.data = actual_bias

for name, actual_input in actual_inputs:
    print(f"\nTesting LayerNorm with {name}")
    expected = torch_ln(actual_input)

    expander = rc.Expander(
        ("ln.input", lambda _: Array(actual_input)),
        ("ln.w.scale", lambda _: Array(actual_weight)),
        ("ln.w.bias", lambda _: Array(actual_bias)),
    )
    computable_ln = expander(ln, fancy_validate=True)
    computable_ln.print(rc.PrintOptions(shape_only_when_necessary=False))
    assert computable_ln.is_explicitly_computable
    t.testing.assert_close(computable_ln.evaluate(), expected)
# %%
"""
## Fancy Validation

You can pass `fancy_validate=True` to have the `Expander` spend time doing additional checks - I recommend always doing this unless it's too slow.

Exercise: With `fancy_validate=True`, mess around with different inputs and see what happens. For example:

- Pass a `Matcher` that doesn't match anything
- Pass an `Array` which is an incompatible shape
- Pass two `Matcher`s that match the same node
"""
# %%
"""
## Circuit Diffing

`print(rc.diff_circuits(circuit1, circuit2))` can take two `Circuit`s and render a colored diff between them. The problem of generating a minimal diff between two arbitrary trees is actually quite hard, and at the time of this writing this function isn't particularly smart. 

If your `layernorm` didn't produce the right output, diff it now against the version produced by `remix_d2_utils.working_layernorm` and see if this helps. Play with some of the keyword arguments to `diff_circuits`. This may or may not be actually helpful, and is mainly to emphasize that doing diffs is a thing you should consider when debugging. (Also, note the argument order: the first argument is `new` and the second is `old`, so added (green) nodes will be from the first argument.) If you feel stuck, call a TA.

Exercise: Once your `layernorm` is correct, diff `working_layernorm` against `mystery_layernorm_a` and `mystery_layernorm_b`. Describe the differences in your own words.

<details>
<summary>Solution - Issues with other layernorm implementations</summary>

A forgot to use the value for epsilon passed in, which makes the results very slightly off.

B hardcodes in the scaling factor for the mean (1/hidden_size) instead of computing it on each forward pass. This is a problem because if we start with abstract circuits, this will be (1/large prime) rather than (1/actual hidden size).

</details>
"""
# %%
if "SOLUTION":
    working = remix_d2_utils.working_layernorm(ln_input, ln_weight, ln_bias)
    print(rc.diff_circuits(working, ln, require_name_same=False))

    ma = remix_d2_utils.mystery_layernorm_a(ln_input, ln_weight, ln_bias)
    print(rc.diff_circuits(ma, working, require_name_same=False))

    mb = remix_d2_utils.mystery_layernorm_b(ln_input, ln_weight, ln_bias)
    print(rc.diff_circuits(mb, working, require_name_same=False))

# %%
"""
## Printing and Serialization

If you just do `print(circuit)` or `repr(circuit)`, this renders a small representation which is limited in depth.

`circuit.print()` however, is a completely different beast. This takes (or constructs) a `rc.PrintOptions` instance which has a ton of arguments you can fiddle with to customize your printing. Generally, you'll want to create your own `PrintOptions` instance which is set up the way you like, and call `printer.print(circuit)` or `printer.repr(circuit)` on that.

The most important argument to `PrintOptions` is `bijection`, which causes your printout to be a complete representation of the circuit (with some caveats like custom GeneralFunctions). When `bijection=True`, you can convert the string representation back into an equivalent tree with `rc.Parser`.
"""
# %%
parser = rc.Parser()
printer = rc.PrintOptions(bijection=True)
text = printer.repr(ln)
print(text)
deserialized_ln = parser(text)
assert ln == deserialized_ln
# %%
"""
## Tensor Storage

Serialization requires a way to serialize any tensors inside the circuit, such as model weights. Storing the tensors directly in the text representation would be very inefficient, so instead we hash the contents of the tensor and store the tensor on a disk in binary format at a path derived from their hash.

Importantly, when `bijection=True` this happens automatically, so unlike `print(circuit)` which is cheap and has no side effects, `circuit.print()` might write a lot of data to disk if you have large tensors.

By default, a series of nested folders on your local disk will be used for storage. For example, if a tensor's hash is `4f2fb80a7d72349dd5353e32c67a921368d6ab8eec58590a0c09b62492000be9`, then by default a file is created in `~/tensors_by_hash_cache/4f2f/b80a/` with the remainder of the hash as the filename.

Exercise: Check the hash of `rand1` below using `Array.tensor_hash_base16()`. Print out the circuit `total`, and then check your local filesystem to find the file that was created during printing. Verify that the file is roughly the size you expect, and that the tensor is only stored once (even though there are two references to it).
"""
# %%
t.manual_seed(0)
rand_tensor = t.randn((256,))
rand1 = Array(rand_tensor, name="rand1")
rand2 = Array(rand_tensor, name="rand2")
total = rand1.add(rand2)

if "SOLUTION":
    print(rand1.tensor_hash_base16())
    printer.print(total)
    """Something like: ls -al tensors_by_hash_cache/4f2f/b80a/ should show one file of size 1KB"""

# %%
"""
## Redwood Research File System (RRFS)

Redwood Research File System (RRFS) is just a big network filesystem managed by AWS where we store things that are too big for version control, such as model weights or datasets for experiments. If you type `ls ~/rrfs` in a terminal, you can check if you have this filesystem mounted on your machine.

If you upload your tensors to RRFS using `sync_tensors=True`, other users can automatically download them as needed over the network. Code for this is included in the `rc.Parser`, so the whole process is transparent - if you parse a `Circuit` and don't have the referenced `Tensor`s on their local machine, the library will automatically copy them over the network.

Use caution when `sync_tensors=True` to avoid using up excessive network bandwidth and disk space on tensors you don't actually care about (like the random tensor above).

If you find yourself creating lots of large temporary tensors, you may want to use `bijection=False` to avoid filling up your local disk.

For today, you won't have to upload any tensors and we'll only be downloading the pretrained weights via this mechanism.

## Circuit Simplification

In the PyTorch version, we needed to specify a flag for `elementwise_affine` and then we had to check this in the forward pass to avoid unnecessary computation. In Circuits, if we didn't want to have a bias in our layer norm, we can just substitute a `Scalar` with the appropriate shape and the value zero. While this won't use much memory (remember, `Scalar` is a view into a 1-element `Tensor`), this will still cause PyTorch to perform an unnecessary addition operation. We can detect and optimize this out with a rewrite.

Exercise: implement `add_elim_zeros`, then replace the bias node in `ln` with a zero `Scalar` and call `add_elim_zeros`. Check the diff and verify that the bias node was completely removed.

<details>
<summary>Help, I get a `ExpandBatchingRankTooLowError` error</summary>

Using the `expander` construction from above to replace the bias node won't work because of the dimension-mismatch, if that's what you tried. Instead use
```python
zero_bias = ln.update("ln.w.bias", lambda _: Scalar(0.0, ()))
```
</details>
"""
# %%
def add_elim_zeros(add: Add) -> Add:
    """Rewrite "add" to eliminate any operands that are Scalar(0.0).

    Return "add" unmodified if nothing was done, and a new Add instance otherwise.
    """
    "SOLUTION"
    new_inps: list[Circuit] = []
    did_anything = False
    for c in add.children:
        if isinstance(c, Scalar) and c.value == 0.0:
            did_anything = True
        else:
            new_inps.append(c)
    if did_anything:
        return Add(*new_inps, name=add.name)
    return add


if "SOLUTION":
    zero_bias = ln.update("ln.w.bias", lambda _: Scalar(0.0, ())).cast_add()
    simplified = add_elim_zeros(zero_bias)
    print(rc.diff_circuits(simplified, ln))


# %%
"""
## Automatic Simplification

The function `rc.simp` automatically perfoms the above rewrite as well as around 20 more common ones for you.

Exercise: Run `rc.simp` and diff the result against the output of your simplification. Are they equal? If not, what else did `rc.simp` do?

<details>

<summary>Solution - rc.simp</summary>

After removing the bias node, it noticed that an `Add` with only one remaining argument, meaning the `Add` can also be replaced with its argument.

</details>
"""
# %%
if "SOLUTION":
    p = rc.PrintOptions(bijection=False)
    auto_simplified = rc.simp(zero_bias)
    print(rc.diff_circuits(auto_simplified, simplified))
    assert simplified.children[0] == auto_simplified

# %%
"""
## "Batchable" Symbols

In GPT, our LayerNorm would run independently at each position of the sequence and independently on each batch element. In other words, the input would actually `(BATCH, SEQ, HIDDEN)` shape, not `(HIDDEN,)` as in our input `Symbol`.

In fact, we could keep adding more leading dimensions to the `(HIDDEN,)` shape, and LayerNorm would work properly until we hit some internal library limitation.

When you can safely replace a `Symbol` with an object that has more leading dimensions but is otherwise compatible, we say that symbol is "batchable".

Exercise: Make a new random `Tensor` that has batch and sequence dimensions and pass it into `Expander`. Print the result and verify that the new `Einsum`s look like what you'd expect. How many leading dimensions can you add? What breaks first once you add too many?
"""
# %%
if "SOLUTION":
    sh: tuple[int, ...] = (8,)
    print("Einsum stops at 28 dimensions and Rust panics at 32 dimensions")
    while len(sh) < 32:
        print("Testing with shape: ", len(sh), sh)
        batched_input = t.randn(sh)

        expander = rc.Expander(
            ("ln.input", lambda _: Array(batched_input)),
            ("ln.w.scale", lambda _: Array(actual_weight)),
            ("ln.w.bias", lambda _: Array(actual_bias)),
        )

        batched_computable = expander(ln, fancy_validate=True)
        try:
            batched_computable.evaluate()
        except ValueError as e:
            print(e)
            batched_computable.print()

        sh = (1,) + sh


# %%
"""
## Loading A Reference Model

When writing your own implementation, it's generally easiest to start with a known-good implementation and check that your version has matching behavior along the way.

We'll load a reference copy of the pretrained GPT2-small now. When people say GPT-2, they are usually referring to the 774M parameter version; the small version has only 117M parameters. You don't need to understand the loading code for today. The first time you run this, it will take some time to download over the network (on later runs, it will use the local cache).

By the end of the day, your model will produce the same output as `ref_circuit`!
"""
# %%
from interp.circuit.interop_rust.module_library import load_model_id


def tokenize(prompts: list[str]) -> tuple[t.Tensor, t.Tensor]:
    out = tokenizer(prompts, padding=True, return_tensors="pt")
    input_ids = out["input_ids"]
    # pad_token_id is past the end of the embedding weights
    input_ids[input_ids == tokenizer.pad_token_id] = 0
    return input_ids, out["attention_mask"]


prompts = [
    "Former President of the United States of America, George",
    "Paris is the capital city of",
]
circ_dict, tokenizer, model_info = load_model_id("gelu_12_tied")
ref_input_ids_tensor, attention_mask_tensor = tokenize(prompts)
ref_input_ids = Array(ref_input_ids_tensor)
# TBD lowpri: why do we need to coerce to float to prevent "MiscInputChildrenMultipleDtypesError: Children multiple dtype"?
ref_attention_mask = Array(attention_mask_tensor.float())
bind_module = circ_dict["t.bind_w"]

tok_embeds = rc.GeneralFunction.gen_index(circ_dict["t.w.tok_embeds"], ref_input_ids, index_dim=0)
assert tok_embeds.shape == (2, 10, 768)
expected = circ_dict["t.w.tok_embeds"].cast_array().value[ref_input_ids_tensor]
t.testing.assert_allclose(tok_embeds.evaluate(), expected)

ref_circuit = model_info.bind_to_input(bind_module, tok_embeds, circ_dict["t.w.pos_embeds"], ref_attention_mask)
assert ref_circuit.shape == (2, 10, 50257)


def eval_circuit(circuit: Circuit) -> None:
    logits = circuit.evaluate()
    for i in range(2):
        end_seq = ref_input_ids_tensor[i].nonzero().max().item()
        topk = t.topk(logits[i, end_seq], k=10).indices
        next_tokens = tokenizer.batch_decode(topk.reshape(-1, 1))
        print("Prompt: ", prompts[i])
        print("Top 10 predictions: ", next_tokens)
        if i == 0:
            assert " Washington" in next_tokens
            assert " Bush" in next_tokens
        elif i == 1:
            assert " France" in next_tokens


eval_circuit(ref_circuit)

# %%
"""
## Intro to Modules

Our LayerNorm has quite a few nodes representing low-level operations, and often we just want to view and work with it as "yeah, that's a box containing a standard LayerNorm" and not care about the specifics unless we specifically need to "open the box" and do rewrites that affect the internals.

In Circuits, we'll use the class `rc.ModuleSpec` to represent the concept of a "standard LayerNorm". All the LayerNorms in our transformer will share one instance of this. Inside the `ModuleSpec`, we have a Circuit with Symbol instances as placeholders - this is called the "spec circuit" or just "spec" for short. The output of our function `layernorm` above is already the appropriate thing.

The `ModuleSpec` also contains a `rc.ModuleArgSpec` instance, one for each unique `Symbol` in the spec circuit. This `ModuleArgSpec` stores information needed for batching.

Below we've written `LN_SPEC` in caps to emphasize that this is an immutable constant throughout our transformer. Note that a `ModuleSpec` is not itself a Circuit and thus can't directly appear in a tree.

<details>

<summary>I feel confused about which `Symbol`s should be batchable.</summary>

The only reason to make something not batchable is to get a helpful error if we mess up and try to pass in an input of the wrong shape. In our case, we already specialized our Circuit to only work when normalizing the last dimension, so it wouldn't make sense to have a weight or bias of more than 1D. On the other hand, in the real transformer we will have an input shape of `(batch, seq, hidden)` where the first two dimensions are treated as batch dimensions.

</details>
"""
# %%
def layernorm_spec() -> ModuleSpec:
    """Return a ModuleSpec representing GPT's standard LayerNorm.

    Note that you only ever need to call this once - all Modules can share it.
    """
    sym_input = sym((HIDDEN,), "ln.input")
    sym_weight = sym((HIDDEN,), "ln.w.scale")
    sym_bias = sym((HIDDEN,), "ln.w.bias")

    spec_circuit = layernorm(sym_input, sym_weight, sym_bias)
    return ModuleSpec(
        spec_circuit,
        arg_specs=[
            ModuleArgSpec(sym_input, batchable=True),
            ModuleArgSpec(sym_weight, batchable=False),
            ModuleArgSpec(sym_bias, batchable=False),
        ],
    )


LN_SPEC = layernorm_spec()

# %%
"""
### Module instances

Each time in our computation tree that a standard LayerNorm appears, we will have a `rc.Module` instance representing "a box containing a standard LayerNorm with these specific weights and these specific inputs." 

In other words, if the `ModuleSpec` is like a function with argument names and a body, the `Module` instance is like a function call site where specific values are supplied for the arguments. 

`Module` has a method `substitute()` which will use the same underlying logic as `Expander` to substitute the arguments for the placeholders. Importantly, this happens by name and not using `Matcher`, so it's necessary that within the `ModuleSpec` all distinct `Symbol`s have unique names. The result of substitution is a regular node, `Add` in this case.

When you `evaluate()` a `Module`, it just does `substitute` to replace itself with a regular node, and then calls `evaluate` on the regular node.

To reiterate: each place in the graph where a LayerNorm is used will be a different `Module` instance because the inputs are different, but all the `Module` instances will share the same `ModuleSpec`.

Below we show how to make a `Module` version of LayerNorm. After this, you'll do the MLP and Attention `Module` implementations on your own.

Exercise: Run the cell and check out the notation in the printed `Module`. Note that the "!" means that when we evaluate the `Module`, all occurrences of the name on the right are replaced with the value on the left.

The first letter of "ftt" means that the first flag (batchable) is `False`; the remaining letters mean that the other two flags (which we'll completely ignore today) are `True`.

Exercise: Why do we need to unpack the dictionary literal with `**` below?

<details>

<summary>Solution - Dictionary Unpacking</summary>

It's because the symbol names have a period in them, which makes them not valid Python identifiers. If the name was "ln_input" instead then we could just pass `ln_input=Array(actual_input)` without requiring unpacking.

</details>
"""
# %%
print("LayerNorm module should match earlier version")

some_layernorm = Module(
    spec=LN_SPEC,
    name="ln1.call",
    **{
        "ln.input": Array(actual_input, name="ln1_actual_input"),
        "ln.w.scale": Array(actual_weight, name="ln1_actual_scale"),
        "ln.w.bias": Array(actual_bias, name="ln1_actual_bias"),
    },
)
some_layernorm.print()
t.testing.assert_close(some_layernorm.evaluate(), computable_ln.evaluate())

print("Two LayerNorm modules should have equal specs:")
another_layernorm = Module(
    spec=LN_SPEC,
    name="ln2.call",
    **{
        "ln.input": Array(t.randn((50,)), name="ln2_real_input"),
        "ln.w.scale": Array(t.randn((50,)), name="ln2_real_scale"),
        "ln.w.bias": Array(t.randn((50,)), name="ln2_real_bias"),
    },
)

assert some_layernorm.spec == another_layernorm.spec

# %%
"""
### Children Matcher

The first child of a `Module` is its `ModuleSpec`'s circuit, followed by the concrete argument values in the same order as they are in the `ModuleArgSpec`s. In my case I wrote the input at index 1, the scale at index 2, and the bias at index 3 but yours may be different.

Exercise: Write an IterativeMatcher(...).children_matcher(...) that matches the `ln1_actual_scale` node when called on `some_layernorm`.

"""
# %%
some_layernorms_scale: Array

if "SOLUTION":
    some_layernorms_scale = (
        rc.IterativeMatcher("ln1.call").children_matcher({2}).get_unique(some_layernorm).cast_array()
    )
assert some_layernorms_scale.name == "ln1_actual_scale"

# %%
"""
### ModuleSpec Sanity Checks

It's always a good idea to do sanity checks when you're writing specs. Two things that should be true are:

- Each `Symbol` should appear somewhere in the "body" of the spec. This is like checking a function for arguments that are provided but not used.
- Each of your `Symbol`s needs a unique name. This is like checking a function for two arguments with the same name - if this happened the library wouldn't know which one to substitute into the body.
"""
# %%
LN_SPEC.check_all_inputs_used()
LN_SPEC.check_unique_arg_names()


# %%
"""
### Module Substitution

We will be writing a `Module` for the attention layer, but often we want to "open the box" on this one and interact with the internals. We need a convenient way to get rid of the `Module` and just have regular nodes that we can do regular rewrites on.

In the analogy where a `Module` node represents a function call, we want to "inline" the call and not have a function call at all but instead substitute the body of the function with all the arguments replaced.

Recall that when we `evaluate` a `Module`, it just calls its own `substitute` method to turn into a regular node. We can just call `substitute` ourselves to do the inlining! This new form is extensionally equal.

Exercise: compare the before and after printouts. Precisely what did `substitute` do?
"""
print("Before substitute:")
some_layernorm.print()

print("\nAfter substitute: ")
some_layernorm.substitute().print()

t.testing.assert_allclose(some_layernorm.evaluate(), some_layernorm.substitute().evaluate())

# %%
"""
### Nesting Modules

Naturally, Modules can have other Modules as arguments. This works as you'd expect, and you can perform matching and update operations as usual.

Exercise: make `updated_outer` be the same as `outer_layernorm` except that the inner module `some_layernorm` is substituted away. It should be one line.
"""
# %%
outer_layernorm = Module(
    spec=LN_SPEC,
    name="ln2.call",
    **{
        "ln.input": some_layernorm,
        "ln.w.scale": Array(t.randn((8,)), name="ln2_real_scale"),
        "ln.w.bias": Array(t.randn((8,)), name="ln2_real_bias"),
    },
)

updated_outer: Circuit
if "SOLUTION":
    updated_outer = outer_layernorm.update("ln1.call", lambda c: c.cast_module().substitute())

ln1_call = rc.IterativeMatcher("ln2.call").children_matcher({1}).get_unique(updated_outer)
assert ln1_call.name == "ln1.call"
assert isinstance(ln1_call, Add)

# %%
"""
### Partially Bound Modules

We'll do one last subtlety with `Module` and then get back to building GPT-2. 

Once we have multiple `Module`s inside each other, it's legal to replace a placeholder `Symbol` with another placeholder `Symbol` during substitution. This works as long as all the `Symbol` instances are gone at the end.

Exercise: inspect the following Circuit and explain to your partner what is happening. When we evaluate this Circuit, does it matter which of the two `Module`s gets removed first via `substitute()`? Why or why not?

<details>

<summary>Solution - substitution order</summary>

`x_doubler.substitute()` means replacing `a` and `b` with `x`, and `bind_x.substitute()` means replacing `x` with `x_arr`.

Just like substituting variables in a system of equations, the order of substitution doesn't affect the final evaluation result.

</details>
"""
# %%
a_sym = sym((), name="a")
b_sym = sym((), name="b")
x_sym = sym((), name="x")

ADDER = ModuleSpec(
    Add(a_sym, b_sym, name="a_plus_b"),
    arg_specs=[
        ModuleArgSpec(a_sym, batchable=True),
        ModuleArgSpec(b_sym, batchable=True),
    ],
)

dbl_module = rc.Module(ADDER, name="x_doubler", a=x_sym, b=x_sym)

BIND_SPEC = ModuleSpec(dbl_module, arg_specs=[ModuleArgSpec(x_sym, batchable=True)])

bind_x = rc.Module(BIND_SPEC, x=Array(t.tensor([2.0, 4.0]), name="x_arr"), name="bind_x")

bind_x.print()

# %%
"""
## MLP Module

Exercise: implement a `Module` version of the MLP in GPT-2. Again, ignore the flags besides `batchable`.

The variable `bindings` in the test code uses our internal conventions for naming the `Symbols` - the `m` stands for MLP and `w` stands for "weight". While these are just conventions, it'll be helpful to get used to reading them as you'll be frequently writing `Matcher`s that match these names.

Note that the input to the MLP `Module` should be declared as shape `(HIDDEN,)`. Just like the LayerNorm, this reminds us that the MLP does the same operation at every sequence position. When we use this in the full transformer, since the MLP input is `batchable` then `Expander` will take care of the leading batch and sequence dimensions for us.

For the nonlinearity, use `rc.gelu`. 

<details>

<summary>I'm confused about what symbolic shapes to use.</summary>

After the first projection, the shape in all common transformers as of this writing will be `4 * HIDDEN` - it's fine to just write it this way.

To be extra generic, you could use `MLP_PROJ` instead because there's nothing special or optimal about the number 4.

</details>

"""
# %%


def mlp(
    inp: Circuit,
    linear1_weight: Circuit,
    linear1_bias,
    linear2_weight: Circuit,
    linear2_bias: Circuit,
) -> Circuit:
    """Return a Circuit computing GPT's standard MLP."""
    "SOLUTION"
    up = Einsum.from_fancy_string("hidden, up hidden -> up", inp, linear1_weight).add(linear1_bias)
    act = rc.gelu(up)
    return Einsum.from_fancy_string("up, hidden up -> hidden", act, linear2_weight).add(linear2_bias)


def mlp_spec() -> ModuleSpec:
    """Return a ModuleSpec representing GPT's standard MLP."""
    "SOLUTION"
    sym_inp = sym((HIDDEN,), "m.input")
    sym_w1 = sym((4 * HIDDEN, HIDDEN), "m.w.proj_in")
    sym_b1 = sym((4 * HIDDEN,), "m.w.in_bias")
    sym_w2 = sym((HIDDEN, 4 * HIDDEN), "m.w.proj_out")
    sym_b2 = sym((HIDDEN,), "m.w.out_bias")
    spec_circuit = mlp(sym_inp, sym_w1, sym_b1, sym_w2, sym_b2)
    argspecs = [
        ModuleArgSpec(sym_inp, batchable=True),
        ModuleArgSpec(sym_w1, batchable=False),
        ModuleArgSpec(sym_b1, batchable=False),
        ModuleArgSpec(sym_w2, batchable=False),
        ModuleArgSpec(sym_b2, batchable=False),
    ]
    return ModuleSpec(spec_circuit, argspecs)


print("MLP: Testing with random weights - should match PyTorch")
m_input = t.randn((8,))
m_proj_in = t.randn((32, 8))
m_in_bias = t.randn((32,))
m_proj_out = t.randn((8, 32))
m_out_bias = t.randn((8,))

t.manual_seed(0)
bindings: dict[str, Circuit] = {
    "m.input": Array(m_input),
    "m.w.proj_in": Array(m_proj_in),
    "m.w.in_bias": Array(m_in_bias),
    "m.w.proj_out": Array(m_proj_out),
    "m.w.out_bias": Array(m_out_bias),
}
MLP_SPEC = mlp_spec()
MLP_SPEC.check_all_inputs_used()
MLP_SPEC.check_unique_arg_names()
mlp_call_site = Module(spec=MLP_SPEC, name="m.call", **bindings)

expected = F.linear(F.gelu(F.linear(m_input, m_proj_in, m_in_bias)), m_proj_out, m_out_bias)
t.testing.assert_close(mlp_call_site.evaluate(), expected)


# %%
"""
Now let's work our way through the network and get each section working in isolation.

## Token Embedding

Circuits provides two ways of indexing. For indexing by values that are known at the time you're creating the Circuit, use `rc.Index` (but note that there are limitation on the types of indexes that `rc.Index` supports). Where the index is based on the output of a Circuit, as is the case here, use `rc.GeneralFunction.gen_index`. When calling `rc.GeneralFunction.gen_index`, leave the `batch_x` argument at the default value `False`.

Exercise: complete `token_embed`. It should be one line.
"""
# %%
def token_embed(embed_weight: Circuit, input_ids: Circuit) -> Circuit:
    """
    embed_weight: shape (vocab_size, hidden_size)
    input_ids: shape (batch, seq), dtype int
    out: shape (batch, seq, hidden_size)

    You can assume sizes aren't symbolic.
    """
    "SOLUTION"
    return rc.GeneralFunction.gen_index(embed_weight, input_ids, index_dim=0)


weight_arr = Array(t.randn((10, 6)))
input_ids_arr = Array(t.tensor([1, 3, 5], dtype=t.int64))
actual_embeds = token_embed(weight_arr, input_ids_arr).evaluate()
t.testing.assert_close(actual_embeds[0], weight_arr.value[1])
t.testing.assert_close(actual_embeds[1], weight_arr.value[3])
t.testing.assert_close(actual_embeds[2], weight_arr.value[5])
# %%
"""
## Positional Embedding

Exercise: complete `pos_embed`. 

Tip: `Index` does support indexing by a `Tensor`, but when the `Tensor` is multidimensional, `Index` indexes into each dimension independently which is different than PyTorch and not what you want. Use `rc.GeneralFunction.gen_index` instead.
"""
# %%
def pos_embed(pos_embed_weight: Circuit, input_ids: Circuit) -> Circuit:
    """
    pos_embed_weight: shape (max_position_embeddings, hidden_size)
    input_ids: shape (batch, seq), dtype int
    out: shape (batch, seq, hidden_size)

    You can assume sizes aren't symbolic.
    """
    "SOLUTION"
    B, S = input_ids.shape
    position = t.arange(S).to(input_ids.device)
    position = repeat(position, "n -> b n", b=B)
    return rc.GeneralFunction.gen_index(pos_embed_weight, Array(position), 0)


max_len = 8
vocab_size = 10
hidden = 4
input_ids = t.tensor([[9, 2, 0, 0, 3], [2, 5, 3, 4, 5]])
pos_embed_weight = t.randn((max_len, hidden))
actual = pos_embed(Array(pos_embed_weight), Array(input_ids)).evaluate()
expected = t.stack((pos_embed_weight[:5], pos_embed_weight[:5]))
t.testing.assert_allclose(actual, expected)


# %%
"""
## Computing the Attention Mask

Before we can do the attention module, we need to compute our attention mask.

- The tokenizer returns a dictionary with a key "attention_mask". The value is a `Tensor` with shape `(batch, seq)` which contains 1 if this is a valid token to attend to, and 0 if this is a padding token that shouldn't be attended to. To disambiguate, I'll call this the "padding mask".

- In an autoregressive transformer, we need a tensor of shape `(seq_q, seq_k)` which has `causal_mask[q][k] = 0 if k > q`. For speed, this tensor can be allocated once and shared between all the attention heads in the network. I'll call this the "causal mask".

- When we actually mask the attention scores, we have a tensor of shape `(batch, seq_q, seq_k)` with 0 for valid (this is the opposite of the meaning of 0 previously) and -10000 for invalid. This doesn't have a standard name, and I'll call this the "additive attention mask" because you add it to the attention scores. You'll need to compute this in `apply_attention_mask` soon.

Exercise: implement `causal_mask`.
"""
# %%
def causal_mask(seq_len: int) -> t.Tensor:
    """
    Return shape (seq_q, seq_k) of dtype float32.
    """
    # TBD lowpri: device should be handled
    "SOLUTION"
    mask = torch.arange(seq_len)[:, None] >= torch.arange(seq_len)[None, :]
    return mask.float()


mask = causal_mask(5)
for q in range(5):
    for k in range(5):
        assert mask[q, k].item() == (0.0 if k > q else 1.0)


# %%
"""
## Self-Attention

Exercise: implement the self-attention block. Note that we're omitting the batch dimension - since our input is batchable, `Expander` is able to prepend a batch dimension later without any issue.

<details>

<summary>I'm confused about how to set invalid attention scores to -10000.0!</summary>

While it's possible to write a custom `GeneralFunction` for this, it's tidier to use the basic classes whenever possible to avoid problems with serialization.

You can do this using `Einsum`, `Scalar`, and `Add` by first multiplying invalid positions by 0, then adding -10000.0.

The multiply by zero is not strictly necessary. Even if the attention score was pretty large to begin with, after subtracting -10000 and exponentiating the result should be basically zero (or identically zero depending on the floating point precision). We're just following the reference implementation here.

</details>

"""
# %%
def apply_attention_mask(attn_scores: Circuit, mask: Circuit) -> Circuit:
    """Return attn_scores with invalid scores set to exactly -10000.0.

    attn_scores: shape (head, seq_q, seq_k).
    mask: shape (seq_q, seq_k). Contains 1.0 if this is a valid token to attend to, and 0.0 otherwise.
    """
    "SOLUTION"
    zero_for_invalid = Einsum.from_einsum_string("hqk,qk->hqk", attn_scores, mask)
    additive_attention_mask = Scalar(1.0).sub(mask).mul_scalar(-10000.0)
    return zero_for_invalid.add(additive_attention_mask, name="masked")


print("Testing attention mask with 1 head, 3 positions and 3rd position is padding:")
sample_mask = t.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
sample_scores = t.arange(9, dtype=t.float32).view((1, 3, 3))
mask_circ = apply_attention_mask(Array(sample_scores), Array(sample_mask))
actual = mask_circ.evaluate()
expected = t.tensor([[
    [0.0, -10000.0, -10000.0],
    [3.0, 4.0, -10000.0],
    [-10000.0, -10000.0, -10000.0]]])  # fmt: skip
t.testing.assert_close(actual, expected)

print("Testing by comparing attention mask application with reference circuit")
seq_len = 10
rand_scores = t.randn((1, seq_len, seq_len))
rand_mask = causal_mask(seq_len).float()
your_mask_circ = apply_attention_mask(Array(rand_scores), Array(rand_mask))
your_mask_circ_actual = your_mask_circ.evaluate()

ref_attention_mask_circ_expanded = remix_d2_utils.get_ref_attn_mask_expanded(
    ref_circuit, Array(rand_scores), Array(rand_mask)
)
ref_attention_mask_circ_expected = ref_attention_mask_circ_expanded.evaluate()
t.testing.assert_close(your_mask_circ_actual, ref_attention_mask_circ_expected)

# %%
"""
### Attention Scores

Exercise: implement `attention_scores`. You'll likely want a helper functon to reduce duplication between the `k`, `q`, and `v` projections.

<details>

<summary>I'm confused about the shapes or the calculations here!</summary>

Try working out the test example's first head scores with pencil and paper.

If you feel stuck, look at w2d1_solution.py for some more guidance or call a TA - don't spend a lot of time stuck here.

</details>

"""
# %%
if "SKIP":

    def project(x: Circuit, weight: Circuit, bias: Circuit, name: str) -> Circuit:
        """Apply projection (with a bias so technically affine transformation)."""
        bias_unsqueezed = Rearrange.from_string(bias, "heads headsize -> heads 1 headsize")
        wx = Einsum.from_fancy_string(
            "seq hidden, head headsize hidden -> head seq headsize",
            x,
            weight,
            name="wx",
        )
        wx_plus_b = wx.add(bias_unsqueezed, name=name)
        return wx_plus_b


def attention_scores(x: Circuit, q_weight: Circuit, q_bias: Circuit, k_weight: Circuit, k_bias: Circuit) -> Circuit:
    """
    Return the attention pattern after scaling but before softmax or attention masking.

    pattern[head, q, k] should be the match between a query at sequence position q and a key at sequence position k.

    IMPORTANT: recall the shape of `rc.last_dim_size` is 1 dimension smaller than the output, which is kinda wonky (I hope to get this changed so it always returns a scalar). Ensure you account for this.

    x: shape (seq, hidden_size)
    q_weight: shape (heads, head_size, hidden_size)
    q_bias: shape (heads, head_size)
    k_weight: shape (heads, head_size, hidden_size)
    k_bias: shape (heads, head_size)
    """
    "SOLUTION"
    q = project(x, q_weight, q_bias, "qx")
    k = project(x, k_weight, k_bias, "kx")
    # WILD: this has shape (head,) and the einsum needs to account for this
    scale_factor = rc.rsqrt(rc.last_dim_size(q_bias))
    return Einsum.from_fancy_string(
        "head seq_q headsize, head seq_k headsize, head -> head seq_q seq_k",
        q,
        k,
        scale_factor,
        name="attn_scores",
    )


print("Testing example with 2 heads, hidden dimension of 2, and head size of 1")
print("Note this won't test your scale factor, since head size is 1.")
sample_q_weight = t.tensor(
    [
        [[1.0, 0.0]],
        [[1.0, -1.0]],
    ]
)
sample_q_bias = t.tensor(
    [
        [0.0],
        [0.0],
    ]
)

sample_k_weight = t.tensor(
    [
        [[0.0, 2.0]],
        [[1.0, 2.0]],
    ]
)
sample_k_bias = t.tensor(
    [
        [1.0],
        [2.0],
    ]
)
sample_x = t.tensor(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
    ]
)

attn_score_circ = attention_scores(
    Array(sample_x),
    Array(sample_q_weight),
    Array(sample_q_bias),
    Array(sample_k_weight),
    Array(sample_k_bias),
)
actual = attn_score_circ.evaluate()
print(actual)
expected = t.tensor([
    [[ 1.,  3.,  1., -1.],
     [ 0.,  0.,  0., -0.],
     [-1., -3., -1.,  1.],
     [ 0.,  0.,  0., -0.]],

    [[ 3.,  4.,  1.,  0.],
     [-3., -4., -1., -0.],
     [-3., -4., -1., -0.],
     [ 3.,  4.,  1.,  0.]]])  # fmt: skip
t.testing.assert_close(actual, expected)

print("Testing attention scores with real weights, random embeddings")
batch, seq_len = ref_input_ids.shape
rand_emb = t.rand((seq_len, config.hidden_size))

pretrained_weights = get_weights(circ_dict, bind_module)
attn_weight = pretrained_weights.weights_by_layer[0].attn
your_attn_score_circ = attention_scores(
    Array(rand_emb),
    attn_weight["a.w.q"],
    attn_weight["a.w.q_bias"],
    attn_weight["a.w.k"],
    attn_weight["a.w.k_bias"],
)
actual = your_attn_score_circ.evaluate()

ref_attn_score_circ_expanded = remix_d2_utils.get_ref_attn_score_expanded(ref_circuit, Array(rand_emb), attn_weight)
ref_expected = ref_attn_score_circ_expanded.evaluate()

t.testing.assert_close(actual[0], ref_expected, atol=1e-4, rtol=1e-4)
# %%
"""
### Self-Attention Module

Exercise: implement `attention` by calling your previous functions, and then doing the remaining steps.
"""
# %%
def attention(
    x: Circuit,
    q_weight: Circuit,
    q_bias: Circuit,
    k_weight: Circuit,
    k_bias: Circuit,
    v_weight: Circuit,
    v_bias: Circuit,
    o_weight: Circuit,
    o_bias: Circuit,
    mask: Circuit,
) -> Circuit:
    "SOLUTION"
    attn_scores = attention_scores(x, q_weight, q_bias, k_weight, k_bias)
    masked = apply_attention_mask(attn_scores, mask)
    probs = rc.softmax(masked, name="probs")
    v = project(x, v_weight, v_bias, "vx")
    combined_v = Einsum.from_einsum_string("nqk, nkh -> nqh", probs, v, name="combined_v")
    out = Einsum.from_einsum_string("nsh, ndh -> sd", combined_v, o_weight, name="ox")
    return out.add(o_bias)


def attention_spec() -> ModuleSpec:
    """Return a ModuleSpec representing GPT's standard Attention Layer.

    This is provided since it's just the same boilerplate as before.

    Note that the mask is passed in; the mask will be calculated once per batch and shared between every attention layer.
    """
    sym_inp = sym(
        (
            SEQ,
            HIDDEN,
        ),
        "a.input",
    )
    sym_q_weight = sym((HEADS, HEAD_SIZE, HIDDEN), "a.w.q")
    sym_q_bias = sym((HEADS, HEAD_SIZE), "a.w.q_bias")
    sym_k_weight = sym((HEADS, HEAD_SIZE, HIDDEN), "a.w.k")
    sym_k_bias = sym((HEADS, HEAD_SIZE), "a.w.k_bias")
    sym_v_weight = sym((HEADS, HEAD_SIZE, HIDDEN), "a.w.v")
    sym_v_bias = sym((HEADS, HEAD_SIZE), "a.w.v_bias")
    sym_o_weight = sym((HEADS, HIDDEN, HEAD_SIZE), "a.w.o")
    sym_o_bias = sym((HIDDEN,), "a.w.o_bias")
    mask = sym((SEQ, SEQ), "a.mask")

    spec_circuit = attention(
        sym_inp,
        sym_q_weight,
        sym_q_bias,
        sym_k_weight,
        sym_k_bias,
        sym_v_weight,
        sym_v_bias,
        sym_o_weight,
        sym_o_bias,
        mask,
    )
    argspecs = [
        ModuleArgSpec(sym_inp, batchable=True),
        ModuleArgSpec(sym_q_weight, batchable=False),
        ModuleArgSpec(sym_q_bias, batchable=False),
        ModuleArgSpec(sym_k_weight, batchable=False),
        ModuleArgSpec(sym_k_bias, batchable=False),
        ModuleArgSpec(sym_v_weight, batchable=False),
        ModuleArgSpec(sym_v_bias, batchable=False),
        ModuleArgSpec(sym_o_weight, batchable=False),
        ModuleArgSpec(sym_o_bias, batchable=False),
        ModuleArgSpec(mask, batchable=False),
    ]
    return ModuleSpec(spec_circuit, argspecs)


ATTN_SPEC = attention_spec()
ATTN_SPEC.check_all_inputs_used()
ATTN_SPEC.check_unique_arg_names()


print("Testing attention with real weights, random embeddings")
batch, seq_len = ref_input_ids.shape
rand_emb = t.rand((batch, seq_len, config.hidden_size))

pretrained_weights = get_weights(circ_dict, bind_module)
pretrained_weights.set_attention_mask(Array(causal_mask(seq_len).float()))
attn_weight = pretrained_weights.weights_by_layer[0].attn

your_attn = Module(spec=ATTN_SPEC, name="a.call", **attn_weight, **{"a.input": Array(rand_emb)})
actual = your_attn.evaluate()

ref_attn_expanded = remix_d2_utils.get_ref_attn_expanded(ref_circuit, Array(rand_emb), attn_weight)
expected = ref_attn_expanded.evaluate()
t.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

# %%
"""
## GPT2 Block

Exercise: implement `gpt2_block`. This doesn't need to itself be a `Module`. 
"""
# %%


def gpt2_block(x: Circuit, weights: LayerWeights) -> Circuit:
    """Build Modules for each part and implement the skip connections."""
    "SOLUTION"
    ln1 = Module(spec=LN_SPEC, name="ln1.call", **weights.ln1, **{"ln.input": x})
    attn = Module(spec=ATTN_SPEC, name="a.call", **weights.attn, **{"a.input": ln1})
    x = x.add(attn, name="attn_skip")
    ln2 = Module(
        spec=LN_SPEC,
        name="ln2.call",
        **weights.ln2,
        **{
            "ln.input": x,
        },
    )
    mlp = Module(spec=MLP_SPEC, name="m.call", **weights.mlp, **{"m.input": ln2})
    out = x.add(mlp, name="mlp_skip")
    return out


print("Testing block with real weights, random embeddings")
weights = pretrained_weights.weights_by_layer[0]
block = gpt2_block(Array(rand_emb), weights)
assert block.shape == (batch, seq_len, config.hidden_size)
actual = block.evaluate()
assert actual.shape == (batch, seq_len, config.hidden_size)

ref_block_expanded = remix_d2_utils.get_ref_block_expanded(ref_circuit, Array(rand_emb), weights)
expected = ref_block_expanded.evaluate()
t.testing.assert_allclose(actual, expected)

# %%
"""
## Unembedding

Exercise: implement `unembed` - it should just be one line.
"""
# %%
def unembed(x: Circuit, unembed_weight: Circuit) -> Circuit:
    """
    x: shape (hidden,)
    unembed_weight: (vocab, hidden)

    Out: (vocab,)
    """
    "SOLUTION"
    return Einsum.from_fancy_string("hidden, vocab hidden -> vocab", x, unembed_weight)


def unembed_spec() -> ModuleSpec:
    """Return a ModuleSpec representing the unembedding layer of GPT.

    This is provided since it's just the same boilerplate as before."""
    sym_inp = sym((HIDDEN,), "unembed.input")
    sym_weight = sym((LOG_LIKELIHOOD_CLASSES, HIDDEN), "unembed.weight")
    spec_circuit = unembed(sym_inp, sym_weight)
    argspecs = [
        ModuleArgSpec(sym_inp, batchable=True),
        ModuleArgSpec(sym_weight, batchable=False),
    ]
    return ModuleSpec(spec_circuit, argspecs)


UNEMBED_SPEC = unembed_spec()
UNEMBED_SPEC.check_all_inputs_used()
UNEMBED_SPEC.check_unique_arg_names()

vocab_size = 10
hidden = 4
weight = Array(t.randn((vocab_size, hidden)))
unembed_x = Array(weight.value[2])
unembed_module = Module(
    spec=UNEMBED_SPEC,
    name="unembed.call",
    **{"unembed.weight": weight},
    **{"unembed.input": unembed_x},
)
assert unembed_module.evaluate().shape == (vocab_size,)
# TBD: should test this also


# %%
"""
## Full GPT-2

Exercise: implement `gpt2` to wire together the pieces. Note that in GPT-2, the unembedding weight is "tied" - it's the same as the embedding weight.
"""
# %%


def gpt2(input_ids: Circuit, weights: GPT2Weights) -> Circuit:
    """
    x: shape (batch, seq, hidden): sum of token and positional embeddings

    Return logits of shape (batch, seq, vocab_size)
    """
    "SOLUTION"
    x = token_embed(weights.tok_embeds, input_ids)
    x = x.add(pos_embed(weights.pos_embeds, input_ids))
    for w in weights.weights_by_layer:
        x = gpt2_block(x, w)
    x = Module(
        spec=LN_SPEC,
        name="final_ln",
        **{"ln.w.scale": weights.final_ln_scale, "ln.w.bias": weights.final_ln_bias},
        **{"ln.input": x},
    )
    x = Module(
        spec=UNEMBED_SPEC,
        name="unembed",
        **{"unembed.weight": weights.tok_embeds},
        **{"unembed.input": x},
    )
    return x


# %%
"""
## Moment of Truth

If you've done everything right, your GPT-2 will make the same predictions as the reference model from before.

Mine did not work on the first try, so don't be discouraged if yours is broken. Debugging is an Authentic ML Experience - try to devise a systematic plan for narrowing down where the issue could be and remember to make liberal use of assertions and write additional test cases.
"""
# %%
print("Real GPT-2 said: ")
eval_circuit(ref_circuit)

print("\n\nYour GPT-2 said: ")
your_circuit = gpt2(ref_input_ids, pretrained_weights)
eval_circuit(your_circuit)

# %%
"""
Congratulations on completing the main content for the day :)

## Bonus

- Practice printing parts of the model. Play with the `PrintOptions` to find a balance where you print enough to see what's going on, but don't feel overwhelmed by too much detail.
"""

# %%
# TBD low: maybe want basic material on ModulePusher (needed later), nested modules.
