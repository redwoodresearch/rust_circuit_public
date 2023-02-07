# %%
# This document serves multiple purposes:
#   - A reproduction of the experiments in the Paren Balancer causal scrubbing post
#   - A causal scrubbing demo
#   - A set of exercises for remix (hense the `if "SOLUTION"` blocks, and instructions)

# %% [markdown]
"""
# Day 4b: Paren Balancer Causal Scrubbing

To start, please read this [less wrong post](https://www.lesswrong.com/s/h95ayYYwMebGEYN5y/p/kjudfaQazMmC74SbF).

We will replicate the experiments today.

<!-- toc -->
"""
import uuid
from typing import Optional, Tuple

import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy_with_logits

import interp.tools.optional as op
import rust_circuit as rc
from interp.circuit.testing.notebook import NotebookInTesting
from interp.tools.indexer import TORCH_INDEXER as I
from rust_circuit.algebric_rewrite import residual_rewrite, split_to_concat
from rust_circuit.causal_scrubbing.experiment import (
    Experiment,
    ExperimentCheck,
    ExperimentEvalSettings,
    ScrubbedExperiment,
)
from rust_circuit.causal_scrubbing.hypothesis import (
    Correspondence,
    ExactSampler,
    FuncSampler,
    InterpNode,
    UncondSampler,
    chain_excluding,
    corr_root_matcher,
)
from rust_circuit.model_rewrites import To, configure_transformer
from rust_circuit.module_library import load_model_id
from rust_circuit.ui.ui import circuit_graph_ui, ui_default_hidden_matcher

from . import test_paren_balancer_exercises as tests
from .setup import ParenDataset, ParenTokenizer, get_h00_open_vector

ui_hidden_matcher = ui_default_hidden_matcher
# %%
# define model
MAIN = __name__ == "__main__"

SEQ_LEN = 42
NUM_EXAMPLES = 4000
MODEL_ID = "jun9_paren_balancer"

PRINT_CIRCUITS = True
ACTUALLY_RUN = True
SLOW_EXPERIMENTS = True
DEFAULT_CHECKS: ExperimentCheck = True
EVAL_DEVICE = "cuda:0"
# If you have less memory, you will want to reduce this and also add a batch_size
MAX_MEMORY = 30_000_000_000
BATCH_SIZE = None

if NotebookInTesting.currently_in_notebook_test:
    EVAL_DEVICE = "cpu"
    SLOW_EXPERIMENTS = False
    MAIN = True  # a bit of a lie but we want to actually run things

# %% [markdown]
"""
## Setup 
No exercises here! It may be helpful to read over the code, however.

### Circuit loading
If any of these operations confuse you, try printing out the circuit before and after!

Step 1: Initial loading
"""

circ_dict, _, model_info = load_model_id(MODEL_ID)
circuit = circ_dict["t.bind_w"]

#%% [markdown]
"""
Step 2: We bind the model to an input by attaching a placeholder symbol input named "tokens" to the model. We then specify the attention mask that prevents attending to padding depends on this tokens array.

We use one hot tokens as this makes defining the attention mask a simple indexing operation.

The symbol has a random fixed uuid (fixed as this gives consistency when comparing in tests).
"""
toks_uuid = uuid.UUID("ce34280e-169f-40bd-b78e-8adeb4274aba")
tokens_arr = rc.Symbol((SEQ_LEN, ParenTokenizer.vocab_size), uuid=toks_uuid, name="tokens")
tok_embeds = rc.Einsum.from_fancy_string(
    "seqlen vocab_size, vocab_size hidden -> seqlen hidden", tokens_arr, circ_dict["t.w.tok_embeds"], name="tok_embeds"
)
attn_mask = rc.Add.minus(
    rc.Scalar(1),
    rc.Index(tokens_arr, I[:, ParenTokenizer.PAD_TOKEN]),
    name="pos_mask",
)
circuit = model_info.bind_to_input(circuit, tok_embeds, circ_dict["t.w.pos_embeds"], attn_mask)

# [markdown]
"""
Step 3: rewrites the circuit into a more conveninet structure to work with using `configure_transformer`.
This flattens out the residual stream (as opposed to the nested layer structure originally) and, pushes down the weight bindings, and separates out each attention layer into a sum of heads.
"""

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

# [markdown]
"""
Some additional misc rewrites.

We substitute the inputs to be duplicated everywhere they apperar in the model instead of being in one outer module bind.

We also index as we only care about the classification at position 0, and use `rc.conform_all_modules` to replace any remaining symbolic shapes with their numeric values.
"""

circuit = circuit.cast_module().substitute()
circuit = rc.Index(circuit, I[0]).rename("logits_pos0")
circuit = rc.conform_all_modules(circuit)

# %% [markdown]
"""
Finally, some custom renames that make the circuit more intuitive.
"""
circuit = circuit.update("t.call", lambda c: c.rename("logits"))
circuit = circuit.update("t.call", lambda c: c.rename("logits_with_bias"))
circuit = circuit.update(rc.Regex(r"[am]\d(.h\d)?$"), lambda c: c.rename(c.name + ".inner"))
circuit = circuit.update("t.inp_tok_pos", lambda c: c.rename("embeds"))
circuit = circuit.update("t.a.mask", lambda c: c.rename("padding_mask"))

for l in range(model_info.params.num_layers):
    circuit = circuit.update(f"b{l}.m", lambda c: c.rename(f"m{l}"))
    circuit = circuit.update(f"b{l}.a.h0", lambda c: c.rename(f"a{l}.h0"))
    circuit = circuit.update(f"b{l}.a.h1", lambda c: c.rename(f"a{l}.h1"))

    next = "final" if l == model_info.params.num_layers - 1 else f"a{l+1}"
    circuit = circuit.update(f"b{l}", lambda c: c.rename(f"{next}.input"))

printer = rc.PrintHtmlOptions(
    shape_only_when_necessary=False,
    traversal=rc.restrict(
        rc.IterativeMatcher("embeds", "padding_mask", "final.norm", rc.Regex(r"^[am]\d(.h\d)?$")), term_if_matches=True
    ),
)
# %%

if PRINT_CIRCUITS:
    printer.print(circuit)
    circuit_graph_ui(circuit, default_hidden=ui_hidden_matcher.get(circuit))


# %% [markdown]
"""
## dataset and experiment code
We have a custom dataset class that precomputes some features of paren sequences, and handles pretty printing / etc.
"""

ds = ParenDataset.load()


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
    return loss, correct


def paren_experiment(
    circuit: rc.Circuit,
    dataset: ParenDataset,
    corr: Correspondence,
    checks: ExperimentCheck = DEFAULT_CHECKS,
    random_seed=1,
    actually_run=ACTUALLY_RUN,
    num_examples=NUM_EXAMPLES,
    batch_size=BATCH_SIZE,
    **kwargs,
) -> Tuple[ScrubbedExperiment, Optional[float]]:
    ex = Experiment(
        circuit,
        dataset,
        corr,
        random_seed=random_seed,
        check=checks,
        **kwargs,
    )
    scrubbed = ex.scrub(num_examples, treeify=actually_run)
    overall_loss: Optional[float] = None
    if actually_run:
        logits = scrubbed.evaluate(
            ExperimentEvalSettings(
                optim_settings=rc.OptimizationSettings(
                    max_memory=MAX_MEMORY,
                ),
                device_dtype=rc.TorchDeviceDtypeOp(device=EVAL_DEVICE),
                optimize=True,
                batch_size=batch_size,
            ),
        )
        ref_ds = ParenDataset.unwrap(scrubbed.ref_ds)
        labels = ref_ds.is_balanced.value

        def loss_str(mask):
            loss, correct = bce_with_logits_loss(logits[mask], labels[mask])
            loss = loss.cpu()
            std_err = loss.std() / len(loss) ** 0.5

            return f"{loss.mean():.3f}  SE={std_err:.3f}  acc={correct.float().mean():.1%} "

        print(f"  overall:               {loss_str(slice(None))}")
        print(f"    on bal:              {loss_str(labels.to(dtype=torch.bool))}")
        print(f"    on unbal:            {loss_str(~labels.to(dtype=torch.bool))}")
        print(f"    on count failures:   {loss_str(~ref_ds.count_test.to(dtype=torch.bool))}")  # type: ignore
        print(f"    on horizon failures: {loss_str(~ref_ds.horizon_test.to(dtype=torch.bool))}")  # type: ignore
        overall_loss = bce_with_logits_loss(logits, labels)[0].mean().item()

    return scrubbed, overall_loss


def check_loss(loss: Optional[float], target: float, std_err: float):
    # we usually set tol to be 4 * SE
    assert loss is not None
    err = abs(loss - target)
    assert err < 4 * std_err, f"err too large! loss ({loss:.2f}) != target ({loss:.2f}) ± 4*SE ({std_err:.2f})"
    if err > 2 * std_err:
        raise Warning("Err is kinda large! loss ({loss:.2f}) != target ({loss:.2f}) ± 2*SE ({std_err:.2f})")


# %% [markdown]
"""
Helpful tidbit on tests:
Most of the tests in this file raise assertion errors that contain extra data, for instance the objects from the comparison that failed. It can be convenient to catch this data to debug. For instance:

```
def check_eq(a, b):
    assert a == b, ("not equal!", a, b)

try:
    check_eq(0, 1)
except AssertionError as e:
    a, b = e.args[0][1], e.args[0][2]
    print(a, b)
```
"""


# %% [markdown]
"""
## Experiment 0
To start with let's measure two baselines:
  - running the model normally
  - interchanging the logits randomly

Make causal scrubbing experiments that impliment both of these. In each case there should be a single interp node named "logits".

The tests do explictly check that the interp nodes in the correspondence are named correctly, in order to facilitate more helpful feedback.
"""

if "SOLUTION":
    corr0a = Correspondence()
    corr0a.add(InterpNode(cond_sampler=ExactSampler(), name="logits"), corr_root_matcher)
else:
    corr0a = Correspondence()
if MAIN:
    tests.t_ex0a_corr(corr0a)
    print("\nEx0a: Exact sampling")
    ex0a, loss0a = paren_experiment(circuit, ds, corr0a)
    check_loss(loss0a, 0, 0.01)

if "SOLUTION":
    corr0b = Correspondence()
    corr0b.add(InterpNode(cond_sampler=UncondSampler(), name="logits"), corr_root_matcher)
else:
    corr0b = Correspondence()
if MAIN:
    tests.t_ex0b_corr(corr0b)
    print("\nEx0b: Interchanging logits")
    ex0b, loss0b = paren_experiment(circuit, ds, corr0b)
    check_loss(loss0b, 4.30, 0.12)

# %% [markdown]
"""
## Experiment 1
Now, let's construct a basic experiment to determine the role that different heads play.
We'll start by testing the following claimed hypothesis:
 - Heads 1.0 and 2.0 compute the count test, and check that there are equal numbers of open and close parentheses
 - Head 2.1 computes the horizon test.

### Matchers
"""
if "SOLUTION":
    # There are several ways to define this matcher, eg:
    # m_10 = rc.IterativeMatcher("final.input").chain(rc.restrict("a1.h0", end_depth=2))
    # or defining all_components = {"a0.h0", "a0.h1", "m0", ..., "m2"} and then
    # m_10 = chain_excluding(corr_root_matcher, "a1.h0", all_components - "a1.h0")
    # It's just down to personal preference / what is clearer in the circumstance.
    m_10 = chain_excluding(corr_root_matcher, "a1.h0", {"m2", "m1", "a2.h0", "a2.h1"})
    m_20 = chain_excluding(corr_root_matcher, "a2.h0", "m2")
    m_21 = chain_excluding(corr_root_matcher, "a2.h1", "m2")
else:
    """
    Define the following matchers. You only want to match _direct_ paths, that is paths through the
    residual stream and not through direct paths. This can be accomplished by calling `rc.restrict` or
    the chain_excluding utilty included in causal scrubbing code.
    """
    m_10 = rc.IterativeMatcher()
    m_20 = rc.IterativeMatcher()
    m_21 = rc.IterativeMatcher()

if MAIN:
    tests.t_m_10(m_10)
    tests.t_m_20(m_20)
    tests.t_m_21(m_21)

# %%  [markdown]
"""
### Cond Samplers 
"""


def passes_count(d: ParenDataset) -> torch.Tensor:
    """
    Returns a bool tensor of shape [len(d)]
    Result is true when the corresponding datum has equal numbers of open and close parens
    """
    # not used in solution, as implimented in ParenDataset
    raise NotImplementedError


def passes_horizon(d: ParenDataset) -> torch.Tensor:
    """
    Returns a bool tensor of shape [len(d)]
    Result is true when the corresponding datum passes the right to left horizon test as described in the [writeup](https://www.lesswrong.com/s/h95ayYYwMebGEYN5y/p/kjudfaQazMmC74SbF#Algorithm).
    """
    # not used in solution, as implimented in ParenDataset
    raise NotImplementedError


if "SOLUTION":
    count_cond = FuncSampler(lambda d: ParenDataset.unwrap(d).count_test)
    horizon_cond = FuncSampler(lambda d: ParenDataset.unwrap(d).horizon_test)
else:
    """
    Define the following cond samplers using FuncSamplers.

    Write your own functions for this based on `ParenDataset.tokens_flat`.

    (Yes, there are predefined properties on ParenDataset. You can use them if you are short on time and want to skip this test. They also use caching to make things faster, so if speed gets to be annoyingly slow for later experiments you could switch over).
    """
    count_cond = FuncSampler(lambda d: passes_count(ParenDataset.unwrap(d)))
    horizon_cond = FuncSampler(lambda d: passes_horizon(ParenDataset.unwrap(d)))

if MAIN:
    tests.t_count_cond(count_cond)
    tests.t_horizon_cond(horizon_cond)

# %% [markdown]
"""
This first correspondence should have 4 nodes:
 - The root node, named "logits", with an ExactSampler (any sampler that agrees on the labels will be equivilant,
 but an exact sampler is somewhat more computationally efficient).
 - Three nodes for the three heads of interest, named "10", "20", and "21". The first two should use the cond sampler provided (`count_cond` for now), the third should use the `horizon_cond` sampler.

The exact interp node names are checked by the test, which allows it to give more meaningful feedback. 
"""


def make_ex1_corr(cs_for_h10_and_h20) -> Correspondence:
    """SOLUTION"""
    corr = Correspondence()
    i_logits = InterpNode(name="logits", cond_sampler=ExactSampler())
    corr.add(i_logits, corr_root_matcher)
    corr.add(i_logits.make_descendant(cs_for_h10_and_h20, "10"), m_10)
    corr.add(i_logits.make_descendant(cs_for_h10_and_h20, "20"), m_20)
    corr.add(i_logits.make_descendant(horizon_cond, "21"), m_21)
    return corr


# %%
if MAIN:
    print("\nEx1a: Just the count cond")
    tests.t_ex1_corr(make_ex1_corr, count_cond)
    ex1a, loss1a = paren_experiment(circuit, ds, make_ex1_corr(count_cond))
    check_loss(loss1a, 0.52, 0.04)

    if PRINT_CIRCUITS:
        ex1a.print()


# %% [markdown]
"""
As discussed in the writeup, we can more accurately capture the equivilance classes of 1.0 and 2.0's output by including if the first parenthesis is open or closed.

This is a natural feature for these heads to use: a sequence will always be unbalanced if it starts with a close parenthesis, and as these heads depend strongly on the residual stream at position 1 anyhow (as we will show in experiement 2) the information is readily accessible.

(reminder: position 0 is always the [START] token, so position 1 is the first parentheses. All sequences in our dataset have >= 2 parentheses in them, so you can assume position 1 is either an open or close paren.)

Define some new cond samplers that incorporate this feature.
"""


def passes_starts_open(d: ParenDataset) -> torch.Tensor:
    """
    Returns a bool tensor of shape [len(d)].
    Result is true when the corresponding datum starts with '('.
    """
    raise NotImplementedError


def passes_count_open(d: ParenDataset) -> torch.Tensor:
    """
    Returns a bool tensor of shape [len(d)].
    Result is true when the corresponding datum starts with '(' and there are equal numbers of open and close parens in the entire sequence.
    """
    raise NotImplementedError


if "SOLUTION":
    start_open_cond = FuncSampler(lambda d: ParenDataset.unwrap(d).starts_with_open)
    count_open_cond = FuncSampler(lambda d: ParenDataset.unwrap(d).count_test & ParenDataset.unwrap(d).starts_with_open)
else:
    """
    Two more samplers! the first that checks that the first paren is an open parentheses,
    the next tests the input passes count test AND the first paren is open.

    We don't use the pure start_open test yet, but we will soon and it's nice to define it here.
    """
    start_open_cond = FuncSampler(lambda d: passes_starts_open(ParenDataset.unwrap(d)))
    count_open_cond = FuncSampler(lambda d: passes_count_open(ParenDataset.unwrap(d)))

if MAIN:
    tests.t_start_open_cond(start_open_cond)
    tests.t_count_open_cond(count_open_cond)


# %%
if MAIN:
    print("\nEx1b: Without a0")
    tests.t_ex1_corr(make_ex1_corr, count_open_cond)
    ex1b, loss1b = paren_experiment(circuit, ds, make_ex1_corr(count_open_cond))
    check_loss(loss1b, 0.30, 0.04)

    if PRINT_CIRCUITS:
        ex1b.print()

# [markdown]
"""
Bonus: Can you improve on the loss by specifying other direct paths, or choosing better features to ensure
agreement along?
"""
# %%
if "SOLUTION":
    if MAIN:
        print("\nEx1 bonus 1: With a0")
        corr = make_ex1_corr(count_open_cond)

        i_00 = corr.get_by_name("logits").make_descendant(count_open_cond, "00")
        m_00 = chain_excluding(corr_root_matcher, "a0.h0", {"m2", "m1", "m0", "a2.h0", "a2.h1", "a1.h0", "a1.h1"})

        corr.add(i_00, m_00)
        ex1_bonus1 = paren_experiment(circuit, ds, corr)


# %%
# We only test up until this point in CircleCI so that it's fast
if "SKIP":
    NotebookInTesting.exit_if_in_testing()

# %% [markdown]
"""
## Experiment 2! Diving into heads 1.0 and 2.0
We are going split up experiment 2 into four parts:
 - Part 1: 1.0 and 2.0 only depend on their input at position 1
 - Part 2 (ex2a in writeup): 1.0 and 2.0 only depend on:
    - the output of 0.0 (which computes $p$, the proportion of open parentheses) and
    - the embeds (which encode if the first paren is open)  
 - Part 3: Projecting the output of 0.0 onto a single direction
 - Part 4 (ex2b in writeup): Estimate the output of 0.0 with a function $\phi(p)$

## Part 1: Splitting up the input to 1.0 and 2.0 by sequence position
One of the claims we'd like to test is that only the input at position 1 (the first paren position) matters for both heads 1.0 and 2.0.
Currently, however, there is no node of our circuit corresponding to "the input at position 1". Let's change that!

Write a `separate_pos1` function that will transform a circuit `node` into:
```
'node_concat' Concat
  'node.pos_0' Index [0:1, :]
    'node'
  'node.pos_1' Index [1:2, :]
    'node'
  'node.pos_2_41' Index [2:42, :]
    'node'
```
This can be acomplished by calling `split_to_concat()` (from algebraic_rewrites.py) and `.rename`-ing the result.

Then split the input node, but only along paths that are reached through head 2.0 and 1.0. (We don't want to split the input to 2.1 in particular, as we'll split that differently later.)

If you are struggling to get the exact correct circuit, you are free to import it from the solution file and print it out. You can also try `print(rc.diff_circuits(your_circuit, our_circuit))` though

Yes, the tests are very strict about naming things exactly correctly.
This is partially because it is convenient for tests, but also because names are really important!
Good names reduce confusion about what that random node of the circuit actually means.
Mis-naming nodes is also a frequent cause of bugs, e.g. a matcher that traverses a path that it wasn't supposted to.
"""

# %%
def separate_pos1(c: rc.Circuit) -> rc.Circuit:
    "SOLUTION"
    return split_to_concat(
        c, 0, [0, 1, torch.arange(2, 42)], partitioning_idx_names=["pos0", "pos1", "pos2_42"], use_dot=True
    ).rename(f"{c.name}_concat")


ex2_part1_circuit = circuit
if "SOLUTION":
    ex2_part1_circuit = ex2_part1_circuit.update(rc.IterativeMatcher("a2.h0").chain("a2.input"), separate_pos1)
    ex2_part1_circuit = ex2_part1_circuit.update(rc.IterativeMatcher("a1.h0").chain("a1.input"), separate_pos1)

if MAIN and PRINT_CIRCUITS:
    subcirc = ex2_part1_circuit.get_unique(rc.IterativeMatcher("a2.h0").chain("a2.input_concat"))
    printer.print(subcirc)
    circuit_graph_ui(subcirc, default_hidden=ui_hidden_matcher.get(subcirc))


if MAIN:
    tests.t_ex2_part1_circuit(ex2_part1_circuit)

# %% [markdown]
"""
Now we can test the claim that both 1.0 and 2.0 only cares about positon 1!

We'll need new matchers, which just matches the pos_1 input.
"""

if "SOLUTION":
    m_10_p1 = m_10.chain("a1.input.at_pos1")
    m_20_p1 = m_20.chain("a2.input.at_pos1")
else:
    m_10_p1 = rc.IterativeMatcher()
    m_20_p1 = rc.IterativeMatcher()

if MAIN:
    tests.t_m_10_p1(m_10_p1)
    tests.t_m_20_p1(m_20_p1)

# %% [markdown]
"""
Then create a correspondence that extends the one returned by `make_ex1_corr(count_open_cond)` so that both 1.0 and 2.0 only use information from position 1. `Correspondence.get_by_name` is useful here.

Have your new nodes be named "10_p1" and "20_p1".
"""


def make_ex2_part1_corr() -> Correspondence:
    "SOLUTION"
    corr = make_ex1_corr(count_open_cond)
    i_10_p1 = corr.get_by_name("10").make_descendant(name="10_p1", cond_sampler=count_open_cond)
    corr.add(i_10_p1, m_10_p1)
    i_20_p1 = corr.get_by_name("20").make_descendant(name="20_p1", cond_sampler=count_open_cond)
    corr.add(i_20_p1, m_20_p1)
    return corr


if MAIN:
    tests.t_make_ex2_part1_corr(make_ex2_part1_corr())
    print("\nEx 2 part 1: 1.0/2.0 depend on position 1 input")
    ex2_p1, loss2_p1 = paren_experiment(ex2_part1_circuit, ds, make_ex2_part1_corr())


# %% [markdown]
"""
### Part 2
We now construct experiment 2a from the writeup. We will be strict about where 1.0 and 2.0 learn the features they depend on. We claim that the 'count test' is determined by head 0.0 checking the exact proportion of open parens in the sequence and outputting this into the residual stream at position 1.

We thus need to also split up the output of attention head 0.0, so we can specify it only cares about the output of this head at position 1. Again, let's only split it for the branch of the circuit we are working with: copies of 0.0 that are upstream of either `m_10_p1` or `m_20_p1`. 
"""

ex2_part2_circuit = ex2_part1_circuit
if "SOLUTION":
    ex2_part2_circuit = ex2_part2_circuit.update((m_10_p1 | m_20_p1).chain("a0.h0"), separate_pos1)

if MAIN and PRINT_CIRCUITS:
    printer.print(ex2_part2_circuit.get_unique(m_10_p1))

if MAIN:
    tests.t_ex2_part2_circuit(ex2_part2_circuit)

# %% [markdown]
"""
First, make a new cond sampler that samples an input that agrees on what is called $p_1^($ in the writeup. This can be done with a FuncSampler based on a function with the following equivalence classes:
 - one class for _all_ inputs that start with a close parenthesis
 - one class for every value of $p$ (proportion of open parentheses in the entire sequence)

Note the actual values returned aren't important, just the equivialance clases.
"""


def p1_if_starts_open(d: ParenDataset):
    """Returns a tensor of size [len(ds)]. The value represents p_1 if the sequence starts open, and is constant otherwise"""
    "SOLUTION"
    return torch.where(
        d.starts_with_open,
        d.p_open_after[:, 1],
        torch.tensor(-1.0, dtype=torch.float32),
    )


p1_open_cond = FuncSampler(lambda d: p1_if_starts_open(ParenDataset.unwrap(d)))

# %% [markdown]
"""And some matchers"""

if "SOLUTION":
    m_10_p1_h00 = m_10_p1.chain("a0.h0.at_pos1")
    m_20_p1_h00 = m_20_p1.chain("a0.h0.at_pos1")
else:
    m_10_p1_h00 = rc.IterativeMatcher()
    m_20_p1_h00 = rc.IterativeMatcher()

# %% [markdown]
"""
Now make the correspondence!

You should add 4 nodes to the correspondence from part 1:
 - "10_p1_00"
 - "20_p1_00"
 - "10_p1_emb"
 - "20_p1_emb"

"""


def make_ex2_part2_corr() -> Correspondence:
    "SOLUTION"
    corr = make_ex2_part1_corr()

    i_10_p1 = corr.get_by_name("10_p1")
    i_20_p1 = corr.get_by_name("20_p1")

    # indirect paths to 0.0
    i_10_p1_00 = i_10_p1.make_descendant(name="10_p1_00", cond_sampler=p1_open_cond)
    i_20_p1_00 = i_20_p1.make_descendant(name="20_p1_00", cond_sampler=p1_open_cond)

    corr.add(i_10_p1_00, m_10_p1_h00)
    corr.add(i_20_p1_00, m_20_p1_h00)

    i_10_p1_emb = i_10_p1.make_descendant(name="10_p1_emb", cond_sampler=start_open_cond)
    i_20_p1_emb = i_20_p1.make_descendant(name="20_p1_emb", cond_sampler=start_open_cond)

    corr.add(i_10_p1_emb, chain_excluding(m_10_p1, "embeds", "a0.h0"))
    corr.add(i_20_p1_emb, chain_excluding(m_20_p1, "embeds", "a0.h0"))

    return corr


if MAIN:
    tests.t_ex2_part2_corr(make_ex2_part2_corr())
    print("\nEx 2 part 2 (2a in writeup): 1.0/2.0 depend on position 0.0 and emb")
    ex2a, loss2a = paren_experiment(ex2_part2_circuit, ds, make_ex2_part2_corr())
    check_loss(loss2a, 0.55, 0.04)


# %% [markdown]
"""
### Part 3: Projecting 0.0 onto a single direction
#### Circuit rewrite
Another claim we would like to test is that only the output of 0.0 written in a particular direction is important.

To do this we will rewrite the output of 0.0 as the sum of two terms: the (projection)[https://en.wikipedia.org/wiki/Vector_projection] and rejection (aka the perpendicular component) along this direction.
"""

# %%

h00_open_vector = get_h00_open_vector(MODEL_ID)


def project_into_direction(c: rc.Circuit, v: torch.Tensor = h00_open_vector) -> rc.Circuit:
    """
    Return a circuit that computes `c`: [seq_len, 56] projected the direction of vector `v`: [56].
    Call the resulting circuit `{c.name}_projected`.
    """
    "SOLUTION"
    v_dir_array = rc.Array(v / torch.linalg.norm(v), "dir")
    return rc.Einsum.from_einsum_string(
        "s e, e, f -> s f", c.rename(c.name + "_orig"), v_dir_array, v_dir_array, name=f"{c.name}_projected"
    )


if MAIN:
    tests.t_project_into_direction(project_into_direction)


def get_ex2_part3_circuit(c: rc.Circuit, project_fn=project_into_direction):
    """
    Uses `residual_rewrite` to write head 0.0 at position 1 (when reached by either `m_10_p1_h00` or `m_20_p1_h00`), as a sum of the projection and the rejection along h00_open_vector. The head retains it's same name, with children named `{head.name}_projected` and `{head.name}_projected_residual`.
    """
    "SOLUTION"
    split = lambda h00: residual_rewrite(h00, project_fn(h00), "projected")[0]
    return c.update((m_10_p1_h00 | m_20_p1_h00).chain("a0.h0"), split)


ex2_part3_circuit = get_ex2_part3_circuit(ex2_part2_circuit)

if MAIN and PRINT_CIRCUITS:
    proj_printer = printer.evolve(
        traversal=rc.new_traversal(term_early_at={"a0.h0.at_pos0", "a0.h0.at_pos2_42", "a0.h0_orig"})
    )
    subcirc = ex2_part3_circuit.get_unique(m_10_p1.chain("a0.h0_concat"))
    proj_printer.print(subcirc)

if MAIN:
    tests.t_ex2_part3_circuit(get_ex2_part3_circuit)

# %% [markdown]
"""
Now make the correspondence. Be sure to avoid the residual node!

This correspondence requires adding two new nodes:
 - "10_p1_00_projected"
 - "20_p1_00_projected"
"""


def make_ex2_part3_corr() -> Correspondence:
    "SOLUTION"
    corr = make_ex2_part2_corr()
    corr.add(
        corr.get_by_name("10_p1_00").make_descendant(p1_open_cond, "10_p1_00_projected"),
        chain_excluding(m_10_p1_h00, "a0.h0_projected", "a0.h0_projected_residual"),
    )
    corr.add(
        corr.get_by_name("20_p1_00").make_descendant(p1_open_cond, "20_p1_00_projected"),
        chain_excluding(m_20_p1_h00, "a0.h0_projected", "a0.h0_projected_residual"),
    )
    return corr


if MAIN:
    tests.t_ex2_part3_corr(make_ex2_part3_corr())
    print("\nEx 2 part 3: Projecting h00 into one direction")
    ex2_p3, loss2_p3 = paren_experiment(ex2_part3_circuit, ds, make_ex2_part3_corr())


# %% [markdown]
"""
### Part 4: The $\phi$ function
"""
# %%
def compute_phi_circuit(tokens: rc.Circuit):
    """
    tokens: [seq_len, vocab_size] array of one hot tokens representing a sequence of parens
    (see ParenTokenizer for the one_hot ordering)

    Returns a circuit that computes phi: tokens -> R^56
    phi = h00_open_vector(2p - 1)
    where p = proportion of parens in `tokens` that are open.

    Returns a circuit with name 'a0.h0_phi'.
    """
    "SOLUTION"
    num_opens = rc.Index(tokens, [slice(None), ParenTokenizer.OPEN_TOKEN], name="is_open").sum(
        axis=-1, name="num_opens"
    )
    num_closes = rc.Index(tokens, [slice(None), ParenTokenizer.CLOSE_TOKEN], name="is_close").sum(
        axis=-1, name="num_closes"
    )
    num_parens = rc.Add(num_opens, num_closes, name="num_parens")
    p_open = num_opens.mul(rc.reciprocal(num_parens), name="p_open")
    p_open_mul_factor = p_open.mul_scalar(2).sub(rc.Scalar(1), name="p_open_mul_factor")
    dir = rc.Array(h00_open_vector, "open direction")
    return dir.mul(p_open_mul_factor, name="a0.h0_phi")


if MAIN:
    tests.t_compute_phi_circuit(compute_phi_circuit)

# %%


def get_ex2_part4_circuit(orig_circuit: rc.Circuit = ex2_part2_circuit, compute_phi_circuit_fn=compute_phi_circuit):
    """
    Split the output of head 0.0 at position 1, when reached through the appropriate paths, into a phi estimate
    and the residual of this estimate.

    The resulting subcircuit should have name 'a0.h0' with children 'a0.h0_phi' and 'a0.h0_phi_residual'.
    """
    "SOLUTION"
    split_by_phi = lambda c: residual_rewrite(c, compute_phi_circuit_fn(c.get_unique("tokens")), "phi")[0]
    return orig_circuit.update((m_10_p1_h00 | m_20_p1_h00).chain("a0.h0"), split_by_phi)


ex2_part4_circuit = get_ex2_part4_circuit()

if MAIN and PRINT_CIRCUITS:
    proj_printer = printer.evolve(
        traversal=rc.new_traversal(term_early_at={"a0.h0.at_pos0", "a0.h0.at_pos2_42", "a0.h0_orig"})
    )
    subcirc = ex2_part4_circuit.get_unique(m_10_p1.chain("a0.h0_concat"))
    proj_printer.print(subcirc)


if MAIN:
    tests.t_ex2_part4_circuit(get_ex2_part4_circuit)

# %% [markdown]
"""
And now make the correspondence -- it should be very similar to the one from part 3. Build on top of the one from part **2**, with new node names "10_p1_00_phi" and "20_p1_00_phi".
"""


def make_ex2_part4_corr() -> Correspondence:
    "SOLUTION"
    corr = make_ex2_part2_corr()
    corr.add(
        corr.get_by_name("10_p1_00").make_descendant(p1_open_cond, "10_p1_00_phi"),
        chain_excluding(m_10_p1_h00, "a0.h0_phi", "a0.h0_phi_residual"),
    )
    corr.add(
        corr.get_by_name("20_p1_00").make_descendant(p1_open_cond, "20_p1_00_phi"),
        chain_excluding(m_20_p1_h00, "a0.h0_phi", "a0.h0_phi_residual"),
    )
    return corr


if MAIN:
    tests.t_ex2_part4_corr(make_ex2_part4_corr())
    print("Ex2 part 4 (2b in writeup): replace a0 by phi(p)")
    ex2b, loss2b = paren_experiment(ex2_part4_circuit, ds, make_ex2_part4_corr())
    check_loss(loss2b, 0.53, 0.04)

# %% [markdown]
"""
Congradulations! This is the end of the main part of today's content. Below is some additional content that covers experiments 3 and 4 from the writeup, although with less detailed testing and instructions.
"""

# %%
if "SKIP":
    ######### Bonus experiments! ########
    # some bonus tests that don't appear in the writeup
    # these test the direct dependances of how information flows from a0 -> 2.0
    # note we don't have a dependance on the tokens here for the 'starts open'
    # which is probably a shortcoming of these hypotheses

    def make_ex2_explicit_paths_corr(from_10=True, to_m0_m1_a0=True, all_to_a0=True) -> Correspondence:
        corr = make_ex1_corr(count_open_cond)
        i_20 = corr.get_by_name("20")
        i_20_p1 = i_20.make_descendant(name="20_p1", cond_sampler=count_open_cond)
        corr.add(i_20_p1, m_20_p1)

        if to_m0_m1_a0:
            i_20_p1_a0 = i_20_p1.make_descendant(name="20_p1_a0", cond_sampler=count_open_cond)
            i_20_p1_m0 = i_20_p1.make_descendant(name="20_p1_m0", cond_sampler=count_open_cond)
            i_20_p1_m1 = i_20_p1.make_descendant(name="20_p1_m1", cond_sampler=count_open_cond)

            m_20_p1_m0 = chain_excluding(m_20_p1, "m0", "m1")
            m_20_p1_m1 = m_20_p1.chain("m1")
            corr.add(i_20_p1_a0, chain_excluding(m_20_p1, "a0.h0", {"m0", "a1.h0", "a1.h1", "m1"}))
            corr.add(i_20_p1_m0, m_20_p1_m0)
            corr.add(i_20_p1_m1, m_20_p1_m1)

            if all_to_a0:
                # m0 depends on a0
                i_20_p1_m0_a0 = i_20_p1_m0.make_descendant(name="20_p1_m0_00", cond_sampler=p1_open_cond)

                # m1 depends on a0, m0
                i_20_p1_m1_a0 = i_20_p1_m1.make_descendant(name="20_p1_m1_00", cond_sampler=p1_open_cond)
                i_20_p1_m1_m0 = i_20_p1_m1.make_descendant(name="20_p1_m1_m0", cond_sampler=p1_open_cond)
                # m1.m0 depends on a0
                i_20_p1_m1_m0_a0 = i_20_p1_m1_m0.make_descendant(name="20_p1_m1_m0_00", cond_sampler=p1_open_cond)

                corr.add(i_20_p1_m0_a0, m_20_p1_m0.chain("a0.h0"))
                corr.add(i_20_p1_m1_a0, chain_excluding(m_20_p1_m1, "a0.h0", "m0"))
                corr.add(i_20_p1_m1_m0, m_20_p1_m1.chain("m0"))
                corr.add(i_20_p1_m1_m0_a0, m_20_p1_m1.chain("m0").chain("a0.h0"))

        if from_10:
            i_10 = corr.get_by_name("10")
            i_10_p1 = i_10.make_descendant(name="10_p1", cond_sampler=count_open_cond)
            corr.add(i_10_p1, m_10_p1)

            if to_m0_m1_a0:
                i_10_p1_a0 = i_10_p1.make_descendant(name="10_p1_a0", cond_sampler=count_open_cond)
                i_10_p1_m0 = i_10_p1.make_descendant(name="10_p1_m0", cond_sampler=count_open_cond)

                corr.add(i_10_p1_a0, chain_excluding(m_10_p1, "a0.h0", {"m0"}))
                m10_p1_m0 = m_10_p1.chain("m0")
                corr.add(i_10_p1_m0, m10_p1_m0)

                if all_to_a0:
                    i_10_p1_m0_a0 = i_10_p1_m0.make_descendant(name="10_p1_m0_00", cond_sampler=p1_open_cond)
                    corr.add(i_10_p1_m0_a0, m10_p1_m0.chain("a0.h0"))

        return corr

    #%%
    if MAIN:
        print("Ex 2_bonus_a: 2.0 depends on only pos1")
        ex2_bonus_a = paren_experiment(ex2_part2_circuit, ds, make_ex2_explicit_paths_corr(False, False, False))

        print("Ex 2_bonus_b: 2.0 depends on only pos1, which depends on a0 + m0 + m1")
        ex2_bonus_b = paren_experiment(ex2_part2_circuit, ds, make_ex2_explicit_paths_corr(False, True, False))

        print("Ex 2_bonus_c: 2.0 depends on only pos1, which depends on a0 + m0 + m1, which bottom out at a0")
        ex2_bonus_c = paren_experiment(ex2_part2_circuit, ds, make_ex2_explicit_paths_corr(False, True, True))

        if PRINT_CIRCUITS:
            ex2_part2_circuit.print(printer)

# %%
"""
# Experiment 3
"""


def separate_all_seqpos(c: rc.Circuit) -> rc.Circuit:
    """
    Separate c into all possible sequence positions.
    c is renamed to `{c.name}_concat`, with children `{c.name}.at_pos{i}`
    """
    "SOLUTION"
    return split_to_concat(c, 0, range(42), partitioning_idx_names=[f"pos{i}" for i in range(42)], use_dot=True).rename(
        f"{c.name}_concat"
    )


if MAIN:
    tests.t_separate_all_seqpos(separate_all_seqpos)

ex3_circuit = circuit
if "SOLUTION":
    ex3_circuit = ex3_circuit.update(rc.IterativeMatcher("a2.h1").chain("a2.input"), separate_all_seqpos)

if MAIN and PRINT_CIRCUITS:
    printer.print(ex3_circuit.get_unique(rc.IterativeMatcher("a2.input_concat")))

if MAIN:
    tests.t_ex3_circuit(ex3_circuit)


# %% [markdown]
"""
When adjusted = True, use the `ds.adjusted_p_open_after` attribute instead of `ds.p_open_after` to compute the horizon test.

One possible gotcha in this section is late-binding-closures messing with the values of i. I think if you follow the outline you should be fine, but if you get strange bugs it's one possibility.
"""


def to_horizon_vals(d: ParenDataset, i: int, adjusted: bool = False) -> torch.Tensor:
    """
    Returns a value for the horizon_i test dividing up the input datums into 5 equivalence classes.
    The actual numerical return values don't have inherent meaning, but are defined as follows:
        0 on padding,
        positive on plausibly-balanced positions,
        negative on unbalance-evidence positions,
        1 / -1 on END_TOKENS,
        2 / -2 on non-end tokens
    """
    "SOLUTION"
    ps: torch.Tensor = d.adjusted_p_open_after if adjusted else d.p_open_after
    toks = d.tokens_flat.value
    assert toks.shape == ps.shape
    conds = [
        toks[:, i] == ParenTokenizer.PAD_TOKEN,
        (toks[:, i] == ParenTokenizer.END_TOKEN) & (toks[:, i - 1] == ParenTokenizer.OPEN_TOKEN),
        (toks[:, i] == ParenTokenizer.END_TOKEN) & (toks[:, i - 1] == ParenTokenizer.CLOSE_TOKEN),
        ps[:, i] > 0.5,
        ps[:, i] <= 0.5,
    ]
    vals = np.select(conds, [0, -1, 1, -2, 2], default=np.nan)
    if np.any(np.isnan(vals)):
        print(i)
        print(d[np.isnan(vals)][:10])
        print(ps[np.isnan(vals)][:10])
        raise AssertionError
    return torch.tensor(vals)


if MAIN:
    tests.t_to_horizon_vals(to_horizon_vals)


def get_horizon_cond(i: int, adjusted: bool) -> FuncSampler:
    """Func sampler for horizon_i"""
    "SOLUTION"
    return FuncSampler(lambda d: to_horizon_vals(ParenDataset.unwrap(d), i, adjusted))


def get_horizon_all_cond(adjusted: bool) -> FuncSampler:
    """Func sampler horizon_all"""
    "SOLUTION"

    def to_horizon_all_vals(d: ParenDataset):
        stacked = torch.stack([to_horizon_vals(d, i, adjusted) for i in range(0, 42)], dim=-1)
        return (stacked >= 0).all(dim=-1)

    return FuncSampler(lambda d: to_horizon_all_vals(ParenDataset.unwrap(d)))


if MAIN:
    tests.t_get_horizon_all_cond(get_horizon_all_cond)


def make_ex3_corr(adjusted: bool = False, corr=None) -> Correspondence:
    """
    `adjusted`: uses `adjusted_p_open_after` based conditions if True, `p_open_after` otherwise.
    `corr`: The starting corr. Uses experiemnt 1b corr by default.

    Makes the following modifications:
     - Changes the cond sampler on node `21` to be the horizon_all cond sampler.
     - Adds one node for each sequence position, called `21_p{i}` with horizon_i cond sampler.
     - Also adds a node `pos_mask`, ensuring the pos_mask of head 2.1 is sampled from an input with the same input length.
    """
    "SOLUTION"
    corr = op.unwrap_or(corr, make_ex1_corr(count_open_cond))

    i_21 = corr.get_by_name("21")
    i_21.cond_sampler = get_horizon_all_cond(adjusted)
    for i in range(0, 42):
        i_pos_i = i_21.make_descendant(name=f"21_p{i}", cond_sampler=get_horizon_cond(i, adjusted))
        corr.add(i_pos_i, m_21.chain(f"a2.input.at_pos{i}"))

    i_pos_mask = i_21.make_descendant(
        name="pos_mask", cond_sampler=FuncSampler(lambda d: ParenDataset.unwrap(d).input_lengths)
    )

    m_pos_mask = chain_excluding(m_21, "pos_mask", "a2.input")
    corr.add(i_pos_mask, m_pos_mask)

    return corr


# %% [markdown]
"""
Note, we have a mini-replication crisis and with the current code can't replicate the exact numbers from the writeup. The handling of the position mask is somewhat different, although much more sane imo. I haven't had time to diagnose the exact cause of the difference.

In any case, expect your loss to be closer to ~1.23 for ex3a and ~1.17 for ex3b
"""
if MAIN:
    print("splitting up 2.1 input by seqpos")

    tests.t_make_ex3_corr(make_ex3_corr)
    print("\nEx3a: first with real open proprotion")
    ex3a, loss3a = paren_experiment(ex3_circuit, ds, make_ex3_corr(adjusted=False))
    check_loss(loss3a, 1.124, 0.1)  # sad sad sad

    print("\nEx3b: now with adjusted open proportion")
    ex3b, loss3b = paren_experiment(ex3_circuit, ds, make_ex3_corr(adjusted=True))
    check_loss(loss3b, 1.140, 0.1)  # sad sad sad

    if PRINT_CIRCUITS:
        ex3b.print()

# %%

if "SKIP":

    def make_ex3c_corr(adjusted: bool):
        corr = make_ex3_corr(adjusted)
        for i in range(42):
            i_pi = corr.get_by_name(f"21_p{i}")
            m_pi = corr[i_pi]
            corr.add(
                i_pi.make_descendant(get_horizon_cond(i, adjusted), f"21_p{i}_a0"),
                chain_excluding(m_pi, "a0.h0", {"m0", "m1"}),
            )
            corr.add(
                i_pi.make_descendant(get_horizon_cond(i, adjusted), f"21_p{i}_m0"), chain_excluding(m_pi, "m0", "m1")
            )
            corr.add(i_pi.make_descendant(get_horizon_cond(i, adjusted), f"21_p{i}_m1"), m_pi.chain("m1"))

        return corr

    if MAIN and SLOW_EXPERIMENTS:
        print("Ex3c: Extending each seqpos to {a0.0, m0, m1}")
        ex3c = paren_experiment(ex3_circuit, ds, make_ex3c_corr(True))

    def make_ex3d_corr(adjusted: bool):
        corr = make_ex3_corr(adjusted)
        for i in range(42):
            i_pi = corr.get_by_name(f"21_p{i}")
            m_pi = corr[i_pi]

            corr.add(i_pi.make_descendant(ExactSampler(), f"21_p{i}_a0"), chain_excluding(m_pi, "a0.h0", {"m0", "m1"}))
            corr.add(i_pi.make_descendant(ExactSampler(), f"21_p{i}_m0"), chain_excluding(m_pi, "m0", "m1"))
            corr.add(i_pi.make_descendant(ExactSampler(), f"21_p{i}_m1"), m_pi.chain("m1"))

        return corr

    if MAIN and SLOW_EXPERIMENTS:
        print("Ex3d: Extending each seqpos to a0.0+m0+m1")
        ex3d = paren_experiment(ex3_circuit, ds, make_ex3d_corr(True))


# %% [markdown]
"""
Now, combine experiments 2 (phi rewrite) and 3 (with adj. proportion)! Expected loss is ~1.64
"""

ex4_circuit = circuit
if "SOLUTION":
    ex4_circuit = ex2_part4_circuit.update(rc.IterativeMatcher("a2.h1").chain("a2.input"), separate_all_seqpos)
    ex4_corr = make_ex3_corr(adjusted=True, corr=make_ex2_part4_corr())
else:
    ex4_corr = Correspondence()

if MAIN and PRINT_CIRCUITS:
    printer.print(ex4_circuit)

if MAIN:
    print("\nEx4: Ex2b (1.0 and 2.0 phi rewrite) + Ex3b (2.1 split by seqpos with p_adj)")
    ex4, loss4 = paren_experiment(ex4_circuit, ds, ex4_corr)
    check_loss(loss4, 1.7, 0.1)  # sad sad sad

# %%
