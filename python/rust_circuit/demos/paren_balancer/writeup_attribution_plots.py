# %%
import jax
import matplotlib.pyplot as plt
import numpy as np
import torch

import interp.tools.optional as op
from interp.model.gpt_model import Gpt
from interp.model.model_loading import load_model_mem_cached
from interp.tools.log import Idxs, KeyIdxs, LoggerCache, LogInfo, construct_mut_log_cache

from .setup import ParenDataset, ParenTokenizer

jax.config.update("jax_platform_name", "cpu")


#%% [markdown]
"""This notebook produces plots for the causal scrubbing paren balancer writeup."""
# %%
# define model

MODEL_ID = "jun9_paren_balancer"

ds = ParenDataset.load(MODEL_ID)
ds = ds[:1000]

# %%
def bce_with_logits_loss(logits, labels):
    targets = torch.tensor(labels, dtype=torch.float, device="cuda:0")
    logit_diff = logits[:, 1] - logits[:, 0]
    correct = (logit_diff > 0) == targets
    return torch.nn.BCEWithLogitsLoss(reduction="none")(logit_diff, targets), correct


def get_bound_jax_model(model_id: str):
    jax_model, jax_params, _ = load_model_mem_cached(model_id)
    return jax_model.bind(jax_params)


def run_dataset_on_jax_model(dataset: ParenDataset, model_id: str, log_info=None):
    mask = (dataset.tokens_flat.value == ParenTokenizer.PAD_TOKEN).cpu().numpy()
    config = Gpt.CallConfig(pos_mask=mask)
    log = construct_mut_log_cache(log_info, None)
    jax_model = get_bound_jax_model(model_id)
    out = jax_model(dataset.tokens_flat.value.cpu().numpy(), log=log, config=config)[:, 0, :]
    probs = jax.nn.softmax(out)
    return out, probs, op.map(log, lambda log: log.cache)


# %%
a2_by_head = KeyIdxs("blocks.attention.out_by_head", idxs=Idxs.single(2))
norm_in = KeyIdxs("final_out.norm.inp")
logger = LoggerCache.from_key_idxs([a2_by_head, norm_in])
out, probs, cache = run_dataset_on_jax_model(ds, MODEL_ID, log_info=LogInfo(logger))
cache = op.unwrap(cache)

h20_out = cache.get(a2_by_head)[:, 0, 0, :]
h21_out = cache.get(a2_by_head)[:, 1, 0, :]
all_terms = cache.get(norm_in)[:, 0, :]

# %%
def attribution_score(head_term, full_sum):
    assert head_term.ndim == 2 and full_sum.ndim == 2  # [batch, hiddendim]
    rem = full_sum - head_term  # []
    possible_sums = head_term[:, None, :] + rem[None, :, :]  # [batch.head, batch.remainders, hidden]
    logits = np.array(get_bound_jax_model(MODEL_ID).out(possible_sums))
    logit_diffs = logits[..., 1] - logits[..., 0]
    return logit_diffs.mean(1) - logit_diffs.mean()


h20_attr = attribution_score(h20_out, all_terms)
h21_attr = attribution_score(h21_out, all_terms)

# %%
# set some globals that later cells can change
# thanks late binding closures!
alpha = 0.5
size = 25


def mk_scatter(filter, label=None, color=None, ax=None):
    ax = plt.gca() if ax is None else ax
    ax.scatter(h20_attr[filter], h21_attr[filter], label=label, color=color, s=size, alpha=alpha)


is_balanced = np.array(ds.is_balanced.value, dtype="bool")
count_test = ds.count_test.bool()
horizon_test = ds.horizon_test.bool()
ele_and_open = count_test & horizon_test

# %%
mk_scatter(is_balanced, "balanced", "#1b9e77")
mk_scatter(count_test & ~horizon_test, "just horizon failure", "#e7298a")
mk_scatter(~count_test & horizon_test, "just count ${}^($ failure", "#d95f02")
mk_scatter(~count_test & ~horizon_test, "both failures", "#7570b3")

plt.xlabel("Logit difference from 2.0")
plt.ylabel("Logit difference from 2.1")
plt.legend()
plt.gcf().set_size_inches(5, 5)

plt.show()

# %%
ele_and_open = count_test & ds.starts_with_open

mk_scatter(~ele_and_open & ~horizon_test, "both failures", "#7570b3")
mk_scatter(~ele_and_open & horizon_test, "just count${}^($ failure", "#d95f02")
mk_scatter(ele_and_open & ~horizon_test, "just horizon failure", "#e7298a")
mk_scatter(is_balanced, "balanced", "#1b9e77")

plt.xlabel("Logit difference from 2.0")
plt.ylabel("Logit difference from 2.1")
plt.legend()
plt.gcf().set_size_inches(5, 5)
plt.show()

# %%

fig, axs = plt.subplots(1, 3, sharey=True, figsize=(10, 3))
axs[0].set_title("Passes horizon test?")
pass_c = "#377eb8"
fail_c = "#e41a1c"
alpha = 0.3
size = 8

mk_scatter(horizon_test, color=pass_c, ax=axs[0])
mk_scatter(~horizon_test, color=fail_c, ax=axs[0])

axs[1].set_title("Passes count test?")
mk_scatter(~count_test, color=fail_c, ax=axs[1])
mk_scatter(count_test, color=pass_c, ax=axs[1])

axs[2].set_title("Passes count${}^($ test?")
mk_scatter(~ele_and_open, "fail", fail_c, ax=axs[2])
mk_scatter(ele_and_open, "pass", pass_c, ax=axs[2])

axs[2].legend()

axs[0].set_xlabel("Logit difference from 2.0")
axs[1].set_xlabel("Logit difference from 2.0")
axs[2].set_xlabel("Logit difference from 2.0")
axs[0].set_ylabel("Logit difference from 2.1")

plt.show()
# %%
