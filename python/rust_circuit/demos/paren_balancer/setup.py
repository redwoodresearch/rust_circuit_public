import warnings
from functools import lru_cache
from typing import Any, Dict, Optional

import attrs
import einops
import numpy as np
import torch
from fancy_einsum import einsum
from scipy.special import softmax
from torch.nn.functional import layer_norm, one_hot

import rust_circuit as rc
from adversarial.simple_task.dataset_utils import load_dataset
from rust_circuit.causal_scrubbing.dataset import Dataset
from rust_circuit.interop_rust import cached_circuit_by_hash
from rust_circuit.module_library import load_model_id

##################  TOKENIZER ##################


class ParenTokenizer:
    START_TOKEN = 0
    PAD_TOKEN = 1
    END_TOKEN = 2
    OPEN_TOKEN = 3
    CLOSE_TOKEN = 4
    vocab_size = 5

    @classmethod
    def tokenize(cls, strs: list[str], max_len: Optional[int] = None) -> torch.Tensor:
        if max_len is None:
            max_len = max((max(len(s) for s in strs), 1))

        tokenizer = rc.CharTokenizer(
            start=cls.START_TOKEN,
            end=cls.END_TOKEN,
            pad=cls.PAD_TOKEN,
            pad_width=max_len + 2,  # this is length of output, so needs space for begin+end
            mapping={"(": cls.OPEN_TOKEN, ")": cls.CLOSE_TOKEN},
            error_if_over=False,
        )

        return tokenizer.tokenize_strings(strs)

    @classmethod
    def decode(cls, tokens: torch.Tensor) -> list[str]:
        def int_to_c(c: float) -> str:
            if c == cls.OPEN_TOKEN:
                return "("
            if c == cls.CLOSE_TOKEN:
                return ")"
            else:
                raise ValueError(c)

        return [
            "".join(int_to_c(i.item()) for i in seq[1:] if i != cls.PAD_TOKEN and i != cls.END_TOKEN) for seq in tokens
        ]


##################  DATA SET ##################


@cached_circuit_by_hash
def compute_p_open_after(circuit: rc.Array) -> torch.Tensor:
    count_open_after = cumsum_reversed(circuit.value == ParenTokenizer.OPEN_TOKEN, 1)
    count_close_after = cumsum_reversed(circuit.value == ParenTokenizer.CLOSE_TOKEN, 1)
    return count_open_after / (count_open_after + count_close_after)


@cached_circuit_by_hash
def compute_adjusted_p_open_after(circuit: rc.Array, model_id: Any) -> torch.Tensor:
    # there are several divide-by-zeros and means of empty slices that result in nan values.
    # that's okay! These nan values shouldn't ever be referenced (except if the dataset has non-empty strs)
    # so we'll just catch the warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weights = get_adjusted_attn_factors(model_id)
    is_open = circuit.value == ParenTokenizer.OPEN_TOKEN
    # broadcast both to b * q * k
    weighted_opens = weights[(circuit.value != ParenTokenizer.PAD_TOKEN).sum(-1)] * is_open[:, None, :]
    return weighted_opens.sum(-1)


def cumsum_reversed(x: torch.Tensor, dim: int):
    return x.sum(dim, keepdim=True) - x.cumsum(dim) + x


@attrs.frozen
class ParenDataset(Dataset):
    model_id: str = attrs.field()

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

    @property
    def p_open_after(self):
        return compute_p_open_after(self.tokens_flat)

    @property
    def starts_with_open(self) -> torch.Tensor:
        return self.tokens_flat.value[:, 1] == ParenTokenizer.OPEN_TOKEN

    @property
    def count_test(self) -> torch.Tensor:
        return self.p_open_after[:, 0] == 0.5

    @property
    def horizon_test(self) -> torch.Tensor:
        return torch.tensor(np.nanmax(self.p_open_after, axis=1) <= 0.5, dtype=self.tokens.value.dtype)

    @property
    def strs(self):
        return ["b" + s + "e" for s in ParenTokenizer.decode(self.tokens_flat.value)]

    @property
    def input_lengths(self) -> torch.Tensor:
        return (self.tokens_flat.value != ParenTokenizer.PAD_TOKEN).sum(-1)

    @classmethod
    def load(cls, model_id="jun9_paren_balancer", dataset_name="random_choice_len_40_extra_yeses_8"):
        data_list = load_dataset("balanced_parens", dataset_name)["dev"]
        inputs, labels = [s for s, a in data_list], [a for s, a in data_list]

        toks = ParenTokenizer.tokenize(inputs, max_len=40)
        labels = torch.tensor(labels, dtype=torch.float32)

        one_hot_toks = one_hot(toks, ParenTokenizer.vocab_size).to(dtype=torch.float32)

        return cls(
            (rc.Array(toks, name="tokens_flat"), rc.Array(one_hot_toks, name="tokens"), rc.Array(labels, name="is_balanced")),  # type: ignore
            model_id=model_id,
        )

    @property
    def adjusted_p_open_after(self):
        return compute_adjusted_p_open_after(self.tokens_flat, self.model_id)

    def str_values(self) -> str:
        return f"{[('strs', self.strs), (self.is_balanced.name, self.is_balanced.value.to(torch.bool))]}"

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if len(self) == 1:
            s = "ParenDS("
            s += " str:" + self.strs[0]
            s += " bal:" + ("T" if self.is_balanced.value.item() else "F")
            s += " count:" + ("T" if self.count_test.item() else "F")
            s += " horz:" + ("T" if self.horizon_test.item() else "F")
            s += " len:" + str(self.input_lengths.item()) + ")"
            return s
        else:
            return f"ParenDataset(len={len(self)})"


@lru_cache
def get_weights(model_id: str) -> Dict[str, torch.Tensor]:
    circ_dict, _, _ = load_model_id(model_id)
    # we don't include spec child
    return {
        child.name[:-4]: child.cast_array().value  # strip _arr from end
        for i, child in enumerate(circ_dict["t.bind_w"].children)
        if i != 0  # don't include spec
    } | {
        "tok_embeds": circ_dict["t.w.tok_embeds"].cast_array().value,
        "pos_embeds": circ_dict["t.w.pos_embeds"].cast_array().value,
    }


################## ADJUSTED ATTN FACTOR COMPUTATION ##################
# (embarassingly long code for something so simple, don't worry about it)


@lru_cache
def get_adjusted_attn_factors(model_id: str):
    """Returns shape [inputlen, qpos] of the proportion of 0.0's-attention-to-paren-positions that are on or after qpos
    rather than before it. Ignores attn not on these positions for the proportion."""
    attn_probs = calc_h00_attn_probs(model_id)  # [inputlen, qpos, kpos]

    masks = get_attn_arr_masks()

    def avg_attn_in_mask(mask):
        mean_where = torch.tensor(np.array(attn_probs).mean(axis=-1, keepdims=True, where=np.array(mask)))
        avg_by_key = torch.nan_to_num(mean_where, nan=0)
        return mask * avg_by_key

    weights = avg_attn_in_mask(masks["paren_before_qpos"]) + avg_attn_in_mask(masks["paren_on_or_after_qpos"])
    weights_normed = weights / weights.sum(dim=-1, keepdim=True)

    return torch.tensor(weights_normed)


def a0_layer_norm(x, model_id: str):
    weights = get_weights(model_id)
    return layer_norm(x, (56,), weight=weights["a0.ln.w.scale"], bias=weights["a0.ln.w.bias"])


def make_input_arr(model_id: str, just_opens=False) -> torch.Tensor:
    """
    Returns an array of [input len, seq pos, d model]
    There are 41 possible input lens (from 2 to 42), but we'll make that dimension 43 long so indexing is easy.
    By default averages the ln-ed open and close positions for the positions corresponding to parens. If just_opens is true, it just uses the open embeds
    """
    weights = get_weights(model_id)
    # [pos, tok, d_model] matrix of possible input embeds
    possible_embeds = weights["pos_embeds"][:42, None, :] + weights["tok_embeds"][None, :, :]
    possible_ln_embeds = a0_layer_norm(possible_embeds, model_id)

    if just_opens:
        paren_vals = possible_ln_embeds[:, ParenTokenizer.OPEN_TOKEN, :]
    else:
        paren_vals = possible_ln_embeds[:, [ParenTokenizer.OPEN_TOKEN, ParenTokenizer.CLOSE_TOKEN], :].mean(1)

    input_lens_arr = einops.repeat(torch.arange(43), "inputlens -> inputlens seqpos dmodel", seqpos=42, dmodel=56)
    seq_pos_arr = einops.repeat(torch.arange(42), "seqpos -> inputlens seqpos dmodel", inputlens=43, dmodel=56)

    arr = torch.where(seq_pos_arr < input_lens_arr, paren_vals, possible_ln_embeds[:, ParenTokenizer.PAD_TOKEN, :])
    arr = torch.where(seq_pos_arr == 0, possible_ln_embeds[None, 0, ParenTokenizer.START_TOKEN, :], arr)
    arr = torch.where(seq_pos_arr == input_lens_arr - 1, possible_ln_embeds[None, :, ParenTokenizer.END_TOKEN, :], arr)
    arr[[0, 1], :, :] = torch.nan
    return arr


def get_attn_arr_masks() -> Dict[str, torch.Tensor]:
    """Returns a bunch of helpful masks of shape (43, 42, 42) = (inputpos, qpos, kpos)"""
    inputlens_arr = einops.repeat(torch.arange(43), "inputlens -> inputlens qpos kpos", qpos=42, kpos=42)
    qpos_arr = einops.repeat(torch.arange(42), "qpos -> inputlens qpos kpos", inputlens=43, kpos=42)
    kpos_arr = einops.repeat(torch.arange(42), "kpos -> inputlens qpos kpos", inputlens=43, qpos=42)

    masks = {
        "paren_pos": (kpos_arr >= 1) & (kpos_arr <= inputlens_arr - 2),
        "padding_pos": kpos_arr >= inputlens_arr,
        "kpos_before_qpos": (kpos_arr < qpos_arr),
        "kpos_on_or_after_qpos": (kpos_arr >= qpos_arr),
    }
    masks["paren_before_qpos"] = masks["paren_pos"] & masks["kpos_before_qpos"]
    masks["paren_on_or_after_qpos"] = masks["paren_pos"] & masks["kpos_on_or_after_qpos"]
    return masks


def calc_h00_attn_probs(model_id: str) -> torch.Tensor:
    weights = get_weights(model_id)

    query_inputs = make_input_arr(model_id, just_opens=True)
    key_inputs = make_input_arr(model_id, just_opens=False)

    queries = (
        einsum("dhead dmodel, inplen qpos dmodel -> inplen qpos dhead", weights["a0.w.q"][0], query_inputs)
        + weights["a0.w.q_bias"][0]
    )
    keys = (
        einsum("dhead dmodel, inplen kpos dmodel -> inplen kpos dhead", weights["a0.w.k"][0], key_inputs)
        + weights["a0.w.k_bias"][0]
    )
    unmasked_scores = einsum("inplen qpos dmodel, inplen kpos dmodel -> inplen qpos kpos", queries, keys)

    masks = get_attn_arr_masks()
    masked_scores = torch.where(masks["padding_pos"], unmasked_scores - 10_000, unmasked_scores)
    scores = masked_scores / (28 ** 0.5)
    return softmax(scores, axis=-1)


def get_h00_open_vector(model_id: str) -> torch.Tensor:
    """
    This function estimates a single vector v which head 0.0 would theoretically output if p=1 (the entire sequence is open parens). It would thus (hypothetically) output -v if p=0, and 0 if p=0.5.
    """
    weights = get_weights(model_id)

    def through_00(x):
        ln_out = a0_layer_norm(x, model_id)
        v_out = torch.einsum("oi, ...i -> ...o", weights["a0.w.v"][0], ln_out) + weights["a0.w.v_bias"][0]
        return torch.einsum("oi, ...i -> ...o", weights["a0.w.o"][0], v_out)

    open_outs = through_00(weights["tok_embeds"][ParenTokenizer.OPEN_TOKEN] + weights["pos_embeds"][1:41])
    close_outs = through_00(weights["tok_embeds"][ParenTokenizer.CLOSE_TOKEN] + weights["pos_embeds"][1:41])

    # empirically all the open outs are close together, and nearly-opposite all hte close outs. But we need to
    # average them together to return a single vector
    return (open_outs.mean(0) - close_outs.mean(0)) / 2
