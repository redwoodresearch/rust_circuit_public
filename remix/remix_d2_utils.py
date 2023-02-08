"""
REMIX Utilities for Day 2

This file contains spoilers for Day 2 of REMIX! Read at your own peril!

"""
import torch as t
import rust_circuit as rc
from rust_circuit import Circuit, Einsum, Scalar, Array
from dataclasses import dataclass
from typing import Callable
import re


def working_layernorm(input: Circuit, weight: Circuit, bias: Circuit, eps=1e-5) -> Circuit:
    """Circuit computing the same thing as TorchLayerNorm."""
    "SOLUTION"
    recip_h = rc.reciprocal(rc.last_dim_size(input))
    centered = input.add(Einsum.from_einsum_string("h,z,->z", input, Scalar(-1, (1,)), recip_h))
    var = Einsum.from_einsum_string("h,h,->", centered, centered, recip_h)
    scale_factor = rc.rsqrt(var.add(Scalar(eps)))
    y = centered.mul(scale_factor)
    return y.mul(weight).add(bias)


def mystery_layernorm_a(input: Circuit, weight: Circuit, bias: Circuit, eps=1e-5) -> Circuit:
    recip_h = rc.reciprocal(rc.last_dim_size(input))
    centered = input.add(Einsum.from_einsum_string("h,z,->z", input, Scalar(-1, (1,)), recip_h))
    var = Einsum.from_einsum_string("h,h,->", centered, centered, recip_h)
    scale_factor = rc.rsqrt(var)
    y = centered.mul(scale_factor)
    return y.mul(weight).add(bias)


def mystery_layernorm_b(input: Circuit, weight: Circuit, bias: Circuit, eps=1e-5) -> Circuit:
    recip_h = 1.0 / input.shape[-1]
    centered = input.add(Einsum.from_einsum_string("h,z->z", input, Scalar(-recip_h, (1,))))
    var = Einsum.scalar_mul(Einsum.from_einsum_string("h,h->", centered, centered), recip_h)
    scale_factor = rc.rsqrt(var.add(Scalar(eps)))
    y = centered.mul(scale_factor)
    return y.mul(weight).add(bias)


def _get(bind_module: Circuit, names: list[str], renamer: Callable[[str], str]) -> dict[str, Circuit]:
    out = {}
    for name in names:
        assert name.endswith("_arr")
        try:
            arr = bind_module.get_unique(name)
        except RuntimeError:
            print(f"Failed to find anything for {name}")
        else:
            without_arr = name[:-4]
            if renamer is not None:
                arr = arr.rename(renamer(without_arr))
            # print(arr.name)
            out[arr.name] = arr
    return out


def _ln_replacer(name: str):
    """Strip "m11." or "a5." prefixes."""
    return re.sub(r"^[am]\d+\.ln", "ln", name)


def _replacer(name: str):
    """Replace a5.w -> a.w and m11.w -> m.w"""
    return re.sub(r"([am])\d+\.w", r"\1.w", name)


@dataclass
class LayerWeights:
    """Container holding the weights for 1 layer of GPT."""

    ln1: dict[str, Circuit]
    attn: dict[str, Circuit]
    ln2: dict[str, Circuit]
    mlp: dict[str, Circuit]


@dataclass
class GPT2Weights:
    # TBD lowpri: this is mutable b/c it has attention mask which is awkward
    weights_by_layer: list[LayerWeights]
    tok_embeds: Array
    pos_embeds: Array
    final_ln_scale: Array
    final_ln_bias: Array

    def set_attention_mask(self, attention_mask: Circuit):
        for lw in self.weights_by_layer:
            lw.attn["a.mask"] = attention_mask


def get_weights(circ_dict, bind_module: Circuit) -> GPT2Weights:
    """Munge weights from the reference transformer into a nicer struct.."""
    weights_by_layer: list[LayerWeights] = []
    for i in range(12):
        ln1 = _get(bind_module, [f"a{i}.ln.w.scale_arr", f"a{i}.ln.w.bias_arr"], _ln_replacer)
        attn1 = _get(
            bind_module,
            [
                f"a{i}.w.k_arr",
                f"a{i}.w.q_arr",
                f"a{i}.w.o_arr",
                f"a{i}.w.v_arr",
                f"a{i}.w.k_bias_arr",
                f"a{i}.w.q_bias_arr",
                f"a{i}.w.o_bias_arr",
                f"a{i}.w.v_bias_arr",
            ],
            _replacer,
        )
        ln2 = _get(bind_module, [f"m{i}.ln.w.scale_arr", f"m{i}.ln.w.bias_arr"], _ln_replacer)
        mlp_weights = _get(
            bind_module,
            [
                f"m{i}.w.in_bias_arr",
                f"m{i}.w.out_bias_arr",
                f"m{i}.w.proj_in_arr",
                f"m{i}.w.proj_out_arr",
            ],
            _replacer,
        )
        weights_by_layer.append(
            LayerWeights(
                ln1,
                attn1,
                ln2,
                mlp_weights,
            )
        )
    return GPT2Weights(
        weights_by_layer,
        circ_dict["t.w.tok_embeds"],
        circ_dict["t.w.pos_embeds"],
        bind_module.get_unique("final.ln.w.scale_arr").cast_array(),
        bind_module.get_unique("final.ln.w.bias_arr").cast_array(),
    )


def get_ref_attn_mask_expanded(ref_circuit: Circuit, rand_scores: Circuit, rand_mask: Circuit) -> Circuit:
    ref_attention_mask_circ = rc.Matcher("b0.call").chain("a.attn_scores").get_unique(ref_circuit)
    ref_attention_mask_circ_expander = rc.Expander(
        ("a.attn_scores_raw", lambda _: rand_scores.mul(rand_mask)),
        ("a.mask", lambda _: rand_mask),
    )
    return ref_attention_mask_circ_expander(ref_attention_mask_circ, fancy_validate=True)


def get_ref_attn_score_expanded(ref_circuit: Circuit, rand_emb: Circuit, attn_weight: dict[str, Circuit]) -> Circuit:
    ref_attn_score_circ = rc.Matcher("b0.call").chain("a.attn_scores_raw").get_unique(ref_circuit)
    ref_attn_score_circ_expander = rc.Expander(
        ("a.mask", lambda _: Array(t.ones((10, 10)))),
        ("a.q.input", lambda _: rand_emb),
        ("a.k.input", lambda _: rand_emb),
        ("a.w.q_h", lambda _: Array(attn_weight["a.w.q"].cast_array().value[0])),
        (
            "a.w.q_bias_h",
            lambda _: Array(attn_weight["a.w.q_bias"].cast_array().value[0]),
        ),
        ("a.w.k_h", lambda _: Array(attn_weight["a.w.k"].cast_array().value[0])),
        (
            "a.w.k_bias_h",
            lambda _: Array(attn_weight["a.w.k_bias"].cast_array().value[0]),
        ),
    )
    return ref_attn_score_circ_expander(ref_attn_score_circ, fancy_validate=True)


def get_ref_attn_expanded(ref_circuit: Circuit, rand_emb: Circuit, attn_weight: dict[str, Circuit]) -> Circuit:
    """Return the reference circuit with the specified values expanded in."""
    ref_attn = rc.Matcher("b0.call").chain("a.p_bias").get_unique(ref_circuit)
    ref_attn_expander = rc.Expander(
        (
            "a.input",
            lambda _: rand_emb,
        ),
        (
            "a.w.q",
            lambda _: attn_weight["a.w.q"],
        ),
        (
            "a.w.q_bias",
            lambda _: attn_weight["a.w.q_bias"],
        ),
        (
            "a.w.k",
            lambda _: attn_weight["a.w.k"],
        ),
        (
            "a.w.k_bias",
            lambda _: attn_weight["a.w.k_bias"],
        ),
        (
            "a.w.v",
            lambda _: attn_weight["a.w.v"],
        ),
        (
            "a.w.v_bias",
            lambda _: attn_weight["a.w.v_bias"],
        ),
        (
            "a.w.o",
            lambda _: attn_weight["a.w.o"],
        ),
        (
            "a.w.o_bias",
            lambda _: attn_weight["a.w.o_bias"],
        ),
        (
            "a.mask",
            lambda _: attn_weight["a.mask"],
        ),
    )
    return ref_attn_expander(ref_attn, fancy_validate=True)


def get_ref_block_expanded(ref_circuit: Circuit, rand_emb: Circuit, weights: LayerWeights) -> Circuit:
    ref_block_expander = rc.Expander(
        ("b.input", lambda _: rand_emb),
        ("a0.ln.w.scale", lambda _: weights.ln1["ln.w.scale"]),
        ("a0.ln.w.bias", lambda _: weights.ln1["ln.w.bias"]),
        ("a0.w.q", lambda _: weights.attn["a.w.q"]),
        ("a0.w.q_bias", lambda _: weights.attn["a.w.q_bias"]),
        ("a0.w.k", lambda _: weights.attn["a.w.k"]),
        ("a0.w.k_bias", lambda _: weights.attn["a.w.k_bias"]),
        ("a0.w.v", lambda _: weights.attn["a.w.v"]),
        ("a0.w.v_bias", lambda _: weights.attn["a.w.v_bias"]),
        ("a0.w.o", lambda _: weights.attn["a.w.o"]),
        ("a0.w.o_bias", lambda _: weights.attn["a.w.o_bias"]),
        ("a.mask", lambda _: weights.attn["a.mask"]),
        ("m0.ln.w.scale", lambda _: weights.ln2["ln.w.scale"]),
        ("m0.ln.w.bias", lambda _: weights.ln2["ln.w.bias"]),
        ("m0.w.proj_in", lambda _: weights.mlp["m.w.proj_in"]),
        ("m0.w.proj_out", lambda _: weights.mlp["m.w.proj_out"]),
        ("m0.w.in_bias", lambda _: weights.mlp["m.w.in_bias"]),
        ("m0.w.out_bias", lambda _: weights.mlp["m.w.out_bias"]),
    )
    ref_block = rc.IterativeMatcher("b0.call").get_unique(ref_circuit)
    return ref_block_expander(ref_block, fancy_validate=True)
