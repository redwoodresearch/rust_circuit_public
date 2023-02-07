from typing import Dict, Literal, Optional

import einops
import jax.numpy as jnp
import numpy as np
import torch

from interp.model.blocks import BatchNormFixed, LayerNorm
from interp.model.gpt_model import Gpt
from interp.model.gpt_modules import Attention, GptBlock, PosEncType, id_norm
from rust_circuit import module_library as mod_l

from . import _rust as rc


def from_converted(arr: jnp.ndarray, name: Optional[str] = None):
    return rc.Array(torch.from_numpy(np.array(arr)), name=name)


def from_converted_tup(arr: jnp.ndarray, s: str):
    return (s, from_converted(arr))


def get_norm(norm_b):
    if not isinstance(norm_b, (LayerNorm, BatchNormFixed)):
        raise NotImplementedError("unsupported norm type, currently only supports ln/bn fixed", type(norm_b))
    if norm_b.epsilon != 1e-5:
        raise NotImplementedError("TODO: support variable epsilon")
    nt = "ln" if isinstance(norm_b, LayerNorm) else "bn"
    out = [
        from_converted_tup(norm_b.variables["params"]["bias"], s=f"{nt}.w.bias"),
        from_converted_tup(norm_b.variables["params"]["scale"], s=f"{nt}.w.scale"),
    ]
    if nt == "bn":
        out.extend(
            [
                from_converted_tup(norm_b.variables["params"]["mean"], s=f"{nt}.mean"),
                from_converted_tup(norm_b.variables["params"]["var"], s=f"{nt}.var"),
            ]
        )

    return out, nt


def get_attention(attention_b: Attention, use_pos: bool):
    if attention_b.softmax_type != "softmax":
        raise NotImplementedError
    q_w, k_w, v_w = attention_b.get_qkv_mats()
    o_w = einops.rearrange(
        attention_b.project_output.get_weights(),
        "hidden_out (num_heads head_size) -> num_heads hidden_out head_size",
        num_heads=attention_b.num_heads,
    )

    out = [
        from_converted_tup(q_w, "a.w.q"),
        from_converted_tup(k_w, "a.w.k"),
        from_converted_tup(v_w, "a.w.v"),
        from_converted_tup(o_w, "a.w.o"),
    ]

    if attention_b.bias:
        q_bias, k_bias, v_bias = einops.rearrange(
            attention_b.attn_weights.get_bias(),
            "(k num_heads head_size) -> k num_heads head_size",
            k=3,
            num_heads=q_w.shape[0],
        )
        o_bias = attention_b.project_output.get_bias()
        out.extend(
            [
                from_converted_tup(q_bias * (2 if use_pos else 1), "a.w.q_bias"),
                from_converted_tup(k_bias * (2 if use_pos else 1), "a.w.k_bias"),
                from_converted_tup(v_bias, "a.w.v_bias"),
                from_converted_tup(o_bias, "a.w.o_bias"),
            ]
        )

    return out


def get_mlp(block_b: GptBlock):
    proj_in = block_b.linear1.get_weights()
    in_bias = block_b.linear1.get_bias()

    proj_out = block_b.linear2.get_weights()

    if block_b.mlp_act_type == "bilinear":
        # only half of params are used in this case
        proj_out = einops.rearrange(proj_out, "hidden_out (a mlp_proj) -> a hidden_out mlp_proj", a=2)[0]

    out = [
        from_converted_tup(proj_in, "m.w.proj_in"),
        from_converted_tup(in_bias, "m.w.in_bias"),
        from_converted_tup(proj_out, "m.w.proj_out"),
    ]

    if block_b.linear2.use_bias:
        out.append(from_converted_tup(block_b.linear2.get_bias(), "m.w.out_bias"))

    return out


def get_block(block_b: GptBlock, pos_enc_type: PosEncType):
    all_inputs: dict[str, rc.Circuit] = {}

    use_norm = block_b.norm1 is not id_norm

    attn_pos = pos_enc_type == "shortformer"

    norm_type = None
    if use_norm:
        norm_circs_attn, norm_type = get_norm(block_b.norm1)
        all_inputs.update(mod_l.apply_prefix(dict(norm_circs_attn), "a"))

    all_inputs.update(get_attention(block_b.attention, use_pos=attn_pos))

    if block_b.use_mlp:
        if use_norm:
            mlp_norm_circs, norm_type_m = get_norm(block_b.norm2)
            assert norm_type_m == norm_type
            all_inputs.update(mod_l.apply_prefix(dict(mlp_norm_circs), "m"))

        all_inputs.update(get_mlp(block_b))

    return all_inputs


def get_model(model_b: Gpt):
    all_inputs: Dict[str, rc.Circuit] = {}
    for block_i in range(model_b.num_layers):
        block_b = model_b.blocks[block_i]
        all_inputs.update(
            {mod_l.add_number(s, block_i): c for s, c in get_block(block_b, pos_enc_type=model_b.pos_enc_type).items()}
        )

    if model_b.norm_type != "none" and model_b.use_norm_output:
        norm, _ = get_norm(model_b.norm_output)
        all_inputs.update(mod_l.apply_prefix(dict(norm), "final"))

    unembed_name = "t.w.unembed"
    if model_b.classifier:
        unembedding = from_converted(model_b.embedding.linear_out.get_weights(), name=unembed_name)
    else:
        unembedding = from_converted(model_b.embedding.token_unembedding.embedding, name=unembed_name)
    all_inputs[unembedding.name] = unembedding

    if model_b.classifier:
        output_bias = from_converted(model_b.embedding.linear_out.get_bias(), name="t.w.unembed_bias")
        all_inputs[output_bias.name] = output_bias

    return all_inputs


def get_model_info(model_b: Gpt, model_class: str = "GPTBeginEndToks"):
    norm_type: Optional[Literal["ln", "bn"]]
    if model_b.norm_type == "none":
        norm_type = None
    elif model_b.norm_type == "layer_norm":
        norm_type = "ln"
    elif model_b.norm_type == "batch_norm_fixed":
        norm_type = "bn"
    else:
        raise NotImplementedError("unsupported norm type, currently only supports ln/bn fixed", model_b.norm_type)
    return mod_l.TransformerInfo(
        mod_l.TransformerParams(
            mod_l.TransformerBlockParams(
                norm_type=norm_type,
                attn_bias=model_b.attn_bias,
                attn_pos=model_b.pos_enc_type == "shortformer",
                use_mlp=model_b.use_mlp,
                mlp_act_type=model_b.mlp_act_type,
                mlp_output_bias=model_b.mlp_bias,
            ),
            num_layers=model_b.num_layers,
            use_norm_output=model_b.use_norm_output,
            output_bias=model_b.classifier,
        ),
        model_class=model_class,
        pos_enc_type=model_b.pos_enc_type,
        causal_mask=model_b.causal_mask,
    )


def get_bound_model(model_b: Gpt, model_class: str = "GPTBeginEndToks"):
    all_inputs = mod_l.rename_circs_to_keys(get_model(model_b), "_arr")
    tok_embeds = from_converted(model_b.embedding.token_embedding.embedding, name="t.w.tok_embeds")
    pos_embeds = from_converted(model_b.embedding.position_embedding.embedding, name="t.w.pos_embeds")
    info = get_model_info(model_b, model_class=model_class)
    model_ret = info.params.get()
    circ = rc.module_new_bind(model_ret.body, *list(all_inputs.items()), name="t.bind_w")

    return circ, (tok_embeds, pos_embeds), info, model_ret
