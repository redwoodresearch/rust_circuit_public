from __future__ import annotations

import functools
import json
import random
from copy import copy
from typing import Any, Callable, Dict, List, Literal, Optional, TypeVar, Union, cast
from uuid import UUID, uuid4

import cattrs
import torch
from attrs import frozen
from cattrs.preconf.json import make_converter as make_json_converter

import rust_circuit.optional as op

from ._rust import (
    Add,
    Array,
    Circuit,
    Einsum,
    GeneralFunction,
    Matcher,
    MatcherIn,
    Module,
    ModuleArgSpec,
    ModuleSpec,
    Parser,
    PrintOptions,
    Scalar,
    Shape,
    Symbol,
    TorchDeviceDtypeOp,
    deep_module_remove_unused_inputs,
    get_free_symbols,
    module_new_bind,
    simp,
    symbolic_sizes,
)
from .py_utils import I

P = Parser()


# WARNING: Not tested for correctness!

# naming convention: module 'spec_circuit' has name 'x' while children are 'x.y'
# weights are 'x.w.y' ('w' isn't a placeholder). Constants are 'x.c.y'.
# then, callers can do 'z.x' or 'x.call' (or similar)

(
    HIDDEN,
    # OUT_HIDDEN,
    SEQ,
    SEQ_Q,
    SEQ_K,
    HEADS,
    MLP_PROJ,
    BILINEAR_MLP_PROJ,
    QK_DIM,
    V_DIM,
    LOG_LIKELYHOOD_CLASSES,
    VOCAB_SIZE,
    HALF_ROTARY_DIM,  # 1/4 of QK_DIM
    *_,
) = symbolic_sizes()
OUT_HIDDEN = HIDDEN  # fix out hidden = hidden as needed (https://github.com/redwoodresearch/unity/issues/1718, https://github.com/redwoodresearch/unity/issues/1716)


def to_name_dict(*circs: Circuit):
    out = {c.name: c for c in circs}
    assert len(out) == len(circs)
    return out


def add_new_circs(s: str, circs: Dict[str, Circuit], parser: Optional[Parser] = None):
    parser = op.unwrap_or(parser, Parser())
    parser.reference_circuits = {**parser.reference_circuits, **circs}

    return {**circs, **to_name_dict(*parser.parse_circuits(s))}


def make_spec(
    body: Circuit, order: List[Circuit] = [], exclude: Union[Callable[[Symbol], bool], MatcherIn] = lambda _: False
):
    exclude_matcher = Matcher(cast(MatcherIn, exclude))
    return ModuleSpec(
        body,
        [
            ModuleArgSpec(x)
            for x in sorted(get_free_symbols(body), key=lambda c: order.index(c) if c in order else -1)
            if not exclude_matcher(x)
        ],
    )


@frozen
class ModuleLibraryRet:
    circs: Dict[str, Circuit]
    body: Circuit
    spec: ModuleSpec
    name: str

    @classmethod
    def mk(cls, circs: Dict[str, Circuit], name: str) -> ModuleLibraryRet:
        body = circs[name]
        spec = make_spec(body, order=list(circs.values()))
        return cls(circs=circs, body=body, spec=spec, name=name)


def zero_input(circs: Dict[str, Circuit], to_zero: str):
    return to_name_dict(
        *(
            # we could make this faster if we needed
            deep_module_remove_unused_inputs(simp(c.update(to_zero, lambda x: Scalar(0.0, shape=x.shape)))).update(
                True, lambda x: x.rename(x.name.removesuffix(" rem_unused"))
            )
            for s, c in circs.items()
            if s != to_zero
        )
    )


layernorm_str = f"""
'ln.y' Einsum h,->h
  'ln.mean_subbed' Add
    'ln.input' [{HIDDEN}] Symbol 981b4d2a-711b-4a9d-a11c-d859c311e80c
    'ln.neg_mean' Einsum h,z,->z # z is padding dim for add
      'ln.input'
      'ln.neg' [1] Scalar -1
      'ln.c.recip_hidden_size' GeneralFunction reciprocal
        'ln.c.hidden_size' GeneralFunction last_dim_size
          'ln.input'
  'ln.rsqrt' GeneralFunction rsqrt
    'ln.var_p_eps' Add
      'ln.c.eps' [] Scalar 1e-5
      'ln.var' Einsum h,h,->
        'ln.mean_subbed'
        'ln.mean_subbed'
        'ln.c.recip_hidden_size'

'ln.input'

'ln' Add
  'ln.w.bias' [{HIDDEN}] Symbol 621c7792-0177-45ab-87c5-7ff1c3bec487
  'ln.y_scaled' Einsum h,h->h
    'ln.y'
    'ln.w.scale' [{HIDDEN}] Symbol 0fa341c3-34b3-4699-847f-08674808b28a

'ln.w.bias'
'ln.w.scale'
"""

layernorm_circs = add_new_circs(layernorm_str, {})
layernorm = ModuleLibraryRet.mk(layernorm_circs, "ln")

batchnorm_str = f"""
'bn.y' Einsum h,h->h
  'bn.mean_subbed' Add
    'bn.input' [{HIDDEN}] Symbol cfe0aa25-1214-4d3e-ad53-417fedbb9b7b
    'bn.neg_mean' Einsum h,->h
      'bn.mean' [{HIDDEN}] Symbol 217403c9-03b6-4444-8f72-de03b95229d0
      'bn.neg' [] Scalar -1
  'bn.rsqrt' GeneralFunction rsqrt
    'bn.var_p_eps' Add
      'bn.c.eps' [] Scalar 1e-5
      'bn.var' [{HIDDEN}] Symbol c4e41ca7-e3ab-429e-81b6-25bb14423fb4

'bn.input'
'bn.mean'
'bn.var'

# same as ln above, we could dedudup if we wanted
'bn' Add
  'bn.w.bias' [{HIDDEN}] Symbol 3e861c08-9a74-4348-87df-3752a8557aea
  'bn.y_scaled' Einsum h,h->h
    'bn.y'
    'bn.w.scale' [{HIDDEN}] Symbol ccbac7e0-c931-422b-9724-1e24ab6e9c63

'bn.w.bias'
'bn.w.scale'
"""

batchnorm_circs = add_new_circs(batchnorm_str, {})
batchnorm = ModuleLibraryRet.mk(batchnorm_circs, "bn")

not_mask_str = """
'not_mask' Add
  'one' [] Scalar 1.
  'not_mask.neg_mask' Einsum ,->
    'not_mask.input' [] Symbol b46f6370-11e1-4535-aabc-94554c234673
    'neg_one' [] Scalar -1.

'not_mask.input'
"""

not_mask_circs = add_new_circs(not_mask_str, {})
not_mask = ModuleLibraryRet.mk(not_mask_circs, "not_mask")

raw_attention_str = f"""
'a.head' Einsum sV,dV->sd
  'a.comb_v' Einsum qk,kV->qV
    'a.attn_probs' GeneralFunction softmax
      'a.attn_scores' Add
        'a.attn_scores_raw' [{SEQ_Q}, {SEQ_K}] Einsum qc,kc,,qk->qk
          'a.q_p_bias' [{SEQ_Q}, {QK_DIM}] Add
            'a.q' Einsum qd,cd->qc
              'a.q.input' [{SEQ_Q}, {HIDDEN}] Symbol 4f80d1a1-86a4-4e44-94f7-909ec7089061
              'a.w.q_h' [{QK_DIM}, {HIDDEN}] Symbol 665efa60-d86c-40d5-92b2-b96d11686a8b
            'a.w.q_bias_h' [{QK_DIM}] Symbol 7d531f53-6cce-4bf3-82db-5fe8b2eef974
          'a.k_p_bias' [{SEQ_K}, {QK_DIM}] Add
            'a.k' Einsum kd,cd->kc
              'a.k.input' [{SEQ_K}, {HIDDEN}] Symbol 664bddee-28ca-47e7-9fb7-9a718de06619
              'a.w.k_h' [{QK_DIM}, {HIDDEN}] Symbol 41177709-446d-4588-b9e5-c2bbf59d53a0
            'a.w.k_bias_h' [{QK_DIM}] Symbol a891aae4-3c24-4713-afc7-8b954c6fc1b5
          'a.c.div_head_size' GeneralFunction rsqrt
            'a.c.head_size' GeneralFunction last_dim_size
              'a.c.bias_for_head_size' [{QK_DIM}] Einsum jk->j # size on this line is just asserted
                'a.w.k_h' [{QK_DIM}, {HIDDEN}]
          # mask is true at where positions *are* allowed. (This differs from old pos_mask code)
          'a.mask' [{SEQ_Q}, {SEQ_K}] Symbol ccfe5bc9-b402-42dd-a5e1-191e6fb7c268
        'a.score_neg_inf_bias' Einsum qk,->qk
          'a.not_mask' Module
            'not_mask'
            'a.mask' ! 'not_mask.input'
          'a.neg_inf' [] Scalar -10_000.0
    'a.v_p_bias' Add
      'a.v' Einsum kd,Vd->kV
        'a.v.input' [{SEQ_K}, {HIDDEN}] Symbol 8fd4c632-7f28-49ee-84cc-3dde997e0693
        'a.w.v_h' [{V_DIM}, {HIDDEN}] Symbol 79b6ebff-f9d0-411a-bcdc-530cc13e1524
      'a.w.v_bias_h' [{V_DIM}] Symbol dfb2c4ec-9378-40ee-a360-7d58f2b96954
  'a.w.o_h' [{OUT_HIDDEN}, {V_DIM}] Symbol 11a116cb-2168-4725-a06f-1b61a8ca6797

'a.q.input'
'a.k.input'
'a.v.input'
'a.mask'
'a.w.q_h'
'a.w.q_bias_h'
'a.w.k_h'
'a.w.k_bias_h'
'a.w.v_h'
'a.w.v_bias_h'
'a.w.o_h'


'a.head.on_inp' Module
  'a.head'
  'a.qk_input' Add ! 'a.q.input'
    'a.input' [{SEQ}, {HIDDEN}] Symbol f9eabd07-e2ab-4ed4-8b4a-c9c039d61835
    'a.pos_input' [{SEQ}, {HIDDEN}] Symbol eab8313e-d910-4174-8dbe-c612954eec34
  'a.qk_input' ! 'a.k.input'
  'a.input' ! 'a.v.input'

'a.input'
'a.pos_input'

'a.p_bias' Add
  'a' Einsum hsd->sd # reduce head dim
    # batch over head
    'a.heads' Module
      'a.head.on_inp'
      'a.w.q' [{HEADS}, {QK_DIM}, {HIDDEN}] Symbol cf8f9c58-1875-45b0-8007-66b66b8a405a ! 'a.w.q_h'
      'a.w.q_bias' [{HEADS}, {QK_DIM}] Symbol e896a43e-ba35-43d5-97e7-e189caf3278b ! 'a.w.q_bias_h'
      'a.w.k' [{HEADS}, {QK_DIM}, {HIDDEN}] Symbol c342f1cd-71e6-4848-86b4-1c3ffdd46753 ! 'a.w.k_h'
      'a.w.k_bias' [{HEADS}, {QK_DIM}] Symbol 13465b29-c28a-4d55-8a27-22c26cc01c69 ! 'a.w.k_bias_h'
      'a.w.v' [{HEADS}, {V_DIM}, {HIDDEN}] Symbol a90db69c-f4ad-47d6-8b76-faa4b107dacd ! 'a.w.v_h'
      'a.w.v_bias' [{HEADS}, {V_DIM}] Symbol ea64ee26-2438-4aa9-b777-0ef4f5f70e74 ! 'a.w.v_bias_h'
      'a.w.o' [{HEADS}, {OUT_HIDDEN}, {V_DIM}] Symbol 24e0e5cf-8b68-4bf2-b198-17237523b237 ! 'a.w.o_h'
  'a.w.o_bias' [{OUT_HIDDEN}] Symbol 61d54061-63e2-4a5d-9c7a-15c1ecc53b95

'a.w.q'
'a.w.q_bias'
'a.w.k'
'a.w.k_bias'
'a.w.v'
'a.w.v_bias'
'a.w.o'
'a.w.o_bias'


# We split into two pieces so we can apply rotary positional embeddings to only the first.
# Also, we use HALF_ROTARY_DIM to enforce that the rotary dim is even.
'qk_with_rot' [{SEQ}, {QK_DIM}] SetSymbolicShape
  'qk_with_rot.re' [{SEQ}, 2*2*{HALF_ROTARY_DIM}] Rearrange q s:2 e:2 rh -> q (s:2 e:2 rh)
    'qk_with_rot.cat' [{SEQ}, 2, 2, {HALF_ROTARY_DIM}] Concat 1
      'qk_with_rot.q_rot' Rearrange q e rh -> q 1 e rh
        'qk_rot' [{SEQ}, 2, {HALF_ROTARY_DIM}] GeneralFunction apply_rotary_pos_emb at interp.circuit.interop_rust.generalfuncs.rotary_pos_emb:ApplyRotaryPosEmb
          'qk_rot.inp' Index [:, 0]
            'qk_split' [{SEQ}, 2, 2, {HALF_ROTARY_DIM}] Rearrange q (s:2 e:2 rh) -> q s:2 e:2 rh
              'qk_split.set' [{SEQ}, 2*2*{HALF_ROTARY_DIM}] SetSymbolicShape
                # q or k
                'qk' [{SEQ}, {QK_DIM}] Symbol 442c7d04-1948-4904-b96a-5c0acf22f19c
      'qk_with_rot.q_pass' Index [:, 1:]
        'qk_split'

# Most of the below is copied from a.head; could factor into a module but backcompat is annoying

'a_rope.head' Einsum sV,dV->sd
  'a_rope.comb_v' Einsum qk,kV->qV
    'a_rope.attn_probs' GeneralFunction softmax
      'a_rope.attn_scores' Add
        'a_rope.attn_scores_raw' Einsum qc,kc,,qk->qk
          'a_rope.q' Module
            'qk_with_rot'
            'a.q_p_bias' ! 'qk'
          'a_rope.k' Module
            'qk_with_rot'
            'a.k_p_bias' ! 'qk'
          'a.c.div_head_size'
          'a.mask'
        'a.score_neg_inf_bias'
    'a.v_p_bias'
  'a.w.o_h'

'a_rope.head.on_inp' Module
  'a_rope.head'
  'a.qk_input' ! 'a.q.input'
  'a.qk_input' ! 'a.k.input'
  'a.input' ! 'a.v.input'

'a_rope.p_bias' Add
  'a_rope' Einsum hsd->sd # reduce head dim
    # batch over head
    'a_rope.heads' Module
      'a_rope.head.on_inp'
      'a.w.q' ! 'a.w.q_h'
      'a.w.q_bias' ! 'a.w.q_bias_h'
      'a.w.k' ! 'a.w.k_h'
      'a.w.k_bias' ! 'a.w.k_bias_h'
      'a.w.v' ! 'a.w.v_h'
      'a.w.v_bias' ! 'a.w.v_bias_h'
      'a.w.o' ! 'a.w.o_h'
  'a.w.o_bias'
"""
raw_attention_circs = add_new_circs(raw_attention_str, not_mask_circs)


def get_attention(bias: bool = False, pos: bool = False, rope: bool = False):
    circs = copy(raw_attention_circs)
    if not bias:
        for weight in ["a.w.q_bias", "a.w.k_bias", "a.w.v_bias", "a.w.o_bias"]:
            circs = zero_input(circs, weight + "_h")
            circs = zero_input(circs, weight)

    if not pos:
        circs = zero_input(circs, "a.pos_input")

    circ = "a"
    if rope:
        circ += "_rope"
    if bias:
        circ += ".p_bias"
    return ModuleLibraryRet.mk(circs, circ)


m_input = Symbol((HIDDEN,), UUID("13ec4cdc-5f25-4969-870b-0bfa2300187b"), name="m.input")
m_base_circs: Dict[str, Circuit] = {"m.input": m_input}


raw_bilinear_mlp_str = f"""
'm.p_bias' Add
  'm' Einsum h,h,oh->o
    'm.pre0' Index [0,:]
      'm.fold_pre' Rearrange (a:2 b) -> a:2 b
        'm.pre_p_bias' Add
          'm.pre' Einsum i,hi->h
            'm.input'
            'm.w.proj_in' [{BILINEAR_MLP_PROJ*2}, {HIDDEN}] Symbol c171d519-8793-4a8b-ac5e-d550347f30a6
          'm.w.in_bias' [{BILINEAR_MLP_PROJ*2}] Symbol 886b5425-cd19-4db4-871a-c46cb1a23114
    'm.pre1' Index [1,:]
      'm.fold_pre'
    'm.w.proj_out' [{OUT_HIDDEN}, {BILINEAR_MLP_PROJ}] Symbol e61637eb-9f17-4325-b2c2-5eb2518026cf
  'm.w.out_bias' [{OUT_HIDDEN}] Symbol 7efddf2e-20af-492c-a94e-1d19468f333f

'm.w.proj_in'
'm.w.in_bias'
'm.w.proj_out'
'm.w.out_bias'
"""

raw_bilinear_mlp_circs = add_new_circs(raw_bilinear_mlp_str, m_base_circs)


@functools.cache
def get_bilinear_mlp(output_bias: bool = False):
    if output_bias:
        circs = raw_bilinear_mlp_circs
    else:
        circs = zero_input(raw_bilinear_mlp_circs, "m.w.out_bias")

    return ModuleLibraryRet.mk(circs, "m.p_bias" if output_bias else "m")


@functools.cache
def get_pointwise_mlp(function_str: str = "gelu", output_bias: bool = False):
    s = f"""
    'm.p_bias' Add
      'm' Einsum h,oh->o
        'm.act' GeneralFunction {function_str}
          'm.pre' Add
            'm.pre_mul' Einsum i,hi->h
              'm.input'
              'm.w.proj_in' [{MLP_PROJ}, {HIDDEN}] Symbol 5217f963-0cdb-460e-bb1f-f82f7fbb3cd9
            'm.w.in_bias' [{MLP_PROJ}] Symbol c870ec00-8c6f-4080-907c-703ea85dde48
        'm.w.proj_out' [{OUT_HIDDEN}, {MLP_PROJ}] Symbol fdefa9af-a7d6-4a38-a7ed-5ce816c6efe7
      'm.w.out_bias' [{OUT_HIDDEN}] Symbol 113c30ba-fa88-4f34-a301-ea7912e03064

    'm.input'
    'm.w.proj_in'
    'm.w.in_bias'
    'm.w.proj_out'
    'm.w.out_bias'
    """
    circs = add_new_circs(s, m_base_circs)
    if not output_bias:
        circs = zero_input(circs, "m.w.out_bias")

    return ModuleLibraryRet.mk(circs, "m.p_bias" if output_bias else "m")


def get_norm_bind(prefix: str, uuids: List[UUID], norm_type: Literal["ln", "bn"]):
    uuids = copy(uuids)
    p = prefix
    nt = norm_type
    bind_str = f"""
    '{p}.norm' Module
      '{nt}'
      '{p}.norm.input' [{SEQ}, {HIDDEN}] Symbol {uuids.pop(4)} ! '{nt}.input'"""

    if norm_type == "bn":
        bind_str += f"""
      '{p}.{nt}.mean' [{HIDDEN}] Symbol {uuids.pop(3)} ! '{nt}.mean'
      '{p}.{nt}.var' [{HIDDEN}] Symbol {uuids.pop(2)} ! '{nt}.var'"""

    bind_str += f"""
      '{p}.{nt}.w.bias' [{HIDDEN}] Symbol {uuids.pop(1)} ! '{nt}.w.bias'
      '{p}.{nt}.w.scale' [{HIDDEN}] Symbol {uuids.pop(0)} ! '{nt}.w.scale'

    '{p}.norm.input'"""

    if norm_type == "bn":
        bind_str += f"""
    '{p}.{nt}.mean'
    '{p}.{nt}.var'"""

    bind_str += f"""
    '{p}.{nt}.w.bias'
    '{p}.{nt}.w.scale'
    """
    circs = add_new_circs(bind_str, layernorm_circs if norm_type == "ln" else batchnorm_circs)
    return ModuleLibraryRet.mk(circs, f"{p}.norm")


def print_uuid_list_repr(n: int):
    """utility for writing this code"""
    print(repr([uuid4() for _ in range(n)]))


m_norm_uuids = [
    UUID("fe330d51-0164-4b49-a504-81769d550bb1"),
    UUID("771e2d50-3414-459e-87ff-879f83d4a15c"),
    UUID("bf7ea689-c481-4ad4-89d0-0a59cd476739"),
    UUID("cf2adbd9-b04f-42af-86d7-cc5c8890607b"),
    UUID("f63f8838-9d69-4da2-a902-8eaf325f07a7"),
]
a_norm_uuids = [
    UUID("c564149d-e226-4e3d-8e47-7d6e2ceea99e"),
    UUID("2c737289-2702-404c-a22e-ad37c2652620"),
    UUID("33774cb3-f047-4a1e-ac0c-e6f8c93d2bb2"),
    UUID("981982a3-253b-4c0d-8789-c0bc9dcd229b"),
    UUID("6a622698-fd68-4d25-aeee-e8d38e68049e"),
]
final_norm_uuids = [
    UUID("2110aef7-70d5-4f77-a929-044e40c31cbd"),
    UUID("10652dc2-e9a8-4f1e-8bad-99d46804c7b3"),
    UUID("6cfcb4e6-7081-4d2d-bef9-1e9db52d4444"),
    UUID("94e8bc7f-1689-429f-bfec-67d33965aef5"),
    UUID("0851aebd-0d39-4f65-89f9-8a8afaed631a"),
]

m_ln_bind = get_norm_bind("m", m_norm_uuids, "ln")
a_ln_bind = get_norm_bind("a", a_norm_uuids, "ln")
p_ln_bind = get_norm_bind("p", a_norm_uuids, "ln")
m_bn_bind = get_norm_bind("m", m_norm_uuids, "bn")
a_bn_bind = get_norm_bind("a", a_norm_uuids, "bn")
p_bn_bind = get_norm_bind("p", a_norm_uuids, "bn")
final_ln_bind = get_norm_bind("final", final_norm_uuids, "ln")
final_bn_bind = get_norm_bind("final", final_norm_uuids, "bn")


# TODO: should this be a module?
def get_norm_call(prefix: str, circs: Dict[str, Circuit], mod: Optional[str] = None):
    norm_call_str = f"""
    '{prefix}.norm_call' Module
      '{op.unwrap(mod, prefix)}'
      '{prefix}.norm' ! '{prefix}.input'
    """

    circs = add_new_circs(norm_call_str, circs)
    return ModuleLibraryRet.mk(circs, f"{prefix}.norm_call")


b_input = Symbol((SEQ, HIDDEN), UUID("5837c4fd-f5ac-4bff-8456-abf3e95bcf36"), name="b.input")


def rename_circs_to_keys(d: Dict[str, Circuit], rename_suffix: str = ""):
    return {s: c.rename(s + rename_suffix) for s, c in d.items()}


T = TypeVar("T")


def apply_prefix(d: Dict[str, T], prefix: str) -> Dict[str, T]:
    return {f"{prefix}.{s}": c for s, c in d.items()}


def add_number(name: str, num: int):
    before, _, after = name.partition(".")
    return f"{before}{num}.{after}"


MlpActType = Literal["gelu", "gelu_new", "relu", "bilinear"]
SoftmaxType = Literal["softmax"]
PosEncType = Literal["gpt", "shortformer", "none"]


@frozen
class TransformerBlockParams:
    norm_type: Optional[Literal["ln", "bn"]] = "ln"
    attn_bias: bool = False
    attn_pos: bool = False
    use_mlp: bool = True
    mlp_act_type: MlpActType = "gelu"  # type: ignore
    mlp_output_bias: bool = False
    use_parallel_residual: bool = False  # currently implies only one shared norm
    use_rope: bool = False

    @functools.cache
    def get(self):
        # order for circs matters
        circs: Dict[str, Circuit] = {}
        if self.use_mlp:
            if self.mlp_act_type == "bilinear":
                mlp = get_bilinear_mlp(output_bias=self.mlp_output_bias)
                circs = {**circs, **mlp.circs}
            else:
                mlp = get_pointwise_mlp(function_str=self.mlp_act_type, output_bias=self.mlp_output_bias)
                circs = {**circs, **mlp.circs}
            if self.norm_type is not None:
                circs = {**circs, **(m_ln_bind if self.norm_type == "ln" else m_bn_bind).circs}
            if self.norm_type is None:
                mlp_with_norm = mlp
            else:
                mlp_with_norm = get_norm_call("m", circs, mod=mlp.name)
                circs = mlp_with_norm.circs
        if self.use_parallel_residual:
            circs = {**circs, **(p_ln_bind if self.norm_type == "ln" else p_bn_bind).circs}

        attn = get_attention(bias=self.attn_bias, pos=self.attn_pos, rope=self.use_rope)
        circs = {**circs, **attn.circs}
        if self.norm_type is not None:
            circs = {**circs, **(a_ln_bind if self.norm_type == "ln" else a_bn_bind).circs}
        if self.norm_type is None:
            attn_with_norm = attn
        else:
            attn_with_norm = get_norm_call("a", circs, mod=attn.name)
            circs = attn_with_norm.circs

        circs = {**circs, "b.input": b_input}

        attn_only_str = f"""
        'b.resid.a' Add
          # 'b.a.set' [{SEQ}, {HIDDEN}] SetSymbolicShape # not currently needed as HIDDEN == OUT_HIDDEN
          'b.a' Module
            '{attn_with_norm.name}'
            'b.input' ! 'a{'.norm' if self.norm_type is not None else ''}.input'
          'b.input'

        'b.input'
        """

        cur_resid_name = "b.resid.a"

        circs = add_new_circs(attn_only_str, circs)

        if self.use_parallel_residual:
            assert self.use_mlp and self.norm_type is not None
            block_str = f"""
          'b.add' Add
            'b.a' Module
              '{attn.name}'
              'p.input' [{SEQ}, {HIDDEN}] Symbol e76c1fd7-a230-4295-9199-17dc8dad79d0 ! 'a.input'
            'b.m' Module
              '{mlp.name}'
              'p.input' ! 'm.input'

          'p.input'
          """
            circs = add_new_circs(block_str, circs)
            circs = get_norm_call("p", circs, mod="b.add").circs
            block_str = """
          'b.resid.p' Add
            'b.p' Module
              'p.norm_call'
              'b.input' ! 'p.norm.input'
            'b.input'
          """
            circs = add_new_circs(block_str, circs)
            cur_resid_name = "b.resid.p"
        elif self.use_mlp:
            mlp_input = cur_resid_name
            block_str = f"""
            'b.resid.m' Add
              # 'b.m.set' [{SEQ}, {HIDDEN}] SetSymbolicShape # not currently needed as HIDDEN == OUT_HIDDEN
              'b.m' Module
                '{mlp_with_norm.name}'
                '{mlp_input}' ! 'm{'.norm' if self.norm_type is not None else ''}.input'
              '{cur_resid_name}'
            """

            cur_resid_name = "b.resid.m"

            circs = add_new_circs(block_str, circs)

        circs["b"] = circs[cur_resid_name].rename("b")

        return ModuleLibraryRet.mk(circs, "b")

    # add function for rebinding inputs also as needed
    @functools.cache
    def get_rebound_weighty(self, num: int):
        gotten = self.get()

        rd = random.Random()
        rd.seed(num)

        non_weighty_inputs = {"b.input", "a.pos_input", "a.mask"}

        circs = copy(gotten.circs)
        spec = make_spec(gotten.body, order=list(gotten.circs.values()), exclude=non_weighty_inputs)
        bound = Module.new_flat(
            spec,
            *[
                Symbol(
                    arg_spec.symbol.shape,
                    UUID(int=rd.getrandbits(128)),
                    add_number(arg_spec.symbol.name, num),
                )
                for arg_spec in spec.arg_specs
            ],
            name=f"b{num}",
        )

        assert bound.name not in circs
        circs[bound.name] = bound

        return ModuleLibraryRet.mk(circs, bound.name)

    def garbage_init_norm_weights(self, hidden_size: int, device_dtype: TorchDeviceDtypeOp = TorchDeviceDtypeOp()):
        """be warned this is not random init, it's GARBAGE, it won't train or behave properly at all"""

        weights: Dict[str, Circuit] = {}
        if self.norm_type is not None:
            if self.norm_type == "bn":
                weights.update(
                    {
                        f"{self.norm_type}.mean": Array.randn(hidden_size, device_dtype=device_dtype),
                        f"{self.norm_type}.var": Array(torch.randn(hidden_size).to(device=device_dtype.device) ** 2),
                    }
                )
            weights.update(
                {
                    f"{self.norm_type}.w.bias": Array.randn(hidden_size, device_dtype=device_dtype),
                    f"{self.norm_type}.w.scale": Array.randn(hidden_size, device_dtype=device_dtype),
                }
            )

        return rename_circs_to_keys(weights, rename_suffix="_arr")

    def garbage_init_weights(
        self, hidden_size: int, head_size: int, num_heads: int, device_dtype: TorchDeviceDtypeOp = TorchDeviceDtypeOp()
    ):
        """be warned this is not random init, it's GARBAGE, it won't train or behave properly at all"""

        weights: Dict[str, Circuit] = {}
        weights.update(
            apply_prefix(self.garbage_init_norm_weights(hidden_size=hidden_size, device_dtype=device_dtype), "a")
        )

        # if self.block_spec.attn_pos:
        #     weights.update({"a.pos_input": Array.randn(*batch_shape, seq_len, hidden_size)})

        weights.update(
            {
                # "a.mask": Array.randn(seq_len, seq_len, device_dtype=device_dtype),
                "a.w.q": Array.randn(num_heads, head_size, hidden_size, device_dtype=device_dtype),
                "a.w.k": Array.randn(num_heads, head_size, hidden_size, device_dtype=device_dtype),
                "a.w.v": Array.randn(num_heads, head_size, hidden_size, device_dtype=device_dtype),
                "a.w.o": Array.randn(num_heads, hidden_size, head_size, device_dtype=device_dtype),
            }
        )

        if self.attn_bias:
            weights.update(
                {
                    "a.w.q_bias": Array.randn(num_heads, head_size, device_dtype=device_dtype),
                    "a.w.k_bias": Array.randn(num_heads, head_size, device_dtype=device_dtype),
                    "a.w.v_bias": Array.randn(num_heads, head_size, device_dtype=device_dtype),
                    "a.w.o_bias": Array.randn(hidden_size, device_dtype=device_dtype),
                }
            )

        if self.use_mlp:
            weights.update(
                apply_prefix(self.garbage_init_norm_weights(hidden_size=hidden_size, device_dtype=device_dtype), "m")
            )

            weights.update(
                {
                    "m.w.proj_in": Array.randn(hidden_size * 4, hidden_size, device_dtype=device_dtype),
                    "m.w.in_bias": Array.randn(hidden_size * 4, device_dtype=device_dtype),
                    "m.w.proj_out": Array.randn(
                        hidden_size,
                        hidden_size * (2 if self.mlp_act_type == "bilinear" else 4),
                        device_dtype=device_dtype,
                    ),
                }
            )

            if self.mlp_output_bias:
                weights.update({"m.w.out_bias": Array.randn(hidden_size, device_dtype=device_dtype)})

        return rename_circs_to_keys(weights, "_arr")


# aka cross entropy, aka log loss
log_likelyhood_str = f"""
'll' GeneralFunction gen_index_at_0_batch_x_c
  'log_probs' GeneralFunction log_softmax
    'll.input' [{LOG_LIKELYHOOD_CLASSES}] Symbol b9d111e5-b793-4f63-84a0-2f8590b9f39c
  'll.label' [] Symbol 74c503b9-fe2c-4f8b-9350-7351a292c351

'll.input'
'll.label'

'nll' Einsum ,->
  'll'
  'nll.neg' [] Scalar -1
"""

log_likelyhood_circs = add_new_circs(log_likelyhood_str, {})
log_likelyhood = ModuleLibraryRet.mk(log_likelyhood_circs, "ll")
negative_log_likelyhood = ModuleLibraryRet.mk(log_likelyhood_circs, "nll")


# TODO: maybe add more bindings on top of this
@frozen
class TransformerParams:
    block_params: TransformerBlockParams = TransformerBlockParams()
    num_layers: int = 2
    use_norm_output: bool = True
    output_bias: bool = False

    @property
    def norm_type(self):
        return self.block_params.norm_type

    @functools.cache
    def get(self):
        blocks = [self.block_params.get_rebound_weighty(i) for i in range(self.num_layers)]
        # keeps right ordering
        circs = dict(x for b in blocks for x in b.circs.items())

        out: Circuit = Symbol((SEQ, HIDDEN), UUID("ece2bb5d-c6d6-4b7a-93a5-32b5ac264888"), "t.input")
        assert "t.input" not in circs
        circs["t.input"] = out
        for i, block in enumerate(blocks):
            out = module_new_bind(block.body, ("b.input", out), name=f"b{i}.call")
            assert out.shape == (SEQ, HIDDEN)
            circs[out.name] = out

        if self.norm_type is not None and self.use_norm_output:
            circs = {**circs, **(final_ln_bind if self.norm_type == "ln" else final_bn_bind).circs}
            final_norm_str = f"""
            'final.call' Module
              'final.norm'
              '{out.name}' ! 'final.norm.input'
            """
            circs = add_new_circs(final_norm_str, circs)
            out = circs["final.call"]

        logits_str = f"""
        't.logits' Einsum sh,vh->sv
          '{out.name}'
          't.w.unembed' [{VOCAB_SIZE}, {HIDDEN}] Symbol 85d5a05a-ef9e-4910-a967-3f27951f67cf
        't.logits_p_bias' Add
          't.logits'
          't.w.unembed_bias' [{VOCAB_SIZE}] Symbol 62d9f91a-5df2-4eb8-af75-5bdfb15eb974

        't.w.unembed'
        't.w.unembed_bias'
        """
        circs = add_new_circs(logits_str, circs)
        if not self.output_bias:
            circs = {s: c for s, c in circs.items() if s not in ["t.w.unembed_bias", "t.logits_p_bias"]}

        return ModuleLibraryRet.mk(circs, name="t.logits" + ("_p_bias" if self.output_bias else ""))

    def garbage_init_weights(
        self,
        hidden_size: int = 3,
        head_size: int = 5,
        num_heads: int = 7,
        vocab_size: int = 11,
        device_dtype: TorchDeviceDtypeOp = TorchDeviceDtypeOp(),
    ):
        """be warned this is not random init, it's GARBAGE, it won't train or behave properly at all"""
        weights: Dict[str, Circuit] = {}
        for i in range(self.num_layers):
            weights.update(
                {
                    add_number(s, i): c
                    for s, c in self.block_params.garbage_init_weights(
                        hidden_size=hidden_size, head_size=head_size, num_heads=num_heads, device_dtype=device_dtype
                    ).items()
                }
            )
        if self.use_norm_output:
            weights.update(
                apply_prefix(
                    self.block_params.garbage_init_norm_weights(hidden_size=hidden_size, device_dtype=device_dtype),
                    "final",
                )
            )

        weights["t.w.unembed"] = Array.randn(vocab_size, hidden_size, device_dtype=device_dtype)

        if self.output_bias:
            weights["t.w.unembed_bias"] = Array.randn(vocab_size, device_dtype=device_dtype)

        return rename_circs_to_keys(weights, rename_suffix="_arr")

    def garbage_call(
        self,
        hidden_size: int = 3,
        head_size: int = 5,
        num_heads: int = 7,
        seq_len: int = 9,
        vocab_size: int = 11,
        batch_shape: Shape = (13, 2),
        device_dtype: TorchDeviceDtypeOp = TorchDeviceDtypeOp(),
    ):
        """be warned this is not random init, it's GARBAGE, it won't train or behave properly at all"""
        weights = self.garbage_init_weights(
            hidden_size=hidden_size,
            head_size=head_size,
            num_heads=num_heads,
            vocab_size=vocab_size,
            device_dtype=device_dtype,
        )

        # this could be made faster if we needed
        # seems fine for now
        bound_weights = module_new_bind(self.get().body, *weights.items(), name="t.bind_w")
        inputs = rename_circs_to_keys(
            {
                "t.input": Array.randn(*batch_shape, seq_len, hidden_size),
                "a.mask": Array.randn(seq_len, seq_len, device_dtype=device_dtype),
                **(
                    {"a.pos_input": Array.randn(*batch_shape, seq_len, hidden_size)}
                    if self.block_params.attn_pos
                    else {}
                ),
            },
            rename_suffix="_rand_inp",
        )
        return module_new_bind(bound_weights, *inputs.items(), name="t.call"), weights, inputs


@frozen
class TransformerInfo:
    params: TransformerParams
    model_class: str = "GPTBeginEndToks"
    # maybe below 2 shouldn't be optional?
    pos_enc_type: Optional[PosEncType] = "gpt"  # type: ignore
    causal_mask: Optional[bool] = True
    extra: Optional[Any] = None

    def dump_model_string(self, *circs: Circuit):
        info_json = make_json_converter().dumps(self)
        assert "\n" not in info_json
        prefix_str = f"# info:{info_json}\n"
        rep = PrintOptions(reference_circuits={c: s for s, c in self.params.get().circs.items()}).repr(
            *circs,
        )

        return prefix_str + rep + "\n"

    def bind_to_input(
        self,
        c: Circuit,
        inp_tok_embeds: Circuit,
        pos_embed_weights: Optional[Circuit] = None,
        inp_mask: Optional[Circuit] = None,
        inp_mask_has_q: bool = False,
        prefix: str = "t",
    ):
        seq_len = inp_tok_embeds.shape[-2]

        if pos_embed_weights is None:
            pos_embeds = None
        else:
            pos_embeds = pos_embed_weights.index(
                I[
                    None:seq_len,
                ],
                name="t.w.pos_embeds_idxed",
            )
        inp: Circuit
        if self.pos_enc_type == "gpt":
            assert pos_embeds is not None
            inp = Add(inp_tok_embeds, pos_embeds, name=f"{prefix}.inp_tok_pos")
        else:
            inp = inp_tok_embeds

        mask: Circuit
        if self.causal_mask:
            mask = Array(
                (torch.arange(seq_len)[:, None] >= torch.arange(seq_len)[None, :]).to(
                    device=inp_tok_embeds.device, dtype=inp_tok_embeds.torch_dtype or torch.float32
                ),
                "t.a.c.causal_mask",  # prefix doesn't apply to constants
            )
        else:
            mask = Scalar(1.0, shape=(seq_len, seq_len), name="t.a.c.full_mask")

        if inp_mask is not None:

            class DimNumMaker:
                num: int = 0

                def __call__(self, n: int):
                    out = tuple(range(self.num, self.num + n))
                    self.num += n
                    return out

            m = DimNumMaker()
            [k] = m(1)
            [q] = m(1)
            batch = m(inp_mask.ndim - 1 - (1 if inp_mask_has_q else 0))
            # TODO: simplify out 1 in this case!
            mask = Einsum(
                (inp_mask, batch + ((q, k) if inp_mask_has_q else (k,))),
                (mask, (q, k)),
                out_axes=batch + (q, k),
                name=f"{prefix}.a.mask",
            )

        pairs = [("t.input", inp), ("a.mask", mask)]
        if self.pos_enc_type == "shortformer":
            assert pos_embeds is not None
            pairs.append(("a.pos_input", pos_embeds))

        return module_new_bind(c, *pairs, name=f"{prefix}.call")

    def bind_to_input_tokens(
        self,
        c: Circuit,
        toks: Circuit,
        tok_embed_weights: Circuit,
        pos_embed_weights: Circuit,
        inp_mask: Optional[Circuit] = None,
        inp_mask_has_q: bool = False,
        prefix: str = "t",
    ):
        embedded = GeneralFunction.gen_index(tok_embed_weights, toks, -2)
        return self.bind_to_input(c, embedded, pos_embed_weights, inp_mask, inp_mask_has_q, prefix)

    def bind_to_input_tokens_int16(
        self,
        c: Circuit,
        toks: torch.Tensor,
        tok_embed_weights: Circuit,
        pos_embed_weights: Circuit,
        inp_mask: Optional[Circuit] = None,
        inp_mask_has_q: bool = False,
        prefix: str = "t",
    ):
        tok_array = Array(toks)
        upcasted = Module(token_upcast_module, "toks_upcasted", **{"upcast_toks.int16_toks": tok_array})
        return self.bind_to_input_tokens(
            c, upcasted, tok_embed_weights, pos_embed_weights, inp_mask, inp_mask_has_q, prefix
        )


def load_transformer_model_string(
    s: str, parser: Optional[Parser] = None
) -> tuple[dict[str, Circuit], Any, TransformerInfo]:
    s = s.strip("\n")
    assert s.startswith("# info:")
    info_s, _, _ = s.partition("\n")

    info = cattrs.structure(json.loads(info_s.removeprefix("# info:")), TransformerInfo)

    circs = info.params.get().circs
    added = add_new_circs(s, circs, parser=parser)
    from interp.model.model_loading import MODEL_CLASS_STR_TO_MODEL_AND_TOKENIZER_FNS

    tokenizer_fn = MODEL_CLASS_STR_TO_MODEL_AND_TOKENIZER_FNS[info.model_class][1]

    return (added, tokenizer_fn(), info)


def get_model_path(model_id: str):
    # we should plausibly eventually remove current circ_models/ and replace with circ_models2/
    from interp.tools.rrfs import RRFS_DIR

    return f"{RRFS_DIR}/circ_models2/{model_id}.circ"


def load_model_id(model_id: str, parser: Optional[Parser] = None) -> tuple[dict[str, Circuit], Any, TransformerInfo]:
    """Returns (circs, tokenizer, info)"""
    with open(get_model_path(model_id)) as f:
        return load_transformer_model_string(f.read(), parser=parser)


token_upcast_module = make_spec(
    P(
        """
'upcast_toks' GeneralFunction cast_from_{device:None,dtype:None}_to_{device:None,dtype:int64}
  'upcast_toks.sub' Add
    'upcast_toks.int16_toks' [] Symbol
    'upcast_toks.signed_int16_min' [] Scalar 32768 # int16 min, torch doesn't support uint16 :(
"""
    ),
    [P("'upcast_toks.int16_toks' [] Symbol")],
)
print(token_upcast_module)
