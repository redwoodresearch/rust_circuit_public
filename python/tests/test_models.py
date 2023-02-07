from typing import Any, Dict, Optional, cast

import einops
import torch
from attrs import frozen

import rust_circuit as rc
import rust_circuit.module_library as mod_l
import rust_circuit.optional as op
from interp.circuit.circuit import Circuit, MemoizedFn
from interp.circuit.circuit_models import attention, batchnorm, bilinear_mlp, layernorm, pointwise_nonlin_mlp
from interp.circuit.circuit_utils import evaluate_fn
from interp.circuit.constant import ArrayConstant
from interp.model.blocks import BatchNormFixed, LayerNorm
from interp.model.gpt_modules import Attention, GptBlock, MlpActType


def evaluate_circ(c: Circuit) -> torch.Tensor:
    return MemoizedFn(evaluate_fn(dtype=torch.float64))(c)


@frozen
class FakeLinear:
    weights: torch.Tensor
    bias: Optional[torch.Tensor]

    @property
    def use_bias(self) -> bool:
        return self.bias is not None

    def get_weights(self) -> torch.Tensor:
        return self.weights

    def get_bias(self) -> torch.Tensor:
        return op.unwrap(self.bias)


@frozen
class FakeGptBlock:
    linear1: FakeLinear
    linear2: FakeLinear
    mlp_act_type: MlpActType = "bilinear"


def raw_test_bilinear_mlp(hidden: int, out_hidden: int, bilinear_mlp_proj: int, batch: rc.Shape, output_bias: bool):
    proj_in = torch.randn(bilinear_mlp_proj * 2, hidden, dtype=torch.float64)
    in_bias = torch.randn(bilinear_mlp_proj * 2, dtype=torch.float64)
    proj_out = torch.randn(out_hidden, bilinear_mlp_proj, dtype=torch.float64)
    out_bias = torch.randn(out_hidden, dtype=torch.float64)
    inp = torch.randn(*batch, hidden, dtype=torch.float64)
    dummy_proj_out = torch.cat([proj_out, torch.randn(proj_out.shape, dtype=torch.float64)], dim=-1)
    b = FakeGptBlock(FakeLinear(proj_in, in_bias), FakeLinear(dummy_proj_out, out_bias if output_bias else None))
    old_circ = bilinear_mlp(cast(GptBlock, b), ArrayConstant(inp.reshape(1, -1, hidden)))

    mod = rc.Module(
        mod_l.get_bilinear_mlp(output_bias=output_bias).spec,
        **{
            "m.w.proj_in": rc.Array(proj_in),
            "m.w.in_bias": rc.Array(in_bias),
            "m.w.proj_out": rc.Array(proj_out),
            "m.input": rc.Array(inp),
        },
        **({"m.w.out_bias": rc.Array(out_bias)} if output_bias else {}),
        name="",
    )

    torch.testing.assert_close(mod.evaluate(), evaluate_circ(old_circ).reshape(*batch, out_hidden))


def test_bilinear_mlp():
    raw_test_bilinear_mlp(2, 2, 3, (), True)
    raw_test_bilinear_mlp(2, 2, 3, (), False)
    raw_test_bilinear_mlp(2, 5, 3, (), False)
    raw_test_bilinear_mlp(2, 5, 3, (1,), False)
    raw_test_bilinear_mlp(2, 5, 3, (7,), False)
    raw_test_bilinear_mlp(2, 5, 3, (7,), True)
    raw_test_bilinear_mlp(2, 5, 3, (7, 3), True)


def raw_test_pointwise_mlp(
    hidden: int, out_hidden: int, mlp_proj: int, batch: rc.Shape, output_bias: bool, mlp_act_type: MlpActType
):
    proj_in = torch.randn(mlp_proj, hidden, dtype=torch.float64)
    in_bias = torch.randn(mlp_proj, dtype=torch.float64)
    proj_out = torch.randn(out_hidden, mlp_proj, dtype=torch.float64)
    out_bias = torch.randn(out_hidden, dtype=torch.float64)
    inp = torch.randn(*batch, hidden, dtype=torch.float64)

    b = FakeGptBlock(
        FakeLinear(proj_in, in_bias), FakeLinear(proj_out, out_bias if output_bias else None), mlp_act_type=mlp_act_type
    )
    old_circ = pointwise_nonlin_mlp(cast(GptBlock, b), ArrayConstant(inp.reshape(1, -1, hidden)))

    mod = rc.Module(
        mod_l.get_pointwise_mlp(function_str=mlp_act_type, output_bias=output_bias).spec,
        **{
            "m.w.proj_in": rc.Array(proj_in),
            "m.w.in_bias": rc.Array(in_bias),
            "m.w.proj_out": rc.Array(proj_out),
            "m.input": rc.Array(inp),
        },
        **({"m.w.out_bias": rc.Array(out_bias)} if output_bias else {}),
        name="",
    )

    torch.testing.assert_close(mod.evaluate(), evaluate_circ(old_circ).reshape(*batch, out_hidden))


def test_pointwise_mlp():
    raw_test_pointwise_mlp(2, 2, 3, (), True, "gelu")
    raw_test_pointwise_mlp(2, 2, 3, (), False, "gelu")
    raw_test_pointwise_mlp(2, 5, 3, (), False, "gelu")
    raw_test_pointwise_mlp(2, 5, 3, (1,), False, "gelu")
    raw_test_pointwise_mlp(2, 5, 3, (7,), False, "gelu")
    raw_test_pointwise_mlp(2, 5, 3, (7,), True, "gelu")
    raw_test_pointwise_mlp(2, 5, 3, (7, 3), True, "gelu")
    raw_test_pointwise_mlp(2, 2, 3, (), True, "relu")
    raw_test_pointwise_mlp(2, 2, 3, (), False, "relu")
    raw_test_pointwise_mlp(2, 5, 3, (7, 3), True, "relu")


@frozen
class FakeAttention:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    o: torch.Tensor
    q_bias: Optional[torch.Tensor]
    k_bias: Optional[torch.Tensor]
    v_bias: Optional[torch.Tensor]
    o_bias: Optional[torch.Tensor]

    @property
    def bias(self) -> bool:
        return self.q_bias is not None

    @property
    def hidden_size(self) -> int:
        return self.q.shape[-1]

    @property
    def num_heads(self) -> int:
        return self.q.shape[0]

    def get_qkv_mats(self):
        return self.q, self.k, self.v

    @property
    def attn_weights(self) -> FakeLinear:
        bias: Optional[torch.Tensor] = None
        if self.q_bias is not None:
            bias = torch.cat(
                [op.unwrap(self.q_bias).flatten(), op.unwrap(self.k_bias).flatten(), op.unwrap(self.v_bias).flatten()],
                dim=0,
            )
        return FakeLinear(
            torch.cat(
                [
                    self.q.reshape(-1, self.hidden_size),
                    self.k.reshape(-1, self.hidden_size),
                    self.v.reshape(-1, self.hidden_size),
                ],
                dim=0,
            ),
            bias,
        )

    @property
    def project_output(self) -> FakeLinear:
        return FakeLinear(einops.rearrange(self.o, "num_heads out v -> out (num_heads v)"), self.o_bias)

    @property
    def causal_mask(self) -> bool:
        return False

    @property
    def dtype(self):
        return torch.float64


def raw_test_attention(
    hidden: int,
    qkv_dim: int,
    num_heads: int,
    out_hidden: int,
    seq_len: int,
    batch: rc.Shape,
    use_bias: bool,
    use_pos: bool,
):
    qk_dim = v_dim = qkv_dim  # coverage isn't great because we can't test different shape case
    q = torch.randn(num_heads, qk_dim, hidden, dtype=torch.float64)
    q_bias = torch.randn(num_heads, qk_dim, dtype=torch.float64)
    k = torch.randn(num_heads, qk_dim, hidden, dtype=torch.float64)
    k_bias = torch.randn(num_heads, qk_dim, dtype=torch.float64)
    v = torch.randn(num_heads, v_dim, hidden, dtype=torch.float64)
    v_bias = torch.randn(num_heads, v_dim, dtype=torch.float64)
    o = torch.randn(num_heads, out_hidden, v_dim, dtype=torch.float64)
    o_bias = torch.randn(out_hidden, dtype=torch.float64)
    inp = torch.randn(*batch, seq_len, hidden, dtype=torch.float64)
    pos_inp = torch.randn(*batch, seq_len, hidden, dtype=torch.float64)

    extra_args: Dict[str, rc.Circuit] = {}

    if not use_bias:
        q_bias_v: Optional[torch.Tensor] = None
        k_bias_v: Optional[torch.Tensor] = None
        v_bias_v: Optional[torch.Tensor] = None
        o_bias_v: Optional[torch.Tensor] = None
    else:
        extra_args = {
            "a.w.q_bias": rc.Array(q_bias * (2 if use_pos else 1)),
            "a.w.k_bias": rc.Array(k_bias * (2 if use_pos else 1)),
            "a.w.v_bias": rc.Array(v_bias),
            "a.w.o_bias": rc.Array(o_bias),
        }

        q_bias_v = q_bias
        k_bias_v = k_bias
        v_bias_v = v_bias
        o_bias_v = o_bias

    if use_pos:
        extra_args["a.pos_input"] = rc.Array(pos_inp)

    mask = torch.bernoulli(torch.full((*batch, seq_len, seq_len), 0.5)).double()

    old_circ = attention(
        cast(Attention, FakeAttention(q, k, v, o, q_bias_v, k_bias_v, v_bias_v, o_bias_v)),
        ArrayConstant(inp.reshape(-1, seq_len, hidden)),
        position_embedding=ArrayConstant(pos_inp.reshape(-1, seq_len, hidden)) if use_pos else None,
        # for some reason, this is negated relative to what you would think...
        pos_mask=ArrayConstant(1.0 - mask.reshape(-1, seq_len, seq_len), name="inp_mask"),
    )

    args: dict[str, rc.Circuit] = {
        "a.w.q": rc.Array(q),
        "a.w.k": rc.Array(k),
        "a.w.v": rc.Array(v),
        "a.w.o": rc.Array(o),
        "a.mask": rc.Array(mask),
        "a.input": rc.Array(inp),
        **extra_args,
    }
    args = {s: c.rename("arr." + s) for s, c in args.items()}

    mod = rc.Module(
        mod_l.get_attention(bias=use_bias, pos=use_pos).spec,
        **args,
        name="",
    )

    torch.testing.assert_close(mod.evaluate(), evaluate_circ(old_circ).reshape(*batch, seq_len, out_hidden))


def test_attention():
    torch.manual_seed(3854)
    raw_test_attention(1, 1, 1, 1, 2, (), use_bias=False, use_pos=False)
    raw_test_attention(1, 1, 2, 1, 3, (), use_bias=False, use_pos=False)
    raw_test_attention(2, 3, 5, 7, 11, (), use_bias=False, use_pos=False)
    raw_test_attention(2, 3, 5, 7, 11, (1,), use_bias=False, use_pos=False)
    raw_test_attention(2, 3, 5, 7, 11, (3,), use_bias=False, use_pos=False)
    raw_test_attention(2, 3, 5, 7, 11, (4, 3, 2), use_bias=False, use_pos=False)
    raw_test_attention(2, 3, 5, 7, 11, (3,), use_bias=True, use_pos=False)
    raw_test_attention(2, 3, 5, 7, 11, (3,), use_bias=False, use_pos=True)
    raw_test_attention(2, 3, 5, 7, 11, (3,), use_bias=True, use_pos=True)
    raw_test_attention(2, 3, 5, 7, 11, (), use_bias=True, use_pos=True)
    raw_test_attention(2, 3, 5, 7, 11, (4, 3), use_bias=True, use_pos=True)


@frozen
class FakeNorm:
    variables: Dict[str, Any]
    epsilon: float


def raw_test_layernorm(hidden: int, batch: rc.Shape):
    scale = torch.randn(hidden, dtype=torch.float64)
    bias = torch.randn(hidden, dtype=torch.float64)
    inp = torch.randn(*batch, hidden, dtype=torch.float64)

    variables = {"params": {"scale": scale, "bias": bias}}
    norm_b = FakeNorm(variables, epsilon=1e-5)
    old_circ = layernorm(cast(LayerNorm, norm_b), ArrayConstant(inp.reshape(1, -1, hidden)))

    mod = rc.Module(
        mod_l.layernorm.spec,
        **{"ln.w.bias": rc.Array(bias), "ln.w.scale": rc.Array(scale), "ln.input": rc.Array(inp)},
        name="",
    )

    torch.testing.assert_close(mod.evaluate(), evaluate_circ(old_circ).reshape(*batch, hidden))


def test_layernorm():
    raw_test_layernorm(2, ())
    raw_test_layernorm(2, (3, 4))
    raw_test_layernorm(2, (3, 4))
    raw_test_layernorm(2, (3, 4, 7))
    raw_test_layernorm(9, (3, 4, 7))


def raw_test_batchnorm(hidden: int, batch: rc.Shape):
    scale = torch.randn(hidden, dtype=torch.float64)
    bias = torch.randn(hidden, dtype=torch.float64)
    mean = torch.randn(hidden, dtype=torch.float64)
    var = torch.randn(hidden, dtype=torch.float64) ** 2
    inp = torch.randn(*batch, hidden, dtype=torch.float64)

    variables = dict(params=dict(scale=scale, bias=bias, mean=mean, var=var))
    norm_b = FakeNorm(variables, epsilon=1e-5)
    old_circ = batchnorm(cast(BatchNormFixed, norm_b), ArrayConstant(inp.reshape(1, -1, hidden)))

    mod = rc.Module(
        mod_l.batchnorm.spec,
        **{
            "bn.w.bias": rc.Array(bias),
            "bn.w.scale": rc.Array(scale),
            "bn.var": rc.Array(var),
            "bn.mean": rc.Array(mean),
            "bn.input": rc.Array(inp),
        },
        name="",
    )

    torch.testing.assert_close(mod.evaluate(), evaluate_circ(old_circ).reshape(*batch, hidden))


def test_batchnorm():
    raw_test_batchnorm(2, ())
    raw_test_batchnorm(2, (3, 4))
    raw_test_batchnorm(2, (3, 4))
    raw_test_batchnorm(2, (3, 4, 7))
    raw_test_batchnorm(9, (3, 4, 7))


def raw_test_nll(classes: int, batch: rc.Shape):
    inp = torch.randn(*batch, classes)
    labels = torch.randint(0, classes, batch).long()
    expected = torch.nn.functional.cross_entropy(
        inp.reshape(-1, classes), labels.flatten().long(), reduction="none"
    ).reshape(batch)
    actual = rc.Module(
        mod_l.negative_log_likelyhood.spec,
        **{
            "ll.input": rc.Array(inp),
            "ll.label": rc.Array(labels),
        },
        name="",
    )

    torch.testing.assert_close(actual.evaluate(), expected)


def test_nll():
    raw_test_nll(1, ())
    raw_test_nll(2, ())
    raw_test_nll(2, (3, 4))
    raw_test_nll(2, (3, 4))
    raw_test_nll(2, (3, 4, 7))
    raw_test_nll(9, (3, 4, 7))
    raw_test_nll(1, (3, 4, 7))
