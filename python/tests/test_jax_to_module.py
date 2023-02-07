from typing import Optional

import jax
import jax.numpy as jnp
import pytest
import torch

import rust_circuit as rc
import rust_circuit.optional as op
from interp.circuit.circuit_models import get_trained_gpt
from interp.circuit.circuit_utils import cast_circuit
from interp.circuit.constant import ArrayConstant
from interp.circuit.get_update_node import FunctionIterativeNodeMatcher, NameMatcher, NodeGetter
from interp.circuit.test_circuit_models import CONFIGS, GptTestConfig, get_gpt_test
from interp.model.gpt_model import Gpt, module_config_dict
from interp.tools.data_loading import get_val_seqs
from interp.tools.variable_dict import variable_dict_map
from rust_circuit.jax_to_module import from_converted, get_bound_model
from rust_circuit.module_library import get_model_path, load_model_id

from .test_models import evaluate_circ


def pos_mask_to_inp_mask(pos_mask: Optional[torch.Tensor]):
    return op.map(pos_mask, lambda pos_mask: rc.Array(1 - pos_mask, name="a.pos_mask"))


@pytest.mark.parametrize("config", CONFIGS)
def test_model_gpt(
    config: GptTestConfig,
    seed=2833333,
    head_size=5,
    seq_len=3,
    num_heads=4,
    batch_size=2,
    num_layers=2,
    vocab_size=27,
):
    jax.config.update("jax_enable_x64", True)
    # TODO: we could maybe cache identical tests instead of removing so we don't unintentionally remove coverage
    if (
        config.use_fits
        or config.norm_type not in {"none", "layer_norm", "batch_norm_fixed"}
        or not config.use_batch
        or config.output_type != "probs"  # we fix output type, so varying this from default does nothing
        or config.use_onehot_tokens
        or config.softmax_type != "softmax"
    ):
        return

    config.output_type = "logits"

    circuit, _, model, params, _, pos_mask = get_gpt_test(
        key=jax.random.PRNGKey(seed),
        head_size=head_size,
        seq_len=seq_len,
        num_heads=num_heads,
        batch_size=batch_size,
        num_layers=num_layers,
        vocab_size=vocab_size,
        config=config,
    )
    circuit = cast_circuit(circuit, dtype=torch.float64)
    model = Gpt(**{**module_config_dict(model), **dict(dtype=jnp.float64)})
    params = variable_dict_map(params, lambda x: x.astype(jnp.float64))

    def get(s: str):
        return NodeGetter(FunctionIterativeNodeMatcher(NameMatcher(s))).get_unique_c(circuit)

    expected_val = evaluate_circ(circuit)
    assert expected_val.shape == circuit.shape

    def get_arr(s: str):
        return ArrayConstant.unwrap(get(s))

    model_b = model.bind(params)
    new_circ, (_, pos_embed_weights), info, _ = get_bound_model(model_b)
    if pos_mask is not None:
        pos_mask_v = from_converted(pos_mask).value
    else:
        pos_mask_v = None
    new_circ_bound = info.bind_to_input(
        new_circ,
        inp_tok_embeds=rc.Array(get_arr("tok_embeds").value.double(), name="t.inp.tok_embeds"),
        pos_embed_weights=pos_embed_weights,
        inp_mask=pos_mask_to_inp_mask(pos_mask_v),
    )
    actual_val = new_circ_bound.evaluate()
    assert actual_val.shape == new_circ_bound.shape

    torch.testing.assert_close(actual_val, expected_val, rtol=1e-5, atol=1e-5)

    jax.config.update("jax_enable_x64", False)


def raw_test_load_circ_model(model_id: str, seq_len: int = 64, batch_size: int = 4):
    s = open(get_model_path(model_id)).read()
    ls = [x.removeprefix("# originally ") for x in s.split("\n") if x.startswith("# originally ")]
    assert len(ls) == 1
    old_model_id = ls[0].split(" ")[0]

    toks = torch.tensor(get_val_seqs(train=True, n_files=1, files_start=0, max_size=seq_len + 1)[:batch_size, :seq_len])

    new_circs, _, info = load_model_id(model_id)
    new_circ = new_circs["t.bind_w"]
    device_dtype = rc.TorchDeviceDtypeOp(dtype="float64", device="cpu")
    new_circ = rc.cast_circuit(new_circ, device_dtype)
    pos_embeds = rc.cast_circuit(new_circs["t.w.pos_embeds"], device_dtype)

    inp_tok_embeds = new_circs["t.w.tok_embeds"].cast_array().value.double().cpu()[toks]
    new_circ = info.bind_to_input(new_circ, rc.Array(inp_tok_embeds, name="t.inp.tok_embeds"), pos_embeds)
    assert new_circ.is_explicitly_computable

    circ, _, _, _, _ = get_trained_gpt(
        toks,
        model_id=old_model_id,
        use_poly_fit=False,
        output_type="logits",
        use_batch=True,
    )
    circ = cast_circuit(circ, dtype=torch.float64)

    torch.testing.assert_close(new_circ.evaluate(), evaluate_circ(circ), atol=1e-5, rtol=1e-5)


@pytest.mark.skip
@pytest.mark.parametrize(
    "model_id",
    [
        "attention_only_bn_2",
        "attention_only_bn_4",
        "attention_only_2",
        "bilinear_bn_2",
        "bilinear_bn_4",
        "bilinear_2",
        "bilinear_4",
        "gelu_1",
        "gelu_2",
    ],
)
def test_load_circ_model(model_id: str):
    raw_test_load_circ_model(model_id, seq_len=16, batch_size=3)


@pytest.mark.skip
@pytest.mark.parametrize("model_id", ["gelu_12", "gelu_24"])
def test_load_circ_model_big(model_id: str):
    raw_test_load_circ_model(model_id, seq_len=4, batch_size=2)
