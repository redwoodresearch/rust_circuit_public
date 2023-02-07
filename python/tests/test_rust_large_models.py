import pytest
import torch

import rust_circuit as rc
import rust_circuit.module_library as mod_l
from rust_circuit import (
    OptimizationSettings,
    TorchDeviceDtypeOp,
    batch_to_concat,
    cast_circuit,
    optimize_and_evaluate,
    substitute_all_modules,
)
from rust_circuit.py_utils import timed


@pytest.mark.cuda
@pytest.mark.parametrize(
    "config",
    [
        {"num_layers": 6, "head_size": 128, "num_heads": 4},
        {"num_layers": 12, "head_size": 64, "num_heads": 12},
        {"num_layers": 16, "head_size": 64, "num_heads": 16},
    ],
)
@pytest.mark.skip("model generation too slow")
def test_big_models_cuda(config):
    raw_test_big_models(config, device="cuda")


@pytest.mark.parametrize("config", [{"num_layers": 4, "head_size": 5, "num_heads": 3}])
def test_big_models(config):
    raw_test_big_models(config, device="cpu")


@pytest.mark.cuda
def test_big_wide():
    raw_test_big_models({"num_layers": 6, "head_size": 128, "num_heads": 8}, device="cuda", batch_size=200)


@pytest.mark.cuda
def test_tall_tiny_batch():
    raw_test_big_models(
        {"num_layers": 12, "head_size": 2, "num_heads": 2},
        device="cuda",
        batch_size=64,
        concat_batch_size=64,
        seq_len=2,
    )


@pytest.mark.cuda
def test_big_batch():
    raw_test_big_models(
        {"num_layers": 4, "head_size": 128, "num_heads": 8}, device="cuda", batch_size=32, concat_batch_size=32
    )


def raw_test_big_models(config, device="cpu", batch_size=1_000, concat_batch_size=None, seq_len=128):
    with timed("model creation"):
        y_rust, _, _ = mod_l.TransformerParams(
            mod_l.TransformerBlockParams(mlp_act_type="gelu", attn_bias=True, mlp_output_bias=True),
            num_layers=config["num_layers"],
        ).garbage_call(
            **{s: c for s, c in config.items() if s != "num_layers"},
            seq_len=seq_len,
            batch_shape=(batch_size,),
        )
        y_rust = substitute_all_modules(y_rust)

    cast_circuit(y_rust, TorchDeviceDtypeOp(device, "float32"))
    with timed("eval"):
        if concat_batch_size is not None:
            with timed("batching"):
                y_rust = batch_to_concat(y_rust, 0, concat_batch_size)
        optimize_and_evaluate(y_rust, OptimizationSettings(verbose=3, max_memory=2_000_000_000, scheduling_naive=False))


@pytest.mark.cuda
def test_len4():
    with open("/home/ubuntu/rrfs/adria/gpt2small/seq_len_4.txt") as f:
        circuit_raw = rc.cast_circuit(rc.Parser()(f.read()), rc.TorchDeviceDtypeOp(device="cpu"))
    # circuit_raw.print()
    circuit_arr = rc.Updater(
        lambda _: rc.DiscreteVar.new_uniform(rc.Array(torch.randint(0, 1, (100, 4), dtype=torch.float32)))
    )(circuit_raw, "tokens")
    # circuit_arr.print()
    circuit_sampled = rc.Sampler(rc.RandomSampleSpec()).sample(circuit_arr)
    # warm up
    circuit_sampled.evaluate()
    circuit_sampled.evaluate()
    with timed(".evaluate"):
        circuit_sampled.evaluate()
    # circuit_sampled.print()
    print("now opevaling")
    print(rc.count_nodes(circuit_sampled))
    with timed("opteval"):
        optimize_and_evaluate(circuit_sampled, OptimizationSettings(verbose=2))


if __name__ == "__main__":
    # raw_test_big_models({"num_layers": 50, "head_size": 16, "num_heads": 4}, "cuda")
    test_len4()
