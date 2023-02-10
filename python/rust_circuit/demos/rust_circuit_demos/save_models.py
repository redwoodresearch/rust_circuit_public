# %%

import os
from typing import Optional, Sequence, Tuple

import rust_circuit as rc
from interp.model.model_loading import load_model_info, load_model_mem_cached
from rust_circuit.demos.notebook_testing import NotebookInTesting
from rust_circuit.jax_to_module import get_bound_model
from rust_circuit.module_library import load_transformer_model_string

RRFS_DIR = os.path.expanduser("~/rrfs")
RRFS_INTERP_MODELS_DIR = f"{RRFS_DIR}/interpretability_models_jax/"
os.environ["INTERPRETABILITY_MODELS_DIR"] = os.environ.get(
    "INTERPRETABILITY_MODELS_DIR",
    os.path.expanduser("~/interp_models_new/")
    if os.path.exists(os.path.expanduser("~/interp_models_new/"))
    else RRFS_INTERP_MODELS_DIR,
)
# %%

all_models: Sequence[Tuple[str, str, Optional[str]]]
if NotebookInTesting.currently_in_notebook_test:
    all_models = [
        ("gelu_two_layers_untied", "gelu_2", None),
    ]
else:
    # TODO: resave before merge!
    all_models = [
        ("attention_only_bn_four_layers_nobias_3_fixed", "attention_only_bn_4", None),
        ("attention_only_bn_two_layers_nobias_3_fixed", "attention_only_bn_2", None),
        ("attention_only_two_layers_untied", "attention_only_2", None),
        ("bilinear_bn_four_layers_nobias_3_fixed", "bilinear_bn_4", None),
        ("bilinear_bn_two_layers_nobias_3_fixed", "bilinear_bn_2", None),
        ("bilinear_four_layers_untied", "bilinear_4", None),
        ("bilinear_two_layers_untied", "bilinear_2", None),
        ("gelu_one_layer_untied", "gelu_1", None),
        ("gelu_two_layers_untied", "gelu_2", None),
        ("gelu_twelve_layers", "gelu_12_tied", "gpt2-small"),
        ("gelu_twenty_four_layers", "gelu_24_tied", "gpt2-medium"),
        ("jun9_paren_balancer", "jun9_paren_balancer", None),
        ("paren_balancer_backdoor_nov_16", "paren_balancer_backdoor_nov_16", None)
        # ("gelu_36_layers", "gelu_36", "gpt2-large"),
        # ("gelu_48_layers", "gelu_48", "gpt2-xlarge"),
    ]

for (orig_model_id, new_model_id, aka) in all_models:
    model_info, _ = load_model_info(orig_model_id)
    model, params, _ = load_model_mem_cached(orig_model_id)
    model.norm_type
    bound_model, both_embeds, info, _ = get_bound_model(model.bind(params), model_class=model_info["model_class"])
    s = info.dump_model_string(*both_embeds, bound_model)
    s += f"\n# originally {orig_model_id}"
    if aka is not None:
        s += f" (aka {aka})"
    s += "\n"
    with open(f"/tmp/{new_model_id}.circ", "w") as f:
        f.write(s)

# %%
# Can be slow!
# Helps if you've cleared ~/tensors_by_hash_cache before running this notebook
rc.sync_all_unsynced_tensors()

# %%

# for running models, see interp/circuit/interop_rust/test_jax_to_module.py
out, tok, extra_args = load_transformer_model_string(open("/tmp/gelu_2.circ").read())
# out, tok, extra_args = load_model_id("gelu_2")
print(extra_args)
out["t.logits"].print()
tok
