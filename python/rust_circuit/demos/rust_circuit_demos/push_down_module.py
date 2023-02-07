# %%

# TODO: make this demo less shitty

import os

import pytest
import torch

import rust_circuit as rc
import rust_circuit.module_library as mod_l
from rust_circuit.model_rewrites import configure_transformer

os.environ["RUST_BACKTRACE"] = "1"

# %%

params = mod_l.TransformerParams(mod_l.TransformerBlockParams(attn_bias=True, mlp_output_bias=True), num_layers=2)
new_circ, _, _ = params.garbage_call(num_heads=2)  # 2 heads to avoid too much clutter
new_circ = new_circ.update("t.bind_w", lambda x: configure_transformer(x, use_pull_up_bias=False))
new_circ.print()

# %%

x = new_circ.get_unique("b1")
print()
x.print()
print()
pushed = rc.ModulePusher()(x, traversal=rc.new_traversal(term_early_at={"m", "ln", "a.on_inp"}))
pushed.print()
rep_push = new_circ.update("b1", lambda _: pushed)
torch.testing.assert_close(rep_push.evaluate(), new_circ.evaluate())

# %%

# crazy pushing!
for flatten_modules in [False, True]:
    for end_depth in range(8):
        full_push = rc.ModulePusher(flatten_modules=flatten_modules)(
            new_circ, traversal=rc.new_traversal(end_depth=end_depth), skip_module=~rc.Matcher(new_circ)
        )
        torch.testing.assert_close(full_push.evaluate(), new_circ.evaluate())

# %%

new_circ_extra_batch = rc.Expander(
    ("t.inp.tok_embeds", lambda x: rc.Array.randn(2, *x.shape, device_dtype=rc.TorchDeviceDtypeOp(dtype="float64")))
)(new_circ)

# more crazy pushing!
for flatten_modules in [False, True]:
    for end_depth in range(8):
        full_push = rc.ModulePusher(flatten_modules=flatten_modules)(
            new_circ_extra_batch, traversal=rc.new_traversal(end_depth=end_depth), skip_module=~rc.Matcher(new_circ)
        )
        torch.testing.assert_close(full_push.evaluate(), new_circ_extra_batch.evaluate())

# %%

# partial substitution with just causal mask: first extract then sub
sub_causal_mask = rc.extract_symbols_get(new_circ, "a.mask").substitute()
# very nice
sub_causal_mask.print()
torch.testing.assert_close(sub_causal_mask.evaluate(), new_circ.evaluate())

# %%

# pushing past a overridden symbol isn't supported atm (but could be supported, just a bit complicated)

# same symbol, different value - the inner value is used
s = """
'a' Module
  'add' Add
    'b' Module
      'sym' [] Symbol c9517c8e-0b5a-4ebf-94c8-6da413c9f5d1
      'b.arg' [] Scalar 1.2 ! 'sym'
    'sym'
  'a.arg' [3] Scalar 1.3 ! 'sym'
"""
circ = rc.Parser()(s)
with pytest.raises(rc.PushDownModulePushingPastModuleWhichOverridesSymError):
    rc.ModulePusher().push_down_modules(circ, traversal=rc.new_traversal(), skip_module="b")

# if you don't skip, this is totally valid
torch.testing.assert_close(
    rc.ModulePusher().push_down_modules(circ, traversal=rc.new_traversal()).evaluate(), circ.evaluate()
)

# %%

# more complex case for the above
s = """
'a' Module
  'add' Add
    'b' Module
      'add_inner' Add
        'sym' [] Symbol c9517c8e-0b5a-4ebf-94c8-6da413c9f5d1
        'sym.outer' [] Symbol 53db5cff-561e-4756-a502-65a26cb2a5f4
      'b.arg' [] Scalar 1.2 ! 'sym'
    'sym'
  'a.arg' [3] Scalar 1.3 ! 'sym'
  'a.outer_arg' [4, 3] Scalar 3.3 ! 'sym.outer'
"""
circ = rc.Parser()(s)
with pytest.raises(rc.PushDownModulePushingPastModuleWhichOverridesSymError):
    rc.ModulePusher().push_down_modules(circ, traversal=rc.new_traversal(), skip_module="b")

torch.testing.assert_close(
    rc.ModulePusher().push_down_modules(circ, traversal=rc.new_traversal(term_early_at="add_inner")).evaluate(),
    circ.evaluate(),
)

rc.ModulePusher().push_down_modules(circ, traversal=rc.new_traversal(term_early_at={"sym", "sym.outer"})).print()
torch.testing.assert_close(
    rc.ModulePusher()
    .push_down_modules(circ, traversal=rc.new_traversal(term_early_at={"sym", "sym.outer"}))
    .evaluate(),
    circ.evaluate(),
)


# %%
