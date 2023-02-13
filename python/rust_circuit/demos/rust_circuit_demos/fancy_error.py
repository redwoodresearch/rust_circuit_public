# import os
# os.environ["RUST_BACKTRACE"] = "1"

import pytest

import rust_circuit as rc
from rust_circuit.py_utils import I

# %% [markdown]

# There are some rust circuit error handling features worth knowing about:
# - Exception per error type
# - Accessing error data
# - error traceback

# %%

softmaxed = rc.softmax(rc.Array.randn(7, 3))
idxed = softmaxed.index(I[:, 2])

# %%

# there are different exception types for most errors.
# If you go to definition on the below error, you can see the full list of exceptions and what they inherit from.
with pytest.raises(rc.PushDownIndexGeneralFunctionSomeAxesNotPossibleError) as excinfo:
    # invalid push down due to being on unbatchable dim
    rc.push_down_index(idxed)

print(excinfo.exconly())

# %%

# errors also contain additional data
try:
    rc.push_down_index(idxed)
    assert False
except rc.PushDownIndexGeneralFunctionSomeAxesNotPossibleError as e:
    assert e.args[0].index_node == idxed
    assert e.args[0].inner_rank == 2
    assert e.args[0].top_axis_indices == [2]
    assert e.args[0].num_non_batch_out == 1
    print(dir(e.args[0]))

    print()

    # you can also use:
    print(e.args[0].items())

# %% [markdown

# If you set the env var RUST_BACKTRACE=1, then errors will have a backtrace through rust
# This may or may not be helpful, but is certainly helpful when debugging issues in the rust code.
# If you rerun this nb with the top lines uncommented, you should see this.

# %% [markdown]

# ## Rust details

# On the rust side, we add additional information to errors using
# `.context(...)` which is from the anyhow library
#
# If errors ever lack context due to passing through multiple rust functions,
# this should be used.

# %%
