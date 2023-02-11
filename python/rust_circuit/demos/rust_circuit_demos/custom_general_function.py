#%%
# import os
# os.environ["RUST_BACKTRACE"] = "1"

import uuid
from typing import Optional, Tuple

import pytest
import torch
from torch.testing import assert_close

import rust_circuit as rc
from rust_circuit.indexer import INDEXER as I

#%%

# order of importance:

# this
# get_update
# pdi
# nest
# (other test files, maybe too)

# %% [markdown]

# # Basic intro

# In this notebook, we'll demo generalfunctions and show how you can extend
# circuits by implementing your own generalfunction.
#
# You should think of this as an extension - if you
# implement the spec incorrectly this can cause pancis or bugs (we might
# improve this later).
#
# That said, there are nice functions for automatically
# testing your generalfunction specs for the properties we'll assume.

# %%

inp = rc.Array.randn(3, 4, 5)

# we can call functions like `softmax`
sm = rc.softmax(inp)
assert_close(sm.evaluate(), torch.softmax(inp.value, dim=-1))

# we can also do this (more verbosely) via 'new_by_name'
verbose_sm = rc.GeneralFunction.new_by_name(inp, spec_name="softmax")
assert sm == verbose_sm

# to see the full list of functions like this, goto definition on softmax
# (which will take you to the stub file for the library)

# %%

# functions will fail to be constructed if inputs shapes aren't viable.
with pytest.raises(rc.GeneralFunctionShapeNDimTooSmallError) as ex:
    rc.softmax(rc.Array.randn())
print(ex.exconly())

# %%

# other than these 'simple' general functions, the only other function we
# currently have implemented is `index_gen`. See interp/circuit/interop_rust/test_gen_index.py
# for examples.

# %%

# we can define additional simple functions like this via subclassing
# rc.GeneralFunctionSpecBase (we could add helpers for simple usecases if this
# becomes common, but hopefully we'll just upstream most simple functions?)


class MyCustomSigmoid(rc.GeneralFunctionSpecBase):
    @property
    def name(self) -> str:
        return "my_sigmoid"

    def compute_hash_bytes(self) -> bytes:
        # IMPORTANT NOTE: make sure you update the hash if you're iterating on a general function in a notebook.
        # If you don't, then caches might be invalid and you may get an old version of the generalfunction!
        return uuid.UUID("36a3ef34-a3cb-49a5-8cae-89f29f269d78").bytes

    def function(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.sigmoid(x)

    def get_shape_info(self, *shapes: rc.Shape) -> rc.GeneralFunctionShapeInfo:
        return rc.get_shape_info_simple(shapes)

    @classmethod
    def new(cls, circ: rc.Circuit, name: Optional[str] = None):
        """convenience function"""
        return rc.GeneralFunction(circ, spec=cls(), name=name)

    # optionally you can implement `get_device_dtype_override`, which validates input device/dtypes and optionally returns the output device/dtype
    # NOTE: if you do fancy stuff here, you'll need to manage dtype device upcasting! (TODO: better docs on this)
    def get_device_dtype_override(self, *device_dtypes: rc.TorchDeviceDtypeOp) -> Optional[rc.TorchDeviceDtypeOp]:
        assert device_dtypes[0].dtype in {
            "float16",
            "float32",
            "float64",
            "bfloat16",
        }  # softmax only applies to floats, not ints
        return None  # output has same device/dtype as input


# %%

func = MyCustomSigmoid.new(inp)
assert_close(func.evaluate(), torch.sigmoid(inp.value))
func

# %%

# this function will support adding extra batch dimensions:
bigger_inp = rc.Array.randn(7, *inp.shape)
bigger_func = rc.Expander((inp, lambda _: bigger_inp))(func)
assert_close(bigger_func.evaluate(), torch.sigmoid(bigger_inp.value))
bigger_func

# %%

# this custom function doesn't support bijection
with pytest.raises(RuntimeError):
    func.print(rc.PrintOptions(bijection=True))
# but support usual printing
func.print(rc.PrintOptions(bijection=False))

# %%
# but we support bijection for custom functions stored in rrfs
# NOTE: python caches imports, so editing the file and rerunning won't work in a notebook!
func_stored_in_rrfs = rc.GeneralFunction.new_by_path(inp, path="/test/general_function.py:MyCustomSigmoid")
func_stored_in_rrfs.print(rc.PrintOptions(bijection=True))

# %%

# now let's define a function which has more interesting batching properties
class MyCustomSoftmaxAndSum(rc.GeneralFunctionSpecBase):
    @property
    def name(self) -> str:
        return "softmax_and_sum"

    def compute_hash_bytes(self) -> bytes:
        return uuid.UUID("55735251-824c-4bf5-a6d3-87257cdbf8c3").bytes

    def function(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.softmax(x, dim=-2).sum(dim=-1)

    def get_shape_info(self, *shapes: rc.Shape) -> rc.GeneralFunctionShapeInfo:
        # here, we set some batching options. See GeneralFunctionShapeInfo docs for what this does!
        #
        # this can throw an exception if the shapes are invalid.
        #
        # note this is just a helper for simple cases - we can do whatever (valid) shape specification we want
        # TODO: these docs could be improved
        return rc.get_shape_info_simple(shapes, num_non_batchable_output_dims=1, removed_from_end=1)

    @classmethod
    def new(cls, circ: rc.Circuit, name: Optional[str] = None):
        """convenience function"""
        return rc.GeneralFunction(circ, spec=cls(), name=name)


# %%

new_func = MyCustomSoftmaxAndSum.new(inp)
assert_close(new_func.evaluate(), torch.softmax(inp.value, dim=-2).sum(dim=-1))
new_func

# %%

# we can test this has the advertized batching properties:
# see GeneralFunctionSpecTester docs for more details
rc.GeneralFunctionSpecTester(test_with_rand=True).test_many_shapes(MyCustomSoftmaxAndSum())
# note that this just checks that our spec is consistent and the function
# respects the shape info - not anything else.
# For instance, if your 'get_shape_info' always errors, that will pass as it's consistent behavior.
#
# Note that if your spec doesn't have these consistency properties, 'rust_circuit' might panic!
# (That's part of why we provide an automatic test suite :) )
# (This isn't that bad, mostly just worse error messages)

# %%

# Note that this function doesn't support batching over later dims.
# this means (for instance) that we can't push down an index through these dims.

indexed = new_func.index(I[:, 1])
with pytest.raises(rc.PushDownIndexError) as ex1:
    rc.push_down_index(indexed)
print(ex1.exconly())

# but we can push down index through first dim
other_indexed = new_func.index(I[1])
pushed = rc.push_down_index(other_indexed)
assert_close(pushed.evaluate(), torch.softmax(inp.value, dim=-2).sum(dim=-1)[1])
pushed

# %%

# suppose we defined the same function but buggy
class MyCustomSoftmaxAndSumBuggy(MyCustomSoftmaxAndSum):
    @property
    def name(self) -> str:
        return "softmax_and_sum_buggy"

    def compute_hash_bytes(self) -> bytes:
        return uuid.UUID("581b9dc8-3090-49a4-a774-75783b74f2d9").bytes

    def get_shape_info(self, *shapes: rc.Shape) -> rc.GeneralFunctionShapeInfo:
        # this dim actually isn't batchable!
        return rc.get_shape_info_simple(shapes, num_non_batchable_output_dims=0, removed_from_end=1)


# %%

# then our test will fail (in this case, the test fails because our spec claims
# that we could operate on inputs with ndim=1, but we actually can't and this
# wil fail with dimension out of range!)
with pytest.raises((rc.ExceptionWithRustContext, IndexError)) as ex2:
    rc.GeneralFunctionSpecTester(test_with_rand=True).test_many_shapes(MyCustomSoftmaxAndSumBuggy())
print(ex2.exconly())

# %%

# here we have a function which actually can operate on an input of any shape, but it actually doesn't support batching at all!
class BuggyMaxThenExpand(rc.GeneralFunctionSpecBase):
    @property
    def name(self) -> str:
        return "buggy_max"

    def compute_hash_bytes(self) -> bytes:
        return uuid.UUID("1b53e84c-3eca-43a2-b13a-ed5e00c22ae2").bytes

    def function(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.numel() == 0:
            return torch.full_like(x, 0.0)
        return torch.full_like(x, x.max().item())

    def get_shape_info(self, *shapes: rc.Shape) -> rc.GeneralFunctionShapeInfo:
        return rc.get_shape_info_simple(shapes, num_non_batchable_output_dims=1)

    @classmethod
    def new(cls, circ: rc.Circuit, name: Optional[str] = None):
        """convenience function"""
        return rc.GeneralFunction(circ, spec=cls(), name=name)


# %%

with pytest.raises((rc.ExceptionWithRustContext, AssertionError)) as ex3:
    rc.GeneralFunctionSpecTester(test_with_rand=True).test_many_shapes(BuggyMaxThenExpand())
print(ex3.exconly())

# %%


# let's do a function with fancier shape related properties
class FancyShape(rc.GeneralFunctionSpecBase):
    correct_x_thingy: int

    def __init__(self, correct_x_thingy: int) -> None:
        super().__init__()  # doesn't do anything atm
        self.correct_x_thingy = correct_x_thingy

    @property
    def name(self) -> str:
        return f"fancy_shape_with={self.correct_x_thingy}"

    def compute_hash_bytes(self) -> bytes:
        return uuid.UUID("59b46c83-d6b0-4267-9915-f372bc722181").bytes

    def function(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # random function with appropriate shape stuff
        [x, y, z] = rc.upcast_tensor_device_dtypes(
            [x, y, z]
        )  # we need this for properly managing scalars and none-dtype in general...
        # TODO: more docs on upcast_tensor_device_dtypes and what general functions have to do
        return torch.log_softmax(torch.einsum("... a b, c, ... -> ... b c", x, y, z), dim=-1)

    def get_shape_info(self, x_shape: rc.Shape, y_shape: rc.Shape, z_shape: rc.Shape) -> rc.GeneralFunctionShapeInfo:  # type: ignore[override]
        assert len(x_shape) >= 2
        assert len(y_shape) == 1
        assert x_shape[-1] == self.correct_x_thingy, f"last dim != thingy={self.correct_x_thingy}"
        batch_shape = z_shape
        assert x_shape[:-2] == batch_shape

        return rc.GeneralFunctionShapeInfo(
            batch_shape + (x_shape[-1],) + y_shape,
            num_non_batchable_output_dims=2,
            input_batchability=[True, False, True],
        )

    @classmethod
    def new(cls, x: rc.Circuit, y: rc.Circuit, z: rc.Circuit, name: Optional[str] = None):
        """convenience function"""
        return rc.GeneralFunction(x, y, z, spec=cls(7), name=name)


# %%

batch_shape: Tuple[int, ...] = (1, 3, 4)
x = rc.Array.randn(*batch_shape, 9, 7)
y = rc.Array.randn(8)
z = rc.Array.randn(*batch_shape)
FancyShape.new(x, y, z)

# %%

batch_shape: Tuple[int, ...] = (1, 3, 4)
x = rc.Array.randn(*batch_shape, 9, 7)
y = rc.Array.randn(8)
z = rc.Scalar(value=384.0, shape=batch_shape)
# this would fail if we didn't upcast tensors due to float64 vs float32
# (ok, technically einsum apparently is fine with differnet dtypes (!?), but in theory)
_ = FancyShape.new(x, y, z).evaluate()


# %%

x_new = rc.Array.randn(*batch_shape, 9, 8)
with pytest.raises((rc.ExceptionWithRustContext, AssertionError)) as ex4:
    FancyShape.new(x_new, y, z)
print(ex4.exconly())

# %%

# torch.einsum w/ opt_einsum doesn't support 0 dims & GeneralFunctionSpecTester generates them
torch.backends.opt_einsum.enabled = False

# by default, this function will basically never find any valid shapes! We
# check that we do sometimes find valid shapes, so this will error (see `min_frac_successful` below)
with pytest.raises(RuntimeError) as ex8:
    rc.GeneralFunctionSpecTester(
        test_with_rand=True,
        samples_per_batch_dims=10,
        base_shapes_samples=1000,
        start_num_inputs=2,
        end_num_inputs=4,
        # this variable can be configured to change the minimum fraction of tests which do anything
        # see docs for more details
        min_frac_successful=0.1,
    ).test_many_shapes(FancyShape(5))
print(ex8.exconly())

# %%

# but we can run a custom generation scheme which gets more hits
for _ in range(1_000):
    batch_shape = tuple(int(x) for x in torch.randint(0, 7, size=(int(torch.randint(0, 4, ())),)))
    rc.GeneralFunctionSpecTester(test_with_rand=True, samples_per_batch_dims=10).test_from_shapes(
        FancyShape(5),
        [(*batch_shape, int(torch.randint(0, 7, ())), 5), (int(torch.randint(0, 7, ())),), batch_shape],
        shapes_must_be_valid=True,
    )

# %%


class FancyShapeBugged1(FancyShape):
    def get_shape_info(  # type: ignore[override]
        self, x_shape: rc.Shape, y_shape: rc.Shape, z_shape: rc.Shape
    ) -> rc.GeneralFunctionShapeInfo:
        out = super().get_shape_info(x_shape, y_shape, z_shape)
        out.num_non_batchable_output_dims = 1
        return out


with pytest.raises(RuntimeError) as ex5:
    for _ in range(1_000):
        batch_shape = tuple(int(x) for x in torch.randint(0, 7, size=(int(torch.randint(0, 4, ())),)))
        rc.GeneralFunctionSpecTester(test_with_rand=True, samples_per_batch_dims=10).test_from_shapes(
            FancyShapeBugged1(5),
            [(*batch_shape, int(torch.randint(0, 7, ())), 5), (int(torch.randint(0, 7, ())),), batch_shape],
            shapes_must_be_valid=True,
        )

print(ex5.exconly())

# %%


class FancyShapeBugged2(FancyShape):
    def function(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = super().function(x, y, z)
        if out.ndim >= 3:
            return torch.log_softmax(out, dim=-3)  # not actually batchable!
        return out


with pytest.raises((rc.ExceptionWithRustContext, AssertionError)) as ex6:
    for _ in range(1_000):
        batch_shape = tuple(int(x) for x in torch.randint(0, 7, size=(int(torch.randint(0, 4, ())),)))
        rc.GeneralFunctionSpecTester(test_with_rand=True, samples_per_batch_dims=10).test_from_shapes(
            FancyShapeBugged2(5),
            [(*batch_shape, int(torch.randint(0, 7, ())), 5), (int(torch.randint(0, 7, ())),), batch_shape],
            shapes_must_be_valid=True,
        )

print(ex6.exconly())

# %%
