from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional, Sequence, Literal

import uuid
from dataclasses import dataclass
from uuid import UUID

import torch

import rust_circuit as rc
import rust_circuit.index_util.internal.ops as ops
import rust_circuit.index_util.internal.parse as parse
import rust_circuit.index_util.internal.resolve as resolve


class Gather(rc.GeneralFunctionSpecBase):
    """Get elements from a 'source' tensor using indices from a 'position' tensor.

    To create a 'Gather' node, use the 'Gather.new' method. The 'Gather.new' method takes a string specifying its
    behavior in a little programming language, similar to 'Einsum'. The program string must always be of the form:

        out[i0, i1, ..., iN] = src[..., pos[...], ...]

    where 'i0, i1, ..., iN' are the indices of the output tensor, 'src' is the name of the source tensor, and 'pos' is
    the name of the position tensor. The 'out' array must always be named 'out', but 'src' and 'pos' can be named
    whatever you want; their names are determined based on the keys of the 'args' dictionary you pass to 'Gather.new'.
    The index expressions used in the square brackets after 'src' and 'pos' can be either the index variables 'i0, i1,
    ..., iN', or constant integers.

    The 'out_shape' argument to 'Gather.new' is optional. If 'out_shape" is not specified, the output shape is inferred
    from the program string and the shapes of the input tensors. If 'out_shape' is specified, it must be a list of
    integers or 'None'. If an element of the list is 'None', the corresponding dimension of the output shape is inferred
    from the program string.

    'Gather' supports arbitrary batching. If any of the tensors have a shape with rank greater than what is implied by
    the program string, any extra leading dimensions are treated as batch dimensions.

    Examples:

        # Get the activations for a different token position for each sequence in a batch:
        #  'activations' has shape [batch_size, seq_len, num_features]
        #  'token_positions' has shape [batch_size] and contains integers in the range [0, seq_len)
        Gather.new(
            "out[i, j] = activations[i, token_positions[i], j]",
            {"activations": activations, "token_positions": token_positions},
        )

        # The previous example can also be done with implicit batching!
        Gather.new(
            "out[j] = activations[token_positions, j]", # equivalent to previous example
            {"activations": activations, "token_positions": token_positions},
        )

        # Get the difference in logit values between two reference vocabulary tokens for each sequence in a batch:
        #   'logits' has shape [batch_size, vocab_size]
        #   'logit_indices' has shape [batch_size, 2] and contains integers in the range [0, vocab_size).
        logits_token0 = Gather.new(
            "out = logits[logit_indices[0]]", # uses implicit batching
            {"logits": logits, "logit_indices": logit_indices},
        )
        logits_token1 = Gather.new(
            "out = logits[logit_indices[1]]", # uses implicit batching
            {"logits": logits, "logit_indices": logit_indices},
        )
        diff = Add.minus(logits_token0, logits_token1)

        # Indices are allowed to repeat:
        Gather.new(
            "out[i, j] = foo[i, i, bar[i, j], j]", # allowed {"foo": foo, "bar": bar},
        )

        # Indices don't need to appear in the same order as the output:
        Gather.new(
            "out[i, j, k] = foo[k, bar[j, k, i], i, j], # allowed {"foo": foo, "bar": bar},
        )

        # When the output shape can't be inferred, you can specify the missing dimensions explicitly:
        Gather.new(
            "out[i, j] = foo[bar[j]]", {"foo": foo, "bar": bar}, out_shape=[20, None],
        )
    """

    _spec: resolve.GatherSpec
    _plan: ops.GatherPlan
    _orig_out_shape: Optional[list[Optional[int]]]

    def __init__(self, s: str, *args: str, out_shape: Optional[Sequence[Optional[int]]] = None) -> None:
        super().__init__()
        out_shape_list = list(out_shape) if out_shape is not None else None
        self._spec = resolve.GatherSpec.parse(s, args)
        self._plan = ops.plan_gather(self._spec)
        self._orig_out_shape = out_shape_list

    @property
    def name(self) -> str:
        out_shape_str = f", out_shape={self._orig_out_shape!r}" if self._orig_out_shape is not None else ""
        return f"Gather({str(self._spec)!r}, {self._spec.src_name!r}, {self._spec.pos_name!r}{out_shape_str})"

    def compute_hash_bytes(self) -> bytes:
        b = bytearray(uuid.UUID("7815ae94-c2bf-4e9b-a1a1-d2c96767c9a8").bytes)

        def extend_str(s: str) -> None:
            b.extend(len(s).to_bytes(8, "little"))
            b.extend(s.encode("utf-8"))

        extend_str(str(self._spec))
        extend_str(self._spec.src_name)
        extend_str(self._spec.pos_name)
        extend_str(str(self._orig_out_shape))

        return bytes(b)

    def get_shape_info(self, *shapes: rc.Shape) -> rc.GeneralFunctionShapeInfo:
        src_shape, pos_shape = shapes
        inferred = ops.infer_gather_shapes(
            self._plan, self._orig_out_shape, list(src_shape), list(pos_shape), self._spec.src_name, self._spec.pos_name
        )
        return rc.GeneralFunctionShapeInfo(
            tuple(inferred.batch_shape + inferred.axis_sizes),
            num_non_batchable_output_dims=len(self._orig_out_shape)
            if self._orig_out_shape is not None
            else len(inferred.axis_sizes),
            input_batchability=[True, True],
        )

    def function(self, *inputs: torch.Tensor) -> torch.Tensor:
        src, pos = inputs
        return ops.execute_gather(self._plan, self._orig_out_shape, src, pos, self._spec.src_name, self._spec.pos_name)

    def get_device_dtype_override(self, *device_dtypes: rc.TorchDeviceDtypeOp) -> Optional[rc.TorchDeviceDtypeOp]:
        src_dtype, pos_dtype = device_dtypes
        assert pos_dtype.dtype == "int64"
        return src_dtype

    @classmethod
    def new(
        cls,
        spec: str,
        args: dict[str, rc.Circuit],
        out_shape: Optional[Sequence[Optional[int]]] = None,
        name: Optional[str] = None,
    ) -> rc.Circuit:
        func_spec = cls(spec, *args.keys(), out_shape=out_shape)
        src_name = func_spec._spec.src_name
        pos_name = func_spec._spec.pos_name
        return rc.GeneralFunction(args[src_name], args[pos_name], spec=func_spec, name=name)


class Scatter(rc.GeneralFunctionSpecBase):
    """Write to selected elements of a 'destination' tensor using values from a 'source' tensor and indices from a
    'position' tensor.

    This operation is non-destructive; the destination tensor is not modified. Instead, a new tensor is created with the
    same shape as the destination tensor, and with the selected elements changed.

    To create a 'Scatter' node, use the 'Scatter.new' method. Like 'Gather.new', this method takes a string specifying
    the operation to be performed in a little programming language. The string must always be of the form:

        dst[..., pos[...], ...] <- src[...]

    where 'dst' is the name of the destination tensor, 'src' is the name of the source tensor, and 'pos' is the name of
    the position tensor. The 'dst', 'src', and 'pos' names can be chosen arbitrarily, but they must be consistent with
    the names used in the 'args' dictionary passed to 'Scatter.new'. The index expressions used in the square brackets after 'dst', 'src', and 'pos' can be either index variables or
    constant integers.

    'Scatter' supports arbitrary batching. If any of the tensors have a shape with rank greater than what is implied by
    the program string, any extra leading dimensions are treated as batch dimensions.

    The 'reduce' argument to 'Scatter.new' specifies how the source values are combined with the destination values. If
    'reduce' is 'None', the source values are written directly to the destination tensor and replace the old values. If
    'reduce' is 'add' or 'multiply', the source values are reduced with the destination values using the specified
    operation.

    Every index variable which appears on the right hand side of the formula must also appear on the left hand side.
    Moreover, due to a limitation of the underlying PyTorch 'scatter' operation, all but at most one of the variables
    appearing in the formula must appear as a direct index on 'dst' (see below for an example of this restriction).

    Examples:

        # Patch the activations from one batch of sequences into another, at different token position for each sequence
        # in the batch:

        #   'activations1' is the destination tensor, with shape [batch_size, seq_len, num_features]
        #   'activations2' is the source tensor, also with shape [batch_size, seq_len, num_features]
        #   'positions' is the position tensor, with shape [batch_size] and values in the range [0, seq_len)

        # First use a 'Gather' to extract the activations at the specified positions
        extracted = Gather.new(
            "out[i, j] = activations2[i, positions[i], j]",
            {"activations2": activations2, "positions": positions},
        )

        # Now, use a 'Scatter' to write the activations at the specified positions
        activations1_patched = Scatter.new(
            "activations1[i, positions[i], j] <- extracted[i, j]",
            {"activations1": activations1, "extracted": extracted},
        )

        # The same operation can be performed with implicit batching in both the gather and the scatter:

        extracted = Gather.new(
            "out[i] = activations2[positions, i]", # uses implicit batching
            {"activations2": activations2, "positions": positions},
        )
        activations1_patched = Scatter.new(
            "activations1[positions, i] <- extracted[i]", # uses implicit batching
            {"activations1": activations1, "extracted": extracted},
        )

        # You are allowed to repeat and reorder indices:
        scatter = Scatter.new(
            "foo[i, i, baz[i, j, i], k] <- bar[k, j, i]", # OK
            {"foo": foo, "bar": bar, "baz": baz},
        )

        # You can use constant integers to index into tensors:
        scatter = Scatter.new(
            "foo[i, 0, baz[i, 2], 1] <- bar[4, i]", # OK
            {"foo": foo, "bar": bar, "baz": baz},
        )

        # Every variable appearing on the right hand side must also appear on the left hand side:
        scatter = Scatter.new(
            "foo[i, baz[j]] <- bar[i, j, k]", # ERROR: 'k' does not appear on the left hand side
            {"foo": foo, "bar": bar, "baz": baz},
        )

        # All but at most one of the variables appearing in the formula must appear as a direct index on 'dst':
        scatter = Scatter.new(
            # ERROR: there are two variables which do not appear as direct indices on 'dst': 'j' and 'k'
            "foo[i, baz[j, k]] <- bar[i, j, k]",
            {"foo": foo, "bar": bar, "baz": baz},
        )
    """

    _spec: resolve.ScatterSpec
    _plan: ops.ScatterPlan

    def __init__(self, s: str, *args: str, reduce: Optional[Literal["add", "multiply"]] = None) -> None:
        super().__init__()
        self._spec = resolve.ScatterSpec.parse(s, args, reduce=reduce)
        self._plan = ops.plan_scatter(self._spec)

    @property
    def name(self) -> str:
        return (
            f"Scatter({str(self._spec)!r}, "
            + f"{self._spec.dst_name!r}, {self._spec.src_name!r}, {self._spec.pos_name!r}, "
            + f"reduce={self._spec.reduce!r})"
        )

    def compute_hash_bytes(self) -> bytes:
        b = bytearray(uuid.UUID("b3b3c3c3-1b2d-4c9b-9f7a-0f3e8b3d3b3b").bytes)

        def extend_str(s: str) -> None:
            b.extend(len(s).to_bytes(8, "little"))
            b.extend(s.encode("utf-8"))

        extend_str(str(self._spec))
        extend_str(self._spec.dst_name)
        extend_str(self._spec.src_name)
        extend_str(self._spec.pos_name)
        extend_str(str(self._spec.reduce))

        return bytes(b)

    def get_shape_info(self, *shapes: rc.Shape) -> rc.GeneralFunctionShapeInfo:
        dst_shape, src_shape, pos_shape = shapes
        inferred, out_shape = ops.infer_scatter_shapes(
            self._plan,
            list(dst_shape),
            list(src_shape),
            list(pos_shape),
            self._spec.dst_name,
            self._spec.src_name,
            self._spec.pos_name,
        )
        return rc.GeneralFunctionShapeInfo(
            tuple(out_shape),
            num_non_batchable_output_dims=len(out_shape) - len(inferred.batch_shape),
            input_batchability=[True, True, True],
        )

    def function(self, *inputs: torch.Tensor) -> torch.Tensor:
        dst, src, pos = inputs
        return ops.execute_scatter(
            self._plan, dst, src, pos, self._spec.dst_name, self._spec.src_name, self._spec.pos_name
        )

    def get_device_dtype_override(self, *device_dtypes: rc.TorchDeviceDtypeOp) -> Optional[rc.TorchDeviceDtypeOp]:
        dst_dtype, src_dtype, pos_dtype = device_dtypes
        assert pos_dtype.dtype == "int64"
        return dst_dtype

    @classmethod
    def new(
        cls,
        spec: str,
        args: dict[str, rc.Circuit],
        reduce: Optional[Literal["add", "multiply"]] = None,
        name: Optional[str] = None,
    ) -> rc.Circuit:
        func_spec = cls(spec, *args.keys(), reduce=reduce)
        dst_name = func_spec._spec.dst_name
        src_name = func_spec._spec.src_name
        pos_name = func_spec._spec.pos_name
        return rc.GeneralFunction(args[dst_name], args[src_name], args[pos_name], spec=func_spec, name=name)
