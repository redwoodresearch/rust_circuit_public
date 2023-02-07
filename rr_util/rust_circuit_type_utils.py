# this file is included with `include_str!()` in py_types.rs
import functools
import itertools
import math
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional


def check_type(x, type_v):
    if not isinstance(x, type_v):
        raise TypeError(f"expected type {repr(type_v)} but got {type(x)}")


def get_tensor_shape(x):
    check_type(x, torch.Tensor)
    return list(x.size())


dtype_from_string = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
    "bool": torch.bool,
    "uint8": torch.uint8,
}
dtype_to_string = {v: k for k, v in dtype_from_string.items()}


def maybe_dtype_to_maybe_string(x):
    return dtype_to_string.get(x)


def scalar_to_tensor(scalar, shape, device_dtype):
    "Makes a stride 0 repeat view into a 1 element tensor"
    scalar_tensor = torch.tensor(scalar, device=device_dtype.device, dtype=dtype_from_string[device_dtype.dtype])
    return torch.broadcast_to(scalar_tensor, tuple(shape))


def tensor_scale(tensor):
    return tensor.abs().mean().cpu().item()


def cast_tensor(tensor, device_dtype):
    kwargs = {}
    if device_dtype.device is not None:
        kwargs["device"] = device_dtype.device
    if device_dtype.dtype is not None:
        kwargs["dtype"] = dtype_from_string[device_dtype.dtype]
    return tensor.to(**kwargs)


zero_tensor = torch.zeros(())


def un_flat_concat(tensor: torch.Tensor, shapes) -> List[torch.Tensor]:
    lens = [math.prod(x) for x in shapes]
    flats = torch.split(tensor, lens, dim=0)
    return [x.reshape(shape) for x, shape in zip(flats, shapes)]


def log_exp_p_1_fn(x: torch.Tensor):
    # piecewise to nicely handle numerics
    addr = 1.0
    return torch.where(x < 0.0, torch.log(torch.exp(x) + addr), torch.log(1.0 + torch.exp(-x) * addr) + x)


generalfunctions = {
    "sin": torch.sin,
    "cos": torch.cos,
    "gelu": torch.nn.functional.gelu,
    "gelu_new": lambda x: 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))),
    "relu": torch.relu,
    "step": lambda x: torch.where(
        x > 0.0, torch.ones((), dtype=x.dtype, device=x.device), torch.zeros((), dtype=x.dtype, device=x.device)
    ),
    "sigmoid": torch.sigmoid,
    "rsqrt": torch.rsqrt,
    "reciprocal": torch.reciprocal,
    "tanh": torch.tanh,
    "softmax": functools.partial(torch.softmax, dim=-1),
    "log_softmax": functools.partial(torch.log_softmax, dim=-1),
    "log_exp_p_1": log_exp_p_1_fn,
    "gaussian_pdf": lambda x: torch.exp(-(x ** 2) / 2) / math.sqrt(2 * math.pi),
    "gaussian_cdf": lambda x: torch.distributions.normal.Normal(0, 1).cdf(x),
    "q_from_qr": lambda x: torch.linalg.qr(x)[0],
    "min": lambda x: torch.min(x, dim=-1)[0],
    "max": lambda x: torch.max(x, dim=-1)[0],
    "last_dim_size": lambda x: torch.full(x.shape[:-1], x.shape[-1], dtype=x.dtype, device=x.device),
    "abs": torch.abs,
    "exp": torch.exp,
    "log": torch.log,
    "logit": torch.logit,
}

pow = torch.pow


def check_canon_idx(i: int, count: int):
    assert count >= 0, count
    if i >= 0:
        assert i < count, (i, count)
    else:
        assert i >= -count, (i, count)
    return i % count


def check_ints(x: torch.Tensor):
    assert (x.long().to(dtype=x.dtype) == x).all()


def gen_index_function(
    x: torch.Tensor, index: torch.Tensor, index_dim: int, batch_x: bool, batch_index: bool, check_index_ints: bool
):
    # copy of  WildIndex from computational_node.py
    if check_index_ints:
        check_ints(index)
    use_index_as_prefix = batch_x and batch_index
    if use_index_as_prefix:
        return explicit_gen_index_function(x, index, index_dim, x.ndim - index.ndim, check_index_ints=check_index_ints)
    elif batch_index:
        moved_out = torch.index_select(x, index_dim, index.long().flatten()).moveaxis(index_dim, 0)
        return moved_out.reshape((*index.shape, *moved_out.shape[1:]))
    else:
        canon_index_dim = check_canon_idx(index_dim, x.ndim)
        out = torch.index_select(x, canon_index_dim, index.long().flatten())
        return out.reshape((*out.shape[:canon_index_dim], *index.shape, *out.shape[canon_index_dim + 1 :]))


def explicit_gen_index_function(
    x: torch.Tensor, index: torch.Tensor, index_dim: int, x_non_batch_dims: int, check_index_ints: bool
):
    if check_index_ints:
        check_ints(index)

    batch_len = x.ndim - x_non_batch_dims
    batch_shape = x.shape[:batch_len]
    assert batch_shape == index.shape[:batch_len]
    # x_len = x.ndim - prefix_len
    moved_x = x.movedim(check_canon_idx(index_dim, x_non_batch_dims) + batch_len, 0)
    if moved_x.ndim == 1:
        return moved_x[index.long()]

    # end_dim is inclusive, so this is right
    flattened_x = moved_x.flatten(start_dim=1, end_dim=batch_len)
    flattened_index = index.flatten(start_dim=0, end_dim=batch_len - 1).long()

    together_arange = torch.arange(math.prod(batch_shape))
    pad_arange = together_arange[(slice(None), *([None] * (index.ndim - batch_len)))]
    new_x = flattened_x[flattened_index, pad_arange]
    return new_x.reshape((*index.shape, *moved_x.shape[1 + batch_len :]))


assert_tensors_close = torch.testing.assert_close


def make_diagonal(tensor: torch.Tensor, tensor_ints: Tuple[int, ...], out_ints: Tuple[int, ...]):
    int_sizes = {x: tensor.shape[i] for i, x in enumerate(tensor_ints)}
    deduped_shape = [int_sizes[x] for x in tensor_ints]
    result = torch.zeros(tuple(int_sizes[x] for x in out_ints), dtype=tensor.dtype, device=tensor.device)
    normal_strides_out = result.stride()
    fancy_strides = []
    for i in tensor_ints:
        indices = [j for j, x in enumerate(out_ints) if x == i]
        strides = [normal_strides_out[k] for k in indices]
        stride_here = sum(strides)
        fancy_strides.append(stride_here)
    fancy_strided = torch.as_strided(result, deduped_shape, fancy_strides)
    fancy_strided += tensor
    return result


def tensor_to_bytes(x: torch.Tensor):
    return x.cpu().numpy().tobytes()


def tensor_from_bytes(device_dtype, shape, bytes, count):
    return (
        torch.frombuffer(bytes, count=count, offset=0, dtype=dtype_from_string[device_dtype.dtype])
        .reshape(shape)
        .to(device=device_dtype.device)
    )


einsum = lambda tensors_and_axes, out_axes: torch.einsum(*itertools.chain(*tensors_and_axes), out_axes)


def random_indices(
    probs: torch.Tensor, shape: Sequence[int], replacement: bool, seed: Optional[Union[int, torch.Tensor]] = None
) -> torch.Tensor:
    shape = tuple(shape)

    assert probs.ndim >= 1

    generator: Optional[torch.Generator] = (
        torch.Generator(device=probs.device).manual_seed(int(seed)) if seed is not None else None
    )

    return torch.multinomial(
        probs.reshape(math.prod(probs.shape[:-1]), probs.shape[-1]),
        num_samples=math.prod(shape),
        replacement=replacement,
        generator=generator,
    ).reshape((*probs.shape[:-1], *shape))


# we use different axis layout than pytorch,
# batch_dims... height_width_ect... channels
# we also allow asymmetric padding to support InceptionV1
# which we have to do manually bc torch.conv doesn't support
def conv(dim, input, filter, stride, padding):
    import einops

    assert len(filter.shape) == 2 + dim, "don't support filter batching yet"
    stride = tuple(stride)

    batch_rank = len(input.shape) - dim - 1
    char_at = lambda i: chr(ord("a") + i)
    input_rearrange_string_i = " ".join([char_at(x) for x in range(len(input.shape))])
    input_rearrange_string_o = f"({' '.join([char_at(x) for x in range(batch_rank)])}) {char_at(len(input.shape)-1)} {' '.join([char_at(x) for x in range(batch_rank,len(input.shape)-1)])}"
    input_rearrange_string = f"{input_rearrange_string_i} -> {input_rearrange_string_o}"
    input_batches_together = einops.rearrange(input, input_rearrange_string)
    if any([x[0] != x[1] for x in padding]):
        padding_min = tuple([min(x) for x in padding])
        input_batches_together = torch.nn.functional.pad(
            input_batches_together,
            tuple(itertools.chain(*[(p[0] - pmin, p[1] - pmin) for p, pmin in zip(padding, padding_min)])),
        )
    padding = tuple([min(x) for x in padding])
    if dim == 1:
        filter = einops.rearrange(filter, "o a i -> o i a")
        result = torch.nn.functional.conv1d(input_batches_together, filter, None, stride, padding)
    elif dim == 2:
        filter = einops.rearrange(filter, "o a b i -> o i a b")
        result = torch.nn.functional.conv2d(input=input_batches_together, weight=filter, stride=stride, padding=padding)
    elif dim == 3:
        filter = einops.rearrange(filter, "o a b c i -> o i a b c")
        result = torch.nn.functional.conv3d(input=input_batches_together, weight=filter, stride=stride, padding=padding)
    else:
        raise ValueError("conv only supports 1, 2, or 3d convolutions")
    return einops.rearrange(
        result,
        f"{input_rearrange_string_o} -> {input_rearrange_string_i}",
        **{char_at(i): input.shape[i] for i in range(batch_rank)},
    )


class OptimizingSymbolicSizeWarning(UserWarning):
    ...


def random_i64():
    return int(torch.randint(-(2 ** 63), 2 ** 63 - 1, size=(), dtype=torch.long))
