# this file is included with `include_str!()` in py_types.rs

import atexit
import functools
import hashlib
from typing import Dict, Tuple

import numpy as np
import torch
from blake3 import blake3

PER_BLOCK_THREAD_COUNT = 128


@functools.lru_cache(maxsize=None)
def get_cuda_kernels():
    import pycuda.autoprimaryctx as apc  # type: ignore

    ctx = apc.context
    # atexit.unregister(apc._finish_up)

    import pycuda.driver as cuda  # type: ignore
    from pycuda.compiler import SourceModule  # type: ignore

    # cuda hash by Peter Schmidt-Nielsen
    mod = SourceModule(
        """
    #include <cstdint>
    static const int PER_BLOCK_THREAD_COUNT = %i;
    // Constants are chunks digits of pi, but |1 to be odd.
    __device__ static const uint32_t ODD_CONSTANT32[8]= { 0x243f6a89ul, 0x85a308d3ul, 0x13198a2ful, 0x03707345ul,
    0xa4093823ul, 0x299f31d1ul, 0x082efa99ul, 0xec4e6c89ul,
    };
    #define MIX_STATE(s) do { \
        s[1] ^= s[0] >> 15; s[1] *= ODD_CONSTANT32[0]; \
        s[2] ^= s[1] >> 15; s[2] *= ODD_CONSTANT32[1]; \
        s[3] ^= s[2] >> 15; s[3] *= ODD_CONSTANT32[2]; \
        s[4] ^= s[3] >> 15; s[4] *= ODD_CONSTANT32[3]; \
        s[5] ^= s[4] >> 15; s[5] *= ODD_CONSTANT32[4]; \
        s[6] ^= s[5] >> 15; s[6] *= ODD_CONSTANT32[5]; \
        s[7] ^= s[6] >> 15; s[7] *= ODD_CONSTANT32[6]; \
        s[0] ^= s[7] >> 15; s[0] *= ODD_CONSTANT32[7]; \
    } while(0)
    __global__ void _gpuhash(
    const uint8_t* data,
    int64_t length,
    uint32_t* block_hash_buffer
    ) {
    extern __shared__ uint32_t sdata[];
    // Each CUDA block hashes one contiguous sub-block of the data -- figure out ours.
    int64_t per_block_length = (length + (int64_t) gridDim.x - 1) / (int64_t) gridDim.x;
    int64_t our_limit = min(length, (1 + (int64_t) blockIdx.x) * per_block_length);
    uint32_t state[8] = { blockIdx.x, threadIdx.x, 1, 2, 3, 4, 5, 6 };
    for (int i = 0; i < 8; i++)
        MIX_STATE(state);
    // Hash the entire sub-block, with each thread having a 256-bit state.
    for (size_t i = threadIdx.x + blockIdx.x * per_block_length; i < our_limit; i += PER_BLOCK_THREAD_COUNT) {
        state[0] ^= data[i];
        MIX_STATE(state);
    }
    for (int i = 0; i < 8; i++)
        sdata[8 * threadIdx.x + i] = state[i];
    __syncthreads();
    // Reduce the individual thread hashes down into a single 256-bit block hash.
    #define REDUCE(n) \
        if (PER_BLOCK_THREAD_COUNT >= 2 * n) { \
        if (threadIdx.x < n) \
            for (int i = 0; i < 8; i++) \
                sdata[8 * threadIdx.x + i] ^= sdata[8 * (threadIdx.x + n) + i] * ODD_CONSTANT32[7 - i]; \
        __syncthreads(); \
        }
    static_assert(PER_BLOCK_THREAD_COUNT <= 1024, "block_size too large");
    REDUCE(512) REDUCE(256) REDUCE(128) REDUCE(64) REDUCE(32)
    REDUCE(16)  REDUCE(8)   REDUCE(4)   REDUCE(2)  REDUCE(1)
    if (threadIdx.x < 8)
        block_hash_buffer[8 * blockIdx.x + threadIdx.x] = sdata[threadIdx.x];
    }
    __global__ void _gpuhash_finalize(
    uint32_t* block_hash_buffer,
    int length
    ) {
    uint32_t state[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    // Absorb all of the 256-bit block hashes.
    for (int i = 0; i < length; i++) {
        state[0] ^= block_hash_buffer[i];
        for (int j = 0; j < 2; j++)
        MIX_STATE(state);
    }
    // Final mixing, and then output the 256-bit hash.
    for (int i = 0; i < 8; i++)
        MIX_STATE(state);
    block_hash_buffer[threadIdx.x] = state[threadIdx.x];
    }
    """
        % PER_BLOCK_THREAD_COUNT
    )

    _gpuhash = mod.get_function("_gpuhash")
    _gpuhash_finalize = mod.get_function("_gpuhash_finalize")
    return (_gpuhash, _gpuhash_finalize, ctx)


dtypes_to_bytes = {
    torch.float32: b"float32",
    torch.float64: b"float64",
    torch.float16: b"float16",
    torch.bool: b"bool---",
    torch.int8: b"int8---",
    torch.int16: b"int16--",
    torch.int32: b"int32--",
    torch.int64: b"int64--",
    torch.uint8: b"uint8--",
}


@functools.lru_cache(maxsize=None)
def int_tuple_to_bytes(s: Tuple[int, ...], b):
    for x in s:
        b += x.to_bytes(8, "big", signed=True)
    return b


import weakref

tensor_hash_cache: Dict[int, Tuple[weakref.ref[torch.Tensor], bytes]] = {}


def hash_tensor(tensor: torch.Tensor) -> bytes:
    if isinstance(tensor, torch.nn.Parameter):
        raise TypeError(
            f"Circuits assumes tensors are immutable, but this tensor " + "is a Parameter! You should call .clone()."
        )

    if (
        (cached := tensor_hash_cache.get(id(tensor))) is not None
        and (cached_tensor := cached[0]()) is not None
        and cached_tensor is tensor
    ):
        # print("hit hash cache!")
        return cached[1]
    # print(f"didn't hit hash cache {id(tensor)=}")
    hash_out = hash_tensor_impl(tensor)
    tensor_hash_cache[id(tensor)] = (weakref.ref(tensor), hash_out)
    return hash_out


def hash_tensor_impl(tensor):
    m = blake3(max_threads=blake3.AUTO)
    m.update(b"c3c5ce3e-7d06-4939-8340-2a7e6b2984b4")
    m.update(int_tuple_to_bytes(tensor.shape, b""))
    m.update(b"fb782993-5fe7-43ac-acc2-d57fb429c2fc")
    m.update(dtypes_to_bytes[tensor.dtype])
    m.update(str(tensor.device).encode("utf-8"))
    m.update(b"9430f5c7-3c58-4339-b863-5449a4d74ee6")
    if str(tensor.device) == "cpu":
        m.update(tensor.detach().numpy().tobytes())  # this copies, could be optimized more
    else:
        if True:
            m.update(tensor.detach().cpu().numpy().tobytes())
        else:
            try:
                m.update(hash_tensor_contents_gpu(tensor))
            except Exception as e:
                print("gpu hashing failed, maybe because cuda version < 11.4 or pycuda not installed")
                print(e)
                m.update(tensor.detach().cpu().numpy().tobytes())

    return m.digest()[:32]


def pop_cuda_context():
    _, _, ctx = get_cuda_kernels()
    ctx.pop()


def hash_tensor_contents_gpu(tensor):
    # this makes pycuda use the default context, whereas the
    # default pycuda.autoinit uses a new context which clashes with pytorch

    tensor = tensor.contiguous()
    data_length = tensor.element_size() * tensor.nelement()
    block_count = 32
    if data_length > 1e6:
        block_count = 64
    if data_length > 1e7:
        block_count = 128
    if data_length > 1e8:
        block_count = 256
    block_hash_buffer = torch.zeros(size=(block_count * 8,), dtype=torch.int32, device=tensor.device)
    _gpuhash, _gpuhash_finalize, _ = get_cuda_kernels()
    _gpuhash(
        tensor,
        np.int64(data_length),
        block_hash_buffer,
        grid=(block_count, 1, 1),
        block=(PER_BLOCK_THREAD_COUNT, 1, 1),
        shared=32 * PER_BLOCK_THREAD_COUNT,
    )
    _gpuhash_finalize(
        block_hash_buffer,
        np.int32(len(block_hash_buffer)),
        grid=(1, 1, 1),
        block=(8, 1, 1),
    )
    b = block_hash_buffer[:8].cpu().numpy().tobytes()
    return b


if __name__ == "__main__":
    # tensor1 = torch.randn(10, 10, device="cuda:0")
    tensor1 = torch.randn(4_000_000_000, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    import time

    for _ in range(3):
        t = time.time()
        hash_tensor_contents_gpu(tensor1)
        torch.cuda.synchronize()
        print(time.time() - t)
