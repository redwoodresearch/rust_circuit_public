import uuid
from functools import lru_cache
from typing import *

import torch

import rust_circuit as rc

# based on https://github.com/huggingface/transformers/blob/6767ce71d661ae57afa38b6cd0d533e3dfd463fa/src/transformers/models/codegen/modeling_codegen.py
# note: different models do rotary pos emb somewhat differently

# NOTE: probably faster to just fuse this with the computation rather than cache
@lru_cache(maxsize=2)
def _rotary_embs(dim, seq_len, device):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    freqs = torch.einsum("i , j -> i j", torch.arange(seq_len, dtype=torch.float), inv_freq).to(device).float()
    sin = duplicate_interleave(freqs.sin())[None, :, None, :]
    cos = duplicate_interleave(freqs.cos())[None, :, None, :]
    return sin, cos


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m


class ApplyRotaryPosEmb(rc.GeneralFunctionSpecBase):
    @property
    def name(self) -> str:
        return "apply_rotary_pos_emb"

    @property
    def path(self) -> str:
        return "rust_circuit.generalfuncs.rotary_pos_emb:ApplyRotaryPosEmb"

    def compute_hash_bytes(self) -> bytes:
        return uuid.UUID("a00070be-b82b-495f-953f-633d0ab6829b").bytes

    def function(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """x can be either q or k"""
        seq_len, two, halfdim = x.shape[-3:]
        assert two == 2
        x = x.view(*x.shape[:-2], 1, -1)
        sin, cos = _rotary_embs(2 * halfdim, seq_len, device=x.device)
        x = (x * cos) + (rotate_every_two(x) * sin)
        x = x.view(*x.shape[:-2], 2, -1)
        return x

    def get_shape_info(self, x_shape: rc.Shape) -> rc.GeneralFunctionShapeInfo:  # type: ignore[override]
        if len(x_shape) < 3:
            raise ValueError(f"x must have at least 3 dimensions, but got {len(x_shape)}")
        if x_shape[-2] != 2:
            raise ValueError(f"Second to last dimension of x must be 2, but got {x_shape[-2]}")
        return rc.GeneralFunctionShapeInfo(
            x_shape,
            num_non_batchable_output_dims=3,
            input_batchability=[True],
        )

    @classmethod
    def new(cls, x: rc.Circuit, name: Optional[str] = None):
        """convenience function"""
        return rc.GeneralFunction(x, spec=cls(), name=name)
