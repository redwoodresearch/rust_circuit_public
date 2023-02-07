from typing import Callable, Tuple

import torch

from rust_circuit import Array
from rust_circuit.causal_scrubbing.dataset import Dataset


def loss_fn(ds: Dataset, out_dict) -> float:
    if isinstance(out_dict, dict):
        out = list(out_dict.values())[0]
    else:
        out = out_dict
    out = out.to(torch.float)
    return torch.abs(out.cpu() - ds.labels.value).mean().item()


class IntDataset(Dataset):
    inp_tensor: torch.Tensor

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        object.__setattr__(self, "inp_tensor", self.xs.value)

    @classmethod
    def of_shape(cls, shape: Tuple, labeler: Callable[[torch.Tensor], torch.Tensor], maxint: int = 10):
        xs = torch.randint(high=maxint, size=shape)
        xs_arr = Array(xs, "xs")
        labels_arr = Array(torch.stack([labeler(x) for x in xs]), "labels")
        ds = cls({"xs": xs_arr, "labels": labels_arr})
        return ds

    def __str__(self):
        if len(self) > 1:
            return f"IntDataset(xs.shape={self.inp_tensor.shape})"
        else:
            xs = self.inp_tensor.item() if self.inp_tensor.numel() == 1 else self.inp_tensor[0].tolist()
            return f"IntDatum(xs={xs}, label={self.labels.value.item()})"
