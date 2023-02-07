import pytest
import torch

import rust_circuit as rc


def test_err_on_param():
    with pytest.raises(TypeError):
        rc.Array(torch.nn.Parameter(torch.randn(3)))
    with pytest.raises(TypeError):
        rc.hash_tensor(torch.nn.Parameter(torch.randn(3)))
    rc.Array(torch.nn.Parameter(torch.randn(3)).clone())
    rc.hash_tensor(torch.nn.Parameter(torch.randn(3)).clone())
