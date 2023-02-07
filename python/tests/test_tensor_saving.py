import torch

from rust_circuit import *


def test_save():
    tensor = torch.randn(3, 3)
    save_tensor(tensor)
    prefix = Array(tensor).tensor_hash_base16()
    loaded = get_tensor_prefix(prefix)
    print(tensor, loaded, torch.allclose(tensor, loaded))
    sync_all_unsynced_tensors()


if __name__ == "__main__":
    sync_all_unsynced_tensors()
    # test_save()
