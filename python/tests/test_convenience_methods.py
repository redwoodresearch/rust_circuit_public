import pytest
import torch

from interp.tools.indexer import I
from rust_circuit import Array, MiscInputItemOutOfBoundsError


def test_convenience_methods():
    a = torch.tensor([[0.0, 1], [2, 3]])
    b = torch.tensor([[0.0, 10], [20, 30]])
    circ = Array(a)
    circ2 = Array(b)

    # circ.add(circ2).print(True, True)
    torch.testing.assert_close(circ.add(circ2).evaluate(), a + b)
    # circ.sub(circ2).print(True, True)
    torch.testing.assert_close(circ.sub(circ2).evaluate(), a - b)
    # circ.mean(0).print(True, True)
    # circ.mean().print(True, True)
    # circ.mean(1, "hooo").print(True, True)
    # circ.min(1, "hooo").print(True, True)
    # circ.max(1, "hooo").print(True, True)

    torch.testing.assert_close(circ.mean().evaluate(), a.mean())
    torch.testing.assert_close(circ.sum().evaluate(), a.sum())
    torch.testing.assert_close(circ.min().evaluate(), a.amin())
    torch.testing.assert_close(circ.max().evaluate(), a.amax())
    torch.testing.assert_close(circ.max(axis=0).evaluate(), a.amax(dim=0))
    torch.testing.assert_close(circ.max(axis=1).evaluate(), a.amax(dim=1))
    torch.testing.assert_close(circ.max(axis=-1).evaluate(), a.amax(dim=-1))
    torch.testing.assert_close(circ.max(axis=-2).evaluate(), a.amax(dim=-2))
    torch.testing.assert_close(circ.max(axis=(0, 1)).evaluate(), a.amax(dim=(0, 1)))
    torch.testing.assert_close(circ.max(axis=(0, -1)).evaluate(), a.amax(dim=(0, -1)))
    torch.testing.assert_close(circ.max(axis=(-2, -1)).evaluate(), a.amax(dim=(-2, -1)))
    torch.testing.assert_close(circ.mean(axis=(0, 1)).evaluate(), a.mean(dim=(0, 1)))
    torch.testing.assert_close(circ.mean(axis=0).evaluate(), a.mean(dim=0))
    torch.testing.assert_close(circ.mean(axis=1).evaluate(), a.mean(dim=1))
    torch.testing.assert_close(circ.mean(axis=-1).evaluate(), a.mean(dim=-1))

    torch.testing.assert_close(circ.mul(circ2).evaluate(), a * b)
    torch.testing.assert_close(circ.index(I[0:1, 1]).evaluate(), a[0:1, 1])


def test_squeeze():
    circ = Array.randn(1, 3, 1)
    circ.squeeze(0).print()
    circ.squeeze([0, 2]).print()
    with pytest.raises(RuntimeError):
        circ.squeeze([0, 1]).print()
    with pytest.raises(MiscInputItemOutOfBoundsError):
        circ.squeeze(3).print()


if __name__ == "__main__":
    test_convenience_methods()
