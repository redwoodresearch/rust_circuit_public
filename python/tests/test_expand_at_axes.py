import pytest
import torch

import rust_circuit.optional as op
from interp.circuit.circuit import MemoizedFn
from interp.circuit.circuit_utils import evaluate_fn
from interp.circuit.computational_node import UnaryRearrange
from interp.circuit.constant import ArrayConstant
from rust_circuit import Array as rArray
from rust_circuit import Rearrange as rRearrange
from rust_circuit import RearrangeSpec as rRearrangeSpec


def test_repeat_at_axes():
    for (t, axes, counts) in [
        (torch.randn(()), (), ()),
        (torch.randn(()), (0,), (3,)),
        (torch.randn(()), (0,), (1,)),
        (torch.randn(()), (0,), None),
        (torch.randn(()), (0, 1, 2), None),
        (torch.randn(()), (0, 1, 2), (2, 1, 3)),
        (torch.randn((3,)), (), ()),
        (torch.randn((3,)), (0,), (2,)),
        (torch.randn((3,)), (0,), None),
        (torch.randn((3,)), (1,), (2,)),
        (torch.randn((3,)), (1,), None),
        (torch.randn((3,)), (0, 2), (2, 4)),
        (torch.randn((3,)), (0, 2), None),
        (torch.randn(3, 4, 2), (1, 2, 3, 5), None),
        (torch.randn(3, 4, 2), (1, 2, 3, 5), (3, 5, 1, 2)),
        (torch.randn(3, 4, 2), (), ()),
    ]:
        basic_r_args = [(axes, counts)]
        if len(axes) == 1:
            basic_r_args.append((axes[0], op.map(counts, lambda x: x[0])))

        new_py_node = UnaryRearrange.repeat_at_axes(ArrayConstant(t), axes, counts)
        for r_axes, r_counts in basic_r_args:
            new_rs_node = rRearrange(rArray(t), rRearrangeSpec.expand_at_axes(t.ndim, r_axes, r_counts))
            assert new_rs_node.shape == new_py_node.shape
            torch.testing.assert_close(new_rs_node.evaluate(), MemoizedFn(evaluate_fn())(new_py_node))

        new_py_node = UnaryRearrange.unsqueeze(ArrayConstant(t), axes)
        for r_axes, _ in basic_r_args:
            new_rs_node = rRearrange(rArray(t), rRearrangeSpec.unsqueeze(t.ndim, r_axes))
            assert new_rs_node.shape == new_py_node.shape
            torch.testing.assert_close(new_rs_node.evaluate(), MemoizedFn(evaluate_fn())(new_py_node))


def test_repeat_at_axes_fail():
    for (ndim, axes, counts) in [
        (0, (2,), ()),
        (0, (0,), (-1)),
        (3, (1, 2, 3, 5, 10), None),
        (3, (1, 2, 3, 5, 1000), None),
        (1000, (1, 2, 3, 5, 1000), None),
    ]:
        with pytest.raises((ValueError, TypeError)):
            rRearrangeSpec.expand_at_axes(ndim, axes, counts)
