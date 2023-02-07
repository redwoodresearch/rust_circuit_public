import torch

from interp.tools.indexer import I
from rust_circuit import (
    Add,
    Array,
    Concat,
    Einsum,
    Index,
    OptimizationSettings,
    Parser,
    Rearrange,
    RearrangeSpec,
    batch_einsum,
    batch_to_concat,
    propagate_named_axes,
)

P = Parser()


def test_named_axes():
    circs = [
        Einsum.from_einsum_string(
            "abc,acd,ade->ba",
            Array(torch.randn(128, 32, 32), "hi").rename_axes({0: "batch"}),
            Array.randn(128, 32, 32),
            Array.randn(128, 32, 32),
        ),
        Index(
            Einsum.from_einsum_string(
                "abc,acd,ade->ba",
                Array(torch.randn(128, 32, 32), "hi").rename_axes({0: "batch"}),
                Array.randn(128, 32, 32),
                Array.randn(128, 32, 32),
            ),
            I[0, :],
        ),
        Rearrange(
            Index(
                Einsum.from_einsum_string(
                    "abc,acd,ade->ba",
                    Array(torch.randn(128, 32, 32), "hi").rename_axes({0: "batch"}),
                    Array.randn(128, 32, 32),
                    Array.randn(128, 32, 32),
                ),
                I[0, :],
            ),
            RearrangeSpec([[0]], [[1], [0], [2]], [None, 2, 3]),
        ),
        Add(
            Rearrange(
                Index(
                    Einsum.from_einsum_string(
                        "abc,acd,ade->ba",
                        Array(torch.randn(128, 32, 32), "hi").rename_axes({0: "batch"}),
                        Array.randn(128, 32, 32),
                        Array.randn(128, 32, 32),
                    ),
                    I[0, :],
                ),
                RearrangeSpec([[0]], [[1], [0], [2]], [None, 2, 3]),
            ),
            Array(torch.randn(128, 3), "hi2").rename_axes({0: "batch2", 1: "rando"}),
        ),
        Concat(
            Rearrange(
                Index(
                    Einsum.from_einsum_string(
                        "abc,acd,ade->ba",
                        Array(torch.randn(128, 32, 32), "hi").rename_axes({0: "batch"}),
                        Array.randn(128, 32, 32),
                        Array.randn(128, 32, 32),
                    ),
                    I[0, :],
                ),
                RearrangeSpec([[0]], [[1], [0], [2]], [None, 2, 3]),
            ),
            Array(torch.randn(2, 128, 3), "hi2").rename_axes({1: "batch2", 2: "rando"}),
            axis=1,
        ),
    ]
    for circ in circs:
        circ.print()
        propagate_named_axes(circ, {}, True).print()


def test_batching_top():
    circs = [
        Einsum.from_einsum_string(
            "ab,bc->ac",
            Add(
                Array.randn(2, 4),
                Array.randn(2, 4),
            ),
            Array.randn(4, 8),
        ),
        P(
            """
0 Einsum ab,bc->ac
  2 Add
    3 [2, 3] Scalar 1.0
    4 [2, 3] Scalar 1.1
  5 Einsum ab,->ab
    9 [3, 5] Scalar 1.2
    6 Einsum ab->
      3
  """
        ),
        P(
            """
0 Einsum ab,bc->ac
  1 Add
    2 [2, 4] Scalar 1.0
    3 [2, 4] Scalar 1.1
  4 [4, 8] Scalar 1.2
  """
        ),
        P(
            """
'outer_b.bilinear_mlp' Module
  'bilinear_mlp' Einsum h,h,oh->o
    'bilinear_mlp.pre0' Index [0,:]
      'bilinear_mlp.fold_pre' Rearrange (a:2 b) -> a:2 b
        'bilinear_mlp.pre' Einsum i,hi->h
          'bilinear_mlp.input' [0s] Symbol 873a937e-2bb9-4f7f-b55e-a100db3dde52
          'bilinear_mlp.w.proj_in' [2*1s,0s] Symbol c171d519-8793-4a8b-ac5e-d550347f30a6
    'bilinear_mlp.pre1' Index [1,:]
      'bilinear_mlp.fold_pre'
    'bilinear_mlp.w.proj_out' [2s,1s] Symbol e61637eb-9f17-4325-b2c2-5eb2518026cf
  0 [64,17] Array rand ! 'bilinear_mlp.input'
  'b.proj_in' [22,17] Array rand ! 'bilinear_mlp.w.proj_in' ftt
  'b.proj_out' [13,11] Array rand ! 'bilinear_mlp.w.proj_out' ftt
  """
        ),
    ]
    for circ in circs:
        print("Circ")
        circ.print()
        batched = batch_to_concat(circ, 0, 2)
        print(batched)
        batched.print()


def test_batch_einsum():
    settings = OptimizationSettings(max_memory=4_000_000, max_single_tensor_memory=400_000)
    circ = P(
        """
0 Einsum ab,bc,bc->
  1 [1000,1000] Scalar 1.0
  2 [1000,1000] Scalar 1.1
  3 [1000,1000] Scalar 1.2"""
    )
    batched = batch_einsum(circ.cast_einsum(), settings)
    batched.print()


if __name__ == "__main__":
    test_named_axes()
    # test_batching_top()
    # test_batch_einsum()
