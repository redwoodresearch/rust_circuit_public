import einops
import torch

import rust_circuit as rc

torch.manual_seed(0)


def test_conv():
    filter = rc.Array(torch.tensor([[[1], [1], [1]]], dtype=torch.float32))
    input = rc.Array.randn(2, 10, 1)
    print(input.shape, filter.shape)
    print("input", input)
    convy = rc.Conv(input, filter, stride=[1], padding=[(0, 1)])
    result = convy.evaluate()
    print("result", result)
    assert convy.shape == result.shape


def test_conv_2d():
    filter = torch.ones(11, 2, 2, 7)
    input = torch.randn(2, 3, 5, 7)
    raw_test_conv(input, filter, [1, 1], [(1, 1), (1, 1)])
    raw_test_conv(input, filter, [2, 1], [(0, 0), (0, 0)])
    raw_test_conv(torch.randn(10, 30, 40, 19), torch.randn(35, 5, 4, 19), [2, 1], [(0, 0), (0, 0)])
    raw_test_conv(torch.randn(10, 30, 40, 8), torch.randn(16, 5, 4, 8), [2, 1], [(0, 0), (0, 0)])


def raw_test_conv(input, filter, stride, padding):
    convy = rc.Conv(rc.Array(input), rc.Array(filter), stride, padding)
    rearranged_input = einops.rearrange(input, "a b c d -> a d b c")
    rearranged_filter = einops.rearrange(filter, "a b c d->a d b c")
    result = convy.evaluate()
    torch_result_rearranged = torch.nn.functional.conv2d(
        rearranged_input, rearranged_filter, None, stride, tuple([x[0] for x in padding])
    )
    torch_result = einops.rearrange(torch_result_rearranged, "a b c d -> a c d b")
    torch.testing.assert_close(result, torch_result)


if __name__ == "__main__":
    # testy1()
    test_conv()
    test_conv_2d()
