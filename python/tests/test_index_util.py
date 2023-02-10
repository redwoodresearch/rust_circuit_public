import pytest
import torch

from rust_circuit import Add, Array
from rust_circuit.index_util import Gather, Scatter

#
# Test Gather
#


def test_gather_1():
    specs = [
        "out[i, j] = activations[i, positions[i], j]",  # explicit batching
        "out[i] = activations[positions, i]",  # implicit batching
    ]

    for spec in specs:
        N, M, K = 3, 4, 5

        activations = torch.randn(N, M, K)
        positions = torch.randint(low=0, high=M, size=(N,))

        out_ref = torch.zeros(N, K)
        for i in range(N):
            for j in range(K):
                out_ref[i, j] = activations[i, positions[i], j]

        gather = Gather.new(
            spec,
            {"activations": Array(activations, name="activations"), "positions": Array(positions, name="positions")},
            name="gather",
        )
        out = gather.evaluate()
        torch.testing.assert_close(out_ref, out)


def test_gather_2():
    spec = "out[i, j, k] = x[y[i, j, k], j, k]"

    N, M, K = 3, 4, 5

    x = torch.randn(N, M, K)
    y = torch.randint(low=0, high=N, size=(N, M, K))

    out_ref = torch.zeros(N, M, K)
    for i in range(N):
        for j in range(M):
            for k in range(K):
                out_ref[i, j, k] = x[y[i, j, k], j, k]

    gather = Gather.new(
        spec,
        {"x": Array(x, name="x"), "y": Array(y, name="y")},
        name="gather",
    )
    out = gather.evaluate()
    torch.testing.assert_close(out_ref, out)


def test_gather_3():
    spec = "out[i, j, k] = x[i, y[i, j, k], k]"

    N, M, K = 3, 4, 5

    x = torch.randn(N, M, K)
    y = torch.randint(low=0, high=M, size=(N, M, K))

    out_ref = torch.zeros(N, M, K)
    for i in range(N):
        for j in range(M):
            for k in range(K):
                out_ref[i, j, k] = x[i, y[i, j, k], k]

    gather = Gather.new(
        spec,
        {"x": Array(x, name="x"), "y": Array(y, name="y")},
        name="gather",
    )
    out = gather.evaluate()
    torch.testing.assert_close(out_ref, out)


def test_gather_4():
    spec = "out[i, j, k] = x[i, j, y[i, j, k]]"

    N, M, K = 3, 4, 5

    x = torch.randn(N, M, K)
    y = torch.randint(low=0, high=K, size=(N, M, K))

    out_ref = torch.zeros(N, M, K)
    for i in range(N):
        for j in range(M):
            for k in range(K):
                out_ref[i, j, k] = x[i, j, y[i, j, k]]

    gather = Gather.new(
        spec,
        {"x": Array(x, name="x"), "y": Array(y, name="y")},
        name="gather",
    )
    out = gather.evaluate()
    torch.testing.assert_close(out_ref, out)


def test_gather_5():
    specs = ["out = x[y]", "out[] = x[y]", "out = x[y[]]", "out[] = x[y[]]"]

    N, M, K = 3, 4, 5

    x = torch.randn(N, M, K)
    y = torch.randint(low=0, high=K, size=(N, M))

    out_ref = torch.zeros(N, M)
    for i in range(N):
        for j in range(M):
            out_ref[i, j] = x[i, j, y[i, j]]

    for spec in specs:
        gather = Gather.new(
            spec,
            {"x": Array(x, name="x"), "y": Array(y, name="y")},
            name="gather",
        )

        out = gather.evaluate()
        torch.testing.assert_close(out_ref, out)


def test_gather_6():
    spec = "out[i] = x[i, i, y[i]]"

    N, M = 3, 4

    x = torch.randn(N, N, M)
    y = torch.randint(low=0, high=M, size=(N,))

    out_ref = torch.zeros(N)
    for i in range(N):
        out_ref[i] = x[i, i, y[i]]

    gather = Gather.new(
        spec,
        {"x": Array(x, name="x"), "y": Array(y, name="y")},
        name="gather",
    )
    out = gather.evaluate()
    torch.testing.assert_close(out_ref, out)


def test_gather_7():
    spec = "out[i] = x[i, y[i, i]]"

    N, M = 3, 4

    x = torch.randn(N, M)
    y = torch.randint(low=0, high=M, size=(N, N))

    out_ref = torch.zeros(N)
    for i in range(N):
        out_ref[i] = x[i, y[i, i]]

    gather = Gather.new(
        spec,
        {"x": Array(x, name="x"), "y": Array(y, name="y")},
        name="gather",
    )
    out = gather.evaluate()
    torch.testing.assert_close(out_ref, out)


def test_gather_8():
    spec = "out[i, j] = x[0, j, y[1, i], 2]"

    N0, N1, N2, N3 = 3, 4, 5, 6
    M0, M1 = 7, 8

    x = torch.randn(N0, N1, N2, N3)
    y = torch.randint(low=0, high=N2, size=(M0, M1))

    out_ref = torch.zeros(M1, N1)
    for i in range(M1):
        for j in range(N1):
            out_ref[i, j] = x[0, j, y[1, i], 2]

    gather = Gather.new(
        spec,
        {"x": Array(x, name="x"), "y": Array(y, name="y")},
        name="gather",
    )
    out = gather.evaluate()
    torch.testing.assert_close(out_ref, out)


def test_gather_9():
    spec0 = "out = logits[logit_indices[0]]"
    spec1 = "out = logits[logit_indices[1]]"

    N, M = 8, 10

    logits = torch.randn(N, M)
    logit_indices = torch.randint(low=0, high=M, size=(N, 2))

    out_ref = torch.zeros(N)
    for i in range(N):
        out_ref[i] = logits[i, logit_indices[i, 0]] - logits[i, logit_indices[i, 1]]

    logits_token0 = Gather.new(
        spec0,
        {"logits": Array(logits, name="logits"), "logit_indices": Array(logit_indices, name="logit_indices")},
        name="logits_token0",
    )
    logits_token1 = Gather.new(
        spec1,
        {"logits": Array(logits, name="logits"), "logit_indices": Array(logit_indices, name="logit_indices")},
        name="logits_token1",
    )
    diff = Add.minus(logits_token0, logits_token1, name="diff")
    out = diff.evaluate()
    torch.testing.assert_close(out_ref, out)


def test_gather_10():
    spec = "out[i, j, k] = foo[k, bar[j, k, i], i, j]"

    I, J, K, L = 3, 4, 5, 6

    foo = torch.randn(K, L, I, J)
    bar = torch.randint(low=0, high=L, size=(J, K, I))

    out_ref = torch.zeros(I, J, K)
    for i in range(I):
        for j in range(J):
            for k in range(K):
                out_ref[i, j, k] = foo[k, bar[j, k, i], i, j]

    gather = Gather.new(
        spec,
        {"foo": Array(foo, name="foo"), "bar": Array(bar, name="bar")},
        name="gather",
    )
    out = gather.evaluate()
    torch.testing.assert_close(out_ref, out)


def test_gather_11():
    spec = "out[i, j] = foo[bar[j]]"

    N, M, K = 3, 4, 5

    foo = torch.randn(K)
    bar = torch.randint(low=0, high=K, size=(M,))

    out_ref = torch.zeros(N, M)
    for i in range(N):
        for j in range(M):
            out_ref[i, j] = foo[bar[j]]

    def make_gather(out_shape):
        return Gather.new(
            spec,
            {"foo": Array(foo, name="foo"), "bar": Array(bar, name="bar")},
            name="gather",
            out_shape=out_shape,
        )

    with pytest.raises(Exception, match="Could not infer"):
        make_gather(None)

    for out_shape in [(N, None), (N, M)]:
        out = make_gather(out_shape).evaluate()
        torch.testing.assert_close(out_ref, out)


#
# Test Scatter
#


def test_scatter_1():
    specs = [
        "activations[i, positions[i], j] <- extracted[i, j]",
        "activations[positions, j] <- extracted[j]",
    ]

    N, M, K = 3, 4, 5

    activations = torch.randn(N, M, K)
    positions = torch.randint(low=0, high=M, size=(N,))
    extracted = torch.randn(N, K)

    for reduce in [None, "add", "multiply"]:
        out_ref = activations.clone()
        for i in range(N):
            for j in range(K):
                if reduce is None:
                    out_ref[i, positions[i], j] = extracted[i, j]
                elif reduce == "add":
                    out_ref[i, positions[i], j] += extracted[i, j]
                elif reduce == "multiply":
                    out_ref[i, positions[i], j] *= extracted[i, j]

        for spec in specs:
            scatter = Scatter.new(
                spec,
                {
                    "activations": Array(activations, name="activations"),
                    "positions": Array(positions, name="positions"),
                    "extracted": Array(extracted, name="extracted"),
                },
                name="scatter",
                reduce=reduce,
            )
            out = scatter.evaluate()
            torch.testing.assert_close(out_ref, out)


def test_scatter_2():
    spec = "dst[pos[i, j, k], j, k] <- src[i, j, k]"

    N, M, K = 3, 4, 5

    dst = torch.randn(N, M, K)
    src = torch.randn(N, M, K)
    pos = torch.randint(low=0, high=N, size=(N, M, K))

    for reduce in [None, "add", "multiply"]:
        out_ref = dst.clone()
        for i in range(N):
            for j in range(M):
                for k in range(K):
                    if reduce is None:
                        out_ref[pos[i, j, k], j, k] = src[i, j, k]
                    elif reduce == "add":
                        out_ref[pos[i, j, k], j, k] += src[i, j, k]
                    elif reduce == "multiply":
                        out_ref[pos[i, j, k], j, k] *= src[i, j, k]

        scatter = Scatter.new(
            spec,
            {"dst": Array(dst, name="dst"), "src": Array(src, name="src"), "pos": Array(pos, name="pos")},
            name="scatter",
            reduce=reduce,
        )
        out = scatter.evaluate()
        torch.testing.assert_close(out_ref, out)


def test_scatter_3():
    spec = "dst[i, pos[i, j, k], k] <- src[i, j, k]"

    N, M, K = 3, 4, 5

    dst = torch.randn(N, M, K)
    src = torch.randn(N, M, K)
    pos = torch.randint(low=0, high=M, size=(N, M, K))

    out_ref = dst.clone()
    for i in range(N):
        for j in range(M):
            for k in range(K):
                out_ref[i, pos[i, j, k], k] = src[i, j, k]

    scatter = Scatter.new(
        spec,
        {"dst": Array(dst, name="dst"), "src": Array(src, name="src"), "pos": Array(pos, name="pos")},
        name="scatter",
    )
    out = scatter.evaluate()
    torch.testing.assert_close(out_ref, out)


def test_scatter_4():
    spec = "dst[i, j, pos[i, j, k]] <- src[i, j, k]"

    N, M, K = 3, 4, 5

    dst = torch.randn(N, M, K)
    src = torch.randn(N, M, K)
    pos = torch.randint(low=0, high=K, size=(N, M, K))

    out_ref = dst.clone()
    for i in range(N):
        for j in range(M):
            for k in range(K):
                out_ref[i, j, pos[i, j, k]] = src[i, j, k]

    scatter = Scatter.new(
        spec,
        {"dst": Array(dst, name="dst"), "src": Array(src, name="src"), "pos": Array(pos, name="pos")},
        name="scatter",
    )
    out = scatter.evaluate()
    torch.testing.assert_close(out_ref, out)


def test_scatter_5():
    specs = ["dst[pos] <- src", "dst[pos[]] <- src", "dst[pos] <- src[]", "dst[pos[]] <- src[]"]

    N, M, K = 3, 4, 5

    dst = torch.randn(N, M, K)
    src = torch.randn(N, M)
    pos = torch.randint(low=0, high=K, size=(N, M))

    out_ref = dst.clone()
    for i in range(N):
        for j in range(M):
            out_ref[i, j, pos[i, j]] = src[i, j]

    for spec in specs:
        scatter = Scatter.new(
            spec,
            {"dst": Array(dst, name="dst"), "src": Array(src, name="src"), "pos": Array(pos, name="pos")},
            name="scatter",
        )
        out = scatter.evaluate()
        torch.testing.assert_close(out_ref, out)


def test_scatter_6():
    spec = "dst[i, i, pos[i]] <- src[i]"

    N, M = 3, 4

    dst = torch.randn(N, N, M)
    src = torch.randn(N)
    pos = torch.randint(low=0, high=M, size=(N,))

    for reduce in [None, "add", "multiply"]:
        out_ref = dst.clone()
        for i in range(N):
            if reduce is None:
                out_ref[i, i, pos[i]] = src[i]
            elif reduce == "add":
                out_ref[i, i, pos[i]] += src[i]
            elif reduce == "multiply":
                out_ref[i, i, pos[i]] *= src[i]

        scatter = Scatter.new(
            spec,
            {"dst": Array(dst, name="dst"), "src": Array(src, name="src"), "pos": Array(pos, name="pos")},
            name="scatter",
            reduce=reduce,
        )
        out = scatter.evaluate()
        torch.testing.assert_close(out_ref, out)


def test_scatter_7():
    spec = "dst[i, pos[i, i]] <- src[i, i]"

    N, M = 3, 4

    dst = torch.randn(N, M)
    src = torch.randn(N, N)
    pos = torch.randint(low=0, high=M, size=(N, N))

    for reduce in [None, "add", "multiply"]:
        out_ref = dst.clone()
        for i in range(N):
            if reduce is None:
                out_ref[i, pos[i, i]] = src[i, i]
            elif reduce == "add":
                out_ref[i, pos[i, i]] += src[i, i]
            elif reduce == "multiply":
                out_ref[i, pos[i, i]] *= src[i, i]

        scatter = Scatter.new(
            spec,
            {"dst": Array(dst, name="dst"), "src": Array(src, name="src"), "pos": Array(pos, name="pos")},
            name="scatter",
            reduce=reduce,
        )
        out = scatter.evaluate()
        torch.testing.assert_close(out_ref, out)


def test_scatter_8():
    spec = "dst[0, j, pos[1, i], 2] <- src[i, 3, j]"

    I, J = 3, 4
    N = 5

    dst = torch.randn(2, J, N, 4)
    src = torch.randn(I, 5, J)
    pos = torch.randint(low=0, high=N, size=(3, I))

    for reduce in [None, "add", "multiply"]:
        out_ref = dst.clone()
        for i in range(I):
            for j in range(J):
                if reduce is None:
                    out_ref[0, j, pos[1, i], 2] = src[i, 3, j]
                elif reduce == "add":
                    out_ref[0, j, pos[1, i], 2] += src[i, 3, j]
                elif reduce == "multiply":
                    out_ref[0, j, pos[1, i], 2] *= src[i, 3, j]

        scatter = Scatter.new(
            spec,
            {"dst": Array(dst, name="dst"), "src": Array(src, name="src"), "pos": Array(pos, name="pos")},
            name="scatter",
            reduce=reduce,
        )
        out = scatter.evaluate()
        torch.testing.assert_close(out_ref, out)


def test_scatter_9():
    spec = "dst[i, j, pos[i, j]] <- src[i, j, k]"

    N, I, J, K = 3, 4, 5, 6

    dst = torch.randn(I, J, N)
    src = torch.randn(I, J, K)
    pos = torch.randint(low=0, high=N, size=(I, J))

    with pytest.raises(ValueError, match="missing"):
        Scatter.new(
            spec,
            {"dst": Array(dst, name="dst"), "src": Array(src, name="src"), "pos": Array(pos, name="pos")},
            name="scatter",
        )


def test_scatter_10():
    spec = "dst[i, j, pos[i, j, k, l]] <- src[i, j, k, l]"

    I, J, K, L = 3, 4, 5, 6
    N = 7

    dst = torch.randn(I, J, N)
    src = torch.randn(I, J, K, L)
    pos = torch.randint(low=0, high=N, size=(I, J, K, L))

    with pytest.raises(ValueError, match="missing"):
        Scatter.new(
            spec,
            {"dst": Array(dst, name="dst"), "src": Array(src, name="src"), "pos": Array(pos, name="pos")},
            name="scatter",
        )
