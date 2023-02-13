import pytest
import torch

from interp.circuit.computational_node import normalize_index
from rust_circuit import Array, Cumulant, DiscreteVar, GeneralFunction, RunDiscreteVarAllSpec, Sampler
from rust_circuit.py_utils import make_index_at


def test_gen_index_at_batch_x():
    torch.manual_seed(1324)
    x_shape = [1, 4, 2, 5, 7]
    index_shape = [1, 4, 2]
    for index_dim in [0, 1, -2, -1]:
        suffix_shape = x_shape[len(index_shape) :]
        use_index_dim = normalize_index(index_dim, len(suffix_shape))
        suffix_post_index_shape = suffix_shape[:use_index_dim] + suffix_shape[use_index_dim + 1 :]
        x_raw = torch.randn(*x_shape, dtype=torch.float64)
        index_raw = torch.randint(low=0, high=x_shape[use_index_dim + len(index_shape)], size=index_shape)

        x = Array(x_raw, name="x")
        index = Array(index_raw.long(), name="index")

        indexed = GeneralFunction.gen_index(x=x, index=index, index_dim=index_dim, batch_x=True)
        evaluated = indexed.evaluate()
        assert evaluated.shape == indexed.shape

        answer = torch.zeros(*index_shape, *suffix_post_index_shape, dtype=torch.float64)
        assert answer.shape == evaluated.shape
        for i in range(index_shape[0]):
            for j in range(index_shape[1]):
                for k in range(index_shape[2]):
                    answer[i, j, k] = x_raw[
                        (i, j, k) + make_index_at(index_raw[i, j, k], use_index_dim)
                    ]  # of course this doesn't generalise to e.g hypothesis tests : )

        torch.testing.assert_close(answer, evaluated)


def test_gen_index_at_batch_x_discrete_var():
    torch.manual_seed(1324)
    x_shape = [12, 4, 2, 5, 7]
    index_shape = [12, 4, 2]
    for index_dim in [0, 1, -2, -1]:
        suffix_shape = x_shape[len(index_shape) :]
        use_index_dim = normalize_index(index_dim, len(suffix_shape))
        suffix_post_index_shape = suffix_shape[:use_index_dim] + suffix_shape[use_index_dim + 1 :]

        x_raw = torch.randn(*x_shape, dtype=torch.float64)
        index_raw = torch.randint(low=0, high=x_shape[use_index_dim + len(index_shape)], size=index_shape)

        x = Array(x_raw, name="x")
        index = Array(index_raw, name="index")

        var_x = DiscreteVar(x, name="var_x")
        var_index = DiscreteVar(index, name="var_index", probs_and_group=var_x.probs_and_group)
        indexed = GeneralFunction.gen_index(x=var_x, index=var_index, index_dim=index_dim, batch_x=True)

        cumulant = Cumulant(indexed, name="cumulant")

        evaluated = (
            Sampler(
                RunDiscreteVarAllSpec(
                    (var_x.probs_and_group,),
                )
            )
            .estimate(cumulant)
            .evaluate()
        )
        assert evaluated.shape == indexed.shape

        answer = torch.zeros(*index_shape, *suffix_post_index_shape, dtype=torch.float64)
        assert answer[0].shape == evaluated.shape
        for i in range(index_shape[0]):
            for j in range(index_shape[1]):
                for k in range(index_shape[2]):
                    answer[i, j, k] = x_raw[
                        (i, j, k) + make_index_at(index_raw[i, j, k], use_index_dim)
                    ]  # of course this doesn't generalise to e.g hypothesis tests : )

        mean_answer = answer.mean(0).cpu()
        torch.testing.assert_close(mean_answer, evaluated.cpu())


def test_gen_index_at_no_batch_x():
    torch.manual_seed(1324)
    x_shape = [5, 7]
    index_shape = [1, 4, 2]
    for index_dim in [0, 1, -2, -1]:
        use_index_dim = normalize_index(index_dim, len(x_shape))
        post_index_shape = x_shape[:use_index_dim] + x_shape[use_index_dim + 1 :]
        x_raw = torch.randn(*x_shape, dtype=torch.float64)
        index_raw = torch.randint(low=0, high=x_shape[use_index_dim], size=index_shape)

        x = Array(x_raw, name="x")
        index = Array(index_raw, name="index")

        indexed = GeneralFunction.gen_index(x=x, index=index, index_dim=index_dim, batch_x=False)
        evaluated = indexed.evaluate()
        assert evaluated.shape == indexed.shape

        answer = torch.zeros(*index_shape, *post_index_shape, dtype=torch.float64)
        assert answer.shape == evaluated.shape
        for i in range(index_shape[0]):
            for j in range(index_shape[1]):
                for k in range(index_shape[2]):
                    answer[i, j, k] = x_raw[
                        make_index_at(index_raw[i, j, k], use_index_dim)
                    ]  # of course this doesn't generalise to e.g hypothesis tests : )

        torch.testing.assert_close(answer, evaluated)


@pytest.mark.parametrize("batch_x", [False, True])
def test_gen_index_at_no_index_batch(batch_x: bool):
    torch.manual_seed(1324)
    x_shape = [5, 7]
    index_shape = [1, 4, 2]
    for index_dim in [0, 1, -2, -1]:
        use_index_dim = normalize_index(index_dim, len(x_shape))
        pre_index_shape = x_shape[:use_index_dim]
        post_index_shape = x_shape[use_index_dim + 1 :]
        x_raw = torch.randn(*x_shape, dtype=torch.float64)
        index_raw = torch.randint(low=0, high=x_shape[use_index_dim], size=index_shape)

        x = Array(x_raw, name="x")
        index = Array(index_raw, name="index")

        if batch_x and index_dim >= 0:
            continue

        indexed = GeneralFunction.gen_index(x=x, index=index, index_dim=index_dim, batch_x=batch_x, batch_index=False)
        evaluated = indexed.evaluate()
        assert evaluated.shape == indexed.shape

        answer = torch.zeros(*pre_index_shape, *index_shape, *post_index_shape, dtype=torch.float64)
        assert answer.shape == evaluated.shape
        for i in range(index_shape[0]):
            for j in range(index_shape[1]):
                for k in range(index_shape[2]):
                    answer[(*[slice(None) for _ in pre_index_shape], i, j, k)] = x_raw[
                        make_index_at(index_raw[i, j, k], use_index_dim)
                    ]  # of course this doesn't generalise to e.g hypothesis tests : )

        torch.testing.assert_close(answer, evaluated)


def test_explicit_index():
    torch.manual_seed(1324)
    x_shape = [1, 4, 2, 5, 7]
    index_shape = [1, 4, 2]
    for x_non_batch_dims in range(2, len(x_shape)):
        for index_dim in range(-x_non_batch_dims, x_non_batch_dims):
            x_non_batch_shape = x_shape[-x_non_batch_dims:]
            num_batch_dims = len(x_shape) - x_non_batch_dims
            batch_shape = x_shape[:num_batch_dims]
            index_non_batch_shape = index_shape[num_batch_dims:]
            use_index_dim = normalize_index(index_dim, x_non_batch_dims)
            suffix_post_index_shape = x_non_batch_shape[:use_index_dim] + x_non_batch_shape[use_index_dim + 1 :]
            x_raw = torch.randn(*x_shape, dtype=torch.float64)
            index_raw = torch.randint(low=0, high=x_non_batch_shape[use_index_dim], size=index_shape)

            x = Array(x_raw, name="x")
            index = Array(index_raw.long(), name="index")

            indexed = GeneralFunction.explicit_index(
                x=x, index=index, index_dim=index_dim, x_non_batch_dims=x_non_batch_dims
            )
            evaluated = indexed.evaluate()
            assert evaluated.shape == indexed.shape

            answer = torch.zeros(*batch_shape, *index_non_batch_shape, *suffix_post_index_shape, dtype=torch.float64)
            assert answer.shape == evaluated.shape
            for i in range(index_shape[0]):
                for j in range(index_shape[1]):
                    for k in range(index_shape[2]):
                        answer[i, j, k] = x_raw[
                            (i, j, k)[:num_batch_dims] + make_index_at(index_raw[i, j, k], use_index_dim)
                        ]  # of course this doesn't generalise to e.g hypothesis tests : )

            torch.testing.assert_close(answer, evaluated)
