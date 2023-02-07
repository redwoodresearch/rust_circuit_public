import random
from typing import Iterable, Optional, Tuple

import einops
import pytest
import torch

import rust_circuit as rc
from interp.tools.indexer import TORCH_INDEXER as I
from rust_circuit import (
    Add,
    Array,
    Circuit,
    Concat,
    Cumulant,
    DiscreteVar,
    Einsum,
    Parser,
    RandomSampleSpec,
    Rearrange,
    RunDiscreteVarAllSpec,
    Sampler,
    Scalar,
    Shape,
    Symbol,
    Tag,
    default_hash_seeder,
    factored_cumulant_expectation_rewrite,
    multinomial,
    optimize_and_evaluate,
)


def test_estim_cumulant_shape_correct(seed=23473):
    random.seed(seed)

    items = [
        Symbol.new_with_random_uuid(shape=()),
        Symbol.new_with_random_uuid(shape=(2,)),
        Symbol.new_with_random_uuid(shape=(3,)),
        Symbol.new_with_random_uuid(shape=(2, 3)),
        Symbol.new_with_random_uuid(shape=(4, 5)),
        Symbol.new_with_random_uuid(shape=(4, 5, 6, 7)),
        Symbol.new_with_random_uuid(shape=(8, 9, 10, 11, 12, 13, 17)),
        Symbol.new_with_random_uuid(shape=(18, 19, 20)),
    ]

    for with_var in [False, True]:
        if with_var:
            n_vals = 7
            probs = DiscreteVar.uniform_probs_and_group(n_vals)
            new_items = [DiscreteVar(x.expand_at_axes(0, n_vals), probs) for x in items]
            all_probs = [probs]
        else:
            all_probs = []
            new_items = items

        for count in range(7):
            for _ in range(5):
                random.shuffle(new_items)
                cumulant = Cumulant(*new_items[:count])
                transformed = Sampler(RunDiscreteVarAllSpec(all_probs)).estimate(cumulant)
                assert transformed.shape == cumulant.shape


def test_correct_cumulant_estimate(seed=283838):
    torch.manual_seed(seed)
    x = torch.randn(5, 3, 2, dtype=torch.float64)
    y = torch.nn.functional.softmax(x, dim=-1).sum(dim=1)
    z = x ** 3
    w = torch.sigmoid(x[:, :, 0])

    vals = [x, y, z, w]

    var_x = DiscreteVar.new_uniform(Array(x))
    vars_unweighted = [DiscreteVar(Array(v), var_x.probs_and_group) for v in vals]

    probs = torch.rand(5, dtype=torch.float64)
    probs = probs / probs.sum()

    probs_and_group = Array(probs)

    vars_weighted = [DiscreteVar(Array(v), probs_and_group) for v in vals]

    def estimate_and_comp(circuit, actual):
        # todo: add is_full to rust
        transformed = (
            Sampler(
                RunDiscreteVarAllSpec.create_full_from_circuits(circuit),
            )
            .estimate(circuit)
            .evaluate()
        )
        torch.testing.assert_close(
            transformed,
            actual,
            atol=1e-6,
            rtol=1e-6,
        )

    for all_vars, weights in [
        [vars_unweighted, torch.full((5,), 1 / 5, dtype=torch.float64)],
        [vars_weighted, probs],
    ]:
        var_x, var_y, var_z, var_w = all_vars
        all_means = [torch.einsum("i ..., i -> ...", v, weights).unsqueeze(0) for v in vals]
        c_x, c_y, c_z, c_w = [v - m for v, m in zip(vals, all_means)]
        x_mean = all_means[0].squeeze(0)
        estimate_and_comp(
            Cumulant(
                var_x,
            ),
            x_mean,
        )

        uncorrected_cov = torch.einsum("b i j, b k, b -> i j k", c_x, c_y, weights)
        estimate_and_comp(Cumulant(var_x, var_y), uncorrected_cov)

        uncorrected_third_cum = torch.einsum("b i j, b k, b l m, b -> i j k l m", c_x, c_y, c_z, weights)
        estimate_and_comp(Cumulant(var_x, var_y, var_z), uncorrected_third_cum)

        centered_4th_mom = torch.einsum("b i j, b k, b l m, b o, b -> i j k l m o", c_x, c_y, c_z, c_w, weights)
        first_pair = torch.einsum(
            "i j k, l m o -> i j k l m o",
            torch.einsum("b i j, b k, b -> i j k", c_x, c_y, weights),
            torch.einsum("b l m, b o, b -> l m o", c_z, c_w, weights),
        )
        snd_pair = torch.einsum(
            "i j l m, k o -> i j k l m o",
            torch.einsum("b i j, b l m, b -> i j l m", c_x, c_z, weights),
            torch.einsum("b k, b o, b-> k o", c_y, c_w, weights),
        )
        thrd_pair = torch.einsum(
            "i j o, k l m -> i j k l m o",
            torch.einsum("b i j, b o, b -> i j o", c_x, c_w, weights),
            torch.einsum("b k, b l m, b -> k l m", c_y, c_z, weights),
        )

        pair_sum = first_pair + snd_pair + thrd_pair

        uncorrected_4th_cum = centered_4th_mom - pair_sum

        estimate_and_comp(Cumulant(var_x, var_y, var_z, var_w), uncorrected_4th_cum)


def test_trivial_factored_cumulant_expectation_rewrite():
    assert factored_cumulant_expectation_rewrite(Cumulant()) == Cumulant()
    cum = Cumulant(DiscreteVar.new_uniform(Array.randn(3, 2)))
    assert factored_cumulant_expectation_rewrite(cum) == cum


def raw_test_factored_cumulant_expectation_rewrite_vs_basic_expectation(cumulant: Cumulant):
    transform = Sampler(RunDiscreteVarAllSpec.create_full_from_circuits(cumulant))
    torch.testing.assert_close(
        transform.estimate(factored_cumulant_expectation_rewrite(cumulant)).evaluate(),
        transform.estimate(cumulant).evaluate(),
    )


def test_factored_cumulant_expectation_rewrite_vs_basic_expectation_simple():
    # could add hypothesis test, but feels extra...
    shapes = (2, 3), (4, 5, 1), (2, 1, 4), (3, 2), (1, 3, 1, 1, 1), (2, 4, 1)
    n_vals = 11
    group = DiscreteVar.uniform_probs_and_group(n_vals)
    variables = [DiscreteVar(Array.randn(*((n_vals,) + s)), group) for s in shapes]

    for i in range(2, len(variables) + 1):
        raw_test_factored_cumulant_expectation_rewrite_vs_basic_expectation(Cumulant(*variables[:i]))


def make_discrete_var(
    values_shape: Shape, batch_size: int, name: Optional[str] = None, group: Optional[Circuit] = None
):
    vals = Array.randn(batch_size, *values_shape)
    out = DiscreteVar(vals, group, name)
    return out, vals.value, out.probs_and_group.evaluate()


def make_discrete_var_group(n: int, name: Optional[str] = None):
    x = torch.rand(n)
    x /= x.sum()
    return Tag.new_with_random_uuid(Array(x), name)


def make_discrete_var_test_circuits(
    seed=238832, just_mean: bool = False
) -> Iterable[Tuple[Circuit, torch.Tensor, torch.Tensor, Circuit]]:
    torch.manual_seed(seed)

    x, x_value, x_probs_and_group = make_discrete_var((3, 2), 17, "x")
    yield x, x_value, x_probs_and_group, x.probs_and_group

    y, y_value, _ = make_discrete_var((3, 42), 17, "y", group=x.probs_and_group)
    yield y, y_value, x_probs_and_group, x.probs_and_group

    z, z_value, _ = make_discrete_var((3, 7, 6, 2), 17, "z", group=x.probs_and_group)
    yield z, z_value, x_probs_and_group, x.probs_and_group

    z_rearrange = Rearrange.from_string(z, "a b c d -> d (c b) a", name="z_rearrange")
    z_rearrange_eval = einops.repeat(z_value, "batch a b c d -> batch d (c b) a")
    yield z_rearrange, z_rearrange_eval, x_probs_and_group, x.probs_and_group

    const_l = Array.randn(6, 2, name="const_l")
    const_r = Array.randn(3, 7, name="const_r")

    z_mul_const = Einsum.from_einsum_string("a b c d, c d, a b -> b c", z, const_l, const_r)
    z_mul_const_eval = torch.einsum("e a b c d, c d, a b -> e b c", z_value, const_l.value, const_r.value)
    yield z_mul_const, z_mul_const_eval, x_probs_and_group, x.probs_and_group

    const_single_l = Array.randn(4, name="const_single_r")
    const_single_r = Array.randn(4, name="const_single_r")

    simple_trace = Einsum.from_einsum_string("a,a->aa", const_single_l, const_single_r)
    simple_trace_eval = einops.repeat(
        torch.diag(torch.einsum("a, a -> a", const_single_l.value, const_single_r.value)),
        "a b -> c a b",
        c=x_probs_and_group.shape[0],
    )
    yield simple_trace, simple_trace_eval, x_probs_and_group, x.probs_and_group

    z_repeated_back_to_orig = z_mul_const.expand_at_axes((0, -1), (3, 2))
    z_repeated_back_to_orig_eval = einops.repeat(z_mul_const_eval, "batch b c -> batch a b c d", a=3, d=2)
    yield z_repeated_back_to_orig, z_repeated_back_to_orig_eval, x_probs_and_group, x.probs_and_group

    z_repeated_back_to_orig_other_order = z_mul_const.expand_at_axes((-1, 0), (2, 3))
    yield z_repeated_back_to_orig_other_order, z_repeated_back_to_orig_eval, x_probs_and_group, x.probs_and_group

    z_unsqueeze = z_mul_const.unsqueeze((0, -1))
    z_unsqueeze_eval = einops.repeat(z_mul_const_eval, "batch b c -> batch 1 b c 1")
    yield z_unsqueeze, z_unsqueeze_eval, x_probs_and_group, x.probs_and_group

    z_mul_const_rearrange = Rearrange.from_string(
        z_repeated_back_to_orig, "a b c d -> d (c b) a", name="z_mul_const_rearrange"
    )
    z_mul_const_rearrange_eval = einops.repeat(z_repeated_back_to_orig_eval, "batch a b c d -> batch d (c b) a")
    yield z_mul_const_rearrange, z_mul_const_rearrange_eval, x_probs_and_group, x.probs_and_group

    y_dup, y_dup_value, _ = make_discrete_var((3, 42), 17, "y_dup", group=x.probs_and_group)
    yield y_dup, y_dup_value, x_probs_and_group, x.probs_and_group

    y_sums = y.add(y_dup)
    y_sums_value = y_value + y_dup_value
    yield y_sums, y_sums_value, x_probs_and_group, x.probs_and_group

    new_group = make_discrete_var_group(7, name="non_uniform_group_and_probs")
    with_variable_weights, with_variable_weights_value, with_variable_weights_group = make_discrete_var(
        (2, 3), 7, "var_weights", group=new_group
    )
    yield (
        with_variable_weights,
        with_variable_weights_value,
        with_variable_weights_group,
        with_variable_weights.probs_and_group,
    )

    with_variable_weights_other, with_variable_weights_other_val, _ = make_discrete_var(
        (2,), 7, "var_weights_other", group=new_group
    )
    yield (
        with_variable_weights_other,
        with_variable_weights_other_val,
        with_variable_weights_group,
        with_variable_weights.probs_and_group,
    )

    index_x = x.index(I[1])
    yield index_x, x_value[:, 1], x_probs_and_group, x.probs_and_group

    index_variable_weights = with_variable_weights.index(I[1])
    yield index_variable_weights, with_variable_weights_value[
        :, 1
    ], with_variable_weights_group, with_variable_weights.probs_and_group

    different_first_dim, different_first_dim_value, _ = make_discrete_var(
        (7, 42),
        17,
        "different_first_dim",
        group=x.probs_and_group,
    )
    yield different_first_dim, different_first_dim_value, x_probs_and_group, x.probs_and_group

    yield Concat(y, axis=0), torch.cat([y_value], dim=1), x_probs_and_group, x.probs_and_group

    yield Concat(y, y_dup, different_first_dim, axis=0), torch.cat(
        [y_value, y_dup_value, different_first_dim_value], dim=1
    ), x_probs_and_group, x.probs_and_group

    yield Concat.stack(y, axis=-1), torch.stack([y_value], dim=-1), x_probs_and_group, x.probs_and_group

    yield Concat.stack(y, y_dup, y, y, axis=-1), torch.stack(
        [y_value, y_dup_value, y_value, y_value], dim=-1
    ), x_probs_and_group, x.probs_and_group

    if just_mean:
        return

    full_mul_str = "a b, a c, d c a, d c a -> d a b c"
    full_mul = Einsum.from_einsum_string(full_mul_str, x, y_sums, z_rearrange, z_mul_const_rearrange)
    full_mul_eval = torch.einsum(
        "e a b, e a c, e d c a, e d c a -> e d a b c",
        x_value,
        y_sums_value,
        z_rearrange_eval,
        z_mul_const_rearrange_eval,
    )
    yield full_mul, full_mul_eval, x_probs_and_group, x.probs_and_group

    # for power in [1, 2, 3, 7, 8, 9, 13]:
    #     power_circuit = Einsum.power(z_rearrange, power=power, name=f"pow_{power}")
    #     yield power_circuit, z_rearrange_eval ** power, x_probs_and_group, x.probs_and_group

    # outer_product_x_y = Einsum.outer_product(x, y)
    # outer_product_x_y_eval = torch.einsum("e a b, e c d -> e a b c d", x_value, y_value)
    # yield outer_product_x_y, outer_product_x_y_eval, x_probs_and_group, x.probs_and_group

    # outer_product_x_y_const_r = Einsum.outer_product(x, y, const_r)
    # outer_product_x_y_const_r_eval = torch.einsum(
    #     "e a b, e c d, g h  -> e a b c d g h", x_value, y_value, const_r.value
    # )
    # yield outer_product_x_y_const_r, outer_product_x_y_const_r_eval, x_probs_and_group, x.probs_and_group

    # outer_product_x_y_x_x = Einsum.outer_product(x, y, x, x)
    # outer_product_x_y_x_x_eval = torch.einsum(
    #     "e a b, e c d, e g h, e o i  -> e a b c d g h o i", x_value, y_value, x_value, x_value
    # )
    # yield outer_product_x_y_x_x, outer_product_x_y_x_x_eval, x_probs_and_group, x.probs_and_group

    # outer_product_x_y_batch = Einsum.outer_product(x, y, name="x_y_outer_batch", num_batch_dims=1)
    # outer_product_x_y_batch_eval = torch.einsum("e a b, e a d -> e a b d", x_value, y_value)
    # yield outer_product_x_y_batch, outer_product_x_y_batch_eval, x_probs_and_group, x.probs_and_group

    # outer_product_x_y_batch_sum = Einsum.outer_product(
    #     x, y, name="x_y_outer_batch_sum", num_batch_dims=1, sum_over_batch=True
    # )
    # outer_product_x_y_batch_sum_eval = torch.einsum("e a b, e a d -> e b d", x_value, y_value)
    # yield outer_product_x_y_batch_sum, outer_product_x_y_batch_sum_eval, x_probs_and_group, x.probs_and_group

    # outer_product_x_y_x_x_batch = Einsum.outer_product(x, y, x, x, name="x_y_x_x_outer_batch", num_batch_dims=1)
    # outer_product_x_y_x_x_batch_eval = torch.einsum(
    #     "e a b, e a d, e a h, e a i -> e a b d h i", x_value, y_value, x_value, x_value
    # )
    # yield outer_product_x_y_x_x_batch, outer_product_x_y_x_x_batch_eval, x_probs_and_group, x.probs_and_group

    # outer_product_x_y_x_x_batch_sum = Einsum.outer_product(
    #     x, y, x, x, name="x_y_x_x_outer_batch_sum", num_batch_dims=1, sum_over_batch=True
    # )
    # outer_product_x_y_x_x_batch_sum_eval = torch.einsum(
    #     "e a b, e a d, e a h, e a i -> e b d h i", x_value, y_value, x_value, x_value
    # )
    # yield outer_product_x_y_x_x_batch_sum, outer_product_x_y_x_x_batch_sum_eval, x_probs_and_group, x.probs_and_group

    # variable_weights_mul_str = "a b, a -> b a"
    # variable_weights_mul = Einsum.from_einsum_str(
    #     variable_weights_mul_str, with_variable_weights, with_variable_weights_other
    # )
    # variable_weights_mul_eval = torch.einsum(
    #     "e a b, e a -> e b a", with_variable_weights_value, with_variable_weights_other_val
    # )
    # yield (
    #     variable_weights_mul,
    #     variable_weights_mul_eval,
    #     with_variable_weights_group,
    #     with_variable_weights.probs_and_group,
    # )

    # sigmoid_of = computational_node.sigmoid(full_mul)
    # sigmoid_of_eval = torch.sigmoid(full_mul_eval)
    # yield sigmoid_of, sigmoid_of_eval, x_probs_and_group, x.probs_and_group

    # sigmoid_of_variable = computational_node.sigmoid(variable_weights_mul)
    # sigmoid_of_variable_eval = torch.sigmoid(variable_weights_mul_eval)
    # yield (
    #     sigmoid_of_variable,
    #     sigmoid_of_variable_eval,
    #     with_variable_weights_group,
    #     with_variable_weights.probs_and_group,
    # )

    # # now for cumulants, oh baby

    # def empirical_mean_and_re_dim(x, group):
    #     return torch.broadcast_to(torch.einsum("i ..., i -> ...", x, group)[None], x.shape)

    # e_x = Cumulant((x,), name="e[x]")
    # e_x_val = empirical_mean_and_re_dim(x_value, x_probs_and_group)
    # yield e_x, e_x_val, x_probs_and_group, x.probs_and_group

    # e_variable_weights = Cumulant((with_variable_weights,), name="e[with_variable_weights]")
    # e_variable_weights_val = empirical_mean_and_re_dim(with_variable_weights_value, with_variable_weights_group)
    # yield e_variable_weights, e_variable_weights_val, with_variable_weights_group, with_variable_weights.probs_and_group

    # x_m_e_x = Add.from_unweighted_list([x, Einsum.scalar_mul(e_x, -1.0)])
    # x_m_e_x_val: torch.Tensor = x_value - e_x_val
    # yield x_m_e_x, x_m_e_x_val, x_probs_and_group, x.probs_and_group

    # # be wary, this code starts getting a bit cursed and insane right around here

    # variable_weights_m_e_variable_weights = Add.from_unweighted_list(
    #     [
    #         with_variable_weights,
    #         Einsum.scalar_mul(e_variable_weights, -1.0),
    #     ]
    # )
    # variable_weights_m_e_variable_weights_val: torch.Tensor = with_variable_weights_value - e_variable_weights_val
    # yield (
    #     variable_weights_m_e_variable_weights,
    #     variable_weights_m_e_variable_weights_val,
    #     with_variable_weights_group,
    #     with_variable_weights.probs_and_group,
    # )

    # centered_sqr_x = Einsum.power(x_m_e_x, 2)
    # centered_sqr_x_eval = x_m_e_x_val ** 2
    # yield centered_sqr_x, centered_sqr_x_eval, x_probs_and_group, x.probs_and_group

    # centered_sqr_variable_weights = Einsum.power(variable_weights_m_e_variable_weights, 2)
    # centered_sqr_variable_weights_eval = variable_weights_m_e_variable_weights_val ** 2
    # yield (
    #     centered_sqr_variable_weights,
    #     centered_sqr_variable_weights_eval,
    #     with_variable_weights_group,
    #     with_variable_weights.probs_and_group,
    # )

    # var_x = Cumulant((x, x), name="var[x]")
    # var_x_eval = empirical_mean_and_re_dim(
    #     torch.einsum("e a b, e c d -> e a b c d", x_m_e_x_val, x_m_e_x_val), x_probs_and_group
    # )
    # yield var_x, var_x_eval, x_probs_and_group, x.probs_and_group

    # var_variable_weights = Cumulant((with_variable_weights, with_variable_weights), name="var[with_variable_weights]")
    # var_variable_weights_eval = empirical_mean_and_re_dim(
    #     torch.einsum(
    #         "e a b, e c d -> e a b c d",
    #         variable_weights_m_e_variable_weights_val,
    #         variable_weights_m_e_variable_weights_val,
    #     ),
    #     with_variable_weights_group,
    # )
    # yield (
    #     var_variable_weights,
    #     var_variable_weights_eval,
    #     with_variable_weights_group,
    #     with_variable_weights.probs_and_group,
    # )

    # sigmoid_of_reduce = Einsum.from_einsum_str("d a b c -> b d", sigmoid_of)
    # sigmoid_of_reduce_eval = torch.einsum("e d a b c -> e b d", sigmoid_of_eval)
    # yield sigmoid_of_reduce, sigmoid_of_reduce_eval, x_probs_and_group, x.probs_and_group

    # new_dims, new_dims_value, _ = make_discrete_var((4, 5), "new_dims", 17, group=x.probs_and_group)
    # yield new_dims, new_dims_value, x_probs_and_group, x.probs_and_group

    # new_dims_1, new_dims_1_value, _ = make_discrete_var((6, 7), "new_dims_1", 17, group=x.probs_and_group)
    # yield new_dims_1, new_dims_1_value, x_probs_and_group, x.probs_and_group

    # c_new_dims: torch.Tensor = new_dims_value - empirical_mean_and_re_dim(new_dims_value, x_probs_and_group)
    # c_new_dims_1: torch.Tensor = new_dims_1_value - empirical_mean_and_re_dim(new_dims_1_value, x_probs_and_group)

    # third_cumulant_new_dims = Cumulant((x, new_dims, new_dims_1), name="new dims cum")

    # third_cumulant_new_dims_eval = empirical_mean_and_re_dim(
    #     torch.einsum("e a b, e c d, e g h -> e a b c d g h", x_m_e_x_val, c_new_dims, c_new_dims_1),
    #     x_probs_and_group,
    # )
    # yield third_cumulant_new_dims, third_cumulant_new_dims_eval, x_probs_and_group, x.probs_and_group

    # third_cumulant_simple_add = Cumulant((Add.from_unweighted_list([y, y_dup]),), name="cum simple add")

    # third_cumulant_simple_add_eval = empirical_mean_and_re_dim(y_sums_value, x_probs_and_group)
    # yield third_cumulant_simple_add, third_cumulant_simple_add_eval, x_probs_and_group, x.probs_and_group

    # for axis in range(0, 1):
    #     third_cumulant_simple_concat = Cumulant((Concat((y, y_dup), axis=axis),), name="cum simple cat")

    #     third_cumulant_simple_concat_eval = empirical_mean_and_re_dim(
    #         torch.cat([y_value, y_dup_value], dim=axis + 1),
    #         x_probs_and_group,
    #     )
    #     yield third_cumulant_simple_concat, third_cumulant_simple_concat_eval, x_probs_and_group, x.probs_and_group

    # for reverse in [False, True]:
    #     for is_concat, concat_axis in [(False, 0), (True, 0), (True, 1)]:
    #         a_items = (x,) * 2
    #         b_items = (new_dims,) * 3
    #         c_items = (new_dims_1,)
    #         if is_concat:
    #             a: Any = Concat(a_items, axis=concat_axis)
    #             b: Any = Concat(b_items, axis=concat_axis)
    #             c: Any = Concat(c_items, axis=concat_axis)
    #         else:
    #             a = Add.from_unweighted_list(a_items)
    #             b = Add.from_unweighted_list(b_items)
    #             c = Add.from_unweighted_list(c_items)
    #         third_cumulant_new_dims_op = Cumulant((c, b, a) if reverse else (a, b, c), name="new dims cum op")

    #         to_val_new_dims_op: Dict[Circuit, torch.Tensor] = {
    #             a: torch.cat([x_m_e_x_val] * 2, dim=concat_axis + 1) if is_concat else 2 * x_m_e_x_val,
    #             b: torch.cat([c_new_dims] * 3, dim=concat_axis + 1) if is_concat else 3 * c_new_dims,
    #             c: c_new_dims_1,
    #         }

    #         third_cumulant_new_dims_op_eval = empirical_mean_and_re_dim(
    #             torch.einsum(
    #                 "e a b, e c d, e g h -> e a b c d g h",
    #                 *(to_val_new_dims_op[c] for c in third_cumulant_new_dims_op.circuits),
    #             ),
    #             x_probs_and_group,
    #         )
    #         yield third_cumulant_new_dims_op, third_cumulant_new_dims_op_eval, x_probs_and_group, x.probs_and_group

    # a = Index(x, I[:, 1])
    # b = Index(new_dims, (3,))
    # c = Index(new_dims_1, I[:, -1])
    # third_cumulant_new_dims_index = Cumulant((a, b, c), name="new dims cum index")
    # third_cumulant_new_dims_index_eval = empirical_mean_and_re_dim(
    #     torch.einsum("e a, e c, e g -> e a c g", x_m_e_x_val[:, :, 1], c_new_dims[:, 3], c_new_dims_1[:, :, -1]),
    #     x_probs_and_group,
    # )
    # yield third_cumulant_new_dims_index, third_cumulant_new_dims_index_eval, x_probs_and_group, x.probs_and_group

    # cum_full_mul = Cumulant((full_mul,), "cum full mul")
    # yield (
    #     cum_full_mul,
    #     empirical_mean_and_re_dim(full_mul_eval, x_probs_and_group),
    #     x_probs_and_group,
    #     x.probs_and_group,
    # )

    # third_cumulant_x = Cumulant(
    #     (x, sigmoid_of_reduce, sigmoid_of_reduce),
    #     name="sigmoid cum",
    # )
    # c_sigmoid_x = sigmoid_of_reduce_eval - empirical_mean_and_re_dim(sigmoid_of_reduce_eval, x_probs_and_group)
    # third_cumulant_x_eval = empirical_mean_and_re_dim(
    #     torch.einsum("e a b, e c d, e g h -> e a b c d g h", x_m_e_x_val, c_sigmoid_x, c_sigmoid_x),
    #     x_probs_and_group,
    # )
    # yield third_cumulant_x, third_cumulant_x_eval, x_probs_and_group, x.probs_and_group

    # just_sig_3_cum_x = computational_node.sigmoid(third_cumulant_x)
    # just_sig_3_cum_x_eval = torch.sigmoid(third_cumulant_x_eval)
    # yield just_sig_3_cum_x, just_sig_3_cum_x_eval, x_probs_and_group, x.probs_and_group

    # sigmoid_third_cumulant_x = Einsum.from_einsum_str(
    #     f"a d, a b c d g h -> a b c d g h",
    #     x,
    #     just_sig_3_cum_x,
    #     name="sigmoid_3_cum_x",
    # )
    # sigmoid_third_cumulant_x_eval = torch.einsum(
    #     f"e a d, e a b c d g h -> e a b c d g h",
    #     x_value,
    #     just_sig_3_cum_x_eval,
    # )
    # yield sigmoid_third_cumulant_x, sigmoid_third_cumulant_x_eval, x_probs_and_group, x.probs_and_group

    # third_cumulant_variable = Cumulant(
    #     (with_variable_weights, sigmoid_of_variable, sigmoid_of_variable), name="sigmoid cum variable"
    # )
    # c_sigmoid_variable: torch.Tensor = sigmoid_of_variable_eval - empirical_mean_and_re_dim(
    #     sigmoid_of_variable_eval, with_variable_weights_group
    # )
    # third_cumulant_variable_eval = empirical_mean_and_re_dim(
    #     torch.einsum(
    #         "e a b, e c d, e g h -> e a b c d g h",
    #         variable_weights_m_e_variable_weights_val,
    #         c_sigmoid_variable,
    #         c_sigmoid_variable,
    #     ),
    #     with_variable_weights_group,
    # )
    # yield (
    #     third_cumulant_variable,
    #     third_cumulant_variable_eval,
    #     with_variable_weights_group,
    #     with_variable_weights.probs_and_group,
    # )

    # just_sig_3_cum_variable = computational_node.sigmoid(third_cumulant_variable)
    # just_sig_3_cum_variable_eval = torch.sigmoid(third_cumulant_variable_eval)
    # yield (
    #     just_sig_3_cum_variable,
    #     just_sig_3_cum_variable_eval,
    #     with_variable_weights_group,
    #     with_variable_weights.probs_and_group,
    # )

    # sigmoid_third_cumulant_variable = Einsum.from_einsum_str(
    #     f"a c, a b c d g h -> a b c d g h",
    #     with_variable_weights,
    #     just_sig_3_cum_variable,
    #     name="sigmoid_3_cum_variable",
    # )
    # sigmoid_third_cumulant_variable_eval = torch.einsum(
    #     f"e a c, e a b c d g h -> e a b c d g h",
    #     with_variable_weights_value,
    #     just_sig_3_cum_variable_eval,
    # )
    # yield (
    #     sigmoid_third_cumulant_variable,
    #     sigmoid_third_cumulant_variable_eval,
    #     with_variable_weights_group,
    #     with_variable_weights.probs_and_group,
    # )


def test_hash_seeder_randomness():
    assert default_hash_seeder()(Scalar(3.0)) != default_hash_seeder()(Scalar(3.0))
    assert default_hash_seeder(7)(Scalar(3.0)) == default_hash_seeder(7)(Scalar(3.0))
    assert default_hash_seeder(9)(Scalar(3.0)) == default_hash_seeder(9)(Scalar(3.0))

    torch.manual_seed(17)
    fst = default_hash_seeder()(Scalar(3.0))
    torch.manual_seed(17)
    snd = default_hash_seeder()(Scalar(3.0))
    assert fst == snd


def test_sample_randomness():
    x, x_value, x_probs_and_group = make_discrete_var((3, 2), 17, "x")

    def get_sampler(base_seed: Optional[int] = None):
        return Sampler(RandomSampleSpec((100,), seeder=default_hash_seeder(base_seed)))

    assert get_sampler().sample(x) != get_sampler().sample(x)
    assert torch.abs(get_sampler().sample(x).evaluate() - get_sampler().sample(x).evaluate()).mean() > 0.1

    assert get_sampler(2378).sample(x) == get_sampler(2378).sample(x)
    torch.testing.assert_allclose(get_sampler(2378).sample(x).evaluate(), get_sampler(2378).sample(x).evaluate())

    assert get_sampler().sample(x) != get_sampler().sample(x)
    assert torch.abs(get_sampler().sample(x).evaluate() - get_sampler().sample(x).evaluate()).mean() > 0.1

    def get_manual_sampler(seed: int = 1782, shape: tuple[int, ...] = (100,)):
        torch.manual_seed(seed)
        return Sampler(RandomSampleSpec(shape))

    assert get_manual_sampler().sample(x) == get_manual_sampler().sample(x)
    torch.testing.assert_allclose(get_manual_sampler().sample(x).evaluate(), get_manual_sampler().sample(x).evaluate())


# TODO: maybe this should be in a test_nb_rewrites.py or such?
def test_simplify_sampled():
    spec = RandomSampleSpec((100,), seeder=lambda _: 11)
    x = rc.DiscreteVar(rc.Array(torch.zeros((12, 37))))
    assert (
        (Sampler(spec).sample(x)).repr()
        == """0 [100,37] Index [tdf3c928b0b [100],:]
  1 [12,37] Array 4e751d26e9a4b2feca958f1a"""
    )
    spec.simplify = False
    assert (
        rc.strip_names_and_tags(Sampler(spec).sample(x)).repr()
        == """0 GeneralFunction gen_index_at_-2_batch_x_no_batch_index_c
  1 [12,37] Array 4e751d26e9a4b2feca958f1a
  2 GeneralFunction multinomial_[100]
    3 [12] Scalar 0.08333333333333333
    4 Tag 00000000-0000-0000-0000-000000000000
      'seed_11' [] Array 6287c955e43db41def49aa5f"""
    )


@pytest.mark.parametrize("replacement", [False, True])
def test_multinomial(replacement: bool):
    probs = torch.rand((5 * 6 * 7, 17 if replacement else 238))
    num_samples = 3 * 3 * 4
    seed = 17

    samps = torch.multinomial(
        probs, num_samples=num_samples, replacement=replacement, generator=torch.Generator().manual_seed(seed)
    )
    samp_circ = multinomial(Array(probs), Array(torch.tensor(seed)), shape=(num_samples,), replacement=replacement)
    samp_circ.print()
    assert Parser()(samp_circ.repr()) == samp_circ

    assert (samp_circ.evaluate() == samps).all()

    samp_circ = multinomial(
        Array(probs.reshape(5, 6 * 7, probs.shape[-1])),
        Array(torch.tensor(seed)),
        shape=(num_samples,),
        replacement=replacement,
    )
    samp_circ.print()
    assert Parser()(samp_circ.repr()) == samp_circ
    assert samp_circ.shape == (5, 6 * 7, num_samples)
    assert samp_circ.evaluate().shape == samp_circ.shape
    assert (samp_circ.evaluate().reshape(-1, num_samples) == samps).all()

    samp_circ = multinomial(
        Array(probs.reshape(5, 6 * 7, probs.shape[-1])),
        Array(torch.tensor(seed)),
        shape=(3, 4, 3),
        replacement=replacement,
    )
    samp_circ.print()
    assert Parser()(samp_circ.repr()) == samp_circ
    assert samp_circ.shape == (5, 6 * 7, 3, 4, 3)
    assert samp_circ.evaluate().shape == samp_circ.shape
    assert (samp_circ.evaluate().reshape(-1, num_samples) == samps).all()

    samp_circ = multinomial(
        Array(probs.reshape(5, 6, 7, probs.shape[-1])),
        Array(torch.tensor(seed)),
        shape=(3, 4, 3),
        replacement=replacement,
    )
    samp_circ.print()
    assert Parser()(samp_circ.repr()) == samp_circ
    assert samp_circ.shape == (5, 6, 7, 3, 4, 3)
    assert samp_circ.evaluate().shape == samp_circ.shape
    assert (samp_circ.evaluate().reshape(-1, num_samples) == samps).all()

    samp_circ = multinomial(
        Array(probs.reshape(5 * 6 * 7, probs.shape[-1])),
        Array(torch.tensor(seed)),
        shape=(3, 4, 3),
        replacement=replacement,
    )
    samp_circ.print()
    assert Parser()(samp_circ.repr()) == samp_circ
    assert samp_circ.shape == (5 * 6 * 7, 3, 4, 3)
    assert samp_circ.evaluate().shape == samp_circ.shape
    assert (samp_circ.evaluate().reshape(-1, num_samples) == samps).all()


@pytest.mark.parametrize("sample_shape", [(3,), (3, 4)])
def test_seed_sample(sample_shape):
    for v, _, *_ in make_discrete_var_test_circuits():
        # Check no error
        orig_shape = v.shape
        torch.manual_seed(1234)
        a1 = Sampler(RandomSampleSpec(sample_shape)).estimate_and_sample(v).evaluate()
        assert a1.shape == sample_shape + orig_shape
        torch.manual_seed(1234)
        a2 = Sampler(RandomSampleSpec(sample_shape)).estimate_and_sample(v).evaluate()
        torch.manual_seed(1234)
        a3 = Sampler(RandomSampleSpec(sample_shape, seeder=None)).estimate_and_sample(v).evaluate()
        torch.testing.assert_close(a1, a2)
        torch.testing.assert_close(a1, a3)
        torch.testing.assert_close(a2, a3)

        a1 = Sampler(RandomSampleSpec(sample_shape, seeder=default_hash_seeder(87))).estimate_and_sample(v).evaluate()
        assert a1.shape == sample_shape + orig_shape
        a2 = Sampler(RandomSampleSpec(sample_shape, seeder=default_hash_seeder(87))).estimate_and_sample(v).evaluate()
        torch.testing.assert_close(a1, a2)


@pytest.mark.parametrize("sample_shape", [(3,), (3, 4)])
def test_seed_per_circuit_sample(sample_shape):
    u, _, _ = make_discrete_var((3, 42), 17, "u")
    v, _, _ = make_discrete_var((3, 42), 17, "v")

    def seeder(c):
        if c == u.probs_and_group:
            return 0
        if c == v.probs_and_group:
            return 1
        return hash(c) ^ 2833

    c1 = Add(v, u)
    a1 = Sampler(RandomSampleSpec(sample_shape, seeder=seeder)).estimate_and_sample(c1).evaluate()
    c2 = Add(u, v)
    a2 = Sampler(RandomSampleSpec(sample_shape, seeder=seeder)).estimate_and_sample(c2).evaluate()
    torch.testing.assert_close(a1, a2)


def test_sample_all():
    for c, val, _, group_circuit in make_discrete_var_test_circuits():
        c.print()
        assert c.shape == val.shape[1:]
        sampled_circ = Sampler(RunDiscreteVarAllSpec([group_circuit])).estimate_and_sample(c)
        transform = sampled_circ.evaluate()
        torch.testing.assert_close(transform, val, atol=1e-5, rtol=1e-5)


def test_probs_dtype():
    int_vals = Array(torch.randint(0, 800, (100,)), name="int_vals")
    s = """
    'int_var' DiscreteVar
      'int_vals'
      'group' Tag 2ef9f1ac-28bb-4bed-ab7b-09b9050483ac
        'uniform_probs' [100] Scalar 0.01
    """

    c = Parser(reference_circuits_by_name=[int_vals]).parse_circuit(s)
    assert c.torch_dtype == torch.int64
    out = Sampler(RandomSampleSpec((8,))).estimate_and_sample(c)
    assert out.torch_dtype == torch.int64
    assert out.evaluate().dtype == torch.int64
    assert optimize_and_evaluate(out).dtype == torch.int64
