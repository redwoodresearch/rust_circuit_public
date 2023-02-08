from typing import List, Optional

import rust_circuit as rc
import torch
from interp.circuit.causal_scrubbing.dataset import Dataset
from interp.circuit.causal_scrubbing.hypothesis import (
    CondSampler,
    Correspondence,
    FuncSampler,
)

from remix_d4_part2_setup import ParenDataset


def check_correspondence_equality(corr_a: Correspondence, corr_b: Correspondence, ds: Dataset, circ: rc.Circuit):
    """
    corr_a is user provided, the one to be checked
    corr_b is the one we are checking equality to

    Details:
        - Assumes matchers were designed for the same circuit (with same names, etc)
        - Checks type matches for all condsamplers. Also checks that FuncSamplers define same equiv classes.
        Wouldn't properly check e.g. FixedUncondSamplers
    """

    # TODO: be less strict about name equality
    check_name_equality(corr_a, corr_b)
    for node_b, matcher_b in corr_b.in_dfs_order():
        node_a = corr_a.get_by_name(node_b.name)
        matcher_a = corr_a.corr[node_a]

        check_cond_sampler_equality(node_a.cond_sampler, node_b.cond_sampler, ds, f"{node_b.name} cond sampler")
        check_cond_sampler_equality(
            node_a.other_inputs_sampler,
            node_b.other_inputs_sampler,
            ds,
            f"{node_b.name} other inputs sampler",
        )
        check_matcher_equality(matcher_a, matcher_b, circ, f"matcher for {node_b.name}")


def check_name_equality(corr_a: Correspondence, corr_b: Correspondence):
    a_names = set(corr_a.i_names.keys())
    b_names = set(corr_b.i_names.keys())

    if a_names != b_names:
        print("Interp node names don't match!")
        print("Missing names:", b_names - a_names)
        print("Extra names:", a_names - b_names)
        assert False


def equiv_classes(arr) -> set[tuple]:
    arr = arr[:, None] if arr.ndim == 1 else arr
    eq_arrs: List[torch.Tensor] = [torch.all(arr == val, dim=1) for val in torch.unique(arr, dim=0)]
    return set((tuple(arr.tolist()) for arr in eq_arrs))


def check_cond_sampler_equality(cs_a: CondSampler, cs_b: CondSampler, ds: Dataset, name: str):
    if cs_a is cs_b:
        return
    assert type(cs_a) == type(cs_b), f"{name} is the wrong type"
    if isinstance(cs_b, FuncSampler):
        assert isinstance(cs_a, FuncSampler)
        assert equiv_classes(cs_a.func(ds)) == equiv_classes(cs_b.func(ds)), (
            f"{name} doesn't match equivalence classes",
            cs_a.func(ds),
            cs_b.func(ds),
        )


def check_matcher_equality(
    matcher_a: rc.IterativeMatcher,
    matcher_b: rc.IterativeMatcher,
    circ: rc.Circuit,
    name: Optional[str] = None,
):
    if matcher_a == matcher_b:
        return
    name = "matcher" if name is None else name
    assert circ.get(matcher_a) == circ.get(matcher_b), (
        f"Circuit gotten by {name} matches incorrect node(s)",
        matcher_a,
        matcher_b,
        circ.get(matcher_a),
        circ.get(matcher_b),
    )

    circ_a = circ.update(matcher_a, lambda m: rc.Symbol.new_with_none_uuid(shape=m.shape))
    circ_b = circ.update(matcher_b, lambda m: rc.Symbol.new_with_none_uuid(shape=m.shape))

    assert circ_a == circ_b, (
        f"After replacements for {name}, the circuits don't match. Probably the paths by which they are allowed to match differ.",
        matcher_a,
        matcher_b,
        circ.get(circ_a),
        circ.get(circ_b),
    )


def evaluate_on_dataset(c: rc.Circuit, ds: ParenDataset):
    group = rc.DiscreteVar.uniform_probs_and_group(len(ds))
    transform = rc.Sampler(rc.RunDiscreteVarAllSpec([group]))
    c = c.update("tokens", lambda _: rc.DiscreteVar(ds.tokens, probs_and_group=group))
    c = rc.substitute_all_modules(c)
    return transform.sample(c).evaluate(rc.TorchDeviceDtypeOp(dtype="float32", device="cpu"))


def check_circuit_equality(circuit_a: rc.Circuit, circuit_b: rc.Circuit, strict=True):
    import remix_d4_part2_solution as sol

    test_ds = sol.ds[:1000]
    eval_a = evaluate_on_dataset(circuit_a, test_ds)
    eval_b = evaluate_on_dataset(circuit_b, test_ds)
    if not torch.allclose(eval_a, eval_b):
        err = (eval_a - eval_b).abs().mean().item()
        assert False, (
            f"circuits do not evaluate to the same thing. Mean abs error {err}",
            circuit_a,
            circuit_b,
            eval_a,
            eval_b,
        )

    if strict:
        assert circuit_a == circuit_b, (
            "Circuits are not strictly equal. This includes names.",
            circuit_a,
            circuit_b,
        )


# prefixed "t_" instead of "test_" since we don't want pytest to collect them.


def t_ex0a_corr(corr):
    import remix_d4_part2_solution as sol

    check_correspondence_equality(corr, sol.corr0a, sol.ds, sol.circuit)


def t_ex0b_corr(corr):
    import remix_d4_part2_solution as sol

    check_correspondence_equality(corr, sol.corr0b, sol.ds, sol.circuit)


def t_m_10(m_10):
    import remix_d4_part2_solution as sol

    check_matcher_equality(m_10, sol.m_10, sol.circuit, "m_10")


def t_m_20(m_20):
    import remix_d4_part2_solution as sol

    check_matcher_equality(m_20, sol.m_20, sol.circuit, "m_20")


def t_m_21(m_21):
    import remix_d4_part2_solution as sol

    check_matcher_equality(m_21, sol.m_21, sol.circuit, "m_21")


def t_count_cond(count_cond):
    import remix_d4_part2_solution as sol

    check_cond_sampler_equality(count_cond, sol.count_cond, sol.ds, "count_cond")


def t_horizon_cond(horizon_cond):
    import remix_d4_part2_solution as sol

    check_cond_sampler_equality(horizon_cond, sol.horizon_cond, sol.ds, "horizon_cond")


def t_ex1_corr(make_ex1_corr, cond: CondSampler):
    import remix_d4_part2_solution as sol

    check_correspondence_equality(make_ex1_corr(cond), sol.make_ex1_corr(cond), sol.ds, sol.circuit)


def t_start_open_cond(start_open_cond):
    import remix_d4_part2_solution as sol

    check_cond_sampler_equality(start_open_cond, sol.start_open_cond, sol.ds, "start_open_cond")


def t_count_open_cond(count_open_cond):
    import remix_d4_part2_solution as sol

    check_cond_sampler_equality(count_open_cond, sol.count_open_cond, sol.ds, "count_open_cond")


def t_ex2_part1_circuit(circuit):
    import remix_d4_part2_solution as sol

    check_circuit_equality(circuit, sol.ex2_part1_circuit)


def t_m_10_p1(m_10_p1):
    import remix_d4_part2_solution as sol

    check_matcher_equality(m_10_p1, sol.m_10_p1, sol.ex2_part1_circuit, "m_10_p1")


def t_m_20_p1(m_20_p1):
    import remix_d4_part2_solution as sol

    check_matcher_equality(m_20_p1, sol.m_20_p1, sol.ex2_part1_circuit, "m_20_p1")


def t_make_ex2_part1_corr(corr):
    import remix_d4_part2_solution as sol

    check_correspondence_equality(corr, sol.make_ex2_part1_corr(), sol.ds, sol.ex2_part1_circuit)


def t_ex2_part2_circuit(circuit):
    import remix_d4_part2_solution as sol

    check_circuit_equality(circuit, sol.ex2_part2_circuit)


def t_ex2_part2_corr(corr):
    import remix_d4_part2_solution as sol

    check_correspondence_equality(corr, sol.make_ex2_part2_corr(), sol.ds, sol.ex2_part2_circuit)


def t_project_into_direction(fn):
    import remix_d4_part2_solution as sol

    h00 = sol.circuit.get_unique("a0.h0")
    assert fn(h00).name == "a0.h0_projected"
    check_circuit_equality(fn(h00), sol.project_into_direction(h00), strict=False)


def t_ex2_part3_circuit(get_circuit):
    import remix_d4_part2_solution as sol

    check_circuit_equality(
        get_circuit(sol.ex2_part2_circuit, sol.project_into_direction),
        sol.ex2_part3_circuit,
    )


def t_ex2_part3_corr(corr):
    import remix_d4_part2_solution as sol

    check_correspondence_equality(corr, sol.make_ex2_part3_corr(), sol.ds, sol.ex2_part3_circuit)


def t_compute_phi_circuit(fn):
    import remix_d4_part2_solution as sol

    tokens = sol.circuit.get_unique("tokens")
    assert fn(tokens).name == "a0.h0_phi"
    check_circuit_equality(fn(tokens), sol.compute_phi_circuit(tokens), strict=False)


def t_ex2_part4_circuit(get_circuit):
    import remix_d4_part2_solution as sol

    check_circuit_equality(
        get_circuit(sol.ex2_part2_circuit, sol.compute_phi_circuit),
        sol.ex2_part4_circuit,
    )


def t_ex2_part4_corr(corr):
    import remix_d4_part2_solution as sol

    check_correspondence_equality(corr, sol.make_ex2_part4_corr(), sol.ds, sol.ex2_part4_circuit)


def t_separate_all_seqpos(fn):
    import remix_d4_part2_solution as sol

    start_circ = rc.Array(torch.randn((42, 56)), name="start_arr")
    check_circuit_equality(fn(start_circ), sol.separate_all_seqpos(start_circ))


def t_ex3_circuit(circ):
    import remix_d4_part2_solution as sol

    check_circuit_equality(circ, sol.ex3_circuit)


def t_to_horizon_vals(fn):
    import remix_d4_part2_solution as sol

    for i in (0, 1, 15, 41):  # random seqpos
        for adj in (True, False):
            assert equiv_classes(fn(sol.ds, i, adj)) == equiv_classes(sol.to_horizon_vals(sol.ds, i, adj)), (
                f"Equiv classes don't match for i={i}, adj={adj}",
                fn(sol.ds, i, adj),
                sol.to_horizon_vals(sol.ds, i, adj),
            )


def t_get_horizon_all_cond(fn):
    import remix_d4_part2_solution as sol

    for adj in (True, False):
        check_cond_sampler_equality(fn(adj), sol.get_horizon_all_cond(adj), sol.ds, "horizon_all cond")


def t_make_ex3_corr(fn):
    import remix_d4_part2_solution as sol

    for adj in (True, False):
        check_correspondence_equality(fn(adj), sol.make_ex3_corr(adj), sol.ds, sol.ex3_circuit)


def t_ex4_corr(corr):
    import remix_d4_part2_solution as sol

    check_correspondence_equality(corr, sol.ex4_corr, sol.ds, sol.ex4_circuit)
