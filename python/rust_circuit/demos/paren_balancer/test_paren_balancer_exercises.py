from typing import List, Optional

import torch

import rust_circuit as rc
from rust_circuit.causal_scrubbing.dataset import Dataset
from rust_circuit.causal_scrubbing.hypothesis import CondSampler, Correspondence, FuncSampler

from .setup import ParenDataset


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
            node_a.other_inputs_sampler, node_b.other_inputs_sampler, ds, f"{node_b.name} other inputs sampler"
        )
        check_matcher_equality(matcher_a, matcher_b, circ, f"matcher for {node_b.name}")


def check_name_equality(corr_a: Correspondence, corr_b: Correspondence):
    a_names = set(corr_a.i_names.keys())
    b_names = set(corr_b.i_names.keys())

    if a_names != b_names:
        print("Interp node names don't match!")
        print("Missing names:", b_names - a_names)
        print("Extra names:", b_names - a_names)
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
            f"{name} doesn't match equivilance classes",
            cs_a.func(ds),
            cs_b.func(ds),
        )


def check_matcher_equality(
    matcher_a: rc.IterativeMatcher, matcher_b: rc.IterativeMatcher, circ: rc.Circuit, name: Optional[str] = None
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
    circ_a = circ
    circ_b = circ
    for match in circ.get(matcher_b):
        symbol = rc.Symbol.new_with_random_uuid(shape=match.shape)
        circ_a = circ_a.update(match, lambda _: symbol)
        circ_b = circ_b.update(match, lambda _: symbol)
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
    return transform.sample(c).evaluate().to(device="cpu", dtype=torch.float32)


def check_circuit_equality(circuit_a: rc.Circuit, circuit_b: rc.Circuit, strict=True):
    from . import causal_scrubbing_experiments as cse

    test_ds = cse.ds[:1000]
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
        assert circuit_a == circuit_b, ("Circuits are not strictly equal. This includes names.", circuit_a, circuit_b)


# prefixed "t_" instead of "test_" since we don't want pytest to collect them.


def t_ex0a_corr(corr):
    from . import causal_scrubbing_experiments as cse

    check_correspondence_equality(corr, cse.corr0a, cse.ds, cse.circuit)


def t_ex0b_corr(corr):
    from . import causal_scrubbing_experiments as cse

    check_correspondence_equality(corr, cse.corr0b, cse.ds, cse.circuit)


def t_m_10(m_10):
    from . import causal_scrubbing_experiments as cse

    check_matcher_equality(m_10, cse.m_10, cse.circuit, "m_10")


def t_m_20(m_20):
    from . import causal_scrubbing_experiments as cse

    check_matcher_equality(m_20, cse.m_20, cse.circuit, "m_20")


def t_m_21(m_21):
    from . import causal_scrubbing_experiments as cse

    check_matcher_equality(m_21, cse.m_21, cse.circuit, "m_21")


def t_count_cond(count_cond):
    from . import causal_scrubbing_experiments as cse

    check_cond_sampler_equality(count_cond, cse.count_cond, cse.ds, "count_cond")


def t_horizon_cond(horizon_cond):
    from . import causal_scrubbing_experiments as cse

    check_cond_sampler_equality(horizon_cond, cse.horizon_cond, cse.ds, "horizon_cond")


def t_ex1_corr(make_ex1_corr, cond: CondSampler):
    from . import causal_scrubbing_experiments as cse

    check_correspondence_equality(make_ex1_corr(cond), cse.make_ex1_corr(cond), cse.ds, cse.circuit)


def t_start_open_cond(start_open_cond):
    from . import causal_scrubbing_experiments as cse

    check_cond_sampler_equality(start_open_cond, cse.start_open_cond, cse.ds, "start_open_cond")


def t_count_open_cond(count_open_cond):
    from . import causal_scrubbing_experiments as cse

    check_cond_sampler_equality(count_open_cond, cse.count_open_cond, cse.ds, "count_open_cond")


def t_ex2_part1_circuit(circuit):
    from . import causal_scrubbing_experiments as cse

    check_circuit_equality(circuit, cse.ex2_part1_circuit)


def t_m_10_p1(m_10_p1):
    from . import causal_scrubbing_experiments as cse

    check_matcher_equality(m_10_p1, cse.m_10_p1, cse.ex2_part1_circuit, "m_10_p1")


def t_m_20_p1(m_20_p1):
    from . import causal_scrubbing_experiments as cse

    check_matcher_equality(m_20_p1, cse.m_20_p1, cse.ex2_part1_circuit, "m_20_p1")


def t_make_ex2_part1_corr(corr):
    from . import causal_scrubbing_experiments as cse

    check_correspondence_equality(corr, cse.make_ex2_part1_corr(), cse.ds, cse.ex2_part1_circuit)


def t_ex2_part2_circuit(circuit):
    from . import causal_scrubbing_experiments as cse

    check_circuit_equality(circuit, cse.ex2_part2_circuit)


def t_ex2_part2_corr(corr):
    from . import causal_scrubbing_experiments as cse

    check_correspondence_equality(corr, cse.make_ex2_part2_corr(), cse.ds, cse.ex2_part2_circuit)


def t_project_into_direction(fn):
    from . import causal_scrubbing_experiments as cse

    h00 = cse.circuit.get_unique("a0.h0")
    assert fn(h00).name == "a0.h0_projected"
    check_circuit_equality(fn(h00), cse.project_into_direction(h00), strict=False)


def t_ex2_part3_circuit(get_circuit):
    from . import causal_scrubbing_experiments as cse

    check_circuit_equality(get_circuit(cse.ex2_part2_circuit, cse.project_into_direction), cse.ex2_part3_circuit)


def t_ex2_part3_corr(corr):
    from . import causal_scrubbing_experiments as cse

    check_correspondence_equality(corr, cse.make_ex2_part3_corr(), cse.ds, cse.ex2_part3_circuit)


def t_compute_phi_circuit(fn):
    from . import causal_scrubbing_experiments as cse

    tokens = cse.circuit.get_unique("tokens")
    assert fn(tokens).name == "a0.h0_phi"
    check_circuit_equality(fn(tokens), cse.compute_phi_circuit(tokens), strict=False)


def t_ex2_part4_circuit(get_circuit):
    from . import causal_scrubbing_experiments as cse

    check_circuit_equality(get_circuit(cse.ex2_part2_circuit, cse.compute_phi_circuit), cse.ex2_part4_circuit)


def t_ex2_part4_corr(corr):
    from . import causal_scrubbing_experiments as cse

    check_correspondence_equality(corr, cse.make_ex2_part4_corr(), cse.ds, cse.ex2_part4_circuit)


def t_separate_all_seqpos(fn):
    from . import causal_scrubbing_experiments as cse

    start_circ = rc.Array(torch.randn((42, 56)), name="start_arr")
    check_circuit_equality(fn(start_circ), cse.separate_all_seqpos(start_circ))


def t_ex3_circuit(circ):
    from . import causal_scrubbing_experiments as cse

    check_circuit_equality(circ, cse.ex3_circuit)


def t_to_horizon_vals(fn):
    from . import causal_scrubbing_experiments as cse

    for i in (0, 1, 15, 41):  # random seqpos
        for adj in (True, False):
            assert equiv_classes(fn(cse.ds, i, adj)) == equiv_classes(cse.to_horizon_vals(cse.ds, i, adj)), (
                f"Equiv classes don't match for i={i}, adj={adj}",
                fn(cse.ds, i, adj),
                cse.to_horizon_vals(cse.ds, i, adj),
            )


def t_get_horizon_all_cond(fn):
    from . import causal_scrubbing_experiments as cse

    for adj in (True, False):
        check_cond_sampler_equality(fn(adj), cse.get_horizon_all_cond(adj), cse.ds, "horizon_all cond")


def t_make_ex3_corr(fn):
    from . import causal_scrubbing_experiments as cse

    for adj in (True, False):
        check_correspondence_equality(fn(adj), cse.make_ex3_corr(adj), cse.ds, cse.ex3_circuit)


def t_ex4_corr(corr):
    from . import causal_scrubbing_experiments as cse

    check_correspondence_equality(corr, cse.ex4_corr, cse.ds, cse.ex4_circuit)
