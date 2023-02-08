import uuid

import rust_circuit as rc


def clean_tags(c):
    return c.update(
        lambda x: x.is_tag(),
        lambda x: rc.Tag(x.node, uuid.UUID("00000000-0000-0000-0000-000000000000")),
    )


def check_matcher_equality(matcher_a: rc.IterativeMatcher, matcher_b: rc.IterativeMatcher, circ: rc.Circuit):
    if matcher_a == matcher_b:
        return
    assert circ.get(matcher_a) == circ.get(matcher_b), (
        f"Matches incorrect node(s)",
        matcher_a,
        matcher_b,
        circ.get(matcher_a),
        circ.get(matcher_b),
    )
    circ_a = circ
    circ_b = circ
    for match in circ.get(matcher_b):
        symbol = rc.Symbol.new_with_random_uuid(shape=match.shape)
        circ_a = circ_a.update(matcher_a, lambda _: symbol)
        circ_b = circ_b.update(matcher_b, lambda _: symbol)
    assert circ_a == circ_b, (
        f"After replacing matched nodes, the circuits don't match. Probably the paths by which they are allowed to match differ.",
        matcher_a,
        matcher_b,
        circ.get(circ_a),
        circ.get(circ_b),
    )


def test_all_inputs_matcher(m_test: rc.IterativeMatcher, c: rc.Circuit):
    import remix_d3_solution

    check_matcher_equality(m_test, remix_d3_solution.unused_baseline_path, c)


def test_by_head(by_head: rc.Circuit):
    import remix_d3_solution

    assert clean_tags(by_head) == clean_tags(remix_d3_solution.by_head)


def test_with_a1_ind_inputs(c: rc.Circuit):
    import remix_d3_solution

    assert clean_tags(c) == clean_tags(remix_d3_solution.with_a1_ind_inputs)


def test_qkv_ind_input_matchers(
    m_test_q: rc.IterativeMatcher, m_test_k: rc.IterativeMatcher, m_test_v: rc.IterativeMatcher, c: rc.Circuit
):
    import remix_d3_solution

    check_matcher_equality(m_test_q, remix_d3_solution.q_ind_input_matcher, c)
    check_matcher_equality(m_test_k, remix_d3_solution.k_ind_input_matcher, c)
    check_matcher_equality(m_test_v, remix_d3_solution.v_ind_input_matcher, c)


def test_v_matcher(m_test: rc.IterativeMatcher, c: rc.Circuit):
    import remix_d3_solution

    check_matcher_equality(m_test, remix_d3_solution.v_matcher, c)


def test_mask_a0(rewritten: rc.Circuit):
    import remix_d3_solution

    assert clean_tags(rewritten) == clean_tags(remix_d3_solution.mask_a0)


def test_add_of_module(rewritten: rc.Circuit):
    import remix_d3_solution

    assert rewritten == remix_d3_solution.add_of_module


def test_fancy_k_matcher(m_test: rc.IterativeMatcher, c: rc.Circuit):
    import remix_d3_solution

    check_matcher_equality(m_test, remix_d3_solution.fancy_k_matcher, c)
