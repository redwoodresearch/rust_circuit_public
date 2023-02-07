import pytest

import rust_circuit as rc


# TODO: more tests
def test_symbolic_expand_simple():
    P = rc.Parser()

    scl, v, new = P.parse_circuits(
        """
    1 [7] Scalar 1.3
    0 Einsum i,i->i
      1
      2 [7] Scalar 1.5
    'new' [0s] Symbol
    """
    )

    assert rc.expand_node(v, [scl, new]).shape == (7,)

    scl0, scl1, v, new0, new1 = P.parse_circuits(
        """
    1 [7] Scalar 1.3
    3 [7,3] Scalar 1.5
    0 Einsum i,i,ij,j->i
      1
      2 [7] Scalar 1.5
      3
      4 [3] Scalar 1.7
    'new0' [0s] Symbol
    'new1' [0s] Symbol
    """
    )

    with pytest.raises(rc.SymbolicSizeSetFailedToSatisfyContraintsError) as exc:
        rc.expand_node(v, [scl0, new0, scl1, new1])
    print(exc.exconly())


def test_symbolic_rearrange_expand():
    P = rc.Parser()
    scl, sym, rearrange = P.parse_circuits(
        """
    0 [6] Scalar 1.3
    'sym' [0s] Symbol rand
    1 Rearrange (a:2 b:3) -> b a
      0
    """
    )

    # should use set symbolic
    expanded = rc.Expander((scl, lambda _: sym))(rearrange)
    assert expanded.shape == (3, 2)

    scl0, scl1, rearrange = P.parse_circuits(
        """
    0 [6] Scalar 1.3
    1 [24] Scalar 1.3
    2 Add
      3 Rearrange (a:2 b:3) -> b a
        0
      4 Rearrange (a:8 b:3) -> a b 1
        1
    """
    )

    with pytest.raises(rc.SymbolicSizeSetFailedToSatisfyContraintsError) as exc:
        rc.Expander(({scl0, scl1}, lambda _: sym))(rearrange)
    print(exc.exconly())

    scl0, scl1, scl2, sym, rearrange = P.parse_circuits(
        """
    'scl0' [3*1s] Scalar 1.3
    'scl1' [24] Scalar 1.3
    'scl2' [17] Scalar 1.3
    'sym' [3*0s] Symbol rand
    2 Add
      3 Rearrange (a:1s b:3) -> b a
        'scl0'
      4 Rearrange (b:3 a:8) -> a b 1
        'scl1'
      8 Einsum i->
        7 Add
          5 Rearrange a:17 -> a
            'scl2'
          6 Rearrange a -> a
            'scl2'
    """
    )

    expanded = rc.Expander(({scl0, scl1, scl2}, lambda _: sym))(rearrange)
    s0, *_ = rc.symbolic_sizes()
    assert expanded.shape == (s0, 3, s0)


def test_multiple_symbolic_product():
    s = """
    'a' [0s*0s*1s*3s] Symbol 07a8025a-d8ba-4c57-946d-cd7cc7e78a66
    'b' Rearrange a b c -> (a b c)
      'c' [1s, 3s, 1s] Symbol 0b108f80-b37f-4e2c-821f-45cbedcbde75
    'd' Rearrange a c -> (a c)
      'e' [1s, 1s] Symbol 3f8aaf28-58f8-44fc-8dbc-497844a5a837
    """
    a, b, d = rc.Parser().parse_circuits(s)
    assert a.repr() == "'a' [0s*0s*1s*3s] Symbol 07a8025a-d8ba-4c57-946d-cd7cc7e78a66"
    assert (
        rc.PrintOptions(shape_only_when_necessary=False).repr(b)
        == "'b' [1s*1s*3s] Rearrange a b c -> (a b c)\n  'c' [1s,3s,1s] Symbol 0b108f80-b37f-4e2c-821f-45cbedcbde75"
    )
    assert (
        rc.PrintOptions(shape_only_when_necessary=False).repr(d)
        == "'d' [1s*1s] Rearrange a b -> (a b)\n  'e' [1s,1s] Symbol 3f8aaf28-58f8-44fc-8dbc-497844a5a837"
    )
