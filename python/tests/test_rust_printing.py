import os
from typing import Dict, List, Tuple

import hypothesis
import hypothesis.strategies as st
import pytest
import torch

import rust_circuit as rc
from interp.circuit.circuit import Circuit as cCircuit
from rust_circuit.interop_rust import py_to_rust
from rust_circuit.py_utils import timed

from .test_rust_rewrite import get_c_st


def print_multiline_escape(s: str):
    uuid = "6c6805a0-3aaf-492c-93a3-00851eacaacf"
    print(s.replace("\n", uuid).encode("unicode_escape").decode("ASCII").replace(uuid, "\n"))


@st.composite
def st_maybe_many_circuits(draw: st.DrawFn, *args, **kwargs) -> Tuple[List[cCircuit], bool]:
    c_st = get_c_st(*args, **kwargs)
    many = draw(st.booleans())
    if many:
        num = draw(st.integers(min_value=0, max_value=4))
        return ([draw(c_st) for _ in range(num)], many)
    else:
        return ([draw(c_st)], many)


@hypothesis.settings(deadline=None)
@hypothesis.given(cs_is_many=st_maybe_many_circuits())
@pytest.mark.parametrize("names", [False, True])
@pytest.mark.parametrize("shapes_only_necessary", [False, True])
@pytest.mark.parametrize("force_use_serial_numbers", [False, True])
@pytest.mark.parametrize("only_child_below", [False, True])
def test_rust_printing_bijection(
    cs_is_many: Tuple[List[cCircuit], bool],
    names: bool,
    shapes_only_necessary: bool,
    force_use_serial_numbers: bool,
    only_child_below: bool,
):
    cs, is_many = cs_is_many
    circs = [py_to_rust(c) for c in cs]
    if not names:
        circs = [rc.strip_names_and_tags(circ) for circ in circs]
    printed = rc.PrintOptions(
        True,
        shape_only_when_necessary=shapes_only_necessary,
        force_use_serial_numbers=force_use_serial_numbers,
        only_child_below=only_child_below,
    ).repr(*circs)
    print(printed)
    # print_multiline_escape(printed)
    if is_many:
        parsed = rc.Parser().parse_circuits(printed)
    else:
        parsed = [rc.Parser().parse_circuit(printed)]
    parsed = [rc.deep_canonicalize(p) for p in parsed]
    circs = [rc.deep_canonicalize(c) for c in circs]
    if parsed is not None:
        if parsed != circs:
            for p, c in zip(parsed, circs):
                c.print()
                p.print()
        assert parsed == circs


def test_empty_einsum_parse():
    assert rc.Parser()("""0 [] Einsum ->""") == rc.Einsum(out_axes=()).rename(None)


def test_autoname_disable_error():
    s = """
    0 [5, 3, 6] Add
    0 AD Add
    """
    with pytest.raises(rc.ParseCircuitOnCircuitRepeatInfoIsNotSameError):
        rc.Parser().parse_circuits(s)


def test_rust_printing_bijection_2():
    strings = [
        """
0 [5, 3, 6] Add
  1 [3, 1] Scatter [0:2,0:1]
    2 [2, 1] Scalar 1.0
  10 [5, 3, 6] Scatter [0:5,0:3,0:3]
    4 [5, 3, 3] Concat 0
      5 '\\\\'\\\\' [2, 3, 3] Scalar 1.2
      6 '\\' hi\\\\' [3, 3, 3] Scalar 1.2""",
        """
3 Tag 1a280c74-fa9f-4dd9-94f4-5cca4a0cf243
  2 [3, 3, 3] Scalar 1.2""",
        """
0 Cumulant
  1 [2, 3, 3] Scalar 1.0
  3 Tag 1a280c74-fa9f-4dd9-94f4-5cca4a0cf243
    2 [3, 3, 3] Scalar 1.2""",
        """
0 DiscreteVar
  2 [10,11] Scalar 2.2
  1 [10] Scalar 0.1
  """,
        """
0 StoredCumulantVar 1,2|1a280c74-fa9f-4dd9-94f4-5cca4a0cf243
  1 [3, 1] Scalar 1.2
  2 [3, 1, 3, 1] Scalar 1.2""",
    ]
    for s in strings:
        c = rc.Parser()(s)
        print(s)
        c.print()


def test_rust_printing_comments():
    strings = [
        """
0 [5, 3, 6] Add
  1 [3, 1] Scatter [0:2,0:1]
    2 [2, 1] Scalar 1.0
  10 [5, 3, 6] Scatter [0:5,0:3,0:3]
    4 '# my name' [5, 3, 3] Concat 0
      5 '#\\\\'\\\\' [2, 3, 3] Scalar 1.2
      6 '\\' hi\\\\' [3, 3, 3] Scalar 1.2""",
        """
        # comment before
0 [5, 3, 6] Add
  1 [3, 1] Scatter [0:2,0:1]
    2 [2, 1] Scalar 1.0
  10 [5, 3, 6] Scatter [0:5,0:3,0:3]
    4 '# my name' [5, 3, 3] Concat 0
      5 '#\\\\'\\\\' [2, 3, 3] Scalar 1.2
      6 '\\' hi\\\\' [3, 3, 3] Scalar 1.2""",
        """
# comment before
0 [5, 3, 6] Add
  1 [3, 1] Scatter [0:2,0:1]
    2 [2, 1] Scalar 1.0
  10 [5, 3, 6] Scatter [0:5,0:3,0:3]
    4 '# my name' [5, 3, 3] Concat 0
      5 '#\\\\'\\\\' [2, 3, 3] Scalar 1.2
      6 '\\' hi\\\\' [3, 3, 3] Scalar 1.2""",
        """
0 [5, 3, 6] Add
  1 [3, 1] Scatter [0:2,0:1]
# comment mid
    2 [2, 1] Scalar 1.0
  10 [5, 3, 6] Scatter [0:5,0:3,0:3]
    4 '# my name' [5, 3, 3] Concat 0
      5 '#\\\\'\\\\' [2, 3, 3] Scalar 1.2
      6 '\\' hi\\\\' [3, 3, 3] Scalar 1.2""",
        """
0 [5, 3, 6] Add
  1 [3, 1] Scatter [0:2,0:1] # comment
    2 [2, 1] Scalar 1.0
  10 [5, 3, 6] Scatter [0:5,0:3,0:3]
    4 '# my name' [5, 3, 3] Concat 0
      5 '#\\\\'\\\\' [2, 3, 3] Scalar 1.2
      6 '\\' hi\\\\' [3, 3, 3] Scalar 1.2""",
        """
0 [5, 3, 6] Add#c
  1 [3, 1] Scatter [0:2,0:1]
    2 [2, 1] Scalar 1.0
  10 [5, 3, 6] Scatter [0:5,0:3,0:3]
    4 '# my name' [5, 3, 3] Concat 0#dfj
      5 '#\\\\'\\\\' [2, 3, 3] Scalar 1.2      #  sldkj#*$&fstuf
      6 '\\' hi\\\\' [3, 3, 3] Scalar 1.2""",
        """
0 [5, 3, 6] Add
  1 [3, 1] Scatter [0:2,0:1]
    2 [2, 1] Scalar 1.0
  10 [5, 3, 6] Scatter [0:5,0:3,0:3]###
    4 '# my name' [5, 3, 3] Concat 0
      5 '#\\\\'\\\\' [2, 3, 3] Scalar 1.2
      6 '\\' hi\\\\' [3, 3, 3] Scalar 1.2""",
        """
0 [5, 3, 6] Add
  1 [3, 1] Scatter [0:2,0:1]
    # comment mid
    2 [2, 1] Scalar 1.0
  10 [5, 3, 6] Scatter [0:5,0:3,0:3]
    4 '# my name' [5, 3, 3] Concat 0
      5 '#\\\\'\\\\' [2, 3, 3] Scalar 1.2
      6 '\\' hi\\\\' [3, 3, 3] Scalar 1.2""",
    ]
    circs = [rc.Parser()(s) for s in strings]
    assert len(set(circs)) == 1, circs


def test_rust_printing_many():
    strings = [
        """
0 [5, 3, 6] Add
  1 [3, 1] Scatter [0:2,0:1]
    2 [2, 1] Scalar 1.0
  10 [5, 3, 6] Scatter [0:5,0:3,0:3]
    4 '# my name' [5, 3, 3] Concat 0
      5 '#\\\\'\\\\' [2, 3, 3] Scalar 1.2
      6 '\\' hi\\\\' [3, 3, 3] Scalar 1.2""",
        """
        # comment before
0 [5, 3, 6] Add
  1 [3, 1] Scatter [0:2,0:1]
    2 [2, 1] Scalar 1.0
  10 [5, 3, 6] Scatter [0:5,0:3,0:3]
    4 '# my name' [5, 3, 3] Concat 0
      5 '#\\\\'\\\\' [2, 3, 3] Scalar 1.2
      6 '\\' hi\\\\' [3, 3, 3] Scalar 1.2""",
        """
# comment before
0 [5, 3, 6] Add
  1 [3, 1] Scatter [0:2,0:1]
    2 [2, 1] Scalar 1.0
  10 [5, 3, 6] Scatter [0:5,0:3,0:3]
    4 '# my name' [5, 3, 3] Concat 0
      5 '#\\\\'\\\\' [2, 3, 3] Scalar 1.2
      6 '\\' hi\\\\' [3, 3, 3] Scalar 1.2""",
        """
0 [5, 3, 6] Add
  1 [3, 1] Scatter [0:2,0:1]
# comment mid
    2 [2, 1] Scalar 1.0
  10 [5, 3, 6] Scatter [0:5,0:3,0:3]
    4 '# my name' [5, 3, 3] Concat 0
      5 '#\\\\'\\\\' [2, 3, 3] Scalar 1.2
      6 '\\' hi\\\\' [3, 3, 3] Scalar 1.2""",
        """
0 [5, 3, 6] Add
  1 [3, 1] Scatter [0:2,0:1]
    # comment mid
    2 [2, 1] Scalar 1.0
  10 [5, 3, 6] Scatter [0:5,0:3,0:3]
    4 '# my name' [5, 3, 3] Concat 0
      5 '#\\\\'\\\\' [2, 3, 3] Scalar 1.2
      6 '\\' hi\\\\' [3, 3, 3] Scalar 1.2""",
    ]
    circs = [rc.Parser()(s) for s in strings]
    assert len(set(circs)) == 1, circs


def test_rust_printing_bijection_many():
    base = """
0 Add
  1 [3, 1] Scatter [0:2,0:1]
    2 [2, 1] Scalar 1
  3 '0' [1, 3, 6] Scalar 7
  4 '1' [5, 3, 6] Scatter [0:5,0:3,0:3]
    5 Concat 0
      6 [2, 3, 3] Scalar 1.2
      7 [3, 3, 3] Scalar 1.2
  8 'ein' Einsum abc->abc
    4
  3"""
    circ = rc.Parser()(base)
    extra_circs = """
# extra
3 '0' [1, 3, 6] Scalar 7
4 '1' [5, 3, 6] Scatter [0:5,0:3,0:3]
  5 Concat 0
    6 [2, 3, 3] Scalar 1.2
    7 [3, 3, 3] Scalar 1.2
# we can include newlines between

0 Add
  1 [3, 1] Scatter [0:2,0:1]
    2 [2, 1] Scalar 1
  # now below is repeated
  3
  4
  8 'ein' Einsum abc->abc
    4
  3"""

    scalar, scatter, same_circ = rc.Parser().parse_circuits(extra_circs)
    assert scalar == rc.Matcher("0").get_unique(circ)
    assert scatter == rc.Matcher("1").get_unique(circ)
    assert same_circ == circ

    expected = """
'0' [1,3,6] Scalar 7
'1' [5,3,6] Scatter [0:5,0:3,0:3]
  0 Concat 0
    1 [2,3,3] Scalar 1.2
    2 [3,3,3] Scalar 1.2
3 Add
  4 [3,1] Scatter [0:2,0:1]
    5 [2,1] Scalar 1
  '0'
  '1'
  'ein' Einsum abc->abc
    '1'
  '0'""".strip(
        "\n"
    )
    # if you update printer, verify print is as expected and then copy in new test strings
    assert expected == rc.PrintOptions().repr(scalar, scatter, same_circ)

    scalar, scatter, same_circ = rc.Parser().parse_circuits(expected)
    assert scalar == rc.Matcher("0").get_unique(circ)
    assert scatter == rc.Matcher("1").get_unique(circ)
    assert same_circ == circ


def test_rust_printing_bijection_indent():
    base = """
0 Add
  1 [3, 1] Scatter [0:2,0:1]
    2 [2, 1] Scalar 1
  3 '0' [1, 3, 6] Scalar 7
  4 '1' [5, 3, 6] Scatter [0:5,0:3,0:3]
    5 Concat 0
      6 [2, 3, 3] Scalar 1.2
      7 [3, 3, 3] Scalar 1.2
  8 'ein' Einsum abc->abc
    4
  3"""
    circ = rc.Parser()(base)
    extra_circs = """
    # extra
    3 '0' [1, 3, 6] Scalar 7
    4 '1' [5, 3, 6] Scatter [0:5,0:3,0:3]
      5 Concat 0
        6 [2, 3, 3] Scalar 1.2
        7 [3, 3, 3] Scalar 1.2
    # we can include newlines between

    0 Add
      1 [3, 1] Scatter [0:2,0:1]
        2 [2, 1] Scalar 1
      # now below is repeated
      3
      4
      8 'ein' Einsum abc->abc
        4
      3
    """

    scalar, scatter, same_circ = rc.Parser().parse_circuits(extra_circs)
    assert scalar == rc.Matcher("0").get_unique(circ)
    assert scatter == rc.Matcher("1").get_unique(circ)
    assert same_circ == circ

    extra_circs = """
        # extra indent for no reason
           # extra
           3 '0' [1, 3, 6] Scalar 7
           4 '1' [5, 3, 6] Scatter [0:5,0:3,0:3]
             5 Concat 0
               6 [2, 3, 3] Scalar 1.2
               7 [3, 3, 3] Scalar 1.2
         # we can include newlines between

           0 Add
             1 [3, 1] Scatter [0:2,0:1]
               2 [2, 1] Scalar 1
             # now below is repeated
             3
             4
             8 'ein' Einsum abc->abc
               4
             3
    """

    scalar, scatter, same_circ = rc.Parser().parse_circuits(extra_circs)
    assert scalar == rc.Matcher("0").get_unique(circ)
    assert scatter == rc.Matcher("1").get_unique(circ)
    assert same_circ == circ

    extra_circs = """
        # extra
           3 '0' [1, 3, 6] Scalar 7
           4 '1' [5, 3, 6] Scatter [0:5,0:3,0:3]
             5 Concat 0
               6 [2, 3, 3] Scalar 1.2
               7 [3, 3, 3] Scalar 1.2
         # we can include newlines between

           0 Add
             1 [3, 1] Scatter [0:2,0:1]
               2 [2, 1] Scalar 1
             # now below is repeated
             3
             4
             8 'ein' Einsum abc->abc
               4
             3"""

    scalar, scatter, same_circ = rc.Parser().parse_circuits(extra_circs)
    assert scalar == rc.Matcher("0").get_unique(circ)
    assert scatter == rc.Matcher("1").get_unique(circ)
    assert same_circ == circ

    extra_circs = """
           3 '0' [1, 3, 6] Scalar 7
           4 '1' [5, 3, 6] Scatter [0:5,0:3,0:3]
             5 Concat 0
               6 [2, 3, 3] Scalar 1.2
               7 [3, 3, 3] Scalar 1.2
           0 Add
             1 [3, 1] Scatter [0:2,0:1]
               2 [2, 1] Scalar 1
             # now below is repeated
             3
             4
             8 'ein' Einsum abc->abc
               4
             3"""

    scalar, scatter, same_circ = rc.Parser().parse_circuits(extra_circs)
    assert scalar == rc.Matcher("0").get_unique(circ)
    assert scatter == rc.Matcher("1").get_unique(circ)
    assert same_circ == circ


def test_parse_less_indent():
    less_indent = """
        8 [2] Scalar 3
    0 Add
      1 [3, 1] Scatter [0:2,0:1]
        2 [2, 1] Scalar 1
      3 '0' [1, 3, 6] Scalar 7
      4 '1' [5, 3, 6] Scatter [0:5,0:3,0:3]
        5 Concat 0
          6 [2, 3, 3] Scalar 1.2
          7 [3, 3, 3] Scalar 1.2
      8 'ein' Einsum abc->abc
        4
      3
    """
    with pytest.raises(rc.ParseCircuitLessIndentationThanFirstItemError):
        rc.Parser()(less_indent)


def test_print_tensor_index():
    array_of_tensor = rc.Array(torch.tensor([1, 2, 3, 3, 2, 1], dtype=torch.int64))
    array_of_tensor.print()
    less_indent = rc.Index(rc.Scalar(1.0, (10, 10)), (array_of_tensor.value, 0))
    assert (
        less_indent.repr()
        == """0 [6] Index [t292d9fdb22 [6],0]
  1 [10,10] Scalar 1"""
    )
    less_indent.print()
    assert P(less_indent.repr()) == less_indent


def test_parse_invalid_bang():
    incorrect_bang = """
    0 Add
      1 [3, 1] Scatter [0:2,0:1]
        2 [2, 1] Scalar 1 ! hi
      3 '0' [1, 3, 6] Scalar 7
      4 '1' [5, 3, 6] Scatter [0:5,0:3,0:3]
        5 Concat 0
          6 [2, 3, 3] Scalar 1.2
          7 [3, 3, 3] Scalar 1.2
      8 'ein' Einsum abc->abc
        4
      3
    """
    with pytest.raises(rc.ParseCircuitUnexpectedChildInfoError):
        rc.Parser()(incorrect_bang)

    incorrect_bang = """
    0 Add
      1 [3, 1] Scatter [0:2,0:1]
        2 [2, 1] Scalar 1
      3 '0' [1, 3, 6] Scalar 7
      4 '1' [5, 3, 6] Scatter [0:5,0:3,0:3]
        5 Concat 0
          6 [2, 3, 3] Scalar 1.2
          7 [3, 3, 3] Scalar 1.2
      8 'ein' Einsum abc->abc
        4 ! bye
      3
    """
    with pytest.raises(rc.ParseCircuitUnexpectedChildInfoError):
        rc.Parser()(incorrect_bang)


def test_rust_printing_repeated_names():
    stuff = """
    0 'hi' Add
      1 ' my' [3, 1] Scatter [0:2,0:1]
        2 '# 43' [2, 1] Scalar 1
      3 '1' [1, 3, 6] Scalar 7
      4 '1' [5, 3, 6] Scatter [0:5,0:3,0:3]
        5 'name stuff  ' Concat 0
          6 'hi' [2, 3, 3] Scalar 1.2
          7 [3, 3, 3] Scalar 1.2
      8 'ein' Einsum abc->abc
        4
      3
      9 'eins' Einsum ab->
        2
      10 'einy' Einsum abc->
        6
    """
    out = rc.Parser()(stuff).repr()
    expected = """
0 'hi' Add
  ' my' [3,1] Scatter [0:2,0:1]
    '# 43' [2,1] Scalar 1
  1 '1' [1,3,6] Scalar 7
  2 '1' [5,3,6] Scatter [0:5,0:3,0:3]
    'name stuff  ' Concat 0
      3 'hi' [2,3,3] Scalar 1.2
      4 [3,3,3] Scalar 1.2
  'ein' Einsum abc->abc
    2 '1'
  1 '1'
  'eins' Einsum ab->
    '# 43'
  'einy' Einsum abc->
    3 'hi'""".strip(
        "\n"
    )
    assert out == expected


def test_rust_printing_bijection_ref():
    setups: List[Tuple[str, Dict, str]] = [
        (
            """
0 [5, 3, 6] Add
  1 [3, 1] Scatter [0:2,0:1]
    'asdfadsf'
  10 [5, 3, 6] Scatter [0:5,0:3,0:3]
    4 [5, 3, 3] Concat 0
      5 'sc1' [2, 3, 3] Scalar 1.0
      6 'sc2' [3, 3, 3] Scalar 1.0""",
            {"asdfadsf": rc.Scalar(1.0, (2, 1), "asdfadsf_the_referred")},
            """
0 Add
  1 [3,1] Scatter [0:2,0:1]
    'asdfadsf_the_referred' [2,1] Scalar 1
  2 [5,3,6] Scatter [0:5,0:3,0:3]
    3 Concat 0
      'sc1' [2,3,3] Scalar 1
      'sc2' [3,3,3] Scalar 1""".strip(
                "\n"
            ),
        ),
    ]
    for string, refs, expected in setups:
        c = rc.Parser(reference_circuits=refs)(string)
        # if you update printer, verify print is as expected and then copy in new test strings
        c.print()
        print(expected)
        assert c.repr() == expected


def test_rust_printing_bijection_errors_correctly():
    strs = [
        """0 [5, 3, 6] Add d""",
        """0 [5, 3, 6] Add
 1 [3, 1] Scalar 1.0""",
        """0 [5, 3, 6] Add
    1 [3, 1] Scalar 1.0""",
        "0 [5, 3, 6] Add\n  1 [3,6] Scalar 1d",
        "0 [5, 3, 6] Add\n  1 [3,6] Index a[:3,:6]\n    2 [3,6] Scalar 1.0",
        "0 [5, 3, 6] Add\n  1 [3,6] Index [:3,:6]\n    2 [1,6] Scalar 1.0",
    ]
    for s in strs:
        try:
            c = rc.Parser()(s)
            c.print()
            print("YIKES!")
            assert False
        except ValueError as e:
            print(e)


def test_print_options():
    circ = rc.Parser()(
        """
0 [5, 3, 6] Add
  1 [3, 1] Scatter [0:2,0:1]
    2 [2, 1] Scalar 1.0
  10 [5, 3, 6] Scatter [0:5,0:3,0:3]
    4 [5, 3, 3] Concat 0
      5 'yoyo ' [2, 3, 3] Scalar 1.0
      6 'hihi' [3, 3, 3] Scalar 1.0"""
    )
    t = rc.new_traversal(end_depth=3)
    printer = rc.PrintOptions(bijection=False, traversal=t)
    # if you update printer, verify print is as expected and then copy in new test strings
    assert (
        printer.repr(circ)
        == """
0 Add
  1 [3,1] Scatter [0:2,0:1]
    2 [2,1] Scalar 1
  3 [5,3,6] Scatter [0:5,0:3,0:3]
    4 Concat 0 ...""".strip(
            "\n"
        )
    )

    assert printer.traversal == rc.new_traversal(end_depth=3)
    assert rc.PrintOptions(traversal=None).traversal == rc.new_traversal()
    assert rc.PrintHtmlOptions().overall_traversal == rc.IterativeMatcher(False)

    assert rc.PrintOptions(traversal="hi", bijection=False).print(circ) == circ.print()

    with pytest.raises(TypeError):
        rc.PrintOptions(traversal=3, bijection=False).print(circ)  # type: ignore

    printer = rc.PrintOptions(bijection=False, traversal=t, force_use_serial_numbers=True)
    # if you update printer, verify print is as expected and then copy in new test strings
    assert (
        printer.repr(circ)
        == """
0 Add
  1 [3,1] Scatter [0:2,0:1]
    2 [2,1] Scalar 1
  3 [5,3,6] Scatter [0:5,0:3,0:3]
    4 Concat 0 ...""".strip(
            "\n"
        )
    )
    assert (
        rc.PrintOptions(bijection=False, arrows=True, leaves_on_top=False).repr(circ)
        == """
0 Add
├‣1 [3,1] Scatter [0:2,0:1]
│ └‣2 [2,1] Scalar 1
└‣3 [5,3,6] Scatter [0:5,0:3,0:3]
  └‣4 Concat 0
    ├‣yoyo  [2,3,3] Scalar 1
    └‣hihi [3,3,3] Scalar 1""".strip(
            "\n"
        )
    )
    assert (
        rc.PrintOptions(bijection=False, arrows=True, leaves_on_top=True).repr(circ)
        == """
    ┌‣hihi [3,3,3] Scalar 1
    ├‣yoyo  [2,3,3] Scalar 1
  ┌‣4 Concat 0
┌‣3 [5,3,6] Scatter [0:5,0:3,0:3]
│ ┌‣2 [2,1] Scalar 1
├‣1 [3,1] Scatter [0:2,0:1]
0 Add""".strip(
            "\n"
        )
    )

    assert (
        rc.PrintOptions.repr_depth(circ, end_depth=2)
        == """0 Add
  1 [3,1] Scatter [0:2,0:1] ...
  2 [5,3,6] Scatter [0:5,0:3,0:3] ..."""
    )


def test_size_threshold_commenter():
    circ = rc.Parser()(
        """
0 [5, 3, 6] Add
  1 [3, 1] Scatter [0:2,0:1]
    2 [2, 1] Scalar 1.0
  10 [5, 3, 6] Scatter [0:5,0:3,0:3]
    4 [5, 3, 3] Concat 0
      5 'yoyo ' [2, 3, 3] Scalar 1.0
      6 'hihi' [3, 3, 3] Scalar 1.0"""
    )

    print(rc.PrintOptions(commenters=[rc.PrintOptions.size_threshold_commenter(16)]).repr(circ))

    assert (
        rc.PrintOptions(commenters=[rc.PrintOptions.size_threshold_commenter(16)]).repr(circ)
        == """0 Add # \x1b[31m90\x1b[0m
  1 [3,1] Scatter [0:2,0:1]
    2 [2,1] Scalar 1
  3 [5,3,6] Scatter [0:5,0:3,0:3] # \x1b[31m90\x1b[0m
    4 Concat 0 # \x1b[31m45\x1b[0m
      'yoyo ' [2,3,3] Scalar 1 # \x1b[31m18\x1b[0m
      'hihi' [3,3,3] Scalar 1 # \x1b[31m27\x1b[0m"""
    )


def test_dtype_commenter():
    circ = rc.Add(rc.Array(torch.zeros(5, dtype=torch.int8)), rc.Array(torch.zeros(5, dtype=torch.int8)))
    assert (
        rc.PrintOptions(commenters=[rc.PrintOptions.dtype_commenter()]).repr(circ)
        == """0 Add # int8
  1 [5] Array 50dba7b000ecd7c013961b6c # int8
  1 Array # int8"""
    )
    assert (
        rc.PrintOptions(commenters=[rc.PrintOptions.dtype_commenter(only_arrays=True)]).repr(circ)
        == """0 Add
  1 [5] Array 50dba7b000ecd7c013961b6c # int8
  1 Array # int8"""
    )
    assert (
        rc.PrintOptions(commenters=[rc.PrintOptions.dtype_commenter()]).repr(rc.Scalar(0.7, [1]))
        == """0 [1] Scalar 0.7"""
    )


def test_default_print():
    assert "RR_DEBUG_END_DEPTH" not in os.environ
    circ = rc.Parser()(
        """
0 [5, 3, 6] Add
  1 [3, 1] Scalar 3.7
  2 [3, 1] Scatter [0:2,0:1]
    3 [2, 1] Scalar 1.0
  10 [5, 3, 6] Scatter [0:5,0:3,0:3]
    4 [5, 3, 3] Concat 0
      5 'yoyo ' [2, 3, 3] Scalar 1.0
      6 'hihi' [3, 3, 3] Scalar 1.0"""
    )
    # if you update printer, verify print is as expected and then copy in new test strings
    assert (
        str(circ)
        == """<0 [5,3,6] Add
  1 [3,1] Scalar 3.7
  2 [3,1] Scatter [0:2,0:1] ...
  3 [5,3,6] Scatter [0:5,0:3,0:3] ...>"""
    )

    try:
        rc.set_debug_print_options(rc.PrintOptions.debug_default().evolve(traversal=rc.new_traversal(end_depth=3)))
        assert (
            str(circ)
            == """<0 [5,3,6] Add
  1 [3,1] Scalar 3.7
  2 [3,1] Scatter [0:2,0:1]
    3 [2,1] Scalar 1
  4 [5,3,6] Scatter [0:5,0:3,0:3]
    5 [5,3,3] Concat 0 ...>"""
        )
    finally:
        rc.set_debug_print_options(rc.PrintOptions.debug_default())


def test_print_color():
    circ = rc.Parser()(
        """
0 Add
  1 [3, 1] Scatter [0:2,0:1]
    2 [2, 1] Scalar 1.0
  4 Concat 0
    5 'yoyo ' [2, 3, 3] Scalar 1.0
    6 'hihi' [3, 3, 3] Scalar 1.0
    7 'hihi' [3, 3, 3] Scalar 1.1
    8 'hihi' [3, 3, 3] Scalar 1.2
    9 'hihi' [3, 3, 3] Scalar 1.3
    10 'hihi' [3, 3, 3] Scalar 1.4
    11 'hihi' [3, 3, 3] Scalar 1.5
    12 'hihi' [3, 3, 3] Scalar 1.6
    13 'hihi' [3, 3, 3] Scalar 1.7
    14 'hihi' [3, 3, 3] Scalar 1.8
    15 'hihi' [3, 3, 3] Scalar 1.9
      """
    )

    colorers = [
        (
            lambda x: abs(hash(x)),
            """
\x1b[95m0 Add\x1b[0m
  \x1b[93m1 [3,1] Scatter [0:2,0:1]\x1b[0m
    \x1b[95m2 [2,1] Scalar 1\x1b[0m
  \x1b[31m3 Concat 0\x1b[0m
    \x1b[92m'yoyo ' [2,3,3] Scalar 1\x1b[0m
    \x1b[92m4 'hihi' [3,3,3] Scalar 1\x1b[0m
    \x1b[32m5 'hihi' [3,3,3] Scalar 1.1\x1b[0m
    \x1b[34m6 'hihi' [3,3,3] Scalar 1.2\x1b[0m
    \x1b[33m7 'hihi' [3,3,3] Scalar 1.3\x1b[0m
    \x1b[95m8 'hihi' [3,3,3] Scalar 1.4\x1b[0m
    \x1b[93m9 'hihi' [3,3,3] Scalar 1.5\x1b[0m
    \x1b[96m10 'hihi' [3,3,3] Scalar 1.6\x1b[0m
    \x1b[32m11 'hihi' [3,3,3] Scalar 1.7\x1b[0m
    \x1b[90m12 'hihi' [3,3,3] Scalar 1.8\x1b[0m
    \x1b[31m13 'hihi' [3,3,3] Scalar 1.9\x1b[0m""".strip(
                "\n"
            ),
        ),
        (
            rc.PrintOptions.hash_colorer(),
            """
\x1b[36m0 Add\x1b[0m
  \x1b[91m1 [3,1] Scatter [0:2,0:1]\x1b[0m
    \x1b[95m2 [2,1] Scalar 1\x1b[0m
  \x1b[33m3 Concat 0\x1b[0m
    \x1b[92m'yoyo ' [2,3,3] Scalar 1\x1b[0m
    \x1b[92m4 'hihi' [3,3,3] Scalar 1\x1b[0m
    \x1b[32m5 'hihi' [3,3,3] Scalar 1.1\x1b[0m
    \x1b[97m6 'hihi' [3,3,3] Scalar 1.2\x1b[0m
    \x1b[31m7 'hihi' [3,3,3] Scalar 1.3\x1b[0m
    \x1b[95m8 'hihi' [3,3,3] Scalar 1.4\x1b[0m
    \x1b[91m9 'hihi' [3,3,3] Scalar 1.5\x1b[0m
    \x1b[96m10 'hihi' [3,3,3] Scalar 1.6\x1b[0m
    \x1b[32m11 'hihi' [3,3,3] Scalar 1.7\x1b[0m
    \x1b[90m12 'hihi' [3,3,3] Scalar 1.8\x1b[0m
    \x1b[33m13 'hihi' [3,3,3] Scalar 1.9\x1b[0m""".strip(
                "\n"
            ),
        ),
        (
            rc.PrintOptions.type_colorer(),
            """
\x1b[96m0 Add\x1b[0m
  \x1b[33m1 [3,1] Scatter [0:2,0:1]\x1b[0m
    \x1b[35m2 [2,1] Scalar 1\x1b[0m
  \x1b[92m3 Concat 0\x1b[0m
    \x1b[35m'yoyo ' [2,3,3] Scalar 1\x1b[0m
    \x1b[35m4 'hihi' [3,3,3] Scalar 1\x1b[0m
    \x1b[35m5 'hihi' [3,3,3] Scalar 1.1\x1b[0m
    \x1b[35m6 'hihi' [3,3,3] Scalar 1.2\x1b[0m
    \x1b[35m7 'hihi' [3,3,3] Scalar 1.3\x1b[0m
    \x1b[35m8 'hihi' [3,3,3] Scalar 1.4\x1b[0m
    \x1b[35m9 'hihi' [3,3,3] Scalar 1.5\x1b[0m
    \x1b[35m10 'hihi' [3,3,3] Scalar 1.6\x1b[0m
    \x1b[35m11 'hihi' [3,3,3] Scalar 1.7\x1b[0m
    \x1b[35m12 'hihi' [3,3,3] Scalar 1.8\x1b[0m
    \x1b[35m13 'hihi' [3,3,3] Scalar 1.9\x1b[0m""".strip(
                "\n"
            ),
        ),
    ]
    print()
    for (colorer, expected) in colorers:
        print()
        rc.PrintOptions(colorer=colorer).print(circ)
        print()
        print_multiline_escape(rc.PrintOptions(colorer=colorer).repr(circ))
        print()
        assert rc.PrintOptions(colorer=colorer).repr(circ) == expected

    # for i in range(20):
    #     rc.PrintOptions(colorer=rc.PrintOptions.fixed_color(i)).print(rc.Scalar(1, (2, 3)))


def test_size_colorer():
    circ = rc.Parser()(
        """
0 [] Add
  1 Einsum i->
    2 [1] Scalar 1.
  3 Einsum i->
    4 [1_000] Scalar 1.
  5 Einsum i->
    6 [1_000_000] Scalar 1.
  7 Einsum i->
    8 [1_000_000_000] Scalar 1.
  9 Einsum i->
    10 [100_000_000_000] Scalar 1.
  11 Einsum i->
    12 [10_000_000_000_000] Scalar 1.
      """
    )
    expected = """
0 Add
  1 Einsum i->
    2 [1] Scalar 1
  3 Einsum i->
    4 [1000] Scalar 1
  5 Einsum i->
    \x1b[32m6 [1000000] Scalar 1\x1b[0m
  7 Einsum i->
    \x1b[33m8 [1000000000] Scalar 1\x1b[0m
  9 Einsum i->
    \x1b[31m10 [100000000000] Scalar 1\x1b[0m
  11 Einsum i->
    \x1b[31m12 [10000000000000] Scalar 1\x1b[0m""".strip(
        "\n"
    )
    colorer = rc.PrintOptions.size_colorer()
    print()
    rc.PrintOptions(colorer=colorer).print(circ)
    print()
    print_multiline_escape(rc.PrintOptions(colorer=colorer).repr(circ))
    print()
    assert rc.PrintOptions(colorer=colorer).repr(circ) == expected


@hypothesis.settings(deadline=None)
@hypothesis.given(c=get_c_st())
def test_self_hash(c: cCircuit):
    try:
        rust_circ = py_to_rust(c)
        replaced = rust_circ.map_children(lambda x: rc.Scalar(1.0, x.shape, None))
        assert rc.compute_self_hash(replaced) == rc.compute_self_hash(rust_circ)
    except Exception as e:
        print(c)
        raise e


P = rc.Parser()


def test_self_hash_2():
    circuits = [
        P("0 Einsum ab,ab->\n  1 [2,3] Scalar 1\n  2 [2,3] Scalar 2"),
        P("0 Einsum ad,ad->\n  1 [2,3] Scalar 1\n  2 [2,3] Scalar 2"),
        P("0 Einsum ab,ab->\n  1 [2,3] Scalar 1.1\n  2 [2,3] Scalar 2"),
        P("0 Einsum ab,ab->a\n  1 [2,3] Scalar 1\n  2 [2,3] Scalar 2"),
        P("0 Einsum ab,ab->a\n  1 [2,5] Scalar 1\n  2 [2,5] Scalar 2"),
    ]
    for circuit in circuits:
        print(rc.compute_self_hash(circuit))


def test_print_diff():
    P = rc.Parser(tensors_as_random=True, allow_hash_with_random=True)
    circuit_pairs = [
        (
            P(
                """0 Einsum ab,bc->ac
  1 [3,4] Scalar 2
  4 [4,5] Scalar 2
    """
            ),
            P(
                """0 Einsum ab,bc->ac
  1 [3,4] Scalar 3
  4 [4,5] Scalar 2
    """
            ),
        ),
        (
            P(
                """0 Einsum ab,bc->ac
  1 Einsum ab,bc->ac
    2 [2,3] Scalar 1
    3 [3,4] Scalar 1
  4 [4,5] Scalar 2
    """
            ),
            P(
                """0 Einsum ab,bc->ac
  1 Einsum ab,bc->ac
    2 [2,3] Scalar 3
    3 [3,4] Scalar 1
  4 [4,5] Scalar 2
    """
            ),
        ),
        (
            P(
                """0 Einsum ab,bc->ac
  1 Einsum ab,bc->ac
    2 [2,3] Scalar 1
    3 [3,4] Scalar 1
  4 [4,5] Add
    5 's0' [4,1] Symbol
    6 's1' [5] Symbol
    """
            ),
            P(
                """0 Einsum ab,bc->ac
  1 Einsum ab,bc->ac
    2 [2,3] Scalar 1
    3 [3,4] Scalar 2
  4 [4,5] Add
    5 's0' [4,1] Symbol
    6 's1' [1,5] Symbol
    """
            ),
        ),
        (
            P(
                """
0 Add
  1 Add
    2 Add
      'hi' [2] Symbol
      'hi' [2] Symbol
    2
  1
  'rando' [2] Symbol
  'randosame' [2] Symbol
    """
            ),
            P(
                """
0 Add
  1 Add
    2 Add
      'ho' [2] Symbol
      'ho' [2] Symbol
    2
  1
  'rando2' [2] Symbol
  'randosame' [2] Symbol
    """
            ),
        ),
        (
            P(
                """
'logit_diff_balanced_unbalanced' Add
  'logit_balanced' Einsum b,b->
    'final.n_pos0' Add
      'final.n.w.bias' [56] Array eeb9a82a41b21694aab7901b
      'final.n.y_out_pos0' Einsum b,->b
        'final.n.y_scale_pos0' Einsum b,b->b
          'final.n.z_mean_pos0' Einsum b,bc->c
            'final.inp_pos0' Add
              'tok_embeds_pos0' Einsum b,bc->c
                'tokens_pos0' Index [0,:]
                  'input_tag_logits' Tag 062c34e2-867b-4930-8ac4-bf2b16abd1e8
                    'tokens_var' DiscreteVar
                      'tokens' [1,22,5] Array 603427427c088ab4dfadc1fb
                      0 Tag 480058d7-0dd4-4105-86cc-9fa7ed06583b
                        1 [1] Scalar 1
                'tok_embed_mat' [5,56] Array 045bc7a114683c5c98190abf
              'w.pos_embeds_pos0' Index [0,:]
                'w.pos_embeds' [22,56] Array 3bf234b0ff1bec7000d09345
              'a0.out_pos0' Add
                'a0.out_pos0_head_0_sum' Einsum ab->b
                  'a0.out_pos0_head_0' Index [0:1,:]
                    'a0.out_pos0_keep_head' Einsum bc,bcd->bd
                      'a0.comb_v_pos0' Einsum ac,acd->ad
                        'a0.probs_pos0' GeneralFunction softmax
                          'a0.scores_padded_pos0' Add
                            'a0.scores_pos_mask_pos0' Einsum ac,c->ac
                              'a0.scores_not_masked_pos0' Einsum ac,adc,->ad
                                'a0.q_p_bias_pos0' Add
                                  'a0.q_pos0' Einsum abc,c->ab
                                    'a0.w.q' [2,28,56] Array d93bfa57cf2a27b30dc250d6
                                    'a0.n_pos0' Add
                                      'a0.n.w.bias' [56] Array a1090771831dccc00a5dcefa
                                      'a0.n.y_out_pos0' Einsum b,->b
                                        'a0.n.y_scale_pos0' Einsum b,b->b
                                          'a0.n.z_mean_pos0' Einsum b,bc->c
                                            'a0.inp_pos0' Add
                                              'tok_embeds_pos0'
                                              'w.pos_embeds_pos0'
                                            'a0.n.c.sub_mean' [56,56] Array 73b0b67535e8dd93d2120db9
                                          'a0.n.w.scale' [56] Array 152c532d249ccfffc16c06c4
                                        'a0.n.full_mul_pos0' GeneralFunction rsqrt
                                          'a0.n.c.var_p_eps_pos0' Add
                                            'a0.n.var_pos0' Einsum b,b,->
                                              'a0.n.z_mean_pos0'
                                              'a0.n.z_mean_pos0'
                                              'a0.n.c.recip_h_size' [] Scalar 0.017857142857142856
                                            'a0.n.eps' [] Scalar 0.00001
                                  'a0.w.q_bias_pos0' Index [:,0,:]
                                    'a0.w.q_bias' [2,1,28] Array 06ef6fea473ce03a5ff36602
                                'a0.k_p_bias' Add
                                  'a0.k' Einsum abc,dc->adb
                                    'a0.w.k' [2,28,56] Array 8917d02e3fed8bb5f5eba8b3
                                    'a0.n' Add
                                      'a0.n.w.bias'
                                      'a0.n.y_out' Einsum ab,a->ab
                                        'a0.n.y_scale' Einsum ab,b->ab
                                          'a0.n.z_mean' Einsum ab,bc->ac
                                            'a0.inp' Add
                                              'tok_embeds' Einsum ab,bc->ac
                                                'input_tag_logits'
                                                'tok_embed_mat'
                                              'w.pos_embeds'
                                            'a0.n.c.sub_mean'
                                          'a0.n.w.scale'
                                        'a0.n.full_mul' GeneralFunction rsqrt
                                          'a0.n.c.var_p_eps' Add
                                            'a0.n.var' Einsum ab,ab,->a
                                              'a0.n.z_mean'
                                              'a0.n.z_mean'
                                              'a0.n.c.recip_h_size'
                                            'a0.n.eps'
                                  'a0.w.k_bias' [2,1,28] Array 64f30e6153bfa9b2f0c5e415
                                'a0.c.div_head_size' [] Scalar 0.1889822365046136
                              'c.neg_pos_mask' Add
                                'one' [] Scalar 1
                                2 'ScalarMul' Einsum a,->a
                                  'pos_mask' Index [:,1]
                                    'input_tag_logits'
                                  3 'unnamed' [] Scalar -1
                            'ScalarMul_pos0' Einsum ac,->ac
                              'Unsqueeze_pos0' Rearrange a:22 -> () a:22
                                'pos_mask'
                              4 'unnamed' [] Scalar -10000
                        'a0.v_p_bias' Add
                          'a0.v' Einsum abc,dc->adb
                            'a0.w.v' [2,28,56] Array 6266ba0d9297e3401ecaa15a
                            'a0.n'
                          'a0.w.v_bias' [2,1,28] Array 5480b6d7d30b03372375f556
                      'a0.w.out' [2,28,56] Array f6abfd1f4f8100780784d1f7
                'a0.out_pos0_head_1_sum' Einsum ab->b
                  'a0.out_pos0_head_1' Index [1:2,:]
                    'a0.out_pos0_keep_head'
              'a0.w.bias_out' [56] Array ef811930aeadb47e844242f0
              'm0_pos0' Einsum b,cb->c
                'm0.act_pos0' GeneralFunction relu
                  'm0.add0_pos0' Add
                    'm0.before_product0_pos0' Einsum b,cb->c
                      'm0.n_pos0' Add
                        'm0.n.w.bias' [56] Array abdfc87093d25769cb5340d8
                        'm0.n.y_out_pos0' Einsum b,->b
                          'm0.n.y_scale_pos0' Einsum b,b->b
                            'm0.n.z_mean_pos0' Einsum b,bc->c
                              'm0.inp_pos0' Add
                                'tok_embeds_pos0'
                                'w.pos_embeds_pos0'
                                'a0.out_pos0'
                                'a0.w.bias_out'
                              'm0.n.c.sub_mean' [56,56] Array 73b0b67535e8dd93d2120db9
                            'm0.n.w.scale' [56] Array eacc41edc6480f2c11a896d2
                          'm0.n.full_mul_pos0' GeneralFunction rsqrt
                            'm0.n.c.var_p_eps_pos0' Add
                              'm0.n.var_pos0' Einsum b,b,->
                                'm0.n.z_mean_pos0'
                                'm0.n.z_mean_pos0'
                                'm0.n.c.recip_h_size' [] Scalar 0.017857142857142856
                              'm0.n.eps' [] Scalar 0.00001
                      'm0.w.w0' [56,56] Array 8e3d05e33535d2ff7a650701
                    'm0.w.b0' [56] Array e533e9975b83858743c9085e
                'm0.w.w1' [56,56] Array 3c486e70d7cec981912fb94c
              'm0.w.b1' [56] Array 75aed749c4c165b60488cd7c
              'a1.out_pos0' Add
                'a1.out_pos0_head_0_sum' Einsum ab->b
                  'a1.out_pos0_head_0' Index [0:1,:]
                    'a1.out_pos0_keep_head' Einsum bc,bcd->bd
                      'a1.comb_v_pos0' Einsum ac,acd->ad
                        'a1.probs_pos0' GeneralFunction softmax
                          'a1.scores_padded_pos0' Add
                            'a1.scores_pos_mask_pos0' Einsum ac,c->ac
                              'a1.scores_not_masked_pos0' Einsum ac,adc,->ad
                                'a1.q_p_bias_pos0' Add
                                  'a1.q_pos0' Einsum abc,c->ab
                                    'a1.w.q' [2,28,56] Array d171720d20b7b5143362d3ac
                                    'a1.n_pos0' Add
                                      'a1.n.w.bias' [56] Array 9c5ca27d53f3f58ec59ec0a0
                                      'a1.n.y_out_pos0' Einsum b,->b
                                        'a1.n.y_scale_pos0' Einsum b,b->b
                                          'a1.n.z_mean_pos0' Einsum b,bc->c
                                            'a1.inp_pos0' Add
                                              'tok_embeds_pos0'
                                              'w.pos_embeds_pos0'
                                              'a0.out_pos0'
                                              'a0.w.bias_out'
                                              'm0_pos0'
                                              'm0.w.b1'
                                            'a1.n.c.sub_mean' [56,56] Array 73b0b67535e8dd93d2120db9
                                          'a1.n.w.scale' [56] Array 35f1b4f1ec7b3f7543667e09
                                        'a1.n.full_mul_pos0' GeneralFunction rsqrt
                                          'a1.n.c.var_p_eps_pos0' Add
                                            'a1.n.var_pos0' Einsum b,b,->
                                              'a1.n.z_mean_pos0'
                                              'a1.n.z_mean_pos0'
                                              'a1.n.c.recip_h_size' [] Scalar 0.017857142857142856
                                            'a1.n.eps' [] Scalar 0.00001
                                  'a1.w.q_bias_pos0' Index [:,0,:]
                                    'a1.w.q_bias' [2,1,28] Array 02a0c59e812c211740fb27bb
                                'a1.k_p_bias' Add
                                  'a1.k' Einsum abc,dc->adb
                                    'a1.w.k' [2,28,56] Array 6a260810e3b7af22a8544a16
                                    'a1.n' Add
                                      'a1.n.w.bias'
                                      'a1.n.y_out' Einsum ab,a->ab
                                        'a1.n.y_scale' Einsum ab,b->ab
                                          'a1.n.z_mean' Einsum ab,bc->ac
                                            'a1.inp' Add
                                              'tok_embeds'
                                              'w.pos_embeds'
                                              'a0.out' Einsum abc,bcd->ad
                                                'a0.comb_v' Einsum abc,acd->bad
                                                  'a0.probs' GeneralFunction softmax
                                                    'a0.scores_padded' Add
                                                      'a0.scores_pos_mask' Einsum abc,c->abc
                                                        'a0.scores_not_masked' Einsum abc,adc,->abd
                                                          'a0.q_p_bias' Add
                                                            'a0.q' Einsum abc,dc->adb
                                                              'a0.w.q'
                                                              'a0.n'
                                                            'a0.w.q_bias'
                                                          'a0.k_p_bias'
                                                          'a0.c.div_head_size'
                                                        'c.neg_pos_mask'
                                                      5 'ScalarMul' Einsum abc,->abc
                                                        'Unsqueeze' Rearrange a:22 -> () () a:22
                                                          'pos_mask'
                                                        4 'unnamed'
                                                  'a0.v_p_bias'
                                                'a0.w.out'
                                              'a0.w.bias_out'
                                              'm0' Einsum ab,cb->ac
                                                'm0.act' GeneralFunction relu
                                                  'm0.add0' Add
                                                    'm0.before_product0' Einsum ab,cb->ac
                                                      'm0.n' Add
                                                        'm0.n.w.bias'
                                                        'm0.n.y_out' Einsum ab,a->ab
                                                          'm0.n.y_scale' Einsum ab,b->ab
                                                            'm0.n.z_mean' Einsum ab,bc->ac
                                                              'm0.inp' Add
                                                                'tok_embeds'
                                                                'w.pos_embeds'
                                                                'a0.out'
                                                                'a0.w.bias_out'
                                                              'm0.n.c.sub_mean'
                                                            'm0.n.w.scale'
                                                          'm0.n.full_mul' GeneralFunction rsqrt
                                                            'm0.n.c.var_p_eps' Add
                                                              'm0.n.var' Einsum ab,ab,->a
                                                                'm0.n.z_mean'
                                                                'm0.n.z_mean'
                                                                'm0.n.c.recip_h_size'
                                                              'm0.n.eps'
                                                      'm0.w.w0'
                                                    'm0.w.b0'
                                                'm0.w.w1'
                                              'm0.w.b1'
                                            'a1.n.c.sub_mean'
                                          'a1.n.w.scale'
                                        'a1.n.full_mul' GeneralFunction rsqrt
                                          'a1.n.c.var_p_eps' Add
                                            'a1.n.var' Einsum ab,ab,->a
                                              'a1.n.z_mean'
                                              'a1.n.z_mean'
                                              'a1.n.c.recip_h_size'
                                            'a1.n.eps'
                                  'a1.w.k_bias' [2,1,28] Array 4f504eb0607ce4368fea6e28
                                'a1.c.div_head_size' [] Scalar 0.1889822365046136
                              'c.neg_pos_mask'
                            'ScalarMul_pos0'
                        'a1.v_p_bias' Add
                          'a1.v' Einsum abc,dc->adb
                            'a1.w.v' [2,28,56] Array 247f61024c75b09c2c1eb7c3
                            'a1.n'
                          'a1.w.v_bias' [2,1,28] Array 6166fa008805ef7f07a08185
                      'a1.w.out' [2,28,56] Array ba174133f8cea9ec3dbd0ad3
                'a1.out_pos0_head_1_sum' Einsum ab->b
                  'a1.out_pos0_head_1' Index [1:2,:]
                    'a1.out_pos0_keep_head'
              'a1.w.bias_out' [56] Array bf309aff1cd4964da4cf8f46
              'm1_pos0' Einsum b,cb->c
                'm1.act_pos0' GeneralFunction relu
                  'm1.add0_pos0' Add
                    'm1.before_product0_pos0' Einsum b,cb->c
                      'm1.n_pos0' Add
                        'm1.n.w.bias' [56] Array a866530bc942a654a3dfede7
                        'm1.n.y_out_pos0' Einsum b,->b
                          'm1.n.y_scale_pos0' Einsum b,b->b
                            'm1.n.z_mean_pos0' Einsum b,bc->c
                              'm1.inp_pos0' Add
                                'tok_embeds_pos0'
                                'w.pos_embeds_pos0'
                                'a0.out_pos0'
                                'a0.w.bias_out'
                                'm0_pos0'
                                'm0.w.b1'
                                'a1.out_pos0'
                                'a1.w.bias_out'
                              'm1.n.c.sub_mean' [56,56] Array 73b0b67535e8dd93d2120db9
                            'm1.n.w.scale' [56] Array f9953fb52e69501d0c0d087b
                          'm1.n.full_mul_pos0' GeneralFunction rsqrt
                            'm1.n.c.var_p_eps_pos0' Add
                              'm1.n.var_pos0' Einsum b,b,->
                                'm1.n.z_mean_pos0'
                                'm1.n.z_mean_pos0'
                                'm1.n.c.recip_h_size' [] Scalar 0.017857142857142856
                              'm1.n.eps' [] Scalar 0.00001
                      'm1.w.w0' [56,56] Array 5816774b9181cb66fd8bb246
                    'm1.w.b0' [56] Array 597a51357ba300db411e6d75
                'm1.w.w1' [56,56] Array 8ddff693c2cdc7e11833a6d7
              'm1.w.b1' [56] Array 74f91880b278f8ab6ec762e4
              'a2.out_pos0' Add
                'a2.out_pos0_head_0_sum' Einsum ab->b
                  'a2.out_pos0_head_0' Index [0:1,:]
                    'a2.out_pos0_keep_head' Einsum bc,bcd->bd
                      'a2.comb_v_pos0' Einsum ac,acd->ad
                        'a2.probs_pos0' GeneralFunction softmax
                          'a2.scores_padded_pos0' Add
                            'a2.scores_pos_mask_pos0' Einsum ac,c->ac
                              'a2.scores_not_masked_pos0' Einsum ac,adc,->ad
                                'a2.q_p_bias_pos0' Add
                                  'a2.q_pos0' Einsum abc,c->ab
                                    'a2.w.q' [2,28,56] Array 9bbbb8061e5f16ca9e30a660
                                    'a2.n_pos0' Add
                                      'a2.n.w.bias' [56] Array efc80eea18b8832c38eae67d
                                      'a2.n.y_out_pos0' Einsum b,->b
                                        'a2.n.y_scale_pos0' Einsum b,b->b
                                          'a2.n.z_mean_pos0' Einsum b,bc->c
                                            'a2.inp_pos0' Add
                                              'tok_embeds_pos0'
                                              'w.pos_embeds_pos0'
                                              'a0.out_pos0'
                                              'a0.w.bias_out'
                                              'm0_pos0'
                                              'm0.w.b1'
                                              'a1.out_pos0'
                                              'a1.w.bias_out'
                                              'm1_pos0'
                                              'm1.w.b1'
                                            'a2.n.c.sub_mean' [56,56] Array 73b0b67535e8dd93d2120db9
                                          'a2.n.w.scale' [56] Array 0ca9ee9df54aeaf7244dc23d
                                        'a2.n.full_mul_pos0' GeneralFunction rsqrt
                                          'a2.n.c.var_p_eps_pos0' Add
                                            'a2.n.var_pos0' Einsum b,b,->
                                              'a2.n.z_mean_pos0'
                                              'a2.n.z_mean_pos0'
                                              'a2.n.c.recip_h_size' [] Scalar 0.017857142857142856
                                            'a2.n.eps' [] Scalar 0.00001
                                  'a2.w.q_bias_pos0' Index [:,0,:]
                                    'a2.w.q_bias' [2,1,28] Array eec5e06f978cc768053299b6
                                'a2.k_p_bias' Add
                                  'a2.k' Einsum abc,dc->adb
                                    'a2.w.k' [2,28,56] Array 65a03f81065243b62fa9bfec
                                    'a2.n' Add
                                      'a2.n.w.bias'
                                      'a2.n.y_out' Einsum ab,a->ab
                                        'a2.n.y_scale' Einsum ab,b->ab
                                          'a2.n.z_mean' Einsum ab,bc->ac
                                            'a2.inp' Add
                                              'tok_embeds'
                                              'w.pos_embeds'
                                              'a0.out'
                                              'a0.w.bias_out'
                                              'm0'
                                              'm0.w.b1'
                                              'a1.out' Einsum abc,bcd->ad
                                                'a1.comb_v' Einsum abc,acd->bad
                                                  'a1.probs' GeneralFunction softmax
                                                    'a1.scores_padded' Add
                                                      'a1.scores_pos_mask' Einsum abc,c->abc
                                                        'a1.scores_not_masked' Einsum abc,adc,->abd
                                                          'a1.q_p_bias' Add
                                                            'a1.q' Einsum abc,dc->adb
                                                              'a1.w.q'
                                                              'a1.n'
                                                            'a1.w.q_bias'
                                                          'a1.k_p_bias'
                                                          'a1.c.div_head_size'
                                                        'c.neg_pos_mask'
                                                      5 'ScalarMul'
                                                  'a1.v_p_bias'
                                                'a1.w.out'
                                              'a1.w.bias_out'
                                              'm1' Einsum ab,cb->ac
                                                'm1.act' GeneralFunction relu
                                                  'm1.add0' Add
                                                    'm1.before_product0' Einsum ab,cb->ac
                                                      'm1.n' Add
                                                        'm1.n.w.bias'
                                                        'm1.n.y_out' Einsum ab,a->ab
                                                          'm1.n.y_scale' Einsum ab,b->ab
                                                            'm1.n.z_mean' Einsum ab,bc->ac
                                                              'm1.inp' Add
                                                                'tok_embeds'
                                                                'w.pos_embeds'
                                                                'a0.out'
                                                                'a0.w.bias_out'
                                                                'm0'
                                                                'm0.w.b1'
                                                                'a1.out'
                                                                'a1.w.bias_out'
                                                              'm1.n.c.sub_mean'
                                                            'm1.n.w.scale'
                                                          'm1.n.full_mul' GeneralFunction rsqrt
                                                            'm1.n.c.var_p_eps' Add
                                                              'm1.n.var' Einsum ab,ab,->a
                                                                'm1.n.z_mean'
                                                                'm1.n.z_mean'
                                                                'm1.n.c.recip_h_size'
                                                              'm1.n.eps'
                                                      'm1.w.w0'
                                                    'm1.w.b0'
                                                'm1.w.w1'
                                              'm1.w.b1'
                                            'a2.n.c.sub_mean'
                                          'a2.n.w.scale'
                                        'a2.n.full_mul' GeneralFunction rsqrt
                                          'a2.n.c.var_p_eps' Add
                                            'a2.n.var' Einsum ab,ab,->a
                                              'a2.n.z_mean'
                                              'a2.n.z_mean'
                                              'a2.n.c.recip_h_size'
                                            'a2.n.eps'
                                  'a2.w.k_bias' [2,1,28] Array 9381312ae69eb63bc827a201
                                'a2.c.div_head_size' [] Scalar 0.1889822365046136
                              'c.neg_pos_mask'
                            'ScalarMul_pos0'
                        'a2.v_p_bias' Add
                          'a2.v' Einsum abc,dc->adb
                            'a2.w.v' [2,28,56] Array b3ee0ca518631de9db93012c
                            'a2.n'
                          'a2.w.v_bias' [2,1,28] Array fee1b26e3976991fb36fc784
                      'a2.w.out' [2,28,56] Array a6b80e015342412235a0dfe1
                'a2.out_pos0_head_1_sum' Einsum ab->b
                  'a2.out_pos0_head_1' Index [1:2,:]
                    'a2.out_pos0_keep_head'
              'a2.w.bias_out' [56] Array 5aab3d97625d2724cd18c894
              'm2_pos0' Einsum b,cb->c
                'm2.act_pos0' GeneralFunction relu
                  'm2.add0_pos0' Add
                    'm2.before_product0_pos0' Einsum b,cb->c
                      'm2.n_pos0' Add
                        'm2.n.w.bias' [56] Array 4928c790a030f627a207af3d
                        'm2.n.y_out_pos0' Einsum b,->b
                          'm2.n.y_scale_pos0' Einsum b,b->b
                            'm2.n.z_mean_pos0' Einsum b,bc->c
                              'm2.inp_pos0' Add
                                'tok_embeds_pos0'
                                'w.pos_embeds_pos0'
                                'a0.out_pos0'
                                'a0.w.bias_out'
                                'm0_pos0'
                                'm0.w.b1'
                                'a1.out_pos0'
                                'a1.w.bias_out'
                                'm1_pos0'
                                'm1.w.b1'
                                'a2.out_pos0'
                                'a2.w.bias_out'
                              'm2.n.c.sub_mean' [56,56] Array 73b0b67535e8dd93d2120db9
                            'm2.n.w.scale' [56] Array 407fac965b99d87edd348b62
                          'm2.n.full_mul_pos0' GeneralFunction rsqrt
                            'm2.n.c.var_p_eps_pos0' Add
                              'm2.n.var_pos0' Einsum b,b,->
                                'm2.n.z_mean_pos0'
                                'm2.n.z_mean_pos0'
                                'm2.n.c.recip_h_size' [] Scalar 0.017857142857142856
                              'm2.n.eps' [] Scalar 0.00001
                      'm2.w.w0' [56,56] Array b481ef72e75d0cd36cca8f96
                    'm2.w.b0' [56] Array 48618f201fe9087f87885333
                'm2.w.w1' [56,56] Array f87c96393e560aae63e9aecd
              'm2.w.b1' [56] Array a1abdf91f20e023361a01060
            'final.n.c.sub_mean' [56,56] Array 73b0b67535e8dd93d2120db9
          'final.n.w.scale' [56] Array 4576df694468664e4ac7ef98
        'final.n.full_mul_pos0' GeneralFunction rsqrt
          'final.n.c.var_p_eps_pos0' Add
            'final.n.var_pos0' Einsum b,b,->
              'final.n.z_mean_pos0'
              'final.n.z_mean_pos0'
              'final.n.c.recip_h_size' [] Scalar 0.017857142857142856
            'final.n.eps' [] Scalar 0.00001
    'w.unembed_bal' Index [1,:]
      'w.unembed' [2,56] Array c2f2e68b7f3fd72f821721a6
  'neg logit_unbalanced' Einsum ,->
    'logit_unbalanced' Einsum b,b->
      'final.n_pos0'
      'w.unembed_unbal' Index [0,:]
        'w.unembed'
    'neg_1' [] Scalar -1
        """
            ),
            P(
                """
'logit_diff_balanced_unbalanced' Add
  'logit_balanced' Einsum b,b->
    'final.n_pos0' Add
      'final.n.w.bias' [56] Array 6b49d14d528b372702a2e3b7
      'final.n.y_out_pos0' Einsum b,->b
        'final.n.y_scale_pos0' Einsum b,b->b
          'final.n.z_mean_pos0' Einsum b,bc->c
            'final.inp_pos0' Add
              'tok_embeds_pos0' Einsum b,bc->c
                'tokens_pos0' Index [0,:]
                  'tokens' [22,5] Symbol d0ce964d-4e8e-46ce-8fa1-91fa1b7f7b92
                'tok_embed_mat' [5,56] Array 24b7cbef931e7cf5f5fc28fa
              'w.pos_embeds_pos0' Index [0,:]
                'w.pos_embeds' [22,56] Array 24e659e5f4fa92a567ab4774
              'a0.out_pos0' Add
                'a0.out_pos0_head_0_sum' Einsum ab->b
                  'a0.out_pos0_head_0' Index [0:1,:]
                    'a0.out_pos0_keep_head' Einsum bc,bcd->bd
                      'a0.comb_v_pos0' Einsum ac,acd->ad
                        'a0.probs_pos0' GeneralFunction softmax
                          'a0.scores_padded_pos0' Add
                            'a0.scores_pos_mask_pos0' Einsum ac,c->ac
                              'a0.scores_not_masked_pos0' Einsum ac,adc,->ad
                                'a0.q_p_bias_pos0' Add
                                  'a0.q_pos0' Einsum abc,c->ab
                                    'a0.w.q' [2,28,56] Array 3077e7db78308fcdb8615772
                                    'a0.n_pos0' Add
                                      'a0.n.w.bias' [56] Array a2f364532da26d9ac3526286
                                      'a0.n.y_out_pos0' Einsum b,->b
                                        'a0.n.y_scale_pos0' Einsum b,b->b
                                          'a0.n.z_mean_pos0' Einsum b,bc->c
                                            'a0.inp_pos0' Add
                                              'tok_embeds_pos0'
                                              'w.pos_embeds_pos0'
                                            'a0.n.c.sub_mean' [56,56] Array 325ba9e9e24767e94536f106
                                          'a0.n.w.scale' [56] Array 8468c210313dea1608d6378d
                                        'a0.n.full_mul_pos0' GeneralFunction rsqrt
                                          'a0.n.c.var_p_eps_pos0' Add
                                            'a0.n.var_pos0' Einsum b,b,->
                                              'a0.n.z_mean_pos0'
                                              'a0.n.z_mean_pos0'
                                              'a0.n.c.recip_h_size' [] Scalar 0.017857142857142856
                                            'a0.n.eps' [] Scalar 0.00001
                                  'a0.w.q_bias_pos0' Index [:,0,:]
                                    'a0.w.q_bias' [2,1,28] Array 6428063697a3febc94f260ed
                                'a0.k_p_bias' Add
                                  'a0.k' Einsum abc,dc->adb
                                    'a0.w.k' [2,28,56] Array a3c22b26f43f450dd4417892
                                    'a0.n' Add
                                      'a0.n.w.bias'
                                      'a0.n.y_out' Einsum ab,a->ab
                                        'a0.n.y_scale' Einsum ab,b->ab
                                          'a0.n.z_mean' Einsum ab,bc->ac
                                            'a0.inp' Add
                                              'tok_embeds' Einsum ab,bc->ac
                                                'tokens'
                                                'tok_embed_mat'
                                              'w.pos_embeds'
                                            'a0.n.c.sub_mean'
                                          'a0.n.w.scale'
                                        'a0.n.full_mul' GeneralFunction rsqrt
                                          'a0.n.c.var_p_eps' Add
                                            'a0.n.var' Einsum ab,ab,->a
                                              'a0.n.z_mean'
                                              'a0.n.z_mean'
                                              'a0.n.c.recip_h_size'
                                            'a0.n.eps'
                                  'a0.w.k_bias' [2,1,28] Array d702254edc06572691386e2a
                                'a0.c.div_head_size' [] Scalar 0.1889822365046136
                              'c.neg_pos_mask' Add
                                'one' [] Scalar 1
                                0 'ScalarMul' Einsum a,->a
                                  'pos_mask' Index [:,1]
                                    'tokens'
                                  1 'unnamed' [] Scalar -1
                            'ScalarMul_pos0' Einsum ac,->ac
                              'Unsqueeze_pos0' Rearrange a:22 -> () a:22
                                'pos_mask'
                              2 'unnamed' [] Scalar -10000
                        'a0.v_p_bias' Add
                          'a0.v' Einsum abc,dc->adb
                            'a0.w.v' [2,28,56] Array 99b1866cf05e772b6645832c
                            'a0.n'
                          'a0.w.v_bias' [2,1,28] Array 8b2f6908b683001dae532509
                      'a0.w.out' [2,28,56] Array fa5cc32c773a90d37646f354
                'a0.out_pos0_head_1_sum' Einsum ab->b
                  'a0.out_pos0_head_1' Index [1:2,:]
                    'a0.out_pos0_keep_head'
              'a0.w.bias_out' [56] Array a6a4e8ab956ae6af221daddf
              'm0_pos0' Einsum b,cb->c
                'm0.act_pos0' GeneralFunction relu
                  'm0.add0_pos0' Add
                    'm0.before_product0_pos0' Einsum b,cb->c
                      'm0.n_pos0' Add
                        'm0.n.w.bias' [56] Array fec0ef838ad56e332ad9958f
                        'm0.n.y_out_pos0' Einsum b,->b
                          'm0.n.y_scale_pos0' Einsum b,b->b
                            'm0.n.z_mean_pos0' Einsum b,bc->c
                              'm0.inp_pos0' Add
                                'tok_embeds_pos0'
                                'w.pos_embeds_pos0'
                                'a0.out_pos0'
                                'a0.w.bias_out'
                              'm0.n.c.sub_mean' [56,56] Array 325ba9e9e24767e94536f106
                            'm0.n.w.scale' [56] Array 932215c27175655d1c5b3447
                          'm0.n.full_mul_pos0' GeneralFunction rsqrt
                            'm0.n.c.var_p_eps_pos0' Add
                              'm0.n.var_pos0' Einsum b,b,->
                                'm0.n.z_mean_pos0'
                                'm0.n.z_mean_pos0'
                                'm0.n.c.recip_h_size' [] Scalar 0.017857142857142856
                              'm0.n.eps' [] Scalar 0.00001
                      'm0.w.w0' [56,56] Array 2775bbed88fc828aeb2d6d81
                    'm0.w.b0' [56] Array d3370780192530e1c174dd54
                'm0.w.w1' [56,56] Array 66d491452d9c628ef37ef6b3
              'm0.w.b1' [56] Array 8985e96b24913782f3cda1fb
              'a1.out_pos0' Add
                'a1.out_pos0_head_0_sum' Einsum ab->b
                  'a1.out_pos0_head_0' Index [0:1,:]
                    'a1.out_pos0_keep_head' Einsum bc,bcd->bd
                      'a1.comb_v_pos0' Einsum ac,acd->ad
                        'a1.probs_pos0' GeneralFunction softmax
                          'a1.scores_padded_pos0' Add
                            'a1.scores_pos_mask_pos0' Einsum ac,c->ac
                              'a1.scores_not_masked_pos0' Einsum ac,adc,->ad
                                'a1.q_p_bias_pos0' Add
                                  'a1.q_pos0' Einsum abc,c->ab
                                    'a1.w.q' [2,28,56] Array 27ded626381bd8ddcbe43e24
                                    'a1.n_pos0' Add
                                      'a1.n.w.bias' [56] Array bec75fb5a4118f2a4da60794
                                      'a1.n.y_out_pos0' Einsum b,->b
                                        'a1.n.y_scale_pos0' Einsum b,b->b
                                          'a1.n.z_mean_pos0' Einsum b,bc->c
                                            'a1.inp_pos0' Add
                                              'tok_embeds_pos0'
                                              'w.pos_embeds_pos0'
                                              'a0.out_pos0'
                                              'a0.w.bias_out'
                                              'm0_pos0'
                                              'm0.w.b1'
                                            'a1.n.c.sub_mean' [56,56] Array 325ba9e9e24767e94536f106
                                          'a1.n.w.scale' [56] Array faba3366e8b1301362c9838a
                                        'a1.n.full_mul_pos0' GeneralFunction rsqrt
                                          'a1.n.c.var_p_eps_pos0' Add
                                            'a1.n.var_pos0' Einsum b,b,->
                                              'a1.n.z_mean_pos0'
                                              'a1.n.z_mean_pos0'
                                              'a1.n.c.recip_h_size' [] Scalar 0.017857142857142856
                                            'a1.n.eps' [] Scalar 0.00001
                                  'a1.w.q_bias_pos0' Index [:,0,:]
                                    'a1.w.q_bias' [2,1,28] Array e6a99b1ee0ce98064e2b5810
                                'a1.k_p_bias' Add
                                  'a1.k' Einsum abc,dc->adb
                                    'a1.w.k' [2,28,56] Array 2447e42db2e9ca421c933a86
                                    'a1.n' Add
                                      'a1.n.w.bias'
                                      'a1.n.y_out' Einsum ab,a->ab
                                        'a1.n.y_scale' Einsum ab,b->ab
                                          'a1.n.z_mean' Einsum ab,bc->ac
                                            'a1.inp' Add
                                              'tok_embeds'
                                              'w.pos_embeds'
                                              'a0.out' Einsum abc,bcd->ad
                                                'a0.comb_v' Einsum abc,acd->bad
                                                  'a0.probs' GeneralFunction softmax
                                                    'a0.scores_padded' Add
                                                      'a0.scores_pos_mask' Einsum abc,c->abc
                                                        'a0.scores_not_masked' Einsum abc,adc,->abd
                                                          'a0.q_p_bias' Add
                                                            'a0.q' Einsum abc,dc->adb
                                                              'a0.w.q'
                                                              'a0.n'
                                                            'a0.w.q_bias'
                                                          'a0.k_p_bias'
                                                          'a0.c.div_head_size'
                                                        'c.neg_pos_mask'
                                                      3 'ScalarMul' Einsum abc,->abc
                                                        'Unsqueeze' Rearrange a:22 -> () () a:22
                                                          'pos_mask'
                                                        2 'unnamed'
                                                  'a0.v_p_bias'
                                                'a0.w.out'
                                              'a0.w.bias_out'
                                              'm0' Einsum ab,cb->ac
                                                'm0.act' GeneralFunction relu
                                                  'm0.add0' Add
                                                    'm0.before_product0' Einsum ab,cb->ac
                                                      'm0.n' Add
                                                        'm0.n.w.bias'
                                                        'm0.n.y_out' Einsum ab,a->ab
                                                          'm0.n.y_scale' Einsum ab,b->ab
                                                            'm0.n.z_mean' Einsum ab,bc->ac
                                                              'm0.inp' Add
                                                                'tok_embeds'
                                                                'w.pos_embeds'
                                                                'a0.out'
                                                                'a0.w.bias_out'
                                                              'm0.n.c.sub_mean'
                                                            'm0.n.w.scale'
                                                          'm0.n.full_mul' GeneralFunction rsqrt
                                                            'm0.n.c.var_p_eps' Add
                                                              'm0.n.var' Einsum ab,ab,->a
                                                                'm0.n.z_mean'
                                                                'm0.n.z_mean'
                                                                'm0.n.c.recip_h_size'
                                                              'm0.n.eps'
                                                      'm0.w.w0'
                                                    'm0.w.b0'
                                                'm0.w.w1'
                                              'm0.w.b1'
                                            'a1.n.c.sub_mean'
                                          'a1.n.w.scale'
                                        'a1.n.full_mul' GeneralFunction rsqrt
                                          'a1.n.c.var_p_eps' Add
                                            'a1.n.var' Einsum ab,ab,->a
                                              'a1.n.z_mean'
                                              'a1.n.z_mean'
                                              'a1.n.c.recip_h_size'
                                            'a1.n.eps'
                                  'a1.w.k_bias' [2,1,28] Array a49b17b7ef53c430d7729666
                                'a1.c.div_head_size' [] Scalar 0.1889822365046136
                              'c.neg_pos_mask'
                            'ScalarMul_pos0'
                        'a1.v_p_bias' Add
                          'a1.v' Einsum abc,dc->adb
                            'a1.w.v' [2,28,56] Array 2b799f4714d3e4946f6ad692
                            'a1.n'
                          'a1.w.v_bias' [2,1,28] Array dd5f6dc596ebd588c223c68e
                      'a1.w.out' [2,28,56] Array bb36cea278ac830b8c38bf56
                'a1.out_pos0_head_1_sum' Einsum ab->b
                  'a1.out_pos0_head_1' Index [1:2,:]
                    'a1.out_pos0_keep_head'
              'a1.w.bias_out' [56] Array 6ff98054537a894d4f864eaf
              'm1_pos0' Einsum b,cb->c
                'm1.act_pos0' GeneralFunction relu
                  'm1.add0_pos0' Add
                    'm1.before_product0_pos0' Einsum b,cb->c
                      'm1.n_pos0' Add
                        'm1.n.w.bias' [56] Array e1427cbf98e903020419c988
                        'm1.n.y_out_pos0' Einsum b,->b
                          'm1.n.y_scale_pos0' Einsum b,b->b
                            'm1.n.z_mean_pos0' Einsum b,bc->c
                              'm1.inp_pos0' Add
                                'tok_embeds_pos0'
                                'w.pos_embeds_pos0'
                                'a0.out_pos0'
                                'a0.w.bias_out'
                                'm0_pos0'
                                'm0.w.b1'
                                'a1.out_pos0'
                                'a1.w.bias_out'
                              'm1.n.c.sub_mean' [56,56] Array 325ba9e9e24767e94536f106
                            'm1.n.w.scale' [56] Array a6d0f045744c39c4cc7e433e
                          'm1.n.full_mul_pos0' GeneralFunction rsqrt
                            'm1.n.c.var_p_eps_pos0' Add
                              'm1.n.var_pos0' Einsum b,b,->
                                'm1.n.z_mean_pos0'
                                'm1.n.z_mean_pos0'
                                'm1.n.c.recip_h_size' [] Scalar 0.017857142857142856
                              'm1.n.eps' [] Scalar 0.00001
                      'm1.w.w0' [56,56] Array 03029605fcf142ed26b4d7c1
                    'm1.w.b0' [56] Array bc840ad268d5ce20d2945c42
                'm1.w.w1' [56,56] Array 8fd43e4886842ad370b0d014
              'm1.w.b1' [56] Array 437b46dac5b85ba3a25ad5d2
              'a2.out_pos0' Add
                'a2.out_pos0_head_0_sum' Einsum ab->b
                  'a2.out_pos0_head_0' Index [0:1,:]
                    'a2.out_pos0_keep_head' Einsum bc,bcd->bd
                      'a2.comb_v_pos0' Einsum ac,acd->ad
                        'a2.probs_pos0' GeneralFunction softmax
                          'a2.scores_padded_pos0' Add
                            'a2.scores_pos_mask_pos0' Einsum ac,c->ac
                              'a2.scores_not_masked_pos0' Einsum ac,adc,->ad
                                'a2.q_p_bias_pos0' Add
                                  'a2.q_pos0' Einsum abc,c->ab
                                    'a2.w.q' [2,28,56] Array de2488f8a3e053ace1d73be6
                                    'a2.n_pos0' Add
                                      'a2.n.w.bias' [56] Array 8b88f4fbcf0be7097423b7c6
                                      'a2.n.y_out_pos0' Einsum b,->b
                                        'a2.n.y_scale_pos0' Einsum b,b->b
                                          'a2.n.z_mean_pos0' Einsum b,bc->c
                                            'a2.inp_pos0' Add
                                              'tok_embeds_pos0'
                                              'w.pos_embeds_pos0'
                                              'a0.out_pos0'
                                              'a0.w.bias_out'
                                              'm0_pos0'
                                              'm0.w.b1'
                                              'a1.out_pos0'
                                              'a1.w.bias_out'
                                              'm1_pos0'
                                              'm1.w.b1'
                                            'a2.n.c.sub_mean' [56,56] Array 325ba9e9e24767e94536f106
                                          'a2.n.w.scale' [56] Array 0db2d344a8397242c3d84925
                                        'a2.n.full_mul_pos0' GeneralFunction rsqrt
                                          'a2.n.c.var_p_eps_pos0' Add
                                            'a2.n.var_pos0' Einsum b,b,->
                                              'a2.n.z_mean_pos0'
                                              'a2.n.z_mean_pos0'
                                              'a2.n.c.recip_h_size' [] Scalar 0.017857142857142856
                                            'a2.n.eps' [] Scalar 0.00001
                                  'a2.w.q_bias_pos0' Index [:,0,:]
                                    'a2.w.q_bias' [2,1,28] Array a7c83d68605a729408f7680e
                                'a2.k_p_bias' Add
                                  'a2.k' Einsum abc,dc->adb
                                    'a2.w.k' [2,28,56] Array 5dbfea26b320218793d7bf39
                                    'a2.n' Add
                                      'a2.n.w.bias'
                                      'a2.n.y_out' Einsum ab,a->ab
                                        'a2.n.y_scale' Einsum ab,b->ab
                                          'a2.n.z_mean' Einsum ab,bc->ac
                                            'a2.inp' Add
                                              'tok_embeds'
                                              'w.pos_embeds'
                                              'a0.out'
                                              'a0.w.bias_out'
                                              'm0'
                                              'm0.w.b1'
                                              'a1.out' Einsum abc,bcd->ad
                                                'a1.comb_v' Einsum abc,acd->bad
                                                  'a1.probs' GeneralFunction softmax
                                                    'a1.scores_padded' Add
                                                      'a1.scores_pos_mask' Einsum abc,c->abc
                                                        'a1.scores_not_masked' Einsum abc,adc,->abd
                                                          'a1.q_p_bias' Add
                                                            'a1.q' Einsum abc,dc->adb
                                                              'a1.w.q'
                                                              'a1.n'
                                                            'a1.w.q_bias'
                                                          'a1.k_p_bias'
                                                          'a1.c.div_head_size'
                                                        'c.neg_pos_mask'
                                                      3 'ScalarMul'
                                                  'a1.v_p_bias'
                                                'a1.w.out'
                                              'a1.w.bias_out'
                                              'm1' Einsum ab,cb->ac
                                                'm1.act' GeneralFunction relu
                                                  'm1.add0' Add
                                                    'm1.before_product0' Einsum ab,cb->ac
                                                      'm1.n' Add
                                                        'm1.n.w.bias'
                                                        'm1.n.y_out' Einsum ab,a->ab
                                                          'm1.n.y_scale' Einsum ab,b->ab
                                                            'm1.n.z_mean' Einsum ab,bc->ac
                                                              'm1.inp' Add
                                                                'tok_embeds'
                                                                'w.pos_embeds'
                                                                'a0.out'
                                                                'a0.w.bias_out'
                                                                'm0'
                                                                'm0.w.b1'
                                                                'a1.out'
                                                                'a1.w.bias_out'
                                                              'm1.n.c.sub_mean'
                                                            'm1.n.w.scale'
                                                          'm1.n.full_mul' GeneralFunction rsqrt
                                                            'm1.n.c.var_p_eps' Add
                                                              'm1.n.var' Einsum ab,ab,->a
                                                                'm1.n.z_mean'
                                                                'm1.n.z_mean'
                                                                'm1.n.c.recip_h_size'
                                                              'm1.n.eps'
                                                      'm1.w.w0'
                                                    'm1.w.b0'
                                                'm1.w.w1'
                                              'm1.w.b1'
                                            'a2.n.c.sub_mean'
                                          'a2.n.w.scale'
                                        'a2.n.full_mul' GeneralFunction rsqrt
                                          'a2.n.c.var_p_eps' Add
                                            'a2.n.var' Einsum ab,ab,->a
                                              'a2.n.z_mean'
                                              'a2.n.z_mean'
                                              'a2.n.c.recip_h_size'
                                            'a2.n.eps'
                                  'a2.w.k_bias' [2,1,28] Array 4e50fd8efe268ed0ee7d4dc5
                                'a2.c.div_head_size' [] Scalar 0.1889822365046136
                              'c.neg_pos_mask'
                            'ScalarMul_pos0'
                        'a2.v_p_bias' Add
                          'a2.v' Einsum abc,dc->adb
                            'a2.w.v' [2,28,56] Array d722b26fc7e19bfececef65a
                            'a2.n'
                          'a2.w.v_bias' [2,1,28] Array 4e19a39f1496d3e67b64cad6
                      'a2.w.out' [2,28,56] Array 682c186f4fe413780c0b1e5e
                'a2.out_pos0_head_1_sum' Einsum ab->b
                  'a2.out_pos0_head_1' Index [1:2,:]
                    'a2.out_pos0_keep_head'
              'a2.w.bias_out' [56] Array f12556b8e170ce3ddde4d0d6
              'm2_pos0' Einsum b,cb->c
                'm2.act_pos0' GeneralFunction relu
                  'm2.add0_pos0' Add
                    'm2.before_product0_pos0' Einsum b,cb->c
                      'm2.n_pos0' Add
                        'm2.n.w.bias' [56] Array e2712658f0e7604ad69ae281
                        'm2.n.y_out_pos0' Einsum b,->b
                          'm2.n.y_scale_pos0' Einsum b,b->b
                            'm2.n.z_mean_pos0' Einsum b,bc->c
                              'm2.inp_pos0' Add
                                'tok_embeds_pos0'
                                'w.pos_embeds_pos0'
                                'a0.out_pos0'
                                'a0.w.bias_out'
                                'm0_pos0'
                                'm0.w.b1'
                                'a1.out_pos0'
                                'a1.w.bias_out'
                                'm1_pos0'
                                'm1.w.b1'
                                'a2.out_pos0'
                                'a2.w.bias_out'
                              'm2.n.c.sub_mean' [56,56] Array 325ba9e9e24767e94536f106
                            'm2.n.w.scale' [56] Array a4318f7b7d6eca0873e4b1c4
                          'm2.n.full_mul_pos0' GeneralFunction rsqrt
                            'm2.n.c.var_p_eps_pos0' Add
                              'm2.n.var_pos0' Einsum b,b,->
                                'm2.n.z_mean_pos0'
                                'm2.n.z_mean_pos0'
                                'm2.n.c.recip_h_size' [] Scalar 0.017857142857142856
                              'm2.n.eps' [] Scalar 0.00001
                      'm2.w.w0' [56,56] Array 378702f5015024bc11c99923
                    'm2.w.b0' [56] Array e5a37c6d6b6b95eb3363aaf9
                'm2.w.w1' [56,56] Array 6ff87833d69056a4f145c5e3
              'm2.w.b1' [56] Array 85abdbba0615b7f2574319e3
            'final.n.c.sub_mean' [56,56] Array 325ba9e9e24767e94536f106
          'final.n.w.scale' [56] Array 49283af11ca865adb6a8bd12
        'final.n.full_mul_pos0' GeneralFunction rsqrt
          'final.n.c.var_p_eps_pos0' Add
            'final.n.var_pos0' Einsum b,b,->
              'final.n.z_mean_pos0'
              'final.n.z_mean_pos0'
              'final.n.c.recip_h_size' [] Scalar 0.017857142857142856
            'final.n.eps' [] Scalar 0.00001
    'w.unembed_bal' Index [1,:]
      'w.unembed' [2,56] Array bdff3a8cee2d823672ce6624
  'neg logit_unbalanced' Einsum ,->
    'logit_unbalanced' Einsum b,b->
      'final.n_pos0'
      'w.unembed_unbal' Index [0,:]
        'w.unembed'
    'neg_1' [] Scalar -1
        """
            ),
        ),
    ]
    print()
    for n, o in circuit_pairs:
        with timed("diffing"):
            print(rc.diff_circuits(n, o, rc.PrintOptions(bijection=False), same_color="Cyan"))


def test_rust_parsing_id_by_name():
    strings = [
        """
        0 'hi' Add
          1 ' my' [3, 1] Scatter [0:2,0:1]
            2 '# 43' [2, 1] Scalar 1
          3 '0' [1, 3, 6] Scalar 7
          4 '1' [5, 3, 6] Scatter [0:5,0:3,0:3]
            5 'name stuff  ' Concat 0
              6 '###$' [2, 3, 3] Scalar 1.2
              7 [3, 3, 3] Scalar 1.2
          8 'ein' Einsum abc->abc
            4
          3
          9 'eins' Einsum ab->
            2
          10 'einy' Einsum abc->
            6
        """,
        """
        'hi' Add
          ' my' [3, 1] Scatter [0:2,0:1]
            '# 43' [2, 1] Scalar 1
          '0' [1, 3, 6] Scalar 7
          '1' [5, 3, 6] Scatter [0:5,0:3,0:3]
            'name stuff  ' Concat 0
              '###$' [2, 3, 3] Scalar 1.2
              7 [3, 3, 3] Scalar 1.2
          'ein' Einsum abc->abc
            '1'
          '0'
          'eins' Einsum ab->
            '# 43'
          'einy' Einsum abc->
            '###$'
        """,
        """
        0 'hi' Add
          1 ' my' [3, 1] Scatter [0:2,0:1]
            '    totally different' '# 43' [2, 1] Scalar 1
          3 '0' [1, 3, 6] Scalar 7
          4 '1' [5, 3, 6] Scatter [0:5,0:3,0:3]
            5 'name stuff  ' Concat 0
              '###$' [2, 3, 3] Scalar 1.2
              7 [3, 3, 3] Scalar 1.2
          8 'ein' Einsum abc->abc
            4
          3
          9 'eins' Einsum ab->
            '    totally different'
          10 'einy' Einsum abc->
            '###$'
        """,
    ]

    circs = [rc.Parser()(s) for s in strings]
    assert len(set(circs)) == 1, circs


def test_print_seen_same_line():
    circ = P(
        """
0 Add
  1 [2] Scalar 1
  2 Add
    1
    1
    3 [2] Scalar 2
    4 Add
      1
      3
    """
    )

    rc.PrintOptions(seen_children_same_line=True, bijection=False).print(circ)
    rc.PrintOptions(seen_children_same_line=True, bijection=False, arrows=True).print(circ)


def test_print_child_below():
    circ = P(
        """
0 Add
  1 Add
    3 [2] Scalar 1
  2 Add
    3
    4 [2] Scalar 1
  """
    )
    rc.PrintOptions(only_child_below=True, bijection=False).print(circ)


def test_print_traversal_by_child():
    # shamelessly taken from nest.py

    # first, we'll create a flat einsum we want to rearrange
    a, b, c, d, f, e = 2, 3, 4, 5, 6, 7
    x = rc.Einsum.from_einsum_string(
        "a b c, b c d, a e, b d, f c, a c d, e a f -> a c d f",
        *(
            rc.Array.randn(*shape, name=name)
            for (shape, name) in [
                ((a, b, c), "abc"),
                ((b, c, d), "bcd"),
                ((a, e), "ae"),
                ((b, d), "bd"),
                ((f, c), "ec"),
                ((a, c, d), "acd"),
                ((e, a, f), "eaf"),
            ]
        ),
    ).normalize_ints()
    z_new = rc.nest_einsums(x, (("acd", ("bcd", rc.Matcher.regex("^bd$"))), ("ae", "ec"), rc.NestRest()))

    fancy_sub_matcher = rc.IterativeMatcher.new_func(
        lambda _: rc.IterateMatchResults(
            [
                rc.new_traversal(term_early_at=rc.Matcher.regex(r"^\w+ \* \w+$")),
                rc.FINISHED,
                True,
            ],
            True,
        )
    )
    assert (
        rc.PrintOptions(bijection=False, traversal=fancy_sub_matcher).repr(z_new)
        == """
abc * bcd * ae * bd * ec * acd * eaf Einsum acef,abcd,abcde->acfb
  acd * bcd * bd Einsum acf,cef->acef
    acd [2,4,5] Array
    bcd * bd Einsum ecf,ef->cef ...
  ae * ec Einsum ad,bc->abcd ...
  abc * eaf Einsum aec,dab->abcde
    abc [2,3,4] Array
    eaf [7,2,6] Array""".strip(
            "\n"
        )
    )


mlp_2_layer_str = """
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
  'b.bilinear_mlp' Module ! 'bilinear_mlp.input'
    'bilinear_mlp'
    'b.input' [3,5,7,17] Scalar 7 ! 'bilinear_mlp.input'
    'b.proj_in' [22,17] Scalar 8 ! 'bilinear_mlp.w.proj_in' ftt
    'b.proj_out' [13,11] Scalar 9 ! 'bilinear_mlp.w.proj_out' ftt
  'b_new.proj_in' [22,13] Scalar 10 ! 'bilinear_mlp.w.proj_in' ftt
  'b_new.proj_out' [20,11] Scalar 11 ! 'bilinear_mlp.w.proj_out' ftt
"""
mlp_2_layer = rc.Parser(tensors_as_random=True)(mlp_2_layer_str)


def test_rust_module_printing():
    r = mlp_2_layer.repr()
    assert rc.Parser()(r) == mlp_2_layer

    def update_mod(x: rc.Circuit):
        mod = x.cast_module()
        snd_spec = mod.spec.arg_specs[1]
        new_spec = rc.ModuleSpec(
            mod.spec.circuit,
            [
                mod.spec.arg_specs[0],
                rc.ModuleArgSpec(snd_spec.symbol, batchable=False, expandable=True, ban_non_symbolic_size_expand=False),
                *mod.spec.arg_specs[2:],
            ],
        )
        return rc.Module.new_flat(new_spec, *mod.nodes, name=mod.name)

    new_mlp_2_layer = rc.IterativeMatcher("b.bilinear_mlp").update(mlp_2_layer, update_mod)
    assert rc.Parser()(new_mlp_2_layer.repr()) == new_mlp_2_layer
    new_mlp_2_layer.print()


def test_print_comments():
    rc.PrintOptions(comment_arg_names=True).print(mlp_2_layer)
    rc.PrintOptions(comment_arg_names=True, arrows=True).print(mlp_2_layer)
    assert (
        rc.PrintOptions(comment_arg_names=True, arrows=True).repr(mlp_2_layer)
        == """
'outer_b.bilinear_mlp' Module
├‣'bilinear_mlp' Einsum h,h,oh->o # Spec
│ ├‣'bilinear_mlp.pre0' [1s] Index [0,:] # h
│ │ └‣'bilinear_mlp.fold_pre' Rearrange (a:2 b) -> a:2 b
│ │   └‣'bilinear_mlp.pre' Einsum i,hi->h
│ │     ├‣'bilinear_mlp.input' [0s] Symbol 873a937e-2bb9-4f7f-b55e-a100db3dde52 # i
│ │     └‣'bilinear_mlp.w.proj_in' [2*1s,0s] Symbol c171d519-8793-4a8b-ac5e-d550347f30a6 # hi
│ ├‣'bilinear_mlp.pre1' [1s] Index [1,:] # h
│ │ └‣'bilinear_mlp.fold_pre'
│ └‣'bilinear_mlp.w.proj_out' [2s,1s] Symbol e61637eb-9f17-4325-b2c2-5eb2518026cf # oh
├‣'b.bilinear_mlp' Module ! 'bilinear_mlp.input'
│ ├‣'bilinear_mlp' # Spec
│ ├‣'b.input' [3,5,7,17] Scalar 7 ! 'bilinear_mlp.input'
│ ├‣'b.proj_in' [22,17] Scalar 8 ! 'bilinear_mlp.w.proj_in' ftt
│ └‣'b.proj_out' [13,11] Scalar 9 ! 'bilinear_mlp.w.proj_out' ftt
├‣'b_new.proj_in' [22,13] Scalar 10 ! 'bilinear_mlp.w.proj_in' ftt
└‣'b_new.proj_out' [20,11] Scalar 11 ! 'bilinear_mlp.w.proj_out' ftt
""".strip(
            "\n"
        )
    )
    rc.PrintOptions(comment_arg_names=True).print(
        rc.Parser()(
            """
0 'hoo' StoredCumulantVar 1,2,4|18af4b4c-e297-4d86-a51f-0e523d809289
  1 'hii' [] Scalar 1
  1
  1
  """
        )
    )


def test_commenters():
    commenter = lambda x: f"line length: {(len(rc.PrintOptions().repr_line_info(x)))}"
    commenter2 = lambda x: f"node count: {rc.count_nodes(x)}"
    print()
    rc.PrintOptions(commenters=[commenter, commenter2]).print(mlp_2_layer)
    assert (
        rc.PrintOptions(commenters=[commenter, commenter2]).repr(mlp_2_layer)
        == """
'outer_b.bilinear_mlp' Module # line length: 7 # node count: 15
  'bilinear_mlp' Einsum h,h,oh->o # line length: 17 # node count: 8
    'bilinear_mlp.pre0' [1s] Index [0,:] # line length: 17 # node count: 5
      'bilinear_mlp.fold_pre' Rearrange (a:2 b) -> a:2 b # line length: 27 # node count: 4
        'bilinear_mlp.pre' Einsum i,hi->h # line length: 15 # node count: 3
          'bilinear_mlp.input' [0s] Symbol 873a937e-2bb9-4f7f-b55e-a100db3dde52 # line length: 49 # node count: 1
          'bilinear_mlp.w.proj_in' [2*1s,0s] Symbol c171d519-8793-4a8b-ac5e-d550347f30a6 # line length: 54 # node count: 1
    'bilinear_mlp.pre1' [1s] Index [1,:] # line length: 17 # node count: 5
      'bilinear_mlp.fold_pre' # line length: 27 # node count: 4
    'bilinear_mlp.w.proj_out' [2s,1s] Symbol e61637eb-9f17-4325-b2c2-5eb2518026cf # line length: 52 # node count: 1
  'b.bilinear_mlp' Module ! 'bilinear_mlp.input' # line length: 7 # node count: 12
    'bilinear_mlp' # line length: 17 # node count: 8
    'b.input' [3,5,7,17] Scalar 7 ! 'bilinear_mlp.input' # line length: 20 # node count: 1
    'b.proj_in' [22,17] Scalar 8 ! 'bilinear_mlp.w.proj_in' ftt # line length: 17 # node count: 1
    'b.proj_out' [13,11] Scalar 9 ! 'bilinear_mlp.w.proj_out' ftt # line length: 17 # node count: 1
  'b_new.proj_in' [22,13] Scalar 10 ! 'bilinear_mlp.w.proj_in' ftt # line length: 18 # node count: 1
  'b_new.proj_out' [20,11] Scalar 11 ! 'bilinear_mlp.w.proj_out' ftt # line length: 18 # node count: 1
""".strip(
            "\n"
        )
    )

    parsed_comment = rc.Parser()(rc.PrintOptions(commenters=[commenter, commenter2]).repr(mlp_2_layer))
    assert parsed_comment == mlp_2_layer

    # term early case
    print()
    opt = rc.PrintOptions(commenters=[commenter, commenter2], traversal=rc.new_traversal(end_depth=3), bijection=False)
    opt.print(mlp_2_layer)
    assert (
        opt.repr(mlp_2_layer)
        == """
outer_b.bilinear_mlp Module # line length: 7 # node count: 15
  bilinear_mlp Einsum h,h,oh->o # line length: 17 # node count: 8
    bilinear_mlp.pre0 [1s] Index [0,:] ... # line length: 17 # node count: 5
    bilinear_mlp.pre1 [1s] Index [1,:] ... # line length: 17 # node count: 5
    bilinear_mlp.w.proj_out [2s,1s] Symbol e61637eb-9f17-4325-b2c2-5eb2518026cf # line length: 52 # node count: 1
  b.bilinear_mlp Module ! bilinear_mlp.input [0s] Symbol 873a937e-2bb9-4f7f-b55e-a100db3dde52 # line length: 7 # node count: 12
    bilinear_mlp # line length: 17 # node count: 8
    b.input [3,5,7,17] Scalar 7 ! bilinear_mlp.input # line length: 20 # node count: 1
    b.proj_in [22,17] Scalar 8 ! bilinear_mlp.w.proj_in [2*1s,0s] Symbol c171d519-8793-4a8b-ac5e-d550347f30a6 ftt # line length: 17 # node count: 1
    b.proj_out [13,11] Scalar 9 ! bilinear_mlp.w.proj_out ftt # line length: 17 # node count: 1
  b_new.proj_in [22,13] Scalar 10 ! bilinear_mlp.w.proj_in ftt # line length: 18 # node count: 1
  b_new.proj_out [20,11] Scalar 11 ! bilinear_mlp.w.proj_out ftt # line length: 18 # node count: 1
""".strip(
            "\n"
        )
    )


def test_parse_fancy():
    fancy_einsum = """
0 Einsum fancy: first second, first third -> first
  1 [3, 1] Scalar 1.0
  2 [3, 1] Scalar 1.5"""
    einsum = """
0 Einsum ab, ac -> a
  1 [3, 1] Scalar 1.0
  2 [3, 1] Scalar 1.5"""
    assert rc.Parser()(fancy_einsum) == rc.Parser()(einsum)
    fancy_einsum = """
0 Einsum fancy:first second -> second
  1 [3, 1] Scalar 1.0"""
    einsum = """
0 Einsum ab -> b
  1 [3, 1] Scalar 1.0"""
    assert rc.Parser()(fancy_einsum) == rc.Parser()(einsum)


def test_number_leaves():
    c = rc.Scalar(1.0)
    many = rc.Add(rc.Add(c, c), c, rc.Add(c, c))
    opt = rc.PrintOptions(number_leaves=True, bijection=False)
    many.print(opt)
    print_multiline_escape(many.repr(opt))
    expected = """
0 Add
  1 Add
    2 [] Scalar 1 # \x1b[35m0\x1b[0m
    2 [] Scalar 1 # \x1b[35m1\x1b[0m
  2 [] Scalar 1 # \x1b[35m2\x1b[0m
  1 Add
    2 [] Scalar 1 # \x1b[35m3\x1b[0m
    2 [] Scalar 1 # \x1b[35m4\x1b[0m""".strip(
        "\n"
    )

    assert many.repr(opt) == expected
    rc.nest_adds(many, (((3, 2), 0), (1, 4))).print()  # nesting works with these numbers

    trav = rc.new_traversal(end_depth=2)
    opt = opt.evolve(traversal=trav)
    many.print(opt)
    print_multiline_escape(many.repr(opt))
    expected = """
0 Add
  1 Add ... # \x1b[35m0\x1b[0m
  2 [] Scalar 1 # \x1b[35m1\x1b[0m
  1 Add ... # \x1b[35m2\x1b[0m""".strip(
        "\n"
    )
    assert many.repr(opt) == expected
    rc.nest_adds(many, (1, 0, 2), traversal=trav).print()  # nesting works with these numbers


def test_parse_named_axes():
    printer = rc.PrintOptions(show_all_named_axes=True)
    einsum = """0 [large:3] Einsum ab,ac->a
  1 [large:3,small:1] Scalar 1
  2 [large:3,small:1] Scalar 1.5"""
    assert printer.repr(rc.Parser()(einsum)) == einsum
    einsum = """0 [small:1,large:3] Einsum ab,cb->bc
  1 [large:3,small:1] Scalar 1
  2 [large:3,small:1] Scalar 1.5"""
    assert printer.repr(rc.Parser()(einsum)) == einsum
    einsum = """0 [small:1,large:3] Einsum ab,cb->bc
  1 [3,1] Scalar 1
  2 [large:3,small:1] Scalar 1.5"""
    assert printer.repr(rc.Parser()(einsum)) == einsum
    einsum = """0 [small:1,3] Einsum ab,cb->bc
  1 [3,1] Scalar 1
  2 [3,small:1] Scalar 1.5"""
    assert printer.repr(rc.Parser()(einsum)) == einsum
    einsum = """0 [small:1,large:3] Einsum fancy: largerrr small, large small -> small large
  1 [largerrr:3,small:1] Scalar 1
  2 [large:3,small:1] Scalar 1.5"""
    assert printer.repr(rc.Parser()(einsum)) == einsum
    einsum = """0 [small:1,large:3] Einsum fancy: largerrr small, large small -> small large
  1 [largerrr:3,small:1] Scalar 1
  2 [large:3,smaller:1] Scalar 1.5"""
    assert printer.repr(rc.Parser()(einsum)) == einsum


def test_different_color_for_common_nodes():
    f = rc.PrintHtmlOptions.type_colorer()

    colors = [
        f(rc.Add()),
        f(rc.Concat(rc.Scalar(1.0, (5,)), axis=0)),
        f(rc.Einsum.scalar_mul(rc.Add(), 2)),
        f(rc.Array(torch.zeros(5))),
        f(rc.sigmoid(rc.Array(torch.zeros(5)))),
        f(rc.Rearrange.from_string(rc.Array(torch.zeros(5)), "a -> a")),
    ]
    assert len(colors) == len(set(colors))


def test_print_arrows_and_only_child_below():  # regression test for #1699
    printer = rc.PrintOptions(arrows=True, only_child_below=True)

    circ = """0 Add\n  1 [1] Scalar 0.7"""
    printer.print(rc.Parser()(circ))
    assert printer.repr(rc.Parser()(circ)) == """0 Add\n▼\n1 [1] Scalar 0.7"""

    circ = """0 Add
  1 Add
    2 [1] Scalar 0.7
  3 Add
    4 [1] Scalar 0.5
    5 [1] Scalar 6.3"""
    printer.print(rc.Parser()(circ))
    assert (
        printer.repr(rc.Parser()(circ))
        == """0 Add
├‣1 Add
│ ▼
│ 2 [1] Scalar 0.7
└‣3 Add
  ├‣4 [1] Scalar 0.5
  └‣5 [1] Scalar 6.3"""
    )

    circ = """0 Add
  1 Add
    2 [1] Scalar 0.5
    3 [1] Scalar 6.3
  4 Add
    5 [1] Scalar 0.7"""
    printer.print(rc.Parser()(circ))
    assert (
        printer.repr(rc.Parser()(circ))
        == """0 Add
├‣1 Add
│ ├‣2 [1] Scalar 0.5
│ └‣3 [1] Scalar 6.3
└‣4 Add
  ▼
  5 [1] Scalar 0.7"""
    )


def test_parse_cycle_error():
    with pytest.raises(rc.ParseCircuitCycleError):
        P(
            """
1 Add
  2 Add
    1
        """
        )
