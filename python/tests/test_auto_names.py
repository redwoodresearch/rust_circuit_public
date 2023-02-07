import rust_circuit as rc
import rust_circuit.optional as op
from rust_circuit import (
    Add,
    Array,
    Circuit,
    Concat,
    Cumulant,
    Einsum,
    Index,
    Module,
    Parser,
    Scalar,
    StoredCumulantVar,
    make_broadcast,
    sigmoid,
)
from rust_circuit.module_library import get_pointwise_mlp


def check_auto_names(do_print: bool = False):
    s = Scalar(1.0, (5, 5), "square_mat")
    r = Scalar(1.0, (1, 5), "rect_mat")

    circuits: list[Circuit] = [
        sigmoid(s).add(r),
        sigmoid(s.add(r)),
        Einsum.scalar_mul(Einsum.scalar_mul(r, 2, scalar_name="2"), 3, scalar_name="3"),
        Add(s, r),
        Add(s, op.unwrap(make_broadcast(r, s.shape))),
        op.unwrap(make_broadcast(Index(Add(s, r), [0]), (1, 5))),
        Concat(s, r, axis=0),
        Einsum.scalar_mul(Concat(s, r, axis=0), 2, scalar_name="2"),
        Concat(s, Einsum.scalar_mul(r, 2, scalar_name="2"), axis=0),
        Einsum.scalar_mul(Add(s, r), 2, scalar_name="2"),
        Add(s, Einsum.scalar_mul(r, 2, scalar_name="2")),
        s.sub(r),
        Cumulant(s, Einsum.scalar_mul(r, 2, scalar_name="2"), s),
        StoredCumulantVar({1: s, 2: Cumulant(s, s), 5: Cumulant(*[s for _ in range(5)])}),
        Module.new_flat(
            get_pointwise_mlp().spec,
            Array.randn(12, name="res_stream").add(Array.randn(12, name="noise")),
            Array.randn(12 * 4, 12, name="mlp2.w.in"),
            Array.randn(12 * 4, name="mlp2.w.bias_in"),
            Array.randn(12, 12 * 4, name="mlp2.w.out"),
        ),
        rc.deep_map(
            Parser().parse_circuits(
                """0 Einsum h,oh->o
  1 GeneralFunction gelu
    2 Einsum i,hi->h
      3 Add
        4 'gelu_mlp.input' [6] Symbol
        5 'noise' [6] Symbol
      6 'gelu_mlp.w.proj_in' [3, 6] Symbol
  7 'gelu_mlp.w.proj_out' [6, 3] Symbol"""
            )[0],
            lambda x: x,
        ),
    ]

    for c in circuits:
        assert c.name
        if do_print:
            print(c.name)


def test_auto_names_are_not_none():
    check_auto_names()


if __name__ == "__main__":
    check_auto_names(do_print=True)
