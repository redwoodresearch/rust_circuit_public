from rust_circuit import *

P = Parser()


# correctness tests are in rea
def test_add_removable():
    circ = P(
        """
0 Add
  1 [4,1,2] Scalar 1
  2 'hi' [1,1,2] Symbol"""
    ).cast_add()
    print(add_elim_removable_axes_weak(circ))


def test_einsum_removable():
    circ = P(
        """
0 Einsum azc,yzc->azc
  1 [4,1,2] Scalar 1
  2 'hi' [1,1,2] Symbol"""
    ).cast_einsum()
    print(einsum_elim_removable_axes_weak(circ))


def test_einsum_permute_rearrange():
    circ = P(
        """
0 Einsum abc->bca
  2 'hi' [4,3,2] Symbol"""
    ).cast_einsum()
    print(einsum_permute_to_rearrange(circ))


def test_basic_simp():
    circ = P(
        """
0 Einsum azc,yzc->azc
  8 Rearrange a b c -> c b a
    1 [3,1,4] Scalar 1
  2 [1,1,3] Rearrange a b c -> a b c
    3 Concat 2
      4 'a' [1,1,1] Symbol
      5 'a' [1,1,1] Symbol
      10 Einsum abc->bca
        6 'b' [1,1,1] Symbol
    """
    )
    print(simp(circ))


if __name__ == "__main__":
    test_add_removable()
    test_einsum_removable()
    test_basic_simp()
    test_einsum_permute_rearrange()
