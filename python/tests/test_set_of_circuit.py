import rust_circuit as rc


def test_set_of_identities():
    setty = rc.SetOfCircuitIdentities()
    circ = rc.Array.randn(10)
    setty.insert(circ)
    print(len(setty))
    print(circ in setty)
    print(rc.Array.randn(11) in setty)
    setty.insert(circ)
    print(len(setty))
