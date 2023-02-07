import rust_circuit as rc

P = rc.Parser()


def test_circuit_flags():
    s = """
    'a' [3] Symbol
    'b' Einsum ij,j->j
      'c' [7, 3] Scalar 2.
      'a'
    'b_arr' Einsum ij,j->j
      'c_arr' [7, 3] Array rand
      'a_arr' [3] Array rand
    'b_var' DiscreteVar
      'vals' [7, 3] Array rand
      'probs' GeneralFunction softmax
        'x' [7] Array rand
    'b_var_sym' DiscreteVar
      'vals_sym' [7, 3] Symbol
      'probs'
    'cum_var' Cumulant
      'b_var'
    'cum_var_sym' Cumulant
      'b_var_sym'
    'cum_sym' Cumulant
      'a'
    'cum_empty' Cumulant
    'cum_const' Cumulant
      'b_arr'
      'c_arr'
    """
    (
        a,
        b,
        b_arr,
        b_var,
        b_var_sym,
        cum_var,
        cum_var_sym,
        cum_sym,
        cum_empty,
        cum_const,
    ) = P.parse_circuits(s)

    def check(
        c: rc.Circuit, is_constant: bool, is_explicitly_computable: bool, can_be_sampled: bool, use_autoname: bool
    ):
        assert c.is_constant == is_constant
        assert c.is_explicitly_computable == is_explicitly_computable
        assert c.can_be_sampled == can_be_sampled
        assert c.use_autoname == use_autoname

        sym = rc.Symbol.new_with_random_uuid((), name="sym")
        mod = rc.module_new_bind(sym, (sym, c))
        assert mod.is_constant == c.is_constant
        assert mod.is_explicitly_computable == c.is_explicitly_computable
        assert mod.can_be_sampled == c.can_be_sampled
        assert mod.no_unbound_symbols == c.no_unbound_symbols
        assert mod.use_autoname == c.use_autoname

    check(a, is_constant=True, is_explicitly_computable=False, can_be_sampled=True, use_autoname=True)
    check(b, is_constant=True, is_explicitly_computable=False, can_be_sampled=True, use_autoname=True)
    check(b_arr, is_constant=True, is_explicitly_computable=True, can_be_sampled=True, use_autoname=True)
    check(b_var, is_constant=False, is_explicitly_computable=False, can_be_sampled=True, use_autoname=True)
    check(b_var_sym, is_constant=False, is_explicitly_computable=False, can_be_sampled=True, use_autoname=True)
    check(cum_var, is_constant=True, is_explicitly_computable=False, can_be_sampled=True, use_autoname=True)
    check(cum_var_sym, is_constant=True, is_explicitly_computable=False, can_be_sampled=True, use_autoname=True)
    check(cum_sym, is_constant=True, is_explicitly_computable=False, can_be_sampled=True, use_autoname=True)
    check(cum_empty, is_constant=True, is_explicitly_computable=True, can_be_sampled=True, use_autoname=True)
    check(cum_const, is_constant=True, is_explicitly_computable=True, can_be_sampled=True, use_autoname=True)

    a_dis = a.with_autoname_disabled()
    check(a_dis, is_constant=True, is_explicitly_computable=False, can_be_sampled=True, use_autoname=False)
    b_arr_dis = b_arr.update("c_arr", lambda x: x.with_autoname_disabled())
    check(b_arr_dis, is_constant=True, is_explicitly_computable=True, can_be_sampled=True, use_autoname=False)

    assert rc.Add(b_arr_dis).op_name is None
    assert rc.Add(b_arr).op_name is not None
