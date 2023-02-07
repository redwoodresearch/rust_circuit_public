import rust_circuit.optional as op

from . import _rust as rc


def get_matching_nodes_and_args(mod: rc.Module, sym_matcher: rc.MatcherIn):
    sym_matcher = rc.Matcher(sym_matcher)
    nodes_arg_specs = [
        (node, arg_spec) for node, arg_spec in zip(mod.nodes, mod.spec.arg_specs) if sym_matcher(arg_spec.symbol)
    ]
    nodes = [n for n, _ in nodes_arg_specs]
    arg_specs = [a for _, a in nodes_arg_specs]
    return nodes, arg_specs


def filter_module_inputs(mod: rc.Module, sym_matcher: rc.MatcherIn):
    nodes, arg_specs = get_matching_nodes_and_args(mod, sym_matcher)
    return rc.Module.new_flat(
        rc.ModuleSpec(mod.spec.circuit, arg_specs, False, False),
        *nodes,
        name=op.map(mod.op_name, lambda x: x + " drop")
    )


def drop_module_inputs(mod: rc.Module, drop_sym_matcher: rc.MatcherIn):
    """filter_module_inputs(mod, ~rc.Matcher(drop_sym_matcher))"""
    return filter_module_inputs(mod, ~rc.Matcher(drop_sym_matcher))
