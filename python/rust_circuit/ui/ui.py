import math
import urllib.parse
import webbrowser
from json import dumps
from typing import Any, Callable, Dict, Optional

import rust_circuit as rc
from rust_circuit import Circuit, PrintOptions

ui_default_hidden_matcher = rc.Matcher(rc.SetSymbolicShape, rc.Scalar, rc.Tag)
ui_no_ln_matcher = rc.Matcher("ln", rc.Matcher.regex(r"\.*ln\.*"))


def circuit_to_json(circ: Circuit) -> Any:
    graph = {}
    printer = PrintOptions()

    def visitor(circ: Circuit):
        childIndices: Dict[str, Any] = {}
        extra_info = printer.repr_extra_info(circ)
        for i, child in enumerate(circ.children):
            to_append = {"i": i, "n": ""}
            if circ.is_einsum():
                to_append["n"] = extra_info.split("->")[0].split(",")[i]
            if circ.is_module():
                circ_mod = circ.cast_module()
                if i == 0:
                    to_append["n"] = "spec"
                else:
                    to_append["n"] = circ_mod.spec.arg_specs[i - 1].symbol.name

            if child.hash_base16 in childIndices:
                childIndices[child.hash_base16].append(to_append)
            else:
                childIndices[child.hash_base16] = [to_append]
        graph[circ.hash_base16] = {
            "hash": circ.hash_base16,
            "childIndices": childIndices,
            "meta": {
                "name": circ.op_name,
                "shape": [
                    {"otherFactor": x.other_factor, "symbolicSizes": x.symbolic_sizes} for x in circ.symbolic_shape()
                ],
                "numel": rc.oom_fmt(math.prod(circ.shape)),
                "kind": circ.__class__.__name__,
                "extra": extra_info,
            },
        }

    circ.visit(visitor)
    return graph


def eval_code_on_circuit(circ_string: str, code: str) -> Any:
    """Big security vulnerability if called remotely"""
    circ = rc.Parser(
        tensors_as_random=True, tensors_as_random_device_dtype=rc.TorchDeviceDtypeOp("cpu", "float32")
    ).parse_circuit(circ_string)
    return eval(code, {"circ": circ, "rc": rc}, {})


def circuit_graph_ui(
    circ: Circuit,
    *,
    base_url: str = "http://interp-tools.redwoodresearch.org",
    annotators: dict[str, Callable[[Circuit], str]] = {},
    enabled_annotators: list[str] = [],
    default_hidden: Optional[set[Circuit]] = None,
    default_shown: Optional[set[Circuit]] = None,
):
    if default_hidden is not None and default_shown is not None:
        raise Exception("only one of default_hidden and default_shown can be set")
    if default_hidden is None:
        default_hidden = set()
    if default_shown is not None:

        def fny(c: Circuit):
            assert isinstance(default_shown, set)  # for linter
            assert isinstance(default_hidden, set)  # for linter
            if c not in default_shown:
                default_hidden.add(c)

        circ.visit(fny)
    default_hidden.discard(circ)  # root can't be hidden
    graph = circuit_to_json(circ)
    data = {
        "graph": graph,
        "rootChildHashes": [circ.hash_base16],
        "annotators": list(annotators.keys()),
        "enabledAnnotators": enabled_annotators,
        "default_hidden": [x.hash_base16 for x in default_hidden],
    }
    url = base_url + "/#/layout?data=" + urllib.parse.quote(dumps(data))
    print(url)
    webbrowser.open(url)
