"""Check that all current notebooks and demos aren't broken with the current code.

You should be more willing than usual to remove tests from this file. If a notebook really isn't being used (for
research or documentation) any more, you should delete its test from here.

"""
import pytest

from interp.circuit.testing.notebook import NotebookInTesting


def test_causal_scrubbing_paren_balancer():
    with NotebookInTesting():
        from rust_circuit.demos.paren_balancer.causal_scrubbing_experiments import __name__  # noqa: F401


def test_demo_rust_basic_scope_manager():
    with NotebookInTesting():
        from rust_circuit.demos.rust_circuit_demos.basic_scope_manager import __name__  # noqa: F401


def test_demo_rust_custom_general_function():
    with NotebookInTesting():
        from rust_circuit.demos.rust_circuit_demos.custom_general_function import __name__  # noqa: F401


def test_demo_rust_fancy_error():
    with NotebookInTesting():
        from rust_circuit.demos.rust_circuit_demos.fancy_error import __name__  # noqa: F401


@pytest.mark.skip(reason="broken demo, but the general NB testing should be on main")
def test_demo_rust_handcrafted_model_cumulants():
    with NotebookInTesting():
        from rust_circuit.demos.rust_circuit_demos.handcrafted_model_cumulants import __name__  # noqa: F401


def test_demo_rust_iterative_matchers():
    with NotebookInTesting():
        from rust_circuit.demos.rust_circuit_demos.iterative_matchers import __name__  # noqa: F401


def test_demo_rust_modules_and_symbols():
    with NotebookInTesting():
        from rust_circuit.demos.rust_circuit_demos.modules_and_symbols import __name__  # noqa: F401


def test_demo_rust_nest():
    with NotebookInTesting():
        from rust_circuit.demos.rust_circuit_demos.nest import __name__  # noqa: F401


def test_demo_rust_printing_parsing():
    with NotebookInTesting():
        from rust_circuit.demos.rust_circuit_demos.printing_parsing import __name__  # noqa: F401


def test_demo_rust_push_down_index():
    with NotebookInTesting():
        from rust_circuit.demos.rust_circuit_demos.push_down_index import __name__  # noqa: F401


def test_demo_rust_simp():
    with NotebookInTesting():
        from rust_circuit.demos.rust_circuit_demos.simp import __name__  # noqa: F401


def test_demo_rust_save_models():
    with NotebookInTesting():
        from rust_circuit.demos.rust_circuit_demos.save_models import __name__  # noqa: F401


def test_demo_push_down_module():
    with NotebookInTesting():
        from rust_circuit.demos.rust_circuit_demos.push_down_module import __name__  # noqa: F401


def test_demo_causal_scrubbing_simple():
    with NotebookInTesting():
        from rust_circuit.demos.causal_scrubbing.causal_scrubbing_simple import __name__  # noqa: F401


def test_demo_causal_scrubbing_pool():
    with NotebookInTesting():
        from rust_circuit.demos.causal_scrubbing.pool import __name__  # noqa: F401


def test_demo_transformer_config():
    with NotebookInTesting():
        from rust_circuit.demos.rust_circuit_demos.transformer_config import __name__  # noqa: F401
