use circuit_base::{
    circuit_utils::{replace_nodes, toposort_circuit},
    evaluate, Array, Circuit, CircuitNode, CircuitRc,
};
use num_bigint::BigUint;
use pyo3::prelude::*;
use rr_util::{
    py_types::{Tensor, PY_UTILS},
    util::HashBytes,
};
use rustc_hash::FxHashMap as HashMap;

use crate::circuit_optimizer::{
    optimize_and_evaluate, optimize_circuit, OptimizationContext, OptimizationSettings,
};

/// We need this to redirect Python stdout to Rust stdout
/// If we don't do this, then Python will print to its own stdout, which is not captured by Rust
/// so you only see the output if your entry point is a Python script
#[pyclass]
struct LoggingStdout;

#[pymethods]
impl LoggingStdout {
    fn write(&self, data: &str) {
        println!("{}", data);
    }
}

/// Print Python objects with Python's print function
pub fn python_print(obj: &PyObject) {
    Python::with_gil(move |py| {
        // import print and overwrite stdout with our logging stdout
        let builtins = py.import("builtins").unwrap();
        let sys = py.import("sys").unwrap();
        let prev = sys.getattr("stdout").unwrap();
        sys.setattr("stdout", LoggingStdout.into_py(py)).unwrap();
        let print = builtins.getattr("print").unwrap();
        // print(obj)
        print.call((obj,), None).unwrap();
        // restore stdout
        sys.setattr("stdout", prev).unwrap();
    });
}

pub fn is_close(a: Tensor, b: Tensor) -> Result<PyObject, PyErr> {
    Python::with_gil(|py| {
        PY_UTILS
            .torch
            .getattr(py, "testing")
            .unwrap()
            .getattr(py, "assert_close")
            .unwrap()
            .call(py, (a, b), None)
    })
}

pub fn assert_is_close(a: CircuitRc, b: CircuitRc) {
    let a_ten = evaluate(a.clone()).unwrap();
    let b_ten = evaluate(b.clone()).unwrap();
    let close_result = is_close(a_ten.clone(), b_ten.clone());
    if close_result.is_err() {
        println!("{:?}", close_result);
        python_print(a_ten.tensor());
        python_print(b_ten.tensor());
        panic!("circuits did not evaluate to the same tensors");
    }
}

#[pyfunction]
pub fn opt_eval_each_subcircuit_until_fail(circuit: CircuitRc, settings: OptimizationSettings) {
    let max_numel = BigUint::from(2_000_000_000usize);
    let topo = toposort_circuit(circuit.clone());
    for circ in topo {
        if circ.info().numel() < max_numel {
            let reference = evaluate(circ.clone()).unwrap();
            let optimized = optimize_and_evaluate(circ.clone(), settings.clone());
            if let Ok(optimized) = optimized {
                let close_result = is_close(reference, optimized);
                if close_result.is_err() {
                    println!("{:?}", close_result);
                    circ.printu();
                    optimize_circuit(
                        circ,
                        &mut OptimizationContext::new_settings(settings.clone()),
                    )
                    .unwrap()
                    .printu();
                }
            }
        }
    }
}

/// Replaces all great-grandchildren of the circuit with normal random array constants
///
/// Note: given an output of this function `ablated`, any current rewrite satisfies
/// `ablated == rewrite(ablated)`. However, if you serialize both sides to Rust via
/// rust_expression_notation_circuit the results may differ. For a concrete example,
/// take Index(Index(X, tensor1), tensor2), which can be simplified to Index(X, tensor3)
pub fn randn_ablate_great_grandchildren(circ: CircuitRc) -> CircuitRc {
    let great_grandchildren = circ
        .children()
        .flat_map(|child| child.children().collect::<Vec<_>>())
        .flat_map(|grandchild| grandchild.children().collect::<Vec<_>>());
    let mut replacements: HashMap<HashBytes, CircuitRc> = HashMap::default();
    for great_grandchild in great_grandchildren {
        let ablated = Circuit::Array(Array::randn_full(
            great_grandchild.info().shape.clone(),
            great_grandchild.info().name,
            great_grandchild.info().device_dtype.clone(),
            Some(great_grandchild.info().hash_usize()),
        ))
        .rc();
        replacements.insert(great_grandchild.info().hash, ablated);
    }
    replace_nodes(circ, &replacements)
}
