use circuit_base::{
    evaluate,
    generalfunction::{SpecTrait, OFFICIAL_GENERALFUNCTION_INVERSES},
    Circuit, CircuitNode, CircuitRc, GeneralFunction, GeneralFunctionSpec, Index, Scalar,
};
use pyo3::prelude::*;
use rr_util::{
    sv,
    symbolic_size::SymbolicSizeProduct,
    tensor_util::{TensorAxisIndex, TensorIndex},
};
#[pyfunction]
pub fn generalfunction_merge_inverses(circuit: &GeneralFunction) -> Option<CircuitRc> {
    if circuit.num_children()==1 && circuit.spec.is_official()
        && let Circuit::GeneralFunction(inner) = &**circuit.info().children[0]
        && inner.spec.is_official()
        && OFFICIAL_GENERALFUNCTION_INVERSES.iter().any(|(a,b)|a==&circuit.spec.name()&&b==&inner.spec.name()) {
        return Some(inner.info().children[0].clone())
    }
    None
}

#[pyfunction]
pub fn generalfunction_special_case_simplification(circuit: &GeneralFunction) -> Option<CircuitRc> {
    if circuit.spec.is_official() {
        let name_str: &str = &circuit.spec.name();
        match name_str {
            "softmax" => {
                if let Circuit::Scalar(_sc) = &**circuit.children_sl()[0] {
                    let scalar: f64 = 1.0
                        / circuit.children_sl()[0].info().shape
                            [circuit.children_sl()[0].info().rank() - 1]
                            as f64;
                    return Some(Scalar::nrc(
                        scalar,
                        circuit.info().shape.clone(),
                        circuit.info().name,
                    ));
                }
            }
            "log_softmax" => {
                if let Circuit::Scalar(_sc) = &**circuit.children_sl()[0] {
                    let scalar: f64 = (1.0
                        / circuit.children_sl()[0].info().shape
                            [circuit.children_sl()[0].info().rank() - 1]
                            as f64)
                        .ln();
                    return Some(Scalar::nrc(
                        scalar,
                        circuit.info().shape.clone(),
                        circuit.info().name,
                    ));
                }
            }
            "last_dim_size" => {
                let last_dim_maybe_symbolic = circuit.children_sl()[0].info().shape
                    [circuit.children_sl()[0].info().rank() - 1];
                if !SymbolicSizeProduct::has_symbolic(last_dim_maybe_symbolic) {
                    return Some(Scalar::nrc(
                        last_dim_maybe_symbolic as f64,
                        circuit.children_sl()[0].info().shape
                            [..circuit.children_sl()[0].info().rank() - 1]
                            .iter()
                            .cloned()
                            .collect(),
                        circuit.info().name,
                    ));
                }
            }
            _ => {}
        }
    }
    None
}

/// this does less than python version, maybe should expand
#[pyfunction]
pub fn generalfunction_evaluate_simple(node: &GeneralFunction) -> Option<CircuitRc> {
    if node.num_non_batchable_output_dims == 0
        && node.info().rank() == 0
        && node.num_children() == 1
        && node.children_sl()[0].info().rank() == 0
    {
        return node.children_sl()[0].as_scalar().map(|_inner| {
            Python::with_gil(|py| {
                Scalar::new(
                    evaluate(node.crc())
                        .unwrap()
                        .tensor()
                        .getattr(py, "item")
                        .unwrap()
                        .call(py, (), None)
                        .unwrap()
                        .extract(py)
                        .unwrap(),
                    sv![],
                    node.info().name,
                )
                .rc()
            })
        });
    }
    None
}

#[pyfunction]
pub fn generalfunction_gen_index_const_to_index(node: &GeneralFunction) -> Option<CircuitRc> {
    if let GeneralFunctionSpec::Index(i) = &node.spec
        && let Circuit::Array(a) = &**node.children_sl()[1]
        && !i.batch_index
        && i.batch_x {
        let x = &node.children_sl()[0];
        let idx = if i.index_dim < 0 {
            (i.index_dim % (a.ndim() as i64)) as usize
        } else {
            i.index_dim as usize
        };
        if a.ndim() == 1 {

        return Some(Index::nrc(
            x.clone(),
            TensorIndex::new_single(TensorAxisIndex::Tensor(a.value.clone().try_into().unwrap()), idx, x.ndim()),
            node.info().name
        ))
        } else {
            return None
        }
    }
    None
}
