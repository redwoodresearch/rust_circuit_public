use circuit_base::{deep_map_op, deep_map_op_context, prelude::*, Add, Einsum, Index, Rearrange};
use pyo3::prelude::*;
use rustc_hash::FxHashMap as HashMap;

use crate::circuit_optimizer::OptimizationContext;

/// takes circuitrc bc convenient
pub fn numel_sort_key(node: &CircuitRc) -> Vec<u8> {
    (usize::MAX - node.info().numel().to_u64_digits()[0] as usize)
        .to_be_bytes()
        .iter()
        .copied()
        .chain(node.variant_string().bytes())
        .chain(node.info().hash)
        .collect::<Vec<u8>>()
}

#[pyfunction]
#[pyo3(name = "canonicalize_node")]
pub fn canonicalize_node_py(circuit: CircuitRc) -> CircuitRc {
    canonicalize_node_op(circuit.clone()).unwrap_or(circuit)
}

pub fn canonicalize_node_op(circuit: CircuitRc) -> Option<CircuitRc> {
    match &**circuit {
        Circuit::Rearrange(rearrange) => Some(Rearrange::nrc(
            rearrange.node().clone(),
            rearrange
                .spec
                .conform_to_input_shape(&rearrange.node().info().shape)
                .unwrap()
                .canonicalize(true),
            circuit.info().name,
        )),
        Circuit::Index(index) => Some(Index::nrc(
            index.node().clone(),
            index.index.canonicalize(&index.node().info().shape),
            index.info().name,
        )),
        Circuit::Add(add) => {
            let mut nodes_sorted = add.children_sl().to_vec();
            nodes_sorted.sort_by_key(numel_sort_key);
            Some(Add::nrc(nodes_sorted, add.info().name))
        }
        Circuit::Einsum(einsum) => {
            let mut args_sorted = einsum.args_cloned();
            args_sorted.sort_by_key(|(node, _ints)| numel_sort_key(node));
            Some(
                Einsum::try_new(args_sorted, einsum.out_axes.clone(), einsum.info().name)
                    .unwrap()
                    .normalize_ints()
                    .rc(),
            )
        }
        _ => None,
    }
}

#[pyfunction]
#[pyo3(name = "deep_canonicalize")]
pub fn deep_canonicalize_py(circuit: CircuitRc) -> CircuitRc {
    deep_canonicalize(circuit, &mut Default::default())
}

pub fn deep_canonicalize(circuit: CircuitRc, context: &mut OptimizationContext) -> CircuitRc {
    deep_map_op_context(
        circuit.clone(),
        &|x, _c: &mut HashMap<(), ()>| canonicalize_node_op(x),
        &mut HashMap::<(), ()>::default(),
        &mut context.cache.canonicalized,
    )
    .unwrap_or(circuit)
}

#[pyfunction]
#[pyo3(name = "canonicalize_node")]
pub fn normalize_node_py(circuit: CircuitRc) -> CircuitRc {
    normalize_node_op(circuit.clone()).unwrap_or(circuit)
}

pub fn normalize_node_op(circuit: CircuitRc) -> Option<CircuitRc> {
    match &**circuit {
        Circuit::Rearrange(rearrange) => Some(Rearrange::nrc(
            rearrange.node().clone(),
            rearrange.spec.canonicalize(false),
            circuit.info().name,
        )),
        Circuit::Einsum(einsum) => Some(einsum.normalize_ints().rc()),
        _ => None,
    }
}

#[pyfunction]
pub fn deep_normalize(circuit: CircuitRc) -> CircuitRc {
    deep_map_op(circuit.clone(), normalize_node_op).unwrap_or(circuit)
}
