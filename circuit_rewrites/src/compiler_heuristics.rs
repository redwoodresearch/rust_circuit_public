use circuit_base::{deep_map_op_context, prelude::*, Einsum};
use num_bigint::BigUint;
use pyo3::prelude::*;
use rr_util::timed;
use rustc_hash::FxHashMap as HashMap;

use crate::{
    algebraic_rewrite::distribute_once_op,
    canonicalize::deep_canonicalize,
    circuit_optimizer::{OptimizationContext, OptimizationSettings},
    deep_rewrite::compiler_simp,
};
#[pyfunction]
pub fn maybe_distribute_py(node: &Einsum) -> Option<CircuitRc> {
    maybe_distribute_uncached(node, &mut Default::default())
}

pub fn maybe_distribute(node: &Einsum, context: &mut OptimizationContext) -> Option<CircuitRc> {
    let key = node.info().hash;
    match context.cache.distributed.get(&key) {
        Some(z) => z.clone(),
        None => {
            let result = maybe_distribute_uncached(node, context);
            context.cache.distributed.insert(key, result.clone());
            result
        }
    }
}

/// only is reasonable if adds have gone through add_pull_removable, but meant to not crash otherwise
/// this is simpler than python version, maybe worse than it
pub fn maybe_distribute_uncached(
    node: &Einsum,
    context: &mut OptimizationContext,
) -> Option<CircuitRc> {
    if context.cache.times_distributed > 10000 {
        println!("compiler hit distribute limit");
        return None;
    }
    for (i, operand) in node.children().enumerate() {
        if let Circuit::Add(add) = &**operand && !(add.num_children() == 0) && ( operand.info().numel() >= BigUint::from(context.settings.distribute_min_size.unwrap_or(context.settings.max_single_tensor_memory)))

        {
            let result = timed!(deep_canonicalize( compiler_simp(distribute_once_op(node, i, true).unwrap().rc(),context),context),1,context.settings.verbose>=3);
            context.cache.times_distributed+=1;
            return Some(result);
        }
    }
    None
}

#[pyfunction]
#[pyo3(name = "deep_maybe_distribute")]
pub fn deep_maybe_distribute_py(node: CircuitRc, settings: OptimizationSettings) -> CircuitRc {
    let context = &mut OptimizationContext::new_settings(settings);
    deep_maybe_distribute(node, context)
}

pub fn deep_maybe_distribute(node: CircuitRc, context: &mut OptimizationContext) -> CircuitRc {
    deep_map_op_context(
        node.clone(),
        &|x: CircuitRc, context: &mut OptimizationContext| match &**x {
            Circuit::Einsum(ein) => maybe_distribute(ein, context),
            _ => None,
        },
        context,
        &mut HashMap::default(),
    )
    .unwrap_or(node)
}
