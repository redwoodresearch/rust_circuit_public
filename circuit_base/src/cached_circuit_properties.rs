use std::{cell::RefCell, iter::once};

use num_bigint::BigUint;
use once_cell::unsync::Lazy;
use rustc_hash::FxHashMap as HashMap;

use crate::{CircuitNode, CircuitRc};

thread_local! {
    static MAX_NON_LEAF_SIZE: RefCell<HashMap<rr_util::util::HashBytes,BigUint>> =
        RefCell::new(HashMap::default());
}

const BIG_UINT_0: Lazy<BigUint> = Lazy::new(|| 0u8.into());
pub fn max_non_leaf_size(circuit: CircuitRc) -> BigUint {
    if let Some(result) =
        MAX_NON_LEAF_SIZE.with(|cache| cache.borrow().get(&circuit.info().hash).cloned())
    {
        return result;
    }
    let result: BigUint = circuit
        .children()
        .map(max_non_leaf_size)
        .chain(once(if circuit.is_leaf() {
            BIG_UINT_0.clone()
        } else {
            circuit.info().naive_mem_use(None)
        }))
        .max()
        .unwrap_or(BIG_UINT_0.clone());
    MAX_NON_LEAF_SIZE.with(|cache| {
        cache
            .borrow_mut()
            .insert(circuit.info().hash, result.clone())
    });
    result
}
