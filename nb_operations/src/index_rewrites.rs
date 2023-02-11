use anyhow::Result;
use circuit_base::{prelude::*, CircuitType, Index};
use circuit_rewrites::algebraic_rewrite::{index_elim_identity, push_down_index_raw};
use get_update_node::{new_traversal, IterativeMatcherRc, Matcher};
use macro_rules_attribute::apply;
use pyo3::prelude::*;
use rr_util::{cached_lambda, util::HashBytes};

#[pyfunction]
pub fn default_index_traversal() -> IterativeMatcherRc {
    new_traversal(
        None,
        None,
        Matcher::types(vec![CircuitType::Index, CircuitType::Array]).rc(),
    )
    .rc()
}

#[pyfunction]
#[pyo3(signature=(
    index,
    traversal = default_index_traversal(),
    suffix = None,
    allow_partial_pushdown = false,
    elim_identity = true,
))]
pub fn push_down_index(
    index: Index,
    traversal: IterativeMatcherRc,
    suffix: Option<String>,
    allow_partial_pushdown: bool,
    elim_identity: bool,
) -> Result<CircuitRc> {
    #[apply(cached_lambda)]
    #[key((index.info().hash, traversal.clone()), (HashBytes, IterativeMatcherRc))]
    #[use_try]
    fn push_down_rec(index: Index, traversal: IterativeMatcherRc) -> Result<CircuitRc> {
        if elim_identity {
            if let Some(removed_node) = index_elim_identity(&index) {
                return Ok(removed_node);
            }
        }
        let updated = traversal
            .match_iterate(index.node().clone())?
            .unwrap_or_same(traversal)
            .0;

        if updated.all_finished() {
            return Ok(index.rc());
        }

        let traversal_per_child = updated.per_child_with_term(index.node().num_children());

        push_down_index_raw(
            &index,
            allow_partial_pushdown,
            &mut |i, new_index| push_down_rec(new_index, traversal_per_child[i].clone()),
            suffix.clone(),
        )
    }

    push_down_rec(index, traversal)
}
