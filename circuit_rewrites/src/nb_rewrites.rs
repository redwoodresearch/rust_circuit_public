use circuit_base::{Add, CircuitNode, Einsum};
use itertools::Itertools;
use pyo3::prelude::*;
use rr_util::util::filter_out_idx;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::algebraic_rewrite::{get_removable_axes, remove_axes_weak};

#[pyfunction]
pub fn einsum_elim_removable_axes_weak(einsum: &Einsum) -> Option<Einsum> {
    let mut int_to_input: HashMap<u8, HashSet<usize>> = HashMap::default();
    for (i, ints) in einsum.in_axes.iter().enumerate() {
        for j in ints {
            int_to_input
                .entry(*j)
                .or_insert(HashSet::default())
                .insert(i);
        }
    }
    let mut int_to_input_removable: HashMap<u8, _> = HashMap::default();
    for (i, (node, ints)) in einsum.args().enumerate() {
        let removable = get_removable_axes(&node);
        for (axis, j) in ints.iter().enumerate() {
            if !removable.contains(&axis) {
                continue;
            }
            int_to_input_removable
                .entry(*j)
                .or_insert(HashSet::default())
                .insert((i, node.is_scalar()));
        }
    }
    for (k, v) in int_to_input_removable.clone() {
        if v.len() == int_to_input[&k].len() {
            if v.iter().map(|(_, is_scalar)| is_scalar).all_equal() {
                // if all of the entries are from things we might want to remove,
                // don't remove any (because it would be ambiguous what to remove)
                int_to_input_removable.remove(&k);
            } else {
                // if some scalar, and some non-scalar, only remove scalars
                int_to_input_removable
                    .get_mut(&k)
                    .unwrap()
                    .retain(|(_, is_scalar)| *is_scalar);
            }
        }
    }
    if int_to_input_removable.len() == 0 {
        return None;
    }

    let int_to_input_removable: HashMap<u8, HashSet<usize>> = int_to_input_removable
        .into_iter()
        .map(|(k, v)| (k, v.into_iter().map(|(i, _)| i).collect()))
        .collect();

    return Some(Einsum::new(
        einsum
            .args()
            .enumerate()
            .map(|(i, (node, ints))| {
                let removable_axes: HashSet<usize> = ints
                    .iter()
                    .enumerate()
                    .filter_map(|(j, int)| {
                        if int_to_input_removable
                            .get(int)
                            .map(|z| z.contains(&i))
                            .unwrap_or(false)
                        {
                            Some(j)
                        } else {
                            None
                        }
                    })
                    .collect();

                // we have to reintersect with removeable because ints can
                // correspond to multiple axes (e.g., ii,i->i)
                let removable_axes: HashSet<usize> = removable_axes
                    .intersection(&get_removable_axes(&node))
                    .cloned()
                    .collect();

                if removable_axes.len() == 0 {
                    (node.clone(), ints.clone())
                } else {
                    (
                        remove_axes_weak(&node, &removable_axes).unwrap(),
                        filter_out_idx(&ints, &removable_axes).into_iter().collect(),
                    )
                }
            })
            .collect(),
        einsum.out_axes.clone(),
        einsum.info().name,
    ));
}

#[pyfunction]
pub fn add_elim_removable_axes_weak(add: &Add) -> Option<Add> {
    let axis_map = add.child_axis_map();
    let mut axis_to_inputs: HashMap<usize, HashSet<usize>> = HashMap::default();
    let mut axis_to_inputs_removable: HashMap<usize, _> = HashMap::default();
    for (i, v) in axis_map.iter().enumerate() {
        let node = &add.children_sl()[i];
        let removable = get_removable_axes(&node);
        for (oj, j) in v
            .iter()
            .enumerate()
            .filter_map(|(axis, v_val)| v_val.as_ref().map(|y| (axis, y)))
        {
            axis_to_inputs
                .entry(*j)
                .or_insert(HashSet::default())
                .insert(i);
            if removable.contains(&oj) {
                axis_to_inputs_removable
                    .entry(*j)
                    .or_insert(HashSet::default())
                    .insert((i, node.is_scalar()));
            }
        }
    }
    for (k, v) in axis_to_inputs_removable.clone() {
        if axis_to_inputs[&k].len() == v.len() {
            if v.iter().map(|(_, is_scalar)| is_scalar).all_equal() {
                // if all of the entries are from things we might want to remove,
                // don't remove any (because it would be ambiguous what to remove)
                axis_to_inputs_removable.remove(&k);
            } else {
                // if some scalar, and some non-scalar, only remove scalars
                axis_to_inputs_removable
                    .get_mut(&k)
                    .unwrap()
                    .retain(|(_, is_scalar)| *is_scalar);
            }
        }
    }
    if axis_to_inputs_removable.len() == 0 {
        return None;
    }

    let axis_to_inputs_removable: HashMap<usize, HashSet<usize>> = axis_to_inputs_removable
        .into_iter()
        .map(|(k, v)| (k, v.into_iter().map(|(i, _)| i).collect()))
        .collect();

    let result = Add::new(
        add.nodes_and_rank_differences()
            .iter()
            .enumerate()
            .map(|(i, (node, rank_dif))| {
                let removable_axes: Vec<usize> = (*rank_dif..add.info().rank())
                    .enumerate()
                    .filter_map(|(j, int)| {
                        if axis_to_inputs_removable
                            .get(&int)
                            .map(|z| z.contains(&i))
                            .unwrap_or(false)
                        {
                            Some(j)
                        } else {
                            None
                        }
                    })
                    .collect();
                // removable_axes is sorted + unique
                let removable_axes: HashSet<_> = removable_axes
                    .into_iter()
                    .enumerate()
                    // we could truncate at first deviation instead of filtering
                    .filter_map(|(i, ax)| (i == ax).then_some(ax))
                    .collect();

                if removable_axes.len() == 0 {
                    node.clone()
                } else {
                    remove_axes_weak(node, &removable_axes).unwrap()
                }
            })
            .collect(),
        add.info().name,
    );
    if &result == add {
        return None;
    }
    return Some(result);
}
