use std::iter::zip;

use circuit_base::{Add, Circuit, CircuitNode, Concat, Einsum, Index, Rearrange};
use itertools::Itertools;
use pyo3::prelude::*;
use rr_util::{
    compact_data::TinyVecU8,
    tensor_util::{TensorAxisIndex, TensorIndex},
    union_find::UnionFind,
    unwrap,
    util::{is_unique, AxisInt, EinsumAxes},
};
use rustc_hash::FxHashSet as HashSet;

use crate::algebraic_rewrite::add_make_broadcasts_explicit;

/// yay! fun when you find a linear time version of a thing you did super slow bc you're lazy
/// transpose so you have ints used by each diag on each axis, hash.
pub fn diags_intersection(diags: &Vec<EinsumAxes>) -> EinsumAxes {
    assert!(!diags.is_empty() && diags.iter().all(|x| x.len() == diags[0].len()));
    let transpose: Vec<EinsumAxes> = (0..diags[0].len())
        .map(|i| diags.iter().map(|d| d[i]).collect::<TinyVecU8>())
        .collect();
    let deduped: Vec<EinsumAxes> = transpose.iter().unique().cloned().collect();
    transpose
        .iter()
        .map(|x| deduped.iter().position(|z| z == x).unwrap() as u8)
        .collect()
}

pub fn diags_union(diags: &Vec<EinsumAxes>) -> EinsumAxes {
    let mut uf = UnionFind::new(diags[0].len());
    for d in diags {
        for (i1, int1) in d.iter().enumerate() {
            for (i2, int2) in d.iter().enumerate() {
                if i1 < i2 && int1 == int2 {
                    println!("{} {}", i1, i2);
                    uf.union(i1 as usize, i2 as usize);
                }
            }
        }
    }
    (0..diags[0].len()).map(|i| uf.find(i) as u8).collect()
}

// some stuff in here should likely be extracted to helpers
#[pyfunction]
pub fn einsum_push_down_trace(einsum: &Einsum) -> Option<Einsum> {
    let mut did_anything = false;
    let new_args = einsum
        .args()
        .map(|(node, ints)| {
            if !is_unique(&ints) {
                match &**node {
                    Circuit::Add(child) => {
                        // have to rearrange
                        let explicit_add =
                            add_make_broadcasts_explicit(&child).unwrap_or(child.clone());
                        let new_add = Add::nrc(
                            explicit_add
                                .children()
                                .map(|node| {
                                    Einsum::new_trace(node.clone(), ints.clone(), None).rc()
                                })
                                .collect(),
                            None,
                        );
                        did_anything = true;
                        return (new_add, ints.into_iter().unique().cloned().collect());
                    }
                    Circuit::Rearrange(child) => {
                        let axis_map = &child.child_axis_map()[0];
                        let ints_for_intersection = (0..child.info().rank())
                            .map(|i| {
                                if axis_map.iter().any(|z| *z == Some(i)) {
                                    0
                                } else {
                                    i as u8 + 1
                                }
                            })
                            .collect();
                        let intersected_ints =
                            diags_intersection(&vec![ints.clone(), ints_for_intersection]);
                        let intersected_deduped: Vec<_> =
                            intersected_ints.iter().unique().collect();
                        if intersected_ints.len() != intersected_deduped.len() {
                            let tuples_to_remove: HashSet<Box<[AxisInt]>> =
                                zip(intersected_ints.iter().enumerate(), &child.spec.output_ints)
                                    .filter_map(|((i, trace_int), oints)| {
                                        if intersected_ints
                                            .iter()
                                            .position(|z| z == trace_int)
                                            .unwrap()
                                            != i
                                        {
                                            Some(oints[..].into())
                                        } else {
                                            None
                                        }
                                    })
                                    .collect();
                            let thingy: EinsumAxes = axis_map
                                .iter()
                                .enumerate()
                                .map(|(i, x)| {
                                    if let Some(z) = x {
                                        intersected_ints[*z]
                                    } else {
                                        i as u8 + child.info().rank() as u8
                                    }
                                })
                                .collect();
                            // can't just use new_trace bc which one gets eliminated is determined by output, not input
                            // so for instance `aba` might go to `ba` instead of normal `ab`
                            let new_trace = Einsum::nrc(
                                vec![(child.node().clone(), thingy.clone())],
                                (0..child.node().info().rank())
                                    .filter(|i| {
                                        !tuples_to_remove.contains(&child.spec.input_ints[*i][..])
                                    })
                                    .map(|i| thingy[i])
                                    .collect(),
                                None,
                            );

                            let new_rearrange = Rearrange::nrc(
                                new_trace,
                                child.spec.filter_all_tuples(&tuples_to_remove).unwrap(),
                                None,
                            );
                            let up_axes = intersected_deduped
                                .iter()
                                .map(|i| {
                                    ints[intersected_ints.iter().position(|z| z == *i).unwrap()]
                                })
                                .collect();
                            did_anything = true;
                            return (new_rearrange, up_axes);
                        }
                    }
                    Circuit::Index(child) => {
                        let ints_for_intersection = zip(&child.index.0, &child.node().info().shape)
                            .enumerate()
                            .filter_map(|(i, (idx, l))| {
                                if matches!(idx, TensorAxisIndex::Single(_)) {
                                    return None;
                                }
                                Some(if idx.is_identity(*l) {
                                    0_u8
                                } else {
                                    (i + 1) as u8
                                })
                            })
                            .collect();
                        let intersected_ints =
                            diags_intersection(&vec![ints_for_intersection, ints.clone()]);
                        let intersected_deduped: EinsumAxes =
                            intersected_ints.iter().unique().cloned().collect();
                        if intersected_ints.len() != intersected_deduped.len() {
                            did_anything = true;
                            let child_map = &child.child_axis_map()[0];
                            let intersected_ints_input: EinsumAxes = child_map
                                .iter()
                                .enumerate()
                                .map(|(i, x)| {
                                    if let Some(z) = x {
                                        intersected_ints[*z] as u8
                                    } else {
                                        (i + child.node().info().rank()) as u8
                                    }
                                })
                                .collect();
                            let intersected_ints_input_deduped: EinsumAxes =
                                intersected_ints_input.iter().unique().cloned().collect();
                            let new_trace = Einsum::new_trace(
                                child.node().clone(),
                                intersected_ints_input.clone(),
                                None,
                            )
                            .rc();
                            let up_axes = intersected_deduped
                                .iter()
                                .map(|i| {
                                    ints[intersected_ints.iter().position(|z| z == i).unwrap()]
                                })
                                .collect();
                            return (
                                Index::nrc(
                                    new_trace,
                                    TensorIndex(
                                        intersected_ints_input_deduped
                                            .iter()
                                            .map(|i| {
                                                child.index.0[intersected_ints_input
                                                    .iter()
                                                    .position(|z| z == i)
                                                    .unwrap()]
                                                .clone()
                                            })
                                            .collect(),
                                    ),
                                    None,
                                ),
                                up_axes,
                            );
                        }
                    }
                    Circuit::Concat(child) => {
                        let mut new_raw_ints = ints.clone();
                        let concat_axis_int = new_raw_ints.iter().max().unwrap_or(&0) + 1;
                        new_raw_ints.as_mut_slice()[child.axis] = concat_axis_int;
                        let deduped_here: EinsumAxes =
                            new_raw_ints.iter().unique().cloned().collect();
                        if deduped_here.len() != new_raw_ints.len() {
                            let new_concat_axis = deduped_here
                                .iter()
                                .position(|x| *x == concat_axis_int)
                                .unwrap();
                            let new_concat = Concat::nrc(
                                child
                                    .children()
                                    .map(|x| {
                                        Einsum::new_trace(x.clone(), new_raw_ints.clone(), None)
                                            .rc()
                                    })
                                    .collect(),
                                new_concat_axis,
                                None,
                            );
                            let up_axes = deduped_here
                                .iter()
                                .map(|x| {
                                    if *x == concat_axis_int {
                                        ints[child.axis]
                                    } else {
                                        *x
                                    }
                                })
                                .collect();
                            did_anything = true;
                            return (new_concat, up_axes);
                        }
                    }
                    _ => {}
                }
            }
            (node, ints.clone())
        })
        .collect();
    if !did_anything {
        return None;
    }
    Some(Einsum::new(
        new_args,
        einsum.out_axes.clone(),
        einsum.info().name,
    ))
}

#[pyfunction]
pub fn add_pull_diags(add: &Add) -> Option<Einsum> {
    if !add.children().all(|x| matches!(**x, Circuit::Einsum(_))) {
        return None;
    }
    // here we're making a Vec<u8> for each child where two elements are the same iff the tensor is diagonal across them
    // and the axis is full size (not a size 1 axis to broadcast over)
    // we do this by filling concat( 0..rank_difference, einsum.out_axes+rank_difference if axis_len!=1 else high_random_number)
    //
    let diags = add
        .nodes_and_rank_differences()
        .iter()
        .map(|(node, rank_difference)| {
            let rank_difference = *rank_difference as u8;
            let einsum = unwrap!(&***node, Circuit::Einsum);
            (0..rank_difference)
                .chain(zip(&einsum.info().shape, &einsum.out_axes).enumerate().map(
                    |(i, (l, x))| {
                        if *l > 1 {
                            x + rank_difference as u8
                        } else {
                            i as u8 + rank_difference + *einsum.out_axes.iter().max().unwrap() + 1
                        }
                    },
                ))
                .collect()
        })
        .collect();
    let overall_diags = diags_intersection(&diags);
    let overall_diags_deduped: EinsumAxes = overall_diags.iter().unique().cloned().collect();
    if overall_diags_deduped.len() == overall_diags.len() {
        return None;
    }
    let new_add = Add::nrc(
        add.nodes_and_rank_differences()
            .iter()
            .map(|(node, rdif)| {
                let ein = unwrap!(&***node, Circuit::Einsum);
                Einsum::nrc(
                    ein.args_cloned(),
                    overall_diags[*rdif..]
                        .iter()
                        .unique()
                        .map(|i| {
                            ein.out_axes
                                [overall_diags[*rdif..].iter().position(|z| z == i).unwrap()]
                        })
                        .collect(),
                    None,
                )
            })
            .collect(),
        add.info().name,
    );
    Some(Einsum::new_diag(new_add, overall_diags, None))
}
