use std::iter::zip;

use circuit_base::{prelude::*, Add, Concat, Einsum, Index, Rearrange, Scalar, Scatter};
use pyo3::prelude::*;
use rr_util::{
    compact_data::TinyVecU8,
    rearrange_spec::RearrangeSpec,
    tensor_util::{
        compose, uslices_shrink_base, uslices_to_index, Shape, TensorAxisIndex, TensorIndex, USlice,
    },
    util::{cumsum, filter_out_idx},
};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::algebraic_rewrite::{get_removable_axes, remove_axes};

/// right now writing scatter_fuse, einsum_pull_scatter, add_pull_scatter
#[pyfunction]
pub fn scatter_fuse(scatter: &Scatter) -> Option<Scatter> {
    // this is just composing indices, lower is top
    if let Circuit::Scatter(inner) = &**scatter.node() {
        let index_composed = compose(&inner.index, &scatter.index)?;
        Some(Scatter::new(
            inner.node().clone(),
            index_composed,
            scatter.info().shape.clone(),
            scatter.info().name,
        ))
    } else {
        None
    }
}

pub fn uslice_map_get_index(ints: &TinyVecU8, int_slices: &HashMap<u8, USlice>) -> TensorIndex {
    TensorIndex(
        ints.iter()
            .map(|i| {
                int_slices
                    .get(i)
                    .map(|s| (*s).into())
                    .unwrap_or(TensorAxisIndex::IDENT)
            })
            .collect(),
    )
}

#[pyfunction]
pub fn einsum_pull_scatter(einsum: &Einsum) -> Option<CircuitRc> {
    let mut did_anything = false;
    let int_sizes = einsum.shape_map().unwrap();
    let mut int_slices: HashMap<u8, USlice> = HashMap::default();
    for (node, ints) in einsum.args() {
        match &**node {
            Circuit::Scatter(scatter) => {
                did_anything = true;
                for (slice, i) in zip(scatter.index.all_uslices().unwrap(), ints.iter()) {
                    int_slices.insert(
                        *i,
                        int_slices
                            .get(i)
                            .unwrap_or(&USlice {
                                start: 0,
                                stop: int_sizes[i],
                            })
                            .intersection(&slice),
                    );
                }
            }
            _ => {}
        }
    }
    if !did_anything {
        None
    } else if int_slices
        .iter()
        .any(|(_i, slice)| slice.start == slice.stop)
    {
        Some(Scalar::new(0.0, einsum.info().shape.clone(), None).rc())
    } else {
        let new_args = einsum
            .args()
            .map(|(node, ints)| match &**node {
                Circuit::Scatter(inner) => {
                    let index_orig_base = ints.iter().map(|i| int_slices[i]).collect();
                    let index =
                        uslices_shrink_base(&index_orig_base, &inner.index.all_uslices().unwrap());
                    (
                        Index::nrc(inner.node().clone(), uslices_to_index(&index), None),
                        ints.clone(),
                    )
                }
                _ => (
                    Index::nrc(node.clone(), uslice_map_get_index(&ints, &int_slices), None),
                    ints.clone(),
                ),
            })
            .collect();
        let new_einsum = Einsum::nrc(new_args, einsum.out_axes.clone(), einsum.info().name);
        Some(Scatter::nrc(
            new_einsum,
            uslice_map_get_index(&einsum.out_axes, &int_slices).canonicalize(&einsum.info().shape),
            einsum.info().shape.clone(),
            None,
        ))
    }
}

#[pyfunction]
pub fn add_pull_scatter(add: &Add) -> Option<Scatter> {
    if !add.children().all(|x| matches!(**x, Circuit::Scatter(_))) {
        return None;
    }
    let mut slices: Vec<USlice> = add
        .info()
        .shape
        .iter()
        .map(|_l| USlice { start: 0, stop: 0 })
        .collect();
    for (operand, rank_difference) in add.nodes_and_rank_differences() {
        let scatter = operand.as_scatter().unwrap();
        for (i, (slice, l)) in
            zip(scatter.index.all_uslices().unwrap(), &scatter.info().shape).enumerate()
        {
            if *l == add.info().shape[i + rank_difference] {
                slices[i + rank_difference] = slices[i + rank_difference].union(&slice);
            } else {
                slices[i + rank_difference] = USlice {
                    start: 0,
                    stop: add.info().shape[i + rank_difference],
                };
            }
        }
        for i in 0..rank_difference {
            slices[i] = USlice {
                start: 0,
                stop: add.info().shape[i],
            };
        }
    }
    if zip(&slices, &add.info().shape).all(|(s, l)| s.start == 0 && s.stop == *l) {
        None
    } else {
        let new_operands = add
            .nodes_and_rank_differences()
            .iter()
            .map(|(node, rank_difference)| {
                let index = TensorIndex(
                    node.info()
                        .shape
                        .iter()
                        .enumerate()
                        .map(|(i, l)| {
                            if *l == add.info().shape[i + rank_difference] {
                                slices[i + rank_difference].into()
                            } else {
                                TensorAxisIndex::IDENT
                            }
                        })
                        .collect(),
                );
                Index::nrc(node.clone(), index, None)
            })
            .collect();
        let new_add = Add::nrc(new_operands, add.info().name);
        Some(Scatter::new(
            new_add,
            uslices_to_index(&(0..add.info().rank()).map(|i| slices[i]).collect()),
            add.info().shape.clone(),
            None,
        ))
    }
}

#[pyfunction]
pub fn scatter_elim_identity(scatter: &Scatter) -> Option<CircuitRc> {
    if scatter.is_identity() {
        Some(scatter.node().clone())
    } else {
        None
    }
}

#[pyfunction]
pub fn index_einsum_to_scatter(node: &Index) -> Option<CircuitRc> {
    let mut did_anything = false;
    if let Circuit::Einsum(inner) = &**node.node() {
        let mut int_slices: HashMap<u8, USlice> = HashMap::default();
        let canon_index = node.index.canonicalize(&inner.info().shape);
        let containing_uslices: Vec<Option<USlice>> = canon_index
            .0
            .iter()
            .map(USlice::containing_uslice)
            .collect();
        for (uslice_here, int) in zip(&containing_uslices, &inner.out_axes) {
            match uslice_here {
                None => {
                    return None;
                }
                Some(uslice_here) => {
                    if int_slices.contains_key(int) {
                        let new_here = uslice_here.intersection(&int_slices[int]);
                        if &new_here != &int_slices[int] || &new_here != uslice_here {
                            did_anything = true;
                            int_slices.insert(*int, new_here);
                        }
                    } else {
                        int_slices.insert(*int, *uslice_here);
                    }
                }
            }
        }
        if !did_anything {
            None
        } else if int_slices.iter().any(|(_i, slice)| slice.length() == 0) {
            Some(Scalar::new(0.0, node.info().shape.clone(), None).rc())
        } else {
            let new_args = inner
                .args()
                .map(|(node, ints)| {
                    (
                        Index::nrc(node, uslice_map_get_index(&ints, &int_slices), None),
                        ints.clone(),
                    )
                })
                .collect();
            let new_einsum = Einsum::nrc(new_args, inner.out_axes.clone(), inner.info().name);
            let new_shape = containing_uslices
                .iter()
                .map(|x| x.unwrap().stop - x.unwrap().start)
                .collect();
            let new_scatter = Scatter::nrc(
                new_einsum,
                TensorIndex(
                    inner
                        .out_axes
                        .iter()
                        .enumerate()
                        .map(|(i, int)| {
                            int_slices
                                .get(int)
                                .map(|s| (s.shrink_base(&containing_uslices[i].unwrap())).into())
                                .unwrap_or(TensorAxisIndex::IDENT)
                        })
                        .collect(),
                )
                .canonicalize(&new_shape),
                new_shape,
                None,
            );
            Some(Index::nrc(
                new_scatter,
                TensorIndex(
                    node.index
                        .0
                        .iter()
                        .map(|x| match x {
                            TensorAxisIndex::Single(_x) => TensorAxisIndex::Single(0),
                            _ => TensorAxisIndex::IDENT,
                        })
                        .collect(),
                ),
                None,
            ))
        }
    } else {
        None
    }
}

#[pyfunction]
pub fn scatter_pull_removable_axes(scatter: &Scatter) -> Option<Rearrange> {
    let removable_axes = get_removable_axes(&scatter.node());
    let scatter_uslices = scatter.index.all_uslices().unwrap();
    let removable_axes: HashSet<usize> = removable_axes
        .iter()
        .filter(|i| {
            scatter_uslices[**i]
                == USlice {
                    start: 0,
                    stop: scatter.info().shape[**i],
                }
        })
        .copied()
        .collect();
    if removable_axes.is_empty() {
        None
    } else {
        let new_inner = remove_axes(&scatter.node(), &removable_axes).unwrap();
        Some(Rearrange::new(
            Scatter::nrc(
                new_inner,
                TensorIndex(filter_out_idx(&scatter.index.0, &removable_axes)),
                filter_out_idx(
                    &scatter.info().shape.iter().cloned().collect::<Vec<_>>(),
                    &removable_axes,
                )
                .into_iter()
                .collect(),
                None,
            ),
            RearrangeSpec::unremove_axes(&removable_axes, &scatter.info().shape),
            None,
        ))
    }
}

#[pyfunction]
pub fn scatter_to_concat(scatter: &Scatter) -> CircuitRc {
    let mut result = scatter.node().clone();
    for (i, (idx, l)) in
        zip(scatter.index.all_uslices().unwrap(), &scatter.info().shape).enumerate()
    {
        let mut lower_pad_shape: Shape = result.info().shape.clone();
        lower_pad_shape[i] = idx.start;
        let mut upper_pad_shape: Shape = result.info().shape.clone();
        upper_pad_shape[i] = l - idx.stop;
        result = Concat::nrc(
            vec![
                Scalar::new(0.0, lower_pad_shape, None).rc(),
                result,
                Scalar::new(0.0, upper_pad_shape, None).rc(),
            ],
            i,
            scatter.info().name,
        );
    }
    result
}

#[pyfunction]
pub fn concat_to_scatter(concat: &Concat) -> Option<Scatter> {
    let pre_zeros = concat
        .children()
        .take_while(|node| {
            if let Circuit::Scalar(sc) = &***node && sc.value==0.0{
            true
        }else{
            false
        }
        })
        .count();
    let post_zeros = concat
        .children()
        .rev()
        .take_while(|node| {
            if let Circuit::Scalar(sc) = &***node && sc.value==0.0{
            true
        }else{
            false
        }
        })
        .count();
    let end = concat.num_children() - post_zeros;
    if pre_zeros == 0 && post_zeros == 0 || (pre_zeros + post_zeros >= concat.num_children()) {
        return None;
    }
    let starts = cumsum(&concat.get_sizes_at_axis());
    let cslice = TensorAxisIndex::new_plain_slice(starts[pre_zeros], starts[end]);
    let scatter_index = TensorIndex::new_single(cslice, concat.axis, concat.info().rank())
        .canonicalize(&concat.info().shape);
    let new_concat = Concat::nrc(
        concat.children_sl()[pre_zeros..end].to_vec(),
        concat.axis,
        concat.info().name,
    );
    if new_concat.info().numel() == concat.info().numel() {
        return None;
    }
    Some(Scatter::new(
        new_concat,
        scatter_index,
        concat.info().shape.clone(),
        None,
    ))
}
