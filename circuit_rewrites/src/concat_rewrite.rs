use std::iter::zip;

use circuit_base::{prelude::*, Add, Concat, Einsum, GeneralFunction, Index};
use num_bigint::BigUint;
use pyo3::prelude::*;
use rr_util::{
    tensor_util::{Slice, TensorAxisIndex, TensorIndex, USlice},
    util::{cumsum, outer_product, AsOp},
};
use rustc_hash::FxHashMap as HashMap;

use crate::algebraic_rewrite::add_make_broadcasts_explicit;
/// this nests arbitrarily deep
#[pyfunction]
pub fn einsum_pull_concat(einsum: &Einsum) -> Option<CircuitRc> {
    let mut did_anything = false;
    let mut int_sections: HashMap<u8, Vec<usize>> = HashMap::default();
    for (operand, ints) in einsum.args() {
        if let Circuit::Concat(concat) = &**operand
            && let int_here = ints[concat.axis]
            && einsum.out_axes.iter().filter(|z| **z == int_here).count() <= 1 {
            did_anything = true;
            let sections_here = concat.get_sizes_at_axis();
            match int_sections.get(&int_here) {
                None => {
                    int_sections.insert(int_here, sections_here);
                }
                Some(prev) if prev[..] != sections_here[..] => {
                    return None;
                }
                _  => {}
            }
        }
    }
    if !did_anything {
        return None;
    }

    // for each concat dimension that's reduced over, compute einsums seperately for each section and Add at the end
    // when there are multiple, we outer product and flat add

    // for each concat dimension that's kept, einsum seperately for each section and Concat at the end
    // if there are multiple, we outer product and nest concats
    let (int_sections_concat, int_sections_add): (Vec<u8>, _) = int_sections
        .keys()
        .cloned()
        .partition(|i| einsum.out_axes.contains(i));

    fn recurse(
        concat_picks: &HashMap<u8, usize>,
        int_sections_concat_left: &[u8],
        int_sections_add: &Vec<u8>,
        int_sections: &HashMap<u8, Vec<usize>>,
        einsum: &Einsum,
    ) -> CircuitRc {
        if int_sections_concat_left.is_empty() {
            let summands: Vec<CircuitRc> = outer_product(
                &int_sections_add
                    .iter()
                    .map(|i| (0..int_sections[i].len()).collect())
                    .collect::<Vec<_>>(),
            )
            .iter()
            .map(|summand_settings| {
                let einsum_args = einsum
                    .args()
                    .map(|(node, ints)| {
                        let index = TensorIndex(
                            ints.iter()
                                .map(|i| {
                                    if let Some(u) = concat_picks.get(i) {
                                        let sections = &int_sections[i];
                                        let cumsums = cumsum(sections);

                                        return TensorAxisIndex::new_plain_slice(
                                            cumsums[*u],
                                            cumsums[u + 1],
                                        );
                                    }
                                    if let Some(u) = int_sections_add.iter().position(|z| z == i) {
                                        let sections = &int_sections[i];
                                        let cumsums = cumsum(sections);

                                        return TensorAxisIndex::new_plain_slice(
                                            cumsums[summand_settings[u]],
                                            cumsums[summand_settings[u] + 1],
                                        );
                                    }
                                    TensorAxisIndex::IDENT
                                })
                                .collect(),
                        );
                        (Index::nrc(node.clone(), index, None), ints.clone())
                    })
                    .collect();

                Einsum::nrc(einsum_args, einsum.out_axes.clone(), einsum.info().name)
            })
            .collect();
            Add::nrc(summands, None)
        } else {
            let (ihere, new_int_sections_concat) = int_sections_concat_left.split_first().unwrap();
            let concattands: Vec<CircuitRc> = (0..int_sections[ihere].len())
                .map(|i| {
                    let mut new_picks = concat_picks.clone();
                    new_picks.insert(*ihere, i);
                    recurse(
                        &new_picks,
                        new_int_sections_concat,
                        int_sections_add,
                        int_sections,
                        einsum,
                    )
                })
                .collect();
            Concat::nrc(
                concattands,
                einsum.out_axes.iter().position(|x| x == ihere).unwrap(),
                None,
            )
        }
    }

    Some(recurse(
        &HashMap::default(),
        &int_sections_concat,
        &int_sections_add,
        &int_sections,
        einsum,
    ))
}

#[pyfunction]
pub fn add_pull_concat(add: &Add) -> Option<CircuitRc> {
    for (child, rank_difference) in add.nodes_and_rank_differences() {
        if let Circuit::Concat(concat) = &**child {
            if concat.info().shape[concat.axis] != 1 {
                let sections_here = concat.get_sizes_at_axis();
                let concat_axis = rank_difference + concat.axis;
                let add_explicit = add_make_broadcasts_explicit(add)
                    .unwrap_or(add.clone())
                    .rc();
                let concat_of_adds = split_to_concat(add_explicit, concat_axis, sections_here).rc();
                if add.info().shape[..] != concat_of_adds.info().shape[..] {
                    add.crc().printu();
                    concat_of_adds.printu();
                    panic!();
                }
                return Some(concat_of_adds);
                // let concattands = (0..sections_here.len()).map(|i|{
                //     Add::try_new(add_of_concats.nodes.iter().map(|)
                // }).collect();
                // return Some(Concat::try_new(concattands,concat_axis,None).unwrap().rc())
            }
        }
    }
    None
}

// pub fn rearrange_pull_concat(rearrange: &Rearrange) -> Option<Concat> {}
#[pyfunction]
pub fn generalfunction_pull_concat(generalfunction: &GeneralFunction) -> Option<Concat> {
    let batch_rank =
        generalfunction.info().rank() - generalfunction.num_non_batchable_output_dims as usize;
    for operand in generalfunction.children() {
        if let Circuit::Concat(concat) = &**operand && concat.axis < batch_rank {
            let mut cur = 0;
            let new_operands = concat
                .children()
                .map(|cnode| {
                    let span_here = cnode.info().shape[concat.axis];
                    let new_operands = zip(
                        generalfunction.children(),
                        &generalfunction.input_batchability,
                    )
                    .map(|(node, &batchability)| {
                        if !batchability {
                            node.clone()
                        } else if node == operand {
                            cnode.clone()
                        } else {
                            Index::nrc(
                                node.clone(),
                                TensorIndex::new_single(
                                    TensorAxisIndex::new_plain_slice(cur, cur + span_here),
                                    concat.axis,
                                    node.info().rank(),
                                ),
                                None,
                            )
                        }
                    })
                    .collect();
                    cur += span_here;
                    GeneralFunction::nrc(new_operands, generalfunction.spec.clone(), None)
                })
                .collect();
            return Some(Concat::new(new_operands, concat.axis, None));
        }
    }
    None
}

#[pyfunction]
pub fn concat_fuse(concat: &Concat) -> Option<Concat> {
    let mut did_anything = false;
    let new_operands = concat
        .children()
        .flat_map(|child| {
            if let Circuit::Concat(inner) = &**child && inner.axis == concat.axis {
                did_anything = true;
                inner.children_sl().to_vec()
            } else{
                vec![child.clone()]
            }
        })
        .collect();
    if !did_anything {
        return None;
    }
    Some(Concat::new(new_operands, concat.axis, concat.info().name))
}

#[pyfunction]
pub fn concat_drop_size_zero(concat: &Concat) -> Option<Concat> {
    let mut did_anything = false;
    let new_nodes = concat
        .children()
        .filter(|x| {
            if x.info().shape[concat.axis] == 0 {
                did_anything = true;
                false
            } else {
                true
            }
        })
        .collect();
    if !did_anything {
        return None;
    }
    Some(Concat::new(new_nodes, concat.axis, concat.info().name))
}

#[pyfunction]
pub fn index_concat_drop_unreached(index: &Index) -> Option<CircuitRc> {
    // canon so we only have to work with positives
    let index_canon = index.index.canonicalize(&index.node().info().shape).0;
    let child = index.node();
    let concat: &Concat = (&**child).as_op()?;
    if concat
        .children()
        .any(|x| x.info().numel() == BigUint::from(0usize))
    {
        return None;
    }
    let axis_index = &index_canon[concat.axis];
    let operand_starts: Vec<usize> = cumsum(
        &concat
            .children()
            .map(|x| x.info().shape[concat.axis])
            .collect::<Vec<_>>(),
    );
    let (new_operands, new_index) = match axis_index {
        TensorAxisIndex::Single(single) => {
            let operand_idx = operand_starts
                .iter()
                .filter(|x| **x <= *single as usize)
                .count()
                - 1;
            (
                vec![concat.children_sl()[operand_idx].clone()],
                TensorAxisIndex::Single(single - operand_starts[operand_idx] as i64),
            )
        }
        TensorAxisIndex::Slice(slice) => {
            let uslice: Option<USlice> = (*slice).into();
            let uslice = uslice.unwrap();
            let operand_idxs: Vec<usize> = (0..concat.num_children())
                .filter(|i| {
                    operand_starts[*i] < uslice.stop && operand_starts[i + 1] > uslice.start
                })
                .collect();
            let start_point = operand_starts[operand_idxs[0]];
            (
                operand_idxs
                    .iter()
                    .map(|x| concat.children_sl()[*x].clone())
                    .collect(),
                TensorAxisIndex::new_plain_slice(
                    uslice.start - start_point,
                    uslice.stop - start_point,
                ),
            )
        }
        _ => (concat.children_sl().to_vec(), axis_index.clone()),
    };
    if new_operands.len() == concat.num_children() {
        return None;
    }
    let mut new_tensor_index = index.index.clone();
    new_tensor_index.0[concat.axis] = new_index;
    Some(Index::nrc(
        Concat::nrc(new_operands, concat.axis, concat.info().name),
        new_tensor_index,
        index.info().name,
    ))
}

#[pyfunction]
pub fn split_to_concat(circuit: CircuitRc, axis: usize, sections: Vec<usize>) -> Concat {
    assert!(axis < circuit.info().rank());
    let starts = cumsum(&sections);
    let operands: Vec<CircuitRc> = (0..sections.len())
        .map(|i| {
            Index::nrc(
                circuit.clone(),
                TensorIndex::new_single(
                    TensorAxisIndex::new_plain_slice(starts[i], starts[i + 1]),
                    axis,
                    circuit.info().rank(),
                ),
                None,
            )
        })
        .collect();
    Concat::new(operands, axis, None)
}

#[pyfunction]
pub fn concat_elim_split(circuit: &Concat) -> Option<Index> {
    let indexes = circuit
        .children()
        .map(|x| x.as_index().cloned())
        .collect::<Option<Vec<Index>>>()?;
    let child = indexes[0].node().clone();
    let child_axis_map = indexes[0].child_axis_map()[0].clone();
    if !indexes
        .iter()
        .all(|x| x.node() == child && x.child_axis_map()[0] == child_axis_map)
    {
        return None;
    }
    let axis_lower = indexes[0].child_axis_map_including_slices()[0]
        .iter()
        .position(|i| *i == Some(circuit.axis))?;
    let sections: Vec<Slice> = indexes
        .iter()
        .map(|x| match x.index.0[axis_lower] {
            TensorAxisIndex::Slice(slice) => Some(slice),
            _ => None,
        })
        .collect::<Option<Vec<Slice>>>()?;
    let sections: Vec<USlice> = sections
        .iter()
        .map(|x| x.to_uslice(child.info().shape[axis_lower]))
        .collect();
    let mut place = 0;
    for s in &sections {
        if s.start != place {
            return None;
        }
        place = s.stop;
    }
    let new_axis_index = TensorAxisIndex::Slice(Slice {
        start: Some(sections[0].start as i64),
        stop: Some(place as i64),
    });
    let mut new_index = vec![];
    for i in 0..child.info().rank() {
        if i == axis_lower {
            new_index.push(new_axis_index.clone());
        } else {
            for idx in &indexes {
                if idx.index.0[i] != indexes[0].index.0[i] {
                    return None;
                }
            }
            new_index.push(indexes[0].index.0[i].clone());
        }
    }
    Some(Index::new(
        child,
        TensorIndex(new_index),
        circuit.info().name,
    ))
}

pub fn split_sections(l: usize, sections: usize) -> Vec<usize> {
    let even_size = l / sections;
    let residual = l % sections;
    [
        vec![even_size + 1; residual],
        vec![even_size; sections - residual],
    ]
    .concat()
}

#[test]
fn test_split_sections() {
    dbg!(split_sections(10, 3));
}

// todo: order concats by axis?
