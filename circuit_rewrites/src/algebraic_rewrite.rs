use std::{
    collections::hash_map::Entry::{self, Vacant},
    iter::{self, zip},
    sync::Arc,
};

use anyhow::{bail, Context, Error, Result};
use bit_iter::BitIter;
use circuit_base::{
    computational_node::EinsumArgs, prelude::*, Add, CircuitType, Concat, DiscreteVar, Einsum,
    GeneralFunction, Index, Module, Rearrange, Scalar, Scatter, Symbol, Tag,
};
use itertools::Itertools;
use macro_rules_attribute::apply;
use pyo3::{exceptions::PyValueError, prelude::*};
use rr_util::{
    compact_data::U8Set,
    filter_by_variant,
    name::Name,
    opt_einsum::optimize_einsum_spec_cached,
    python_error_exception,
    rearrange_spec::{shape_to_op_shape, OpSize, RInts, RearrangeSpec},
    sv,
    tensor_util::{
        check_canon_idxs, compose, IndexTensor, Shape, Slice, TensorAxisIndex, TensorIndex, USlice,
    },
    tu8v,
    union_find::UnionFind,
    unwrap,
    util::{
        arc_ref_clone, cumsum, filter_out_idx, filter_to_idx, intersection_all,
        inverse_permutation, is_unique, unique_to_appearance, AxisInt, BitMask128, EinsumAxes,
        HashBytes,
    },
};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use thiserror::Error;

use crate::circuit_optimizer::OptimizationContext;

#[pyfunction]
pub fn add_flatten_once(add: &Add) -> Option<Add> {
    let mut did_anything = false;
    let new_operands: Vec<CircuitRc> = add
        .children()
        .flat_map(|node| match &**node {
            Circuit::Add(inner_add) => {
                did_anything = true;
                inner_add.children_sl().to_vec()
            }
            _ => {
                vec![node.clone()]
            }
        })
        .collect();
    if !did_anything {
        return None;
    }
    Some(Add::new(new_operands, add.info().name))
}

#[pyfunction]
pub fn add_collapse_scalar_inputs(add: &Add) -> Option<Add> {
    let (scalar_inputs, non_scalar_inputs): (Vec<Scalar>, Vec<Arc<Circuit>>) = filter_by_variant!(
        add.children().map(|x| ((*x).clone())),
        Circuit,
        Scalar,
        Scalar
    );
    if scalar_inputs.len() <= 1 {
        None
    } else {
        let new_scalar: f64 = scalar_inputs.iter().map(|node| node.value).sum();
        let mut new_inputs: Vec<CircuitRc> =
            non_scalar_inputs.iter().map(|x| x.clone().into()).collect();
        let new_scalar_input = Scalar::new(new_scalar, add.info().shape.clone(), None).rc();
        new_inputs.push(new_scalar_input);
        Some(Add::new(new_inputs, add.info().name))
    }
}

#[pyfunction]
pub fn add_elim_zeros(add: &Add) -> Option<Add> {
    let mut did_anything = false;
    let new_operands = add
        .children()
        .filter_map(|node| match &**node {
            Circuit::Scalar(scalar) => {
                if scalar.value == 0.0 {
                    did_anything = true;
                    None
                } else {
                    Some(node)
                }
            }
            _ => Some(node),
        })
        .collect();
    if !did_anything {
        None
    } else {
        let out = Add::new(new_operands, add.info().name);
        if out.shape() == add.shape() {
            Some(out)
        } else {
            None
        } // in case the scalars caused broadcasting
    }
}

#[pyfunction]
pub fn add_deduplicate(add: &Add) -> Option<Add> {
    let mut duplicate_counts: HashMap<HashBytes, i32> = HashMap::default();
    let mut hash_to_node: HashMap<HashBytes, CircuitRc> = HashMap::default();
    for node in add.children() {
        *duplicate_counts.entry(node.info().hash).or_insert(0) += 1;
        hash_to_node.insert(node.info().hash, node.clone());
    }
    if duplicate_counts.len() == add.num_children() {
        None
    } else {
        let deduped_inputs = hash_to_node
            .into_iter()
            .map(|(hash, node)| {
                let count = duplicate_counts[&hash];
                if count > 1 {
                    let node_count_name =
                        node.info().name.map(|s| format!("{}_mul_count", s).into());
                    let count_name = node
                        .info()
                        .name
                        .is_some()
                        .then(|| "count_from_dedup".into());
                    Einsum::scalar_mul(node, count as f64, node_count_name, count_name).rc()
                } else {
                    node
                }
            })
            .collect();
        Some(Add::new(deduped_inputs, add.info().name))
    }
}

#[pyfunction]
pub fn remove_add_few_input(add: &Add) -> Option<CircuitRc> {
    match add.num_children() {
        0 => Some(Scalar::new(0f64, add.info().shape.clone(), add.info().name).rc()),
        1 => Some(add.children_sl()[0].clone()),
        _ => None,
    }
}

pub fn make_einsum_ints_same_one_layer(einsum: &Einsum) -> Einsum {
    make_einsum_ints_same_one_layer_and_int_info(einsum).0
}

pub fn make_einsum_ints_same_one_layer_and_int_info(
    einsum: &Einsum,
) -> (
    Einsum,
    UnionFind,
    HashMap<u8, u8>,
    Vec<Option<HashMap<u8, u8>>>,
) {
    let mut top_int_map: HashMap<u8, u8> = HashMap::default();
    let mut next_global_int: u8 = 0;
    for i in einsum.axes_in_input() {
        if let Vacant(e) = top_int_map.entry(i) {
            e.insert(next_global_int);
            next_global_int += 1;
        }
    }
    // don't know the number of result ints, so UF for really high amount?
    let mut unionfind = UnionFind::new(300);
    let int_maps: Vec<Option<HashMap<u8, u8>>> = einsum
        .args()
        .map(|(node, ints)| match &**node {
            Circuit::Einsum(einsum) => {
                let mut bottom_int_map: HashMap<u8, u8> = HashMap::default();

                for (bottom_int, top_int) in zip(&einsum.out_axes, ints) {
                    if !bottom_int_map.contains_key(bottom_int) {
                        bottom_int_map.insert(*bottom_int, top_int_map[top_int]);
                    } else {
                        unionfind.union(
                            bottom_int_map[bottom_int] as usize,
                            top_int_map[top_int] as usize,
                        )
                    }
                }
                for bottom_int in einsum.axes_in_input() {
                    if let Vacant(e) = bottom_int_map.entry(bottom_int) {
                        e.insert(next_global_int);
                        next_global_int += 1;
                    }
                }
                Some(bottom_int_map)
            }

            _ => None,
        })
        .collect();

    let new_args = zip(einsum.args(), int_maps.clone())
        .map(|((node, ints), int_map_op)| {
            let new_operand_axes = ints
                .iter()
                .map(|x| unionfind.find(top_int_map[x] as usize) as u8)
                .collect();
            match &**node {
                Circuit::Einsum(inner) => {
                    let int_map = int_map_op.unwrap();
                    let new_args_inner = inner
                        .args()
                        .map(|(child, inner_op_ints)| {
                            (
                                child,
                                inner_op_ints
                                    .iter()
                                    .map(|i| unionfind.find(int_map[i] as usize) as u8)
                                    .collect(),
                            )
                        })
                        .collect();
                    (
                        inner
                            .evolve(
                                Some(new_args_inner),
                                Some(
                                    inner
                                        .out_axes
                                        .iter()
                                        .map(|i| unionfind.find(int_map[i] as usize) as u8)
                                        .collect(),
                                ),
                            )
                            .rc(),
                        new_operand_axes,
                    )
                }
                _ => (node.clone(), new_operand_axes),
            }
        })
        .collect();
    let ein_out = einsum.evolve(
        Some(new_args),
        Some(
            einsum
                .out_axes
                .iter()
                .map(|x| unionfind.find(top_int_map[x] as usize) as u8)
                .collect(),
        ),
    );

    (ein_out, unionfind, top_int_map, int_maps)
}

#[pyfunction]
pub fn einsum_flatten_once(einsum: &Einsum) -> Option<Einsum> {
    if einsum
        .children()
        .all(|node| !matches!(**node, Circuit::Einsum(_)))
    {
        return None;
    }
    let with_ints_same = make_einsum_ints_same_one_layer(einsum);
    Some(Einsum::new(
        with_ints_same
            .args()
            .flat_map(|(node, ints)| match &**node {
                Circuit::Einsum(einsum) => einsum.args_cloned(),
                _ => vec![(node, ints.clone())],
            })
            .collect(),
        with_ints_same.out_axes,
        None,
    ))
}

/// nests einsum (similar to "rearrange_muls") according to an einops-style path with recycled IDs
#[pyfunction]
pub fn einsum_nest_path(einsum: &Einsum, path: Vec<Vec<usize>>) -> Einsum {
    let mut stack = einsum.args_cloned();
    for contraction in path {
        let mut contraction = contraction.clone();
        contraction.sort();
        contraction.reverse();
        let mut args_here = vec![];
        for i in contraction {
            args_here.push(stack.remove(i))
        }
        let outer_indices = {
            if stack.is_empty() {
                einsum.out_axes.clone()
            } else {
                let outer_indices_set: HashSet<AxisInt> = stack
                    .iter()
                    .map(|(_node, ints)| ints)
                    .chain(std::iter::once(&einsum.out_axes))
                    .flatten()
                    .copied()
                    .collect();
                let inner_indices_set: HashSet<AxisInt> = args_here
                    .iter()
                    .flat_map(|(_node, ints)| ints)
                    .copied()
                    .collect();
                let mut outer_indices: EinsumAxes = inner_indices_set
                    .intersection(&outer_indices_set)
                    .copied()
                    .collect();
                outer_indices.as_mut_slice().sort();
                outer_indices
            }
        };
        stack.push((
            Einsum::nrc(args_here, outer_indices.clone(), None),
            outer_indices,
        ))
    }
    assert!(stack.len() == 1);
    unwrap!(arc_ref_clone(&stack[0].0), Circuit::Einsum)
}

#[pyfunction]
#[pyo3(name = "einsum_nest_optimize")]
pub fn einsum_nest_optimize_py(einsum: &Einsum) -> Result<Einsum> {
    einsum_nest_optimize(
        einsum,
        &mut OptimizationContext::new_settings(Default::default()),
    )
}

pub fn einsum_nest_optimize(einsum: &Einsum, context: &mut OptimizationContext) -> Result<Einsum> {
    // by default dont pre-contract subsets of arrayconstants
    let path = optimize_einsum_spec_cached(
        einsum.get_spec(),
        None,
        Some(context.settings.max_single_tensor_memory / einsum.info().device_dtype.size()),
        None,
        Some(filter_to_idx(einsum.children(), |x| {
            matches!(***x, Circuit::Array(_))
        })),
    )?;
    // let path = einsum.get_spec().optimize_dp(None,None,None);
    Ok(einsum_nest_path(einsum, path))
}

/// corresponds to python function `construct_diag`
pub fn expand_maybe_diag(
    node: &CircuitRc,
    in_ints: &EinsumAxes,
    out_ints: &EinsumAxes,
    int_sizes: &HashMap<u8, usize>,
) -> CircuitRc {
    if in_ints[..] == out_ints[..] {
        return node.clone();
    }

    // if no diag, just expand
    if is_unique(in_ints) && is_unique(out_ints) {
        return Rearrange::nrc_elim_identity(
            node.clone(),
            RearrangeSpec::expand(in_ints, out_ints, int_sizes).unwrap(),
            None,
        );
    }

    let mut cur_ints = in_ints.clone();
    let mut result = node.clone();

    // broadcast to have all diag dims
    let out_int_appearances = unique_to_appearance(&out_ints.iter().collect());
    let ints_to_add_before_einsum: EinsumAxes = out_int_appearances
        .iter()
        .filter(|(int, appearances)| appearances.len() > 1 && !in_ints.contains(int))
        .map(|x| **x.0)
        .collect();
    if !ints_to_add_before_einsum.is_empty() {
        cur_ints = in_ints
            .iter()
            .chain(ints_to_add_before_einsum.iter())
            .cloned()
            .collect();
        result = Rearrange::nrc_elim_identity(
            result,
            RearrangeSpec::expand(
                &(0_u8..in_ints.len() as u8).collect(),
                &(0_u8..cur_ints.len() as u8).collect(),
                &cur_ints
                    .iter()
                    .enumerate()
                    .map(|(i, int)| (i as u8, int_sizes[int]))
                    .collect(),
            )
            .unwrap(),
            None,
        )
    }

    // make diag
    let einsum_out_ints: EinsumAxes = out_ints
        .iter()
        .filter(|i| cur_ints.contains(i))
        .cloned()
        .collect();
    result = Einsum::nrc(vec![(result, cur_ints)], einsum_out_ints.clone(), None);
    cur_ints = einsum_out_ints;

    // add non-diag broadcast dims
    if cur_ints[..] != out_ints[..] {
        result = Rearrange::nrc_elim_identity(
            result,
            RearrangeSpec::expand(
                &(0_u8..out_ints.len() as u8)
                    .filter(|i| cur_ints.contains(&out_ints[*i as usize]))
                    .collect(),
                &(0_u8..out_ints.len() as u8).collect(),
                &out_ints
                    .iter()
                    .enumerate()
                    .map(|(i, int)| (i as u8, int_sizes[int]))
                    .collect(),
            )
            .unwrap(),
            None,
        );
    }

    result
}

#[pyfunction]
#[pyo3(name = "make_broadcast")]
pub fn make_broadcast_py(node: CircuitRc, out_shape: Shape) -> Option<CircuitRc> {
    match make_broadcast(&node, &out_shape) {
        Ok(c) => Some(c),
        Err(_) => None,
    }
}

pub fn make_broadcast(node: &CircuitRc, out_shape: &Shape) -> Result<CircuitRc> {
    let rank_dif = out_shape.len() - node.info().rank();
    let input_ints = node
        .info()
        .shape
        .iter()
        .enumerate()
        .map(|(i, x)| {
            if *x == 1 {
                tu8v![]
            } else {
                tu8v![(i + rank_dif) as u8]
            }
        })
        .collect();
    let output_ints: RInts = (0..(out_shape.len())).map(|i| tu8v![i as u8]).collect();
    let int_sizes = out_shape.clone();
    Ok(Rearrange::nrc_elim_identity(
        node.clone(),
        RearrangeSpec::new_canon(input_ints, output_ints, shape_to_op_shape(&int_sizes)),
        None,
    ))
}

#[pyfunction]
pub fn einsum_elim_identity(einsum: &Einsum) -> Option<CircuitRc> {
    if einsum.num_children() == 1
        && einsum.out_axes[..] == einsum.in_axes[0][..]
        && is_unique(&einsum.out_axes)
    {
        Some(einsum.children_sl()[0].clone())
    } else if einsum.num_children() == 0 {
        Some(Scalar::new(1.0, sv![], einsum.info().name).rc())
    } else {
        None
    }
}

#[pyfunction]
pub fn index_merge_scalar(index: &Index) -> Option<CircuitRc> {
    index
        .node()
        .as_scalar()
        .map(|scalar| Scalar::new(scalar.value, index.info().shape.clone(), index.info().name).rc())
}

#[pyfunction]
pub fn index_elim_identity(index: &Index) -> Option<CircuitRc> {
    if index.index.is_identity(&index.node().info().shape) {
        Some(index.node().clone())
    } else {
        None
    }
}

#[pyfunction]
pub fn index_fuse(index: &Index) -> Option<Index> {
    index.node().as_index().and_then(|inner| {
        Some(Index::new(
            inner.node().clone(),
            compose(
                &index.index,
                &inner.index.canonicalize(inner.node().shape()),
            )
            .unwrap(),
            index.info().name.map(|n| {
                if n.ends_with(" idx idx") {
                    n.strip_suffix(" idx").unwrap().into()
                } else {
                    n
                }
            }),
        ))
    })
}

#[pyfunction]
pub fn rearrange_merge_scalar(rearrange: &Rearrange) -> Option<CircuitRc> {
    rearrange.node().as_scalar().map(|scalar| {
        Scalar::new(
            scalar.value,
            rearrange.info().shape.clone(),
            rearrange.info().name,
        )
        .rc()
    })
}

#[pyfunction]
pub fn rearrange_elim_identity(rearrange: &Rearrange) -> Option<CircuitRc> {
    if rearrange.spec.is_identity() {
        Some(rearrange.node().clone())
    } else {
        None
    }
}

#[pyfunction]
pub fn concat_merge_uniform(concat: &Concat) -> Option<CircuitRc> {
    let maybe_scalars: Vec<Option<f64>> = concat
        .children()
        .map(|x| match &**x {
            Circuit::Scalar(scalar) => Some(scalar.value),
            _ => None,
        })
        .collect();
    if maybe_scalars[0].is_some() && maybe_scalars.iter().all(|x| *x == maybe_scalars[0]) {
        Some(
            Scalar::new(
                maybe_scalars[0].unwrap(),
                concat.info().shape.clone(),
                concat.info().name,
            )
            .rc(),
        )
    } else {
        None
    }
}

#[pyfunction]
pub fn concat_elim_identity(concat: &Concat) -> Option<CircuitRc> {
    if concat.num_children() == 1 {
        Some(concat.children_sl()[0].clone())
    } else {
        None
    }
}

pub fn get_removable_axes(circuit: &CircuitRc) -> HashSet<usize> {
    // plausibly this should have a flag for conforming to input shape
    match &***circuit {
        Circuit::Scalar(_scalar) => (0..circuit.info().rank()).collect(),
        Circuit::Rearrange(rearrange) => rearrange
            .spec
            .out_broadcast_axes()
            .iter()
            .copied()
            .collect(),
        _ => HashSet::default(),
    }
}

pub fn remove_axes(circuit: &CircuitRc, axes: &HashSet<usize>) -> Option<CircuitRc> {
    remove_axes_gen(circuit, axes, true)
}

pub fn remove_axes_weak(circuit: &CircuitRc, axes: &HashSet<usize>) -> Option<CircuitRc> {
    remove_axes_gen(circuit, axes, false)
}

pub fn remove_axes_gen(
    circuit: &CircuitRc,
    axes: &HashSet<usize>,
    rearrange_conform_for_remove_1: bool,
) -> Option<CircuitRc> {
    if axes.is_empty() {
        return Some(circuit.clone());
    }
    fn get_name<T: CircuitNode>(t: &T) -> Option<Name> {
        t.info().name.map(|x| format!("{}_rem_ax", x).into())
    }
    match &***circuit {
        Circuit::Scalar(scalar) => Some(
            Scalar::new(
                scalar.value,
                filter_out_idx(&scalar.info().shape, axes)
                    .into_iter()
                    .collect(),
                get_name(scalar),
            )
            .rc(),
        ),
        Circuit::Rearrange(rearrange) => {
            // canonicalize first to make input 1s (), allowing us to remove them from output
            let spec = if rearrange_conform_for_remove_1 {
                rearrange.conform_to_input_shape_spec().canonicalize(true)
            } else {
                (*rearrange.spec).clone()
            };
            Some(Rearrange::nrc_elim_identity(
                rearrange.node().clone(),
                spec.filter_out_axes(axes).unwrap(),
                get_name(rearrange),
            ))
        }
        _ => {
            if circuit
                .info()
                .shape
                .iter()
                .enumerate()
                .all(|(i, l)| !axes.contains(&i) || *l == 1)
            {
                Some(Rearrange::nrc_elim_identity(
                    circuit.clone(),
                    // wow this chained construction is long, should change at some point
                    RearrangeSpec::ident_usize(circuit.info().rank())
                        .unwrap()
                        .conform_to_input_shape(&circuit.info().shape)
                        .unwrap()
                        .canonicalize(true)
                        .filter_out_axes(axes)
                        .unwrap(),
                    get_name(circuit),
                ))
            } else {
                None
            }
        }
    }
}

#[pyfunction]
pub fn generalfunction_pull_removable_axes(node: &GeneralFunction) -> Option<CircuitRc> {
    if node.num_children() != 1 {
        return None;
    }
    let removable_axes_inp = get_removable_axes(&node.children_sl()[0]);
    let removable_batchable_inp: HashSet<usize> = removable_axes_inp
        .iter()
        .filter(|x| (**x + node.num_non_batchable_output_dims as usize) < node.info().rank())
        .cloned()
        .collect();
    if removable_batchable_inp.is_empty() {
        return None;
    }
    let new_generalfunction = GeneralFunction::new(
        vec![remove_axes(&node.children_sl()[0], &removable_batchable_inp).unwrap()],
        node.spec.clone(),
        node.info().name,
    );
    Some(Rearrange::nrc_elim_identity(
        new_generalfunction.rc(),
        RearrangeSpec::unremove_axes(&removable_batchable_inp, &node.info().shape),
        None,
    ))
}

#[pyfunction]
pub fn concat_pull_removable_axes(node: &Concat) -> Option<CircuitRc> {
    let removable_axes_per: Vec<HashSet<usize>> =
        node.children().map(|x| get_removable_axes(&x)).collect();
    let removable_axes = intersection_all(&removable_axes_per);
    let removable_non_axis: HashSet<usize> = removable_axes
        .iter()
        .filter(|x| **x != node.axis)
        .cloned()
        .collect();
    if removable_non_axis.is_empty() {
        return None;
    }
    let new_axis = node.axis
        - removable_non_axis
            .iter()
            .filter(|i| **i < node.axis)
            .count();
    let new_node = Concat::new(
        node.children()
            .map(|node| remove_axes(&node, &removable_non_axis).unwrap())
            .collect(),
        new_axis,
        node.info().name,
    );
    Some(Rearrange::nrc_elim_identity(
        new_node.rc(),
        RearrangeSpec::unremove_axes(&removable_non_axis, &node.info().shape),
        None,
    ))
}

#[pyfunction]
pub fn einsum_pull_removable_axes(einsum: &Einsum) -> Option<CircuitRc> {
    let mut did_anything = false;
    let mut new_args: EinsumArgs = einsum
        .args()
        .map(|(node, ints)| {
            let removable_axes = get_removable_axes(&node);
            if !removable_axes.is_empty() {
                did_anything = true;
                (
                    remove_axes(&node, &removable_axes).unwrap(),
                    filter_out_idx(ints, &removable_axes).into_iter().collect(),
                )
            } else {
                (node, ints.clone())
            }
        })
        .collect();
    if !did_anything {
        return None;
    }
    let ints_in_new_args: U8Set = new_args
        .iter()
        .flat_map(|(_node, ints)| ints.as_slice())
        .cloned()
        .collect();

    // multiply by product of removed reduced axes
    let shape_map = einsum.shape_map().unwrap();
    let reduced_axes = einsum.reduced_axes();
    let scalar_mul: f64 = reduced_axes
        .into_iter()
        .filter(|i| !ints_in_new_args.contains(*i))
        .map(|i| shape_map[&i] as f64)
        .product();
    if scalar_mul != 1.0 {
        new_args.push((Scalar::new(scalar_mul, sv![], None).rc(), tu8v![]));
    }

    let new_out_ints: EinsumAxes = einsum
        .out_axes
        .iter()
        .filter(|i| ints_in_new_args.contains(**i))
        .copied()
        .collect();
    let new_einsum = Einsum::nrc(new_args, new_out_ints.clone(), einsum.info().name);
    if new_out_ints[..] == einsum.out_axes[..] {
        Some(new_einsum)
    } else {
        // for now failing on diags bc simp loop
        if is_unique(&einsum.out_axes) {
            Some(expand_maybe_diag(
                &new_einsum,
                &new_out_ints,
                &einsum.out_axes,
                &shape_map,
            ))
        } else {
            None
        }
    }
}

#[pyfunction]
pub fn add_pull_removable_axes(add: &Add, remove_non_common_axes: bool) -> Option<CircuitRc> {
    let mut removed_any_non_one_axes = false;
    let removable_axes: Vec<HashSet<usize>> = add
        .nodes_and_rank_differences()
        .iter()
        .map(|(x, rank_difference)| {
            let pre_broadcast = get_removable_axes(x);
            let pre_broadcast_with_ones: HashSet<usize> = pre_broadcast
                .union(
                    &x.info()
                        .shape
                        .iter()
                        .enumerate()
                        .filter_map(|(i, l)| {
                            if *l != add.info().shape[i + rank_difference] {
                                Some(i)
                            } else {
                                None
                            }
                        })
                        .collect(),
                )
                .copied()
                .collect();
            if remove_non_common_axes
                && !removed_any_non_one_axes
                && pre_broadcast_with_ones
                    .iter()
                    .any(|i| x.info().shape[*i] != 1)
            {
                removed_any_non_one_axes = true;
            }
            pre_broadcast_with_ones
                .iter()
                .map(|i| i + rank_difference)
                .chain(0..*rank_difference)
                .collect()
        })
        .collect();

    let intersection = intersection_all(&removable_axes);
    if (intersection.is_empty() && !remove_non_common_axes)
        || (remove_non_common_axes && !removed_any_non_one_axes)
    {
        return None;
    }
    let post_rearrange_spec = RearrangeSpec::new_canon(
        (0..add.info().rank())
            .filter(|x| !intersection.contains(x))
            .map(|x| tu8v![x as u8])
            .collect(),
        (0..add.info().rank()).map(|x| tu8v![x as u8]).collect(),
        shape_to_op_shape(&add.info().shape),
    );
    let new_operands = zip(add.nodes_and_rank_differences(), removable_axes)
        .map(|((node, _rank_dif), removable_here_base)| {
            let rank_difference = add.info().rank() - node.info().rank();
            if !remove_non_common_axes {
                let removable_common_in_rank: HashSet<usize> = intersection
                    .iter()
                    .filter_map(|i| i.checked_sub(rank_difference))
                    .collect();
                remove_axes(&node, &removable_common_in_rank).unwrap()
            } else {
                let removable_here = removable_here_base
                    .iter()
                    .filter_map(|i| i.checked_sub(rank_difference))
                    .collect();
                let raw_removed = remove_axes(&node, &removable_here).unwrap();

                // let removable_here_to_add_back = removable_here.iter().filter(|i|!intersection.contains(i+rank_dif))

                let mut output_ints: RInts = sv![];
                let mut count = 0;
                let mut started = false;
                for i in 0..node.info().rank() {
                    if removable_here.contains(&i) {
                        if started && !intersection.contains(&(i + rank_difference)) {
                            output_ints.push(tu8v![]);
                        }
                    } else {
                        output_ints.push(tu8v![count]);
                        count += 1;
                        started = true;
                    }
                }

                let broadcast_rspec = RearrangeSpec::new_canon(
                    (0..raw_removed.info().rank())
                        .map(|i| tu8v![i as u8])
                        .collect(),
                    output_ints,
                    shape_to_op_shape(&raw_removed.info().shape),
                );
                Rearrange::nrc_elim_identity(raw_removed, broadcast_rspec, None)
            }
        })
        .collect();
    let add_circuit = Add::nrc(new_operands, add.info().name);
    Some(Rearrange::nrc_elim_identity(
        add_circuit,
        post_rearrange_spec,
        None,
    ))
}

#[pyfunction]
pub fn add_make_broadcasts_explicit(add: &Add) -> Option<Add> {
    let mut did_anything = false;
    let new_nodes = add
        .children()
        .map(|node| {
            if node.info().shape[..] != add.info().shape[..] {
                did_anything = true;
                make_broadcast(&node, &add.info().shape).unwrap()
            } else {
                node.clone()
            }
        })
        .collect();
    if !did_anything {
        return None;
    }
    Some(Add::new(new_nodes, add.info().name))
}

pub fn distribute_once_raw(
    einsum: &Einsum,
    operand_idx: usize,
    do_broadcasts: bool,
    prefix: Option<&str>,
    suffix: Option<&str>,
    mut call_on_sub: impl FnMut(usize, Einsum) -> Result<CircuitRc>,
) -> Result<Add> {
    let mut on_sub = |child_idx, node: CircuitRc, einsum: Einsum| -> Result<CircuitRc> {
        let einsum = if prefix.is_some() || suffix.is_some() {
            let suffix = suffix.unwrap_or("");
            let prefix = prefix.unwrap_or("");

            einsum.rename(
                node.info()
                    .name
                    .map(|n_s| format!("{prefix}{n_s}{suffix}").into()),
            )
        } else {
            einsum
        };
        call_on_sub(child_idx, einsum)
    };
    match einsum.children_sl()[operand_idx].as_add() {
        Some(add_node) => {
            let mut add = add_node.clone();
            if do_broadcasts && add.has_broadcast() {
                add = add_make_broadcasts_explicit(&add).unwrap();
            }
            if add.has_broadcast() {
                bail!(DistributeError::CouldntBroadcast { add, do_broadcasts });
            }
            if add.num_children() == 0 {
                bail!(DistributeError::EmptyAddUnsupported {
                    einsum: einsum.clone(),
                    operand_idx
                });
            }
            let summands = add
                .children()
                .enumerate()
                .map(|(i, node)| {
                    let mut new_args = einsum.args_cloned();
                    new_args[operand_idx].0 = node.clone();
                    on_sub(
                        i,
                        node.clone(),
                        Einsum::new(new_args, einsum.out_axes.clone(), None),
                    )
                })
                .collect::<Result<Vec<CircuitRc>>>()?;
            Ok(Add::new(summands, einsum.info().name))
        }
        None => bail!(DistributeError::OperandIsNotAdd {
            einsum: einsum.clone(),
            operand_idx,
        }),
    }
}

#[pyfunction]
#[pyo3(signature=(einsum, operand_idx, do_broadcasts = true, prefix = None, suffix = None))]
pub fn distribute_once(
    einsum: &Einsum,
    operand_idx: usize,
    do_broadcasts: bool,
    prefix: Option<String>,
    suffix: Option<String>,
) -> Result<Add> {
    if operand_idx >= einsum.num_children() {
        bail!(DistributeError::OperandIdxTooLarge {
            einsum: einsum.clone(),
            operand_idx,
        })
    }
    distribute_once_raw(
        einsum,
        operand_idx,
        do_broadcasts,
        prefix.as_deref(),
        suffix.as_deref(),
        |_, c| Ok(c.rc()),
    )
}

pub fn distribute_once_op(einsum: &Einsum, operand_idx: usize, do_broadcasts: bool) -> Option<Add> {
    distribute_once_raw(einsum, operand_idx, do_broadcasts, None, None, |_, c| {
        Ok(c.rc())
    })
    .ok()
}

/// distribute all Adds that are direct children of this einsum
/// not necessarily very useful?
#[pyfunction]
pub fn distribute_all(einsum: &Einsum) -> Option<Add> {
    let add_op_idxs = filter_to_idx(einsum.args(), |(node, _ints)| {
        matches!(&***node, Circuit::Add(_)) && node.num_children() > 0
    });
    if add_op_idxs.is_empty() {
        return None;
    }
    let mut distributed = distribute_once_op(einsum, add_op_idxs[0], true).unwrap();
    if add_op_idxs.len() > 1 {
        distributed = distributed
            .map_children(&mut |node: CircuitRc| match &**node {
                Circuit::Einsum(inner) => Ok(distribute_all(inner).unwrap().rc()),
                _ => {
                    panic!();
                }
            })
            .unwrap()
    }
    Some(add_flatten_once(&distributed).unwrap_or(distributed))
}

#[pyfunction]
pub fn einsum_of_permute_merge(einsum: &Einsum) -> Option<Einsum> {
    let mut did_anything = false;
    let new_args = einsum
        .args()
        .map(|(node, ints)| {
            node.as_rearrange()
                .and_then(|rearrange| {
                    rearrange.spec.get_fwd_permutation().map(|permutation| {
                        did_anything = true;
                        (
                            rearrange.node().clone(),
                            permutation.iter().map(|i| ints[*i]).collect(),
                        )
                    })
                })
                .unwrap_or_else(|| (node.clone(), ints.clone()))
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
pub fn permute_of_einsum_merge(rearrange: &Rearrange) -> Option<Einsum> {
    rearrange.spec.get_fwd_permutation().and_then(|perm| {
        rearrange.node().as_einsum().map(|einsum| {
            Einsum::new(
                einsum.args_cloned(),
                inverse_permutation(&perm)
                    .iter()
                    .map(|i| einsum.out_axes[*i])
                    .collect(),
                rearrange.info().name,
            )
        })
    })
}

#[pyfunction]
pub fn einsum_elim_zero(einsum: &Einsum) -> Option<Scalar> {
    if einsum.children().any(|x| match &**x {
        Circuit::Scalar(scalar) => scalar.value == 0.0,
        _ => false,
    }) {
        Some(Scalar::new(
            0.0,
            einsum.info().shape.clone(),
            einsum.info().name,
        ))
    } else {
        None
    }
}

pub fn einsum_merge_scalars_impl(einsum: &Einsum, only_one: bool) -> Option<Einsum> {
    let mut num_scalars_found = 0;
    let mut scalar_mul: f64 = 1.0;
    let mut new_args: Vec<(CircuitRc, EinsumAxes)> = einsum
        .args_cloned()
        .into_iter()
        .filter(|(node, _ints)| match &***node {
            Circuit::Scalar(scalar) => {
                if scalar.info().shape.is_empty() && (!only_one || scalar.value == 1.) {
                    num_scalars_found += 1;
                    scalar_mul *= scalar.value;
                    false
                } else {
                    true
                }
            }
            _ => true,
        })
        .collect();
    if num_scalars_found == 0 || (num_scalars_found == 1 && scalar_mul != 1.) {
        return None;
    }
    if scalar_mul != 1.0 {
        new_args.push((Scalar::new(scalar_mul, sv![], None).rc(), tu8v![]))
    }
    Some(Einsum::new(new_args, einsum.out_axes.clone(), None))
}

#[pyfunction]
pub fn einsum_merge_scalars(einsum: &Einsum) -> Option<Einsum> {
    einsum_merge_scalars_impl(einsum, false)
}

#[pyfunction]
pub fn einsum_remove_one(einsum: &Einsum) -> Option<Einsum> {
    einsum_merge_scalars_impl(einsum, true)
}

/// We want to turn einsum ab->b, where a is concat, to be Add [a1b->b, a2b->b...]
/// This allows us to not realize all of ab at once to save memory.
#[pyfunction]
pub fn einsum_concat_to_add(einsum: &Einsum) -> Option<Add> {
    // Which nodes contain a given einsum int, e.g. "ab,ab->" will produce {0: [0, 1], 1: [0, 1]}
    let mut nodes_by_einsum_dim = HashMap::default();
    for (concat_idx, (_circ, einsum_ints)) in einsum.args().enumerate() {
        for einsum_int in einsum_ints {
            nodes_by_einsum_dim
                .entry(einsum_int)
                .or_insert(HashSet::default())
                .insert(concat_idx);
        }
    }

    let potential_concats_and_idx: Vec<(usize, CircuitRc)> = einsum
        .args()
        .enumerate()
        .filter_map(|(concat_idx, (circ, in_axes))| {
            if let Circuit::Concat(concat) = &**circ {
                // `concat.axis` is in terms of this particular node, but we need the einsum dimension, so
                // look it up, e.g. Einsum([0,1],[2,3] -> [0,3]), if the second input is a concat with
                // concat.axis=1, needs to pull out 3.
                let concat_dim = in_axes[concat.axis];
                // If this axis appears multiple times in `in_axes`, we're doing the magic diagonal thing and this
                // reduction won't work without more specialization.
                if in_axes.iter().filter(|ax| **ax == concat_dim).count() > 1 {
                    return None;
                }
                // If this input int appears in multiple nodes' input string, we can't use it
                // TODO: Unless we get smarter and confirm the other nodes' concat shapes, or do some splitting, etc.
                if nodes_by_einsum_dim[&concat_dim].len() > 1 {
                    return None;
                }
                // This transformation only works if we're summing across the concat dimension.
                if !einsum.out_axes.contains(&(concat_dim as u8)) {
                    return Some((concat_idx.clone(), circ));
                }
            }
            None
        })
        .collect();
    if potential_concats_and_idx.is_empty() {
        return None;
    }

    // TODO: Maybe make this work across multiple Concat nodes in an Einsum.
    let (concat_idx, concat_node) = &potential_concats_and_idx[0];

    let inner_einsum: Vec<CircuitRc> = concat_node
        .children()
        .map(|node| {
            let mut new_args = einsum.args_cloned();
            // Replace the concatted `a` dimension with the partial `a` we're mapping over, then
            // leave everything else in the output alone.
            new_args[*concat_idx] = (node.clone(), einsum.in_axes[*concat_idx].clone());
            Einsum::nrc(new_args, einsum.out_axes.clone(), None)
        })
        .collect();
    Some(Add::new(inner_einsum, None))
}

#[pyfunction]
pub fn index_split_axes(node: &Index, top_axes: HashSet<usize>) -> Option<Index> {
    let mut bottom: Vec<TensorAxisIndex> = vec![];
    let mut top: Vec<TensorAxisIndex> = vec![];
    if top_axes.iter().any(|i| *i >= node.node().info().rank()) {
        return None;
    }
    for (i, idx) in node.index.0.iter().enumerate() {
        if top_axes.contains(&i) {
            bottom.push(TensorAxisIndex::Slice(Slice {
                start: None,
                stop: None,
            }));
            top.push(idx.clone());
        } else {
            bottom.push(idx.clone());
            match idx {
                TensorAxisIndex::Single(_single) => {}
                _ => top.push(TensorAxisIndex::Slice(Slice {
                    start: None,
                    stop: None,
                })),
            }
        }
    }
    Some(Index::new(
        Index::nrc(node.node(), TensorIndex(bottom), None),
        TensorIndex(top),
        None,
    ))
}

#[pyfunction]
pub fn rearrange_fuse(node: &Rearrange) -> Option<Rearrange> {
    node.node().as_rearrange().and_then(|inner| {
        RearrangeSpec::fuse(
            &inner.conform_to_input_shape_spec().canonicalize(true),
            &node.conform_to_input_shape_spec().canonicalize(true),
        )
        .map(|spec| Rearrange::new(inner.node(), spec.canonicalize(true), node.info().name))
        .ok()
    })
}

pub fn axis_index_split_sections_for_concat(
    axis_index: &TensorAxisIndex,
    sections: &Vec<usize>,
) -> Result<Vec<Option<TensorAxisIndex>>> {
    let starts = cumsum(sections);
    let len = sections.iter().sum();
    let out = match axis_index {
        TensorAxisIndex::Single(single) => {
            let single = check_canon_idxs(len, &[*single])
                .context("out of bounds when checking index for concat push down index")?[0];
            let outer_idx = starts.iter().take_while(|start| **start <= single).count() - 1;
            let inner_idx = single - starts[outer_idx];
            let mut result = vec![None; sections.len()];
            result[outer_idx] = Some(TensorAxisIndex::Single(inner_idx as i64));
            result
        }
        &TensorAxisIndex::Slice(slice) => {
            let uslice: USlice = slice.to_uslice(len);
            let mut result = vec![None; sections.len()];
            for (i, (start, sec_len)) in zip(starts.iter(), sections.iter()).enumerate() {
                let stop = start + sec_len;
                if *start < uslice.stop
                    && stop > uslice.start
                    && ((*start != stop) || (*start < uslice.stop - 1))
                {
                    result[i] = Some(TensorAxisIndex::new_plain_slice(
                        uslice.start.saturating_sub(*start),
                        std::cmp::min(*sec_len, uslice.stop - start),
                    ))
                }
            }
            if uslice.start == uslice.stop && result.iter().all(|x| x.is_none()) {
                if uslice.start == 0 {
                    result[0] = Some(TensorAxisIndex::Slice(uslice.into()))
                } else if uslice.stop == len {
                    let start = uslice.start.saturating_sub(*starts.last().unwrap());
                    *result.last_mut().unwrap() =
                        Some(TensorAxisIndex::new_plain_slice(start, start))
                } else {
                    // at boundary between concattands
                    bail!(PushDownIndexError::ZeroLengthIndexAmbiguous {})
                }
            }
            if sections.len() == 1 && result[0] == None {
                result[0] = Some(TensorAxisIndex::Slice(uslice.into()))
            }
            result
        }
        TensorAxisIndex::Tensor(tensor) => {
            bail!(PushDownIndexError::ConcatSplitSectionsTensorUnsupported {
                tensor: tensor.clone()
            })
        }
    };
    Ok(out)
}

pub fn push_down_index_raw(
    node: &Index,
    allow_partial_pushdown: bool,
    call_on_sub: &mut dyn FnMut(usize, Index) -> Result<CircuitRc>,
    suffix: Option<String>,
) -> Result<CircuitRc> {
    let out_name = node.node().info().name.map(|n_s| {
        suffix
            .as_ref()
            .map(|s| format!("{n_s}{s}").into())
            .unwrap_or(n_s)
    });
    let mut on_sub = |child_idx,
                      node: CircuitRc,
                      index,
                      force_name: Option<Option<Name>>|
     -> Result<CircuitRc> {
        let new_name: Option<Name> = force_name.unwrap_or_else(|| {
            suffix
                .as_ref()
                .and_then(|s| node.info().name.map(|n_s| format!("{n_s}_idx_{s}").into()))
        });
        call_on_sub(child_idx, Index::new(node, index, new_name))
    };
    let idx_partial_name = || {
        suffix.as_ref().and_then(|s| {
            node.node()
                .info()
                .name
                .map(|inner_name| format!("{inner_name}_partial_{s}").into())
        })
    };

    let handle_non_batchable_remaining = |rank_to_pass, new_circ: CircuitRc| {
        let passed_rank_now = rank_to_pass
            - (0..rank_to_pass)
                .filter(|i| matches!(node.index.0[*i], TensorAxisIndex::Single(_)))
                .count();
        let top_axis_indices: Vec<_> = node.index.0[rank_to_pass..].iter().cloned().collect();
        if top_axis_indices
            .iter()
            .zip(&new_circ.shape()[passed_rank_now..])
            .all(|(x, s)| x.is_identity(*s))
        {
            Ok(new_circ)
        } else if allow_partial_pushdown {
            let index_top = TensorIndex(
                vec![TensorAxisIndex::IDENT; passed_rank_now]
                    .iter()
                    .cloned()
                    .chain(top_axis_indices)
                    .collect(),
            );
            Ok(Index::nrc(new_circ, index_top, idx_partial_name()))
        } else {
            Err((passed_rank_now, top_axis_indices))
        }
    };

    // TODO: maybe handle symbolic axes? Possibilities:
    // Check we don't index into sum of symbolic axes
    // Handle rearrange case.
    let out: CircuitRc = match &**node.node() {
        Circuit::Add(inner) => {
            let new_operands = inner
                .nodes_and_rank_differences()
                .iter()
                .enumerate()
                .map(|(i, (operand, rank_difference))| {
                    let index_here = (0..operand.info().rank())
                        .map(|i| {
                            let idx_here = node.index.0[i + rank_difference].clone();
                            if operand.info().shape[i] == inner.info().shape[i + rank_difference] {
                                idx_here
                            } else {
                                match idx_here {
                                    TensorAxisIndex::Single(_s) => TensorAxisIndex::Single(0),
                                    _ => TensorAxisIndex::IDENT,
                                }
                            }
                        })
                        .collect();
                    on_sub(i, operand.clone(), TensorIndex(index_here), None)
                })
                .collect::<Result<_>>()?;
            Add::nrc(new_operands, out_name)
        }
        Circuit::Concat(inner) => {
            let index_non_axis = TensorIndex(
                node.index
                    .0
                    .iter()
                    .enumerate()
                    .map(|(i, idx)| {
                        if i == inner.axis {
                            TensorAxisIndex::IDENT
                        } else {
                            idx.clone()
                        }
                    })
                    .collect(),
            );
            let sections = &inner.get_sizes_at_axis();
            let old_index_on_concat_axis = node.index.0[inner.axis].clone();
            let concat_axis_indices_maybe =
                axis_index_split_sections_for_concat(&old_index_on_concat_axis, sections);
            let concat_axis_indices_maybe = if allow_partial_pushdown {
                if let Err(Some(PushDownIndexError::ConcatSplitSectionsTensorUnsupported {
                    ..
                })) = &concat_axis_indices_maybe
                    .as_ref()
                    .map_err(|e| e.downcast_ref())
                {
                    None
                } else {
                    Some(concat_axis_indices_maybe?)
                }
            } else {
                Some(concat_axis_indices_maybe?)
            };
            if index_non_axis.is_identity(&inner.info().shape)
                && concat_axis_indices_maybe.is_none()
            {
                bail!(PushDownIndexError::NoopOnConcat {
                    index_node: node.clone(),
                    index_non_axis,
                    index: node.index.clone()
                })
            }
            // todo: drop concat parts that get indexed out
            let new_axis = inner.axis
                - node.index.0[..inner.axis]
                    .iter()
                    .filter(|idx| matches!(idx, TensorAxisIndex::Single(_)))
                    .count();
            if let Some(concat_indices) = concat_axis_indices_maybe {
                if matches!(old_index_on_concat_axis, TensorAxisIndex::Single(_)) {
                    let mut index = index_non_axis;
                    for (i, ax) in concat_indices.iter().enumerate() {
                        // should only be once match
                        if let Some(z) = ax {
                            index.0[inner.axis] = z.clone();
                            return on_sub(
                                i,
                                inner.children_sl()[i].clone(),
                                index,
                                Some(out_name),
                            );
                        }
                    }
                    unreachable!()
                } else {
                    Concat::nrc(
                        zip(inner.children(), concat_indices)
                            .enumerate()
                            .filter_map(|(i, (child, idx))| {
                                idx.map(|idx| {
                                    let mut index = index_non_axis.clone();
                                    index.0[inner.axis] = idx;
                                    on_sub(i, child.clone(), index, None)
                                })
                            })
                            .collect::<Result<_>>()?,
                        new_axis,
                        out_name,
                    )
                }
            } else {
                let new_concat = Concat::nrc(
                    inner
                        .children()
                        .enumerate()
                        .map(|(i, operand)| on_sub(i, operand, index_non_axis.clone(), None))
                        .collect::<Result<_>>()?,
                    new_axis,
                    out_name,
                );
                let final_index = TensorIndex::new_single(
                    node.index.0[inner.axis].clone(),
                    new_axis,
                    new_concat.info().rank(),
                );
                Index::nrc(new_concat, final_index, idx_partial_name())
            }
        }
        Circuit::GeneralFunction(inner) => {
            if !inner.is_batchable()
                || inner.num_non_batchable_output_dims as usize == inner.info().rank()
            {
                bail!(PushDownIndexError::NoopOnGeneralFunction {
                    index_node: node.clone(),
                    is_batchable: inner.is_batchable(),
                    num_non_batch_out: inner.num_non_batchable_output_dims,
                    inner_rank: inner.rank()
                })
            } else {
                let rank_to_pass =
                    inner.info().rank() - inner.num_non_batchable_output_dims as usize;
                let new_operands = zip(inner.children(), &inner.input_batchability)
                    .enumerate()
                    .map(|(i, (child, &batchable))| {
                        if batchable {
                            let ident_rank = child.info().rank() - rank_to_pass;
                            let index_passed = TensorIndex(
                                node.index.0[..rank_to_pass]
                                    .iter()
                                    .cloned()
                                    .chain(vec![TensorAxisIndex::IDENT; ident_rank])
                                    .collect(),
                            );
                            on_sub(i, child, index_passed, None)
                        } else {
                            Ok(child)
                        }
                    })
                    .collect::<Result<_>>()?;
                let new_gf =
                    GeneralFunction::nrc(new_operands, inner.spec.clone(), out_name.clone());

                handle_non_batchable_remaining(rank_to_pass, new_gf).map_err(
                    |(_, top_axis_indices)| {
                        PushDownIndexError::GeneralFunctionSomeAxesNotPossible {
                            index_node: node.clone(),
                            is_batchable: inner.is_batchable(),
                            num_non_batch_out: inner.num_non_batchable_output_dims,
                            inner_rank: inner.rank(),
                            top_axis_indices,
                        }
                    },
                )?
            }
        }
        Circuit::Einsum(inner) => {
            let mut int_indices: HashMap<u8, Option<TensorAxisIndex>> = HashMap::default();
            for (idx, int) in zip(&node.index.0, &inner.out_axes) {
                match int_indices.get(int) {
                    Some(_prev_idx) => {
                        // todo: equality on tensoraxisindex so we can pass if same
                        int_indices.insert(*int, None);
                    }
                    None => {
                        int_indices.insert(*int, Some(idx.clone()));
                    }
                }
            }
            let ints_filter_index = |ints: &EinsumAxes| -> EinsumAxes {
                ints.iter()
                    .filter_map(|i| {
                        match int_indices
                            .get(i)
                            .cloned()
                            .flatten()
                            .unwrap_or(TensorAxisIndex::IDENT)
                        {
                            TensorAxisIndex::Single(_) => None,
                            _ => Some(*i),
                        }
                    })
                    .collect()
            };
            let mut any_operand_index_nontrivial = false;
            let new_operands: Vec<(CircuitRc, EinsumAxes)> = inner
                .args()
                .enumerate()
                .map(|(i, (node, ints))| {
                    let index_here = TensorIndex(
                        ints.iter()
                            .map(|i| {
                                int_indices
                                    .get(i)
                                    .cloned()
                                    .flatten()
                                    .unwrap_or(TensorAxisIndex::IDENT)
                            })
                            .collect(),
                    );
                    if !index_here.is_identity(&node.info().shape) {
                        any_operand_index_nontrivial = true;
                    }
                    Ok((
                        on_sub(i, node, index_here, None)?,
                        ints_filter_index(&ints), // todo make function work with sv
                    ))
                })
                .collect::<Result<_>>()?;
            if !any_operand_index_nontrivial {
                bail!(PushDownIndexError::EinsumNoop {
                    index_node: node.clone()
                });
            }
            let new_einsum = Einsum::nrc(
                new_operands,
                ints_filter_index(&inner.out_axes),
                out_name.clone(),
            );
            let out_index = TensorIndex(
                zip(&inner.out_axes, &node.index.0)
                    .filter_map(|(i, idx)| match int_indices.get(i).unwrap() {
                        None => Some(idx.clone()),
                        Some(int_idx) => match int_idx {
                            TensorAxisIndex::Single(_) => None,
                            _ => Some(TensorAxisIndex::IDENT),
                        },
                    })
                    .collect(),
            );
            if out_index.is_identity(new_einsum.shape()) {
                new_einsum
            } else if allow_partial_pushdown {
                Index::nrc(new_einsum, out_index, idx_partial_name())
            } else {
                bail!(PushDownIndexError::EinsumSomeAxesNotPossible {
                    index_node: node.clone(),
                    out_index,
                    out_axes: inner.out_axes.clone(),
                    int_indices
                })
            }
        }
        Circuit::Rearrange(inner) => {
            let (index, spec): (TensorIndex, RearrangeSpec) = if let Some(x) = inner
                .conform_to_input_shape_spec()
                .canonicalize(true)
                .index_before(
                    node.index.canonicalize(&node.node().shape()),
                    node.info().device_dtype.clone().unwrap_or_defaults().device,
                ) {
                x
            } else {
                bail!(PushDownIndexError::RearrangeNotPossible {
                    index_node: node.clone(),
                    spec: (*inner.spec).clone()
                })
            };
            let new_index = on_sub(0, inner.node().clone(), index, None)?;
            Rearrange::nrc_elim_identity(new_index, spec, out_name)
        }
        Circuit::Scatter(inner) => {
            // for now it just pushes indexes of Single(0) through axes size 1
            let mut new_outer = vec![];
            let mut new_scatter = vec![];
            let mut new_scatter_shape: Shape = sv![];
            let mut did_anything = false;
            let passed_index = zip(
                &node.index.0,
                zip(inner.index.all_uslices().unwrap(), &inner.info().shape),
            )
            .map(|(top_index, (bottom_scatter, l))| {
                if let TensorAxisIndex::Single(single) = top_index {
                    if bottom_scatter == (USlice { start: 0, stop: 1 }) && *single == 0 {
                        did_anything = true;
                        return top_index.clone();
                    }
                }
                new_outer.push(top_index.clone());
                new_scatter.push(bottom_scatter);
                new_scatter_shape.push(*l);
                TensorAxisIndex::IDENT
            })
            .collect();
            if !did_anything {
                bail!(PushDownIndexError::ScatterNoop {
                    index_node: node.clone()
                });
            }
            let inner_node = on_sub(0, inner.node().clone(), TensorIndex(passed_index), None)?;

            let new_scatter = Scatter::nrc(
                inner_node,
                TensorIndex(new_scatter.iter().cloned().map(USlice::into).collect()),
                new_scatter_shape,
                out_name,
            );

            let out_index = TensorIndex(new_outer);

            if out_index.is_identity(new_scatter.shape()) {
                new_scatter
            } else if allow_partial_pushdown {
                Index::nrc(new_scatter, out_index, idx_partial_name())
            } else {
                bail!(PushDownIndexError::ScatterSomeAxesNotPossible {
                    index_node: node.clone(),
                    out_index,
                })
            }
        }
        Circuit::Tag(tag) => Tag::nrc(
            on_sub(0, tag.node().clone(), node.index.clone(), None)?,
            tag.uuid,
            out_name,
        ),
        Circuit::DiscreteVar(var) => DiscreteVar::nrc(
            on_sub(
                0,
                var.values().clone(),
                TensorIndex(
                    iter::once(TensorAxisIndex::IDENT) // skip sample axis
                        .chain(node.index.0.iter().cloned())
                        .collect(),
                ),
                None,
            )?,
            var.probs_and_group().clone(),
            out_name,
        ),
        Circuit::Scalar(_) => index_merge_scalar(node).unwrap(),
        Circuit::Module(m) => {
            // doesn't push anything into spec,
            let batch_shape = m.aligned_batch_shape();
            let rank_to_pass = batch_shape.len();

            let new_nodes = m
                .args()
                .zip(&m.spec.arg_specs)
                .enumerate()
                .map(|(i, (input_node, arg_spec))| {
                    let this_num_batch = input_node.rank() - arg_spec.symbol.rank();
                    assert!(this_num_batch <= rank_to_pass);
                    let start = rank_to_pass - this_num_batch;
                    let index_passed = TensorIndex(
                        node.index.0[start..rank_to_pass]
                            .iter()
                            .cloned()
                            .chain(vec![TensorAxisIndex::IDENT; arg_spec.symbol.rank()])
                            .collect(),
                    );
                    on_sub(i + 1, input_node.clone(), index_passed, None)
                })
                .collect::<Result<_>>()?;
            let new_mod = Module::nrc(new_nodes, m.spec.clone(), out_name.clone());
            handle_non_batchable_remaining(rank_to_pass, new_mod).map_err(
                |(_, top_axis_indices)| PushDownIndexError::ModuleSomeAxesNotPossible {
                    index_node: node.clone(),
                    num_non_batch_out: m.rank() - rank_to_pass,
                    inner_rank: m.rank(),
                    top_axis_indices,
                },
            )?
        }

        // Circuit::Cumulant(cum) => Cumulant::nrc(
        //     cum.dims_for_self().map(...)
        //     out_name,
        // ),
        Circuit::Index(_) => bail!(Error::from(PushDownIndexError::ThroughIndex {
            index_node: node.clone(),
            circ_type: CircuitType::Index,
        })),
        inner => bail!(PushDownIndexError::UnimplementedType {
            index_node: node.clone(),
            circ_type: inner.type_tag()
        }),
    };

    Ok(out)
}

#[pyfunction]
pub fn push_down_index_once(
    node: Index,
    allow_partial_pushdown: Option<bool>,
    suffix: Option<String>,
) -> Result<CircuitRc> {
    push_down_index_raw(
        &node,
        allow_partial_pushdown.unwrap_or(false),
        &mut |_, x| Ok(x.rc()),
        suffix,
    )
}

pub fn push_down_index_op(node: &Index) -> Option<CircuitRc> {
    push_down_index_raw(node, true, &mut |_, x| Ok(x.rc()), None).ok()
}

#[pyfunction]
pub fn concat_repeat_to_rearrange(concat: &Concat) -> Option<Concat> {
    if concat.num_children() == 0 {
        return None;
    }
    let mut prev_node: CircuitRc = concat.children_sl()[0].clone();
    let mut same_seen = 1;
    let mut did_anything = false;
    let mut new_nodes: Vec<CircuitRc> = vec![];

    // add unique additional node to the end of the chain so that we have node != &prev_node at the end
    let extra_unique_node = Symbol::new_with_random_uuid(concat.info().shape.clone(), None).rc();

    for node in concat
        .children()
        .dropping(1)
        .chain(std::iter::once(extra_unique_node))
    {
        if node != prev_node {
            if same_seen > 1 {
                did_anything = true;
                let mut int_sizes = prev_node.info().shape.clone();
                int_sizes.push(same_seen);
                new_nodes.push(Rearrange::nrc_elim_identity(
                    prev_node.clone(),
                    RearrangeSpec::new(
                        (0..prev_node.info().rank())
                            .map(|x| tu8v![x as u8])
                            .collect(),
                        (0..prev_node.info().rank())
                            .map(|x| {
                                if x == concat.axis {
                                    // create group which is (count, axis_size) for total dim size of
                                    // count * axis_size()
                                    tu8v![prev_node.info().rank() as u8, x as u8]
                                } else {
                                    tu8v![x as u8]
                                }
                            })
                            .collect(),
                        shape_to_op_shape(&int_sizes),
                    )
                    .unwrap(),
                    None,
                ));
            } else {
                new_nodes.push(prev_node.clone())
            }
            same_seen = 1;
        } else {
            same_seen += 1
        }
        prev_node = node.clone();
    }
    if !did_anything {
        return None;
    }
    Some(Concat::new(new_nodes, concat.axis, concat.info().name))
}

#[test]
fn basic_check_same_concat_shape() {
    let a = Symbol::new_with_random_uuid(sv![2, 3, 4, 5, 6], None).rc();
    let out =
        concat_repeat_to_rearrange(&Concat::new(vec![a.clone(), a.clone(), a], 3, None)).unwrap();
    assert_eq!(out.info().shape[3], 15)
}

#[inline]
pub fn is_not_outer(a: BitMask128, b: BitMask128) -> bool {
    let union = a & b;
    union == a || union == b
}

/// this takes bitmasks of axes and produces bitmasks of indices into argument bitmasks that each don't have any outer products
pub fn bitmask_outer_product_sets(bitmasks: &Vec<BitMask128>) -> Vec<BitMask128> {
    let mut result: Vec<BitMask128> = vec![];
    let mut result_axes: Vec<BitMask128> = vec![];
    for (input_idx, bm) in bitmasks.iter().enumerate() {
        let mut did_anything = false;
        let mut result_index_out: usize = 0;
        for (result_index, result_ax) in result_axes.iter().enumerate() {
            if is_not_outer(*bm, *result_ax) {
                result[result_index] |= 1 << input_idx;
                did_anything = true;
                result_index_out = result_index;
                break;
            }
        }
        if !did_anything {
            result.push(1 << input_idx);
            result_axes.push(*bm);
        } else {
            result_axes[result_index_out] |= bm;
        }
    }
    result
}

#[test]
fn test_bitmask_outer_product_sets() {
    let ex: Vec<u128> = vec![1 + 4, 2 + 4, 4, 1];
    dbg!(&ex);
    dbg!(bitmask_outer_product_sets(&ex));
    dbg!(bitmask_outer_product_sets(&ex)
        .iter()
        .map(|x| BitIter::from(*x).collect())
        .collect::<Vec<Vec<usize>>>());
}

#[pyfunction]
pub fn add_outer_product_broadcasts_on_top(add: &Add) -> Option<Add> {
    let axis_sets: Vec<BitMask128> = add
        .nodes_and_rank_differences()
        .iter()
        .map(|(child, rank_dif)| {
            let mut bitmask: BitMask128 = 0;
            for (i, l) in child.info().shape.iter().enumerate() {
                if *l != 1 {
                    bitmask |= 1 << (rank_dif + i)
                }
            }
            bitmask
        })
        .collect();
    let outer_product_sets = bitmask_outer_product_sets(&axis_sets);
    if outer_product_sets.len() <= 1 {
        return None;
    }
    let new_operands = outer_product_sets
        .iter()
        .map(|operand_set_bitmask| {
            Add::nrc(
                BitIter::from(*operand_set_bitmask)
                    .map(|i| add.children_sl()[i].clone())
                    .collect(),
                None,
            )
        })
        .collect();
    Some(Add::new(new_operands, None))
}

/// returns None if sub isn't subset
#[pyfunction]
pub fn extract_add(add: &Add, sub: &Add) -> Option<Add> {
    let mut counts = add.to_counts();

    for item in sub.children() {
        match counts.entry(item) {
            Entry::Occupied(mut entry) => {
                let new = entry.get().saturating_sub(1);
                if new == 0 {
                    entry.remove_entry();
                } else {
                    *entry.get_mut() = new;
                }
            }
            Entry::Vacant(_) => return None,
        }
    }
    *counts.entry(sub.crc()).or_insert(0) += 1;

    Some(Add::try_from_counts(&counts, add.info().name).unwrap())
}

/// De-duplicate children of Add nodes that are the same except for a scalar product
#[pyfunction]
pub fn add_fuse_scalar_multiples(add: &Add) -> Option<Add> {
    if !add.children().any(|x| x.as_einsum().is_some()) {
        return None;
    }

    let mut reduced_any = 0;

    let scalars_and_children: Vec<(Einsum, f64, Option<Name>)> = add
        .children()
        .map(|c| match &**c {
            Circuit::Einsum(e) => {
                let mut scalar: f64 = 1.0;
                let mut new_name: Option<Name> = None;
                let no_scalar_children: EinsumArgs = e
                    .args_cloned().into_iter()
                    .filter(|(child, _)| {
                        if let Some(x) = child.as_scalar() && x.info().shape.len() == 0  {
                            scalar *= x.value;
                            reduced_any += 1;
                            if x.info().name.is_some() {
                                if let Some(new_name_string) = &new_name {
                                    new_name = Some(
                                       ( new_name_string.string() + " + " + &x.info().name.unwrap()).into()
                                    );
                                } else {
                                    new_name = x.info().name;
                                }
                            }
                            false
                        }else{
                            true
                        }
                    })
                    .collect();

                (
                    Einsum::new(no_scalar_children, e.out_axes.clone(), e.info().name),
                    scalar,
                    new_name,
                )
            }
            Circuit::Scalar(s) => {
                if s.info().shape.len() == 0 {
                    (Einsum::empty(None), s.value, s.info().name)
                } else {
                    (Einsum::identity(c, None), 1.0, None)
                }
            }
            _ => (Einsum::identity(c, None), 1.0, None),
        })
        .collect();

    if reduced_any <= 1 {
        return None;
    }

    let mut hm: HashMap<Einsum, (f64, Option<Name>)> = HashMap::default();
    for (einsum, scalar, name) in scalars_and_children {
        let (hm_val, hm_name) = hm.entry(einsum).or_insert((0.0, None));
        *hm_val += scalar;
        if hm_name.is_none() {
            *hm_name = name;
        }
    }

    if hm.len() == add.num_children() {
        return None;
    }

    let nodes: Vec<CircuitRc> = hm
        .into_iter()
        .map(|(einsum, (scalar, scalar_name))| {
            if scalar == 1.0 {
                einsum.rc()
            } else {
                let name = einsum.info().name;
                let out_axes = einsum.out_axes.clone();
                let mut v = einsum.args_cloned();
                v.push((Scalar::new(scalar, sv![], scalar_name).rc(), tu8v![]));
                Einsum::nrc(v, out_axes, name)
            }
        })
        .collect();

    Some(Add::new(nodes, add.info().name))
}

#[pyfunction]
pub fn einsum_permute_to_rearrange(einsum: &Einsum) -> Option<Rearrange> {
    if einsum.num_children() == 1
        && einsum.out_axes.len() > 0
        && einsum.in_axes[0].len() == einsum.out_axes.len()
        && is_unique(&einsum.out_axes)
        && is_unique(&einsum.in_axes[0])
    {
        return Some(Rearrange::new(
            einsum.children_sl()[0].clone(),
            RearrangeSpec::new(
                einsum.in_axes[0].iter().map(|z| tu8v![*z]).collect(),
                einsum.out_axes.iter().map(|z| tu8v![*z]).collect(),
                sv![OpSize::NONE;(*einsum.out_axes.iter().max().unwrap()+1) as usize],
            )
            .unwrap()
            .canonicalize(false),
            einsum.info().name,
        ));
    }
    None
}

const MODULE_AXIS_NO_MSG: &str = concat!(
    "Some axes would be pushed into spec circuit, so aren't possible.\n",
    "You could substitute the module if you want to push into the spec circuit."
);

#[apply(python_error_exception)]
#[base_error_name(PushDownIndex)]
#[base_exception(PyValueError)]
#[derive(Error, Debug, Clone)]
pub enum PushDownIndexError {
    #[error("index_node={index_node:?} index_non_axis={index_non_axis} index={index} ({e_name})")]
    NoopOnConcat {
        index_node: Index,
        index_non_axis: TensorIndex,
        index: TensorIndex,
    },
    #[error("index_node={index_node:?} is_batchable={is_batchable} num_non_batch_out={num_non_batch_out} inner_rank={inner_rank} ({e_name})")]
    NoopOnGeneralFunction {
        index_node: Index,
        is_batchable: bool,
        num_non_batch_out: u8,
        inner_rank: usize,
    },
    #[error("index_node={index_node:?} is_batchable={is_batchable} num_non_batch_out={num_non_batch_out} inner_rank={inner_rank} top_axis_indices={top_axis_indices:?} ({e_name})")]
    GeneralFunctionSomeAxesNotPossible {
        index_node: Index,
        is_batchable: bool,
        num_non_batch_out: u8,
        inner_rank: usize,
        top_axis_indices: Vec<TensorAxisIndex>,
    },
    #[error("{MODULE_AXIS_NO_MSG}\nindex_node={index_node:?} num_non_batch_out={num_non_batch_out} inner_rank={inner_rank} top_axis_indices={top_axis_indices:?} ({e_name})")]
    ModuleSomeAxesNotPossible {
        index_node: Index,
        num_non_batch_out: usize,
        inner_rank: usize,
        top_axis_indices: Vec<TensorAxisIndex>,
    },
    // TODO: improve errors as needed
    #[error("index_node={index_node:?} ({e_name})")]
    EinsumNoop { index_node: Index },
    #[error("index_node={index_node:?} out_index={out_index:?} out_axes={out_axes:?} int_indices={int_indices:?} ({e_name})")]
    EinsumSomeAxesNotPossible {
        index_node: Index,
        out_index: TensorIndex,
        out_axes: EinsumAxes,
        int_indices: HashMap<u8, Option<TensorAxisIndex>>,
    },
    #[error("index_node={index_node:?} spec={spec:?} ({e_name})")]
    RearrangeNotPossible {
        index_node: Index,
        spec: RearrangeSpec,
    },
    #[error("index_node={index_node:?} ({e_name})")]
    ScatterNoop { index_node: Index },
    #[error("index_node={index_node:?} out_index={out_index:?} ({e_name})")]
    ScatterSomeAxesNotPossible {
        index_node: Index,
        out_index: TensorIndex,
        // TODO: more
    },
    #[error("can't push down index through other index, but you can use 'index_fuse'\nindex_node={index_node:?} circ_type={circ_type:?} ({e_name})")]
    ThroughIndex {
        index_node: Index,
        circ_type: CircuitType,
    },
    #[error("index_node={index_node:?} circ_type={circ_type:?} ({e_name})")]
    UnimplementedType {
        index_node: Index,
        circ_type: CircuitType,
    },

    #[error("tensors aren't currently supported tensor={tensor:?} ({e_name})")]
    ConcatSplitSectionsTensorUnsupported { tensor: IndexTensor },

    #[error("zero-length index and no unique choice ({e_name})")]
    ZeroLengthIndexAmbiguous {},
}

#[apply(python_error_exception)]
#[base_error_name(Distribute)]
#[base_exception(PyValueError)]
#[derive(Error, Debug, Clone)]
pub enum DistributeError {
    #[error("add={add:?} do_broadcasts={do_broadcasts} ({e_name})")]
    CouldntBroadcast { add: Add, do_broadcasts: bool },
    #[error("einsum={einsum:?} operand_idx={operand_idx} ({e_name})")]
    OperandIsNotAdd { einsum: Einsum, operand_idx: usize },
    #[error("einsum={einsum:?} operand_idx={operand_idx} ({e_name})")]
    OperandIdxTooLarge { einsum: Einsum, operand_idx: usize },
    #[error("einsum={einsum:?} operand_idx={operand_idx} ({e_name})")]
    EmptyAddUnsupported { einsum: Einsum, operand_idx: usize },
    #[error("einsum={einsum:?} operand_idx={operand_idx} ({e_name})")]
    Noop { einsum: Einsum, operand_idx: usize },
}
