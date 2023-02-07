use std::{collections::HashSet, iter::zip};

use anyhow::{bail, Result};
use circuit_base::{
    cumulant::{dim_permutation_for_circuits, partitions},
    Add, CircResult, Circuit, CircuitNode, CircuitRc, Concat, Cumulant, Einsum, Index, Rearrange,
    Scalar, Tag,
};
use circuit_rewrites::algebraic_rewrite::add_make_broadcasts_explicit;
use get_update_node::IterativeMatcher;
use itertools::Itertools;
use macro_rules_attribute::apply;
use pyo3::{exceptions::PyValueError, prelude::*};
use rr_util::{
    name::Name,
    python_error_exception,
    rearrange_spec::{OpSize, RInts, RearrangeSpec},
    sv,
    tensor_util::{Slice, TensorAxisIndex, TensorIndex},
    tu8v,
};
use thiserror::Error;

use crate::nest::einsum_flatten;

/// namer is a callable Python function fn(cumulant) -> name
/// on_sub_cumulant_fn is a callable Python function fn(cumulant) -> circuit
#[pyfunction]
#[pyo3(signature=(cumulant, node, namer = None, on_sub_cumulant_fn = None))]
pub fn rewrite_cum_to_circuit_of_cum(
    cumulant: Cumulant,
    node: CircuitRc,
    namer: Option<PyObject>,
    on_sub_cumulant_fn: Option<PyObject>,
) -> CircResult {
    if cumulant.num_children() == 0 {
        return Ok(Scalar::nrc(1., sv![], cumulant.info().name));
    }

    let on_sub_cumulant = get_on_sub_cumulant(&namer, &on_sub_cumulant_fn);

    let (node_position, start_axis) = find_position_and_axis_of_start(&cumulant, &node)?;

    let original_cumulant_name = cumulant.info().name;

    let r = match &**node {
        Circuit::Rearrange(n) => {
            let spec = n.spec.clone();
            let nb_axis_before = start_axis as u8;
            let nb_axis_after = (cumulant.rank() - start_axis - node.rank()) as u8;
            let new_spec_axis = spec.next_axis();
            let before_axis: RInts = (new_spec_axis..(new_spec_axis + nb_axis_before))
                .map(|i| tu8v![i as u8])
                .collect();
            let after_axis: RInts = ((new_spec_axis + nb_axis_before)
                ..(new_spec_axis + nb_axis_before + nb_axis_after))
                .map(|i| tu8v![i as u8])
                .collect();
            let new_input_ints = [
                before_axis.clone(),
                spec.clone().input_ints,
                after_axis.clone(),
            ]
            .concat()
            .into();
            let new_output_ints = [
                before_axis.clone(),
                spec.clone().output_ints,
                after_axis.clone(),
            ]
            .concat()
            .into();
            let new_int_sizes = spec
                .int_sizes
                .clone()
                .into_iter()
                .chain((0..(nb_axis_after + nb_axis_after)).map(|_| OpSize::NONE))
                .collect();

            let new_spec = RearrangeSpec::new(new_input_ints, new_output_ints, new_int_sizes)?;
            Rearrange::nrc(
                on_sub_cumulant(replace_node(cumulant, n.node().clone(), node_position))?,
                new_spec,
                original_cumulant_name,
            )
        }
        Circuit::Index(n) => {
            let mut new_index: Vec<TensorAxisIndex> = (0..start_axis)
                .map(|_| TensorAxisIndex::Slice(Slice::IDENT))
                .collect();
            new_index.append(&mut n.index.clone().0);
            Index::nrc(
                on_sub_cumulant(replace_node(cumulant, n.node().clone(), node_position))?,
                TensorIndex(new_index),
                original_cumulant_name,
            )
        }
        Circuit::Tag(n) => Tag::nrc(
            on_sub_cumulant(replace_node(cumulant, n.node().clone(), node_position))?,
            n.uuid,
            original_cumulant_name,
        ),
        Circuit::Concat(n) => {
            let axis = start_axis + n.axis;
            Concat::nrc(
                n.children()
                    .map(|c| replace_node(cumulant.clone(), c.clone(), node_position))
                    .map(on_sub_cumulant)
                    .collect::<Result<Vec<CircuitRc>>>()?,
                axis,
                original_cumulant_name,
            )
        }
        Circuit::Add(n) => {
            // Might create rearranges WITHIN the new cumulant when some node is broadcasted.
            let explicit_add = add_make_broadcasts_explicit(n).unwrap_or(n.clone()); // If None, means NOOP

            Add::nrc(
                explicit_add
                    .children()
                    .map(|c| replace_node(cumulant.clone(), c, node_position))
                    .map(on_sub_cumulant)
                    .collect::<Result<Vec<CircuitRc>>>()?,
                original_cumulant_name,
            )
        }
        Circuit::Einsum(_) => cum_of_einsum_to_einsum_of_cum_raw(
            cumulant,
            node_position,
            start_axis,
            false,
            on_sub_cumulant,
        )?,
        Circuit::Scatter(_)
        | Circuit::Conv(_)
        | Circuit::Module(_)
        | Circuit::SetSymbolicShape(_)
        | Circuit::DiscreteVar(_)
        | Circuit::StoredCumulantVar(_) => {
            bail!(CumulantRewriteError::UnimplementedType { node })
        }
        Circuit::Array(_) | Circuit::Scalar(_) | Circuit::Cumulant(_) => {
            if cumulant.num_children() == 1 {
                node.rename(cumulant.info().name).rc()
            } else {
                Scalar::nrc(0., cumulant.shape().clone(), cumulant.info().name)
            }
        }
        Circuit::GeneralFunction(_) | Circuit::Symbol(_) => {
            bail!(CumulantRewriteError::UnexpandableType { node })
        }
    };

    Ok(r)
}

fn replace_node(cumulant: Cumulant, node: CircuitRc, position: usize) -> Cumulant {
    let new_name = cumulant.info().name;
    let mut new_nodes = cumulant.children_sl().to_vec();
    new_nodes[position] = node;
    Cumulant::new(new_nodes, new_name)
}

fn cum_of_einsum_to_einsum_of_cum_raw(
    cumulant: Cumulant,
    position: usize,
    start_axis: usize,
    just_fully_touching_partitions: bool,
    on_sub_cumulant: impl Fn(Cumulant) -> CircResult,
) -> CircResult {
    let einsum = cumulant.children_sl()[position].as_einsum_unwrap();

    let other_nodes: Vec<CircuitRc> = cumulant
        .children()
        .enumerate()
        .filter(|(i, _)| *i != position)
        .map(|(_, c)| c)
        .collect();
    let other_rank = cumulant.rank() - einsum.rank();
    let other_nums: Vec<u8> =
        (einsum.next_axis()..(einsum.next_axis() + other_rank as u8)).collect();
    let out_nums = [
        &other_nums[0..start_axis],
        &einsum.out_axes,
        &other_nums[start_axis..(other_nums.len())],
    ]
    .concat();

    let count = einsum.num_children() + other_nodes.len();

    let mut to_sum: Vec<CircuitRc> = vec![];

    for p_other in partition_into_blocks(
        zip(einsum.num_children()..count, other_nodes.clone()).collect(),
        0,
        einsum.num_children(),
    ) {
        for p_mine in partition_into_blocks(
            einsum.children().enumerate().collect(),
            p_other.len(),
            if just_fully_touching_partitions {
                p_other.len()
            } else {
                2usize.pow(31)
            }, // Lowered from 63 to avoid crash on 32 bits machines
        ) {
            for mine_perm in (0..(p_mine.len())).permutations(p_other.len()) {
                debug_assert!(mine_perm.len() == p_other.len());
                let mut outside_perm: Vec<usize> = (0..(p_mine.len()))
                    .collect::<HashSet<usize>>()
                    .difference(&mine_perm.clone().into_iter().collect::<HashSet<usize>>())
                    .map(|i| i.clone())
                    .collect::<Vec<usize>>();
                outside_perm.sort_unstable();

                debug_assert!(
                    !(just_fully_touching_partitions
                        && other_nodes.len() > 0
                        && outside_perm.len() > 0)
                );

                let fused_per: Vec<Vec<(usize, CircuitRc)>> = [
                    zip(mine_perm.clone(), p_other.clone())
                        .map(|(idx_mine, b_other)| {
                            [p_mine[idx_mine].clone(), b_other.clone()].concat()
                        })
                        .collect::<Vec<Vec<(usize, CircuitRc)>>>(),
                    outside_perm
                        .iter()
                        .map(|other_idx_mine| p_mine[*other_idx_mine].clone())
                        .collect(),
                ]
                .concat();

                let (prod, _new_cums) = kappa_term(fused_per, &on_sub_cumulant)?;

                let operated_on_prod = Einsum::try_new(
                    vec![(
                        prod.rc(),
                        einsum.in_axes.iter().flatten().cloned().collect(),
                    )],
                    out_nums.clone().into(),
                    Some("perm".into()), /* name=Cumulant.get_name(*new_cums, getter=namer) + " perm",  # naming to ensure no duplicates */
                )?; // TODO: better name

                to_sum.push(
                    einsum_flatten(operated_on_prod, IterativeMatcher::noop_traversal().rc())?.rc(),
                );
            }
        }
    }

    if to_sum.len() == 0 {
        Ok(Scalar::nrc(
            0.,
            cumulant.shape().clone(),
            cumulant.info().name,
        ))
    } else {
        Ok(Add::nrc(to_sum, cumulant.info().name))
    }
}

fn partition_into_blocks<T>(ns: Vec<T>, min_blocks: usize, max_blocks: usize) -> Vec<Vec<Vec<T>>>
where
    T: Clone,
{
    // Could be improved by making this return an iterable, but it makes memory management harder...
    partitions(ns.len())
        .collect::<Vec<Vec<Vec<usize>>>>()
        .iter()
        .filter(|p| min_blocks <= p.len() && p.len() <= max_blocks)
        .map(|p| {
            p.iter()
                .map(|s| s.iter().map(|i| ns[*i].clone()).collect())
                .collect()
        })
        .collect()
}

fn kappa_term(
    args: Vec<Vec<(usize, CircuitRc)>>,
    on_sub_cumulant: impl Fn(Cumulant) -> CircResult,
) -> Result<(Einsum, Vec<Cumulant>)> {
    let count = args.clone().concat().len();
    let new_cums: Vec<Cumulant> = args
        .iter()
        .map(|node_list| Cumulant::new(node_list.iter().map(|(_, c)| c.clone()).collect(), None))
        .collect();
    let new_cums_sub = new_cums
        .iter()
        .map(|c| on_sub_cumulant(c.clone()))
        .collect::<Result<Vec<CircuitRc>>>()?;

    Ok((
        Einsum::new_outer_product(
            new_cums_sub,
            None,
            Some(dim_permutation_for_circuits(
                args,
                new_cums.iter().map(|c| c.children_sl()).collect(),
                count,
            )),
        ),
        new_cums,
    ))
}

#[pyfunction]
#[pyo3(name = "kappa_term", signature=(args, namer = None, on_sub_cumulant_fn = None))]
pub fn kappa_term_py(
    args: Vec<Vec<(usize, CircuitRc)>>,
    namer: Option<PyObject>,
    on_sub_cumulant_fn: Option<PyObject>,
) -> Result<(Einsum, Vec<Cumulant>)> {
    let on_sub_cumulant = get_on_sub_cumulant(&namer, &on_sub_cumulant_fn);

    kappa_term(args, on_sub_cumulant)
}

fn get_on_sub_cumulant<'a>(
    namer: &'a Option<PyObject>,
    on_sub_cumulant_fn: &'a Option<PyObject>,
) -> Box<dyn Fn(Cumulant) -> CircResult + 'a> {
    Box::new(move |c: Cumulant| -> CircResult {
        let renamed_cumulant = match namer {
            Some(namer_fn) => {
                let cumulant_new_name: Option<Name> =
                    Python::with_gil(|py| -> Result<Option<Name>> {
                        Ok(namer_fn
                            .call(py, (c.clone(),), None)?
                            .extract::<Option<String>>(py)?
                            .map(|z| z.into()))
                    })?;
                c.rename(cumulant_new_name)
            }
            None => c, // Keep the name generated by autonaming
        };

        if let Some(f) = on_sub_cumulant_fn {
            Python::with_gil(|py| {
                Ok(f.call(py, (renamed_cumulant,), None)?
                    .extract::<CircuitRc>(py)?)
            })
        } else {
            Ok(renamed_cumulant.rc())
        }
    })
}

fn find_position_and_axis_of_start(
    cumulant: &Cumulant,
    node: &CircuitRc,
) -> Result<(usize, usize)> {
    let mut node_position = -1;
    let mut start_axis = 0;
    for i in 0..cumulant.num_children() {
        if cumulant.children_sl()[i] == *node {
            node_position = i as i32;
            break;
        } else {
            start_axis += cumulant.children_sl()[i].rank();
        }
    }
    if node_position == -1 {
        bail!(CumulantRewriteError::NodeToExpandNotFound {
            cumulant: cumulant.clone(),
            node: node.clone()
        })
    }
    let node_position = node_position as usize;
    Ok((node_position, start_axis))
}

#[apply(python_error_exception)]
#[base_error_name(CumulantRewrite)]
#[base_exception(PyValueError)]
#[derive(Error, Debug, Clone)]
pub enum CumulantRewriteError {
    #[error("cumulant={cumulant:?} node={node:?} ({e_name})")]
    NodeToExpandNotFound { cumulant: Cumulant, node: CircuitRc },
    #[error("node={node:?} ({e_name})")]
    UnexpandableType { node: CircuitRc },
    #[error("node={node:?} ({e_name})")]
    UnimplementedType { node: CircuitRc },
}
