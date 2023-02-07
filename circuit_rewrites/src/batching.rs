use std::collections::BTreeMap;

use anyhow::{bail, Context, Result};
use circuit_base::{
    deep_map_op, deep_map_unwrap,
    named_axes::{axis_of_name, get_axis_leaves, set_named_axes},
    prelude::*,
    visit_circuit_unwrap, visit_circuit_with_parents, Concat, Einsum, Index, Module, ModuleArgSpec,
    ModuleSpec, Symbol,
};
use macro_rules_attribute::apply;
use pyo3::{exceptions::PyValueError, prelude::*};
use rr_util::{
    python_error_exception,
    set_cover::set_cover,
    tensor_util::{TensorAxisIndex, TensorIndex},
    util::EinsumAxes,
};
use thiserror::Error;

use crate::{
    algebraic_rewrite::einsum_nest_optimize,
    circuit_optimizer::{OptimizationContext, OptimizationSettings},
};

#[pyfunction]
#[pyo3(signature=(circuit, axis, batch_size, keep_non_axis_leaves = false, allow_partial_batch = true))]
pub fn batch_to_concat(
    circuit: CircuitRc,
    axis: usize,
    batch_size: usize,
    keep_non_axis_leaves: bool, // THIS IS A HACK
    allow_partial_batch: bool,
) -> Result<CircuitRc> {
    if axis >= circuit.info().shape.len() {
        bail!("batch_to_concat on nonexistent axis");
    }
    let l = circuit.info().shape[axis];
    if batch_size == 0 {
        bail!(BatchError::BatchSizeIsZero {});
    }
    if l % batch_size != 0 && !allow_partial_batch {
        bail!(BatchError::PartialBatchNotAllowed {});
    }
    let circuit = deep_map_unwrap(circuit, |x| {
        x.as_index()
            .map(|x| Index::slice_edges_to_none(x).rc())
            .unwrap_or(x.clone())
    });

    let (name, circuit, leaves_axis, leaves_non_axis) = get_axis_leaves(circuit, axis)?;
    let leaves_non_axis = if keep_non_axis_leaves {
        Default::default()
    } else {
        leaves_non_axis
    };

    let input_specs_axis = leaves_axis
        .iter()
        .map(|(sub, i)| {
            Ok((
                sub.clone(),
                ModuleArgSpec {
                    symbol: set_named_axes(
                        Symbol::new_with_random_uuid(sub.info().shape.clone(), None),
                        BTreeMap::from([(*i as u8, name.clone())]),
                    )
                    .context("Failed extracting named axis from old name.")?,
                    ban_non_symbolic_size_expand: false, // I think this needs to be false?
                    ..Default::default()
                },
            ))
        })
        .collect::<Result<Vec<(CircuitRc, ModuleArgSpec)>>>()?;
    let input_specs: Vec<(CircuitRc, ModuleArgSpec)> = input_specs_axis
        .iter()
        .cloned()
        .chain(leaves_non_axis.iter().map(|sub| {
            (
                sub.clone(),
                ModuleArgSpec {
                    symbol: Symbol::new_with_random_uuid(sub.info().shape.clone(), None),
                    ban_non_symbolic_size_expand: false, // I think this needs to be false?
                    ..Default::default()
                },
            )
        }))
        .collect();
    if input_specs.is_empty() {
        bail!(BatchError::AxisOriginatesTooHigh {});
    }
    let check_all_inputs_used = true;
    let module_spec = ModuleSpec::new_extract(
        circuit.clone(),
        input_specs.clone(),
        check_all_inputs_used,
        false,
    )
    .unwrap();

    let mk_module_spec = |axis_size: usize| -> Result<ModuleSpec> {
        module_spec.resize(
            module_spec
                .arg_specs
                .iter()
                .map(|inp_spec| {
                    if let Some((c, _is)) = input_specs_axis
                        .iter()
                        .find(|(_c, spec)| spec.symbol.uuid == inp_spec.symbol.uuid)
                    {
                        let mut result = c.info().shape.clone();
                        result[leaves_axis[c]] = axis_size;
                        return result;
                    }
                    inp_spec.symbol.info().shape.clone()
                })
                .collect(),
        )
    };
    let mk_args = |index: TensorAxisIndex| -> Vec<CircuitRc> {
        module_spec
            .arg_specs
            .iter()
            .map(|inp_spec| {
                if let Some((c, _is)) = input_specs_axis
                    .iter()
                    .find(|z| z.1.symbol.uuid == inp_spec.symbol.uuid)
                {
                    return Index::nrc(
                        c.clone(),
                        TensorIndex::new_single(index.clone(), leaves_axis[c], c.info().rank()),
                        None,
                    );
                }
                input_specs
                    .iter()
                    .find(|(_c, s)| s.symbol.uuid == inp_spec.symbol.uuid)
                    .unwrap()
                    .0
                    .clone()
            })
            .collect()
    };
    let module_spec_main = mk_module_spec(batch_size)?;
    let num_even_batches = l / batch_size;
    let mut concattands: Vec<CircuitRc> = (0..num_even_batches)
        .map(|i| {
            Module::nrc(
                mk_args(TensorAxisIndex::new_plain_slice(
                    i * batch_size,
                    (i + 1) * batch_size,
                )),
                module_spec_main.clone(),
                None,
            )
        })
        .collect();
    if l % batch_size != 0 {
        concattands.push(Module::nrc(
            mk_args(TensorAxisIndex::new_plain_slice(
                num_even_batches * batch_size,
                l,
            )),
            mk_module_spec(l % batch_size)?,
            None,
        ));
    }
    let result = Concat::nrc(concattands, axis, None);
    if result.info().shape != circuit.info().shape {
        println!(
            "shapes not equal {:?} {:?}",
            result.info().shape,
            circuit.info().shape
        );
        println!("old");
        circuit.printu();
        println!("new");
        result.printu();
        panic!();
    }
    Ok(result)
}

#[apply(python_error_exception)]
#[base_error_name(Batch)]
#[base_exception(PyValueError)]
#[derive(Error, Debug, Clone)]
pub enum BatchError {
    #[error("Would need to batch multiple axes, only supports one ({e_name})")]
    RequiresMultipleAxes {},

    #[error("Batching axis originates too high ({e_name})")]
    AxisOriginatesTooHigh {},

    #[error("batch_size must be greater than 0 ({e_name})")]
    BatchSizeIsZero {},

    #[error("axis not evenly dividable by batch_size and allow_partial_batch is False ({e_name})")]
    PartialBatchNotAllowed {},
}

#[pyfunction]
#[pyo3(name = "batch_einsum")]
pub fn batch_einsum_py(einsum: Einsum, settings: OptimizationSettings) -> Result<CircuitRc> {
    batch_einsum(&einsum, &mut OptimizationContext::new_settings(settings))
}

pub fn batch_einsum(einsum: &Einsum, context: &mut OptimizationContext) -> Result<CircuitRc> {
    let max_single_mem =
        context.settings.max_single_tensor_memory / einsum.info().device_dtype.size();
    assert!(einsum.info().numel_usize() <= max_single_mem);

    let mut no_memory_limit_context: OptimizationContext = Default::default();
    no_memory_limit_context.settings.max_single_tensor_memory = usize::MAX;
    // use named axes to record the ints used in intermediate einsums
    let einsum_args_named_axes: Vec<(CircuitRc, EinsumAxes)> = einsum
        .args()
        .enumerate()
        .map(|(i, (c, ints))| {
            (
                set_named_axes(
                    Symbol::new_with_random_uuid(
                        c.info().shape.clone(),
                        Some(i.to_string().into()),
                    ),
                    ints.iter()
                        .enumerate()
                        .map(|(i, int)| (i as u8, int.to_string().into()))
                        .collect(),
                )
                .unwrap() // u8 don't have spaces
                .rc(),
                ints.clone(),
            )
        })
        .collect();
    let einsum_named_axes = Einsum::try_new(
        einsum_args_named_axes.clone(),
        einsum.out_axes.clone(),
        einsum.info().name,
    )
    .unwrap();
    let nested_big = einsum_nest_optimize(&einsum_named_axes, &mut no_memory_limit_context)?;

    let mut intermediates_above_mem = vec![];
    visit_circuit_unwrap(nested_big.crc(), |c| {
        if c.info().numel_usize() > max_single_mem {
            intermediates_above_mem.push(c);
        }
    });
    assert!(!intermediates_above_mem.is_empty());
    // get the set cover of axes to batch
    let mut covers: Vec<u128> = vec![0; 64];
    for (i, intermediate) in intermediates_above_mem.iter().enumerate() {
        for (_, v) in &intermediate.info().named_axes {
            let int = v.parse::<usize>().unwrap();
            let v = covers[int] | 1 << i;
            covers[int] = v
        }
    }
    let ints_needed = set_cover(&covers);
    if ints_needed.len() != 1 {
        bail!(BatchError::RequiresMultipleAxes {});
    }

    let batch_int = ints_needed[0];
    let batch_str = batch_int.to_string().into();

    let batch_int_len = einsum.shape_map().unwrap()[&(batch_int as u8)];
    let max_interm_mem = intermediates_above_mem
        .iter()
        .map(|x| x.info().numel_usize())
        .max()
        .unwrap();
    let mut batch_size = ((max_interm_mem as f64) / (max_single_mem as f64)) as usize;
    while batch_int_len % batch_size != 0 {
        batch_size -= 1;
    }

    let mut batchheads: Vec<CircuitRc> = vec![];
    visit_circuit_with_parents(nested_big.crc(), |x, parents| {
        if x.info().named_axes.values().any(|n| n == &batch_str)
            && !parents
                .iter()
                .any(|p| p.info().named_axes.values().any(|n| n == &batch_str))
        {
            batchheads.push(x);
        }
    });
    let batchheads_from: Vec<CircuitRc> = batchheads
        .iter()
        .map(|c| {
            batch_to_concat(
                c.clone(),
                axis_of_name(c, batch_str).unwrap(),
                batch_size,
                false,
                false,
            )
            .unwrap()
        })
        .collect();
    let out_still_symbols = deep_map_op(nested_big.crc(), |c| {
        if let Some(p) = batchheads
            .iter()
            .position(|bh| c.info().hash == bh.info().hash)
        {
            return Some(batchheads_from[p].clone());
        }
        None
    })
    .unwrap();
    let out = deep_map_op(out_still_symbols, |c| {
        if let Some(sym) = c.as_symbol() && let Some(p) = einsum_args_named_axes
            .iter()
            .position(|(s, _ints)| sym.uuid==s.as_symbol().unwrap().uuid)
        {
            return Some(einsum.children_sl()[p].clone());
        }
        None
    })
    .unwrap();
    Ok(out)
}
