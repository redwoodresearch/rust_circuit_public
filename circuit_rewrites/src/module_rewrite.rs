use std::iter::zip;

use anyhow::{anyhow, bail, Context, Result};
use circuit_base::{
    circuit_utils::count_nodes, deep_map_op_context, CircuitNode, CircuitRc, Concat, Module,
    ModuleArgSpec, ModuleSpec, Rearrange,
};
use pyo3::prelude::*;
use rr_util::{
    name::Name, py_types::PyOpAtAxes, rearrange_spec::RearrangeSpec,
    tensor_util::right_align_shapes, util::HashBytes,
};
use rustc_hash::FxHashMap as HashMap;

use crate::{circuit_manipulation::replace_nodes_py, concat_rewrite::split_to_concat};

#[pyfunction]
pub fn elim_empty_module(module: &Module) -> Option<CircuitRc> {
    (count_nodes(module.spec.circuit.clone()) == 1).then(|| module.substitute(None, None))
}

#[pyfunction]
pub fn elim_no_input_module(module: &Module) -> Option<CircuitRc> {
    module
        .args_slice()
        .is_empty()
        .then(|| module.spec.circuit.clone())
}

#[pyfunction]
#[pyo3(name = "deep_module_remove_unused_inputs", signature=(
    circuit,
    add_suffix_on_remove_unused = true,
    use_elim_no_input_module = true,
    elim_empty = false
))]
pub fn py_deep_module_remove_unused_inputs(
    circuit: CircuitRc,
    add_suffix_on_remove_unused: bool,
    use_elim_no_input_module: bool,
    elim_empty: bool,
) -> CircuitRc {
    deep_module_remove_unused_inputs(
        circuit,
        &mut Default::default(),
        add_suffix_on_remove_unused,
        use_elim_no_input_module,
        elim_empty,
    )
}

pub fn deep_module_remove_unused_inputs(
    circuit: CircuitRc,
    cache: &mut HashMap<HashBytes, Option<CircuitRc>>,
    add_suffix_on_remove_unused: bool,
    use_elim_no_input_module: bool,
    elim_empty: bool,
) -> CircuitRc {
    deep_map_op_context(
        circuit.clone(),
        &|c, _| {
            c.as_module().and_then(|m| {
                if elim_empty && let Some(eem) = elim_empty_module(m) {
                    return Some(eem);
                }
                module_remove_unused_inputs(
                    m,
                    add_suffix_on_remove_unused,
                    use_elim_no_input_module,
                )
                .ok()
                // ok case is rank overflow, rare.
            })
        },
        &mut (),
        cache,
    )
    .unwrap_or(circuit)
}

pub fn module_remove_unused_inputs_op(
    m: &Module,
    add_suffix_on_remove_unused: bool,
    use_elim_no_input_module: bool,
) -> Result<Option<CircuitRc>> {
    let used_inputs = m.spec.are_args_used();

    if used_inputs.iter().all(|x| *x) {
        if use_elim_no_input_module {
            return Ok(elim_no_input_module(m));
        }
        return Ok(None);
    }

    let (nodes, arg_specs) = m
        .arg_items()
        .into_iter()
        .zip(used_inputs)
        .filter_map(|(x, used)| used.then_some(x))
        .unzip();

    let m_out = Module::new(
        nodes,
        ModuleSpec {
            circuit: m.spec.circuit.clone(),
            arg_specs,
        },
        m.info().name.map(|s| {
            if add_suffix_on_remove_unused {
                format!("{} rem_unused", s).into()
            } else {
                s.into()
            }
        }),
    );

    let out_batch_shape = m_out.aligned_batch_shape();
    let orig_batch_shape = m.aligned_batch_shape();
    right_align_shapes(&[&out_batch_shape[..], &orig_batch_shape[..]]).unwrap();
    assert!(out_batch_shape.len() <= orig_batch_shape.len());
    let out = if use_elim_no_input_module {
        elim_no_input_module(&m_out).unwrap_or_else(|| m_out.rc())
    } else {
        m_out.rc()
    };
    let out = if out_batch_shape.len() < orig_batch_shape.len() {
        let spec = RearrangeSpec::prepend_batch_shape(
            orig_batch_shape[..orig_batch_shape.len() - out_batch_shape.len()]
                .iter()
                .cloned()
                .collect(),
            out.rank(),
        )
        .context("rank error in module remove unused")?;
        Rearrange::nrc(out, spec, m.info().name)
    } else {
        out
    };

    Ok(Some(out))
}

#[pyfunction]
#[pyo3(signature=(m, add_suffix_on_remove_unused = true, use_elim_no_input_module = true))]
pub fn module_remove_unused_inputs(
    m: &Module,
    add_suffix_on_remove_unused: bool,
    use_elim_no_input_module: bool,
) -> Result<CircuitRc> {
    Ok(
        module_remove_unused_inputs_op(m, add_suffix_on_remove_unused, use_elim_no_input_module)?
            .unwrap_or_else(|| m.crc()),
    )
}

// not a rewrite, doesn't keep shape!
pub fn module_strip_args(m: &Module) -> Module {
    let used_inputs = m.spec.are_args_used();
    let (nodes, arg_specs) = m
        .arg_items()
        .into_iter()
        .zip(used_inputs)
        .filter_map(|(x, used)| used.then_some(x))
        .unzip();

    Module::new(
        nodes,
        ModuleSpec {
            circuit: m.spec.circuit.clone(),
            arg_specs,
        },
        m.info().name,
    )
}

#[pyfunction]
#[pyo3(signature=(
    circuit,
    input_specs,
    prefix_to_strip = None,
    module_name = None,
    check_all_inputs_used = true,
    check_unique_arg_names = true
))]
pub fn extract_rewrite_raw(
    circuit: CircuitRc,
    input_specs: Vec<(CircuitRc, ModuleArgSpec)>,
    prefix_to_strip: Option<String>,
    module_name: Option<Name>,
    check_all_inputs_used: bool,
    check_unique_arg_names: bool,
) -> Result<Module> {
    let mut spec = ModuleSpec::new_extract(
        circuit,
        input_specs.clone(),
        check_all_inputs_used,
        check_unique_arg_names,
    )?;
    if let Some(pref) = &prefix_to_strip {
        spec = spec.map_circuit_unwrap(|x| {
            if let Some(name) = x.info().name {
                x.clone().rename(Some(
                    name.strip_prefix(pref)
                        .map(|z| Name::new(z))
                        .unwrap_or(name),
                ))
            } else {
                x
            }
        })
    }
    Module::try_new(
        spec.arg_specs
            .iter()
            .map(|x| input_specs.iter().find(|z| &z.1 == x).unwrap().0.clone())
            .collect(),
        spec,
        module_name,
    )
}

// right now this only handles batch rank 0 or 1, todo handle arbitrary batch rank
#[pyfunction]
pub fn fuse_concat_modules(
    circuit: CircuitRc,
    modules: Vec<Module>,
    name: Option<Name>,
) -> Result<CircuitRc> {
    if modules.is_empty() {
        bail!(anyhow!("no modules specified"))
    }
    let spec = modules[0].spec.clone();
    if !modules.iter().all(|x| x.spec == spec) {
        bail!(anyhow!("modules_one_batch not all same"))
    }
    let pure_rank = spec.circuit.info().rank();
    let modules_one_batch = modules
        .iter()
        .map(|x| {
            let dif = x.info().rank() - spec.circuit.info().rank();
            match dif {
                0 => Ok(Module::new(
                    zip(x.args(), &x.spec.arg_specs)
                        .map(|(n, s)| {
                            if s.batchable {
                                n.unsqueeze(vec![0], None).unwrap().rc()
                            } else {
                                n.clone()
                            }
                        })
                        .collect(),
                    spec.clone(),
                    x.info().name,
                )),
                1 => Ok(x.clone()),
                _ => Err(anyhow!("module has batch rank > 1")),
            }
        })
        .collect::<Result<Vec<_>>>()?;
    if !modules_one_batch[0]
        .spec
        .arg_specs
        .iter()
        .enumerate()
        .all(|(i, x)| {
            x.batchable
                || modules_one_batch
                    .iter()
                    .all(|node| node.args_slice()[i] == modules_one_batch[0].args_slice()[i])
        })
    {
        bail!(anyhow!(
            "all inputs must be batchable or have the same child on all inputs"
        ));
    }
    let concatted_input = Module::nrc(
        (0..spec.arg_specs.len())
            .map(|i| {
                if spec.arg_specs[i].batchable {
                    Concat::nrc(
                        modules_one_batch
                            .iter()
                            .map(|m| m.args_slice()[i].clone())
                            .collect(),
                        0,
                        None,
                    )
                } else {
                    modules_one_batch[0].args_slice()[i].clone()
                }
            })
            .collect(),
        spec.clone(),
        name,
    );
    let splitted_modules: Vec<CircuitRc> = split_to_concat(
        concatted_input,
        0,
        modules_one_batch
            .iter()
            .map(|x| x.info().shape[0])
            .collect(),
    )
    .children()
    .collect();
    let splitted_unsqueezed: Vec<CircuitRc> = (0..modules.len())
        .map(|i| {
            if modules[i].info().rank() == pure_rank {
                splitted_modules[i]
                    .squeeze(PyOpAtAxes::Single(0), None)
                    .unwrap()
                    .rc()
            } else {
                splitted_modules[i].clone()
            }
        })
        .collect();
    Ok(replace_nodes_py(
        circuit,
        zip(
            modules.into_iter().map(|x| x.rc()),
            splitted_unsqueezed.into_iter().map(|x| x.rc()),
        )
        .collect(),
    ))
}
