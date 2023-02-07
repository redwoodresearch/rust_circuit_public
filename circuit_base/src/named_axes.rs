use std::collections::BTreeMap;

use anyhow::{bail, Result};
use macro_rules_attribute::apply;
use pyo3::exceptions::PyRuntimeError;
use rr_util::{
    name::Name,
    python_error_exception,
    util::{is_unique, HashBytes, NamedAxes},
};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use thiserror::Error;
use uuid::Uuid;

use crate::{circuit_utils::deep_map_pass_down_branching, deep_map_unwrap, prelude::*, Module};

pub fn set_named_axes<T: CircuitNode>(node: T, named_axes: NamedAxes) -> Result<T> {
    for name in named_axes.values() {
        if name
            .chars()
            .any(|c| c.is_whitespace() || c == ']' || c == '[' || c == ':')
        {
            bail!(NamedAxesError::ForbiddenCharacter {
                name: name.string()
            })
        }
    }
    Ok(node.update_info(|i| i.named_axes = named_axes).unwrap())
}
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(name = "set_named_axes")]
pub fn set_named_axes_py(circuit: CircuitRc, named_axes: NamedAxes) -> Result<CircuitRc> {
    set_named_axes(circuit, named_axes)
}

pub fn merge_named_axes(a: &NamedAxes, b: &NamedAxes) -> NamedAxes {
    let mut result = a.clone();
    result.extend(b.clone());
    result
}

pub fn named_axes_backward<T: CircuitNode + Clone>(
    circuit: &T,
    named_axes: &NamedAxes,
) -> Vec<NamedAxes> {
    circuit
        .child_axis_map()
        .iter()
        .map(|z| {
            z.iter()
                .enumerate()
                .filter_map(|(child_i, ax)| {
                    ax.and_then(|i| {
                        named_axes
                            .get(&(i as u8))
                            .map(|name| (child_i as u8, name.clone()))
                    })
                })
                .collect()
        })
        .collect()
}

#[pyfunction]
pub fn propagate_named_axes(
    circuit: CircuitRc,
    named_axes: NamedAxes,
    abort_on_branch: bool,
) -> CircuitRc {
    deep_map_pass_down_branching(
        circuit,
        |circuit, axes| {
            let child_axis_names: Vec<NamedAxes> = if abort_on_branch
                && circuit
                    .as_einsum()
                    .map(|x| x.args().any(|(_, i)| !is_unique(&i)))
                    .unwrap_or(false)
            {
                circuit
                    .children()
                    .map(|_c| Default::default())
                    .collect::<Vec<NamedAxes>>()
            } else {
                named_axes_backward(&**circuit, axes)
            };
            child_axis_names
        },
        |circuit, named_axes, children| {
            set_named_axes(
                circuit.map_children_unwrap_idxs(|i| children[i].clone()),
                merge_named_axes(&circuit.info().named_axes, named_axes),
            )
            .unwrap() // Ok because inputs don't have spaces
            .rc()
        },
        named_axes,
    )
}

pub fn axis_of_name(circuit: &Circuit, name: Name) -> Option<usize> {
    circuit
        .info()
        .named_axes
        .iter()
        .find(|(_i, s)| **s == name)
        .map(|(i, _s)| *i as usize)
}

#[pyfunction]
/// returns tuple (leavs with axis, leavs without axis)
pub fn get_axis_leaves(
    circuit: CircuitRc,
    axis: usize,
) -> Result<(
    Name,
    CircuitRc,
    HashMap<CircuitRc, usize>,
    HashSet<CircuitRc>,
)> {
    let name = Uuid::new_v4().to_string().into();
    if circuit
        .as_einsum()
        .map(|x| {
            x.out_axes
                .iter()
                .filter(|i| x.out_axes[axis] == **i)
                .count()
                != 1
        })
        .unwrap_or(false)
    {
        return Ok((
            name,
            circuit.clone(),
            HashMap::from_iter([(circuit, axis)]),
            HashSet::default(),
        ));
    }
    let circ_named_axes = propagate_named_axes(circuit, BTreeMap::from([(axis as u8, name)]), true);

    let mut result_axis: HashMap<CircuitRc, usize> = Default::default();
    let mut result_no_axis: HashSet<CircuitRc> = Default::default();
    let mut seen: HashMap<HashBytes, CircuitRc> = Default::default();

    fn f(
        ra: &mut HashMap<CircuitRc, usize>,
        rna: &mut HashSet<CircuitRc>,
        seen: &mut HashMap<HashBytes, CircuitRc>,
        name: Name,
        x: CircuitRc,
    ) -> Result<CircuitRc> {
        let h = x.info().hash;
        if let Some(r) = seen.get(&h) {
            Ok(r.clone())
        } else {
            {
            if axis_of_name(&x, name).is_none() {
                rna.insert(x.clone());
                return Ok(x);
            }

            if let Circuit::Module(m) = &**x {
                if axis_of_name(&x, name).unwrap() < m.aligned_batch_shape().len() {
                    let nodes2 = m
                        .args()
                        .map(|c| f(ra, rna, seen, name, c.clone()))
                        .collect::<Result<Vec<CircuitRc>>>()?;
                    return Ok(Module::nrc(nodes2, m.spec.clone(), m.info().name));
                } else {
                    bail!("batch_to_concat/get_axis_leaves: modules not fully supported, please substitute first")
                }
            }

            if !x
                .children()
                .any(|child| axis_of_name(&child, name).is_some())
            {
                if let Some(i) = axis_of_name(&x, name) {
                    ra.insert(x.clone(), i);
                    return Ok(x);
                } else {
                    panic!("get_axis_leaves: axis gone before traversal done")
                }
            }

            x.map_children(|c| f(ra, rna, seen, name, c))
        }.map(|r2|{seen.insert(h, r2.clone()); r2})
        }
    }
    let circ_new = f(
        &mut result_axis,
        &mut result_no_axis,
        &mut seen,
        name,
        circ_named_axes.clone(),
    )?;
    Ok((name, circ_new, result_axis, result_no_axis))
}

/// remove all instances of this axis name from circuit
pub fn deep_strip_axis_names(circuit: CircuitRc, names: &Option<HashSet<Name>>) -> CircuitRc {
    deep_map_unwrap(circuit, |x| {
        let axis_names: NamedAxes = if let Some(blacklist) = &names {
            x.info()
                .named_axes
                .clone()
                .into_iter()
                .filter(|(_i, name)| !blacklist.contains(name))
                .collect()
        } else {
            BTreeMap::new()
        };
        set_named_axes(x, axis_names).unwrap() // Ok because we are removing names
    })
}

#[apply(python_error_exception)]
#[base_error_name(NamedAxes)]
#[base_exception(PyRuntimeError)]
#[derive(Error, Debug, Clone)]
pub enum NamedAxesError {
    #[error("No whitespace, colon, or square brackets allowed in axis name '{name}'\n({e_name})")]
    ForbiddenCharacter { name: String },
}
