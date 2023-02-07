use anyhow::Result;
use circuit_base::{deep_map_preorder, deep_map_preorder_unwrap, prelude::*, visit_circuit};
use macro_rules_attribute::apply;
use pyo3::{pyfunction, PyObject};
use rr_util::{py_types::PyCallable, pycall, pycallable};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

#[apply(pycallable)]
#[pyo3(name = "filter_nodes")]
pub fn filter_nodes<F>(circuit: CircuitRc, filter: F) -> Result<HashSet<CircuitRc>>
where
    F: Fn((circuit, CircuitRc)) -> Result<bool>,
{
    let mut result: HashSet<CircuitRc> = HashSet::default();
    visit_circuit(circuit, |circuit: CircuitRc| {
        if filter(circuit.clone())? {
            result.insert(circuit);
        }
        Ok(())
    })?;
    Ok(result)
}

#[pyfunction]
#[pyo3(name = "replace_nodes")]
pub fn replace_nodes_py(circuit: CircuitRc, map: HashMap<CircuitRc, CircuitRc>) -> CircuitRc {
    deep_map_preorder_unwrap(circuit, |x: CircuitRc| -> CircuitRc {
        map.get(&x).cloned().unwrap_or(x)
    })
}

#[pyfunction]
#[pyo3(name = "update_nodes")]
pub fn update_nodes_py(
    circuit: CircuitRc,
    matcher: PyCallable,
    updater: PyCallable,
) -> Result<CircuitRc> {
    let nodes = filter_nodes_py(circuit.clone(), matcher)?;
    deep_map_preorder(circuit, |x| {
        if nodes.contains(&x) {
            pycall!(updater, (x,), anyhow)
        } else {
            Ok(x)
        }
    })
}

pub type CircuitPath = Vec<usize>;

#[pyfunction]
pub fn path_get(circuit: CircuitRc, path: CircuitPath) -> Option<CircuitRc> {
    let mut cur = circuit;
    for i in path {
        let children: Vec<CircuitRc> = cur.children().collect();
        if i >= children.len() {
            return None;
        }
        cur = children[i].clone()
    }
    Some(cur)
}

pub fn update_path<F>(circuit: CircuitRc, path: &CircuitPath, updater: F) -> Result<CircuitRc>
where
    F: Fn(CircuitRc) -> Result<CircuitRc>,
{
    fn recurse<F>(
        circuit: CircuitRc,
        path: &CircuitPath,
        path_idx: usize,
        updater: &F,
    ) -> Result<CircuitRc>
    where
        F: Fn(CircuitRc) -> Result<CircuitRc>,
    {
        if path_idx == path.len() {
            return updater(circuit);
        }
        circuit
            .map_children_enumerate(|i, circuit| {
                if i == path[path_idx] {
                    recurse(circuit, path, path_idx + 1, updater)
                } else {
                    Ok(circuit)
                }
            })
            .map(|z| z.rc())
    }
    recurse(circuit, path, 0, &updater)
}

#[pyfunction]
#[pyo3(name = "update_path")]
pub fn update_path_py(
    circuit: CircuitRc,
    path: CircuitPath,
    updater: PyObject,
) -> Result<CircuitRc> {
    update_path(circuit, &path, |x| pycall!(updater, (x.clone(),), anyhow))
}
