use std::{
    collections::hash_map::Entry,
    fmt::{self, Display},
};

use anyhow::{anyhow, bail, Context, Result};
use circuit_base::{
    cached_circuit_properties::max_non_leaf_size,
    circuit_utils::{hash_to_node_non_free, is_definitely_view_on_child, toposort_circuit},
    flat_concat,
    module::{any_children_with_symbolic_sizes, substitute_all_modules},
    prelude::*,
    print::{oom_fmt, PrintOptions},
    visit_circuit_non_free, Array, IrreducibleNode, Leaf, ModuleSpec, Scalar, Symbol,
};
use itertools::Itertools;
use macro_rules_attribute::apply;
use miniserde::json;
use num_bigint::BigUint;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};
use rr_util::{
    lru_cache::TensorCacheRrfs,
    py_types::{
        is_python_running, scalar_to_tensor, tensor_scale, un_flat_concat, ExtraPySelfOps, Tensor,
        PY_UTILS,
    },
    pycall, python_error_exception, sv,
    tensor_util::{Shape, TorchDevice},
    timed, unwrap,
    util::{arc_ref_clone, with_context_failable, HashBytes},
};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use thiserror::Error;

use crate::{
    circuit_optimizer::{OptimizationContext, OptimizationSettings},
    module_rewrite::deep_module_remove_unused_inputs,
    schedule_send::ScheduleToSend,
    scheduling_alg::{Dag, DagSimpSettings},
};

#[derive(Clone, Debug)]
pub enum Instruction {
    Drop(usize),
    Compute(usize, CircuitRc),
}

impl IntoPy<PyObject> for Instruction {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            Instruction::Drop(hb) => hb.into_py(py),
            Instruction::Compute(key, circ) => (key, circ).into_py(py),
        }
    }
}

/// allow for having tensors directly so we can bypass hashing (which isn't
/// important for scheduling)
#[derive(Clone, Debug, FromPyObject)]
pub enum ScheduleConstant {
    Circ(IrreducibleNode),
    Tensor(Tensor),
}

impl IntoPy<PyObject> for ScheduleConstant {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            ScheduleConstant::Circ(circ) => circ.into_py(py),
            ScheduleConstant::Tensor(tensor) => tensor.into_py(py),
        }
    }
}

impl ScheduleConstant {
    pub fn as_symbol(&self) -> Option<&Symbol> {
        if let Self::Circ(circ) = self {
            circ.as_symbol()
        } else {
            None
        }
    }
    pub fn as_array(&self) -> Option<&Array> {
        if let Self::Circ(circ) = self {
            circ.as_array()
        } else {
            None
        }
    }
    pub fn as_tensor(&self) -> Option<&Tensor> {
        if let Self::Tensor(t) = self {
            Some(t)
        } else {
            None
        }
    }
    pub fn convert_as_tensor(&self) -> Option<&Tensor> {
        match self {
            Self::Circ(circ) => circ.as_array().map(|x| &x.value),
            Self::Tensor(tensor) => Some(tensor),
        }
    }
    pub fn shape(&self) -> &Shape {
        match self {
            Self::Circ(circ) => &circ.info().shape,
            Self::Tensor(tensor) => tensor.shape(),
        }
    }
}

/// Schedule is an optimized sequence of operations to perform to compute a circuit
/// it keeps arrayconstants seperately, and replacing arrayconstants in a schedule should produce the same
/// output + ~performance as optimizing the circuit with different arrayconstants from the beginning
/// Schedule stores intermediate nodes as circuits with Symbol children, where the name of the symbol
/// contains the "id" of that node
#[pyclass]
#[derive(Clone, Debug)]
pub struct Schedule {
    #[pyo3(get)]
    pub instructions: Vec<Instruction>,
    #[pyo3(get)]
    pub constants: HashMap<usize, ScheduleConstant>,
    // keep scalar constants seperate so adjust numerical scale can work without losing precision
    // before when these were in tensors, they had to be right dtype and therefore overflowed when
    // adjustment needed
    #[pyo3(get)]
    pub scalars: HashMap<usize, Scalar>,
    pub output_circuit: Option<(usize, CircuitRc)>,
    pub split_shapes: Option<Vec<Shape>>,
    pub old_constant_hashes: HashMap<HashBytes, usize>,
}

#[pymethods]
impl Schedule {
    pub fn validate(&self, validate_output: bool) -> Result<()> {
        // NOTE: values can't be computed or dropped multiple times atm.
        let mut prior_compute_numbers = HashSet::default();
        let mut prior_drop_numbers = HashSet::default();

        for num in self.scalars.keys() {
            if self.constants.contains_key(num) {
                bail!("num={num} used for both constants and scalars!");
            }
        }
        for instruction in &self.instructions {
            match instruction {
                Instruction::Drop(drop) => {
                    if !prior_compute_numbers.contains(drop) {
                        bail!("drop={drop} not previously computed");
                    }
                    if !prior_drop_numbers.insert(drop) {
                        bail!("drop={drop} was already dropped!");
                    }
                }
                Instruction::Compute(key, circuit) => {
                    if !prior_compute_numbers.insert(key) {
                        bail!("key={key} was already computed!");
                    }
                    if circuit.is_leaf() {
                        bail!("leaf circuit is being computed circuit={circuit:?}");
                    }
                }
            }
        }

        if validate_output {
            let output_id = self.output_circuit.as_ref().ok_or(anyhow!("no output!"))?.0;
            for computed in prior_compute_numbers {
                if !prior_drop_numbers.contains(computed) && *computed != output_id {
                    bail!("computed={computed} is never dropped (or returned)");
                }
            }
        }

        Ok(())
    }

    #[pyo3(signature=(map, allow_missing = false))]
    pub fn replace_tensors(
        &self,
        map: HashMap<HashBytes, Tensor>,
        allow_missing: bool,
    ) -> Result<Self> {
        let mut result = self.clone();
        for (k, v) in map {
            if !result.old_constant_hashes.contains_key(&k) {
                if allow_missing {
                    continue;
                }
                return Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                    "key circuit wasn't present in original",
                )
                .into());
            }
            // should be present because we checked above
            let entry = unwrap!(
                result.constants.entry(result.old_constant_hashes[&k]),
                Entry::Occupied
            );
            if entry.get().shape() != v.shape() {
                bail!("tensor shape doesn't match entry shape!");
            }

            *entry.into_mut() = ScheduleConstant::Tensor(v);
        }
        Ok(result)
    }

    #[pyo3(name = "map_tensors")]
    pub fn map_tensors_py(&self, f: PyObject) -> PyResult<Self> {
        let mut result = self.clone();

        result.old_constant_hashes.iter().for_each(|(key, id)| {
            let maybe_tensor: Option<Tensor> =
                Python::with_gil(|py| pycall!(f, (PyBytes::new(py, key),)));
            if let Some(t) = maybe_tensor {
                result.constants.insert(*id, ScheduleConstant::Tensor(t));
            }
        });
        Ok(result)
    }
    #[pyo3(signature=(settings = Default::default()))]
    pub fn evaluate(&self, settings: OptimizationSettings) -> Result<Tensor> {
        let timed = settings.verbose >= 1;
        let eval = || -> Result<_> {
            let result = if settings.adjust_numerical_scale {
                evaluate_schedule_adjust_numerical_scale(self, settings)?
                    [&self.output_circuit.as_ref().unwrap().0]
                    .clone()
            } else {
                evaluate_schedule(self)?[&self.output_circuit.as_ref().unwrap().0].clone()
            };
            Ok(result)
        };
        Ok(timed!(eval()?, 10, timed))
    }

    // TODO: I think we can still panic on other non-explicitly computable stuff? (e.g., cumulants etc)
    pub fn check_no_syms(&self) -> Result<()> {
        let mut iter = self
            .constants
            .iter()
            .filter_map(|x| x.1.as_symbol())
            .peekable();
        if iter.peek().is_some() {
            bail!(SchedulingError::EvaluateCalledWithSyms {
                syms: iter.cloned().collect()
            })
        }
        Ok(())
    }
    pub fn check_no_raw_tensor_constants(&self) -> Result<()> {
        let mut iter = self
            .constants
            .iter()
            .filter_map(|x| x.1.as_tensor())
            .peekable();
        if iter.peek().is_some() {
            bail!("has raw tensor constant")
        }
        Ok(())
    }

    #[pyo3(signature=(settings = Default::default()))]
    pub fn evaluate_many(&self, settings: OptimizationSettings) -> Result<Vec<Tensor>> {
        let single = self.evaluate(settings)?;
        Ok(self.split(single))
    }

    pub fn split(&self, tensor: Tensor) -> Vec<Tensor> {
        un_flat_concat(&tensor, self.split_shapes.clone().unwrap()).unwrap()
    }

    pub fn get_stats(&self) -> ScheduleStats {
        let mut mem: BigUint = BigUint::from(0usize);
        let mut max_mem: BigUint = BigUint::from(0usize);
        let mut biggest: HashMap<usize, CircuitRc> = HashMap::default();
        let mut current: HashMap<usize, CircuitRc> = HashMap::default();
        for instruction in self.instructions.clone() {
            match instruction {
                Instruction::Drop(drop) => {
                    let dropped = current.remove(&drop).unwrap();
                    mem -= dropped.info().numel();
                }
                Instruction::Compute(key, circuit) => {
                    current.insert(key, circuit.clone());
                    mem += circuit.info().numel();
                    if mem > max_mem {
                        max_mem = mem.clone();
                        biggest = current.clone();
                    }
                }
            }
        }
        ScheduleStats {
            max_mem,
            constant_mem: self
                .constants
                .iter()
                .map(|(_h, t)| t.shape().iter().product::<usize>())
                .sum(),
            max_circuit_set: biggest.values().cloned().collect(),
        }
    }

    pub fn next_key(&self) -> usize {
        *self
            .constants
            .keys()
            .chain(self.scalars.keys())
            .chain(self.instructions.iter().filter_map(|ins| match ins {
                Instruction::Compute(k, _c) => Some(k),
                _ => None,
            }))
            .max()
            .unwrap_or(&0)
            + 1
    }

    pub fn serialize(&self) -> Result<String> {
        let tosend: ScheduleToSend = self.try_into()?;
        Ok(json::to_string(&tosend))
    }

    #[staticmethod]
    #[pyo3(name = "deserialize")]
    pub fn deserialize_py(string: String, tensor_cache: Option<TensorCacheRrfs>) -> Result<Self> {
        let mut tensor_cache = tensor_cache;
        Schedule::deserialize(string, &mut tensor_cache)
    }

    pub fn tosend(&self) -> Result<ScheduleToSend> {
        self.try_into()
    }

    pub fn evaluate_remote(&self, remote_url: String) -> Result<Option<Tensor>> {
        let out = self.tosend()?.evaluate_remote(remote_url, TorchDevice::Cpu);
        Ok(out)
    }
    pub fn evaluate_remote_many(&self, remote_url: String) -> Result<Option<Vec<Tensor>>> {
        let out = self
            .tosend()?
            .evaluate_remote_many(remote_url, TorchDevice::Cpu);
        Ok(out)
    }
    fn __repr__(&self) -> String {
        format!("{}", &self)
    }
}
impl Schedule {
    pub fn deserialize(string: String, tensor_cache: &mut Option<TensorCacheRrfs>) -> Result<Self> {
        let sent: ScheduleToSend =
            json::from_str(&string).context("schedule deserialization failed due to json error")?;
        sent.load(tensor_cache)
    }
}

impl Display for Schedule {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Schedule: instructions\n{}\nTensors: {} Arrays: {} Symbols: {} Scalars: {}",
            self.instructions
                .iter()
                .filter_map(|i| {
                    if let Instruction::Compute(k, c) = i {
                        return Some(
                            k.to_string()
                                + " "
                                + &c.variant_string()
                                + " "
                                + &format!("{:?}", c.info().device_dtype.dtype)
                                + " "
                                + &c.children()
                                    .map(|x| get_child_key(x).to_string())
                                    .collect::<Vec<String>>()
                                    .join(" "),
                        );
                    }
                    None
                })
                .collect::<Vec<String>>()
                .join("\n"),
            self.constants
                .iter()
                .filter_map(|(k, c)| c.as_tensor().map(|_| k.to_string()))
                .collect::<Vec<String>>()
                .join(" "),
            self.constants
                .iter()
                .filter_map(|(k, c)| c.as_array().map(|_| k.to_string()))
                .collect::<Vec<String>>()
                .join(" "),
            self.constants
                .iter()
                .filter_map(|(k, c)| c.as_symbol().map(|_| k.to_string()))
                .collect::<Vec<String>>()
                .join(" "),
            self.scalars
                .iter()
                .map(|(k, _c)| k.to_string())
                .collect::<Vec<String>>()
                .join(" "),
        )
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct ScheduleStats {
    #[pyo3(get)]
    max_mem: BigUint,
    #[pyo3(get)]
    constant_mem: BigUint, // this has already been allocated so it can't be over 2^64
    #[pyo3(get)]
    max_circuit_set: HashSet<CircuitRc>,
}

#[apply(python_error_exception)]
#[base_error_name(SchedulingOOM)]
#[base_exception(PyValueError)]
#[derive(Error, Clone, Debug)]
pub enum SchedulingOOMError {
    #[error("{string} ({e_name})")]
    Many {
        max_memory: usize,
        memory_chunks: usize,
        node_memories: Vec<usize>,
        string: String,
    },
    #[error("Single element doesn't fit {formatted} ({e_name})")]
    Single { numel: BigUint, formatted: String },
    #[error("Simplification can't find anything within bounds, likely each isolated subdag fits but together they don't. maybe try scheduling_simplify=False opt setting graph {cyto} ({e_name})")]
    Simp { cyto: String },
    #[error("Threading failed! bug! ({e_name})")]
    ThreadsLost {},

    #[error("Schedule exhaustive timed out, at size {size} iters {iters} graph (click to see circuit structure) {cyto} ({e_name})")]
    ExhaustiveTimeout {
        cyto: String,
        iters: usize,
        size: usize,
    },
}

#[apply(python_error_exception)]
#[base_error_name(Scheduling)]
#[base_exception(PyValueError)]
#[derive(Error, Debug)]
pub enum SchedulingError {
    #[error("circuit={circuit:?} ({e_name})")]
    NotExplicitlyComputable { circuit: CircuitRc },

    #[error("syms={syms:?} ({e_name})")]
    EvaluateCalledWithSyms { syms: Vec<Symbol> },
}

impl Display for ScheduleStats {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut shapes = self
            .max_circuit_set
            .iter()
            .map(|x| x.info().shape.clone())
            .collect::<Vec<Shape>>();
        shapes.sort_by_key(|x| std::cmp::Reverse(x.iter().product::<usize>()));
        let shapes_and_percents: String = shapes
            .iter()
            .map(|x| {
                format!(
                    "{:?} {}%",
                    x,
                    // biguint doesn't have cast f64? that would be less lossy than truncate
                    (x.iter().product::<usize>() as f64 / self.max_mem.to_u64_digits()[0] as f64
                        * 100.0) as i64
                )
            })
            .collect::<Vec<String>>()
            .join(", ");
        let result = format!(
            "ScheduleStats: max: {} const: {} shapes: {}",
            oom_fmt(self.max_mem.clone()),
            oom_fmt(self.constant_mem.clone()),
            shapes_and_percents
        );
        write!(f, "{}", result)
    }
}

fn get_child_key(circuit: CircuitRc) -> usize {
    circuit
        .as_symbol()
        .unwrap()
        .info()
        .name
        .unwrap()
        .parse::<usize>()
        .unwrap()
}

fn child_from_key(key: usize, shape: Shape) -> Symbol {
    Symbol::new_with_random_uuid(shape, Some(key.to_string().into()))
}

pub fn get_children_keys(circuit: CircuitRc) -> Vec<usize> {
    circuit.children().map(get_child_key).collect()
}

/// see https://pyo3.rs/main/doc/pyo3/marker/struct.python#method.new_pool
pub unsafe fn with_gil_pool<F, R>(f: F) -> R
where
    F: for<'py> FnOnce(Python<'py>) -> R,
{
    Python::with_gil(|py| {
        let pool = unsafe { py.new_pool() };
        let py = pool.python();
        f(py)
    })
}

pub fn evaluate_schedule(schedule: &Schedule) -> Result<HashMap<usize, Tensor>> {
    schedule.check_no_syms()?;
    let mut live: HashMap<usize, Tensor> = schedule
        .constants
        .iter()
        .map(|x| x.1.convert_as_tensor().map(|z| (*x.0, z.clone())))
        .collect::<Option<HashMap<_, _>>>()
        .unwrap();
    live.extend(
        schedule
            .scalars
            .iter()
            .map(|x| (*x.0, x.1.eval_tensors(&[]).unwrap())),
    );
    for s in &schedule.instructions {
        let run = |_py: Python| match s {
            Instruction::Compute(key, circ) => {
                let child_keys: Vec<usize> = circ.children().map(get_child_key).collect();
                child_keys.iter().for_each(|child_key| {
                    if !live.contains_key(child_key) {
                        panic!("FAIL");
                    }
                });
                let tensors: Vec<Tensor> = child_keys
                    .iter()
                    .map(|child_key| live[child_key].clone())
                    .collect();
                let result_err = circ.eval_tensors(&tensors);
                if result_err.is_err() {
                    println!("errored evaluate");
                    circ.print().unwrap()
                }
                let result = result_err.unwrap();
                assert!(
                    result.shape() == circ.shape(),
                    "circ={:?}, result_shape={:?}",
                    circ,
                    result.shape()
                );
                live.insert(*key, result);
            }
            Instruction::Drop(key) => {
                live.remove(key);
            }
        };
        unsafe { with_gil_pool(run) }
    }
    Ok(live)
}

/// evaluate a circuit while measuring the numerical scale of tensor contents
/// and computing "10^10 * x" or such instead of x to avoid numerical overflow
pub fn evaluate_schedule_adjust_numerical_scale(
    schedule: &Schedule,
    settings: OptimizationSettings,
) -> Result<HashMap<usize, Tensor>> {
    schedule.check_no_syms()?;

    // we store (tensor, scale) where scale is a number the tensor's been multiplied by
    // so (tensor(1e10),1.0) evaluates to same as (tensor(1),1e-10)
    let mul = |tup: &(Tensor, f64), m: f64| -> (Tensor, f64) {
        Python::with_gil(|py| (tup.0.clone().py_mul(py, m).unwrap(), tup.1 * m))
    };
    let set_scale = |tup: &(Tensor, f64), new_scale: f64| -> (Tensor, f64) {
        Python::with_gil(|py| {
            (
                tup.0.clone().py_mul(py, new_scale / tup.1).unwrap(),
                new_scale,
            )
        })
    };
    let clamp = |tup: &(Tensor, f64)| -> (Tensor, f64) {
        let scale = tensor_scale(&tup.0).unwrap();
        if (scale > settings.numerical_scale_max || scale < settings.numerical_scale_min)
            && scale != 0.0
        {
            mul(tup, 1.0 / scale)
        } else {
            tup.clone()
        }
    };
    let uniformize = |tups: &Vec<(Tensor, f64)>| -> Vec<(Tensor, f64)> {
        if tups.is_empty() || tups.iter().all(|x| x.1 == tups[0].1) {
            tups.clone()
        } else {
            let new_scale: f64 = tups
                .iter()
                .map(|x| x.1)
                .reduce(|a, b| if a > b { a } else { b })
                .unwrap();
            tups.iter().map(|x| set_scale(x, new_scale)).collect()
        }
    };

    let mut live: HashMap<usize, (Tensor, f64)> = schedule
        .constants
        .iter()
        .map(|(key, x)| (*key, clamp(&(x.convert_as_tensor().unwrap().clone(), 1.0))))
        .collect();
    live.extend(schedule.scalars.iter().map(|(h, s)| {
        (*h, {
            let value_scale = s.value.abs();
            if value_scale > settings.numerical_scale_max
                || value_scale < settings.numerical_scale_min && value_scale != 0.0
            {
                (
                    scalar_to_tensor(s.value.signum(), s.info().shape.clone(), Default::default())
                        .unwrap(),
                    1.0 / value_scale,
                )
            } else {
                (
                    scalar_to_tensor(s.value, s.info().shape.clone(), Default::default()).unwrap(),
                    1.0,
                )
            }
        })
    }));
    for s in &schedule.instructions {
        let run = |_py: Python| match s {
            Instruction::Compute(key, circ) => {
                circ.children().for_each(|x: CircuitRc| {
                    if !live.contains_key(&get_child_key(x)) {
                        panic!("FAIL");
                    }
                });
                let tensors_and_scales: Vec<(Tensor, f64)> = circ
                    .children()
                    .map(|x| live[&get_child_key(x)].clone())
                    .collect();
                let tensors: Vec<Tensor> = tensors_and_scales.iter().map(|x| x.0.clone()).collect();
                let result = match &***circ {
                    Circuit::Einsum(_) => {
                        let new_scale = tensors_and_scales.iter().map(|x| x.1).product();
                        clamp(&(circ.eval_tensors(&tensors).unwrap(), new_scale))
                    }
                    Circuit::Add(_) | Circuit::Concat(_) => {
                        let new_ts = uniformize(&tensors_and_scales);
                        (
                            circ.eval_tensors(
                                &new_ts.iter().map(|x| x.0.clone()).collect::<Vec<_>>(),
                            )
                            .unwrap(),
                            new_ts[0].1,
                        )
                    }
                    Circuit::GeneralFunction(_) => {
                        let new_ts: Vec<(Tensor, f64)> = tensors_and_scales
                            .iter()
                            .map(|x| set_scale(x, 1.0))
                            .collect();
                        clamp(&(
                            circ.eval_tensors(
                                &new_ts.iter().map(|x| x.0.clone()).collect::<Vec<_>>(),
                            )
                            .unwrap(),
                            1.0,
                        ))
                    }
                    Circuit::Index(_) | Circuit::Rearrange(_) | Circuit::Scatter(_) => (
                        circ.eval_tensors(
                            &tensors_and_scales
                                .iter()
                                .map(|x| x.0.clone())
                                .collect::<Vec<_>>(),
                        )
                        .unwrap(),
                        tensors_and_scales[0].1,
                    ),
                    Circuit::Scalar(_) | Circuit::Array(_) | Circuit::Symbol(_) => {
                        panic!("constant found as schedule instruction, not supposed to happen")
                    }
                    Circuit::Tag(_) => tensors_and_scales[0].clone(),
                    _ => {
                        unimplemented!()
                    }
                };
                assert!(result.0.shape()[..] == circ.info().shape[..]);
                live.insert(*key, result);
            }
            Instruction::Drop(hash) => {
                live.remove(hash);
            }
        };
        unsafe { with_gil_pool(run) }
    }
    let out = live
        .iter()
        .map(|(k, v)| (*k, set_scale(v, 1.0).0))
        .collect();
    Ok(out)
}

/// this supports dropping and recomputing. if you have an Evaluate(CircuitRc) of something you already evaluated
/// it's assumed you dropped this and are recomputing it
pub fn order_to_schedule(
    order: &Vec<CircuitRc>,
    constants: &Vec<IrreducibleNode>,
    scalars: &Vec<Scalar>,
    to_keep: HashSet<HashBytes>,
) -> (Schedule, Vec<usize>) {
    let mut circ_to_id: HashMap<HashBytes, usize> = Default::default();
    let constants: HashMap<usize, _> = constants
        .iter()
        .map(|x| {
            circ_to_id.insert(x.info().hash, circ_to_id.len());
            (circ_to_id.len() - 1, x.clone())
        })
        .collect();
    let mut result: Schedule = Schedule {
        instructions: vec![],
        scalars: scalars
            .iter()
            .map(|x| {
                circ_to_id.insert(x.info().hash, circ_to_id.len());
                (circ_to_id.len() - 1, x.clone())
            })
            .collect(),
        split_shapes: None,
        output_circuit: None,
        old_constant_hashes: constants
            .iter()
            .map(|(id, node)| (node.info().hash, *id))
            .collect(),
        constants: constants
            .into_iter()
            .map(|(h, n)| (h, ScheduleConstant::Circ(n)))
            .collect(),
    };
    let mut seen_dependencies: HashSet<usize> = HashSet::default();
    for ex in order.iter().rev() {
        for child in ex.non_free_children() {
            let next_id = circ_to_id.len();
            let dep = *circ_to_id.entry(child.info().hash).or_insert(next_id);
            if !Leaf::matches(&child)
                && seen_dependencies.insert(dep)
                && !to_keep.contains(&child.info().hash)
            {
                result.instructions.push(Instruction::Drop(dep));
            }
        }
        let next_id = circ_to_id.len();
        let our_id = *circ_to_id.entry(ex.info().hash).or_insert(next_id);
        let node_here_symbol_children = (**ex.map_non_free_children_unwrap(|child| {
            child_from_key(circ_to_id[&child.info().hash], child.info().shape.clone()).rc()
        }))
        .clone()
        .rc();
        result
            .instructions
            .push(Instruction::Compute(our_id, node_here_symbol_children));
        // seen_dependencies.remove(&ex.info().hash);
    }
    result.instructions.reverse();
    (result, to_keep.iter().map(|x| circ_to_id[x]).collect())
}

pub fn circuits_to_dag(circuits: &[CircuitRc]) -> Dag {
    let mut result: Dag = Default::default();
    let mut to_merge_with_child: Vec<u32> = vec![];
    // append the circuit to the dag if it's not already there, and return its index
    let number_node = |c: CircuitRc, result: &mut Dag, to_merge_with_child: &mut Vec<u32>| -> u32 {
        if let Some(idx) = result.hash_to_node.get(&c.info().hash) {
            return *idx as u32;
        }
        result.node_costs.push(c.info().naive_mem_use_usize(None));
        result.node_hashes.push(c.info().hash);
        result
            .hash_to_node
            .insert(c.info().hash, result.node_hashes.len() - 1);
        result.children.push(sv![]);
        result.parents.push(sv![]);
        let output = (result.node_hashes.len() - 1) as u32;

        if is_definitely_view_on_child(c.clone()) {
            to_merge_with_child.push(output)
        }
        output
    };
    for circuit in circuits {
        // non free so we avoid recuring into module spec.circuit
        visit_circuit_non_free(
            circuit.clone(),
            |c: CircuitRc| {
                // Arrays are never added to the dag to begin with because they're always 0 cost

                if !c.is_leaf() {
                    let my_number: u32 =
                        number_node(c.clone(), &mut result, &mut to_merge_with_child);

                    let children_to_consider: Vec<CircuitRc> = c
                        .non_free_children()
                        .filter(|child| !Leaf::matches(child))
                        .collect();
                    result.children[my_number as usize] = children_to_consider
                        .iter()
                        .map(|child| {
                            number_node(child.clone(), &mut result, &mut to_merge_with_child)
                        })
                        .unique()
                        .collect();
                    for child in children_to_consider {
                        let new_num =
                            number_node(child.clone(), &mut result, &mut to_merge_with_child);
                        if !result.parents[new_num as usize]
                            .iter()
                            .contains(&(my_number as u32))
                        {
                            result.parents[new_num as usize].push(my_number as u32);
                        }
                    }
                }

                Ok(())
            },
            false,
        )
        .unwrap();
    }
    result.node_to_orig = (0..result.node_costs.len() as u32)
        .map(|x| (x, sv![x].clone()))
        .collect();
    for tmp in to_merge_with_child {
        assert!(result.children[tmp as usize].len() < 2);
        if result.children[tmp as usize].len() == 1 {
            result.merge_larger(result.children[tmp as usize][0], tmp, false)
        }
    }
    result
}

pub fn circuit_to_schedule(
    circuit: CircuitRc,
    context: &mut OptimizationContext,
) -> Result<Schedule> {
    // we could avoid this if we wanted, but a bit annoying
    // (requires handle rearrange due to unused arg + empty instruction case!)
    let circuit = deep_module_remove_unused_inputs(
        circuit,
        &mut context.cache.module_removed_unused,
        true,
        true,
        true,
    );
    if is_python_running() && any_children_with_symbolic_sizes(circuit.clone()) {
        Python::with_gil(|py| {
            pyo3::PyErr::warn(
                py,
                PY_UTILS.optimizing_symbolic_size_warning.as_ref(py),
                concat!(
                    "you're scheduling with symbolic sizes.",
                    " Perhaps you need to conform_all_modules?",
                    " (this is typically done in optimize_circuit)."
                ),
                1,
            )
        })?;
    }

    let mut dag = circuits_to_dag(&[circuit.clone()]);
    if max_non_leaf_size(circuit.clone()) > BigUint::from(context.settings.max_single_tensor_memory)
    {
        bail!(SchedulingOOMError::Single {
            numel: max_non_leaf_size(circuit.clone()),
            formatted: oom_fmt(max_non_leaf_size(circuit.clone()))
        });
    }
    let order_result = {
        let mut result: Result<Vec<u32>, SchedulingOOMError> = Err(SchedulingOOMError::Many {
            max_memory: 0,
            memory_chunks: 0,
            node_memories: vec![],
            string: "".to_owned(),
        });
        let mut mem = context.settings.max_memory;
        while mem
            <= context
                .settings
                .max_memory_fallback
                .unwrap_or(context.settings.max_memory)
            && result.is_err()
        {
            if context.settings.scheduling_simplify {
                let mut dag_simp_settings: DagSimpSettings = Default::default();
                dag_simp_settings.acceptable_subdag_max = context.settings.max_memory;
                dag_simp_settings.mem_limit = mem;
                dag_simp_settings.num_mem_chunks = context.settings.scheduling_num_mem_chunks;
                dag_simp_settings.timeout = context.settings.scheduling_timeout;
                dag_simp_settings.verbose = context.settings.verbose;
                dag_simp_settings.parallelism = context.settings.optimization_parallelism;
                timed!(
                    dag.simplify(&dag_simp_settings)?,
                    10,
                    context.settings.verbose >= 2
                );
            }
            result = dag.compute_schedule(
                context.settings.verbose,
                mem,
                context.settings.scheduling_num_mem_chunks,
                context.settings.scheduling_timeout,
            );
            mem *= 2;
        }
        result
    };
    let order = with_context_failable(order_result, || {
        Ok(format!(
            "getting schedule failed for circuit:\n{}",
            PrintOptions::compiler_default().repr(circuit.clone())?
        ))
    })?;

    let to_node = hash_to_node_non_free(circuit.clone(), false);
    let circuit_order: Vec<CircuitRc> = order
        .iter()
        .map(|x| to_node[&dag.node_hashes[*x as usize]].clone())
        .collect();

    let (mut out, kept_keys) = order_to_schedule(
        &circuit_order,
        &to_node
            .iter()
            .filter_map(|x| Option::<IrreducibleNode>::from(arc_ref_clone(&x.1)))
            .collect(),
        &to_node
            .iter()
            .filter_map(|x| x.1.as_scalar().cloned())
            .collect(),
        HashSet::from_iter([circuit.info().hash]),
    );
    out.output_circuit = Some((kept_keys[0], circuit));
    let out = replace_module_schedules(&out, context);
    out.validate(true).expect("should be valid");
    Ok(out)
}

pub fn circuit_to_schedule_naive_toposort(circuit: CircuitRc) -> Result<Schedule> {
    let circuit = substitute_all_modules(circuit);
    let toposorted = toposort_circuit(circuit.clone());
    let mut order = Vec::new();
    let mut scalars = Vec::new();
    let mut constants = Vec::new();

    for c in toposorted {
        if let Some(sc) = c.as_scalar() {
            scalars.push(sc.clone())
        } else if let Some(constant) = Option::<IrreducibleNode>::from(c.clone().c()) {
            constants.push(constant)
        } else {
            order.push(c)
        }
    }

    let (mut result, kept_keys) = order_to_schedule(
        &order,
        &constants,
        &scalars,
        HashSet::from_iter([circuit.info().hash]),
    );
    result.output_circuit = Some((kept_keys[0], circuit));
    result.validate(true).expect("should be valid");
    Ok(result)
}

#[pyfunction]
pub fn scheduled_evaluate(circuit: CircuitRc, settings: OptimizationSettings) -> Result<Tensor> {
    let schedule = if settings.scheduling_naive {
        circuit_to_schedule_naive_toposort(circuit)
    } else {
        circuit_to_schedule(
            circuit.clone(),
            &mut OptimizationContext::new_settings(settings),
        )
    }?;
    Ok(evaluate_schedule(&schedule)?[&schedule.output_circuit.unwrap().0].clone())
}

#[pyfunction]
#[pyo3(name = "circuit_to_schedule")]
pub fn py_circuit_to_schedule(
    circuit: CircuitRc,
    settings: OptimizationSettings,
) -> Result<Schedule> {
    circuit_to_schedule(
        circuit.clone(),
        &mut OptimizationContext::new_settings(settings),
    )
}

#[pyfunction]
#[pyo3(name = "circuit_to_schedule_many")]
pub fn py_circuit_to_schedule_many(
    circuits: Vec<CircuitRc>,
    settings: OptimizationSettings,
) -> Result<Schedule> {
    let flat_concatted = flat_concat(circuits.clone())?.rc();
    let mut schedule = py_circuit_to_schedule(flat_concatted, settings)?;
    schedule.split_shapes = Some(
        circuits
            .iter()
            .map(|x| x.info().shape.clone())
            .collect::<Vec<Shape>>(),
    );
    Ok(schedule)
}

fn schedule_module_spec(spec: &ModuleSpec, context: &mut OptimizationContext) -> Schedule {
    context
        .cache
        .module_specs_scheduled_same_settings
        .get(spec)
        .cloned()
        .unwrap_or_else(|| {
            let result = circuit_to_schedule(spec.circuit.clone(), context).unwrap();
            context
                .cache
                .module_specs_scheduled_same_settings
                .insert(spec.clone(), result.clone());
            result
        })
}

/// note: this doesn't allow for any tensors in constants
fn replace_module_schedules(schedule: &Schedule, context: &mut OptimizationContext) -> Schedule {
    let mut result = schedule.clone();
    let mut scalar_to_id: HashMap<Scalar, usize> = schedule
        .scalars
        .iter()
        .map(|(a, b)| (b.clone(), *a))
        .collect();
    let mut irreducible_to_id: HashMap<IrreducibleNode, usize> = schedule
        .constants
        .iter()
        .map(|(a, b)| (unwrap!(b, ScheduleConstant::Circ).clone(), *a))
        .collect();
    result.instructions = vec![];
    let mut next_key = schedule.next_key();
    for ins in &schedule.instructions {
        match ins {
            Instruction::Drop(_d) => result.instructions.push(ins.clone()),
            Instruction::Compute(outermost_k, v) => match &***v {
                Circuit::Module(mn) => {
                    mn.spec
                        .check_all_inputs_used()
                        .expect("unused args removed at start of circuit_to_schedule!");

                    let original_children: Vec<usize> =
                        mn.args().cloned().map(get_child_key).collect();

                    let inner_schedule = schedule_module_spec(&mn.spec, context);
                    let mut inner_to_outer_key = HashMap::default();
                    for (inner_k, sc) in &inner_schedule.scalars {
                        let outer = scalar_to_id.get(sc).cloned().unwrap_or_else(|| {
                            let this_key = next_key;
                            scalar_to_id.insert(sc.clone(), this_key);
                            result.scalars.insert(this_key, sc.clone());
                            next_key += 1;
                            this_key
                        });
                        inner_to_outer_key.try_insert(*inner_k, outer).unwrap();
                    }
                    for (inner_k, irreducible) in &inner_schedule.constants {
                        let irreducible = unwrap!(irreducible, ScheduleConstant::Circ);
                        let outer =
                            irreducible_to_id
                                .get(irreducible)
                                .cloned()
                                .unwrap_or_else(|| {
                                    if let Some(pos) =
                                        mn.spec.arg_specs.iter().position(|argspec| {
                                            argspec.symbol.info().hash == irreducible.info().hash
                                        })
                                    {
                                        original_children[pos]
                                    } else {
                                        let this_key = next_key;
                                        irreducible_to_id.insert(irreducible.clone(), this_key);
                                        result.constants.insert(
                                            this_key,
                                            ScheduleConstant::Circ(irreducible.clone()),
                                        );
                                        next_key += 1;
                                        this_key
                                    }
                                });
                        inner_to_outer_key.try_insert(*inner_k, outer).unwrap();
                    }

                    assert!(
                        !inner_schedule.instructions.is_empty(),
                        "empty_modules removed at start of circuit to schedule"
                    );

                    inner_to_outer_key
                        .try_insert(
                            inner_schedule.output_circuit.clone().unwrap().0,
                            *outermost_k,
                        )
                        .unwrap();

                    for inner_ins in &inner_schedule.instructions {
                        match inner_ins {
                            Instruction::Drop(drop) => {
                                result
                                    .instructions
                                    .push(Instruction::Drop(inner_to_outer_key[drop]));
                            }
                            Instruction::Compute(inner_k, c) => {
                                assert!(!c.is_module());
                                let new_c = c
                                    .map_children_unwrap(|child| {
                                        let child_inner_k = get_child_key(child.clone());
                                        let child_outer_k =
                                            inner_to_outer_key.get(&child_inner_k).unwrap();
                                        child_from_key(*child_outer_k, child.info().shape.clone())
                                            .rc()
                                    })
                                    .rc();
                                let key =
                                    inner_to_outer_key.get(inner_k).cloned().unwrap_or_else(|| {
                                        let this_key = next_key;
                                        inner_to_outer_key.insert(*inner_k, this_key);
                                        next_key += 1;
                                        this_key
                                    });
                                result.instructions.push(Instruction::Compute(key, new_c));
                                inner_to_outer_key.insert(*inner_k, key);
                            }
                        }
                    }
                }
                // Note: if other node types have free we'd have to handle that here!
                _ => {
                    result.instructions.push(ins.clone());
                }
            },
        }
    }
    result.validate(result.output_circuit.is_some()).unwrap();

    result
}
