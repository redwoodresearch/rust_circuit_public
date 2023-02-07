use anyhow::Result;
use macro_rules_attribute::apply;
use pyo3::prelude::*;
use rr_util::{
    name::Name,
    py_types::Tensor,
    rearrange_spec::OpShape,
    symbolic_size::{SymbolicSizeConstraint, SymbolicSizeProduct},
    tensor_util::Shape,
};

use crate::{
    circuit_node_auto_impl, circuit_node_extra_impl,
    circuit_node_private::{CircuitNodeComputeInfoImpl, CircuitNodeHashItems},
    circuit_utils::{child_name_with_maybe_paren, OperatorPriority},
    new_rc_unwrap, CachedCircuitInfo, CircuitFlags, CircuitNode, CircuitNodeAutoName, CircuitRc,
    PyCircuitBase, TensorEvalError,
};

#[pyclass(extends=PyCircuitBase)]
#[derive(Clone)]
pub struct SetSymbolicShape {
    info: CachedCircuitInfo,
}

impl SetSymbolicShape {
    #[apply(new_rc_unwrap)]
    pub fn try_new(node: CircuitRc, shape: Shape, name: Option<Name>) -> Result<Self> {
        let out = Self {
            info: CachedCircuitInfo::incomplete(name, shape, vec![node]),
        };
        out.initial_init_info()
    }
}

circuit_node_extra_impl!(SetSymbolicShape, self_hash_default);

impl CircuitNodeComputeInfoImpl for SetSymbolicShape {
    fn symbolic_size_constraints_extra(&self) -> Result<Vec<SymbolicSizeConstraint>> {
        let out = self
            .node()
            .shape()
            .iter()
            .zip(self.shape())
            .map(|(&in_size, &set_to)| SymbolicSizeConstraint::get_new_from(in_size, set_to))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .filter_map(|x| x)
            .collect();
        Ok(out)
    }

    fn compute_flags(&self) -> CircuitFlags {
        self.node().compute_flags_default() & !CircuitFlags::IS_EXPLICITLY_COMPUTABLE
    }
}

impl CircuitNodeHashItems for SetSymbolicShape {
    fn compute_hash_non_name_non_children(&self, hasher: &mut blake3::Hasher) {
        for i in &self.info().shape {
            hasher.update(&i.to_le_bytes());
        }
    }
}

impl CircuitNode for SetSymbolicShape {
    circuit_node_auto_impl!("25332a58-39d6-443d-b225-65c810a09aa4");

    fn child_axis_map(&self) -> Vec<Vec<Option<usize>>> {
        vec![(0..self.rank()).map(Some).collect()]
    }

    fn _replace_children(&self, children: Vec<CircuitRc>) -> Result<Self> {
        Self::try_new(
            children[0].clone(),
            self.info().shape.clone(),
            self.info().name,
        )
    }

    fn eval_tensors(&self, _tensors: &[Tensor]) -> Result<Tensor> {
        Err(TensorEvalError::NotExplicitlyComputableInternal {
            circuit: self.crc(),
        }
        .into())
    }
}

impl CircuitNodeAutoName for SetSymbolicShape {
    const PRIORITY: OperatorPriority = OperatorPriority::PostFix {};

    fn auto_name(&self) -> Option<Name> {
        child_name_with_maybe_paren(&Self::PRIORITY, self.node().clone())
            .map(|n| (n.string() + " set_shape").into())
    }
}

#[pymethods]
impl SetSymbolicShape {
    #[new]
    #[pyo3(signature=(node, shape, name = None))]
    fn new_py(
        node: CircuitRc,
        shape: Shape,
        name: Option<Name>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let out = Self::try_new(node, shape, name)?;
        Ok(out.into_init())
    }

    #[staticmethod]
    pub fn some_set_neq(node: CircuitRc, shape: OpShape, name: Option<Name>) -> Result<CircuitRc> {
        assert_eq!(node.ndim(), shape.len()); // we should probably fix calling from python being able to trigger a panic!
        if shape
            .iter()
            .zip(node.shape())
            .all(|(&x, &s)| x.is_none() || x.unwrap() as usize == s)
        {
            return Ok(node);
        }
        let shape = shape
            .into_iter()
            .zip(node.shape())
            .map(|(x, s)| Option::from(x).unwrap_or(*s))
            .collect();
        Self::try_new(node, shape, name).map(|x| x.rc())
    }

    #[staticmethod]
    pub fn some_set_and_symbolic_neq(
        node: CircuitRc,
        shape: OpShape,
        name: Option<Name>,
    ) -> Result<CircuitRc> {
        let shape = shape
            .into_iter()
            .zip(node.shape())
            .map(|(x, &n_s)| {
                SymbolicSizeProduct::has_symbolic(n_s)
                    .then(|| Option::from(x))
                    .flatten()
                    .into()
            })
            .collect();
        Self::some_set_neq(node, shape, name)
    }

    #[getter]
    pub fn node(&self) -> CircuitRc {
        self.info.children[0].clone()
    }
}
