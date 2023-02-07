use std::{
    fmt::{self, Debug},
    sync::Arc,
};

use anyhow::Result;
use circuit_base::CircuitRc;
use pyo3::{prelude::*, AsPyPointer};
use rr_util::{setup_callable, simple_default};
use uuid::uuid;

use super::Updater;

#[derive(Clone, Debug)]
pub enum TransformData {
    Raw(RawTransform),
    Ident,
    PyFunc(PyObject),
}

#[derive(Clone, FromPyObject)]
pub enum TransformFromPy {
    Transform(Transform),
    #[pyo3(transparent)]
    PyFunc(PyObject),
}

setup_callable!(Transform, TransformData, TransformFromPy, run(circuit : CircuitRc) -> CircuitRc);

simple_default!(Transform { Self::ident() });
simple_default!(TransformFromPy { Self::Transform(Default::default()) });

impl TransformData {
    fn uuid(&self) -> [u8; 16] {
        match self {
            Self::Raw(_) => uuid!("8a3cdd61-09be-4881-bb78-759e66a0a63b"),
            Self::Ident => uuid!("61a4dfe4-194d-4f00-bc26-8b3f47c119b9"),
            Self::PyFunc(_) => uuid!("b847ee54-adc2-418d-8edb-4ba846be50fe"),
        }
        .into_bytes()
    }

    fn item_hash(&self, hasher: &mut blake3::Hasher) {
        match &self {
            Self::Raw(x) => {
                hasher.update(&(Arc::as_ptr(&x.0) as *const () as usize).to_le_bytes());
            }
            Self::Ident => {}
            Self::PyFunc(x) => {
                hasher.update(&(x.as_ptr() as usize).to_le_bytes());
            }
        }
    }
}

impl From<TransformFromPy> for Transform {
    fn from(m: TransformFromPy) -> Self {
        match m {
            TransformFromPy::Transform(x) => x,
            TransformFromPy::PyFunc(x) => TransformData::PyFunc(x).into(),
        }
    }
}

#[pyo3::pymethods]
impl Transform {
    #[new]
    fn py_new(inp: TransformFromPy) -> Self {
        inp.into()
    }

    pub fn run(&self, circuit: CircuitRc) -> Result<CircuitRc> {
        let ret = match &self.data {
            TransformData::Raw(f) => f.0(circuit)?,
            TransformData::Ident => circuit.clone(),
            TransformData::PyFunc(pyfunc) => {
                Python::with_gil(|py| pyfunc.call1(py, (circuit,)).and_then(|r| r.extract(py)))?
            }
        };

        Ok(ret)
    }

    #[staticmethod]
    pub fn ident() -> Self {
        TransformData::Ident.into()
    }

    #[pyo3(signature=(
        cache_transform = Updater::default().cache_transform,
        cache_update = Updater::default().cache_update
    ))]
    fn updater(&self, cache_transform: bool, cache_update: bool) -> Updater {
        Updater {
            transform: self.clone().into(),
            cache_transform,
            cache_update,
            ..Default::default()
        }
    }
}
