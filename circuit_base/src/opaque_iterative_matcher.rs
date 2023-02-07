use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Debug,
    ops::Deref,
    sync::Arc,
};

use anyhow::Result;
use pyo3::{prelude::*, pyclass::CompareOp, types::PyDict, PyObject};
use rr_util::{
    atr,
    py_types::{use_rust_comp, SELF_MODULE},
    pycall,
};

use crate::{CircuitRc, CircuitType};

#[derive(Debug, Clone)]
pub enum OpaqueIterativeMatcherVal {
    Py(PyObject),
    Dyn(Arc<dyn OpaqueIterativeMatcher + Send + Sync>),
}

impl<'source> FromPyObject<'source> for OpaqueIterativeMatcherVal {
    fn extract(inp: &'source PyAny) -> PyResult<Self> {
        Python::with_gil(|py| {
            Ok(Self::Py(
                atr!(SELF_MODULE, IterativeMatcher, raw).call1(py, (inp,))?,
            ))
        })
    }
}

#[pyclass]
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub struct Finished;

#[pymethods]
impl Finished {
    fn __richcmp__(&self, object: &Self, comp_op: CompareOp) -> bool {
        use_rust_comp(&self, &object, comp_op)
    }
    fn __hash__(&self) -> u64 {
        15802471944074381489
    }
}

#[derive(FromPyObject, Clone)]
pub enum UpdateImpl<Matcher> {
    Finished(Finished),
    Update(Matcher),
}

#[derive(Debug, Clone)]
// none if finished
pub struct Update<Matcher>(pub Option<Matcher>);

impl<'source, Matcher: FromPyObject<'source>> FromPyObject<'source> for Update<Matcher> {
    fn extract(inp: &'source PyAny) -> PyResult<Self> {
        let x: UpdateImpl<_> = inp.extract()?;
        let out = match x {
            UpdateImpl::Update(x) => Some(x),
            UpdateImpl::Finished(Finished) => None,
        };
        Ok(Update(out))
    }
}

impl<Matcher: IntoPy<PyObject>> IntoPy<PyObject> for Update<Matcher> {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self.0 {
            Some(x) => x.into_py(py),
            None => Finished.into_py(py),
        }
    }
}

impl<Matcher> From<Option<Matcher>> for Update<Matcher> {
    fn from(value: Option<Matcher>) -> Self {
        Update(value)
    }
}

impl<Matcher> From<Update<Matcher>> for Option<Matcher> {
    fn from(value: Update<Matcher>) -> Self {
        value.0
    }
}

impl<Matcher> Deref for Update<Matcher> {
    type Target = Option<Matcher>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Clone, Debug, FromPyObject)]
pub enum UpdatedIterativeMatcher<Matcher> {
    Many(Vec<Update<Matcher>>),
    Single(Update<Matcher>),
}

impl<Matcher: IntoPy<PyObject>> IntoPy<PyObject> for UpdatedIterativeMatcher<Matcher> {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            Self::Many(x) => x.into_py(py),
            Self::Single(x) => x.into_py(py),
        }
    }
}

impl<Matcher> From<Update<Matcher>> for UpdatedIterativeMatcher<Matcher> {
    fn from(value: Update<Matcher>) -> Self {
        Self::Single(value)
    }
}
impl<Matcher> From<Option<Matcher>> for UpdatedIterativeMatcher<Matcher> {
    fn from(value: Option<Matcher>) -> Self {
        Self::Single(value.into())
    }
}
impl<Matcher> From<Vec<Update<Matcher>>> for UpdatedIterativeMatcher<Matcher> {
    fn from(value: Vec<Update<Matcher>>) -> Self {
        Self::Many(value)
    }
}
impl<Matcher> From<Vec<Option<Matcher>>> for UpdatedIterativeMatcher<Matcher> {
    fn from(value: Vec<Option<Matcher>>) -> Self {
        value
            .into_iter()
            .map(Update::from)
            .collect::<Vec<_>>()
            .into()
    }
}

pub trait HasTerm {
    fn term() -> Self;
}

impl<Matcher: Clone + HasTerm> UpdatedIterativeMatcher<Matcher> {
    // make this pyfunction as needed
    pub fn per_child(self, num_children: usize) -> Vec<Option<Matcher>> {
        let out = match self {
            Self::Single(Update(item)) => vec![item; num_children],
            Self::Many(items) => {
                // should already have been checked!
                assert_eq!(items.len(), num_children);
                items.into_iter().map(|x| x.0).collect::<Vec<_>>()
            }
        };
        out
    }

    pub fn per_child_with_term(self, num_children: usize) -> Vec<Matcher> {
        self.per_child(num_children)
            .into_iter()
            .map(|x| x.unwrap_or_else(|| Matcher::term().into()))
            .collect()
    }

    pub fn all_finished(&self) -> bool {
        match self {
            Self::Single(x) => x.is_none(),
            Self::Many(x) => x.iter().all(|x| x.is_none()),
        }
    }

    pub fn map_updated(self, mut f: impl FnMut(Matcher) -> Matcher) -> Self {
        match self {
            Self::Single(x) => Self::Single(x.0.map(f).into()),
            Self::Many(x) => Self::Many(x.into_iter().map(|x| x.0.map(&mut f).into()).collect()),
        }
    }
}

pub fn all_finished<Matcher: Clone + HasTerm>(
    x: &Option<UpdatedIterativeMatcher<Matcher>>,
) -> bool {
    x.as_ref().map(|x| x.all_finished()).unwrap_or(false)
}
type OpaqueUpdatedIterativeMatcher = UpdatedIterativeMatcher<OpaqueIterativeMatcherVal>;

impl HasTerm for OpaqueIterativeMatcherVal {
    fn term() -> Self {
        Self::for_end_depth(0)
    }
}

#[derive(Debug, FromPyObject)]
pub struct OpaqueIterateMatchResults {
    pub updated: Option<OpaqueUpdatedIterativeMatcher>,
    pub found: bool,
}

pub trait OpaqueIterativeMatcher: Debug + ToPyObject {
    fn opaque_match_iterate(&self, circuit: CircuitRc) -> Result<OpaqueIterateMatchResults>;
}

impl ToPyObject for OpaqueIterativeMatcherVal {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        match self {
            Self::Py(obj) => obj.clone(),
            Self::Dyn(d) => d.to_object(py),
        }
    }
}

impl OpaqueIterativeMatcher for OpaqueIterativeMatcherVal {
    fn opaque_match_iterate(&self, circuit: CircuitRc) -> Result<OpaqueIterateMatchResults> {
        match self {
            Self::Py(obj) => Python::with_gil(|py| {
                Ok(obj
                    .call_method1(py, "match_iterate", (circuit,))?
                    .extract(py)?)
            }),
            Self::Dyn(d) => d.opaque_match_iterate(circuit),
        }
    }
}

#[derive(Clone, Debug)]
pub struct EndDepthOpaqueIterativeMatcher {
    pub end_depth: usize,
    pub depth: usize,
}

// why bother impl-ing if we also need to have python conversion?
// We do this so that this so depth matcher can be used in rust without assuming python library is installed!
// (same for below with Never matcher)
//
// This is a bit gross and sad...
impl ToPyObject for EndDepthOpaqueIterativeMatcher {
    fn to_object(&self, _py: Python<'_>) -> PyObject {
        assert_eq!(
            self.depth, 0,
            "shouldn't be called if depth has been updated"
        );
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("end_depth", self.end_depth).unwrap();

            SELF_MODULE
                .call_method(
                    py,
                    "restrict",
                    (OpaqueIterativeMatcherVal::noop_traversal_raw(),),
                    Some(dict),
                )
                .unwrap()
        })
    }
}

impl OpaqueIterativeMatcher for EndDepthOpaqueIterativeMatcher {
    fn opaque_match_iterate(&self, _: CircuitRc) -> Result<OpaqueIterateMatchResults> {
        let found = self.depth < self.end_depth;
        if self.depth < self.end_depth.saturating_sub(1) {
            Ok(OpaqueIterateMatchResults {
                updated: Some(
                    Some(OpaqueIterativeMatcherVal::Dyn(Arc::new(Self {
                        end_depth: self.end_depth,
                        depth: self.depth + 1,
                    })))
                    .into(),
                ),
                found,
            })
        } else {
            Ok(OpaqueIterateMatchResults {
                updated: Some(None.into()),
                found,
            })
        }
    }
}
#[derive(Clone, Debug)]
pub struct NeverOpaqueIterativeMatcher;

impl ToPyObject for NeverOpaqueIterativeMatcher {
    fn to_object(&self, _: Python<'_>) -> PyObject {
        pycall!(atr!(SELF_MODULE, IterativeMatcher, raw), (false,))
    }
}

impl OpaqueIterativeMatcher for NeverOpaqueIterativeMatcher {
    fn opaque_match_iterate(&self, _: CircuitRc) -> Result<OpaqueIterateMatchResults> {
        Ok(OpaqueIterateMatchResults {
            updated: None,
            found: false,
        })
    }
}

macro_rules! defer_to_py {
    (
    $(
        fn $fn_name:ident(
            &self,
            $($arg:ident : $arg_ty:ty),* $(,)?
        ) -> $ret_ty:ty;
    )*
    ) => (
    $(
        pub(super) fn $fn_name(
            &self,
            $($arg : $arg_ty),*
        ) -> $ret_ty {
            Python::with_gil(|py| {
                atr!(self.to_object(py), $fn_name, raw).call1(py, ($($arg,)*)).map(|x| x.extract(py).unwrap())
            })
        }
    )*

    )
}

impl OpaqueIterativeMatcherVal {
    pub fn noop_traversal_raw() -> PyObject {
        pycall!(atr!(SELF_MODULE, new_traversal, raw), (), raw)
    }

    pub fn for_end_depth(end_depth: usize) -> Self {
        Self::Dyn(Arc::new(EndDepthOpaqueIterativeMatcher {
            end_depth,
            depth: 0,
        }))
    }

    pub fn never() -> Self {
        Self::Dyn(Arc::new(NeverOpaqueIterativeMatcher))
    }

    pub fn op_to_object(x: &Option<Self>) -> PyObject {
        Python::with_gil(|py| {
            x.as_ref()
                .map(|x| x.to_object(py))
                .unwrap_or_else(|| Self::noop_traversal_raw())
        })
    }

    // below methods should just be used for PyCircuitBase
    defer_to_py!(
        fn update(
            &self,
            circuit: CircuitRc,
            transform: PyObject,
            cache_transform: bool,
            cache_update: bool,
            fancy_validate: bool,
            assert_any_found: bool,
        ) -> PyResult<CircuitRc>;

        fn get(&self, circuit: CircuitRc, fancy_validate: bool) -> PyResult<BTreeSet<CircuitRc>>;

        fn get_unique_op(
            &self,
            circuit: CircuitRc,
            fancy_validate: bool,
        ) -> PyResult<Option<CircuitRc>>;

        fn get_unique(&self, circuit: CircuitRc, fancy_validate: bool) -> PyResult<CircuitRc>;

        fn get_paths(&self, circuit: CircuitRc) -> PyResult<BTreeMap<CircuitRc, Vec<CircuitRc>>>;
        fn get_all_paths(
            &self,
            circuit: CircuitRc,
        ) -> PyResult<BTreeMap<CircuitRc, Vec<Vec<CircuitRc>>>>;

        fn are_any_found(&self, circuit: CircuitRc) -> PyResult<bool>;
    );
}

pub fn get_opaque_type_matcher(
    circuit_type: CircuitType,
    traversal: Option<OpaqueIterativeMatcherVal>,
) -> OpaqueIterativeMatcherVal {
    Python::with_gil(|py| {
        let kwargs = PyDict::new(py);
        let not_type_matcher = SELF_MODULE
            .getattr(py, "Matcher")
            .unwrap()
            .call1(py, (circuit_type,))
            .unwrap()
            .call_method0(py, "new_not")
            .unwrap();
        kwargs.set_item("term_early_at", not_type_matcher).unwrap();
        SELF_MODULE
            .call_method(
                py,
                "restrict",
                (OpaqueIterativeMatcherVal::op_to_object(&traversal),),
                Some(kwargs),
            )
            .unwrap()
            .extract(py)
            .unwrap()
    })
}
