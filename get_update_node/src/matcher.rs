//! TODO: fuzzy string matching for debugging (maybe???)

use std::{
    collections::BTreeSet,
    fmt::{self, Debug, Display},
    hash,
    sync::{Arc, Mutex},
    vec,
};

use anyhow::{bail, Context, Result};
use circuit_base::{visit_circuit, CircuitNode, CircuitNodeUnion, CircuitRc, CircuitType};
use pyo3::{exceptions::PyTypeError, prelude::*, AsPyPointer};
use regex::{Captures, Regex};
use rr_util::{
    eq_by_big_hash::EqByBigHash,
    make_single_many,
    name::Name,
    py_types::PyCallable,
    setup_callable, simple_default, simple_from,
    util::{arc_unwrap_or_clone, EmptySingleMany as ESM},
};
use thiserror::Error;
use uuid::uuid;

use crate::{restrict, BoundAnyFound, IterativeMatcher, IterativeMatcherRc};

make_single_many!(TypeTags, CircuitType, BTreeSet);
make_single_many!(Strings, String, BTreeSet);
make_single_many!(Names, Name, BTreeSet);
make_single_many!(Circuits, CircuitRc, BTreeSet);

#[derive(Clone, FromPyObject)]
pub enum MatcherFromPyBase {
    Always(bool),
    Name(Names),
    Type(TypeTags),
    Regex(RegexWrap),
    EqM(Circuits),
    AnyFound(BoundAnyFound),
    Matcher(Matcher),
}

#[derive(Clone, FromPyObject)]
pub enum MatcherFromPy {
    Base(MatcherFromPyBase),
    #[pyo3(transparent)]
    PyFunc(PyCallable),
}

#[derive(Clone, Debug)]
pub enum MatcherData {
    Always(bool),
    Name(BTreeSet<Name>),
    Type(BTreeSet<CircuitType>),
    Regex(RegexWrap),
    EqM(BTreeSet<CircuitRc>),
    AnyFound(Arc<Mutex<BoundAnyFound>>), // sad mutex
    Raw(RawMatcher),
    PyFunc(PyCallable),
    Not(MatcherRc),
    Any(Vec<MatcherRc>),
    All(Vec<MatcherRc>),
}

setup_callable!(Matcher, MatcherData, MatcherFromPy, call (circuit : CircuitRc) -> bool);

simple_from!(|x: MatcherFromPyBase| -> MatcherFromPy { MatcherFromPy::Base(x) });
simple_from!(|x: MatcherFromPyBase| -> MatcherRc { Matcher::from(x).rc() });
simple_default!(MatcherFromPyBase { Self::Always(true) });
simple_default!(MatcherFromPy { MatcherFromPyBase::default().into() });
simple_default!(Matcher { MatcherFromPy::default().into() });
simple_default!(MatcherRc { Matcher::default().into() });

impl From<MatcherFromPyBase> for Matcher {
    fn from(m: MatcherFromPyBase) -> Self {
        match m {
            MatcherFromPyBase::Always(x) => MatcherData::Always(x),
            MatcherFromPyBase::Name(x) => MatcherData::Name(x.into_many()),
            MatcherFromPyBase::Type(x) => MatcherData::Type(x.into_many()),
            MatcherFromPyBase::Regex(x) => MatcherData::Regex(x),
            MatcherFromPyBase::EqM(x) => MatcherData::EqM(x.into_many()),
            MatcherFromPyBase::AnyFound(x) => MatcherData::AnyFound(Arc::new(Mutex::new(x))),
            MatcherFromPyBase::Matcher(x) => return x,
        }
        .into()
    }
}
impl From<MatcherFromPy> for Matcher {
    fn from(m: MatcherFromPy) -> Self {
        match m {
            MatcherFromPy::Base(x) => x.into(),
            MatcherFromPy::PyFunc(x) => MatcherData::PyFunc(x).into(),
        }
    }
}

impl MatcherFromPyBase {
    pub fn to_matcher(self) -> Matcher {
        self.into()
    }
}

#[pyclass]
#[pyo3(name = "Regex")]
#[derive(Debug, Clone)]
pub struct RegexWrap {
    regex: Regex,
    pattern: String,
    escape_dot: bool,
}

impl RegexWrap {}

#[pymethods]
impl RegexWrap {
    #[new]
    #[pyo3(signature=(pattern, escape_dot = true))]
    pub fn new(pattern: String, escape_dot: bool) -> Result<Self> {
        let new_pattern = if escape_dot {
            Regex::new(r"(\\\.)|(\.)")
                .unwrap()
                .replace_all(&pattern, |captures: &Captures| {
                    if captures.get(1).is_some() {
                        ".".to_owned()
                    } else {
                        r"\.".to_owned()
                    }
                })
                .to_string()
        } else {
            pattern.clone()
        };
        let regex = Regex::new(&new_pattern)?;

        Ok(Self {
            regex,
            pattern,
            escape_dot,
        })
    }

    pub fn call(&self, s: &str) -> bool {
        self.regex.is_match(s)
    }

    pub fn __repr__(&self) -> String {
        let escape_dot_str = if self.escape_dot { "True" } else { "False" };
        format!(
            "Regex(\"{}\", escape_dot={})",
            self.regex
                .to_string()
                .replace('\\', r"\\")
                .replace('"', "\\\""),
            escape_dot_str
        )
    }

    #[getter]
    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    #[getter]
    pub fn escape_dot(&self) -> bool {
        self.escape_dot
    }
}

impl MatcherData {
    fn uuid(&self) -> [u8; 16] {
        match self {
            Self::Always(_) => uuid!("70ca26a9-43be-4d8f-8962-655697e50b2a"),
            Self::Name(_) => uuid!("f7d89984-abb7-4ab5-a685-6c5cb65624da"),
            Self::Type(_) => uuid!("16afef04-e938-457b-93fc-6e781e97a63d"),
            Self::Regex(_) => uuid!("29859287-5945-4d04-8c01-6974a2bd3a1d"),
            Self::EqM(_) => uuid!("3d9ee0b2-5075-47f3-9bcd-4e77eb14e5f9"),
            Self::AnyFound(_) => uuid!("9afd5f57-f738-4e47-a109-da7803546a08"),
            Self::Raw(_) => uuid!("eff8833f-b3c6-4842-8f9b-4404f7abc356"),
            Self::PyFunc(_) => uuid!("f5934590-a3c1-471d-a0a3-1dece4344326"),
            Self::Not(_) => uuid!("b4e14744-4a07-40e4-acd4-a0001e8ffbb0"),
            Self::Any(_) => uuid!("99e756b9-1ce2-49ba-98b0-56f10578fa76"),
            Self::All(_) => uuid!("c43f7201-6869-4fa9-b494-472a56d0699a"),
        }
        .into_bytes()
    }

    fn item_hash(&self, hasher: &mut blake3::Hasher) {
        match self {
            Self::Always(x) => {
                hasher.update(&[*x as u8]);
            }
            Self::Name(x) => {
                for s in x {
                    hasher.update(s.as_bytes());
                    // variable size so we need to delimit
                    hasher.update(&uuid!("a4b56c19-c5d2-41c2-be29-1742955c1299").into_bytes());
                }
            }
            Self::Type(x) => {
                for t in x {
                    hasher.update(&(*t as u32).to_le_bytes());
                }
            }
            Self::Regex(x) => {
                hasher.update(&[x.escape_dot as u8]);
                hasher.update(x.pattern.as_bytes());
            }
            Self::EqM(x) => {
                for t in x {
                    hasher.update(&t.info().hash);
                }
            }
            Self::AnyFound(x) => {
                hasher.update(&x.lock().unwrap().matcher.hash());
            }
            Self::Raw(x) => {
                hasher.update(&(Arc::as_ptr(&x.0) as *const () as usize).to_le_bytes());
            }
            Self::PyFunc(x) => {
                hasher.update(&(x.as_ptr() as usize).to_le_bytes());
            }
            Self::Not(x) => {
                hasher.update(&x.hash);
            }
            Self::Any(x) | Self::All(x) => {
                for sub in x {
                    hasher.update(&sub.hash);
                }
            }
        }
    }
}

impl Matcher {
    pub fn and(self, other: MatcherRc) -> Self {
        Self::all(vec![self.rc(), other])
    }

    pub fn or(self, other: MatcherRc) -> Self {
        Self::any(vec![self.rc(), other])
    }

    pub fn to_iterative_matcher_rc(&self) -> IterativeMatcherRc {
        self.to_iterative_matcher().rc()
    }
    // TODO: more rust niceness funcs like the py ones!

    pub fn validate_matched(&self, matched: &BTreeSet<CircuitRc>) -> Result<()> {
        fn run_on_set<T: Eq + hash::Hash + Ord + Debug + Clone, D: Display>(
            matcher_set: &BTreeSet<T>,
            found_set: &BTreeSet<T>,
            items: &str,
            matcher_type: &str,
            convert: impl Fn(&BTreeSet<T>) -> D,
        ) -> Result<()> {
            if !matcher_set.is_subset(found_set) {
                bail!(
                    concat!(
                        "Didn't match all {} contained in this {} matcher!\n",
                        "matcher: {}, found: {}, missing: {}"
                    ),
                    items,
                    matcher_type,
                    convert(matcher_set),
                    convert(found_set),
                    convert(
                        &matcher_set
                            .difference(&found_set)
                            .into_iter()
                            .cloned()
                            .collect()
                    )
                )
            } else {
                Ok(())
            }
        }

        match &self.data {
            MatcherData::Name(names) => {
                let found_names = matched.into_iter().filter_map(|x| x.info().name).collect();
                run_on_set(names, &found_names, "names", "name", |x| format!("{:?}", x))
            }
            MatcherData::Type(types) => {
                let found_types = matched.into_iter().map(|x| x.type_tag()).collect();
                run_on_set(types, &found_types, "types", "type", |x| format!("{:?}", x))
            }
            MatcherData::EqM(circs) => run_on_set(
                circs,
                &matched,
                "circuits",
                "circuit equality",
                |x| -> String {
                    "[".to_owned()
                        + &x.into_iter()
                            .map(|x| format!("{}({:?})", x.variant_string(), x.info().name))
                            .collect::<Vec<_>>()
                            .join(", ")
                        + "]"
                },
            ),
            MatcherData::AnyFound(_) => Ok(()), // TODO
            MatcherData::Not(_) => Ok(()),      // nothing we can check here!
            MatcherData::Any(matchers) | MatcherData::All(matchers) => {
                // TODO: improve errors here?
                for m in matchers {
                    m.validate_matched(matched)?;
                }

                Ok(())
            }
            MatcherData::Regex(_)
            | MatcherData::Always(_)
            | MatcherData::Raw(_)
            | MatcherData::PyFunc(_) => {
                for c in matched {
                    if self.call(c.clone())? {
                        return Ok(());
                    }
                }
                bail!(
                    "This matcher matched nothing: {:?}\ncircuits: {:?}",
                    self,
                    matched
                )
            }
        }
    }
}

/// if needed, we could add explicit types
#[pymethods]
impl Matcher {
    #[new]
    #[pyo3(signature=(*inps))]
    fn py_new(inps: Vec<MatcherRc>) -> Self {
        match inps.into() {
            ESM::Empty => MatcherData::Always(false).into(),
            ESM::Single(x) => arc_unwrap_or_clone(x.0),
            ESM::Many(x) => Self::any(x),
        }
    }

    pub fn call(&self, circuit: CircuitRc) -> Result<bool> {
        let ret = match &self.data {
            &MatcherData::Always(x) => x,
            MatcherData::Name(names) => circuit
                .info()
                .name
                .map(|x| names.contains(&x))
                .unwrap_or(false),
            MatcherData::Type(type_tags) => type_tags.contains(&circuit.type_tag()),
            MatcherData::Regex(r) => circuit
                .info()
                .name
                .map(|x| r.call(x.str()))
                .unwrap_or(false),
            MatcherData::EqM(circs) => circs.contains(&circuit),
            MatcherData::AnyFound(any_found) => any_found.lock().unwrap().are_any_found(circuit)?,
            MatcherData::Raw(f) => f.0(circuit)?,
            MatcherData::PyFunc(pyfunc) => Python::with_gil(|py| {
                pyfunc
                    .call1(py, (circuit,))
                    .context("calling python matcher failed")
                    .and_then(|r| {
                        r.extract(py)
                            .context("extracting from python matcher failed")
                    })
            })?,
            MatcherData::Not(m) => !m.call(circuit)?,
            MatcherData::Any(ms) => {
                for m in ms {
                    if m.call(circuit.clone())? {
                        return Ok(true);
                    }
                }
                false
            }
            MatcherData::All(ms) => {
                for m in ms {
                    if !m.call(circuit.clone())? {
                        return Ok(false);
                    }
                }
                true
            }
        };

        Ok(ret)
    }

    // TODO: write flatten/simplify method if we want the extra speed + niceness!

    #[pyo3(name = "validate_matched")]
    fn validate_matched_py(&self, matched: BTreeSet<CircuitRc>) -> Result<()> {
        self.validate_matched(&matched)
    }

    pub fn get_first(&self, circuit: CircuitRc) -> Result<Option<CircuitRc>> {
        #[derive(Error, Debug, Clone)]
        #[error("Stop iteration")]
        struct StopIteration;

        let mut result: Option<CircuitRc> = None;
        let err = visit_circuit(circuit, |x| {
            if self.call(x.clone())? {
                result = Some(x);
                bail!(StopIteration);
            }
            Ok(())
        });

        if let Err(e) = err {
            match e.downcast_ref::<StopIteration>() {
                Some(StopIteration) => (),
                None => return Err(e),
            }
        }

        Ok(result)
    }

    #[staticmethod]
    pub fn true_matcher() -> Self {
        MatcherData::Always(true).into()
    }

    #[staticmethod]
    pub fn false_matcher() -> Self {
        MatcherData::Always(false).into()
    }

    #[staticmethod]
    #[pyo3(signature=(pattern, escape_dot = true))]
    pub fn regex(pattern: String, escape_dot: bool) -> Result<Self> {
        Ok(MatcherData::Regex(RegexWrap::new(pattern, escape_dot)?).into())
    }

    #[staticmethod]
    pub fn match_any_found(finder: IterativeMatcherRc) -> Self {
        MatcherData::AnyFound(Arc::new(Mutex::new(finder.any_found()))).into()
    }

    #[staticmethod]
    pub fn match_any_child_found(finder: IterativeMatcherRc) -> Self {
        Self::match_any_found(
            restrict(
                finder,
                false,
                Some(1),
                None,
                MatcherFromPyBase::Always(false).into(),
            )
            .into(),
        )
    }

    #[staticmethod]
    #[pyo3(signature=(*types))]
    pub fn types(types: Vec<CircuitType>) -> Self {
        MatcherData::Type(types.into_iter().collect::<BTreeSet<_>>()).into()
    }

    #[staticmethod]
    #[pyo3(signature=(*circuits))]
    pub fn circuits(circuits: Vec<CircuitRc>) -> Self {
        MatcherData::EqM(circuits.into_iter().collect::<BTreeSet<_>>()).into()
    }

    #[staticmethod]
    #[pyo3(signature=(*matchers))]
    pub fn all(matchers: Vec<MatcherRc>) -> Self {
        MatcherData::All(matchers).into()
    }

    #[staticmethod]
    #[pyo3(signature=(*matchers))]
    pub fn any(matchers: Vec<MatcherRc>) -> Self {
        MatcherData::Any(matchers).into()
    }

    #[pyo3(name = "new_not")]
    pub fn not(&self) -> Self {
        MatcherData::Not(self.crc()).into()
    }

    #[pyo3(signature=(*others))]
    pub fn new_and(&self, others: Vec<MatcherRc>) -> Self {
        Self::all([self.crc()].into_iter().chain(others).collect())
    }

    #[pyo3(signature=(*others))]
    pub fn new_or(&self, others: Vec<MatcherRc>) -> Self {
        Self::any([self.crc()].into_iter().chain(others).collect())
    }

    fn __invert__(&self) -> Self {
        self.not()
    }
    fn __and__(&self, other: MatcherRc) -> Self {
        self.clone().and(other)
    }
    fn __or__(&self, other: MatcherRc) -> Self {
        self.clone().or(other)
    }
    fn __rand__(&self, other: MatcherRc) -> Self {
        arc_unwrap_or_clone(other.0).and(self.crc())
    }
    fn __ror__(&self, other: MatcherRc) -> Self {
        arc_unwrap_or_clone(other.0).or(self.crc())
    }
    fn __bool__(&self) -> PyResult<bool> {
        PyResult::Err(PyTypeError::new_err("Matcher was coerced to a boolean. Did you mean to use & or | instead of \"and\" or \"or\"?"))
    }

    pub fn to_iterative_matcher(&self) -> IterativeMatcher {
        self.clone().into()
    }
}
