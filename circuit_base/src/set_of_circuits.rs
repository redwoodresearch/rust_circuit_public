use pyo3::prelude::*;
use rr_util::util::HashBytes;
use rustc_hash::FxHashSet as HashSet;

use crate::{CircuitNode, CircuitRc};
#[pyclass]
pub struct SetOfCircuitIdentities(HashSet<HashBytes>); // could use something that avoids rehashing, but not worth
#[pymethods]
impl SetOfCircuitIdentities {
    #[new]
    #[pyo3(signature=(hs = Default::default()))]
    pub fn new(hs: HashSet<HashBytes>) -> Self {
        Self(hs)
    }
    pub fn union(&self, other: &SetOfCircuitIdentities) -> Self {
        Self(self.0.iter().chain(&other.0).cloned().collect())
    }
    pub fn intersection(&self, other: &SetOfCircuitIdentities) -> Self {
        Self(
            self.0
                .iter()
                .filter(|x| other.0.contains(*x))
                .cloned()
                .collect(),
        )
    }
    pub fn extend(&mut self, other: &SetOfCircuitIdentities) {
        self.0.extend(&other.0)
    }
    pub fn insert(&mut self, circ: CircuitRc) {
        self.0.insert(circ.info().hash);
    }
    pub fn __contains__(&self, key: CircuitRc) -> bool {
        self.0.contains(&key.info().hash)
    }
    pub fn __len__(&self) -> usize {
        self.0.len()
    }
}
impl Into<SetOfCircuitIdentities> for HashSet<HashBytes> {
    fn into(self) -> SetOfCircuitIdentities {
        SetOfCircuitIdentities(self)
    }
}
