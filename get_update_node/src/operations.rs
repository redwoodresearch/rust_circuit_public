use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use anyhow::{anyhow, bail, Result};
use circuit_base::{
    expand_node::{expand_node, ReplaceMapRc},
    set_of_circuits::SetOfCircuitIdentities,
    CircuitNode, CircuitRc,
};
use macro_rules_attribute::apply;
use pyo3::prelude::*;
use rr_util::{
    cached_method,
    caching::FastUnboundedCache,
    eq_by_big_hash::EqByBigHash,
    util::{transpose, HashBytes},
};
use rustc_hash::FxHashSet as HashSet;

use crate::{
    iterative_matcher::{all_finished, function_per_child, per_child, per_child_with_term},
    IterateMatchResults, IterativeMatcherRc, Transform, TransformData, TransformRc,
};

#[derive(Debug, Clone)]
pub(super) struct UpdateOutput {
    pub(super) updated: CircuitRc,
    pub(super) any_found: bool,
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct Updater {
    #[pyo3(get)]
    pub(super) transform: TransformRc,
    #[pyo3(get)]
    pub(super) cache_transform: bool,
    #[pyo3(get)]
    pub(super) cache_update: bool,
    #[pyo3(get, set)]
    pub(super) default_fancy_validate: bool,
    #[pyo3(get, set)]
    pub(super) default_assert_any_found: bool,
    pub(super) transform_cache: FastUnboundedCache<HashBytes, CircuitRc>,
    pub(super) updated_cache: FastUnboundedCache<(HashBytes, IterativeMatcherRc), UpdateOutput>,
    pub(super) validation_getter: Getter,
}

impl Default for Updater {
    fn default() -> Self {
        Self {
            transform: Transform::ident().into(),
            cache_transform: true,
            cache_update: true,
            default_fancy_validate: false,
            default_assert_any_found: false,
            transform_cache: FastUnboundedCache::default(),
            updated_cache: FastUnboundedCache::default(),
            validation_getter: Default::default(),
        }
    }
}

#[pymethods]
impl Updater {
    #[new]
    #[pyo3(signature=(
        transform,
        cache_transform = Updater::default().cache_transform,
        cache_update = Updater::default().cache_update,
        default_fancy_validate = Updater::default().default_fancy_validate,
        default_assert_any_found = Updater::default().default_assert_any_found
    ))]
    pub fn new(
        transform: TransformRc,
        cache_transform: bool,
        cache_update: bool,
        default_fancy_validate: bool,
        default_assert_any_found: bool,
    ) -> Self {
        Self {
            transform,
            cache_transform,
            cache_update,
            default_fancy_validate,
            default_assert_any_found,
            ..Default::default()
        }
    }

    fn __call__(
        &mut self,
        _py: Python<'_>,
        circuit: CircuitRc,
        matcher: IterativeMatcherRc,
        fancy_validate: Option<bool>,
        assert_any_found: Option<bool>,
    ) -> Result<CircuitRc> {
        self.update(circuit, matcher, fancy_validate, assert_any_found)
    }

    pub fn update(
        &mut self,
        circuit: CircuitRc,
        matcher: IterativeMatcherRc,
        fancy_validate: Option<bool>,
        assert_any_found: Option<bool>,
    ) -> Result<CircuitRc> {
        if fancy_validate.unwrap_or(self.default_fancy_validate) {
            self.validation_getter
                .validate(circuit.clone(), matcher.clone())?;
        }
        let out = self.update_impl(circuit.crc(), matcher.crc())?;
        if assert_any_found.unwrap_or(self.default_assert_any_found) && !out.any_found {
            bail!(
                concat!(
                    "No matches found during update\n",
                    "matcher: {matcher:?}, circuit: {circuit:?}"
                ),
                matcher = matcher,
                circuit = circuit
            );
        }
        Ok(out.updated)
    }

    pub fn bind(&self, matcher: IterativeMatcherRc) -> BoundUpdater {
        BoundUpdater {
            updater: self.clone(),
            matcher,
        }
    }
}

impl Updater {
    #[apply(cached_method)]
    #[self_id(self_)]
    #[key((circuit.info().hash, matcher.clone()))]
    #[use_try]
    #[cache_expr(updated_cache)]
    fn update_impl(
        &mut self,
        circuit: CircuitRc,
        matcher: IterativeMatcherRc,
    ) -> Result<UpdateOutput> {
        let IterateMatchResults { updated, found } = matcher.match_iterate(circuit.clone())?;

        let mut any_found = false;
        let mut new_circuit =
            function_per_child(updated, matcher, circuit.clone(), |circuit, new_matcher| {
                let child_out = self_.update_impl(circuit, new_matcher)?;
                any_found |= child_out.any_found;
                Ok(child_out.updated)
            })?;

        if found {
            any_found = true;
            if !matches!(self_.transform.data(), TransformData::Ident) {
                new_circuit = self_.run_transform(new_circuit)?;
            }
        }

        Ok(UpdateOutput {
            updated: new_circuit,
            any_found,
        })
    }

    #[apply(cached_method)]
    #[self_id(self_)]
    #[key(circuit.info().hash)]
    #[use_try]
    #[cache_expr(transform_cache)]
    fn run_transform(&mut self, circuit: CircuitRc) -> Result<CircuitRc> {
        self_.transform.run(circuit)
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct BoundUpdater {
    #[pyo3(get, set)]
    pub updater: Updater,
    #[pyo3(get, set)]
    pub matcher: IterativeMatcherRc,
}

#[pymethods]
impl BoundUpdater {
    #[new]
    pub fn new(updater: Updater, matcher: IterativeMatcherRc) -> Self {
        Self { updater, matcher }
    }

    fn __call__(
        &mut self,
        _py: Python<'_>,
        circuit: CircuitRc,
        fancy_validate: Option<bool>,
        assert_any_found: Option<bool>,
    ) -> Result<CircuitRc> {
        self.update(circuit, fancy_validate, assert_any_found)
    }

    pub fn update(
        &mut self,
        circuit: CircuitRc,
        fancy_validate: Option<bool>,
        assert_any_found: Option<bool>,
    ) -> Result<CircuitRc> {
        self.updater.update(
            circuit,
            self.matcher.clone(),
            fancy_validate,
            assert_any_found,
        )
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct Getter {
    #[pyo3(get, set)]
    pub(super) default_fancy_validate: bool,
    pub(super) cache: FastUnboundedCache<(HashBytes, IterativeMatcherRc), BTreeSet<CircuitRc>>,
    pub(super) paths_cache:
        FastUnboundedCache<(HashBytes, IterativeMatcherRc), BTreeMap<CircuitRc, Vec<CircuitRc>>>,
    pub(super) all_paths_cache:
        FastUnboundedCache<(HashBytes, IterativeMatcherRc), Vec<CircuitLinkedList>>,
}

impl Default for Getter {
    fn default() -> Self {
        Self {
            default_fancy_validate: false,
            cache: FastUnboundedCache::default(),
            paths_cache: FastUnboundedCache::default(),
            all_paths_cache: FastUnboundedCache::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub enum CircuitLinkedList {
    None,
    Some(CircuitRc, Arc<CircuitLinkedList>),
    // this is arc bc its used in cached crate which uses mutex blah blah blah
    // it isn't actually meant to (or likely actually) go between threads
    // maybe i should just unsafe impl send instead
}

impl CircuitLinkedList {
    fn list(&self) -> Vec<CircuitRc> {
        let mut out = vec![];
        let mut head = self;
        loop {
            if let CircuitLinkedList::Some(c, n) = &head {
                out.push(c.clone());
                head = &*n;
            } else {
                break;
            }
        }
        out
    }
}

#[pymethods]
impl Getter {
    #[new]
    #[pyo3(signature=(default_fancy_validate = Getter::default().default_fancy_validate))]
    pub fn new(default_fancy_validate: bool) -> Self {
        Self {
            default_fancy_validate,
            ..Default::default()
        }
    }

    fn __call__(
        &mut self,
        _py: Python<'_>,
        circuit: CircuitRc,
        matcher: IterativeMatcherRc,
        fancy_validate: Option<bool>,
    ) -> Result<BTreeSet<CircuitRc>> {
        self.get(circuit, matcher, fancy_validate)
    }

    pub fn get(
        &mut self,
        circuit: CircuitRc,
        matcher: IterativeMatcherRc,
        fancy_validate: Option<bool>,
    ) -> Result<BTreeSet<CircuitRc>> {
        let out = self.get_impl(circuit, matcher.clone())?;
        if fancy_validate.unwrap_or(self.default_fancy_validate) {
            matcher.validate_matched(&out)?;
        }
        Ok(out)
    }

    pub fn get_paths(
        &mut self,
        circuit: CircuitRc,
        matcher: IterativeMatcherRc,
    ) -> Result<BTreeMap<CircuitRc, Vec<CircuitRc>>> {
        self.get_paths_impl(circuit, matcher) // TODO: add validation?
    }

    pub fn get_all_circuits_in_paths(
        &mut self,
        circuit: CircuitRc,
        matcher: IterativeMatcherRc,
    ) -> Result<SetOfCircuitIdentities> {
        Ok(self
            .get_all_paths_impl(circuit, matcher)?
            .into_values()
            .flatten()
            .flatten()
            .map(|c| c.info().hash)
            .collect::<HashSet<HashBytes>>()
            .into()) // TODO: add validation?
    }

    pub fn get_all_paths(
        &mut self,
        circuit: CircuitRc,
        matcher: IterativeMatcherRc,
    ) -> Result<BTreeMap<CircuitRc, Vec<Vec<CircuitRc>>>> {
        self.get_all_paths_impl(circuit, matcher) // TODO: add validation?
    }

    pub fn get_unique_op(
        &mut self,
        circuit: CircuitRc,
        matcher: IterativeMatcherRc,
        fancy_validate: Option<bool>,
    ) -> Result<Option<CircuitRc>> {
        let out = self.get(circuit, matcher, fancy_validate)?;
        if out.len() > 1 {
            bail!("found {} matches which is > 1", out.len());
        }
        Ok(out.into_iter().next())
    }

    pub fn get_unique(
        &mut self,
        circuit: CircuitRc,
        matcher: IterativeMatcherRc,
        fancy_validate: Option<bool>,
    ) -> Result<CircuitRc> {
        self.get_unique_op(circuit, matcher, fancy_validate)?
            .ok_or_else(|| anyhow!("found no matches!"))
    }

    pub fn validate(&mut self, circuit: CircuitRc, matcher: IterativeMatcherRc) -> Result<()> {
        self.get(circuit, matcher, Some(true))?;
        Ok(())
    }

    pub fn bind(&self, matcher: IterativeMatcherRc) -> BoundGetter {
        BoundGetter {
            getter: self.clone(),
            matcher,
        }
    }

    // TODO: add support for paths as needed!
}

impl Getter {
    #[apply(cached_method)]
    #[self_id(self_)]
    #[key((circuit.info().hash, matcher.clone()))]
    #[use_try]
    #[cache_expr(cache)]
    fn get_impl(
        &mut self,
        circuit: CircuitRc,
        matcher: IterativeMatcherRc,
    ) -> Result<BTreeSet<CircuitRc>> {
        let IterateMatchResults { updated, found } = matcher.match_iterate(circuit.clone())?;

        let mut out: BTreeSet<CircuitRc> = Default::default();
        if found {
            out.insert(circuit.clone());
        }
        if !all_finished(&updated) {
            let new_matchers = per_child(updated, matcher, circuit.num_children());
            for (child, new_matcher) in circuit.children().zip(new_matchers) {
                if let Some(new_matcher) = new_matcher {
                    out.extend(self_.get_impl(child, new_matcher)?);
                }
            }
        }
        Ok(out)
    }

    fn get_all_paths_impl(
        &mut self,
        circuit: CircuitRc,
        matcher: IterativeMatcherRc,
    ) -> Result<BTreeMap<CircuitRc, Vec<Vec<CircuitRc>>>> {
        let result_raw = self.get_all_paths_impl_intermediate(circuit, matcher)?;
        let mut result: BTreeMap<CircuitRc, Vec<Vec<CircuitRc>>> = Default::default();
        for ll in result_raw {
            let mut listy = ll.list();
            listy.reverse();
            let foundy = listy[0].clone();
            if let Some(existing) = result.get_mut(&foundy) {
                existing.push(listy);
            } else {
                result.insert(foundy, vec![listy]);
            }
        }
        Ok(result)
    }

    #[apply(cached_method)]
    #[self_id(self_)]
    #[key((circuit.info().hash, matcher.clone()))]
    #[use_try]
    #[cache_expr(all_paths_cache)]
    fn get_all_paths_impl_intermediate(
        &mut self,
        circuit: CircuitRc,
        matcher: IterativeMatcherRc,
    ) -> Result<Vec<CircuitLinkedList>> {
        let IterateMatchResults { updated, found } = matcher.match_iterate(circuit.clone())?;

        let mut out: Vec<CircuitLinkedList> = Default::default();
        if found {
            out.push(CircuitLinkedList::Some(
                circuit.clone(),
                Arc::new(CircuitLinkedList::None),
            ));
        }
        if !all_finished(&updated) {
            let new_matchers = per_child(updated, matcher, circuit.num_children());
            for (child, new_matcher) in circuit.children().zip(new_matchers) {
                if let Some(new_matcher) = new_matcher {
                    let child_paths = self_.get_all_paths_impl_intermediate(child, new_matcher)?;
                    out.extend(
                        child_paths
                            .into_iter()
                            .map(|ll| CircuitLinkedList::Some(circuit.clone(), Arc::new(ll))),
                    );
                }
            }
        }
        Ok(out)
    }

    #[apply(cached_method)]
    #[self_id(self_)]
    #[key((circuit.info().hash, matcher.clone()))]
    #[use_try]
    #[cache_expr(paths_cache)]
    fn get_paths_impl(
        &mut self,
        circuit: CircuitRc,
        matcher: IterativeMatcherRc,
    ) -> Result<BTreeMap<CircuitRc, Vec<CircuitRc>>> {
        let IterateMatchResults { updated, found } = matcher.match_iterate(circuit.clone())?;

        let mut out: BTreeMap<CircuitRc, Vec<CircuitRc>> = Default::default();
        if found {
            out.insert(circuit.clone(), vec![circuit.clone()]);
        }
        if !all_finished(&updated) {
            let new_matchers = per_child(updated, matcher, circuit.num_children());
            for (child, new_matcher) in circuit.children().zip(new_matchers) {
                if let Some(new_matcher) = new_matcher {
                    let child_paths = self_.get_paths_impl(child, new_matcher)?;
                    for (child, mut path) in child_paths.into_iter() {
                        path.push(circuit.clone());
                        out.insert(child, path);
                    }
                }
            }
        }
        Ok(out)
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct BoundGetter {
    #[pyo3(get, set)]
    pub getter: Getter,
    #[pyo3(get, set)]
    pub matcher: IterativeMatcherRc,
}

#[pymethods]
impl BoundGetter {
    #[new]
    pub fn new(getter: Getter, matcher: IterativeMatcherRc) -> Self {
        Self {
            getter,
            matcher: matcher.into(),
        }
    }

    fn __call__(
        &mut self,
        _py: Python<'_>,
        circuit: CircuitRc,
        fancy_validate: Option<bool>,
    ) -> Result<BTreeSet<CircuitRc>> {
        self.get(circuit, fancy_validate)
    }

    pub fn get(
        &mut self,
        circuit: CircuitRc,
        fancy_validate: Option<bool>,
    ) -> Result<BTreeSet<CircuitRc>> {
        self.getter
            .get(circuit, self.matcher.clone(), fancy_validate)
    }

    pub fn get_unique_op(
        &mut self,
        circuit: CircuitRc,
        fancy_validate: Option<bool>,
    ) -> Result<Option<CircuitRc>> {
        self.getter
            .get_unique_op(circuit, self.matcher.clone(), fancy_validate)
    }

    pub fn get_unique(
        &mut self,
        circuit: CircuitRc,
        fancy_validate: Option<bool>,
    ) -> Result<CircuitRc> {
        self.getter
            .get_unique(circuit, self.matcher.clone(), fancy_validate)
    }

    pub fn get_paths(&mut self, circuit: CircuitRc) -> Result<BTreeMap<CircuitRc, Vec<CircuitRc>>> {
        self.getter.get_paths(circuit, self.matcher.clone())
    }

    pub fn get_all_paths(
        &mut self,
        circuit: CircuitRc,
    ) -> Result<BTreeMap<CircuitRc, Vec<Vec<CircuitRc>>>> {
        self.getter.get_all_paths(circuit, self.matcher.clone())
    }

    pub fn validate(&mut self, circuit: CircuitRc) -> Result<()> {
        self.getter.validate(circuit, self.matcher.clone())
    }
}

#[pyclass]
#[derive(Debug, Clone, Default)]
pub struct AnyFound {
    pub(super) cache: FastUnboundedCache<(HashBytes, IterativeMatcherRc), bool>,
}

#[pymethods]
impl AnyFound {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }
    fn __call__(
        &mut self,
        _py: Python<'_>,
        circuit: CircuitRc,
        matcher: IterativeMatcherRc,
    ) -> Result<bool> {
        self.are_any_found(circuit, matcher)
    }

    pub fn are_any_found(
        &mut self,
        circuit: CircuitRc,
        matcher: IterativeMatcherRc,
    ) -> Result<bool> {
        let out = self.any_impl(circuit, matcher.clone())?;
        Ok(out)
    }

    pub fn bind(&self, matcher: IterativeMatcherRc) -> BoundAnyFound {
        BoundAnyFound {
            any_found: self.clone(),
            matcher,
        }
    }
}

impl AnyFound {
    #[apply(cached_method)]
    #[self_id(self_)]
    #[key((circuit.info().hash, matcher.clone()))]
    #[use_try]
    #[cache_expr(cache)]
    fn any_impl(&mut self, circuit: CircuitRc, matcher: IterativeMatcherRc) -> Result<bool> {
        let IterateMatchResults { updated, found } = matcher.match_iterate(circuit.clone())?;
        if found {
            return Ok(true);
        }

        if !all_finished(&updated) {
            let new_matchers = per_child(updated, matcher, circuit.num_children());
            for (child, new_matcher) in circuit.children().zip(new_matchers) {
                if let Some(new_matcher) = new_matcher {
                    if self_.any_impl(child, new_matcher)? {
                        return Ok(true);
                    }
                }
            }
        }
        Ok(false)
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct BoundAnyFound {
    #[pyo3(get, set)]
    pub any_found: AnyFound,
    #[pyo3(get, set)]
    pub matcher: IterativeMatcherRc,
}

#[pymethods]
impl BoundAnyFound {
    #[new]
    pub fn new(any_found: AnyFound, matcher: IterativeMatcherRc) -> Self {
        Self {
            any_found,
            matcher: matcher.into(),
        }
    }

    fn __call__(&mut self, _py: Python<'_>, circuit: CircuitRc) -> Result<bool> {
        self.are_any_found(circuit)
    }

    pub fn are_any_found(&mut self, circuit: CircuitRc) -> Result<bool> {
        self.any_found.are_any_found(circuit, self.matcher.clone())
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct Expander {
    /// Note: we don't currently cache these transforms individually. We could
    /// do this.
    #[pyo3(get)]
    pub(super) replacements: Vec<TransformRc>,
    /// Technically, having all of these matchers stored here isn't important
    /// for key functionality (like unused for caching).
    /// This is just nice for calling from python.
    ///
    /// invariant: replacements.len() == matchers.len()
    #[pyo3(get)]
    pub(super) matchers: Vec<IterativeMatcherRc>,
    #[pyo3(set, get)]
    pub ban_multiple_matches_on_node: bool,
    #[pyo3(set, get)]
    pub default_fancy_validate: bool,
    #[pyo3(set, get)]
    pub default_assert_any_found: bool,
    #[pyo3(get)]
    pub(super) suffix: Option<String>,
    pub(super) batch_cache:
        FastUnboundedCache<(HashBytes, Vec<IterativeMatcherRc>, HashBytes), UpdateOutput>,
    pub(super) validation_getter: Getter,
}

impl Default for Expander {
    fn default() -> Self {
        Self {
            replacements: Vec::new(),
            matchers: Vec::new(),
            ban_multiple_matches_on_node: false,
            default_fancy_validate: false,
            default_assert_any_found: false,
            suffix: None,
            batch_cache: FastUnboundedCache::default(),
            validation_getter: Default::default(),
        }
    }
}

#[pymethods]
impl Expander {
    #[new]
    #[pyo3(signature=(
        *expanders,
        ban_multiple_matches_on_node = Expander::default().ban_multiple_matches_on_node,
        default_fancy_validate = Expander::default().default_fancy_validate,
        default_assert_any_found = Expander::default().default_assert_any_found,
        suffix = None
    ))]
    pub fn new(
        expanders: Vec<(IterativeMatcherRc, TransformRc)>,
        ban_multiple_matches_on_node: bool,
        default_fancy_validate: bool,
        default_assert_any_found: bool,
        suffix: Option<String>,
    ) -> Self {
        let (matchers, replacements) = expanders
            .into_iter()
            .map(|(a, b)| (a.into(), b.into()))
            .unzip();
        Self {
            replacements,
            matchers,
            ban_multiple_matches_on_node,
            default_fancy_validate,
            default_assert_any_found,
            suffix,
            ..Default::default()
        }
    }

    fn __call__(
        &mut self,
        _py: Python<'_>,
        circuit: CircuitRc,
        fancy_validate: Option<bool>,
        assert_any_found: Option<bool>,
    ) -> Result<CircuitRc> {
        self.batch(circuit, fancy_validate, assert_any_found)
    }

    pub fn batch(
        &mut self,
        circuit: CircuitRc,
        fancy_validate: Option<bool>,
        assert_any_found: Option<bool>,
    ) -> Result<CircuitRc> {
        if fancy_validate.unwrap_or(self.default_fancy_validate) {
            for m in &self.matchers {
                self.validation_getter
                    .validate(circuit.clone(), m.clone())?;
            }
        }
        let out = self.batch_impl(circuit.crc(), self.matchers.clone(), &Default::default())?;
        if assert_any_found.unwrap_or(self.default_assert_any_found) && !out.any_found {
            bail!(
                concat!(
                    "No matches found during expand\n",
                    "matchers: {matchers:?}, circuit: {circuit:?}"
                ),
                matchers = self.matchers,
                circuit = circuit,
            );
        }
        Ok(out.updated)
    }
}

impl Expander {
    #[apply(cached_method)]
    #[self_id(self_)]
    #[key((circuit.info().hash, matchers.clone(), extra_replacements.hash()))]
    #[use_try]
    #[cache_expr(batch_cache)]
    fn batch_impl(
        &mut self,
        circuit: CircuitRc,
        matchers: Vec<IterativeMatcherRc>,
        extra_replacements: &ReplaceMapRc,
    ) -> Result<UpdateOutput> {
        let results = matchers
            .iter()
            .map(|m| m.match_iterate(circuit.clone()))
            .collect::<Result<Vec<_>>>()?;

        let filtered = results.iter().enumerate().filter(|(_, res)| res.found);

        if let Some((idx, _)) = filtered.clone().next() {
            if self_.ban_multiple_matches_on_node {
                let n_matches = filtered.count();
                if n_matches != 1 {
                    bail!("multiple matches! got {} != 1", n_matches);
                }
            }

            return Ok(UpdateOutput {
                updated: self_.replacements[idx].run(circuit)?,
                any_found: true,
            });
        }
        if let Some(replaced) = extra_replacements.get(&circuit) {
            return Ok(UpdateOutput {
                updated: replaced.clone(),
                any_found: false,
            });
        }

        if results.iter().all(|x| all_finished(&x.updated)) && extra_replacements.is_empty() {
            return Ok(UpdateOutput {
                updated: circuit,
                any_found: false,
            });
        }

        let num_children = circuit.num_children();

        let new_matchers: Vec<_> = results
            .into_iter()
            .zip(matchers)
            .map(|(res, matcher)| per_child_with_term(res.updated, matcher, num_children))
            .collect();
        let new_matchers_per = transpose(new_matchers, num_children);
        assert_eq!(new_matchers_per.len(), circuit.num_children());

        let mut any_found = false;
        let new_children = circuit
            .children()
            .zip(extra_replacements.per_child(&circuit))
            .zip(new_matchers_per.clone())
            .map(|((c, rep), new_matchers)| {
                let child_out = self_.batch_impl(c, new_matchers, &rep)?;
                any_found |= child_out.any_found;
                Ok(child_out.updated)
            })
            .collect::<Result<_>>()?;

        let expanded = expand_node(circuit.clone(), &new_children, &mut |c, rep, child_idx| {
            let out = self_.batch_impl(
                c,
                new_matchers_per[child_idx].clone(),
                &extra_replacements.extend_into(rep),
            )?;
            any_found |= out.any_found;
            Ok(out.updated)
        })?;
        let expanded = if expanded != circuit {
            expanded.add_suffix(self_.suffix.as_ref().map(|x| &**x))
        } else {
            expanded
        };

        Ok(UpdateOutput {
            updated: expanded,
            any_found,
        })
    }

    pub fn suffix(&self) -> Option<&str> {
        self.suffix.as_ref().map(|x| &**x)
    }
}
