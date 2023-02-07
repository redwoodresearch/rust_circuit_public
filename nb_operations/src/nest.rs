use std::vec;

use anyhow::{bail, Context, Result};
use circuit_base::{computational_node::EinsumArgs, Add, Circuit, CircuitNode, CircuitRc, Einsum};
use circuit_rewrites::algebraic_rewrite::make_einsum_ints_same_one_layer_and_int_info;
use get_update_node::{
    iterative_matcher::UpdatedIterativeMatcher, IterativeMatcher, IterativeMatcherRc,
};
use macro_rules_attribute::apply;
use pyo3::{exceptions::PyValueError, prelude::*};
use rr_util::{
    compact_data::U8Set,
    name::Name,
    python_error_exception,
    rearrange_spec::{check_permutation, PermError},
    union_find::UnionFind,
    util::{is_unique, transpose, AsOp, EinsumAxes, Multizip},
};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use thiserror::Error;

#[pyclass]
#[derive(Clone, Debug)]
pub struct NestRest {
    #[pyo3(get, set)]
    pub flat: bool,
}

#[pymethods]
impl NestRest {
    #[new]
    #[pyo3(signature=(flat = false))]
    pub fn new(flat: bool) -> Self {
        Self { flat }
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct NestMatcher {
    #[pyo3(get, set)]
    matcher: IterativeMatcherRc,
    #[pyo3(get, set)]
    pub flat: bool,
    #[pyo3(get, set)]
    pub assert_exists: bool,
    #[pyo3(get, set)]
    pub assert_unique: bool,
    #[pyo3(get, set)]
    pub fancy_validate: bool,
}

impl Default for NestMatcher {
    fn default() -> Self {
        Self {
            matcher: Default::default(),
            flat: false,
            assert_exists: true,
            assert_unique: false,
            fancy_validate: false,
        }
    }
}

#[pymethods]
impl NestMatcher {
    #[new]
    #[pyo3(signature=(
        matcher,
        flat = NestMatcher::default().flat,
        assert_exists = NestMatcher::default().assert_exists,
        assert_unique = NestMatcher::default().assert_unique,
        fancy_validate = NestMatcher::default().fancy_validate
    ))]
    pub fn new(
        matcher: IterativeMatcherRc,
        flat: bool,
        assert_exists: bool,
        assert_unique: bool,
        fancy_validate: bool,
    ) -> Self {
        Self {
            matcher,
            flat,
            assert_exists,
            assert_unique,
            fancy_validate,
        }
    }
}

#[derive(Clone, Debug, FromPyObject, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum IntOrMatcher {
    Int(usize),
    Matcher(IterativeMatcherRc), // equivalent to NestMatcher with assert_unique
}

pub trait NestInfo: Clone {
    type Full: NestFullInfo;
    fn to_full(self, orig_full: Option<&Self::Full>) -> Self::Full;
}

type FlattenImplRec<Full> = (
    CircuitRc,
    Option<NestEnumerationItem<Full>>,
    Vec<Vec<usize>>,
);

pub trait NestFullInfo: Clone + Default {
    type CircuitType: CircuitNode + Clone;
    type Extra<'a>
    where
        Self: 'a;
    fn from_circ(circ: CircuitRc) -> Self;
    fn map_extra(x: NestIdxsInfo<Self>, _: CircuitRc, _: &Self::Extra<'_>) -> NestIdxsInfo<Self> {
        x
    }
    fn finish_after_rec(
        rec_on_children: &[FlattenImplRec<Self>],
        x: Self::CircuitType,
    ) -> FlattenImplRec<Self>;
    fn nest_flat_strict_impl(
        flat: &Self::CircuitType,
        specs: Vec<NestIdxsInfo<Self>>,
    ) -> Result<Self::CircuitType>;
}

#[derive(Clone, Debug)]
pub enum NestSpecMultiple<I: NestInfo> {
    Rest(NestRest),
    Matcher(NestMatcher),
    Many(Vec<NestSpec<I>>),
}

#[derive(Clone, Debug)]
pub enum NestSpecSub<I: NestInfo> {
    Multiple(NestSpecMultiple<I>),
    Val(IntOrMatcher),
}

#[derive(Clone, Debug)]
pub struct NestSpecInfo<I: NestInfo> {
    pub spec: NestSpecMultiple<I>,
    pub info: I,
}

#[derive(Clone, Debug)]
pub enum NestSpec<I: NestInfo> {
    Info(NestSpecInfo<I>),
    Sub(NestSpecSub<I>),
}

macro_rules! nest_spec_for {
    ($name:ident {
        $(
            $field:ident : $field_ty:ty,
        )*
    }) => {
    paste::paste!{
    #[derive(Clone, Debug)]
    pub struct [<Nest $name sInfo>] {
        $(
            pub $field: $field_ty,
        )*
    }

    mod [<$name:snake _spec>] {
        use anyhow::Result;
        use pyo3::prelude::*;
        use rr_util::{util::IterInto, py_types::MaybeNotSet};
        use circuit_base::{$name, CircuitNode};

        use super::{
            nest_gen, IntOrMatcher, [<Nest $name sInfo>] as NestInfo, NestMatcher, NestRest, NestSpec,
            NestSpecInfo, NestSpecMultiple, NestSpecSub, IterativeMatcherRc, IterativeMatcher,
        };

        #[derive(Clone, Debug, FromPyObject)]
        pub enum PyNestSpecMultiple {
            Rest(NestRest),
            Matcher(NestMatcher),
            Many(Vec<PyNestSpec>),
        }
        #[derive(Clone, Debug, FromPyObject)]
        pub enum PyNestSpecSub {
            Multiple(PyNestSpecMultiple),
            Val(IntOrMatcher),
        }
        #[pyclass]
        #[derive(Clone, Debug)]
        pub struct [<Nest $name sSpecInfo>] {
            pub spec: PyNestSpecMultiple,
            $(
                pub $field: $field_ty,
            )*
        }

        type PyNestSpecInfo = [<Nest $name sSpecInfo>];

        #[derive(Clone, Debug, FromPyObject)]
        pub enum PyNestSpec {
            Info(PyNestSpecInfo),
            Sub(PyNestSpecSub),
        }

        #[pymethods]
        impl PyNestSpecInfo {
            #[new]
            #[pyo3(signature=(spec, $($field = Default::default(),)*))]
            pub fn new(spec: PyNestSpecMultiple, $($field : MaybeNotSet<$field_ty>,)*) -> Result<Self> {
                let default_nest_info = NestInfo::default();
                let NestInfo {
                    $($field,)*
                } = NestInfo::new($($field.0.unwrap_or(default_nest_info.$field),)*)?; // check for errors
                Ok(Self {
                    spec,
                    $($field,)*
                })
            }
        }

        impl From<PyNestSpecMultiple> for NestSpecMultiple<NestInfo> {
            fn from(x: PyNestSpecMultiple) -> Self {
                match x {
                    PyNestSpecMultiple::Rest(v) => NestSpecMultiple::Rest(v),
                    PyNestSpecMultiple::Matcher(v) => NestSpecMultiple::Matcher(v),
                    PyNestSpecMultiple::Many(v) => NestSpecMultiple::Many(v.into_collect()),
                }
            }
        }
        impl From<PyNestSpecSub> for NestSpecSub<NestInfo> {
            fn from(x: PyNestSpecSub) -> Self {
                match x {
                    PyNestSpecSub::Multiple(v) => NestSpecSub::Multiple(v.into()),
                    PyNestSpecSub::Val(v) => NestSpecSub::Val(v),
                }
            }
        }
        impl From<PyNestSpecInfo> for NestSpecInfo<NestInfo> {
            fn from(x: PyNestSpecInfo) -> Self {
                NestSpecInfo {
                    spec: x.spec.into(),
                    info: NestInfo {
                        $(
                        $field : x.$field,
                        )*
                    },
                }
            }
        }
        impl From<PyNestSpec> for NestSpec<NestInfo> {
            fn from(x: PyNestSpec) -> Self {
                match x {
                    PyNestSpec::Info(v) => NestSpec::Info(v.into()),
                    PyNestSpec::Sub(v) => NestSpec::Sub(v.into()),
                }
            }
        }

        #[pyfunction]
        #[pyo3(signature=([<$name:snake>], spec, traversal = IterativeMatcher::noop_traversal().rc()))]
        pub fn [<nest_ $name:snake s>](
            [<$name:snake>] : $name,
            spec: PyNestSpecSub, // use sub to disallow top level info which does nothing
            traversal: IterativeMatcherRc,
        ) -> Result<$name> {
            let spec = NestSpec::Sub(spec.into());
            nest_gen([<$name:snake>].rc(), spec, traversal)
        }


        pub fn register(_: Python<'_>, m: &PyModule) -> PyResult<()> {
            m.add_class::<PyNestSpecInfo>()?;
            m.add_function(wrap_pyfunction!([<nest_ $name:snake s>], m)?)
        }
    }
    pub use [<$name:snake _spec>]::[<nest_ $name:snake s>];
    }
    };
}

nest_spec_for!(Einsum {
    name: Option<rr_util::name::Name>,
    out_axes_perm: Option<rr_util::util::EinsumAxes>,
    shrink_out_axes: bool,
});
nest_spec_for!(Add {
    name: Option<rr_util::name::Name>,
});

impl Default for NestEinsumsInfo {
    fn default() -> Self {
        Self {
            name: None,
            out_axes_perm: None,
            shrink_out_axes: false,
        }
    }
}
impl Default for NestAddsInfo {
    fn default() -> Self {
        Self { name: None }
    }
}

impl NestInfo for NestEinsumsInfo {
    type Full = NestEinsumsFullInfo;
    fn to_full(self, orig_full: Option<&Self::Full>) -> Self::Full {
        let out_axes = orig_full.map(|x| x.out_axes.clone()).flatten();
        Self::Full {
            name: self.name,
            out_axes,
            out_axes_perm: self.out_axes_perm,
            shrink_out_axes: self.shrink_out_axes,
        }
    }
}
impl NestInfo for NestAddsInfo {
    type Full = NestAddsFullInfo;
    fn to_full(self, _: Option<&Self::Full>) -> Self::Full {
        self
    }
}

impl NestEinsumsInfo {
    pub fn new(
        name: Option<Name>,
        out_axes_perm: Option<EinsumAxes>,
        shrink_out_axes: bool,
    ) -> Result<Self> {
        if let Some(axes) = &out_axes_perm {
            check_permutation(axes).context("out_axes_perm not permutation")?
        }
        Ok(Self {
            name,
            out_axes_perm,
            shrink_out_axes,
        })
    }
}
impl NestAddsInfo {
    pub fn new(name: Option<Name>) -> Result<Self> {
        Ok(Self { name })
    }
}

mod nest_match_prelude {
    pub use super::{
        NestMatcher as Matcher, NestSpec::*, NestSpecInfo as Info, NestSpecMultiple::*,
        NestSpecSub::*,
    };
}

#[derive(Debug, Clone)]
pub struct NestIdxsInfo<I: NestFullInfo> {
    pub idxs: NestIdxsItem<I>,
    pub info: I,
}

#[derive(Debug, Clone)]
pub enum NestIdxsItem<I: NestFullInfo> {
    Single(usize),
    Many(Vec<NestIdxsInfo<I>>),
}

#[derive(Debug, Clone)]
pub struct NestEinsumsFullInfo {
    pub name: Option<Name>,
    pub out_axes: Option<EinsumAxes>,
    pub out_axes_perm: Option<EinsumAxes>,
    pub shrink_out_axes: bool,
}
type NestAddsFullInfo = NestAddsInfo;

impl NestEinsumsFullInfo {
    fn new(
        name: Option<Name>,
        out_axes: Option<EinsumAxes>,
        out_axes_perm: Option<EinsumAxes>,
        shrink_out_axes: bool,
    ) -> Result<Self> {
        if let Some(axes) = &out_axes_perm {
            check_permutation(axes).context("sorted_out_axes_perm not permutation")?
        }

        Ok(Self {
            name,
            out_axes,
            out_axes_perm,
            shrink_out_axes,
        })
    }
}
impl Default for NestEinsumsFullInfo {
    fn default() -> Self {
        NestEinsumsFullInfo::new(None, None, None, false).unwrap()
    }
}
impl NestFullInfo for NestEinsumsFullInfo {
    type CircuitType = Einsum;
    type Extra<'a> = (&'a UnionFind, Option<HashMap<u8, u8>>);

    fn from_circ(circ: CircuitRc) -> Self {
        Self {
            name: circ.info().name,
            out_axes: circ.as_einsum().map(|x| x.out_axes.clone()),
            ..Default::default()
        }
    }

    fn map_extra(
        x: NestIdxsInfo<Self>,
        circ: CircuitRc,
        extra: &Self::Extra<'_>,
    ) -> NestIdxsInfo<Self> {
        let (union_find, int_map) = extra;
        assert_eq!(circ.is_einsum(), int_map.is_some());
        if let Some(int_map) = int_map {
            let get_new_nums = |x: u8| union_find.find_(int_map[&x] as usize) as u8;
            x.map_out_axes(get_new_nums) // quadratic in tree depth fwiw
        } else {
            x
        }
    }

    fn finish_after_rec(
        rec_on_children: &[FlattenImplRec<Self>],
        einsum: Einsum,
    ) -> FlattenImplRec<Self> {
        let (einsum, unionfind, _, per_arg_int_maps) =
            make_einsum_ints_same_one_layer_and_int_info(&einsum);
        let extra: Vec<_> = per_arg_int_maps
            .into_iter()
            .map(|m| (&unionfind, m))
            .collect();

        let new_args = einsum
            .args()
            .zip(rec_on_children)
            .flat_map(|((node, ints), (_, named_idxs, _))| {
                if named_idxs.is_some() {
                    node.as_einsum().unwrap().args_cloned()
                } else {
                    vec![(node, ints.clone())]
                }
            })
            .collect();

        let out_circ = Einsum::nrc(new_args, einsum.out_axes.clone(), einsum.info().name);

        finish_with_extra_and_circ(&rec_on_children, &extra, out_circ)
    }

    fn nest_flat_strict_impl(
        flat: &Self::CircuitType,
        specs: Vec<NestIdxsInfo<Self>>,
    ) -> Result<Self::CircuitType> {
        nest_flat_einsum_strict_rec(
            &flat.args_cloned(),
            flat.out_axes.clone(),
            specs,
            flat.info().name,
        )
    }
}
impl NestFullInfo for NestAddsFullInfo {
    type CircuitType = Add;
    type Extra<'a> = ();

    fn from_circ(circ: CircuitRc) -> Self {
        Self {
            name: circ.info().name,
        }
    }

    fn finish_after_rec(
        rec_on_children: &[FlattenImplRec<Self>],
        add: Add,
    ) -> FlattenImplRec<Self> {
        let new_operands: Vec<CircuitRc> = add
            .children()
            .zip(rec_on_children)
            .flat_map(|(node, (_, named_idxs, _))| {
                if named_idxs.is_some() {
                    node.as_add().unwrap().children_sl().to_vec()
                } else {
                    vec![node]
                }
            })
            .collect();
        let out_circ = Add::nrc(new_operands, add.info().name);

        finish_with_extra_and_circ(&rec_on_children, &vec![(); rec_on_children.len()], out_circ)
    }

    fn nest_flat_strict_impl(
        flat: &Self::CircuitType,
        specs: Vec<NestIdxsInfo<Self>>,
    ) -> Result<Self::CircuitType> {
        Ok(nest_flat_add_strict_rec(
            flat.children_sl(),
            specs,
            flat.info().name,
        ))
    }
}

impl NestIdxsInfo<NestEinsumsFullInfo> {
    pub fn map_out_axes(&self, mapping: impl Fn(u8) -> u8 + Clone) -> Self {
        let new_idxs = match &self.idxs {
            NestIdxsItem::Single(x) => NestIdxsItem::Single(*x),
            NestIdxsItem::Many(items) => NestIdxsItem::Many(
                items
                    .iter()
                    .map(|x| x.map_out_axes(mapping.clone()))
                    .collect(),
            ),
        };

        Self {
            idxs: new_idxs,
            info: NestEinsumsFullInfo::new(
                self.info.name.clone(),
                self.info
                    .out_axes
                    .as_ref()
                    .map(|axes| axes.iter().cloned().map(mapping).collect()),
                self.info.out_axes_perm.clone(),
                self.info.shrink_out_axes,
            )
            .unwrap(),
        }
    }
}

impl<I: NestFullInfo> NestIdxsItem<I> {
    fn all_indices(&self) -> Vec<usize> {
        match self {
            Self::Single(x) => vec![*x],
            Self::Many(x) => x.into_iter().flat_map(|x| x.idxs.all_indices()).collect(),
        }
    }
    fn add(&self, n: usize) -> Self {
        match self {
            Self::Single(x) => Self::Single(x + n),
            Self::Many(x) => Self::Many(
                x.into_iter()
                    .map(|x| NestIdxsInfo {
                        idxs: x.idxs.add(n),
                        info: x.info.clone(),
                    })
                    .collect(),
            ),
        }
    }

    fn insert_exact_subset(
        &self,
        parent_info: Option<I>,
        exact_subsets: &mut HashMap<(usize, usize), NestIdxsInfo<I>>,
    ) -> Option<(usize, usize)> {
        let out = match self {
            Self::Single(item) => (*item, *item + 1),
            Self::Many(items) => {
                let (start, end) =
                    items
                        .iter()
                        .fold((usize::MAX, usize::MIN), |(start, end), item| {
                            if let Some((new_start, new_end)) = item
                                .idxs
                                .insert_exact_subset(Some(item.info.clone()), exact_subsets)
                            {
                                // we assume contiguous, so this is valid!
                                if end != usize::MIN {
                                    assert_eq!(end, new_start);
                                }
                                (start.min(new_start), end.max(new_end)) // this fold is a bit silly, we could just get first + last
                            } else {
                                return (start, end);
                            }
                        });
                if start == usize::MAX {
                    // empty einsum
                    return None;
                } else {
                    assert_ne!(end, usize::MIN);
                    (start, end)
                }
            }
        };
        if let Some(info) = parent_info {
            exact_subsets.insert(
                out,
                NestIdxsInfo {
                    idxs: self.clone(),
                    info,
                },
            );
        }
        Some(out)
    }
}

/// maintains invariant that indices are in sorted enumeration order
#[derive(Clone, Debug)]
pub struct NestEnumerationItem<I: NestFullInfo>(NestIdxsItem<I>);

impl<I: NestFullInfo> NestEnumerationItem<I> {
    pub fn new(idxs: NestIdxsItem<I>) -> Self {
        let all_indices = idxs.all_indices();
        assert!(&all_indices == &(0..all_indices.len()).collect::<Vec<usize>>());
        Self(idxs)
    }

    fn exact_subset_to_named_idxs(&self) -> HashMap<(usize, usize), NestIdxsInfo<I>> {
        let mut out_map = HashMap::default();
        self.0.insert_exact_subset(None, &mut out_map);
        out_map
    }
}

pub fn check_permutation_rest(
    perm: &[usize],
    count: usize,
    allow_rest: bool,
) -> Result<HashSet<usize>> {
    let perm_set: HashSet<_> = perm.iter().cloned().collect();
    if perm.len() != perm_set.len() {
        bail!(PermError::IntsNotUnique {
            ints: perm.iter().cloned().collect()
        })
    }
    let count_set = (0..count).collect::<HashSet<_>>();
    if !perm_set.is_subset(&count_set) {
        bail!(NestError::IntNotContainedInRangeCount {
            ints: perm.to_vec(),
            count,
            extra_ints: perm_set.difference(&count_set).cloned().collect()
        })
    }
    let rest: HashSet<usize> = count_set.difference(&perm_set).cloned().collect();
    if !allow_rest && rest.len() != 0 {
        bail!(NestError::PermutationMissesIdxsAndNoRestInSpec { missed_idxs: rest })
    }
    Ok(rest)
}

impl<I: NestInfo> NestSpec<I> {
    fn count_rest(&self) -> usize {
        use nest_match_prelude::*;
        match self {
            Sub(Val(_)) => 0,
            Sub(Multiple(Many(items)))
            | Info(Info {
                spec: Many(items), ..
            }) => items.iter().map(|x| x.count_rest()).sum(),
            Sub(Multiple(Rest(_))) | Info(Info { spec: Rest(_), .. }) => 1,
            Sub(Multiple(Matcher(_)))
            | Info(Info {
                spec: Matcher(_), ..
            }) => 0,
        }
    }

    fn check_rest_valid(&self) -> Result<()> {
        let count_rest = self.count_rest();
        if count_rest > 1 {
            bail!(NestError::MultipleRest { count_rest });
        }
        Ok(())
    }

    pub fn all_matchers(&self) -> Vec<IterativeMatcherRc> {
        use nest_match_prelude::*;
        match self {
            Sub(Val(IntOrMatcher::Matcher(val))) => vec![val.clone()],
            Sub(Val(IntOrMatcher::Int(_))) => Vec::new(),
            Sub(Multiple(Many(items)))
            | Info(Info {
                spec: Many(items), ..
            }) => items.iter().flat_map(|x| x.all_matchers()).collect(),
            Sub(Multiple(Matcher(Matcher { matcher, .. })))
            | Info(Info {
                spec: Matcher(Matcher { matcher, .. }),
                ..
            }) => vec![matcher.clone()],
            Sub(Multiple(Rest(_))) | Info(Info { spec: Rest(_), .. }) => Vec::new(),
        }
    }

    pub fn all_ints(&self) -> Vec<usize> {
        use nest_match_prelude::*;
        match self {
            Sub(Val(IntOrMatcher::Int(v))) => vec![*v],
            Sub(Val(IntOrMatcher::Matcher(_))) => Vec::new(),
            Sub(Multiple(Many(items)))
            | Info(Info {
                spec: Many(items), ..
            }) => items.iter().flat_map(|x| x.all_ints()).collect(),
            Sub(Multiple(Matcher(_)))
            | Sub(Multiple(Rest(_)))
            | Info(Info { spec: Rest(_), .. })
            | Info(Info {
                spec: Matcher(_), ..
            }) => Vec::new(),
        }
    }

    pub fn convert_to_named_idxs(
        &self,
        mapping: &HashMap<(usize, usize), NestIdxsInfo<I::Full>>,
        all_matchers: &[IterativeMatcherRc],
        ints_per_matcher: &[Vec<usize>],
        circuits: &[CircuitRc],
    ) -> Result<NestIdxsItem<I::Full>> {
        let all_ints = self.all_ints();
        let all_ints_set = all_ints.iter().cloned().collect();
        // more unneeded quadratic running time...
        // (Note that this would also be caught by 'check_permutation_rest', we check here also to improve error messages)
        for (k, v) in all_matchers.iter().zip(ints_per_matcher) {
            for (other_k, other_v) in all_matchers.iter().zip(ints_per_matcher) {
                if k == other_k {
                    continue;
                }

                let v_set = v.iter().cloned().collect::<HashSet<_>>();
                let other_v_set = other_v.iter().cloned().collect();
                if v_set.intersection(&other_v_set).count() > 0 {
                    let intersection: Vec<_> = v_set.intersection(&other_v_set).cloned().collect();
                    bail!(NestError::MatchersOverlap {
                        matcher: k.clone(),
                        other_matcher: other_k.clone(),
                        ints: v.clone(),
                        other_ints: other_v.clone(),
                        intersection_circs: intersection
                            .iter()
                            .map(|x| circuits[*x].clone())
                            .collect(),
                        intersection: intersection,
                    });
                }
                if v_set.intersection(&all_ints_set).count() > 0 {
                    let intersection: Vec<_> = v_set.intersection(&all_ints_set).cloned().collect();
                    bail!(NestError::MatcherOverlapsWithExplicitInts {
                        matcher: k.clone(),
                        matcher_ints: v.clone(),
                        explicit_ints: all_ints,
                        intersection_circs: intersection
                            .iter()
                            .map(|x| circuits[*x].clone())
                            .collect(),
                        intersection: intersection,
                    });
                }
            }
        }
        let all_ints_vals: Vec<_> = ints_per_matcher
            .iter()
            .chain(std::iter::once(&all_ints))
            .flatten()
            .cloned()
            .collect();

        let allow_rest = self.count_rest() == 1;
        let rest = check_permutation_rest(&all_ints_vals, circuits.len(), allow_rest)
            .context("ints from flattened spec not valid permutation")?;
        if circuits.is_empty() {
            return Ok(NestIdxsItem::Many(Vec::new()));
        }
        let rest_items = Self::find_rest(rest, mapping);
        let mut all_items = self
            .convert_to_named_idxs_impl(
                mapping,
                &all_matchers
                    .into_iter()
                    .zip(ints_per_matcher)
                    .map(|(a, b)| (a, &b[..]))
                    .collect(),
                circuits,
                &rest_items,
                true,
            )?
            .0;
        // this assert is valid because we pass in 'is_outer=True' above.
        assert_eq!(all_items.len(), 1);
        Ok(all_items.pop().unwrap().idxs)
    }

    fn find_rest(
        rest: HashSet<usize>,
        mapping: &HashMap<(usize, usize), NestIdxsInfo<I::Full>>,
    ) -> Vec<NestIdxsInfo<I::Full>> {
        let mut subset_keys: Vec<_> = mapping
            .keys()
            .cloned()
            .filter(|&(k_start, k_end)| rest.is_superset(&(k_start..k_end).collect()))
            .collect();
        subset_keys.sort();
        let final_keys: Vec<_> = subset_keys
            .iter()
            .cloned()
            .filter(|(k_start, k_end)| {
                // TODO: could toposort for efficiency
                !subset_keys.iter().any(|(k_inner_start, k_inner_end)| {
                    let is_different = k_start != k_inner_start || k_end != k_inner_end;
                    let inner_includes = k_inner_start <= k_start && k_inner_end >= k_end;
                    is_different && inner_includes
                })
            })
            .collect();

        // some quick debug checking
        for ((_, prior_end), (next_start, _)) in final_keys.iter().zip(final_keys.iter().skip(1)) {
            assert!(prior_end <= next_start);
        }
        let all_key_vals: HashSet<usize> = final_keys
            .iter()
            .flat_map(|&(start, end)| start..end)
            .collect();
        assert_eq!(all_key_vals, rest);

        final_keys.iter().map(|k| mapping[k].clone()).collect()
    }

    pub fn get_full(
        &self,
        all_ints: &[usize],
        mapping: &HashMap<(usize, usize), NestIdxsInfo<I::Full>>,
    ) -> I::Full {
        let start_r = *all_ints.iter().min().unwrap();
        let end_r = *all_ints.iter().max().unwrap() + 1;
        let orig_info = if all_ints.len() == end_r - start_r {
            mapping.get(&(start_r, end_r)).map(|x| &x.info)
        } else {
            None
        };
        match self {
            Self::Info(NestSpecInfo { info, .. }) => info.clone().to_full(orig_info).clone(),
            Self::Sub(_) => orig_info
                .map(|x| x.clone())
                .unwrap_or_else(Default::default),
        }
    }

    pub fn convert_to_named_idxs_impl(
        &self,
        mapping: &HashMap<(usize, usize), NestIdxsInfo<I::Full>>,
        matcher_to_ints: &HashMap<&IterativeMatcherRc, &[usize]>,
        circuits: &[CircuitRc],
        rest_items: &[NestIdxsInfo<I::Full>],
        is_outer: bool,
    ) -> Result<(Vec<NestIdxsInfo<I::Full>>, Vec<usize>)> {
        use nest_match_prelude::*;

        let check_matcher = |matcher: &NestMatcher| {
            let ints = matcher_to_ints[&matcher.matcher];
            let get_matches = || ints.iter().map(|i| circuits[*i].clone());

            if matcher.assert_exists && ints.is_empty() {
                bail!(NestError::MatcherMatchedNoneAndMustExist {
                    matcher: matcher.matcher.clone()
                })
            }
            if matcher.assert_unique && ints.len() > 1 {
                bail!(NestError::MatcherMatchedMultipleAndMustBeUnique {
                    matcher: matcher.matcher.clone(),
                    matches: get_matches().collect(),
                    int_matches: ints.to_vec(),
                })
            }
            if matcher.fancy_validate {
                matcher
                    .matcher
                    .validate_matched(&get_matches().collect())
                    .context("fancy validate failed for a matcher in nest")?;
            }

            Ok(ints)
        };

        let find_rest =
            |ints: &[usize]| Self::find_rest(ints.into_iter().cloned().collect(), mapping);

        let handle_many = |all_ints: Vec<_>, items, flat| {
            assert!(is_unique(&all_ints));
            let out_idxs = if (flat && !is_outer) || all_ints.is_empty() {
                items
            } else {
                let full = self.get_full(&all_ints, mapping);
                vec![NestIdxsInfo {
                    idxs: NestIdxsItem::Many(items),
                    info: full,
                }]
            };
            (out_idxs, all_ints)
        };

        let res = match self {
            Sub(Val(val)) => {
                let int = match val {
                    IntOrMatcher::Int(int) => *int,
                    IntOrMatcher::Matcher(matcher) => {
                        let ints = check_matcher(&NestMatcher {
                            matcher: matcher.clone(),
                            assert_exists: true,
                            assert_unique: true,
                            ..Default::default()
                        })?;
                        assert_eq!(ints.len(), 1);
                        ints[0]
                    }
                };
                return Ok((vec![mapping[&(int, int + 1)].clone()], vec![int]));
            }
            Sub(Multiple(Many(items)))
            | Info(Info {
                spec: Many(items), ..
            }) => {
                let (items_vec, subsets): (Vec<_>, Vec<_>) = items
                    .iter()
                    .map(|x| {
                        x.convert_to_named_idxs_impl(
                            mapping,
                            matcher_to_ints,
                            circuits,
                            rest_items,
                            false,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .unzip();
                let all_ints: Vec<_> = subsets.into_iter().flatten().collect();
                handle_many(all_ints, items_vec.into_iter().flatten().collect(), false)
            }
            Sub(Multiple(Rest(NestRest { flat })))
            | Info(Info {
                spec: Rest(NestRest { flat }),
                ..
            }) => {
                let all_ints: Vec<usize> = rest_items
                    .iter()
                    .flat_map(|x| x.idxs.all_indices())
                    .collect();
                handle_many(all_ints, rest_items.to_vec(), *flat)
            }
            Sub(Multiple(Matcher(matcher)))
            | Info(Info {
                spec: Matcher(matcher),
                ..
            }) => {
                let all_ints = check_matcher(matcher)?.to_vec();
                let items = find_rest(&all_ints);
                handle_many(all_ints, items, matcher.flat)
            }
        };
        Ok(res)
    }
}

pub fn finish_with_extra_and_circ<I: NestFullInfo>(
    rec_on_children: &[FlattenImplRec<I>],
    extra_items: &[I::Extra<'_>],
    out_circ: CircuitRc,
) -> FlattenImplRec<I> {
    assert_eq!(rec_on_children.len(), extra_items.len());
    let mut running_count = 0;
    let (out, all_mappings): (_, Vec<_>) = rec_on_children
        .into_iter()
        .zip(extra_items)
        .map(|((circ, maybe_items, mapping), extra)| {
            let (new_item, additional_count) = match maybe_items {
                Some(items) => {
                    // not efficient but whatever
                    (items.0.add(running_count), items.0.all_indices().len())
                }
                None => (NestIdxsItem::Single(running_count), 1),
            };
            let mut mapping = mapping.clone();
            for v in &mut mapping {
                for i in v {
                    *i += running_count
                }
            }

            running_count += additional_count;

            (
                I::map_extra(
                    NestIdxsInfo {
                        idxs: new_item,
                        info: I::from_circ(circ.clone()),
                    },
                    circ.clone(),
                    extra,
                ),
                mapping,
            )
        })
        .unzip();

    // combine all nums for each matcher
    let fused: Vec<_> = Multizip(all_mappings.into_iter().map(|x| x.into_iter()).collect())
        .map(|items| items.into_iter().flatten().collect::<Vec<_>>())
        .collect();

    for v in &fused {
        assert!(is_unique(v), "internal error in nest");
    }

    (
        out_circ,
        Some(NestEnumerationItem::new(NestIdxsItem::Many(out))),
        fused,
    )
}

pub fn run_matcher<T>(
    circ: &CircuitRc,
    matcher: IterativeMatcherRc,
) -> Result<Option<(&T, UpdatedIterativeMatcher)>>
where
    Circuit: AsOp<T>,
{
    if let Some(out) = AsOp::<T>::as_op(&***circ) {
        // if let chain ICE's compiler : / (fixed on new version)
        let updated = matcher
            .match_iterate(circ.clone())?
            .unwrap_or_same(matcher)
            .0;
        if !updated.all_finished() {
            return Ok(Some((out, updated)));
        }
    }
    Ok(None)
}

/// ignores 'found' and just looks at whether or not matcher has terminated yet.
/// In other words, uses the matcher as a 'traversal'.
pub fn flatten_impl<Full: NestFullInfo>(
    circ: CircuitRc,
    matcher: IterativeMatcherRc,
    current_matcher_to_pair: Vec<IterativeMatcherRc>,
) -> Result<FlattenImplRec<Full>>
where
    Circuit: AsOp<Full::CircuitType>,
{
    let num_children = circ.num_children();
    let (updated_current_matcher_to_pair, found): (Vec<_>, Vec<_>) = current_matcher_to_pair
        .into_iter()
        .map(|m| {
            m.match_iterate(circ.clone()).map(|out| {
                let (new, found) = out.unwrap_or_same(m);
                (new.per_child_with_term(num_children), found)
            })
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .unzip();
    let updated_current_matcher_to_pair = transpose(updated_current_matcher_to_pair, num_children);

    // TODO: we could cache, but probably overkill
    // no if let chain is sad, but format is broke.
    let f = || -> Result<_> {
        if let Some(out) = AsOp::<Full::CircuitType>::as_op(&**circ) {
            let updated = matcher
                .match_iterate(circ.clone())?
                .unwrap_or_same(matcher)
                .0;
            if !updated.all_finished() {
                return Ok(Some((out, updated)));
            }
        }
        Ok(None)
    };

    let (x, updated) = if let Some(x) = f()? {
        x
    } else {
        return Ok((
            circ,
            None,
            found
                .into_iter()
                .map(|found| if found { vec![0] } else { vec![] })
                .collect(),
        ));
    };

    let rec_on_children = x
        .children()
        .zip(updated.per_child_with_term(num_children))
        .zip(updated_current_matcher_to_pair)
        .map(|((c, new_matcher), updated_matchers_to_pair)| {
            flatten_impl(c, new_matcher.clone(), updated_matchers_to_pair)
        })
        .collect::<Result<Vec<_>>>()?;
    let x = x.map_children_unwrap_idxs(|i| rec_on_children[i].0.clone());

    Ok(Full::finish_after_rec(&rec_on_children, x))
}

pub fn flatten_gen<Full: NestFullInfo>(
    x: Full::CircuitType,
    traversal: IterativeMatcherRc,
) -> Result<Full::CircuitType>
where
    Circuit: AsOp<Full::CircuitType>,
{
    let (new_x, _, _) = flatten_impl::<Full>(x.rc(), traversal, vec![])?;
    Ok((**new_x).clone().into_op().unwrap())
}
#[pyfunction]
#[pyo3(signature=(einsum, traversal = IterativeMatcher::noop_traversal().rc()))]
pub fn einsum_flatten(einsum: Einsum, traversal: IterativeMatcherRc) -> Result<Einsum> {
    flatten_gen::<NestEinsumsFullInfo>(einsum, traversal).map(|x| x.normalize_ints())
}
#[pyfunction]
#[pyo3(signature=(add, traversal = IterativeMatcher::noop_traversal().rc()))]
pub fn add_flatten(add: Add, traversal: IterativeMatcherRc) -> Result<Add> {
    flatten_gen::<NestAddsFullInfo>(add, traversal)
}

fn nest_gen<I: NestInfo>(
    circuit: CircuitRc,
    spec: NestSpec<I>, // use sub to disallow top level name which does nothing
    traversal: IterativeMatcherRc,
) -> Result<<I::Full as NestFullInfo>::CircuitType>
where
    Circuit: AsOp<<I::Full as NestFullInfo>::CircuitType>,
{
    spec.check_rest_valid()?;
    let all_matchers = spec.all_matchers();
    let (flat, enumeration, ints_per_matcher) =
        flatten_impl(circuit.clone(), traversal.clone(), all_matchers.clone())?;
    if enumeration.is_none() {
        bail!(NestError::TraversalMatchedNothing { traversal, circuit })
    }
    let flat: &<I::Full as NestFullInfo>::CircuitType = (**flat).as_op().unwrap();
    let enumeration = enumeration.unwrap();
    let new_item = spec.convert_to_named_idxs(
        &enumeration.exact_subset_to_named_idxs(),
        &all_matchers,
        &ints_per_matcher,
        &flat.children().collect::<Vec<_>>(),
    )?;

    match new_item {
        NestIdxsItem::Single(_) => {
            assert_eq!(flat.num_children(), 1);
            Ok(flat.clone())
        }
        NestIdxsItem::Many(specs) => I::Full::nest_flat_strict_impl(flat, specs),
    }
}

pub fn nest_flat_einsum_strict_rec(
    flat_einsum: &EinsumArgs,
    out_axes: EinsumAxes,
    specs: Vec<NestIdxsInfo<NestEinsumsFullInfo>>,
    name: Option<Name>,
) -> Result<Einsum> {
    // TODO: avoid quadratic running time...
    let all_ints: Vec<U8Set> = specs
        .iter()
        .map(|spec| {
            spec.idxs
                .all_indices()
                .into_iter()
                .flat_map(|idx| &flat_einsum[idx].1)
                .cloned()
                .collect()
        })
        .collect();
    let out_axes_set = out_axes.iter().cloned().collect();

    let args = specs
        .into_iter()
        .enumerate()
        .map(|(i, spec)| {
            let out = match spec.idxs {
                NestIdxsItem::Single(v) => flat_einsum[v].clone(),
                NestIdxsItem::Many(sub_specs) => {
                    let my_ints = &all_ints[i];
                    let all_other_ints: U8Set = all_ints
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| j != &i)
                        .flat_map(|(_, ints)| *ints)
                        .collect();
                    let all_other_ints: U8Set = all_other_ints.union(&out_axes_set);
                    let out_ints = all_other_ints.intersection(my_ints);

                    let mut sub_out_axes = if let Some(sub_out_axes) = spec.info.out_axes {
                        let out_ints_actual: U8Set =
                            sub_out_axes.iter().cloned().collect::<U8Set>();
                        assert!(out_ints_actual.is_subset(my_ints));
                        let minimal_out_num_set: U8Set = out_ints;
                        assert!(out_ints_actual.is_superset(&minimal_out_num_set));
                        if spec.info.shrink_out_axes {
                            // keep ordering for user simplicity
                            sub_out_axes
                                .iter()
                                .filter(|x| minimal_out_num_set.contains(**x))
                                .cloned()
                                .collect()
                        } else {
                            sub_out_axes
                        }
                    } else {
                        let mut sub_out_axes: EinsumAxes = out_ints.into_iter().collect();
                        sub_out_axes.as_mut_slice().sort();
                        sub_out_axes
                    };

                    // then permute
                    if let Some(perm) = spec.info.out_axes_perm {
                        if perm.len() != sub_out_axes.len() {
                            bail!(NestError::PermHasWrongLen {
                                perm: perm.clone(),
                                expected_len: sub_out_axes.len()
                            })
                        }
                        sub_out_axes = perm.iter().map(|i| sub_out_axes[*i as usize]).collect();
                    }

                    (
                        nest_flat_einsum_strict_rec(
                            flat_einsum,
                            sub_out_axes.clone(),
                            sub_specs,
                            spec.info.name,
                        )?
                        .rc(),
                        sub_out_axes,
                    )
                }
            };
            Ok(out)
        })
        .collect::<Result<EinsumArgs>>()?;

    Ok(Einsum::new(args, out_axes, name))
}

pub fn nest_flat_add_strict_rec(
    flat_nodes: &[CircuitRc],
    specs: Vec<NestIdxsInfo<NestAddsFullInfo>>,
    name: Option<Name>,
) -> Add {
    let nodes = specs
        .into_iter()
        .map(|spec| match spec.idxs {
            NestIdxsItem::Single(v) => flat_nodes[v].clone(),
            NestIdxsItem::Many(sub_specs) => {
                nest_flat_add_strict_rec(flat_nodes, sub_specs, spec.info.name).rc()
            }
        })
        .collect();

    Add::new(nodes, name)
}

pub fn register(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    einsum_spec::register(py, m)?;
    add_spec::register(py, m)?;
    m.add_class::<NestRest>()?;
    m.add_class::<NestMatcher>()?;
    m.add_function(wrap_pyfunction!(einsum_flatten, m)?)?;
    m.add_function(wrap_pyfunction!(add_flatten, m)?)
}

#[apply(python_error_exception)]
#[base_error_name(Nest)]
#[base_exception(PyValueError)]
#[derive(Error, Debug, Clone)]
pub enum NestError {
    #[error("count_rest={count_rest} ({e_name})")]
    MultipleRest { count_rest: usize },
    #[error("matcher={matcher:?} {}={other_matcher:?}\n{ints:?} intersects with {other_ints:?}\nintersection={intersection:?}\nintersection_circs={intersection_circs:?} ({e_name})",
        "overlaps with other_matcher")]
    MatchersOverlap {
        matcher: IterativeMatcherRc,
        other_matcher: IterativeMatcherRc,
        ints: Vec<usize>,
        other_ints: Vec<usize>,
        intersection: Vec<usize>,
        intersection_circs: Vec<CircuitRc>,
    },
    #[error(
        "matcher={matcher:?}, matcher_ints={matcher_ints:?} {}{explicit_ints:?}\nintersection={intersection:?}\n{}{intersection_circs:?} ({e_name})",
        "intersects with explicit_ints=",
        "intersection_circs=")]
    MatcherOverlapsWithExplicitInts {
        matcher: IterativeMatcherRc,
        matcher_ints: Vec<usize>,
        explicit_ints: Vec<usize>,
        intersection: Vec<usize>,
        intersection_circs: Vec<CircuitRc>,
    },
    #[error("matcher={matcher:?} matches={matches:?} (int_matches={int_matches:?}) ({e_name})")]
    MatcherMatchedMultipleAndMustBeUnique {
        matcher: IterativeMatcherRc,
        matches: Vec<CircuitRc>,
        int_matches: Vec<usize>,
    },
    #[error("matcher={matcher:?} ({e_name})")]
    MatcherMatchedNoneAndMustExist { matcher: IterativeMatcherRc },
    #[error("traversal={traversal:?} circuit={circuit:?} ({e_name})")]
    TraversalMatchedNothing {
        traversal: IterativeMatcherRc,
        circuit: CircuitRc,
    },
    #[error("ints={ints:?} count={count} extra_ints={extra_ints:?} ({e_name})")]
    IntNotContainedInRangeCount {
        ints: Vec<usize>,
        count: usize,
        extra_ints: HashSet<usize>,
    },
    #[error("perm={perm:?} expected_len={expected_len} ({e_name})")]
    PermHasWrongLen {
        perm: EinsumAxes,
        expected_len: usize,
    },
    #[error("This num wasn't present in orig! ({e_name})")]
    OrigNumPermWhenNotPresentInOrig {},
    #[error("missed_idxs={missed_idxs:?} ({e_name})")]
    PermutationMissesIdxsAndNoRestInSpec { missed_idxs: HashSet<usize> },
}
