use std::{
    collections::BTreeMap,
    fmt::{Debug, Display},
    hash::{BuildHasher, Hash},
    mem,
    rc::Rc,
    sync::Arc,
};

use anyhow::{Context, Result};
use itertools::Itertools;
use macro_rules_attribute::apply;
use once_cell::sync::Lazy;
use pyo3::{types::PyBytes, FromPyObject, IntoPy, PyObject, Python};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::{compact_data::TinyVecU8, make_lazy, name::Name, simple_from};
pub type AxisInt = u8;
pub type HashBytes = [u8; 32];
// we could use u64 instead of u8 if we wanted.
// this is u8, so we want to use multiple of 8 inline so we don't pad
pub type EinsumAxes = TinyVecU8;
pub type NamedAxes = BTreeMap<AxisInt, Name>;

#[derive(FromPyObject, Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct HashBytesToPy(pub HashBytes);

simple_from!(|b: HashBytes| -> HashBytesToPy { Self(b) });
impl IntoPy<PyObject> for HashBytesToPy {
    fn into_py(self, py: Python<'_>) -> PyObject {
        PyBytes::new(py, &self.0).into()
    }
}

/// letters used in einsum and rearrange strings
/// i wanted this to be a constant, but that is annoying in rust bc can't do any computation to produce constants
/// so instead it's cached function
#[apply(make_lazy)]
#[lazy_ty(Lazy)]
pub fn alphabet() -> Vec<String> {
    let alphabet_upper: Vec<String> = ('A'..'[').map(|x| x.to_string()).collect(); // char after Z
    let alphabet_lower: Vec<String> = ('a'..'{').map(|x| x.to_string()).collect();
    let alphabet_greek_lower: Vec<String> = ('α'..'ω').map(|x| x.to_string()).collect();
    let result = alphabet_lower
        .iter()
        .chain(alphabet_upper.iter())
        .chain(alphabet_greek_lower.iter())
        .cloned()
        .collect();
    result
}

#[apply(make_lazy)]
#[lazy_ty(Lazy)]
pub fn alphabet_inv() -> HashMap<String, usize> {
    ALPHABET
        .iter()
        .cloned()
        .enumerate()
        .map(|(a, b)| (b, a))
        .collect()
}

pub fn is_unique<T: Eq + Hash>(col: &[T]) -> bool {
    let set: HashSet<&T> = col.iter().collect();
    col.len() == set.len()
}

pub fn counts<T: Eq + Hash>(x: impl IntoIterator<Item = T>) -> HashMap<T, usize> {
    let mut map = HashMap::default();
    for item in x {
        map.entry(item).and_modify(|x| *x += 1).or_insert(1);
    }
    map
}

pub fn counts_g_1<T: Eq + Hash>(x: impl IntoIterator<Item = T>) -> HashMap<T, usize> {
    counts(x)
        .into_iter()
        .filter_map(|(x, count)| (count > 1).then_some((x, count)))
        .collect()
}

pub fn unique_to_appearance<T: Eq + Hash + Clone>(vec: &Vec<T>) -> HashMap<T, Vec<usize>> {
    let mut map: HashMap<T, Vec<usize>> = HashMap::default();
    for (i, x) in vec.iter().enumerate() {
        map.entry(x.clone()).or_insert(vec![]).push(i);
    }
    map
}

pub fn filter_to_idx<T>(col: impl Iterator<Item = T>, f: impl Fn(&T) -> bool) -> Vec<usize> {
    col.enumerate()
        .filter(|(_i, x)| f(x))
        .map(|(i, _x)| i)
        .collect()
}

pub fn filter_out_idx<T: Clone>(col: &[T], idxs: &HashSet<usize>) -> Vec<T> {
    col.iter()
        .enumerate()
        .filter(|(i, _x)| !idxs.contains(i))
        .map(|(_i, x)| x)
        .cloned()
        .collect()
}

pub fn intersection_all<T: Eq + Hash + Copy>(sets: &Vec<HashSet<T>>) -> HashSet<T> {
    if sets.is_empty() {
        return HashSet::default();
    }
    let (first, rest) = sets.split_first().unwrap();
    rest.iter().fold(first.clone(), |acc, new| {
        acc.intersection(new).copied().collect()
    })
}

pub fn inverse_permutation(perm: &Vec<usize>) -> Vec<usize> {
    let mut result = vec![0; perm.len()];
    for (i, x) in perm.iter().enumerate() {
        result[*x] = i;
    }
    result
}

/// element at k in vec is v, max
pub fn dict_to_list(dict: &HashMap<usize, usize>, max: Option<usize>) -> Vec<usize> {
    let max = max.unwrap_or(*dict.keys().max().unwrap_or(&0));
    let mut result = vec![0; max + 1];
    for (k, v) in dict.iter() {
        result[*k] = *v;
    }
    result
}

/// Convenience function for managing a hashmap of vecs
pub fn vec_map_insert<T: Eq + Hash>(map: &mut HashMap<T, Vec<T>>, k: T, v: T) {
    map.entry(k).or_insert_with(Vec::new).push(v);
}

// wow this is a badly written macro. I'd really expect something like this to already exist
#[macro_export]
macro_rules! filter_by_variant {
    ($iterator:expr, $enum_name:ident, $variant:ident, $return_type:ty) => {{
        let mut yes: Vec<$return_type> = vec![];
        let mut no = vec![];
        for x in $iterator {
            match &*x {
                $enum_name::$variant(inner) => yes.push(inner.clone()),
                _ => no.push(x),
            }
        }
        (yes, no)
    }};
}

#[macro_export]
macro_rules! unwrap {
    ($target: expr, $pat: path) => {{
        if let $pat(a) = $target {
            // #1
            a
        } else {
            panic!("mismatch variant when cast to {}", stringify!($pat)); // #2
        }
    }};
}

#[macro_export]
macro_rules! timed {
    ($x:expr) => {{
        let timed_macro_now = std::time::Instant::now();

        let result = $x;

        let elapsed = timed_macro_now.elapsed();
        $crate::python_println!("{} took {:.2?}", stringify!($x), elapsed);
        result
    }};
    ($x:expr,$min_to_print_milis:expr,$for_real:expr) => {{
        let timed_macro_now = std::time::Instant::now();

        let result = $x;

        let elapsed = timed_macro_now.elapsed();
        if $for_real && elapsed > std::time::Duration::new(0, $min_to_print_milis * 1_000_000) {
            $crate::python_println!("{} took {:.2?}", stringify!($x), elapsed);
        }
        result
    }};
}
#[macro_export]
macro_rules! timed_value {
    ($x:expr) => {{
        let timed_macro_now = std::time::Instant::now();

        let result = $x;

        let elapsed = timed_macro_now.elapsed();
        (result, elapsed)
    }};
}

pub trait AsOp<T> {
    fn into_op(self) -> Option<T>
    where
        Self: Sized;
    fn as_op(&self) -> Option<&T>;
    fn as_mut_op(&mut self) -> Option<&mut T>;

    fn into_unwrap(self) -> T
    where
        Self: Sized,
    {
        self.into_op().unwrap()
    }
    fn as_unwrap(&self) -> &T {
        self.as_op().unwrap()
    }
    fn as_mut_unwrap(&mut self) -> &mut T {
        self.as_mut_op().unwrap()
    }

    fn map_or_clone<'a, F, O, OF>(&'a self, f: F) -> O
    where
        Self: Into<O>,
        OF: Into<O>,
        T: 'a,
        Self: Clone,
        F: FnOnce(&'a T) -> OF,
    {
        self.and_then_or_clone(|x| Some(f(x)))
    }
    fn and_then_or_clone<'a, F, O, OF>(&'a self, f: F) -> O
    where
        Self: Into<O>,
        OF: Into<O>,
        T: 'a,
        Self: Clone,
        F: FnOnce(&'a T) -> Option<OF>,
    {
        self.as_op()
            .and_then(f)
            .map(Into::into)
            .unwrap_or_else(|| self.clone().into())
    }
}

impl<T> AsOp<T> for T {
    fn into_op(self) -> Option<T> {
        Some(self)
    }
    fn as_op(&self) -> Option<&T> {
        Some(self)
    }
    fn as_mut_op(&mut self) -> Option<&mut T> {
        Some(self)
    }
}

pub fn mapping_until_end<T: Eq + Hash + Clone>(x: &T, mapping: &HashMap<T, T>) -> T {
    let mut result = x.clone();
    for _ in 0..1000 {
        match mapping.get(&result) {
            None => {
                return result;
            }
            Some(next) => {
                result = next.clone();
            }
        }
    }
    panic!("mapping_until_end didnt finish");
}

pub fn apply_fn_until_same<T: Eq + Hash + Clone + Debug + Display, F>(x: &T, f: F) -> T
where
    F: FnMut(&T) -> T,
{
    let mut f = f;
    let mut result = x.clone();
    for i in 0..1000 {
        let next = f(&result);
        if next == result {
            return result;
        }
        result = next;
        if i == 1000 - 1 {
            dbg!(&result);
            panic!("apply until same didnt finish");
        }
    }
    result
}

pub fn apply_fn_until_none<T: Clone + Debug, F>(x: &T, f: F) -> T
where
    F: FnMut(&T) -> Option<T>,
{
    let mut f = f;
    let mut result = x.clone();
    for i in 0..1000 {
        match f(&result) {
            None => return result,
            Some(new) => result = new,
        }
        if i == 1000 - 1 {
            dbg!(&result);
            panic!("apply until none didnt finish");
        }
    }
    result
}

// might be fun to make memcpy based optimized outer product where we just copy blocks into place
pub fn outer_product<T: Clone>(cols: &Vec<Vec<T>>) -> Vec<Vec<T>> {
    assert!(!cols.iter().any(|x| x.is_empty()));
    if cols.is_empty() {
        return vec![vec![]];
    }

    let mut result: Vec<Vec<T>> = Vec::with_capacity(cols.iter().map(|x| x.len()).product());
    let mut places: Vec<usize> = vec![0; cols.len()];
    loop {
        result.push(
            places
                .iter()
                .enumerate()
                .map(|(i, place)| cols[i][*place].clone())
                .collect(),
        );

        let mut moving_place = cols.len() - 1;
        loop {
            places[moving_place] += 1;
            if places[moving_place] == cols[moving_place].len() {
                places[moving_place] = 0;
                if moving_place == 0 {
                    return result;
                }
                moving_place -= 1;
            } else {
                break;
            }
        }
    }
}

#[test]
pub fn test_outer_product() {
    let ex = vec![vec![0, 1], vec![2, 3, 4]];
    dbg!(outer_product(&ex));
}

// note this adds an element at the end so you can always look up arr[i+1] to get end
pub fn cumsum<T: std::ops::AddAssign + Default + Copy>(col: &[T]) -> Vec<T> {
    col.iter()
        .chain(std::iter::once(&Default::default()))
        .scan(Default::default(), |state: &mut T, el| {
            let old_state = *state;
            *state += *el;
            Some(old_state)
        })
        .collect()
}

pub fn hashmap_collect_except_duplicates<K: Eq + Hash + Clone, V: Eq + Hash + Clone>(
    it: impl Iterator<Item = (K, V)>,
) -> HashMap<K, V> {
    let mut result = HashMap::default();
    let mut dead: HashSet<K> = HashSet::default();
    for (k, v) in it {
        if !dead.contains(&k) {
            if let Some(old) = result.insert(k.clone(), v.clone()) {
                if old != v {
                    result.remove(&k);
                    dead.insert(k);
                }
            }
        }
    }
    result
}

pub type BitMask64 = usize;
pub type BitMask128 = u128;

#[macro_export]
macro_rules! ss {
    ($s:literal) => {
        Some($s.to_owned())
    };
}

/// this exists bc cursed jupyter bug that makes prints from rust hang
#[macro_export]
macro_rules! python_println{
    ($($tt:tt)*)=>{
        {
            let mut s = format!($($tt)*);
            if s.len()>1_000_000{
                s=format!("TRIED TO PRINT TOO LONG {}",&s[..1000]);
            }
            pyo3::Python::with_gil(|py| {
                $crate::py_types::PY_UTILS.print.call1(py, (s,)).unwrap()
            });
        }
    }
}
trait FromCollect<ItemNew>: FromIterator<ItemNew> {
    fn from_collect(x: impl IntoIterator<Item = impl Into<ItemNew>>) -> Self;
}

impl<ItemNew, T: FromIterator<ItemNew>> FromCollect<ItemNew> for T {
    fn from_collect(x: impl IntoIterator<Item = impl Into<ItemNew>>) -> Self {
        x.into_iter().map(Into::into).collect()
    }
}

pub trait IterInto<Item>: IntoIterator<Item = Item> {
    fn into_collect<ItemNew, B: FromIterator<ItemNew>>(self) -> B
    where
        Item: Into<ItemNew>;

    fn into_iter_into<ItemNew>(
        self,
    ) -> std::iter::Map<<Self as IntoIterator>::IntoIter, fn(Item) -> ItemNew>
    where
        Item: Into<ItemNew>;
}

impl<Item, T: IntoIterator<Item = Item>> IterInto<Item> for T {
    fn into_collect<ItemNew, B: FromIterator<ItemNew>>(self) -> B
    where
        Item: Into<ItemNew>,
    {
        self.into_iter_into().collect()
    }

    fn into_iter_into<ItemNew>(
        self,
    ) -> std::iter::Map<<Self as IntoIterator>::IntoIter, fn(Item) -> ItemNew>
    where
        Item: Into<ItemNew>,
    {
        self.into_iter().map(Into::into)
    }
}

pub enum EmptySingleMany<T> {
    Empty,
    Single(T),
    Many(Vec<T>),
}

impl<T> EmptySingleMany<T> {
    pub fn new(mut x: Vec<T>) -> Self {
        if x.is_empty() {
            Self::Empty
        } else if x.len() == 1 {
            Self::Single(x.pop().unwrap().into())
        } else {
            Self::Many(x)
        }
    }
}

impl<T> From<Vec<T>> for EmptySingleMany<T> {
    fn from(x: Vec<T>) -> Self {
        EmptySingleMany::new(x)
    }
}

pub fn split_first_take<T: Default>(x: &mut [T]) -> Option<(T, impl Iterator<Item = T> + '_)> {
    x.split_first_mut()
        .map(|(l, r)| (mem::take(l), r.into_iter().map(mem::take)))
}

pub fn flip_op_result<T, E>(x: Option<Result<T, E>>) -> Result<Option<T>, E> {
    x.map_or(Ok(None), |x| x.map(Some))
}

pub fn flip_result_op<T, E>(x: Result<Option<T>, E>) -> Option<Result<T, E>> {
    match x {
        Ok(None) => None,
        Ok(Some(x)) => Some(Ok(x)),
        Err(e) => Some(Err(e)),
    }
}

pub fn as_sorted<Item: Ord>(x: impl IntoIterator<Item = Item>) -> Vec<Item> {
    let mut out: Vec<_> = x.into_iter().collect();
    out.sort();
    out
}

// https://github.com/rust-lang/rust/pull/91589
#[inline]
pub fn arc_ref_clone<T: Clone>(x: &Arc<T>) -> T {
    arc_unwrap_or_clone(x.clone())
}
#[inline]
pub fn arc_unwrap_or_clone<T: Clone>(x: Arc<T>) -> T {
    Arc::try_unwrap(x).unwrap_or_else(|arc| (*arc).clone())
}
#[inline]
pub fn rc_unwrap_or_clone<T: Clone>(x: Rc<T>) -> T {
    Rc::try_unwrap(x).unwrap_or_else(|rc| (*rc).clone())
}

#[macro_export]
macro_rules! simple_from {
    (|$var:ident : $a:ty| -> $b:ty { $($fill:tt)* }) => {
        impl From<$a> for $b {
            fn from($var: $a) -> Self {
                $($fill)*
            }
        }

    };
}

#[macro_export]
macro_rules! simple_default {
    ($a:ty { $($fill:tt)* }) => {
        impl Default for $a {
            fn default() -> Self {
                $($fill)*
            }
        }

    };
}

pub fn with_context_failable<T, E, Con: Context<T, E>, F, C>(context: Con, f: F) -> Result<T>
where
    C: std::fmt::Display + Send + Sync + 'static + Default,
    F: FnOnce() -> Result<C>,
{
    let mut res: Result<()> = Ok(());
    let out = context.with_context(|| match f() {
        Ok(x) => x,
        Err(e) => {
            res = Err(e);
            Default::default()
        }
    });
    res.context("failed to get context for earlier error!")?;
    out
}

/// UNTESTED!
pub fn fuse_maps<K, V, NewV, S>(
    mut maps: Vec<::std::collections::HashMap<K, V, S>>,
    mut f: impl FnMut(Box<dyn Iterator<Item = V> + '_>) -> NewV,
) -> ::std::collections::HashMap<K, NewV, S>
where
    S: Default + BuildHasher,
    K: Clone + Hash + Debug + Eq,
{
    match split_first_take(&mut maps) {
        None => ::std::collections::HashMap::default(),
        Some((first, rest)) => {
            let mut rest: Vec<_> = rest.collect();
            let first_keys: HashSet<_> = first.keys().cloned().collect();
            for r in &rest {
                let sub = r.keys().cloned().collect();
                assert_eq!(first_keys, sub);
            }

            first
                .into_iter()
                .map(|(k, v)| {
                    let out = f(Box::new(
                        std::iter::once(v).chain(rest.iter_mut().map(|m| m.remove(&k).unwrap())),
                    ));
                    (k, out)
                })
                .collect()
        }
    }
}

#[derive(Debug, Clone)]
pub struct Multizip<T>(pub Vec<T>);

impl<T> Iterator for Multizip<T>
where
    T: Iterator,
    <T as Iterator>::Item: Debug,
{
    type Item = Vec<T::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.0.is_empty() {
            return None;
        }
        self.0.iter_mut().map(Iterator::next).collect()
    }
}

#[macro_export]
macro_rules! fn_struct{
    ($vis:vis $name:ident : $fn_ty:tt($($iname:ident:$ity:ty),*)->$retty:ty
     $(;
    {
        $(
        $enum_name:ident ($type_name:ty),
        )*
    })?
     )=>{
        paste::paste!{
            #[derive(Clone,Debug)]
            $vis enum $name{
                Dyn([< $name DynStruct >]),
                Py($crate::py_types::PyCallable),
                $($(
                    $enum_name($type_name),
                )*)*
            }
            impl $name{
                fn new_dyn(f:Box<dyn Fn($($ity),*)->anyhow::Result<$retty> + Send + Sync>)->Self{
                    Self::Dyn([< $name DynStruct >] (std::sync::Arc::new(f)))
                }
            }

            $crate::fn_struct!(@mk_call $name $fn_ty($($iname:$ity),*)->$retty; ($($($enum_name,)*)*));

            #[pyclass]
            #[derive(Clone)]
            $vis struct [< $name DynStruct >] (
                std::sync::Arc<dyn Fn($($ity),*)->anyhow::Result<$retty> + Send + Sync>
            );
            #[pymethods]
            impl  [< $name DynStruct >]{
                fn __call__( &self,
                    _py: Python<'_>,$($iname:$ity),*)->anyhow::Result<$retty>{
                        self.0($($iname),*)
                }
            }
            impl<'source> pyo3::FromPyObject<'source> for $name{
                fn extract(inp: &'source pyo3::PyAny) -> pyo3::PyResult<Self> {
                    if let Ok(d) = inp.extract() {
                        return Ok(Self::Dyn(d));
                    }
                    Ok(Self::Py(inp.extract()?))
                }
            }
            impl IntoPy<PyObject> for $name{
                fn into_py(self, py: Python<'_>) -> PyObject {
                    match self{
                        Self::Py(x)=>x.into_py(py),
                        Self::Dyn(dynf)=>dynf.into_py(py),
                        $($(
                        Self::$enum_name(x)=>x.into_py(py),
                        )*)*
                    }
                }
            }
            impl std::fmt::Debug for [< $name DynStruct >]{
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    write!(f,"dyn_fn_struct")
                }
            }
        }
    };
    (@mk_call $name:ident Fn($($iname:ident:$ity:ty),*) -> $retty:ty; ($($enum_name:ident,)*)) => {
        impl $name {
            fn call(&self,$($iname:$ity),*)->anyhow::Result<$retty>{
                // nested macro to deal with expansion resolution
                #[allow(unused_macros)]
                macro_rules! stuff {
                    ($x:expr) => (
                        $x.call($($iname,)*)
                    )
                }

                match &self{
                    Self::Py(pyobj)=>rr_util::pycall!(pyobj,($($iname,)*),anyhow),
                    Self::Dyn(dynf)=>dynf.0($($iname),*),
                    $(
                        Self::$enum_name(x)=>stuff!(x),
                    )*
                }
            }
        }
    };
    (@mk_call $name:ident FnMut($($iname:ident:$ity:ty),*) -> $retty:ty; ($($enum_name:ident,)*)) => {
        impl $name {
            fn call(&mut self,$($iname:$ity),*)->anyhow::Result<$retty>{
                // nested macro to deal with expansion resolution
                #[allow(unused_macros)]
                macro_rules! stuff {
                    ($x:expr) => (
                        $x.call($($iname,)*)
                    )
                }

                match self{
                    Self::Py(pyobj)=>rr_util::pycall!(pyobj,($($iname,)*),anyhow),
                    Self::Dyn(dynf)=>dynf.0($($iname),*),
                    $(
                        Self::$enum_name(x)=>stuff!(x),
                    )*
                }
            }
        }

    }


}

pub fn indent(x: String, width: usize) -> String {
    let tab = " ".repeat(width);

    x.lines().map(|s| tab.clone() + s).join("\n")
}

pub fn indent_butfirst(x: String, width: usize) -> String {
    let tab = " ".repeat(width) + "\n";

    x.lines().join(&tab)
}

/// https://stackoverflow.com/questions/64498617/how-to-transpose-a-vector-of-vectors-in-rust
pub fn transpose<T>(v: Vec<Vec<T>>, inner_len: usize) -> Vec<Vec<T>> {
    for x in &v {
        assert_eq!(x.len(), inner_len);
    }

    let mut iters: Vec<_> = v.into_iter().map(|n| n.into_iter()).collect();
    (0..inner_len)
        .map(|_| {
            iters
                .iter_mut()
                .map(|n| n.next().unwrap())
                .collect::<Vec<T>>()
        })
        .collect()
}

pub struct DimNumMaker {
    pub running: usize,
}

impl Default for DimNumMaker {
    fn default() -> Self {
        Self { running: 0 }
    }
}

impl DimNumMaker {
    pub fn next(&mut self) -> usize {
        let out = self.running;
        self.running += 1;
        out
    }
    pub fn next_range(&mut self, len: usize) -> std::ops::Range<usize> {
        let out = self.running..self.running + len;
        self.running += len;
        out
    }
}
