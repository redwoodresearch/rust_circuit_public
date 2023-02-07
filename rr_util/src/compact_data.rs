use std::{
    iter::zip,
    num::NonZeroUsize,
    ops::{Deref, DerefMut},
};

use itertools::Itertools;
use pyo3::{FromPyObject, IntoPy, PyAny, PyObject, PyResult, Python};
use rustc_hash::FxHashSet as HashSet;

/// stores 7 elements inline
// This is 8 bytes long and stores either 7 u8s or an owning pointer (like Box) to a Vec<u8>
// It tracks whether its a pointer with the lowest order bit - if this is set, its inline, if not its pointer
// bc rust pointers must be aligned, pointer will always have last bit unset
// if inline, rest of first byte shifted down one is length
// data starts from the second byte
// similar to https://docs.rs/thin_str/latest/thin_str/, but isn't utf8
// this relies on little endian (pointer is little endian) and 64 bit (checked in rust_circuit/src/lib.rs)
#[repr(transparent)]
pub struct TinyVecU8(NonZeroUsize);

impl TinyVecU8 {
    // raw stuff
    unsafe fn raw_mut(&mut self) -> &mut [u8; 8] {
        std::mem::transmute::<&mut NonZeroUsize, &mut [u8; 8]>(&mut self.0)
    }
    unsafe fn raw(&self) -> &[u8; 8] {
        std::mem::transmute::<&NonZeroUsize, &[u8; 8]>(&self.0)
    }
    unsafe fn pointer_mut(&mut self) -> &mut Box<Vec<u8>> {
        std::mem::transmute::<&mut NonZeroUsize, &mut Box<Vec<u8>>>(&mut self.0)
    }
    unsafe fn pointer(&self) -> &Box<Vec<u8>> {
        std::mem::transmute::<&NonZeroUsize, &Box<Vec<u8>>>(&self.0)
    }
    #[inline]
    unsafe fn inline_len(&self) -> usize {
        (self.raw()[0] >> 1) as usize
    }
    #[inline]
    unsafe fn set_inline_len(&mut self, value: usize) {
        self.raw_mut()[0] = (value as u8) << 1 | 0b1;
    }

    // safe api stuff
    #[inline]
    pub fn is_inline(&self) -> bool {
        unsafe { self.raw()[0] & 0b1 != 0 }
    }
    pub fn as_slice(&self) -> &[u8] {
        if self.is_inline() {
            unsafe { &self.raw()[1..1 + self.inline_len()] }
        } else {
            unsafe { &self.pointer()[..] }
        }
    }
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.is_inline() {
            unsafe {
                let lenny = self.inline_len();
                &mut self.raw_mut()[1..1 + lenny]
            }
        } else {
            unsafe { &mut self.pointer_mut()[..] }
        }
    }
    pub unsafe fn new_raw(x: usize) -> Self {
        Self(NonZeroUsize::new_unchecked(x))
    }
    // wanted this to call set_slice_outline, but that would break aliasing rules
    fn convert_to_outline(&mut self) {
        assert!(self.is_inline());
        let values = self.as_slice();
        let boxed = Box::new(Vec::from(values));
        let box_raw: *const Vec<u8> = Box::into_raw(boxed);
        self.0 = NonZeroUsize::new(box_raw as usize).unwrap();
        assert!(!self.is_inline());
    }
    pub fn push(&mut self, value: u8) {
        if self.is_inline() {
            unsafe {
                let inline_len = self.inline_len();
                if inline_len == 7 {
                    self.convert_to_outline();
                    self.pointer_mut().push(value);
                } else {
                    let raw = self.raw_mut();
                    raw[inline_len + 1] = value;
                    self.set_inline_len(inline_len + 1);
                }
            }
        } else {
            unsafe {
                self.pointer_mut().push(value);
            }
        }
    }
    pub fn new() -> Self {
        Self(NonZeroUsize::new(1).unwrap()) // yes inline, 0 inline size, 0 other things
    }
    pub fn set_slice_outline(&mut self, values: &[u8]) {
        let boxed = Box::new(Vec::from(values));
        self.0 = NonZeroUsize::new(Box::into_raw(boxed) as usize).unwrap();
    }
    pub fn from_slice(values: &[u8]) -> Self {
        let mut result = Self::new();
        if values.len() > 7 {
            result.set_slice_outline(values);
        } else {
            unsafe {
                result.set_inline_len(values.len());
                result.raw_mut()[1..1 + values.len()].copy_from_slice(values);
            }
        }
        return result;
    }
    pub fn unique(&self) -> Self {
        self.iter().unique().cloned().collect()
    }
}

impl From<Vec<u8>> for TinyVecU8 {
    fn from(value: Vec<u8>) -> Self {
        Self::from_slice(&value[..])
    }
}

impl From<TinyVecU8> for Vec<u8> {
    fn from(value: TinyVecU8) -> Self {
        Vec::from(value.as_slice())
    }
}

impl Deref for TinyVecU8 {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}
impl DerefMut for TinyVecU8 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl Drop for TinyVecU8 {
    fn drop(&mut self) {
        if !self.is_inline() {
            // println!("dropping");
            drop(unsafe { self.pointer() });
        }
    }
}
impl Clone for TinyVecU8 {
    fn clone(&self) -> Self {
        if self.is_inline() {
            return Self(self.0);
        }
        Self::from_slice(self.as_slice())
    }
}

impl IntoPy<PyObject> for TinyVecU8 {
    fn into_py(self, py: Python<'_>) -> PyObject {
        Vec::from(self).into_py(py)
    }
}
impl<'source> FromPyObject<'source> for TinyVecU8 {
    fn extract(args_obj: &'source PyAny) -> PyResult<Self> {
        let vec: Vec<u8> = args_obj.extract()?;

        Ok(vec.into())
    }
}

impl FromIterator<u8> for TinyVecU8 {
    fn from_iter<I: IntoIterator<Item = u8>>(iter: I) -> TinyVecU8 {
        let mut c = TinyVecU8::new();

        for i in iter {
            c.push(i);
        }

        c
    }
}
impl<'a> IntoIterator for &'a TinyVecU8 {
    type Item = &'a u8;
    type IntoIter = core::slice::Iter<'a, u8>;
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

impl PartialEq for TinyVecU8 {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}
impl Eq for TinyVecU8 {}

impl std::fmt::Debug for TinyVecU8 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_slice().fmt(f)
    }
}
impl ::std::hash::Hash for TinyVecU8 {
    fn hash<H: ::std::hash::Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state)
    }
}

#[macro_export]
macro_rules! tu8v{
    ()=>{
        $crate::compact_data::TinyVecU8::new()
    };
    ($single:expr)=>{
        {
            let x:u8 = $single; // for callsite typedness
            unsafe{$crate::compact_data::TinyVecU8::new_raw(0b11|((x as usize)<<8))} //
        }
    };
    ($($tt:tt)*)=>{
        vec![$($tt)*].into()
    };
}

#[derive(Default, Copy, Clone)]
pub struct U8Set([u64; 4]); // 64 bits * 4 = 256 bits
impl U8Set {
    #[inline]
    pub fn contains(&self, x: u8) -> bool {
        self.0[(x >> 6) as usize] & (1 << (x & 0b00111111)) != 0
    }
    #[inline]
    pub fn insert(&mut self, x: u8) {
        self.0[(x >> 6) as usize] |= 1 << (x & 0b00111111);
    }
    pub fn len(&self) -> usize {
        self.0.into_iter().map(|x| x.count_ones() as usize).sum() // portable_simd doesn't have count_ones
    }
    pub fn is_subset(&self, other: &U8Set) -> bool {
        zip(self.0, other.0).all(|(sub, sup)| sub & !sup == 0) // could use simd
    }
    pub fn is_superset(&self, other: &U8Set) -> bool {
        other.is_subset(&self)
    }
    pub fn union(&self, other: &U8Set) -> U8Set {
        Self([
            self.0[0] | other.0[0],
            self.0[1] | other.0[1],
            self.0[2] | other.0[2],
            self.0[3] | other.0[3],
        ])
    }
    pub fn intersection(&self, other: &U8Set) -> U8Set {
        Self([
            self.0[0] & other.0[0],
            self.0[1] & other.0[1],
            self.0[2] & other.0[2],
            self.0[3] & other.0[3],
        ])
    }
}

impl FromIterator<u8> for U8Set {
    fn from_iter<I: IntoIterator<Item = u8>>(iter: I) -> U8Set {
        let mut c = U8Set::default();

        for i in iter {
            c.insert(i);
        }

        c
    }
}
pub struct U8SetIterator {
    u8set: U8Set,
    chunk: usize,
}
impl Iterator for U8SetIterator {
    type Item = u8;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.u8set.0[self.chunk].trailing_zeros() {
                64 => {
                    if self.chunk == 3 {
                        return None;
                    }
                    self.chunk += 1;
                }
                num_zeros => {
                    self.u8set.0[self.chunk] &= !(1 << num_zeros);
                    return Some(num_zeros as u8 + 64 * self.chunk as u8);
                }
            }
        }
    }
}

impl IntoIterator for U8Set {
    type Item = u8;
    type IntoIter = U8SetIterator;
    fn into_iter(self) -> Self::IntoIter {
        U8SetIterator {
            u8set: self,
            chunk: 0,
        }
    }
}

impl std::fmt::Debug for U8Set {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.into_iter().collect::<HashSet<u8>>().fmt(f)
    }
}
impl IntoPy<PyObject> for U8Set {
    fn into_py(self, py: Python<'_>) -> PyObject {
        self.into_iter().collect::<HashSet<u8>>().into_py(py)
    }
}
impl<'source> FromPyObject<'source> for U8Set {
    fn extract(args_obj: &'source PyAny) -> PyResult<Self> {
        let vec: HashSet<u8> = args_obj.extract()?;

        Ok(vec.into_iter().collect())
    }
}
#[test]
fn test_u8set() {
    let mut u8s = U8Set::default();
    dbg!(u8s.len());
    u8s.insert(0);
    u8s.insert(243);
    u8s.insert(150);
    u8s.insert(255);
    assert!(u8s.into_iter().collect::<Vec<u8>>()[..] == vec![0, 150, 243, 255]);
    assert!(u8s.len() == 4);
    dbg!(u8s.into_iter().collect::<Vec<u8>>(), u8s.len());
    assert!(u8s.contains(150) && !u8s.contains(1));
    let u8s2: U8Set = vec![0, 1, 2, 2, 50].into_iter().collect();
    assert!(u8s2.union(&u8s).into_iter().collect::<Vec<u8>>() == vec![0, 1, 2, 50, 150, 243, 255]);
    dbg!(u8s2.intersection(&u8s).into_iter().collect::<Vec<u8>>() == vec![0]);
}

#[test]
fn test_tiny_u8() {
    let mut col = TinyVecU8::new();
    dbg!(col.len());
    dbg!(col.is_empty());
    dbg!(&col);
    col.push(1);
    dbg!(col[0]);
    dbg!(col.len());
    dbg!(col.is_empty());
    dbg!(&col);
    col.push(2);
    dbg!(&col);
    col.push(3);
    dbg!(&col);
    col.push(4);
    dbg!(col[2]);
    dbg!(&col);
    col.push(5);
    dbg!(&col);
    col.push(6);
    dbg!(&col);
    col.push(7);
    dbg!(&col);
    col.push(8);
    dbg!(col.len());
    dbg!(col.is_empty());

    dbg!(&col);
    let col2 = TinyVecU8::from_slice(&[10, 12, 13, 14, 15, 16, 17, 18, 19, 20]);
    dbg!(&col2);
    for i in &*col2 {
        println!("{}", i);
    }
    let col3 = TinyVecU8::from_slice(&[10, 12, 13]);
    dbg!(&col3);
    for i in &*col3 {
        println!("{}", i);
    }
}
