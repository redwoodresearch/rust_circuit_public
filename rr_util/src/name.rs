use std::{num::NonZeroU32, sync::Mutex};

use once_cell::sync::Lazy;
use pyo3::prelude::*;
use rustc_hash::FxHashMap as HashMap;
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Name(pub NonZeroU32); // nonzero for null pointer optimization (Option<Name> is only 32 bits)
impl Name {
    pub fn new(string: &str) -> Self {
        let mut interner = NAME_INTERNER.lock().unwrap();
        if let Some(z) = interner.to_name.get(string) {
            return *z;
        }
        let to_leak = string.to_owned();
        let staticy = to_leak.leak();
        let result = Self(unsafe { NonZeroU32::new_unchecked(interner.names.len() as u32 + 1) });
        interner.to_name.insert(staticy, result);
        interner.names.push(staticy);
        result
    }
    pub fn string(&self) -> String {
        self.str().to_owned()
    }
    pub fn str(&self) -> &'static str {
        NAME_INTERNER.lock().unwrap().names[self.0.get() as usize - 1]
    }
    pub fn string_maybe_empty(x: Option<Name>) -> String {
        x.map(|x| x.into()).unwrap_or("".to_owned())
    }
    pub fn str_maybe_empty(x: Option<Name>) -> &'static str {
        x.map(|x| x.into()).unwrap_or("")
    }
}

impl PartialOrd for Name {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.str().partial_cmp(other.str())
    }
}
impl Ord for Name {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.str().cmp(other.str())
    }
}

impl Into<Name> for &str {
    fn into(self) -> Name {
        Name::new(self)
    }
}
impl Into<Name> for String {
    fn into(self) -> Name {
        Name::new(&self)
    }
}

impl Into<&'static str> for Name {
    fn into(self) -> &'static str {
        self.str()
    }
}

// Not recommended! just use &'static str!
impl Into<String> for Name {
    fn into(self) -> String {
        self.str().to_owned()
    }
}

impl std::ops::Deref for Name {
    type Target = str;

    fn deref(&self) -> &'static Self::Target {
        self.str()
    }
}

impl std::fmt::Display for Name {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.str().fmt(f)
    }
}
impl std::fmt::Debug for Name {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.str().fmt(f)
    }
}

impl IntoPy<PyObject> for Name {
    fn into_py(self, py: Python<'_>) -> PyObject {
        self.string().into_py(py)
    }
}
impl<'source> FromPyObject<'source> for Name {
    fn extract(args_obj: &'source PyAny) -> PyResult<Self> {
        let string: String = args_obj.extract()?;
        Ok(Name::new(&string))
    }
}

#[derive(Default, Debug)]
pub struct NameInterner {
    names: Vec<&'static str>,
    to_name: HashMap<&'static str, Name>,
}

impl NameInterner {
    pub fn num_interned_strings() -> usize {
        NAME_INTERNER.lock().unwrap().names.len()
    }
    pub fn total_interned_string_len() -> usize {
        NAME_INTERNER
            .lock()
            .unwrap()
            .names
            .iter()
            .map(|z| z.len())
            .sum()
    }
}

static NAME_INTERNER: Lazy<Mutex<NameInterner>> = Lazy::new(|| Mutex::new(NameInterner::default()));

#[test]
fn test_name() {
    let s1 = "hi1";
    let n1: Name = s1.into();
    dbg!(n1);
    let s2 = "hi11";
    let n2: Name = s2.into();
    dbg!(n2);
    let n3 = Name::new(&(n1.string() + "1"));
    assert!(
        n1.0 == NonZeroU32::new(1).unwrap()
            && n2.0 == NonZeroU32::new(2).unwrap()
            && n3.0 == NonZeroU32::new(2).unwrap()
    );
    assert!(n1.str() == s1 && n2.str() == s2 && n3.str() == s2);
    dbg!(n1.0, n2.0, n3.0);
    println!("{} {} {} {}", s1, n1, s2, n2);
    dbg!(n1 == n2);
}
