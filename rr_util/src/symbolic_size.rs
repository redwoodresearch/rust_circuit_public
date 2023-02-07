use std::{collections::BTreeSet, fmt, iter, str::FromStr};

use anyhow::{bail, Result};
use macro_rules_attribute::apply;
use once_cell::sync::Lazy;
use pyo3::{exceptions::PyOverflowError, prelude::*};
use rustc_hash::FxHashMap as HashMap;
use thiserror::Error;
use z3::{self, ast, ast::Ast};

use crate::{
    python_error_exception,
    tensor_util::{parse_numeric, ParseError, Shape, UINT_WITH_UNDERSCORE},
};

const PRIME_BYTES: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/primes.bin"));

/// these start at >10_000.
/// Maybe this is too small?
pub static SYMBOLIC_SIZES: Lazy<Vec<usize>> = Lazy::new(|| {
    PRIME_BYTES
        .chunks_exact(8)
        .map(|x| usize::from_le_bytes(x.try_into().unwrap()))
        .take(256)
        .collect()
});

#[pyfunction]
pub fn symbolic_sizes() -> Vec<usize> {
    SYMBOLIC_SIZES.clone()
}

#[derive(Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
#[pyclass]
pub struct SymbolicSizeProduct {
    #[pyo3(get)]
    pub other_factor: usize,
    #[pyo3(get)]
    pub symbolic_sizes: Vec<usize>,
}

impl Default for SymbolicSizeProduct {
    fn default() -> Self {
        SymbolicSizeProduct {
            other_factor: 1,
            symbolic_sizes: Vec::new(),
        }
    }
}

impl From<usize> for SymbolicSizeProduct {
    // TODO: make this faster as needed
    fn from(value: usize) -> Self {
        if value < SYMBOLIC_SIZES[0] {
            return SymbolicSizeProduct {
                other_factor: value,
                symbolic_sizes: Vec::new(),
            };
        }
        let mut running_value = value;
        let mut symbolic_sizes = Vec::new();
        for (i, p) in SYMBOLIC_SIZES.iter().enumerate() {
            if running_value < *p {
                break;
            }
            while running_value % p == 0 {
                symbolic_sizes.push(i);
                running_value /= p;
            }
        }
        return SymbolicSizeProduct {
            symbolic_sizes,
            other_factor: running_value,
        };
    }
}

impl TryFrom<SymbolicSizeProduct> for usize {
    type Error = SymbolicSizeOverflowError;
    fn try_from(value: SymbolicSizeProduct) -> Result<Self, Self::Error> {
        let sizes_iter = value
            .symbolic_sizes
            .iter()
            .map(|i| SYMBOLIC_SIZES[*i])
            .chain(iter::once(value.other_factor));
        checked_product(sizes_iter).ok_or(SymbolicSizeOverflowError::ProductTooLarge {
            symbolic_size_product: value,
        })
    }
}

fn checked_product(x: impl IntoIterator<Item = usize>) -> Option<usize> {
    x.into_iter().fold(Some(1usize), |old, new| {
        old.map(|old| old.checked_mul(new)).flatten()
    })
}

impl fmt::Debug for SymbolicSizeProduct {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self, f)
    }
}
impl fmt::Display for SymbolicSizeProduct {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.symbolic_sizes.is_empty() {
            f.write_str(&self.other_factor.to_string())
        } else {
            f.write_str(
                &(self.other_factor != 1)
                    .then(|| self.other_factor.to_string())
                    .into_iter()
                    .chain(self.symbolic_sizes.iter().map(|x| format!("{}s", x)))
                    .collect::<Vec<_>>()
                    .join("*"),
            )
        }
    }
}

pub static MAYBE_SYMBOLIC_SIZE: Lazy<String> = Lazy::new(|| format!(r"{}s?", UINT_WITH_UNDERSCORE)); // we allow underscores
pub static SIZE_PROD_MATCH: Lazy<String> =
    Lazy::new(|| format!(r"(?:{d}\s*\*\s*)*{d}", d = *MAYBE_SYMBOLIC_SIZE));

impl FromStr for SymbolicSizeProduct {
    type Err = ParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parse_item = |x: &str| -> Result<_, _> {
            let x = x.trim();
            let out = if x.chars().last() == Some('s') {
                let mut chars = x.chars().collect::<Vec<_>>();
                chars.pop(); // drop last
                let symbolic_idx = parse_numeric(&chars.into_iter().collect::<String>())?;
                if symbolic_idx >= SYMBOLIC_SIZES.len() {
                    return Err(ParseError::SymbolicSizeNumberOutOfBounds {
                        i: symbolic_idx,
                        bound: SYMBOLIC_SIZES.len(),
                    });
                }

                (true, symbolic_idx)
            } else {
                (false, parse_numeric(x)?)
            };
            Ok(out)
        };

        let all_sizes = if s.contains('*') {
            s.split('*').map(parse_item).collect()
        } else {
            parse_item(s).map(|x| vec![x])
        };
        let all_sizes = all_sizes?;

        let get_factors = || {
            all_sizes
                .iter()
                .filter_map(|(is_sym, s)| (!*is_sym).then_some(*s))
        };

        let other_factor =
            checked_product(get_factors()).ok_or_else(|| ParseError::FactorProductTooLarge {
                factors: get_factors().collect(),
                string: s.to_owned(),
            })?;
        let symbolic_sizes = all_sizes
            .iter()
            .filter_map(|(is_sym, s)| (*is_sym).then_some(*s))
            .collect();

        Ok(SymbolicSizeProduct {
            other_factor,
            symbolic_sizes,
        })
    }
}

impl SymbolicSizeProduct {
    pub fn has_symbolic(x: usize) -> bool {
        !Self::from(x).symbolic_sizes.is_empty()
    }
    pub fn canon(self) -> Self {
        if self.other_factor == 0 {
            return Self {
                other_factor: 0,
                symbolic_sizes: Vec::new(),
            };
        }

        let mut symbolic_sizes = self.symbolic_sizes;
        symbolic_sizes.sort();

        Self {
            other_factor: self.other_factor,
            symbolic_sizes,
        }
    }
    pub fn parse_to_usize(s: &str) -> Result<usize> {
        Ok(Self::from_str(s)?.try_into()?)
    }
}

#[pymethods]
impl SymbolicSizeProduct {
    pub fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}

/// TODO: printing + maybe sending to python and whatever
/// TODO: simplify instead of just using a set. (can be done with z3)
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct SymbolicSizeConstraints(BTreeSet<SymbolicSizeConstraint>);

impl IntoPy<PyObject> for SymbolicSizeConstraints {
    fn into_py(self, py: Python<'_>) -> PyObject {
        {
            self.0.into_py(py)
        }
    }
}

#[pyclass]
#[derive(Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct SymbolicSizeConstraint {
    #[pyo3(get)]
    pub l: SymbolicSizeProduct,
    #[pyo3(get)]
    pub r: SymbolicSizeProduct,
}

impl fmt::Display for SymbolicSizeConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format!("{}={}", self.l, self.r))
    }
}

impl fmt::Debug for SymbolicSizeConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl SymbolicSizeConstraint {
    // TODO: canonicalize, etc
    pub fn get_new_from(l: usize, r: usize) -> Result<Option<Self>> {
        Self::get_new(l.into(), r.into())
    }

    pub fn get_new(
        l_size_prod: SymbolicSizeProduct,
        r_size_prod: SymbolicSizeProduct,
    ) -> Result<Option<Self>> {
        let mut l_size_prod: SymbolicSizeProduct = l_size_prod.canon();
        let mut r_size_prod: SymbolicSizeProduct = r_size_prod.canon();

        if l_size_prod.other_factor == 0 && r_size_prod.other_factor == 0 {
            return Ok(None);
        }
        if l_size_prod == r_size_prod {
            return Ok(None);
        }

        for l_fst in [false, true] {
            let (l_size_prod, r_size_prod) = if l_fst {
                (&mut l_size_prod, &mut r_size_prod)
            } else {
                (&mut r_size_prod, &mut l_size_prod)
            };
            if l_size_prod.symbolic_sizes.is_empty()
                && (r_size_prod.other_factor == 0
                    || l_size_prod.other_factor % r_size_prod.other_factor != 0)
            {
                bail!(SymbolicSizeSetError::FactorDoesntDivideSetTo {
                    factor: r_size_prod.other_factor,
                    set_to: l_size_prod.other_factor,
                    size_prod: std::mem::take(r_size_prod),
                })
            }

            if l_size_prod.other_factor == 0 {
                *r_size_prod = SymbolicSizeProduct {
                    other_factor: 1,
                    ..r_size_prod.clone()
                };
            }
        }

        let (l, r) = if l_size_prod.symbolic_sizes.len() > r_size_prod.symbolic_sizes.len() {
            (l_size_prod, r_size_prod)
        } else {
            (r_size_prod, l_size_prod)
        };

        Ok(Some(Self { l, r }))
    }
}

#[pymethods]
impl SymbolicSizeConstraint {
    fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}

impl Default for SymbolicSizeConstraints {
    fn default() -> Self {
        Self::empty()
    }
}

impl From<SymbolicSizeConstraints> for BTreeSet<SymbolicSizeConstraint> {
    fn from(value: SymbolicSizeConstraints) -> Self {
        value.0
    }
}

impl TryFrom<BTreeSet<SymbolicSizeConstraint>> for SymbolicSizeConstraints {
    type Error = anyhow::Error;
    fn try_from(constraints: BTreeSet<SymbolicSizeConstraint>) -> Result<Self, Self::Error> {
        Self::new(constraints)
    }
}

impl SymbolicSizeConstraints {
    pub fn empty() -> Self {
        Self(Default::default())
    }

    pub fn constraints(&self) -> &BTreeSet<SymbolicSizeConstraint> {
        &self.0
    }

    pub fn into_constraints(self) -> BTreeSet<SymbolicSizeConstraint> {
        self.into()
    }

    pub fn new(constraints: BTreeSet<SymbolicSizeConstraint>) -> Result<Self> {
        if constraints.is_empty() {
            return Ok(Self(constraints));
        }

        // TODO: incremental or other clever stuff maybe?
        // This is in fact a bit sad...
        if check_constraints(constraints.clone()) == z3::SatResult::Unsat {
            bail!(SymbolicSizeSetError::FailedToSatisfyContraints { constraints })
        }

        Ok(Self(constraints.clone()))
    }
}

#[cached::proc_macro::cached]
fn check_constraints(constraints: BTreeSet<SymbolicSizeConstraint>) -> z3::SatResult {
    let mut all_symbolic_sizes = BTreeSet::default();

    for constraint in &constraints {
        all_symbolic_sizes.extend(constraint.l.symbolic_sizes.clone());
        all_symbolic_sizes.extend(constraint.r.symbolic_sizes.clone());
    }
    let sym_mapping: HashMap<_, _> = all_symbolic_sizes
        .into_iter()
        .enumerate()
        .map(|(i, x)| (x, i))
        .collect();

    let cfg = z3::Config::new();
    let ctx = z3::Context::new(&cfg);
    let opt = z3::Solver::new(&ctx);
    let syms: Vec<_> = (0..sym_mapping.len())
        .map(|i| ast::Int::new_const(&ctx, i as u32))
        .collect();
    for constraint in constraints.clone() {
        let get_expr = |x: SymbolicSizeProduct| {
            let to_mul: Vec<_> = x
                .symbolic_sizes
                .iter()
                .map(|i| syms[sym_mapping[i]].clone())
                .chain(std::iter::once(ast::Int::from_u64(
                    &ctx,
                    x.other_factor as u64,
                )))
                .collect();
            ast::Int::mul(&ctx, &to_mul.iter().collect::<Vec<_>>())
        };
        opt.assert(&(get_expr(constraint.l)._eq(&get_expr(constraint.r))));
    }
    opt.check()
}

#[apply(python_error_exception)]
#[base_error_name(SymbolicSizeOverflow)]
#[base_exception(PyOverflowError)]
#[derive(Error, Debug, Clone)]
pub enum SymbolicSizeOverflowError {
    #[error("symbolic_size_product={symbolic_size_product:?} ({e_name})")]
    ProductTooLarge {
        symbolic_size_product: SymbolicSizeProduct,
    },

    #[error("i={i} >= {} ({e_name})", SYMBOLIC_SIZES.len())]
    SymbolicSizeNumberOutOfBounds { i: usize },
}

#[apply(python_error_exception)]
#[base_error_name(SymbolicSizeSet)]
#[base_exception(PyOverflowError)]
#[derive(Error, Debug, Clone)]
pub enum SymbolicSizeSetError {
    #[error("size_prod={size_prod} is not 0, but tried to set to 0 ({e_name})")]
    TriedToSetNonZeroToZero { size_prod: SymbolicSizeProduct },

    #[error("factor={factor} doesn't divide set_to={set_to} (trying to set symbolic size). size_prod={size_prod} ({e_name})")]
    FactorDoesntDivideSetTo {
        factor: usize,
        set_to: usize,
        size_prod: SymbolicSizeProduct,
    },

    #[error("l_factor={l_factor} != r_factor={r_factor}\nl_size_prod={l_size_prod} r_size_prod={r_size_prod} ({e_name})")]
    NoSymbolicAndSizesNotEqual {
        l_factor: usize,
        r_factor: usize,
        l_size_prod: SymbolicSizeProduct,
        r_size_prod: SymbolicSizeProduct,
    },

    #[error("constraints={constraints:?} ({e_name})")]
    FailedToSatisfyContraints {
        constraints: BTreeSet<SymbolicSizeConstraint>,
    },
}

pub fn shape_first_symbolic_dim(shape: &Shape) -> Option<usize> {
    shape
        .iter()
        .position(|x| SymbolicSizeProduct::has_symbolic(*x))
}
