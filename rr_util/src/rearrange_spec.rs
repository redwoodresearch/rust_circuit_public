use std::{
    fmt::{self, Debug},
    hash,
    iter::{self, zip},
    str::FromStr,
};

use anyhow::{bail, Context, Result};
use itertools::{izip, Itertools};
use macro_rules_attribute::apply;
use once_cell::sync::Lazy;
use pyo3::{exceptions::PyValueError, prelude::*};
use regex::Regex;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use smallvec::SmallVec as Sv;
use thiserror::Error;
use uuid::uuid;

use crate::{
    compact_data::{TinyVecU8, U8Set},
    make_single_many,
    name::Name,
    py_types::{einops_repeat, PyOpAtAxes, Tensor, PY_CIRCUIT_ITEMS},
    python_error_exception, sv,
    symbolic_size::{SymbolicSizeProduct, SIZE_PROD_MATCH},
    tensor_util::{check_canon_idxs, PyParseError, Shape},
    tu8v,
    util::{
        dict_to_list, filter_out_idx, is_unique, vec_map_insert, AxisInt, EinsumAxes, HashBytes,
        NamedAxes, ALPHABET,
    },
};
/// OpSize is a memory optimization of Option<usize> that stores a 63-bit int and one "is this none" bit
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct OpSize(pub u64);
pub type OpShape = Sv<[OpSize; 8]>;

impl OpSize {
    pub const SHIFT: usize = 63;
    pub const NONE: OpSize = OpSize(1 << Self::SHIFT);
    // sometimes into/from can't do type interference, so we have aliases
    fn t(self) -> Option<usize> {
        self.into()
    }

    fn f(val: Option<usize>) -> Self {
        val.into()
    }
    pub fn is_some(&self) -> bool {
        self.0 >> Self::SHIFT == 0
    }
    pub fn is_none(&self) -> bool {
        self.0 >> Self::SHIFT != 0
    }
    pub fn unwrap(self) -> usize {
        assert!(self.is_some());
        self.0 as usize
    }
    pub fn unwrap_or(self, x: usize) -> usize {
        if self.is_none() {
            x
        } else {
            self.unwrap()
        }
    }
}

impl Debug for OpSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.t().fmt(f)
    }
}

impl From<Option<usize>> for OpSize {
    fn from(x: Option<usize>) -> Self {
        assert!(std::mem::size_of::<usize>() <= std::mem::size_of::<u64>()); // optimized out I assume
        match x {
            None => Self::NONE,
            Some(value) => {
                assert_eq!(value >> Self::SHIFT, 0);
                OpSize(value as u64)
            }
        }
    }
}

impl From<OpSize> for Option<usize> {
    fn from(x: OpSize) -> Self {
        if x.0 >> OpSize::SHIFT == 1 {
            None
        } else {
            Some(x.0 as usize)
        }
    }
}

impl<'source> FromPyObject<'source> for OpSize {
    fn extract(obj: &'source PyAny) -> PyResult<Self> {
        Ok(OpSize::f(obj.extract()?))
    }
}

impl IntoPy<PyObject> for OpSize {
    fn into_py(self, py: Python<'_>) -> PyObject {
        self.t().into_py(py)
    }
}

pub fn shape_to_op_shape(shape: &Shape) -> OpShape {
    shape.iter().map(|x| OpSize(*x as u64)).collect()
}

// RInnerInts distribution is heavy tailed - almost everything's 1, but there might be up to 10.
// we use multiple of 8 bc they're u8 and we'd just pad if we used less
pub type RInnerInts = TinyVecU8;
pub type RInts = Sv<[RInnerInts; 4]>;

#[pyclass]
#[derive(Clone, PartialEq, Eq)]
pub struct RearrangeSpec {
    #[pyo3(get)]
    pub input_ints: RInts,
    #[pyo3(get)]
    pub output_ints: RInts,
    #[pyo3(get)]
    pub int_sizes: OpShape,
}

make_single_many!(PyCountsAtAxes, usize, Vec);

pub fn check_permutation<T: Eq + hash::Hash + Clone + Into<usize>>(perm: &[T]) -> Result<()> {
    let perm_set: HashSet<_> = perm.iter().cloned().map(Into::into).collect();
    if perm.len() != perm_set.len() {
        bail!(PermError::IntsNotUnique {
            ints: perm.iter().cloned().map(Into::into).collect()
        })
    }
    if &perm_set != &(0..perm.len()).collect::<HashSet<_>>() {
        bail!(PermError::NotContiguousInts {
            ints: perm.iter().cloned().map(Into::into).collect(),
            count: perm.len(),
        })
    }
    Ok(())
}

impl fmt::Display for RearrangeSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // plausible that default display shouldn't show_size_if_present, but debug should. (idk)
        f.write_str(&self.to_string_impl(true, true, true))
    }
}
impl fmt::Debug for RearrangeSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl FromStr for RearrangeSpec {
    type Err = anyhow::Error;
    fn from_str(string: &str) -> Result<Self, Self::Err> {
        const ALPHA_NON_NUMERIC: &str = "[a-zA-Z]";
        static RE_TOP_LEVEL: Lazy<Regex> = Lazy::new(|| {
            Regex::new(&format!(
                r"({p})|({a}\w*(?::{p})?)|\(\s*((?:({a}\w*(?::{p})?|{p})\s*)*)\)",
                a = ALPHA_NON_NUMERIC,
                p = *SIZE_PROD_MATCH
            ))
            .unwrap()
        });

        let (input, output) =
            if let [input, output] = string.split("->").map(str::trim).collect::<Vec<_>>()[..] {
                (input, output)
            } else {
                bail!(RearrangeParseError::ArrowIssue {
                    string: string.to_owned(),
                });
            };

        let mut sizes: OpShape = sv![];
        let mut string_to_int: HashMap<String, u8> = HashMap::default();
        let new_int = |s: usize| {
            assert!(s > 0);
            if s - 1 > (u8::MAX as usize) {
                bail!(RearrangeParseError::TooManyAxes {
                    string: string.to_owned()
                });
            }
            Ok((s - 1) as u8)
        };
        let mut f = |s: &str| -> Result<RInts> {
            let handle_int = |num, sizes: &mut OpShape| {
                sizes.push(Some(num).into());
                new_int(sizes.len())
            };
            let parse_num = |s: &str| {
                SymbolicSizeProduct::parse_to_usize(s).context("rearrange num didn't parse")
            };
            let mut handle_string_nosize = |s: String, sizes: &mut OpShape| {
                if let Some(old) = string_to_int.get(&s) {
                    Ok(*old)
                } else {
                    sizes.push(None.into());
                    let new_int = new_int(sizes.len())?;
                    string_to_int.insert(s.to_owned(), new_int);
                    Ok(new_int)
                }
            };
            let mut handle_string = |string: String, sizes: &mut OpShape| -> Result<u8> {
                if let [string, num] = string.split(':').collect::<Vec<_>>()[..] {
                    let num = parse_num(num)?;
                    let result = handle_string_nosize(string.to_owned(), sizes)?;
                    sizes[result as usize] = Some(num).into();
                    Ok(result)
                } else {
                    handle_string_nosize(string, sizes)
                }
            };
            RE_TOP_LEVEL
                .captures_iter(s)
                .map(|tl| {
                    if let Some(c) = tl.get(1) {
                        let num = parse_num(c.as_str())?;
                        Ok(tu8v![handle_int(num, &mut sizes)?])
                    } else if let Some(c) = tl.get(2) {
                        Ok(tu8v![handle_string(c.as_str().to_owned(), &mut sizes)?])
                    } else if let Some(c) = tl.get(3) {
                        c.as_str()
                            .split_whitespace()
                            .filter(|z| !z.is_empty())
                            .map(|s| {
                                parse_num(s)
                                    .and_then(|z| handle_int(z, &mut sizes))
                                    .or_else(|_e| handle_string(s.to_owned(), &mut sizes))
                            })
                            .collect::<Result<_>>()
                    } else {
                        bail!(RearrangeParseError::FailedToMatchRegex {
                            string: string.to_owned(),
                        });
                    }
                })
                .collect()
        };

        Self::new(f(input)?, f(output)?, sizes)
            .context("failed to construct rearrange spec after parsing")
    }
}

/// RearrangeSpec encodes the same thing as a python Einops.repeat string https://einops.rocks/api/repeat/
///
/// Dimension names are encoded as integers, so the einops operation
/// repeat('(a b) -> 10 a b', tensor, a=5) is encoded as
///   input_ints = [[0, 1]]
///   output_ints = [[2], [0], [1]]
///   int_sizes = [Some(5), None, Some(10)]
///
/// If a variable i is only on the right, it is a repeat dimension, and the number of repeats
/// must be specified as int_sizes[i] = Some(repeats).
/// If a variable is on the left, it must appear on the right (we don't allow reductions). This means
/// this doesn't support squeezes normally: '1 a -> a' is not allowed, but unsqueezes are allowed.
/// (You can however get a squeeze by doing '() a -> a' or `b:1 a -> (b:1 a)`, although you should use Index instead.)
#[pymethods]
impl RearrangeSpec {
    #[new]
    pub fn new(input_ints: RInts, output_ints: RInts, int_sizes: OpShape) -> Result<Self> {
        let out = RearrangeSpec {
            input_ints,
            output_ints,
            int_sizes,
        };
        out.check_valid()
            .context(format!("Invalid rearrange spec: {:?}", out))?;
        Ok(out)
    }

    #[staticmethod]
    fn check_rank(rank: usize) -> Result<()> {
        // we could do u8::MAX+1 in most cases...
        if rank > (u8::MAX as usize) {
            bail!(RearrangeSpecError::LenShapeTooLarge { len_shape: rank })
        }
        Ok(())
    }

    #[staticmethod]
    pub fn flatten(rank: u8) -> Self {
        Self::new(
            (0..rank).map(|i| tu8v![i]).collect(),
            sv![(0..rank).collect()],
            sv![OpSize::from(None); rank as usize],
        )
        .unwrap()
    }

    #[staticmethod]
    pub fn unflatten(shape: Shape) -> Result<Self> {
        let rank = shape.len();
        Self::check_rank(rank)?;
        Ok(Self::new(
            sv![(0..rank as u8).collect()],
            (0..rank).map(|i| tu8v![i as u8]).collect(),
            shape_to_op_shape(&shape),
        )
        .unwrap())
    }

    #[staticmethod]
    pub fn unflatten_axis(ndim: usize, axis: i64, shape: Shape) -> Result<Self> {
        if ndim == 0 {
            bail!(RearrangeSpecError::CantUnflattenScalar {})
        }
        let axis = check_canon_idxs(ndim, &[axis]).context("axis out of ndim bounds")?[0];
        let new_ndim = (ndim + shape.len()) - 1;
        Self::check_rank(new_ndim)?;

        let mut initial_inp: RInts = (0..(ndim - 1) as u8).map(|i| tu8v![i]).collect();
        let unflatten_ints = (ndim - 1) as u8..(shape.len() + ndim - 1) as u8;
        initial_inp.insert(axis, unflatten_ints.clone().collect());
        let out_ints: RInts = (0..axis as u8)
            .chain(unflatten_ints)
            .chain(axis as u8..(ndim - 1) as u8)
            .map(|i| tu8v![i])
            .collect();
        let op_shape: OpShape = iter::repeat(OpSize::NONE)
            .take(ndim - 1)
            .chain(shape_to_op_shape(&shape))
            .collect();

        assert_eq!(op_shape.len(), out_ints.len());
        assert_eq!(new_ndim, out_ints.len());

        Ok(Self::new(initial_inp, out_ints, op_shape).unwrap())
    }

    #[staticmethod]
    pub fn ident(rank: u8) -> Self {
        Self::new(
            (0..rank).map(|i| tu8v![i]).collect(),
            (0..rank).map(|i| tu8v![i]).collect(),
            sv![OpSize::from(None); rank as usize],
        )
        .unwrap()
    }

    /// counts defaults to unsqueeze (1)
    #[staticmethod]
    #[pyo3(name = "expand_at_axes")]
    pub fn expand_at_axes_py(
        orig_ndim: usize,
        axes: PyOpAtAxes,
        counts: Option<PyCountsAtAxes>,
    ) -> Result<Self> {
        Self::expand_at_axes_impl(orig_ndim, axes.into_many(), counts.map(|x| x.into_many()))
    }

    #[staticmethod]
    #[pyo3(name = "unsqueeze")]
    pub fn unsqueeze_py(orig_ndim: usize, axes: PyOpAtAxes) -> Result<Self> {
        Self::expand_at_axes_impl(orig_ndim, axes.into_many(), None)
    }

    #[staticmethod]
    pub fn prepend_batch_shape(new_shape: Shape, old_rank: usize) -> Result<Self> {
        let rank = new_shape.len() + old_rank;
        Self::check_rank(rank)?;
        let out = Self::new(
            (new_shape.len()..new_shape.len() + old_rank)
                .map(|i| tu8v![i as u8])
                .collect(),
            (0..new_shape.len() + old_rank)
                .map(|i| tu8v![i as u8])
                .collect(),
            new_shape
                .iter()
                .map(|i| OpSize(*i as u64))
                .chain(std::iter::repeat(OpSize::NONE).take(old_rank))
                .collect(),
        )
        .expect("Internal error - this should always work");
        Ok(out)
    }

    pub fn to_py_rearrange_spec(&self, input_shape: Shape) -> PyObject {
        Python::with_gil(|py| {
            PY_CIRCUIT_ITEMS
                .circ_compiler_util
                .getattr(py, "RearrangeSpec")
                .unwrap()
                .getattr(py, "from_rust")
                .unwrap()
                .call(
                    py,
                    (self
                        .canonicalize(true)
                        .conform_to_input_shape(&input_shape)
                        .unwrap()
                        .fill_empty_ints_allow_invalid(),),
                    None,
                )
                .unwrap()
        })
    }
    pub fn ints_in_inp(&self) -> HashSet<AxisInt> {
        self.ints_in_inp_it().collect()
    }
    pub fn ints_in_out(&self) -> HashSet<AxisInt> {
        self.ints_in_out_it().collect()
    }
    #[staticmethod]
    pub fn new_permute(permutation: Vec<usize>) -> Result<RearrangeSpec> {
        let rank = permutation.len();
        Self::check_rank(rank)?;
        check_permutation(&permutation)?;

        let out = Self::new(
            (0..permutation.len() as u8).map(|i| tu8v![i]).collect(),
            permutation.iter().map(|i| tu8v![*i as u8]).collect(),
            sv![OpSize::NONE;permutation.len()],
        )
        .unwrap();

        Ok(out)
    }
    #[staticmethod]
    pub fn new_permute_combine(permute_combine: RInts) -> Result<RearrangeSpec> {
        let count = permute_combine.iter().flatten().count();

        Self::check_rank(count)?;

        let out = Self::new(
            (0..count as u8).map(|i| tu8v![i]).collect(),
            permute_combine,
            sv![OpSize::NONE; count],
        )
        .unwrap();
        Ok(out)
    }
    #[staticmethod]
    pub fn combine_axes_at_end(n: u8, axes_to_combine: RInnerInts) -> Result<RearrangeSpec> {
        if !axes_to_combine
            .iter()
            .cloned()
            .collect::<HashSet<_>>()
            .is_subset(&(0..n).collect())
        {
            bail!(RearrangeSpecError::AxesToCombineNotSubset { axes_to_combine, n })
        }
        let out = Self::new_permute_combine(
            (0..n)
                .filter_map(|z| {
                    if axes_to_combine.contains(&z) {
                        None
                    } else {
                        Some(tu8v![z])
                    }
                })
                .chain(std::iter::once(axes_to_combine.clone()))
                .collect(),
        )
        .expect("we can't have too high of a rank if axes to combine is subset of 0..n");
        Ok(out)
    }
    pub fn is_permute(&self) -> bool {
        // because there are no squeezes, all we need is no splits/joins and equal input and output rank
        self.input_ints.iter().all(|x| x.len() == 1)
            && self.output_ints.iter().all(|x| x.len() == 1)
            && self.input_ints.len() == self.output_ints.len()
    }
    pub fn get_fwd_permutation(&self) -> Option<Vec<usize>> {
        if !self.is_permute() {
            return None;
        }
        let output_ints_single: Vec<u8> = self.output_ints.iter().map(|x| x[0]).collect();
        Some(
            self.input_ints
                .iter()
                .map(|ints| {
                    output_ints_single
                        .iter()
                        .position(|in_int| ints[0] == *in_int)
                        .unwrap()
                })
                .collect(),
        )
    }
    pub fn is_identity(&self) -> bool {
        self.input_ints == self.output_ints
    }

    pub fn is_valid(&self) -> bool {
        self.check_valid().is_ok()
    }

    pub fn to_einops_string(&self) -> String {
        self.to_string_impl(false, false, false)
    }

    #[pyo3(name = "to_string", signature = (
        size_on_letter = true,
        use_symbolic = true,
        show_size_if_present = true
    ))]
    pub fn to_string_impl(
        &self,
        size_on_letter: bool,
        use_symbolic: bool,
        show_size_if_present: bool,
    ) -> String {
        let ints_needing_sizes = (!show_size_if_present).then(|| self.ints_needing_sizes());
        let ints_needing_sizes = ints_needing_sizes.as_ref();

        let int_to_str = |i: usize| {
            if use_symbolic {
                SymbolicSizeProduct::from(i).to_string()
            } else {
                i.to_string()
            }
        };

        let get_str = |ints: &RInts, ints_in_other: &HashSet<AxisInt>| {
            ints.iter()
                .map(|one_axis_ints| {
                    let one_axis_letters = one_axis_ints
                        .iter()
                        .map(|i| {
                            if ints_in_other.contains(i) || self.int_sizes[*i as usize].is_none() {
                                if size_on_letter
                                    && ints_needing_sizes.map(|x| x.contains(i)).unwrap_or(true)
                                    && self.int_sizes[*i as usize].is_some()
                                {
                                    format!(
                                        "{}:{}",
                                        ALPHABET[*i as usize].to_owned(),
                                        int_to_str(self.int_sizes[*i as usize].unwrap() as usize)
                                    )
                                } else {
                                    ALPHABET[*i as usize].to_owned()
                                }
                            } else {
                                int_to_str(self.int_sizes[*i as usize].unwrap() as usize)
                            }
                        })
                        .collect::<Vec<String>>();
                    if one_axis_letters.len() == 1 {
                        one_axis_letters[0].to_owned()
                    } else {
                        format!("({})", one_axis_letters.join(" "))
                    }
                })
                .collect::<Vec<String>>()
                .join(" ")
        };

        format!(
            "{} -> {}",
            get_str(&self.input_ints, &self.ints_in_out()),
            get_str(&self.output_ints, &self.ints_in_inp())
        )
    }

    /// Only supports a small subsets of rearrange specs. Return None if not supported.
    /// Self's named_axes preceeds over node's named axes.
    /// Does not check for conflicts between different names.
    pub fn to_fancy_einops_string(
        &self,
        node_named_axes: NamedAxes,
        self_named_axes: NamedAxes,
    ) -> Option<String> {
        let input_int_unpacked = self
            .input_ints
            .iter()
            .map(|v| if v.len() == 1 { Some(v[0]) } else { None })
            .collect::<Option<Vec<AxisInt>>>()?;
        let output_int_unpacked = self
            .output_ints
            .iter()
            .map(|v| if v.len() == 1 { Some(v[0]) } else { None })
            .collect::<Option<Vec<AxisInt>>>()?;
        let mut axis_to_name: HashMap<AxisInt, Name> = Default::default();
        for (i, a) in input_int_unpacked.iter().enumerate() {
            if let Some(name) = node_named_axes.get(&(i as u8)) {
                axis_to_name.insert(*a, *name);
            }
        }
        for (i, a) in output_int_unpacked.iter().enumerate() {
            if let Some(name) = self_named_axes.get(&(i as u8)) {
                axis_to_name.insert(*a, *name);
            }
        }
        if axis_to_name.len() > HashSet::from_iter(axis_to_name.values().cloned()).len() {
            return None; // Can't tolerate duplicate names
        }
        let inp_words = input_int_unpacked
            .iter()
            .map(|a| Some((*axis_to_name.get(a)?).into()))
            .collect::<Option<Vec<String>>>()?
            .join(" ");
        let out_words = output_int_unpacked
            .iter()
            .map(|a| Some((*axis_to_name.get(a)?).into()))
            .collect::<Option<Vec<String>>>()?
            .join(" ");
        Some(inp_words + " -> " + &out_words)
    }

    pub fn to_maybe_fancy_einops_string(
        &self,
        node_named_axes: NamedAxes,
        self_named_axes: NamedAxes,
    ) -> String {
        if let Some(fancy_string) = self.to_fancy_einops_string(node_named_axes, self_named_axes) {
            fancy_string
        } else {
            self.to_string()
        }
    }

    #[staticmethod]
    #[pyo3(name = "from_string")]
    pub fn py_from_string(string: &str) -> Result<Self> {
        string.parse()
    }

    pub fn add_batch_dims(&self, batch_rank: usize) -> Self {
        let to_prepend =
            || (self.int_sizes.len()..self.int_sizes.len() + batch_rank).map(|x| tu8v![x as u8]);
        Self::new(
            to_prepend()
                .chain(self.input_ints.iter().cloned())
                .collect(),
            to_prepend()
                .chain(self.output_ints.iter().cloned())
                .collect(),
            self.int_sizes
                .iter()
                .cloned()
                .chain(vec![OpSize::f(None); batch_rank].into_iter())
                .collect(),
        )
        .unwrap()
    }

    /// does this int ever appear in an inner group with >1 element?
    pub fn ints_needing_sizes(&self) -> HashSet<AxisInt> {
        self.input_ints
            .iter()
            .filter(|x| x.len() > 1)
            .flatten()
            .copied()
            .collect()
    }

    /// apply rearrange to tensor
    pub fn apply(&self, tensor: Tensor) -> Result<Tensor> {
        let (string, letter_sizes) = self
            .conform_to_input_shape(&tensor.shape())
            .context("conform to input shape failed in apply")?
            .to_einops_string_and_letter_sizes();
        einops_repeat(&tensor, string, letter_sizes)
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
    pub fn shapes(&self) -> Result<(Shape, Shape)> {
        let used_ints: U8Set = self
            .input_ints
            .iter()
            .chain(&self.output_ints)
            .flatten()
            .cloned()
            .collect();
        if self
            .int_sizes
            .iter()
            .enumerate()
            .any(|(i, x)| x.is_none() && used_ints.contains(i as u8))
        {
            bail!(RearrangeSpecError::HasWildcardSizes { spec: self.clone() });
        }
        Ok((
            self.input_ints
                .iter()
                .map(|x| {
                    x.iter()
                        .map(|y| self.int_sizes[*y as usize].0 as usize)
                        .product()
                })
                .collect(),
            self.output_ints
                .iter()
                .map(|x| {
                    x.iter()
                        .map(|y| self.int_sizes[*y as usize].0 as usize)
                        .product()
                })
                .collect(),
        ))
    }
    /// which output axes are introduced entirely by rearrange, eg `a -> a b`->1, `a -> b (a c)`->0
    pub fn out_broadcast_axes(&self) -> Vec<usize> {
        let ints_in_inp = self.ints_in_inp();
        (0..self.output_ints.len())
            .filter(|i| {
                self.output_ints[*i]
                    .iter()
                    .all(|int| !ints_in_inp.contains(int))
            })
            .collect()
    }

    // TODO: Clean up the code examples once we implement RearrangeSpec.from_string
    /// Simplifies a RearrangeSpec using three rules (described below).
    /// # Examples
    /// 1. if special_case_ones, remove all indices of size 1
    /// ```
    /// # use rr_util::{sv,tu8v, rearrange_spec::{OpSize, RearrangeSpec}};
    /// let spec = RearrangeSpec::new(
    ///     sv![tu8v![0, 1], tu8v![2, 3]],
    ///     sv![tu8v![3, 2, 1, 0]],
    ///     sv![OpSize(3), OpSize(1), OpSize(1), OpSize(1)],
    /// )
    /// .unwrap();
    /// let canonicalized_spec =
    ///     RearrangeSpec::new(sv![tu8v![0], tu8v![]], sv![tu8v![0]], sv![OpSize(3)]).unwrap();
    /// assert_eq!(spec.canonicalize(true), canonicalized_spec);
    /// ```
    /// 2. Merge all sequences of indices that always appear together in the same order inside parentheses
    /// 3. Renumber the int indices sequentially based on when they first appear, inputs first then outputs
    /// ```
    /// # use rr_util::{sv,tu8v, rearrange_spec::{OpSize, RearrangeSpec}};
    /// let spec = RearrangeSpec::new(
    ///     sv![tu8v![0], tu8v![1, 2], tu8v![3, 4, 5]],
    ///     sv![tu8v![1, 2], tu8v![5], tu8v![3, 4], tu8v![0]],
    ///     sv![
    ///         OpSize(2),
    ///         OpSize(2),
    ///         OpSize(2),
    ///         OpSize(2),
    ///         OpSize(2),
    ///         OpSize(2),
    ///     ],
    /// )
    /// .unwrap();
    /// let canonicalized_spec = RearrangeSpec::new(
    ///     sv![tu8v![0], tu8v![1], tu8v![2, 3]],
    ///     sv![tu8v![1], tu8v![3], tu8v![2], tu8v![0]],
    ///     sv![OpSize(2), OpSize(4), OpSize(4), OpSize(2)],
    /// )
    /// .unwrap();
    /// assert_eq!(spec.canonicalize(true), canonicalized_spec);
    /// ```
    #[pyo3(signature=(special_case_ones = true))]
    pub fn canonicalize(&self, special_case_ones: bool) -> Self {
        // Remove all indices of size 1
        let int_sizes: Vec<Option<usize>> = self.int_sizes.iter().map(|x| (*x).into()).collect();
        let mut input_ints = self.input_ints.clone();
        let mut output_ints = self.output_ints.clone();
        if special_case_ones {
            let drop_einints = |ints_list: RInts| {
                ints_list
                    .iter()
                    .map(|ints| {
                        ints.iter()
                            .filter(|i| int_sizes[**i as usize] != Some(1))
                            .copied()
                            .collect()
                    })
                    .collect()
            };
            input_ints = drop_einints(input_ints);
            output_ints = drop_einints(output_ints);
        }

        // Find sequences of indices that always appear together by storing the idx ints that
        // appear before and after this idx int
        #[derive(Copy, Clone, PartialEq, Debug)]
        enum SeqPos {
            First,
            Last,
            Middle(AxisInt),
        }
        use SeqPos::*;
        let mut prev_idx = vec![None; int_sizes.len()];
        let mut next_idx = vec![None; int_sizes.len()];
        for lst in input_ints.iter().chain(output_ints.iter()) {
            if let (Some(&first), Some(&last)) = (lst.first(), lst.last()) {
                prev_idx[first as usize] = Some(First);
                next_idx[last as usize] = Some(Last);
            }

            for (&prev, &curr) in lst.iter().tuple_windows() {
                let prev_i = prev as usize;
                let curr_i = curr as usize;
                if int_sizes[prev_i].is_some() == int_sizes[curr_i].is_some() {
                    if prev_idx[curr_i].is_none() {
                        prev_idx[curr_i] = Some(Middle(prev));
                    }
                    if next_idx[prev_i].is_none() {
                        next_idx[prev_i] = Some(Middle(curr));
                    }
                }
                if prev_idx[curr_i] != Some(Middle(prev)) || next_idx[prev_i] != Some(Middle(curr))
                {
                    prev_idx[curr_i] = Some(First);
                    next_idx[prev_i] = Some(Last);
                }
            }
        }

        // find the first index in a sequence of repeated indices
        // similar to union find - caches results to make future lookups faster
        fn find(i: AxisInt, prev_idx: &mut [Option<SeqPos>]) -> AxisInt {
            let mut pos = i;
            while let Some(Middle(prev_pos)) = prev_idx[pos as usize] {
                pos = prev_pos;
            }
            assert!(prev_idx[pos as usize] == Some(First));

            let first_pos = pos;
            pos = i;
            while let Some(Middle(prev_pos)) = prev_idx[pos as usize] {
                prev_idx[pos as usize] = Some(Middle(first_pos));
                pos = prev_pos;
            }
            first_pos
        }

        let mut map: HashMap<AxisInt, AxisInt> = HashMap::default();
        let mut new_int_sizes: OpShape = sv![];
        let mut already_accounted_shape = vec![false; int_sizes.len()];

        let mut merge_tuples_and_renumber_fn = |ints: RInts| {
            ints.iter()
                .map(|single_dim| {
                    let mut new_dim: TinyVecU8 = tu8v![];
                    for &i in single_dim.iter() {
                        if let Some(&new_i) = map.get(&i) {
                            new_dim.push(new_i);
                        } else if prev_idx[i as usize] == Some(First) {
                            let new_i: u8 = new_int_sizes.len().try_into().unwrap();
                            map.insert(i, new_i);
                            new_dim.push(new_i);
                            new_int_sizes.push(int_sizes[i as usize].into());
                        } else if !already_accounted_shape[i as usize] {
                            if let Some(old_size) = int_sizes[i as usize] {
                                let new_int_sizes_idx =
                                    *map.get(&find(i, &mut prev_idx)).unwrap() as usize;
                                new_int_sizes[new_int_sizes_idx] = Some(
                                    new_int_sizes[new_int_sizes_idx].unwrap() as usize * old_size,
                                )
                                .into();
                            }
                            already_accounted_shape[i as usize] = true;
                        }
                    }
                    new_dim
                })
                .collect()
        };

        let input_ints: RInts = merge_tuples_and_renumber_fn(input_ints);
        let output_ints: RInts = merge_tuples_and_renumber_fn(output_ints);

        RearrangeSpec::new(input_ints, output_ints, new_int_sizes).unwrap()
    }

    fn fill_empty_ints_allow_invalid(&self) -> RearrangeSpec {
        self.fill_empty_ints_impl(true)
            .expect("can't error if we allow invalid")
    }
    pub fn fill_empty_ints(&self) -> Result<RearrangeSpec> {
        self.fill_empty_ints_impl(false)
    }

    fn fill_empty_ints_impl(&self, allow_rust_invalid: bool) -> Result<RearrangeSpec> {
        let mut next_int = self.all_ints().max().map(|i| i + 1).unwrap_or(0);
        let first_int_to_add = next_int;
        let input_ints = self
            .input_ints
            .iter()
            .map(|ints| {
                if ints.is_empty() {
                    next_int += 1;
                    tu8v![next_int - 1]
                } else {
                    ints.clone()
                }
            })
            .collect();
        let int_after_input = next_int;
        let mut output_ints: RInts = self
            .output_ints
            .iter()
            .map(|ints| {
                if ints.is_empty() {
                    next_int += 1;
                    tu8v![next_int - 1]
                } else {
                    ints.clone()
                }
            })
            .collect();
        if int_after_input > first_int_to_add {
            if !output_ints.is_empty() {
                for i in first_int_to_add..int_after_input {
                    output_ints[0].push(i);
                }
            } else if !allow_rust_invalid {
                bail!(RearrangeSpecError::NotConvertable { spec: self.clone() });
            }
        }
        let result = RearrangeSpec {
            input_ints,
            output_ints,
            int_sizes: self
                .int_sizes
                .iter()
                .cloned()
                .chain(vec![OpSize(1); (next_int - first_int_to_add) as usize])
                .collect(),
        };
        if !allow_rust_invalid {
            result.check_valid()?;
        }
        Ok(result)
    }

    #[staticmethod]
    #[pyo3(name = "fuse")]
    pub fn fuse_py(inner: Self, outer: Self) -> Result<Self> {
        RearrangeSpec::fuse(&inner, &outer)
    }
    #[pyo3(name = "conform_to_input_shape")]
    fn conform_to_input_shape_py(&self, shape: Shape) -> Result<RearrangeSpec> {
        self.conform_to_input_shape(&shape)
    }
    #[pyo3(name = "conform_to_output_shape")]
    fn conform_to_output_shape_py(&self, shape: Shape) -> Result<RearrangeSpec> {
        self.conform_to_output_shape(&shape)
    }

    pub fn next_axis(&self) -> u8 {
        // input_ints are included in output_ints, so we don't check them
        self.output_ints
            .iter()
            .flatten()
            .max()
            .map(|x| x + 1)
            .unwrap_or(0)
    }

    #[pyo3(name = "expand_to")]
    fn expand_to_py(&self, shape: Shape) -> Result<RearrangeSpec> {
        self.expand_to(&shape)
    }
}

pub enum ExpandToSpecOrShape {
    Spec(RearrangeSpec),
    SetShape(OpShape),
}

impl RearrangeSpec {
    pub fn ints_in_inp_it(&self) -> impl Iterator<Item = AxisInt> + '_ {
        self.input_ints.iter().flatten().copied()
    }
    pub fn ints_in_out_it(&self) -> impl Iterator<Item = AxisInt> + '_ {
        self.output_ints.iter().flatten().copied()
    }
    /// All integers must occur in the output, so this is valid
    pub fn all_ints(&self) -> impl Iterator<Item = AxisInt> + '_ {
        self.ints_in_out_it()
    }

    pub fn flatten_usize(rank: usize) -> Result<Self> {
        Self::check_rank(rank)?;
        Ok(Self::flatten(rank as u8))
    }

    pub fn ident_usize(rank: usize) -> Result<Self> {
        Self::check_rank(rank)?;
        Ok(Self::ident(rank as u8))
    }

    pub fn new_canon(input_ints: RInts, output_ints: RInts, int_sizes: OpShape) -> Self {
        RearrangeSpec::new(input_ints, output_ints, int_sizes)
            .unwrap()
            .canonicalize(true)
    }

    pub fn expand_to(&self, shape: &Shape) -> Result<RearrangeSpec> {
        match self.expand_to_spec_or_shape(shape, true)? {
            ExpandToSpecOrShape::Spec(spec) => Ok(spec),
            ExpandToSpecOrShape::SetShape(_) => unreachable!(),
        }
    }

    pub fn expand_to_spec_or_shape(
        &self,
        shape: &Shape,
        error_on_ambiguous: bool,
    ) -> Result<ExpandToSpecOrShape> {
        if shape.len() != self.input_ints.len() {
            bail!(RearrangeSpecError::NotConformable {
                spec: self.clone(),
                shape: shape.clone(),
            });
        }

        let mut out_shape: OpShape = sv![OpSize::NONE; shape.len()];
        let mut int_sizes = self.int_sizes.clone();
        for (dim, (inner_ints, &expand_size, out_size)) in
            izip!(&self.input_ints, shape, &mut out_shape).enumerate()
        {
            if inner_ints
                .iter()
                .any(|x| self.int_sizes[*x as usize].is_none())
            {
                // none size, so this is fine
                continue;
            }
            let current_size = inner_ints
                .iter()
                .map(|x| self.int_sizes[*x as usize].unwrap())
                .product::<usize>();
            if current_size == expand_size {
                continue;
            }
            if inner_ints.len() == 1 {
                // change to wild card if single non-matching int
                int_sizes[inner_ints[0] as usize] = OpSize::NONE;
                continue;
            }
            let inner_int_sizes: Vec<_> = inner_ints
                .iter()
                .map(|x| self.int_sizes[*x as usize].unwrap())
                .collect();
            let divides_size: Vec<_> = inner_int_sizes
                .iter()
                .map(|&x| x != 0 && expand_size % x == 0)
                .collect();
            let num_failing_to_divide = divides_size
                .iter()
                .map(|divides| (!divides) as usize)
                .sum::<usize>();
            if num_failing_to_divide != 1 {
                if error_on_ambiguous {
                    bail!(RearrangeSpecError::AmbiguousExpand {
                        expand_size,
                        current_size,
                        inner_ints: inner_ints.clone(),
                        inner_int_sizes: inner_int_sizes.clone(),
                        divides_size,
                        num_failing_to_divide,
                        dim,
                        shape: shape.clone(),
                        spec: self.clone()
                    })
                }
                *out_size = Some(current_size).into();
                continue;
            }

            // if we have exactly 1 symbolic idx, set this size to None
            let non_dividing_idx = divides_size
                .iter()
                .enumerate()
                .find_map(|(i, b)| (!b).then_some(i))
                .unwrap();
            int_sizes[inner_ints[non_dividing_idx] as usize] = OpSize::NONE;
        }
        if out_shape.iter().any(|x| x.is_some()) {
            return Ok(ExpandToSpecOrShape::SetShape(out_shape));
        }

        Ok(ExpandToSpecOrShape::Spec(
            Self::new(self.input_ints.clone(), self.output_ints.clone(), int_sizes).unwrap(),
        ))
    }

    pub fn conform_to_input_shape(&self, shape: &Shape) -> Result<RearrangeSpec> {
        if shape.len() != self.input_ints.len() {
            bail!(RearrangeSpecError::NotConformable {
                spec: self.clone(),
                shape: shape.clone(),
            });
        }
        let mut int_sizes = self.int_sizes.clone();
        for (ints, &l) in zip(&self.input_ints, shape) {
            let none_indices: Vec<usize> = ints
                .iter()
                .enumerate()
                .filter(|(_i, x)| self.int_sizes[**x as usize].is_none())
                .map(|(i, _x)| i)
                .collect();
            let non_wildcard_size: usize = ints
                .iter()
                .map(|x| self.int_sizes[*x as usize].unwrap_or(1))
                .product();
            let has_wildcard = none_indices.len() > 0;
            if (!has_wildcard && l != non_wildcard_size)
                || (has_wildcard && l % non_wildcard_size != 0)
            {
                bail!(RearrangeSpecError::NotConformable {
                    spec: self.clone(),
                    shape: shape.clone(),
                });
            }

            if has_wildcard {
                if none_indices.len() > 1 {
                    bail!(RearrangeSpecError::TooManyWildcardSizes { spec: self.clone() });
                } else if none_indices.len() == 1 {
                    int_sizes[ints[none_indices[0]] as usize] =
                        OpSize((l / non_wildcard_size) as u64);
                }
            }
        }
        Ok(Self::new(self.input_ints.clone(), self.output_ints.clone(), int_sizes).unwrap())
    }

    pub fn conform_to_output_shape(&self, shape: &Shape) -> Result<RearrangeSpec> {
        if shape.len() != self.output_ints.len() {
            bail!(RearrangeSpecError::NotConformable {
                spec: self.clone(),
                shape: shape.clone(),
            });
        }
        let mut int_sizes = self.int_sizes.clone();
        for (ints, &l) in zip(&self.output_ints, shape) {
            let none_indices: Vec<usize> = ints
                .iter()
                .enumerate()
                .filter(|(_i, x)| self.int_sizes[**x as usize].is_none())
                .map(|(i, _x)| i)
                .collect();
            let non_wildcard_size: usize = ints
                .iter()
                .map(|x| self.int_sizes[*x as usize].unwrap_or(1))
                .product();
            let has_wildcard = none_indices.len() > 0;
            if (!has_wildcard && l != non_wildcard_size)
                || (has_wildcard && l % non_wildcard_size != 0)
            {
                bail!(RearrangeSpecError::NotConformable {
                    spec: self.clone(),
                    shape: shape.clone(),
                });
            }

            if has_wildcard {
                if none_indices.len() > 1 {
                    bail!(RearrangeSpecError::TooManyWildcardSizes { spec: self.clone() });
                } else if none_indices.len() == 1 {
                    int_sizes[ints[none_indices[0]] as usize] =
                        OpSize((l / non_wildcard_size) as u64);
                }
            }
        }
        Ok(Self::new(self.input_ints.clone(), self.output_ints.clone(), int_sizes).unwrap())
    }

    pub fn unflatten_axis_usize(ndim: usize, axis: usize, shape: Shape) -> Result<Self> {
        Self::unflatten_axis(ndim, axis as i64, shape)
    }

    pub fn check_valid(&self) -> Result<()> {
        // check each int appears once in input and output
        let ints: Vec<_> = self.ints_in_inp_it().collect();
        if !is_unique(&ints) {
            bail!(RearrangeSpecError::IntsNotUnique {
                is_input: true,
                ints: ints,
            })
        }
        let ints: Vec<_> = self.ints_in_out_it().collect();
        if !is_unique(&ints) {
            bail!(RearrangeSpecError::IntsNotUnique {
                is_input: false,
                ints: ints,
            })
        }

        let ints_in_out = self.ints_in_out();
        let ints_in_inp = self.ints_in_inp();

        // check each input is in output
        if ints_in_inp.difference(&ints_in_out).count() > 0 {
            bail!(RearrangeSpecError::InpNotInOut {
                difference: ints_in_inp.difference(&ints_in_out).cloned().collect(),
                ints_in_inp,
                ints_in_out,
            })
        }

        // check int_sizes is long enough
        if ints_in_out
            .iter()
            .any(|&i| (i as usize) >= self.int_sizes.len())
        {
            bail!(RearrangeSpecError::IntNotInSizes {
                max_int: ints_in_out.iter().cloned().max().unwrap(),
                len_sizes: self.int_sizes.len(),
            })
        }

        let only_in_output: Vec<AxisInt> = ints_in_out.difference(&ints_in_inp).copied().collect();

        // check each int only in output has size
        if only_in_output
            .iter()
            .any(|x| self.int_sizes[*x as usize].is_none())
        {
            bail!(RearrangeSpecError::IntOnlyInOutputWithoutSize {
                corresponding_sizes: only_in_output
                    .iter()
                    .map(|x| self.int_sizes[*x as usize])
                    .collect(),
                only_in_output,
                int_sizes: self.int_sizes.clone(),
            })
        }

        // check only one wildcard per input axis
        if !(self.input_ints.iter().all(|x| {
            x.iter()
                .filter(|y| self.int_sizes[**y as usize].is_none())
                .count()
                <= 1
        })) {
            bail!(RearrangeSpecError::InputAxisHasMultipleWildCards {
                ints: self.input_ints.clone(),
                int_sizes: self.int_sizes.clone(),
            })
        }

        Ok(())
    }

    pub fn filter_out_axes(&self, out_axes_to_skip: &HashSet<usize>) -> Result<RearrangeSpec> {
        RearrangeSpec::new(
            self.input_ints.clone(),
            filter_out_idx(&self.output_ints, out_axes_to_skip)
                .into_iter()
                .collect(),
            self.int_sizes.clone(),
        )
    }
    pub fn filter_all_tuples(
        &self,
        tuples_to_skip: &HashSet<Box<[AxisInt]>>,
    ) -> Result<RearrangeSpec> {
        RearrangeSpec::new(
            self.input_ints
                .iter()
                .filter(|x| !tuples_to_skip.contains(&x[..]))
                .cloned()
                .collect(),
            self.output_ints
                .iter()
                .filter(|x| !tuples_to_skip.contains(&x[..]))
                .cloned()
                .collect(),
            self.int_sizes.clone(),
        )
    }

    /// input_ints and output_ints cant contain repeats!
    pub fn unremove_axes(removed_axes: &HashSet<usize>, output_shape: &Shape) -> RearrangeSpec {
        RearrangeSpec::new(
            (0..output_shape.len())
                .filter(|i| !removed_axes.contains(i))
                .map(|i| tu8v![i as u8])
                .collect(),
            (0..output_shape.len()).map(|i| tu8v![i as u8]).collect(),
            shape_to_op_shape(output_shape),
        )
        .unwrap()
    }

    pub fn expand(
        input_ints: &EinsumAxes,
        output_ints: &EinsumAxes,
        int_sizes: &HashMap<u8, usize>,
    ) -> Result<RearrangeSpec> {
        let int_sizes_usize: HashMap<usize, usize> =
            int_sizes.iter().map(|(k, v)| (*k as usize, *v)).collect();
        RearrangeSpec::new(
            input_ints.iter().map(|i| tu8v![*i]).collect(),
            output_ints.iter().map(|i| tu8v![*i]).collect(),
            shape_to_op_shape(&dict_to_list(&int_sizes_usize, None).into_iter().collect()),
        )
    }

    pub fn compute_hash(&self) -> HashBytes {
        let mut hasher = blake3::Hasher::new();
        for axis in self.input_ints.iter() {
            hasher.update(&uuid!("8718e8a1-bf0a-46fe-be66-be8894bc41bd").into_bytes()); // delimit with uuid
            hasher.update(axis);
        }
        for axis in self.output_ints.iter() {
            hasher.update(&uuid!("ebd66768-bb20-43a4-8f5a-fd51e76a0333").into_bytes()); // delimit with uuid
            hasher.update(axis);
        }
        hasher.update(&uuid!("ebd66768-bb20-43a4-8f5a-fd51e76a0333").into_bytes()); // delimit with uuid
        for size in self.int_sizes.iter() {
            hasher.update(&uuid!("ebd66768-bb20-43a4-8f5a-fd51e76a0333").into_bytes()); // delimit with uuid
            hasher.update(&size.0.to_le_bytes());
        }
        hasher.finalize().into()
    }

    fn get_unwrapped_sizes(&self) -> Vec<usize> {
        self.int_sizes.iter().map(|x| x.unwrap()).collect()
    }

    // TODO: Clean up the code examples once we implement RearrangeSpec.from_string
    /// Composes two rearrange specs
    ///
    /// # Examples
    /// ```
    /// # use rr_util::{sv,tu8v, rearrange_spec::{OpSize, RearrangeSpec}};
    /// let inner = RearrangeSpec::new(
    ///     sv![tu8v![0], tu8v![1]],
    ///     sv![tu8v![0, 1]],
    ///     sv![OpSize(2), OpSize(3)],
    /// )
    /// .unwrap();
    /// let outer = RearrangeSpec::new(
    ///     sv![tu8v![0]],
    ///     sv![tu8v![0], tu8v![1]],
    ///     sv![OpSize(6), OpSize(4)],
    /// )
    /// .unwrap();
    /// let fused = RearrangeSpec::new(
    ///     sv![tu8v![0], tu8v![1]],
    ///     sv![tu8v![0, 1], tu8v![2]],
    ///     sv![OpSize(2), OpSize(3), OpSize(4)],
    /// )
    /// .unwrap();
    /// assert_eq!(RearrangeSpec::fuse(&inner, &outer).unwrap(), fused);
    /// ```
    pub fn fuse(inner: &Self, outer: &Self) -> Result<Self> {
        let mut inner_sizes: Vec<_> = inner.get_unwrapped_sizes();
        let mut outer_sizes: Vec<_> = outer.get_unwrapped_sizes();
        let mut inner_replaces: HashMap<AxisInt, Vec<AxisInt>> = HashMap::default();
        let mut outer_replaces: HashMap<AxisInt, Vec<AxisInt>> = HashMap::default();
        let mut in_to_out: HashMap<AxisInt, AxisInt> = HashMap::default();

        // if possible, split up inner.outer_ints and outer.inner_ints until they have the same size for each subdimension (if not possible, raise error)
        for (outer_ints, inner_ints) in izip!(&outer.input_ints, &inner.output_ints) {
            let mut new_inner_ints = inner_ints.clone();
            let mut new_outer_ints = outer_ints.clone();
            let mut inner_iter = inner_ints.iter().peekable();
            let mut outer_iter = outer_ints.iter().peekable();
            while let Some(&&in_i) = inner_iter.peek() && let Some(&&out_i) = outer_iter.peek() {
                let in_s = inner_sizes[in_i as usize];
                let out_s = outer_sizes[out_i as usize];
                if in_s == out_s {
                    inner_iter.next();
                    outer_iter.next();
                    in_to_out.insert(in_i, out_i);
                    vec_map_insert(&mut inner_replaces, in_i, in_i);
                    vec_map_insert(&mut outer_replaces, out_i, out_i);
                } else {
                    let (
                        old_i,
                        new_i,
                        small_i,
                        small_s,
                        small_iter,
                        small_replaces,
                        big_s,
                        big_ints,
                        big_sizes,
                        big_replaces,
                    ) = if in_s > out_s {
                        (
                            in_i,
                            inner_sizes.len().try_into().unwrap(),
                            out_i,
                            out_s,
                            &mut outer_iter,
                            &mut outer_replaces,
                            in_s,
                            &mut new_inner_ints,
                            &mut inner_sizes,
                            &mut inner_replaces,
                        )
                    } else {
                        (
                            out_i,
                            outer_sizes.len().try_into().unwrap(),
                            in_i,
                            in_s,
                            &mut inner_iter,
                            &mut inner_replaces,
                            out_s,
                            &mut new_outer_ints,
                            &mut outer_sizes,
                            &mut outer_replaces,
                        )
                    };

                    if big_s % small_s != 0 {
                        bail!(RearrangeSpecError::NotFusable { inner:inner.clone(), outer:outer.clone() });
                    }
                    if in_s > out_s {
                        in_to_out.insert(new_i, out_i);
                    } else {
                        in_to_out.insert(in_i, new_i);
                    }
                    small_iter.next();
                    big_ints.push(new_i);
                    big_sizes[old_i as usize] /= small_s;
                    big_sizes.push(small_s);
                    vec_map_insert(small_replaces, small_i, small_i);
                    vec_map_insert(big_replaces, old_i, new_i);
                }
            }

            if inner_iter.any(|&i| inner_sizes[i as usize] != 1)
                || outer_iter.any(|&i| outer_sizes[i as usize] != 1)
            {
                bail!(RearrangeSpecError::NotFusable {
                    inner: inner.clone(),
                    outer: outer.clone(),
                });
            }
        }

        // update inner.inner_ints and outer.inner_ints using the splits we generated above
        fn expand_dims(
            all_dim_ints: &[RInnerInts],
            replaces: &mut HashMap<AxisInt, Vec<AxisInt>>,
        ) -> RInts {
            all_dim_ints
                .iter()
                .map(|dim_ints| {
                    dim_ints
                        .iter()
                        .flat_map(|dim_int| {
                            replaces.remove(dim_int).unwrap_or_else(|| vec![*dim_int])
                        })
                        .collect()
                })
                .collect()
        }
        let input_ints: RInts = expand_dims(&inner.input_ints, &mut inner_replaces);
        let output_ints: RInts = expand_dims(&outer.output_ints, &mut outer_replaces);

        // change inputs to use output labels and drop input terms that don't have a mapping
        // (this should only happen if those indices have size 1)
        let input_ints: RInts = input_ints
            .iter()
            .map(|dim_ints| {
                dim_ints
                    .iter()
                    .filter_map(|dim_int| in_to_out.get(dim_int).copied())
                    .collect()
            })
            .collect();

        Ok(Self::new(
            input_ints,
            output_ints,
            outer_sizes
                .iter()
                .map(|&x| OpSize::from(Some(x as usize)))
                .collect(),
        )
        .unwrap()
        .canonicalize(true))
    }

    pub fn expand_at_axes(orig_ndim: usize, axes: Vec<usize>, counts: Vec<usize>) -> Result<Self> {
        Self::expand_at_axes_impl(
            orig_ndim,
            axes.into_iter().map(|x| x as i64).collect(),
            Some(counts),
        )
    }

    pub fn unsqueeze(orig_ndim: usize, axes: Vec<usize>) -> Result<Self> {
        Self::expand_at_axes_impl(
            orig_ndim,
            axes.into_iter().map(|x| x as i64).collect(),
            None,
        )
    }

    fn expand_at_axes_impl(
        orig_ndim: usize,
        axes: Vec<i64>,
        counts: Option<Vec<usize>>,
    ) -> Result<Self> {
        let rank = orig_ndim + axes.len();
        Self::check_rank(rank)?;
        let counts = counts.unwrap_or_else(|| vec![1; axes.len()]);
        if counts.len() != axes.len() {
            bail!(RearrangeSpecError::AxesAndCountsDontMatch {
                counts_len: counts.len(),
                axes_len: axes.len()
            });
        }

        let (sorted_axes, sorted_counts): (Vec<_>, Vec<_>) =
            check_canon_idxs(orig_ndim + axes.len(), &axes)
                .context("axis out of bounds even after expanding number of axes")?
                .into_iter()
                .zip(counts)
                .sorted()
                .unzip();

        let unique_count = sorted_axes.iter().unique().count();
        if unique_count != sorted_axes.len() {
            bail!(RearrangeSpecError::RepeatedAxes {
                unique_count,
                actual_count: sorted_axes.len(),
                axes: sorted_axes,
            })
        }

        let input_ints: Vec<_> = (0..orig_ndim).collect();
        let mut next_sorted_axis = 0;
        let mut next_orig_axis = 0;

        let mut output_ints = Vec::new();

        while next_sorted_axis < sorted_axes.len() || next_orig_axis < input_ints.len() {
            let current_axis = next_sorted_axis + next_orig_axis;
            if next_sorted_axis < sorted_axes.len() && current_axis == sorted_axes[next_sorted_axis]
            {
                output_ints.push(next_sorted_axis + orig_ndim);
                next_sorted_axis += 1
            } else {
                output_ints.push(next_orig_axis);
                next_orig_axis += 1
            }
        }

        assert_eq!(next_orig_axis, orig_ndim);
        assert_eq!(next_sorted_axis, sorted_axes.len());

        let out = Self::new(
            input_ints.into_iter().map(|i| tu8v![i as u8]).collect(),
            output_ints.into_iter().map(|i| tu8v![i as u8]).collect(),
            std::iter::repeat(OpSize::NONE)
                .take(orig_ndim)
                .map(Ok)
                .chain(sorted_counts.iter().map(|x| Ok(OpSize((*x).try_into()?))))
                .collect::<Result<_>>()?,
        )
        .unwrap();

        Ok(out)
    }

    /// assumes has size when 'needed' (needed if int ever appears in an inner group with >1 element)
    /// Can be achieved with conform.
    pub fn letter_sizes(&self) -> Vec<(String, usize)> {
        // because to_einops_string uses numerals on all added dims, we only need letter sizes for splits

        self.ints_needing_sizes()
            .into_iter()
            .map(|x| {
                let size_here = self.int_sizes[x as usize].t().unwrap();
                (ALPHABET[x as usize].to_owned(), size_here)
            })
            .collect()
    }

    /// assumes has size when 'needed' (needed if int ever appears in an inner group with >1 element)
    /// Can be achieved with conform.
    pub fn to_einops_string_and_letter_sizes(&self) -> (String, Vec<(String, usize)>) {
        (self.to_einops_string(), self.letter_sizes())
    }
}

#[test]
fn test_expand_to() {
    let spec = RearrangeSpec::from_str("a->a").unwrap();
    assert_eq!(spec.expand_to(&[8].into_iter().collect()).unwrap(), spec);
    assert_eq!(spec.expand_to(&[2].into_iter().collect()).unwrap(), spec);

    let spec_sized = RearrangeSpec::from_str("a:5->a").unwrap();
    assert_eq!(
        spec_sized.expand_to(&[5].into_iter().collect()).unwrap(),
        spec_sized
    );
    assert_eq!(
        spec_sized.expand_to(&[2].into_iter().collect()).unwrap(),
        spec
    );
    assert_eq!(
        spec_sized.expand_to(&[6].into_iter().collect()).unwrap(),
        spec
    );

    let spec = RearrangeSpec::from_str("(a:5) b (k:2 y:7)-> (k a b) y").unwrap();
    assert_eq!(
        spec.expand_to(&[5, 3, 14].into_iter().collect()).unwrap(),
        spec
    );
    assert_eq!(
        spec.expand_to(&[2, 3, 14].into_iter().collect()).unwrap(),
        RearrangeSpec::from_str("a b (k:2 y:7)-> (k a b) y").unwrap()
    );
    assert_eq!(
        spec.expand_to(&[7, 3, 14].into_iter().collect()).unwrap(),
        RearrangeSpec::from_str("a b (k:2 y:7)-> (k a b) y").unwrap()
    );
    assert_eq!(
        spec.expand_to(&[7, 3, 21].into_iter().collect()).unwrap(),
        RearrangeSpec::from_str("a b (k y:7)-> (k a b) y").unwrap()
    );
    assert_eq!(
        spec.expand_to(&[7, 3, 18].into_iter().collect()).unwrap(),
        RearrangeSpec::from_str("a b (k:2 y)-> (k a b) y").unwrap()
    );
    assert!(spec.expand_to(&[5, 3, 5].into_iter().collect()).is_err());
    assert!(spec
        .expand_to(&[2, 5, 3, 14].into_iter().collect())
        .is_err());
    assert!(spec.expand_to(&[5, 3, 28].into_iter().collect()).is_err());
}

fn fancy_size_format<'a, T: IntoIterator<Item = &'a usize>>(x: T) -> String {
    format!(
        "[{}]",
        x.into_iter()
            .map(|s| SymbolicSizeProduct::from(*s).to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
}

#[apply(python_error_exception)]
#[base_error_name(RearrangeSpec)]
#[base_exception(PyValueError)]
#[derive(Error, Debug, Clone)]
pub enum RearrangeSpecError {
    #[error("RearrangeSpec not conformable, {spec:?} {shape:?} ({e_name})")]
    NotConformable { spec: RearrangeSpec, shape: Shape },

    #[error("RearrangeSpec has wildcard sizes, {spec:?} ({e_name})")]
    HasWildcardSizes { spec: RearrangeSpec },

    #[error("RearrangeSpec cannot be converted to no-(), no-squeeze format, {spec:?} ({e_name})")]
    NotConvertable { spec: RearrangeSpec },

    #[error("RearrangeSpec has too many wildcard sizes, {spec:?} ({e_name})")]
    TooManyWildcardSizes { spec: RearrangeSpec },

    #[error("counts.len()={counts_len} != axes.len()={axes_len} ({e_name})")]
    AxesAndCountsDontMatch { axes_len: usize, counts_len: usize },

    #[error("Rearranges can't be fused, {inner:?}\n{outer:?} ({e_name})")]
    NotFusable {
        inner: RearrangeSpec,
        outer: RearrangeSpec,
    },

    #[error(
        "expanding to expand_size={fancy_size} (!= current_size={fancy_current_size}) {} {}\n{s}\n{c}\n({e_name})",
        "was ambiguous for",
        format!(
            "inner_ints={:?}, inner_int_sizes={}, divides_size={:?}, num_failing_to_divide={}!=1",
            inner_ints,
            fancy_size_format(inner_int_sizes.iter()),
            divides_size,
            num_failing_to_divide
        ),
        fancy_size = SymbolicSizeProduct::from(*expand_size).to_string(),
        fancy_current_size =  SymbolicSizeProduct::from(*current_size).to_string(),
        s = concat!(
            "ambiguous expansion happens if:\n",
            "  - there aren't any wildcards\n",
            "  - there are multiple ints\n",
            "  - and it isn't the case that exactly one of these ints sizes",
            " fails to divide the input size ",
            "(in which case, we can unambiguously replace that size with None)"
        ),
        c = format!(
            "dim={}, shape={:?}, spec={}",
            dim,
            fancy_size_format(shape.iter()),
            spec
        )
    )]
    AmbiguousExpand {
        expand_size: usize,
        current_size: usize,
        inner_ints: RInnerInts,
        inner_int_sizes: Vec<usize>,
        divides_size: Vec<bool>,
        num_failing_to_divide: usize,
        dim: usize,
        shape: Shape,
        spec: RearrangeSpec,
    },

    #[error(
        "Repeated axes={axes:?} unique_count={unique_count} actual_count={actual_count} ({e_name})"
    )]
    RepeatedAxes {
        axes: Vec<usize>,
        unique_count: usize,
        actual_count: usize,
    },

    #[error("ints not unique is_input={is_input}, ints={ints:?} ({e_name})")]
    IntsNotUnique { is_input: bool, ints: Vec<AxisInt> },

    #[error(
        "inp int not in output difference={difference:?}, ints_in_inp={ints_in_inp:?}, ints_in_out={ints_in_out:?} ({e_name})"
    )]
    InpNotInOut {
        difference: HashSet<AxisInt>,
        ints_in_inp: HashSet<AxisInt>,
        ints_in_out: HashSet<AxisInt>,
    },

    #[error("int not in sizes vec, len_sizes={len_sizes}, max_int={max_int} ({e_name})")]
    IntNotInSizes { max_int: u8, len_sizes: usize },

    #[error("int which is only in output doesn't have size, only_in_output={only_in_output:?}, corresponding_sizes={corresponding_sizes:?}, int_sizes={int_sizes:?} ({e_name})")]
    IntOnlyInOutputWithoutSize {
        only_in_output: Vec<AxisInt>,
        corresponding_sizes: Vec<OpSize>,
        int_sizes: OpShape,
    },

    // TODO: improve error message to highlight failure
    #[error("ints={ints:?} int_sizes={int_sizes:?} ({e_name})")]
    InputAxisHasMultipleWildCards { ints: RInts, int_sizes: OpShape },

    // TODO: improve error message to highlight failure
    #[error("len_shape={len_shape} > 255 ({e_name})")]
    LenShapeTooLarge { len_shape: usize },

    #[error("axes_to_combine={axes_to_combine:?} for range 0..{n} ({e_name})")]
    AxesToCombineNotSubset { axes_to_combine: RInnerInts, n: u8 },

    #[error("Tried to unflatten with ndim=0 ({e_name})")]
    CantUnflattenScalar {},
}

#[apply(python_error_exception)]
#[base_error_name(Perm)]
#[base_exception(PyValueError)]
#[derive(Error, Debug, Clone)]
pub enum PermError {
    // TODO: improve errors
    #[error("ints={ints:?} ({e_name})")]
    IntsNotUnique { ints: Vec<usize> },

    #[error("ints={ints:?} count={count} ({e_name})")]
    NotContiguousInts { ints: Vec<usize>, count: usize },
}

#[apply(python_error_exception)]
#[base_error_name(RearrangeParse)]
#[base_exception(PyParseError)]
#[derive(Error, Debug, Clone)]
pub enum RearrangeParseError {
    // TODO: improve errors
    #[error("string={string:?} ({e_name})")]
    ArrowIssue { string: String },
    #[error("string={string:?} ({e_name})")]
    FailedToMatchRegex { string: String },
    #[error("string={string:?} ({e_name})")]
    TooManyAxes { string: String },
}
