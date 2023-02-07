use std::{iter::zip, str::FromStr};

use anyhow::{bail, Context, Result};
use itertools::Itertools;
use macro_rules_attribute::apply;
use once_cell::sync::Lazy;
use pyo3::{exceptions::PyValueError, prelude::*};
use regex::Regex;
use rr_util::{
    lru_cache::TensorCacheRrfs,
    name::Name,
    python_error_exception,
    symbolic_size::{SymbolicSizeProduct, SIZE_PROD_MATCH},
    tensor_db::ensure_all_tensors_local,
    tensor_util::{
        parse_numeric, ParseError, PyParseError, Shape, TensorIndex, TorchDeviceDtypeOp,
    },
    timed,
    util::{counts_g_1, flip_op_result, is_unique, NamedAxes},
};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use thiserror::Error;
use uuid::Uuid;

use crate::{
    named_axes::set_named_axes, print::TerseBool, Add, Array, CircuitNode, CircuitRc, CircuitType,
    Concat, ConstructError, Cumulant, DiscreteVar, Einsum, GeneralFunction, Index, Module,
    ModuleArgSpec, ModuleSpec, Rearrange, Scalar, Scatter, SetSymbolicShape, StoredCumulantVar,
    Symbol, Tag,
};
const FANCY_PREFIX: &str = "fancy:";

#[pyclass]
#[derive(Clone, Debug)]
pub struct Parser {
    #[pyo3(get, set)]
    pub reference_circuits: HashMap<Name, CircuitRc>,
    #[pyo3(get, set)]
    pub tensors_as_random: bool,
    #[pyo3(get, set)]
    pub tensors_as_random_device_dtype: TorchDeviceDtypeOp,
    #[pyo3(get, set)]
    pub allow_hash_with_random: bool,
    #[pyo3(get, set)]
    pub on_repeat_check_info_same: bool,
    #[pyo3(get, set)]
    pub module_check_all_inputs_used: bool,
    #[pyo3(get, set)]
    pub module_check_unique_arg_names: bool,
}

impl Default for Parser {
    fn default() -> Self {
        Self {
            reference_circuits: Default::default(),
            tensors_as_random: false,
            tensors_as_random_device_dtype: TorchDeviceDtypeOp::TENSOR_DEFAULT,
            allow_hash_with_random: false,
            on_repeat_check_info_same: true,
            module_check_all_inputs_used: true,
            module_check_unique_arg_names: false,
        }
    }
}

pub(super) fn get_reference_circuits(
    reference_circuits: HashMap<Name, CircuitRc>,
    reference_circuits_by_name: Vec<CircuitRc>,
) -> Result<HashMap<Name, CircuitRc>> {
    let ref_circ_names = reference_circuits_by_name
        .into_iter()
        .map(|circ| {
            Ok((
                circ.info()
                    .name
                    .ok_or_else(|| ReferenceCircError::ByNameHasNoneName {
                        circuit: circ.clone(),
                    })?,
                circ,
            ))
        })
        .collect::<Result<Vec<_>>>()?;

    let all_idents: Vec<_> = ref_circ_names
        .iter()
        .map(|(name, _)| name)
        .chain(reference_circuits.keys().map(|x| x))
        .collect();
    if !is_unique(&all_idents) {
        bail!(ReferenceCircError::DuplicateIdentifier {
            dup_idents: counts_g_1(all_idents.into_iter().cloned())
        })
    }

    Ok(ref_circ_names
        .into_iter()
        .chain(reference_circuits)
        .collect())
}

#[pymethods]
impl Parser {
    #[pyo3(signature=(
        reference_circuits = Parser::default().reference_circuits,
        reference_circuits_by_name = vec![],
        tensors_as_random = Parser::default().tensors_as_random,
        tensors_as_random_device_dtype = Parser::default().tensors_as_random_device_dtype,
        allow_hash_with_random = Parser::default().allow_hash_with_random,
        on_repeat_check_info_same = Parser::default().on_repeat_check_info_same,
        module_check_all_inputs_used = Parser::default().module_check_all_inputs_used,
        module_check_unique_arg_names = Parser::default().module_check_unique_arg_names
    ))]
    #[new]
    pub fn new(
        reference_circuits: HashMap<Name, CircuitRc>,
        reference_circuits_by_name: Vec<CircuitRc>,
        tensors_as_random: bool,
        tensors_as_random_device_dtype: TorchDeviceDtypeOp,
        allow_hash_with_random: bool,
        on_repeat_check_info_same: bool,
        module_check_all_inputs_used: bool,
        module_check_unique_arg_names: bool,
    ) -> Result<Self> {
        Ok(Self {
            reference_circuits: get_reference_circuits(
                reference_circuits,
                reference_circuits_by_name,
            )?,
            tensors_as_random,
            tensors_as_random_device_dtype,
            allow_hash_with_random,
            on_repeat_check_info_same,
            module_check_all_inputs_used,
            module_check_unique_arg_names,
        })
    }

    #[pyo3(name = "parse_circuit")]
    pub fn parse_circuit_py(
        &self,
        string: &str,
        mut tensor_cache: Option<TensorCacheRrfs>,
    ) -> Result<CircuitRc> {
        self.parse_circuit(string, &mut tensor_cache)
    }

    pub fn __call__(
        &self,
        string: &str,
        tensor_cache: Option<TensorCacheRrfs>,
    ) -> Result<CircuitRc> {
        self.parse_circuit_py(string, tensor_cache)
    }

    #[pyo3(name = "parse_circuits")]
    pub fn parse_circuits_py(
        &self,
        string: &str,
        mut tensor_cache: Option<TensorCacheRrfs>,
    ) -> Result<Vec<CircuitRc>> {
        self.parse_circuits(string, &mut tensor_cache)
    }
}

#[derive(Debug, Clone, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub enum CircuitIdent {
    Num(usize),
    Str(String),
}

impl IntoPy<PyObject> for CircuitIdent {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            Num(x) => x.into_py(py),
            Str(x) => x.into_py(py),
        }
    }
}

use CircuitIdent::*;

impl CircuitIdent {
    fn as_str(&self) -> Option<&str> {
        match self {
            Num(_) => None,
            Str(s) => Some(s),
        }
    }
}

// make separate struct that can deeply mutate, can't use immutable Circuit bc see children later
#[derive(Debug, Clone)]
struct PartialCirc {
    pub variant: CircuitType,
    pub extra: String,
    pub shape: Option<Shape>,
    pub name: Option<Name>,
    pub children: Vec<CircuitIdent>,
    pub named_axes: Option<NamedAxes>,
    pub module_arg_specs: Vec<ModuleArgSpec>,
    pub autoname_disabled: bool,
}

impl Parser {
    pub fn parse_circuit(
        &self,
        string: &str,
        tensor_cache: &mut Option<TensorCacheRrfs>,
    ) -> Result<CircuitRc> {
        let circuits = self.parse_circuits(string, tensor_cache)?;
        if circuits.len() != 1 {
            bail!(ParseCircuitError::ExpectedOneCircuitGotMultiple {
                actual_num_circuits: circuits.len()
            })
        }
        Ok(circuits.into_iter().next().unwrap())
    }

    pub fn parse_circuits(
        &self,
        string: &str,
        tensor_cache: &mut Option<TensorCacheRrfs>,
    ) -> Result<Vec<CircuitRc>> {
        self.parse_circuits_impl(
            string,
            &mut Default::default(),
            &mut Default::default(),
            tensor_cache,
        )
    }

    fn parse_circuits_impl(
        &self,
        string: &str,
        partial_circuits: &mut HashMap<CircuitIdent, PartialCirc>,
        context: &mut HashMap<CircuitIdent, CircuitRc>,
        tensor_cache: &mut Option<TensorCacheRrfs>,
    ) -> Result<Vec<CircuitRc>> {
        let lines: Vec<_> = string.lines().collect();

        let tab_width: usize = 2;
        // ident and if it's a new node
        let mut stack: Vec<(CircuitIdent, bool)> = vec![];
        let mut was_previous_only_child_marker = false;
        let mut top_level = vec![];
        const WHITESPACE: &str = r"[\s│└├‣─]*";
        const NAME_INSIDE_MATCH: &str = r"(?:(?:\\')?[^']*)*";
        const AXIS_NAME_MATCH: &str = r"[^\s:\[\]]*"; // No whitespace, colon, or square brackets

        let (
            prefix_whitespace_cap,
            num_ident_cap,
            name_ident_cap,
            name_cap,
            autoname_disabled_cap,
            shape_cap,
            variant_cap,
            extra_cap,
        ) = (1, 2, 3, 4, 5, 6, 7, 8);
        static RE_STR: Lazy<(Regex, String)> = Lazy::new(|| {
            let name_match_capture: String = format!("'({})'", NAME_INSIDE_MATCH);
            // first match named axis name, then size. Name is optional. We handle whitespace.
            let shape_axis_match: String =
                format!(r"(?:{}\s*:\s*)?{}\s*", AXIS_NAME_MATCH, *SIZE_PROD_MATCH);
            let trailing_comment_match = r"(?:#.*?)?"; // TODO: test this never matches stuff it's not supposed to...
            let full_regex = format!(
                r"^({ws})(?:(\d+)|{nm})(?: \s*{nm})?( \s*AD)?(?: \s*\[\s*((?:{sh},\s*)*(?:{sh})?)\])?(?: \s*([a-zA-Z]+))?(?: \s*(.*?))?(?:{ws}){com}$",
                ws = WHITESPACE,
                nm = name_match_capture,
                sh = shape_axis_match,
                com = trailing_comment_match,
            );

            (Regex::new(&full_regex).unwrap(), full_regex)
        });
        let re = &RE_STR.0;
        let regex_string = &RE_STR.1;
        static RE_SKIP_LINE: Lazy<Regex> = Lazy::new(|| {
            // supports newlines and comment lines starting with #
            Regex::new(&format!(r"^{ws}(#.*)?$", ws = WHITESPACE)).unwrap()
        });
        static RE_ONLY_CHILD_MARKER: Lazy<Regex> = Lazy::new(|| {
            // supports newlines and comment lines starting with #
            Regex::new(&format!(r"^{ws}(?:▼|\|/)(#.*)?$", ws = WHITESPACE)).unwrap()
        });
        let mut first_num_spaces = None;
        let mut first_line = None;
        for line in lines {
            if RE_SKIP_LINE.is_match(line) {
                continue;
            }
            if RE_ONLY_CHILD_MARKER.is_match(line) {
                was_previous_only_child_marker = true;
                continue;
            }

            let re_captures =
                re.captures(line)
                    .ok_or_else(|| ParseCircuitError::RegexDidntMatch {
                        line: line.to_owned(),
                        regex_string: regex_string.clone(),
                    })?;
            let num_spaces_base = re_captures
                .get(prefix_whitespace_cap)
                .expect("if regex matches, group should be present")
                .as_str()
                .chars()
                .count();
            if first_num_spaces.is_none() {
                first_num_spaces = Some(num_spaces_base);
                first_line = Some(line.to_owned());
            }
            let first_num_spaces = first_num_spaces.unwrap();
            if num_spaces_base < first_num_spaces {
                bail!(ParseCircuitError::LessIndentationThanFirstItem {
                    first_num_spaces,
                    this_num_spaces_base: num_spaces_base,
                    first_line: first_line.unwrap(),
                    this_line: line.to_owned(),
                });
            }
            let num_spaces = num_spaces_base - first_num_spaces;
            let get_unindented_line = || line.chars().skip(first_num_spaces).collect();
            if num_spaces % tab_width != 0 {
                bail!(ParseCircuitError::InvalidIndentation {
                    tab_width,
                    spaces: num_spaces,
                    stack_indentation: stack.len(),
                    line: get_unindented_line(),
                });
            }
            let indentation_level =
                num_spaces / tab_width + (was_previous_only_child_marker as usize);
            if indentation_level > stack.len() {
                bail!(ParseCircuitError::InvalidIndentation {
                    tab_width,
                    spaces: num_spaces,
                    stack_indentation: stack.len(),
                    line: get_unindented_line(),
                });
            }
            stack.truncate(indentation_level);

            let unescape_name =
                |x: regex::Match| x.as_str().replace(r"\'", "'").replace(r"\\", r"\");

            let circuit_ident = match (
                re_captures.get(num_ident_cap),
                re_captures.get(name_ident_cap),
            ) {
                (None, None) => unreachable!(),
                (Some(c), None) => Num(c
                    .as_str()
                    .parse()
                    .context("failed to parse serial number")?),
                (None, Some(c)) => Str(unescape_name(c)),
                (Some(_), Some(_)) => unreachable!(),
            };

            let is_new_node = !partial_circuits.contains_key(&circuit_ident);

            let extra_and_parent_info =
                re_captures.get(extra_cap).map(|z| z.as_str()).unwrap_or("");
            let (extra, parent_info) = extra_and_parent_info
                .split_once('!')
                .map(|(x, y)| (x.trim(), Some(y.trim())))
                .unwrap_or((extra_and_parent_info, None));
            let extra = (!extra.is_empty()).then(|| extra.to_owned());
            let autoname_disabled = re_captures.get(autoname_disabled_cap).is_some();

            let is_ref = circuit_ident
                .as_str()
                .map(|s| self.reference_circuits.contains_key(&Name::new(s)))
                .unwrap_or(false);
            if is_ref {
                if re_captures.get(name_cap).is_some()
                    || re_captures.get(shape_cap).is_some()
                    || re_captures.get(variant_cap).is_some()
                    || extra.is_some()
                    || autoname_disabled
                {
                    bail!(
                        ParseCircuitError::ReferenceCircuitNameFollowedByAdditionalInfo {
                            reference_name: circuit_ident.as_str().unwrap().to_owned(),
                            line: get_unindented_line()
                        }
                    );
                }
            } else {
                let name = re_captures
                    .get(name_cap)
                    .map(unescape_name)
                    .map(|z| Name::new(&z))
                    .or_else(|| {
                        if is_new_node {
                            circuit_ident.as_str().map(|z| Name::new(z))
                        } else {
                            // if not new node, than this is just for checking and we want to avoid checking in this case
                            None
                        }
                    });

                let (shape, named_axes) = if let Some(shape_cap) = re_captures.get(shape_cap) {
                    let mut axis_strs: Vec<_> =
                        shape_cap.as_str().split(',').map(|z| z.trim()).collect();
                    if axis_strs.last() == Some(&"") {
                        // allow last axis to be empty due to trailing comma
                        // (Regex guarantees that only last axis has this I think, but better to be clear)
                        axis_strs.pop();
                    }

                    let parse_axis_size = |x| -> Result<(usize, Option<&str>)> {
                        let (striped_x, named_axis) = extract_named_axis(x)?;

                        Ok((
                            SymbolicSizeProduct::from_str(striped_x)
                                .context(
                                    "failed to parse number (including symbolic product) for shape",
                                )?
                                .try_into()
                                .context(
                                    "failed to convert size product to usize in parse of shape",
                                )?,
                            named_axis,
                        ))
                    };

                    let axes_info = axis_strs
                        .into_iter()
                        .map(parse_axis_size)
                        .collect::<Result<Vec<(usize, Option<&str>)>, _>>()?;

                    let shape = axes_info.iter().map(|(a, _)| *a).collect::<Shape>();
                    let mut named_axes = NamedAxes::default();
                    for (i, (_, name)) in axes_info.iter().enumerate() {
                        if let Some(name_) = name {
                            named_axes.insert(i.try_into().unwrap(), Name::new(*name_));
                        }
                    }

                    (Some(shape), Some(named_axes))
                } else {
                    (None, None)
                };
                let variant = flip_op_result(re_captures.get(variant_cap).map(|s| {
                    s.as_str()
                        .parse()
                        .context("failed to parse variant in parse_circuit")
                }))?;

                if is_new_node {
                    partial_circuits.insert(
                        circuit_ident.clone(),
                        PartialCirc {
                            name,
                            shape,
                            variant: variant.ok_or_else(|| {
                                ParseCircuitError::RegexDidntMatchGroup {
                                    group_name: "variant".to_owned(),
                                    line: get_unindented_line(),
                                    regex_string: regex_string.clone(),
                                    group: variant_cap,
                                }
                            })?,
                            extra: extra.unwrap_or_else(|| "".to_owned()),
                            children: vec![],
                            named_axes,
                            module_arg_specs: vec![],
                            autoname_disabled,
                        },
                    );
                } else if self.on_repeat_check_info_same {
                    let old_node = &partial_circuits[&circuit_ident];

                    let mut any_fail = false;
                    let mut err_strs = String::new();

                    macro_rules! fail_check {
                        ($n:ident, $w:expr) => {{
                            let failed = $n.clone().map(|x| $w(x) != old_node.$n).unwrap_or(false);
                            if failed {
                                any_fail = true;
                                err_strs += &format!(
                                    "{} mismatch. new={:?} != old={:?}\n",
                                    stringify!($n),
                                    $n.unwrap(),
                                    old_node.$n
                                );
                            }
                        }};
                        ($n:ident) => {
                            fail_check!($n, |x| x)
                        };
                    }
                    fail_check!(name, Some);
                    fail_check!(shape, Some);
                    fail_check!(variant);
                    fail_check!(extra);
                    let autoname_disabled = autoname_disabled.then_some(true);
                    fail_check!(autoname_disabled);
                    if any_fail {
                        bail!(ParseCircuitError::OnCircuitRepeatInfoIsNotSame {
                            repeat_ident: format!("{:?}", circuit_ident),
                            err_strs,
                            line: get_unindented_line()
                        });
                    }
                }
            }

            // now manipulate stack
            if stack.is_empty() {
                top_level.push(circuit_ident.clone())
            }
            if let Some((l, is_new)) = stack.last() {
                if let Str(s) = l {
                    if self.reference_circuits.contains_key(&Name::new(&s)) {
                        bail!(ParseCircuitError::ReferenceCircuitHasChildren {
                            reference_name: s.to_owned(),
                            child_line: get_unindented_line(),
                        });
                    }
                }
                if !is_new {
                    bail!(ParseCircuitError::RepeatedCircuitHasChildren {
                        repeated_ident: l.clone(),
                        child_line: get_unindented_line(),
                    });
                }

                self.add_child_info(
                    l,
                    &circuit_ident,
                    parent_info,
                    partial_circuits,
                    context,
                    tensor_cache,
                )?;
                let partial_parent = partial_circuits.get_mut(l).unwrap();
                partial_parent.children.push(circuit_ident.clone());
            } else if parent_info.is_some() {
                bail!(ParseCircuitError::HasParentInfoButNoParent {
                    ident: circuit_ident.clone(),
                    line: get_unindented_line(),
                });
            }
            if was_previous_only_child_marker {
                stack.pop();
                was_previous_only_child_marker = false;
            }
            stack.push((circuit_ident, is_new_node));
        }

        let arrayconstant_tensor_hashes: Vec<String> = partial_circuits
            .values()
            .filter_map(|x| {
                if !self.tensors_as_random && let CircuitType::Array = &x.variant && x.extra!="rand" {
                    return Some(x.extra.clone());
                }
                None
            })
            .collect();
        timed!(
            ensure_all_tensors_local(arrayconstant_tensor_hashes),
            100,
            true
        )?;
        top_level
            .iter()
            .map(|ident| {
                self.deep_convert_partial_circ(
                    ident,
                    partial_circuits,
                    context,
                    tensor_cache,
                    &mut Default::default(),
                )
            })
            .collect()
    }

    fn add_module_child_info(
        &self,
        module_ident: &CircuitIdent,
        child_ident: &CircuitIdent,
        child_info: &str,
        partial_circuits: &mut HashMap<CircuitIdent, PartialCirc>,
        context: &mut HashMap<CircuitIdent, CircuitRc>,
        tensor_cache: &mut Option<TensorCacheRrfs>,
    ) -> Result<()> {
        static RE_FLAGS: Lazy<Regex> = Lazy::new(|| Regex::new(&r"(.*) ([tf]{3})$").unwrap());

        let (sym_str, flags_str) = RE_FLAGS
            .captures(child_info)
            .map(|x| (x.get(1).unwrap().as_str(), Some(x.get(2).unwrap().as_str())))
            .unwrap_or((child_info, None));

        let [batchable, expandable, ban_non_symbolic_size_expand] = flags_str
            .map(|s| {
                s.chars()
                    .map(|x| TerseBool::try_from(x).unwrap().0)
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            })
            .unwrap_or([true, true, true]);
        // this is very obviously kind of a hack...
        // not exactly sure what we should do instead.
        let symbol_out_circs = self
            .parse_circuits_impl(sym_str, partial_circuits, context, tensor_cache)
            .with_context(|| {
                format!(
                    "{} child_ident={:?} {}",
                    "failed to parse symbol circuit for module",
                    child_ident,
                    "extra info (perhaps flags are given incorrectly?)"
                )
            })?;
        assert_eq!(symbol_out_circs.len(), 1);
        let symbol = symbol_out_circs.into_iter().next().unwrap();
        let symbol = symbol
            .as_symbol()
            .ok_or_else(|| ConstructError::ModuleExpectedSymbol {
                actual_circuit: symbol.clone(),
            })
            .context("module expected symbol in parse (via child extra info)")?
            .clone();

        partial_circuits
            .get_mut(module_ident)
            .unwrap()
            .module_arg_specs
            .push(ModuleArgSpec {
                symbol,
                batchable,
                expandable,
                ban_non_symbolic_size_expand,
            });

        Ok(())
    }

    fn add_child_info(
        &self,
        parent_ident: &CircuitIdent,
        child_ident: &CircuitIdent,
        child_info: Option<&str>,
        partial_circuits: &mut HashMap<CircuitIdent, PartialCirc>,
        context: &mut HashMap<CircuitIdent, CircuitRc>,
        tensor_cache: &mut Option<TensorCacheRrfs>,
    ) -> Result<()> {
        let parent = partial_circuits.get(parent_ident).unwrap();
        let unexpected_child_info = || -> anyhow::Error {
            ParseCircuitError::UnexpectedChildInfo {
                child_info: child_info.unwrap().to_owned(),
                parent_ident: parent_ident.clone(),
                child_ident: child_ident.clone(),
                parent_variant: parent.variant,
            }
            .into()
        };

        if parent.variant != CircuitType::Module {
            if child_info.is_some() {
                bail!(unexpected_child_info())
            }
            return Ok(());
        }

        if parent.children.is_empty() {
            if child_info.is_some() {
                bail!(unexpected_child_info().context("spec circuit had child info for module"))
            }
            return Ok(());
        } else {
            if child_info.is_none() {
                bail!(
                    ParseCircuitError::ModuleRequiresInputChildToHaveParentInfo {
                        module_ident: parent_ident.clone(),
                        child_ident: child_ident.clone(),
                    }
                )
            }
        }

        self.add_module_child_info(
            parent_ident,
            child_ident,
            child_info.unwrap(),
            partial_circuits,
            context,
            tensor_cache,
        )
    }

    fn deep_convert_partial_circ(
        &self,
        ident: &CircuitIdent,
        partial_circuits: &HashMap<CircuitIdent, PartialCirc>,
        context: &mut HashMap<CircuitIdent, CircuitRc>,
        tensor_cache: &mut Option<TensorCacheRrfs>,
        in_progress: &mut HashSet<CircuitIdent>,
        // mutable set of ancestor idents, insert when we start and remove when we finish
        // to detect cycles (when something is an ancestor of itself)
    ) -> Result<CircuitRc> {
        if let Some(ref_circ) = ident
            .as_str()
            .map(|x| self.reference_circuits.get(&Name::new(x)))
            .flatten()
        {
            return Ok(ref_circ.clone());
        }
        if let Some(already) = context.get(ident) {
            return Ok(already.clone());
        }
        if !in_progress.insert(ident.clone()) {
            bail!(ParseCircuitError::Cycle {
                ident: ident.clone()
            });
        }
        let ps = &partial_circuits[ident];
        let children: Vec<CircuitRc> = ps
            .children
            .iter()
            .map(|x| {
                self.deep_convert_partial_circ(
                    x,
                    partial_circuits,
                    context,
                    tensor_cache,
                    in_progress,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        let result = self
            .deep_convert_partial_circ_children(ident, ps, children, tensor_cache)
            .with_context(|| {
                format!(
                    concat!(
                        "in parse, failed to convert ident={:?}, ",
                        "variant={}\nfull partial circuit: {:?}"
                    ),
                    ident, ps.variant, ps
                )
            })?
            .rename(partial_circuits[ident].name.into());

        context.insert(ident.clone(), result.clone());
        in_progress.remove(ident);
        Ok(result)
    }

    fn deep_convert_partial_circ_children(
        &self,
        ident: &CircuitIdent,
        ps: &PartialCirc,
        mut children: Vec<CircuitRc>,
        tensor_cache: &mut Option<TensorCacheRrfs>,
    ) -> Result<CircuitRc> {
        let expected_k_children = |k| {
            if children.len() != k {
                bail!(ParseCircuitError::WrongNumberChildren {
                    expected: k,
                    found: children.len(),
                    ident: ident.clone()
                })
            }
            Ok(())
        };

        let extra_should_be_empty = || {
            if !ps.extra.is_empty() {
                bail!(ParseCircuitError::ExpectedNoExtraInfo {
                    extra: ps.extra.clone(),
                    variant: ps.variant
                })
            }
            Ok(())
        };

        type T = CircuitType;

        if ps.variant != T::Module {
            assert!(ps.module_arg_specs.is_empty());
        }

        let result = match ps.variant {
            T::Array => {
                expected_k_children(0)?;
                if self.tensors_as_random || ps.extra == "rand" {
                    if ps.extra != "rand" && !self.allow_hash_with_random {
                        // we allow "rand" even when self.tensors_as_random is true

                        extra_should_be_empty().context(
                            "self.tensors_as_random was passed, so array hashes should not be passed!",
                        )?
                    }
                    Array::randn_named(
                        ps.shape.clone().ok_or(ParseCircuitError::ShapeNeeded {
                            variant: ps.variant,
                        })?,
                        ps.name,
                        self.tensors_as_random_device_dtype.clone(),
                    )
                    .rc()
                } else {
                    let out = Array::from_hash_prefix(ps.name, &ps.extra, tensor_cache)
                        .context("failed to parse array constant from hash prefix")?
                        .rc();
                    if let Some(shape) = &ps.shape {
                        if shape != out.shape() {
                            bail!(ParseCircuitError::ArrayShapeLoadedFromHashDiffersFromProvidedShape {
                                loaded_from_hash_shape : out.shape().clone(),
                                provided_shape : shape.clone(),
                            });
                        }
                    }
                    out
                }
            }
            T::Scalar => {
                expected_k_children(0)?;
                Scalar::nrc(
                    parse_numeric(&ps.extra).context("failed to parse out number for Scalar")?,
                    ps.shape
                        .as_ref()
                        .ok_or(ParseCircuitError::ShapeNeeded {
                            variant: ps.variant,
                        })?
                        .clone(),
                    ps.name,
                )
            }
            T::Add => {
                extra_should_be_empty()?;
                Add::try_new(children, ps.name)?.rc()
            }
            T::Concat => Concat::new_signed_axis(
                children,
                parse_numeric(&ps.extra).context("failed to parse out concat axis")?,
                ps.name,
            )?
            .rc(),
            T::Einsum => if ps.extra.starts_with(FANCY_PREFIX) {
                Einsum::from_fancy_string(&ps.extra[FANCY_PREFIX.len()..], children, ps.name)
                    .context(concat!(
                        "Couldn't parse fancy einsum string. If you wanted ",
                        "to parse a regular einsum, remove \"fancy:\"."
                    ))
            } else {
                Einsum::from_einsum_string(&ps.extra, children.clone(), ps.name).context(concat!(
                    "Couldn't parse einsum string. If you wanted ",
                    "to parse a fancy einsum string, prefix it by \"fancy:\"."
                ))
            }?
            .rc(),
            T::Rearrange => {
                expected_k_children(1)?;
                Rearrange::from_string(children[0].clone(), &ps.extra, ps.name)?.rc()
            }
            T::Symbol => {
                expected_k_children(0)?;
                let shape = ps
                    .shape
                    .as_ref()
                    .ok_or(ParseCircuitError::ShapeNeeded {
                        variant: ps.variant,
                    })?
                    .clone();
                if ps.extra.is_empty() {
                    Symbol::new_with_none_uuid(shape, ps.name).rc()
                } else if ps.extra == "rand" {
                    Symbol::new_with_random_uuid(shape, ps.name).rc()
                } else {
                    Symbol::nrc(
                        shape,
                        Uuid::from_str(&ps.extra).map_err(|e| ParseError::InvalidUuid {
                            string: ps.extra.to_owned(),
                            err_msg: e.to_string(),
                        })?,
                        ps.name,
                    )
                }
            }
            T::GeneralFunction => {
                GeneralFunction::new_from_parse(children, ps.extra.clone(), ps.name)?.rc()
            }
            T::Index => {
                expected_k_children(1)?;
                Index::try_new(
                    children[0].clone(),
                    TensorIndex::from_bijection_string(&ps.extra, tensor_cache)?,
                    ps.name,
                )?
                .rc()
            }
            T::Scatter => {
                expected_k_children(1)?;
                Scatter::try_new(
                    children[0].clone(),
                    TensorIndex::from_bijection_string(&ps.extra, tensor_cache)?,
                    ps.shape
                        .as_ref()
                        .ok_or(ParseCircuitError::ShapeNeeded {
                            variant: ps.variant,
                        })?
                        .clone(),
                    ps.name,
                )?
                .rc()
            }
            T::Module => {
                extra_should_be_empty()?;
                if children.len() == 0 {
                    bail!(ParseCircuitError::ModuleNoSpecCircuit { name: ps.name })
                }

                let nodes = children.split_off(1);
                let spec_circuit = children.pop().unwrap();
                let arg_specs = ps.module_arg_specs.clone();
                assert_eq!(arg_specs.len(), nodes.len());

                Module::try_new(
                    nodes,
                    ModuleSpec::new(
                        spec_circuit,
                        arg_specs,
                        self.module_check_all_inputs_used,
                        self.module_check_unique_arg_names,
                    )
                    .context("spec construction failed in parse")?,
                    ps.name,
                )
                .context("module construction failed in parse")?
                .rc()
            }
            T::Tag => {
                expected_k_children(1)?;
                Uuid::from_str(&ps.extra)
                    .map_err(|e| ParseError::InvalidUuid {
                        string: ps.extra.clone(),
                        err_msg: e.to_string(),
                    })
                    .map(|uuid| Tag::nrc(children[0].clone(), uuid, ps.name))?
            }
            T::Cumulant => {
                extra_should_be_empty()?;
                Cumulant::nrc(children, ps.name)
            }
            T::DiscreteVar => {
                expected_k_children(2)?;
                DiscreteVar::try_new(children[0].clone(), children[1].clone(), ps.name)?.rc()
            }
            T::StoredCumulantVar => {
                let (cum_nums, uuid) =
                    if let [cum_nums, uuid] = ps.extra.split("|").collect::<Vec<_>>()[..] {
                        (cum_nums, uuid)
                    } else {
                        bail!(ParseCircuitError::StoredCumulantVarExtraInvalid {
                            extra: ps.extra.clone(),
                        })
                    };
                let keys = cum_nums
                    .split(",")
                    .map(|s| parse_numeric(s.trim()))
                    .collect::<Result<Vec<_>, _>>()
                    .context("failed to parse keys for stored cumulant var")?;

                let uuid = Uuid::from_str(uuid).map_err(|e| ParseError::InvalidUuid {
                    string: ps.extra.clone(),
                    err_msg: e.to_string(),
                })?;
                if keys.len() != children.len() {
                    bail!(ParseCircuitError::WrongNumberChildren {
                        expected: keys.len(),
                        found: children.len(),
                        ident: ident.clone()
                    })
                } else {
                    StoredCumulantVar::try_new(zip(keys, children).collect(), uuid, ps.name)?.rc()
                }
            }
            T::SetSymbolicShape => {
                expected_k_children(1)?;
                extra_should_be_empty()?;
                SetSymbolicShape::try_new(
                    children[0].clone(),
                    ps.shape
                        .as_ref()
                        .ok_or(ParseCircuitError::ShapeNeeded {
                            variant: ps.variant,
                        })?
                        .clone(),
                    ps.name,
                )?
                .rc()
            }
            T::Conv => {
                unimplemented!()
            }
        };
        if let Some(provided_shape) = &ps.shape {
            if result.shape() != provided_shape {
                bail!(ParseCircuitError::PassedInShapeDoesntMatchComputedShape {
                    provided_shape: provided_shape.clone(),
                    computed_shape: result.shape().clone(),
                })
            }
        }
        let result = if let Some(named_axes) = ps.named_axes.clone() {
            set_named_axes(result, named_axes)?
        } else {
            result
        };
        let result = if ps.autoname_disabled {
            result.with_autoname_disabled(true)
        } else {
            result
        };

        Ok(result)
    }
}

fn extract_named_axis<'a>(axis: &'a str) -> Result<(&'a str, Option<&'a str>)> {
    let trimed = axis.trim();
    if trimed.contains(':') {
        let split = trimed.split(":").collect_vec();
        if split.len() != 2 {
            bail!(ParseCircuitError::InvalidAxisShapeFormat {
                axis: axis.to_owned()
            })
        }
        let axis_name = split[0];
        let striped_axis = split[1];
        Ok((striped_axis, Some(axis_name)))
    } else {
        Ok((axis, None))
    }
}

#[apply(python_error_exception)]
#[base_error_name(ReferenceCirc)]
#[base_exception(PyValueError)]
#[derive(Error, Debug, Clone)]
pub enum ReferenceCircError {
    #[error("circuit={circuit:?} ({e_name})")]
    ByNameHasNoneName { circuit: CircuitRc },

    #[error("dup_idents={dup_idents:?} ({e_name})")]
    DuplicateIdentifier { dup_idents: HashMap<Name, usize> },
}

const REGEX_101_MSG: &str = "\nTry using regex101.com : )\n(TODO: auto generate regex test page via https://github.com/firasdib/Regex101/wiki/API#python-3)";

#[apply(python_error_exception)]
#[base_error_name(ParseCircuit)]
#[base_exception(PyParseError)]
#[derive(Error, Debug, Clone)]
pub enum ParseCircuitError {
    #[error(
        "regex didn't match '{line}'\nregex='{regex_string}'{}\n({e_name})",
        REGEX_101_MSG
    )]
    RegexDidntMatch { line: String, regex_string: String },

    #[error(
        "regex didn't match '{group_name}' group={group} for '{line}'\n{}\nregex='{regex_string}'{}\n({e_name})",
        "Note: this error can also be caused by failing to lookup a reference circuit",
        REGEX_101_MSG
    )]
    RegexDidntMatchGroup {
        group_name: String,
        line: String,
        regex_string: String,
        group: usize,
    },

    #[error("effectively negative indent!\n{}\n{}\n{}\n({e_name})",
        format!("this_num_spaces_base={} < first_num_spaces={}", this_num_spaces_base, first_num_spaces),
        format!("first_line='{}'", first_line),
        format!("this_line='{}'", this_line)
        )]
    LessIndentationThanFirstItem {
        this_num_spaces_base: usize,
        first_num_spaces: usize,
        first_line: String,
        this_line: String,
    },

    #[error("Parsing invalid indentation, tab width {tab_width} num spaces {spaces} stack indentation {stack_indentation}\nline='{line}'\n({e_name})")]
    InvalidIndentation {
        spaces: usize,
        tab_width: usize,
        stack_indentation: usize,
        line: String,
    },

    #[error("extra='{extra}' for variant={variant}, but expected empty ({e_name})")]
    ExpectedNoExtraInfo { extra: String, variant: CircuitType },

    #[error("Trying to parse cycle! identifier {ident:?} ({e_name})")]
    Cycle { ident: CircuitIdent },

    #[error("Parsing wrong number of children, expected {expected} found {found}, ident={ident:?} ({e_name})")]
    WrongNumberChildren {
        expected: usize,
        found: usize,
        ident: CircuitIdent,
    },

    #[error("child_info={child_info:?} for child_ident={child_ident:?}, parent_ident={parent_ident:?}, and parent_variant={parent_variant}, but expected None ({e_name})")]
    UnexpectedChildInfo {
        child_info: String,
        child_ident: CircuitIdent,
        parent_ident: CircuitIdent,
        parent_variant: CircuitType,
    },

    #[error("Parsing invalid circuit variant {v} ({e_name})")]
    InvalidVariant { v: String },

    #[error("Parsing shape needed but not provided on {variant} ({e_name})")]
    ShapeNeeded { variant: CircuitType },

    #[error("actual_num_circuits={actual_num_circuits} ({e_name})")]
    ExpectedOneCircuitGotMultiple { actual_num_circuits: usize },

    #[error("expected 't' or 'f' got='{got}' ({e_name})")]
    InvalidTerseBool { got: String },

    #[error("reference_name={reference_name} child_line='{child_line}' ({e_name})")]
    ReferenceCircuitHasChildren {
        reference_name: String,
        child_line: String,
    },

    #[error("repeated_ident={repeated_ident:?} child_line='{child_line}' ({e_name})")]
    RepeatedCircuitHasChildren {
        repeated_ident: CircuitIdent,
        child_line: String,
    },

    #[error("ident={ident:?} line='{line}' ({e_name})")]
    HasParentInfoButNoParent { ident: CircuitIdent, line: String },

    #[error("reference_name={reference_name} ({e_name})")]
    ReferenceCircuitNameFollowedByAdditionalInfo {
        reference_name: String,
        line: String,
    },

    #[error("computed_shape={computed_shape:?} != provided_shape={provided_shape:?} ({e_name})")]
    PassedInShapeDoesntMatchComputedShape {
        computed_shape: Shape,
        provided_shape: Shape,
    },

    // different error type from above for clarity
    #[error(
        "loaded_from_hash_shape={loaded_from_hash_shape:?} != provided_shape={provided_shape:?}.{}\n({e_name})",
        " Note that you don't have to provide the shape if you provide a hash"
    )]
    ArrayShapeLoadedFromHashDiffersFromProvidedShape {
        loaded_from_hash_shape: Shape,
        provided_shape: Shape,
    },

    #[error("repeat_ident={repeat_ident} \n{err_strs}line='{line}' ({e_name})")]
    OnCircuitRepeatInfoIsNotSame {
        repeat_ident: String,
        line: String,
        err_strs: String,
    },

    #[error(
        "Module has no children, but should have at least a spec circuit name={name:?} ({e_name})"
    )]
    ModuleNoSpecCircuit { name: Option<Name> },

    #[error(
        "for module_ident={module_ident:?} the input child_ident={child_ident:?} {}\n{}\n{}\n({e_name})",
        "doesn't have any extra parent info.",
        "modules must indicate which symbol is replaced for each input.",
        "this is done with '!', as in: \"'my_input' Einsum a->a ! 'my_sym'\""
    )]
    ModuleRequiresInputChildToHaveParentInfo {
        module_ident: CircuitIdent,
        child_ident: CircuitIdent,
    },

    #[error("extra='{extra}' invalid. Should contain | (TODO: example). ({e_name})")]
    StoredCumulantVarExtraInvalid { extra: String },

    #[error("Invalid formating for shape axis {axis} ({e_name})")]
    InvalidAxisShapeFormat { axis: String },
}
