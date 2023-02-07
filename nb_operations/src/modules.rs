use std::{
    collections::{btree_map::Entry, BTreeMap},
    iter,
    sync::Arc,
};

use anyhow::{anyhow, bail, Context, Result};
use circuit_base::{
    expand_node::{expand_node, MapReplaceExpander, ReplaceMapRc},
    module::{are_args_used, is_intersecting_free_syms, SymbolSetRc},
    CircuitNode, CircuitRc, Module, ModuleArgSpec, ModuleSpec, Rearrange, Symbol,
};
use circuit_rewrites::{
    deep_rewrite::SimpFnSubset,
    module_rewrite::{
        elim_no_input_module, extract_rewrite_raw, module_remove_unused_inputs, module_strip_args,
    },
};
use get_update_node::{AnyFound, IterativeMatcher, IterativeMatcherRc, MatcherData};
use itertools::{izip, multiunzip};
use macro_rules_attribute::apply;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    PyObject,
};
use rr_util::{
    cached_method,
    caching::FastUnboundedCache,
    eq_by_big_hash::EqByBigHash,
    fn_struct, impl_eq_by_big_hash,
    name::Name,
    pycall, python_error_exception,
    rearrange_spec::{OpSize, RearrangeSpec, RearrangeSpecError},
    sv, tu8v, unwrap,
    util::{DimNumMaker, HashBytes},
    IndexSet,
};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use thiserror::Error;
use uuid::uuid;

#[pyfunction]
#[pyo3(signature=(
    circuit,
    matcher,
    prefix_to_strip = None,
    module_name = None,
    check_all_inputs_used = true,
    check_unique_arg_names = true,
    circuit_to_arg_spec = None
))]
pub fn extract_rewrite(
    circuit: CircuitRc,
    matcher: IterativeMatcherRc,
    prefix_to_strip: Option<String>,
    module_name: Option<Name>,
    check_all_inputs_used: bool,
    check_unique_arg_names: bool,
    circuit_to_arg_spec: Option<PyObject>,
) -> Result<Module> {
    let edges: Vec<CircuitRc> = matcher.get(circuit.clone(), false)?.into_iter().collect();
    let mut specs: Vec<(CircuitRc, ModuleArgSpec)> = edges
        .into_iter()
        .map(|n| {
            if let Some(cts) = &circuit_to_arg_spec {
                pycall!(cts, (n.clone(),), anyhow)
            } else {
                Ok(ModuleArgSpec::just_name_shape(n.clone(), true, true, false))
            }
            .map(|z| (n, z))
        })
        .collect::<Result<Vec<_>>>()?;
    specs.sort_by_key(|x| x.1.symbol.info().name);
    extract_rewrite_raw(
        circuit,
        specs,
        prefix_to_strip,
        module_name,
        check_all_inputs_used,
        check_unique_arg_names,
    )
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct BindItem {
    #[pyo3(get, set)]
    pub matcher: IterativeMatcherRc,
    #[pyo3(get, set)]
    pub input_circuit: CircuitRc,
    #[pyo3(get, set)]
    pub batchable: bool,
    #[pyo3(get, set)]
    pub expandable: bool,
    #[pyo3(get, set)]
    pub ban_non_symbolic_size_expand: bool,
}

#[pymethods]
impl BindItem {
    #[new]
    #[pyo3(signature=(
        matcher,
        input_circuit,
        batchable = ModuleArgSpec::default().batchable,
        expandable = ModuleArgSpec::default().expandable,
        ban_non_symbolic_size_expand = ModuleArgSpec::default().ban_non_symbolic_size_expand
    ))]
    fn new(
        matcher: IterativeMatcherRc,
        input_circuit: CircuitRc,
        batchable: bool,
        expandable: bool,
        ban_non_symbolic_size_expand: bool,
    ) -> Self {
        Self {
            matcher,
            input_circuit,
            batchable,
            expandable,
            ban_non_symbolic_size_expand,
        }
    }
}

impl Binder {
    fn into_item(self) -> BindItem {
        match self {
            Self::Tup(matcher, input_circuit) => BindItem {
                matcher,
                input_circuit,
                batchable: ModuleArgSpec::default().batchable,
                expandable: ModuleArgSpec::default().expandable,
                ban_non_symbolic_size_expand: ModuleArgSpec::default().ban_non_symbolic_size_expand,
            },
            Self::Item(item) => item,
        }
    }
}

#[derive(Debug, FromPyObject)]
pub enum Binder {
    Tup(IterativeMatcherRc, CircuitRc),
    Item(BindItem),
}

#[pyfunction]
#[pyo3(signature=(spec_circuit, *binders, check_unique_arg_names = true, name = None))]
pub fn module_new_bind(
    spec_circuit: CircuitRc,
    binders: Vec<Binder>,
    check_unique_arg_names: bool,
    name: Option<Name>,
) -> Result<Module> {
    let (nodes, arg_specs) = binders
        .into_iter()
        .map(|binder| {
            let BindItem {
                matcher,
                input_circuit,
                batchable,
                expandable,
                ban_non_symbolic_size_expand,
            } = binder.into_item();
            let matched_circuit =
                matcher
                    .get_unique(spec_circuit.clone(), false)
                    .context(format!(
                        "failed to get unique for matcher={} in bind",
                        *matcher
                    ))?;

            let symbol = matched_circuit.as_symbol().cloned().ok_or_else(|| {
                ModuleBindError::ExpectedSymbol {
                    matched_circuit,
                    matcher,
                    spec_circuit: spec_circuit.clone(),
                }
            })?;

            Ok((
                input_circuit,
                ModuleArgSpec {
                    symbol,
                    batchable,
                    expandable,
                    ban_non_symbolic_size_expand,
                },
            ))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .unzip();

    Module::try_new(
        nodes,
        ModuleSpec::new(spec_circuit, arg_specs, true, check_unique_arg_names)?,
        name,
    )
}

#[apply(python_error_exception)]
#[base_error_name(ModuleBind)]
#[base_exception(PyRuntimeError)]
#[derive(Error, Debug, Clone)]
pub enum ModuleBindError {
    #[error("expected to match symbol, matched_circuit={matched_circuit:?}\nfor matcher={matcher:?}\nspec_circuit={spec_circuit:?}\n({e_name})")]
    ExpectedSymbol {
        matched_circuit: CircuitRc,
        matcher: IterativeMatcherRc,
        spec_circuit: CircuitRc,
    },
}

/// args are outer to inner
pub fn fuse_modules_impl(
    orig_modules: &[Module],
    new_nodes: &[Vec<CircuitRc>],
) -> Result<Vec<(CircuitRc, ModuleArgSpec)>> {
    // reverse to get inner shapes inside of outer
    let batch_shapes_per_mod: Vec<_> = orig_modules
        .into_iter()
        .map(|m| m.aligned_batch_shape())
        .collect();
    let cumulative_batch_shapes = batch_shapes_per_mod
        .iter()
        .rev()
        .scan(vec![], |state, sh| {
            let old_state = state.clone();
            state.extend(sh.iter().cloned());
            Some(old_state)
        })
        .collect::<Vec<_>>()
        .into_iter()
        .rev(); // reverse again

    let out = new_nodes
        .into_iter()
        .zip(cumulative_batch_shapes)
        .zip(orig_modules)
        .zip(batch_shapes_per_mod)
        .map(|(((items, cum_batch_shape), m), overall_mod_batch_shape)| {
            (*items)
                .to_owned()
                .into_iter()
                .zip(m.args())
                .zip(m.spec.arg_specs.clone())
                .zip(m.spec.batch_shapes(m.args_slice()))
                .map(|(((node, orig_node), arg_spec), orig_batch_shape)| {
                    let missing_ndims = overall_mod_batch_shape.len() - orig_batch_shape.len();
                    let non_batch = orig_node.ndim() - orig_batch_shape.len();
                    let extra_dims = node.ndim() - orig_node.ndim();
                    let any_batching = extra_dims > 0 || orig_batch_shape.len() > 0;
                    let needs_pad_orig = missing_ndims > 0 && extra_dims > 0;
                    let new_node = if (any_batching && cum_batch_shape.len() > 0) || needs_pad_orig
                    {
                        let mut maker = DimNumMaker::default();
                        let same_outer = maker.next_range(extra_dims);
                        let pad_for_orig = if needs_pad_orig {
                            maker.next_range(missing_ndims)
                        } else {
                            0..0
                        };
                        let same_orig_batch = maker.next_range(orig_batch_shape.len());
                        let cum = maker.next_range(cum_batch_shape.len());
                        let same_non_batch = maker.next_range(non_batch);

                        if maker.running > u8::MAX as usize {
                            bail!(anyhow!(RearrangeSpecError::LenShapeTooLarge {
                                len_shape: maker.running
                            })
                            .context("too many dims in fuse module for flatten"));
                        }

                        let mut sizes = sv![OpSize::NONE; maker.running];
                        for i in pad_for_orig.clone() {
                            sizes[i] = Some(overall_mod_batch_shape[i - pad_for_orig.start]).into();
                        }
                        for i in cum.clone() {
                            sizes[i] = Some(cum_batch_shape[i - cum.start]).into();
                        }

                        let spec = RearrangeSpec::new(
                            same_outer
                                .clone()
                                .chain(same_orig_batch.clone())
                                .chain(same_non_batch.clone())
                                .map(|i| tu8v![i as u8])
                                .collect(),
                            same_outer
                                .chain(pad_for_orig)
                                .chain(same_orig_batch)
                                .chain(cum)
                                .chain(same_non_batch.clone())
                                .map(|i| tu8v![i as u8])
                                .collect(),
                            sizes,
                        )
                        .unwrap();

                        let rep_name = node.info().name.map(|x| format!("{} rep_fuse", x).into());
                        Rearrange::nrc(node, spec, rep_name)
                    } else {
                        node
                    };
                    Ok((new_node, arg_spec))
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();

    Ok(out)
}

#[derive(Debug, Clone)]
struct NestedModuleItems {
    modules: Vec<Module>,
    new_nodes_flat: Vec<Vec<CircuitRc>>,
    new_nodes_base: Vec<Vec<CircuitRc>>,
    flat: Vec<(CircuitRc, ModuleArgSpec)>,
    flat_sym_to_arg: HashMap<Symbol, CircuitRc>,
    hash: HashBytes,
}

impl Default for NestedModuleItems {
    fn default() -> Self {
        Self::new(vec![], vec![], vec![]).unwrap()
    }
}

impl NestedModuleItems {
    fn new(
        modules: Vec<Module>,
        new_nodes_flat: Vec<Vec<CircuitRc>>,
        new_nodes_base: Vec<Vec<CircuitRc>>,
    ) -> Result<Self> {
        assert_eq!(modules.len(), new_nodes_flat.len());
        assert_eq!(modules.len(), new_nodes_base.len());
        for ((self_m, new), base) in modules.iter().zip(&new_nodes_flat).zip(&new_nodes_base) {
            assert_eq!(self_m.num_args(), new.len());
            assert_eq!(self_m.num_args(), base.len());
        }

        let flat = fuse_modules_impl(&modules, &new_nodes_flat)?;
        let flat_sym_to_arg = flat
            .iter()
            .map(|(circ, arg_spec)| (arg_spec.symbol.clone(), circ.clone()))
            .collect();

        let mut hasher = blake3::Hasher::new();

        for ((m, this_new_nodes_flat), this_new_nodes_base) in
            modules.iter().zip(&new_nodes_flat).zip(&new_nodes_base)
        {
            for n in this_new_nodes_flat {
                hasher.update(&n.hash());
            }
            for n in this_new_nodes_base {
                hasher.update(&n.hash());
            }
            hasher.update(uuid!("a5685fe8-00a8-4680-bf64-a5802b36fd7d").as_bytes());
            hasher.update(&m.hash());
        }

        Ok(Self {
            modules,
            new_nodes_flat,
            new_nodes_base,
            flat,
            flat_sym_to_arg,
            hash: hasher.finalize().into(),
        })
    }

    fn push(
        &self,
        m: Module,
        this_new_nodes_flat: Vec<CircuitRc>,
        this_new_nodes_base: Vec<CircuitRc>,
    ) -> Result<Self> {
        let mut modules = self.modules.clone();
        let mut new_nodes_flat = self.new_nodes_flat.clone();
        let mut new_nodes_base = self.new_nodes_base.clone();
        modules.push(m);
        new_nodes_flat.push(this_new_nodes_flat);
        new_nodes_base.push(this_new_nodes_base);
        Self::new(modules, new_nodes_flat, new_nodes_base)
    }
}

impl EqByBigHash for NestedModuleItems {
    fn hash(&self) -> HashBytes {
        self.hash
    }
}
impl_eq_by_big_hash!(NestedModuleItems);

// inner to outer
// NOTE: this is reverse order from how we store in NestedModuleItems!!!
fn_struct!(
    pub NestedModuleNamer: Fn(
        base_circuit: CircuitRc,
        running_circuit: CircuitRc,
        modules: Vec<Module>,
        pushed_overall_mod_count: Option<usize>
    ) -> Option<Name>
);

#[pyfunction]
#[pyo3(signature=(bind_name = "bind".to_owned()))]
pub fn default_nested_module_namer(bind_name: String) -> NestedModuleNamer {
    NestedModuleNamer::Dyn(NestedModuleNamerDynStruct(Arc::new(
        move |base_circuit, _, modules, _| {
            if modules.is_empty() {
                return Ok(base_circuit.info().name);
            }

            if base_circuit.info().name.is_none() || modules.iter().any(|m| m.info().name.is_none())
            {
                return Ok(None);
            }

            Ok(Some(
                format!(
                    "{} {bind_name}:{}",
                    base_circuit.info().name.unwrap(),
                    modules
                        .iter()
                        .map(|x| x.info().name.unwrap().string())
                        .collect::<Vec<String>>()
                        .join(",")
                )
                .into(),
            ))
        },
    )))
}

impl Default for NestedModuleNamer {
    fn default() -> Self {
        default_nested_module_namer("bind".to_owned())
    }
}

fn_struct!(pub ModuleConstructCallback: FnMut(m: Module, applied_modules: Vec<Module>, pushed_overall_mod_count: usize) -> CircuitRc;
{
    UpdateBindings(ModuleConstructUpdateBindings),
});

#[pyclass]
#[derive(Debug, Clone)]
pub struct ModulePusher {
    #[pyo3(get)]
    flatten_modules: bool,
    #[pyo3(get)]
    module_construct_callback: ModuleConstructCallback,
    #[pyo3(get)]
    bind_encountered_symbols: bool,
    #[pyo3(get)]
    namer: NestedModuleNamer,
    any_found: AnyFound,
    cache: FastUnboundedCache<
        (
            HashBytes,
            IterativeMatcherRc,
            IterativeMatcherRc,
            PushDownMode,
            HashBytes,
            HashBytes,
        ),
        (CircOrGet, (bool, HashSet<CircuitRc>)),
    >,
    replace_expander: MapReplaceExpander,
}

impl Default for ModulePusher {
    fn default() -> Self {
        Self {
            flatten_modules: true,
            module_construct_callback: Self::remove_unused_callback(false, true),
            bind_encountered_symbols: true,
            namer: Default::default(),
            any_found: Default::default(),
            cache: Default::default(),
            replace_expander: MapReplaceExpander::new_noop(),
        }
    }
}

#[derive(Clone, Debug)]
enum CircOrGet {
    Circ(CircuitRc),
    Get(HashSet<CircuitRc>),
}

#[pymethods]
impl ModulePusher {
    #[new]
    #[pyo3(signature=(
        flatten_modules = Self::default().flatten_modules,
        module_construct_callback = Self::default().module_construct_callback,
        bind_encountered_symbols = Self::default().bind_encountered_symbols,
        namer = Self::default().namer
    ))]
    pub fn new(
        flatten_modules: bool,
        module_construct_callback: ModuleConstructCallback,
        bind_encountered_symbols: bool,
        namer: NestedModuleNamer,
    ) -> Self {
        Self {
            flatten_modules,
            module_construct_callback,
            bind_encountered_symbols,
            namer,
            ..Default::default()
        }
    }

    #[pyo3(signature=(circuit, traversal, skip_module = MatcherData::Always(false).into()))]
    fn __call__(
        &mut self,
        _py: Python<'_>,
        circuit: CircuitRc,
        traversal: IterativeMatcherRc,
        skip_module: IterativeMatcherRc,
    ) -> Result<CircuitRc> {
        self.push_down_modules(circuit, traversal, skip_module)
    }

    #[pyo3(signature=(circuit, traversal, skip_module = MatcherData::Always(false).into()))]
    pub fn push_down_modules(
        &mut self,
        circuit: CircuitRc,
        traversal: IterativeMatcherRc,
        skip_module: IterativeMatcherRc,
    ) -> Result<CircuitRc> {
        self.push_down_modules_impl(circuit, traversal, skip_module, PushDownMode::Circ)
            .map(|x| unwrap!(x, CircOrGet::Circ))
    }

    // TODO: fancy validate
    #[pyo3(signature=(circuit, get, skip_module = MatcherData::Always(false).into()))]
    pub fn get_push_down_modules(
        &mut self,
        circuit: CircuitRc,
        get: IterativeMatcherRc,
        skip_module: IterativeMatcherRc,
    ) -> Result<HashSet<CircuitRc>> {
        self.push_down_modules_impl(circuit, get, skip_module, PushDownMode::Get)
            .map(|x| unwrap!(x, CircOrGet::Get))
    }

    #[pyo3(signature=(circuit, get, skip_module = MatcherData::Always(false).into()))]
    pub fn get_unique_op_push_down_modules(
        &mut self,
        circuit: CircuitRc,
        get: IterativeMatcherRc,
        skip_module: IterativeMatcherRc,
    ) -> Result<Option<CircuitRc>> {
        let out = self.get_push_down_modules(circuit, get, skip_module)?;
        if out.len() > 1 {
            bail!("found {} matches which is > 1", out.len());
        }
        Ok(out.into_iter().next())
    }

    #[pyo3(signature=(circuit, get, skip_module = MatcherData::Always(false).into()))]
    pub fn get_unique_push_down_modules(
        &mut self,
        circuit: CircuitRc,
        get: IterativeMatcherRc,
        skip_module: IterativeMatcherRc,
    ) -> Result<CircuitRc> {
        self.get_unique_op_push_down_modules(circuit, get, skip_module)?
            .ok_or_else(|| anyhow!("found no matches!"))
    }

    #[staticmethod]
    #[pyo3(signature=(
        remove_unused_inputs = true,
        add_suffix_on_remove_unused = false,
        elim_no_input_modules = true
    ))]
    pub fn remove_and_elim_callback(
        remove_unused_inputs: bool,
        add_suffix_on_remove_unused: bool,
        elim_no_input_modules: bool,
    ) -> ModuleConstructCallback {
        if !remove_unused_inputs && !elim_no_input_modules {
            Self::noop_callback()
        } else if !remove_unused_inputs {
            Self::elim_no_input_modules_callback()
        } else {
            Self::remove_unused_callback(add_suffix_on_remove_unused, elim_no_input_modules)
        }
    }

    #[staticmethod]
    #[pyo3(signature=(add_suffix_on_remove_unused = false, elim_no_input_modules = true))]
    pub fn remove_unused_callback(
        add_suffix_on_remove_unused: bool,
        elim_no_input_modules: bool,
    ) -> ModuleConstructCallback {
        ModuleConstructCallback::Dyn(ModuleConstructCallbackDynStruct(Arc::new(
            move |m, _, _| {
                module_remove_unused_inputs(&m, add_suffix_on_remove_unused, elim_no_input_modules)
            },
        )))
    }

    #[staticmethod]
    pub fn noop_callback() -> ModuleConstructCallback {
        ModuleConstructCallback::Dyn(ModuleConstructCallbackDynStruct(Arc::new(
            move |m, _, _| Ok(m.rc()),
        )))
    }

    #[staticmethod]
    pub fn elim_no_input_modules_callback() -> ModuleConstructCallback {
        ModuleConstructCallback::Dyn(ModuleConstructCallbackDynStruct(Arc::new(
            move |m, _, _| Ok(elim_no_input_module(&m).unwrap_or_else(|| m.rc())),
        )))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum PushDownMode {
    PushOverride,
    Circ,
    Get,
}

impl ModulePusher {
    fn push_down_modules_impl(
        &mut self,
        circuit: CircuitRc,
        get: IterativeMatcherRc,
        skip_module: IterativeMatcherRc,
        mode: PushDownMode,
    ) -> Result<CircOrGet> {
        self.push_down_modules_rec(
            circuit,
            get,
            skip_module,
            mode,
            &Default::default(),
            &Default::default(),
        )
        .map(|x| x.0)
    }

    // we could do a generic function on the bools, but I assume this has better compile time
    #[apply(cached_method)]
    #[self_id(self_)]
    #[key((
        circuit.info().hash,
        get.clone(),
        skip_module.clone(),
        mode,
        items.hash(),
        extra_replacements.hash(),
    ))]
    #[use_try]
    #[cache_expr(cache)]
    fn push_down_modules_rec(
        &mut self,
        circuit: CircuitRc,
        get: IterativeMatcherRc,
        skip_module: IterativeMatcherRc,
        mode: PushDownMode,
        items: &NestedModuleItems,
        extra_replacements: &ReplaceMapRc,
    ) -> Result<(CircOrGet, (bool, HashSet<CircuitRc>))> {
        let push_or_get = mode == PushDownMode::PushOverride || mode == PushDownMode::Get;
        if push_or_get {
            assert_eq!(extra_replacements.len(), 0);
        }
        let (new_get, found_get) = get
            .match_iterate(circuit.clone())?
            .unwrap_or_same(get.clone());

        let is_get = mode == PushDownMode::Get;
        let all_finished = new_get.all_finished();
        let new_get = new_get.clone().per_child_with_term(circuit.num_children());
        // before get_fin for borrow check reasons
        let no_child_will_get = push_or_get
            && !circuit
                .children()
                .zip(&new_get)
                .map(|(c, get_per)| self_.any_found.are_any_found(c, get_per.clone()))
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .any(|x| x);

        let mut get_fin = || self_.finalize(circuit.clone(), items, extra_replacements);
        let mut out_set = HashSet::default();

        if is_get && found_get {
            let out = get_fin()?;
            out_set.insert(out.clone());
        }

        let found_override = found_get && mode == PushDownMode::PushOverride;
        let all_done = all_finished || found_override || no_child_will_get;

        let extra_default = (true, Default::default());

        if all_done {
            let out = match (found_override, mode) {
                (_, PushDownMode::Get) => CircOrGet::Get(out_set),
                (_, PushDownMode::Circ) | (true, PushDownMode::PushOverride) => {
                    CircOrGet::Circ(get_fin()?)
                }
                (false, PushDownMode::PushOverride) => CircOrGet::Circ(circuit),
            };
            return Ok((out, extra_default));
        }

        if let Some(arg) = circuit
            .as_symbol()
            .and_then(|sym| items.flat_sym_to_arg.get(sym))
        {
            assert!(!push_or_get);
            if !self_.bind_encountered_symbols {
                bail!(PushDownModuleError::PushPastPreviouslyBoundSymbol {
                    symbol: circuit.as_symbol().unwrap().clone()
                })
            }

            return Ok((
                CircOrGet::Circ(arg.clone()),
                (true, [arg.clone()].into_iter().collect()),
            ));
        }
        if let Some(out) = extra_replacements.get(&circuit) {
            return Ok((
                CircOrGet::Circ(out.clone()),
                (true, [out.clone()].into_iter().collect()),
            ));
        }

        let (new_skip_module, found_skip_module) = skip_module
            .match_iterate(circuit.clone())?
            .unwrap_or_same(skip_module);
        let new_skip_module = new_skip_module.per_child_with_term(circuit.num_children());

        let mut rec = |self_: &mut Self, c, get, skip, items, extra_replacements: &_| match self_
            .push_down_modules_rec(c, get, skip, mode, items, extra_replacements)?
        {
            (CircOrGet::Circ(c), b) => {
                assert!(!is_get);
                Ok((Some(c), b))
            }
            (CircOrGet::Get(new_set), b) => {
                assert!(is_get);
                out_set.extend(new_set);
                Ok((None, b))
            }
        };

        let (out, prior_inner_circuits, circuit_is_from_term) = if let Some(m) = circuit
            .as_module()
            .and_then(|x| (!found_skip_module).then_some(x))
        {
            let mut new_iter = new_get.into_iter().zip(new_skip_module);
            let (spec_circuit_get, spec_circuit_skip_module) = new_iter.next().unwrap();
            let node_get_skip: Vec<_> = new_iter.collect();
            assert_eq!(node_get_skip.len(), m.num_args());

            // intentionally don't use extra_replacements per child
            let new_nodes_flat_orig = m
                .args()
                .zip(node_get_skip.clone())
                .map(|(c, (get, skip))| {
                    rec(self_, c.clone(), get, skip, items, extra_replacements).map(|x| x.0)
                })
                .collect::<Result<Vec<_>>>()?;

            let (new_nodes_flat, new_nodes_base): (Vec<_>, _) = if push_or_get {
                assert_eq!(extra_replacements.len(), 0);
                let new_nodes_base = m.args_cloned();

                // we fully substitute in the 'get' case
                // Argueably, this should push down for naming

                (
                    m.args()
                        .map(|circuit| {
                            let out = self_
                                .build_module_flat_no_call(
                                    circuit.clone(),
                                    circuit.clone(),
                                    items.flat.clone(),
                                    items.modules.clone().into_iter().rev().collect(),
                                    items.modules.len(),
                                )?
                                .substitute(None, None);
                            Ok(out)
                        })
                        .collect::<Result<_>>()?,
                    new_nodes_base,
                )
            } else {
                let new_nodes_base = m
                    .args()
                    .zip(node_get_skip)
                    // don't use rec to avoid adding nodes to out_set
                    .map(|(c, (get, skip))| {
                        let out = unwrap!(self_.push_down_modules_rec(
                            c.clone(),
                            get,
                            skip,
                            mode,
                            &Default::default(), // no items (only difference from recursive call above)
                            extra_replacements,
                        )?.0, CircOrGet::Circ);
                        assert_eq!(
                            out.ndim(),
                            c.ndim(),
                            "expected to have same ndim: \n{}\n{}",
                            out.repru(),
                            c.repru()
                        );
                        Ok(out)
                    })
                    .collect::<Result<Vec<_>>>()?;
                (
                    new_nodes_flat_orig
                        .iter()
                        .map(|x| x.clone().unwrap())
                        .collect(),
                    new_nodes_base,
                )
            };

            let (out, (termed_prior_iter, mut prior_inner_circuits)) = rec(
                self_,
                m.spec.circuit.clone(),
                spec_circuit_get,
                spec_circuit_skip_module,
                &items.push(m.clone(), new_nodes_flat.clone(), new_nodes_base)?,
                extra_replacements,
            )?;
            if mode == PushDownMode::PushOverride {
                let out = Some(
                    m.map_children_unwrap_idxs(|i| {
                        if i == 0 {
                            out.clone().unwrap()
                        } else {
                            new_nodes_flat_orig[i - 1].clone().unwrap()
                        }
                    })
                    .rc(),
                );
                (out, Default::default(), false)
            } else {
                prior_inner_circuits.insert(m.spec.circuit.clone());
                (out, prior_inner_circuits, termed_prior_iter)
            }
        } else {
            if let Some(m) = circuit.as_module() {
                for arg_spec in &m.spec.arg_specs {
                    if items.flat_sym_to_arg.contains_key(&arg_spec.symbol) {
                        // we could add support for this case, but quite annoying
                        bail!(PushDownModuleError::PushingPastModuleWhichOverridesSym {
                            symbol: arg_spec.symbol.clone(),
                            skipped_module: m.clone()
                        });
                    }
                }
            }

            let new_children = izip!(
                circuit.children(),
                new_get.clone(),
                new_skip_module.clone(),
                extra_replacements.per_child(&circuit)
            )
            .map(|(c, get, skip, rep)| rec(self_, c, get, skip, items, &rep).map(|x| x.0))
            .collect::<Result<Vec<_>>>()?;

            if is_get {
                return Ok((CircOrGet::Get(out_set), extra_default));
            }

            let out = expand_node(
                circuit.clone(),
                &new_children.into_iter().map(|x| x.unwrap()).collect(),
                &mut |c, rep, child_idx| {
                    rec(
                        self_,
                        c,
                        new_get[child_idx].clone(),
                        new_skip_module[child_idx].clone(),
                        items,
                        &extra_replacements.extend_into(rep),
                    )
                    .map(|x| x.0.unwrap())
                },
            )
            // maybe we're supposed to just panic on rearrange rank errors?
            .context(concat!(
                "expand fail in push down modules, should be rearrange ",
                "overflow error (otherwise internal error)"
            ))?;
            (Some(out), Default::default(), false)
        };

        if is_get {
            return Ok((CircOrGet::Get(out_set), extra_default));
        }
        let out = out.unwrap();

        // rename if we didn't create new
        //
        // we also special case module 'stacks' with no intermediate circuits:
        // this is what prior_spec_circuits and circuit_is_from_term are for.
        let out = if out != circuit
            && !prior_inner_circuits.contains(&out)
            // if at top of module stack, ignore from term
            && (!circuit_is_from_term || items.modules.is_empty())
        {
            let new_name = self_
                .namer
                .call(
                    circuit.clone(),
                    circuit.clone(),
                    items.modules.iter().rev().cloned().collect(),
                    None,
                )
                .context("namer failed in push down module")?;
            out.rename(new_name)
        } else {
            out
        };

        Ok((
            CircOrGet::Circ(out),
            (circuit_is_from_term, prior_inner_circuits),
        ))
    }

    fn finalize(
        &mut self,
        circuit: CircuitRc,
        items: &NestedModuleItems,
        extra_replacements: &ReplaceMapRc,
    ) -> Result<CircuitRc> {
        let finished_circuit = self
            .replace_expander
            .replace_expand_with_map(circuit.clone(), extra_replacements)?;
        let rep_expand_extra_dims = finished_circuit.ndim() - circuit.ndim();

        let mut updated_extra_replacements = (**extra_replacements).clone();
        let mut update_flat = |flat: Vec<(CircuitRc, ModuleArgSpec)>, extra_dims_here| -> Vec<_> {
            // compare to conform_to_input_batch_shape in Module
            flat.into_iter()
                .map(|(c, arg_spec)| {
                    let current_batch_shape = &c.shape()[..c.ndim() - arg_spec.symbol.ndim()];
                    let batch_start = current_batch_shape.len().saturating_sub(extra_dims_here);
                    if batch_start == current_batch_shape.len() {
                        return (c, arg_spec);
                    }

                    let new_sym = Symbol::new(
                        current_batch_shape[batch_start..]
                            .iter()
                            .chain(arg_spec.symbol.shape())
                            .copied()
                            .collect(),
                        arg_spec.symbol.uuid.clone(),
                        arg_spec.symbol.info().name,
                    );

                    let orig_sim = arg_spec.symbol.clone();

                    updated_extra_replacements.insert(orig_sim.rc(), new_sym.crc());

                    (
                        c,
                        ModuleArgSpec {
                            symbol: new_sym,
                            ..arg_spec
                        },
                    )
                })
                .collect()
        };

        if self.flatten_modules {
            let (finished_circuit, new_flat) = if rep_expand_extra_dims > 0 {
                // if we have extra dims, we have to pull the symbols and resub to sync up batching
                let new_flat = update_flat(items.flat.clone(), rep_expand_extra_dims);
                let new_finished_circuit = self.replace_expander.replace_expand_with_map(
                    circuit.clone(),
                    &ReplaceMapRc::new(updated_extra_replacements),
                )?;
                (new_finished_circuit, new_flat)
            } else {
                (finished_circuit, items.flat.clone())
            };

            self.build_module_flat(
                finished_circuit.clone(),
                finished_circuit,
                new_flat,
                items.modules.clone().into_iter().rev().collect(),
                items.modules.len(),
            )
        } else {
            let (finished_circuit, rev_nested) = if rep_expand_extra_dims > 0 {
                // if we have extra dims, we have to pull the symbols and resub to sync up batching
                let rev_nested: Vec<_> = items
                    .modules
                    .iter()
                    .zip(&items.new_nodes_base)
                    .rev()
                    .scan(0, |prev_dims_covered, (m, new_nodes)| {
                        let extra_dims_here =
                            rep_expand_extra_dims.saturating_sub(*prev_dims_covered);
                        *prev_dims_covered += m.aligned_batch_shape().len();
                        Some(update_flat(
                            new_nodes
                                .clone()
                                .into_iter()
                                .zip(m.spec.arg_specs.clone())
                                .collect(),
                            extra_dims_here,
                        ))
                    })
                    .collect();
                let new_finished_circuit = self.replace_expander.replace_expand_with_map(
                    circuit.clone(),
                    &ReplaceMapRc::new(updated_extra_replacements),
                )?;
                (new_finished_circuit, rev_nested)
            } else {
                (
                    finished_circuit,
                    items
                        .modules
                        .iter()
                        .zip(&items.new_nodes_base)
                        .rev()
                        .map(|(m, new_nodes)| {
                            new_nodes
                                .clone()
                                .into_iter()
                                .zip(m.spec.arg_specs.clone())
                                .collect()
                        })
                        .collect(),
                )
            };

            self.build_module_nested(finished_circuit, rev_nested, items.modules.clone())
        }
    }

    fn build_module_flat_no_call(
        &mut self,
        base_circuit: CircuitRc, // just for naming
        circuit: CircuitRc,
        flat: Vec<(CircuitRc, ModuleArgSpec)>,
        rev_modules: Vec<Module>,
        overall_mod_count: usize,
    ) -> Result<Module> {
        let (nodes, arg_specs) = flat.into_iter().unzip();
        let name = self
            .namer
            .call(
                base_circuit,
                circuit.clone(),
                rev_modules,
                Some(overall_mod_count),
            )
            .context("namer failed in push down module")?;
        Ok(Module::new(
            nodes,
            ModuleSpec { circuit, arg_specs },
            name.clone(),
        ))
    }

    fn build_module_flat(
        &mut self,
        base_circuit: CircuitRc, // just for naming
        circuit: CircuitRc,
        flat: Vec<(CircuitRc, ModuleArgSpec)>,
        rev_modules: Vec<Module>,
        overall_mod_count: usize,
    ) -> Result<CircuitRc> {
        let out = self.build_module_flat_no_call(
            base_circuit,
            circuit,
            flat,
            rev_modules.clone(),
            overall_mod_count,
        )?;
        self.module_construct_callback
            .call(out, rev_modules, overall_mod_count)
    }

    fn build_module_nested(
        &mut self,
        base_circuit: CircuitRc,
        rev_nested: Vec<Vec<(CircuitRc, ModuleArgSpec)>>,
        modules: Vec<Module>,
    ) -> Result<CircuitRc> {
        if modules.is_empty() {
            return Ok(base_circuit.clone());
        }
        let cum_rev_mods = modules
            .clone()
            .into_iter()
            .rev()
            .scan(vec![], |state, new_mod| {
                state.push(new_mod);
                Some(state.clone())
            });

        rev_nested.into_iter().zip(cum_rev_mods).fold(
            Ok(base_circuit.clone()),
            |circuit, (arg_items, running_modules)| {
                self.build_module_flat(
                    base_circuit.clone(),
                    circuit?,
                    arg_items,
                    running_modules,
                    modules.len(),
                )
            },
        )
    }
}

fn_struct!(pub MaybeUpdate: Fn(m : CircuitRc) -> Option<CircuitRc>);

#[derive(Debug, Clone)]
pub struct ModuleConstructUpdateBindings {
    // TODO
    cache: FastUnboundedCache<HashBytes, Option<CircuitRc>>,
    update: MaybeUpdate,
    run_update_on_new_spec_circuits: bool,
    namer: NestedModuleNamer,
    is_flatten: bool,
}

impl IntoPy<PyObject> for ModuleConstructUpdateBindings {
    fn into_py(self, _: Python<'_>) -> PyObject {
        // fix as needed
        unreachable!()
    }
}

impl ModuleConstructUpdateBindings {
    fn call(
        &mut self,
        m: Module,
        these_applied_modules: Vec<Module>,
        overall_count: usize,
    ) -> Result<CircuitRc> {
        assert!(these_applied_modules.len() <= overall_count);
        if these_applied_modules.len() != overall_count {
            return Ok(m.rc());
        }

        let m = self.deep_remove_unused(m, overall_count);

        self.bind_module(
            m,
            overall_count,
            Default::default(),
            &these_applied_modules,
            overall_count,
        )
        .map(|(c, _, _)| c)
    }

    fn deep_remove_unused(&self, m: Module, current_count: usize) -> Module {
        let m = if current_count > 1 && !self.is_flatten {
            let new_spec_circuit = self
                .deep_remove_unused(m.spec.circuit.as_module_unwrap().clone(), current_count - 1)
                .rc();
            let name = m.info().name;
            Module::new(
                m.args_cloned(),
                ModuleSpec {
                    circuit: new_spec_circuit,
                    arg_specs: m.spec.arg_specs,
                },
                name,
            )
        } else {
            m
        };

        module_strip_args(&m)
    }

    fn bind_module(
        &mut self,
        m: Module,
        count_mods: usize,
        outer_updated_syms: IndexSet<Symbol>,
        applied_modules: &[Module],
        overall_count: usize,
    ) -> Result<(CircuitRc, CircuitRc, Vec<Module>)> {
        assert_eq!(applied_modules.len(), count_mods);
        let m = if (self.is_flatten || count_mods == 1) && self.run_update_on_new_spec_circuits {
            if let Some(new_spec) = self.deep_update(m.spec.circuit.clone())? {
                Module::try_new(
                    m.args_cloned(),
                    ModuleSpec {
                        circuit: new_spec,
                        arg_specs: m.spec.arg_specs.clone(),
                    },
                    m.info().name,
                )
                .context(
                    "module construction failed in update bindings after updating spec circuit",
                )?
            } else {
                m
            }
        } else {
            m
        };

        // this assumes that unused args of children were already removed
        let used_inputs = m.spec.are_args_used();
        let (nodes, arg_specs): (Vec<_>, Vec<_>) = m
            .arg_items()
            .into_iter()
            .zip(used_inputs)
            .map(|((node, arg_spec), used)| {
                if used {
                    self.deep_update(node.clone()).map(|x| {
                        x.or_else(|| {
                            is_intersecting_free_syms(&node, &outer_updated_syms)
                                .then(|| node.clone())
                        })
                        .map(|x| (x, arg_spec))
                    })
                } else {
                    Ok(None)
                }
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .filter_map(|x| x)
            .unzip();

        let (new_spec_circuit, base_circuit, prior_modules) = if count_mods > 1 && !self.is_flatten
        {
            self.bind_module(
                m.spec.circuit.as_module_unwrap().clone(),
                count_mods - 1,
                outer_updated_syms
                    .into_iter()
                    .chain(arg_specs.iter().map(|x| x.symbol.clone()))
                    .collect(),
                &applied_modules[..applied_modules.len() - 1],
                overall_count,
            )?
        } else {
            (m.spec.circuit.clone(), m.spec.circuit.clone(), Vec::new())
        };

        let new_modules = if self.is_flatten {
            applied_modules.to_vec()
        } else {
            let mut new_modules = prior_modules.clone();
            new_modules.push(applied_modules.last().unwrap().clone());
            new_modules
        };
        let name = self.namer.call(
            base_circuit.clone(),
            new_spec_circuit.clone(),
            new_modules.clone(),
            Some(overall_count),
        )?;

        let m_out = Module::try_new(
            nodes,
            ModuleSpec {
                circuit: new_spec_circuit,
                arg_specs,
            },
            name,
        )
        .context("module construction failed in update bindings after updating nodes")?;

        if m_out.aligned_batch_shape().len() > 0 {
            bail!("batching update bindings case not supported, consider doing some conforming (TODO: better err)");
        }

        let (new_circ, prior_mods) = elim_no_input_module(&m_out)
            .map(|x| (x, prior_modules))
            .unwrap_or_else(|| (m_out.rc(), new_modules));
        Ok((new_circ, base_circuit, prior_mods))
    }

    #[apply(cached_method)]
    #[self_id(self_)]
    #[key(circuit.info().hash)]
    #[use_try]
    #[cache_expr(cache)]
    fn deep_update(&mut self, circuit: CircuitRc) -> Result<Option<CircuitRc>> {
        // somewhat different ordering than deep_map_op
        let out = if let Some(new) = self_.update.call(circuit.clone())? {
            Some(new)
        } else {
            let new_children = circuit
                .children()
                .map(|c| self_.deep_update(c))
                .collect::<Result<Vec<_>>>()?;
            new_children.iter().any(|x| x.is_some()).then(|| {
                circuit.map_children_unwrap_enumerate(|i, c| new_children[i].clone().unwrap_or(c))
            })
        };
        Ok(out)
    }
}

#[pyfunction]
#[pyo3(signature=(
    bind_name = "upd_bind".to_owned(),
    short_if_not_leaf = true,
    keep_name_if_not_leaf = false
))]
pub fn default_update_bindings_nested_namer(
    bind_name: String,
    short_if_not_leaf: bool,
    keep_name_if_not_leaf: bool,
) -> NestedModuleNamer {
    let default_overall_namer = default_nested_module_namer(bind_name.clone());
    NestedModuleNamer::Dyn(NestedModuleNamerDynStruct(Arc::new(
        move |base_circuit, running_circuit, modules, pushed_overall_mod_count| {
            if pushed_overall_mod_count.is_none() && keep_name_if_not_leaf {
                Ok(base_circuit.info().name)
            } else if pushed_overall_mod_count.is_none() && short_if_not_leaf {
                Ok(base_circuit
                    .info()
                    .name
                    .map(|name| format!("{name} {bind_name}").into()))
            } else {
                default_overall_namer.call(
                    base_circuit,
                    running_circuit,
                    modules,
                    pushed_overall_mod_count,
                )
            }
        },
    )))
}

#[pyfunction]
#[pyo3(signature=(
    circuit,
    update,
    matcher,
    namer = default_update_bindings_nested_namer("upd_bind".to_owned(), true, false),
    skip_module = MatcherData::Always(false).into(),
    run_update_on_new_spec_circuits = false,
    flatten_modules = false
))]
pub fn update_bindings_nested(
    circuit: CircuitRc,
    update: MaybeUpdate,
    matcher: IterativeMatcherRc,
    namer: NestedModuleNamer,
    skip_module: IterativeMatcherRc,
    run_update_on_new_spec_circuits: bool,
    flatten_modules: bool,
) -> Result<CircuitRc> {
    let module_construct_callback =
        ModuleConstructCallback::UpdateBindings(ModuleConstructUpdateBindings {
            cache: Default::default(),
            update,
            run_update_on_new_spec_circuits,
            namer: namer.clone(),
            is_flatten: flatten_modules,
        });

    ModulePusher {
        flatten_modules,
        module_construct_callback,
        namer,
        ..Default::default()
    }
    .push_down_modules_impl(circuit, matcher, skip_module, PushDownMode::PushOverride)
    .map(|x| unwrap!(x, CircOrGet::Circ))
}

struct SymbolExtractor {
    cache: FastUnboundedCache<
        (HashBytes, HashBytes, IterativeMatcherRc),
        (CircuitRc, BTreeMap<Symbol, (CircuitRc, ModuleArgSpec)>),
    >,
    conform_batch_if_needed: bool,
}

impl Default for SymbolExtractor {
    fn default() -> Self {
        Self {
            cache: FastUnboundedCache::default(),
            conform_batch_if_needed: false,
        }
    }
}

impl SymbolExtractor {
    #[apply(cached_method)]
    #[self_id(self_)]
    #[key((circ.info().hash, symbols.hash(), traversal.clone()))]
    #[use_try]
    #[cache_expr(cache)]
    fn extract_symbols_rec(
        &mut self,
        circ: CircuitRc,
        symbols: &SymbolSetRc,
        traversal: IterativeMatcherRc,
    ) -> Result<(CircuitRc, BTreeMap<Symbol, (CircuitRc, ModuleArgSpec)>)> {
        let updated = traversal
            .match_iterate(circ.clone())?
            .unwrap_or_same(traversal)
            .0;

        if updated.all_finished() {
            return Ok((circ, Default::default()));
        }

        let traversal_per_child = updated.per_child_with_term(circ.num_children());

        let mut out_syms = BTreeMap::default();
        let add_sym_circ = |out_syms: &mut BTreeMap<_, (CircuitRc, ModuleArgSpec)>,
                            arg_spec: ModuleArgSpec,
                            circ| {
            match out_syms.entry(arg_spec.symbol.clone()) {
                Entry::Occupied(entry) => {
                    if &entry.get().0 != &circ {
                        bail!(ExtractSymbolsError::BoundInputInconsistent {
                            symbol: arg_spec.symbol,
                            old_bound: entry.get().0.clone(),
                            new_bound: circ
                        })
                    }
                    if &entry.get().1 != &arg_spec {
                        bail!(ExtractSymbolsError::ArgSpecInconsistent {
                            old_arg_spec: entry.get().1.clone(),
                            new_arg_spec: arg_spec
                        })
                    }
                }
                Entry::Vacant(entry) => {
                    entry.insert((circ, arg_spec));
                }
            }
            Ok(())
        };

        let mut freshly_bound_syms = HashSet::default(); // so we extract outer -> inner (left -> right)
        let (circ, traversal_per_child) = if let Some(m) = circ.as_module() {
            let new_mod = if self_.conform_batch_if_needed {
                let batch = m
                    .args()
                    .zip(&m.spec.arg_specs)
                    .filter_map(|(node, arg_spec)| {
                        assert!(node.ndim() >= arg_spec.symbol.ndim());
                        symbols
                            .contains(&arg_spec.symbol)
                            .then_some(node.ndim() - arg_spec.symbol.ndim())
                    })
                    .max();
                batch.map(|dims| m.conform_to_input_batch_shape(Some(dims)).unwrap())
            } else {
                None
            };
            let new_m = new_mod.as_ref().unwrap_or(m);
            let mut traversals_iter = traversal_per_child.into_iter();
            let spec_traversal = traversals_iter.next().unwrap();
            let iter = new_m
                .arg_items()
                .into_iter()
                .zip(&m.spec.arg_specs)
                .zip(traversals_iter)
                .filter_map(|(((node, arg_spec), orig_arg_spec), traversal)| {
                    if !symbols.contains(&orig_arg_spec.symbol)
                        || freshly_bound_syms.contains(&orig_arg_spec.symbol)
                    {
                        return Some(Ok((node, arg_spec, traversal)));
                    }
                    if node.ndim() > arg_spec.symbol.ndim() {
                        assert!(!self_.conform_batch_if_needed); // this should have been handled
                        return Some(Err(ExtractSymbolsError::BatchedInput {
                            node_ndim: node.ndim(),
                            symbol_ndim: arg_spec.symbol.ndim(),
                            symbol: arg_spec.symbol.clone(),
                            node,
                        }
                        .into()));
                    }
                    if let Err(e) = add_sym_circ(&mut out_syms, arg_spec, node) {
                        return Some(Err(e));
                    }
                    freshly_bound_syms.insert(orig_arg_spec.symbol.clone());
                    None
                })
                .collect::<Result<Vec<_>>>()?;
            let (nodes, arg_specs, traversal_per_node): (Vec<_>, Vec<_>, Vec<_>) = multiunzip(iter);
            let circ = if nodes.len() == m.num_args() {
                m.crc()
            } else {
                Module::nrc(
                    nodes,
                    ModuleSpec {
                        circuit: new_m.spec.circuit.clone(),
                        arg_specs,
                    },
                    m.info().name.map(|s| format!("{} extracted", s).into()),
                )
            };
            let traversal_per_child = iter::once(spec_traversal)
                .chain(traversal_per_node)
                .collect();
            (circ, traversal_per_child)
        } else {
            (circ, traversal_per_child)
        };

        let per_child_syms = if !freshly_bound_syms.is_empty() {
            let mut removed = (**symbols).clone();
            for bound in &freshly_bound_syms {
                removed.remove(bound);
            }

            iter::once(SymbolSetRc::new(removed))
                .chain(iter::repeat(symbols).cloned())
                .take(circ.num_children())
                .collect()
        } else {
            vec![symbols.clone(); circ.num_children()]
        };

        assert_eq!(per_child_syms.len(), circ.num_children());
        assert_eq!(traversal_per_child.len(), circ.num_children());
        let new_circ = circ.map_children_enumerate(|i, c| {
            let (new, map) =
                self_.extract_symbols_rec(c, &per_child_syms[i], traversal_per_child[i].clone())?;
            for (_, (circ, arg_spec)) in map {
                add_sym_circ(&mut out_syms, arg_spec, circ)?;
            }
            Ok(new)
        })?;

        if let Some(m) = circ.as_module() {
            for (sym, (bound_input, _)) in &out_syms {
                let used_args = are_args_used(bound_input, &m.spec.arg_specs);
                if used_args.iter().any(|x| *x) {
                    bail!(ExtractSymbolsError::HasBindingsFromOuterModule {
                        bound_input: bound_input.clone(),
                        symbol: sym.clone(),
                        free_outer_symbols: m
                            .spec
                            .arg_specs
                            .iter()
                            .zip(used_args)
                            .filter_map(|(arg_spec, used)| used.then(|| arg_spec.symbol.clone()))
                            .collect(),
                    })
                }
            }
        }

        Ok((new_circ, out_syms))
    }
}

#[pyfunction]
#[pyo3(signature=(
    circuit,
    symbols,
    use_elim_no_input_modules = true,
    conform_batch_if_needed = false,
    traversal = IterativeMatcher::noop_traversal().rc()
))]
pub fn extract_symbols(
    circuit: CircuitRc,
    symbols: SymbolSetRc,
    use_elim_no_input_modules: bool,
    conform_batch_if_needed: bool,
    traversal: IterativeMatcherRc,
) -> Result<Module> {
    let name = circuit.info().name;
    let (extracted_circ, extracted_syms) = SymbolExtractor {
        conform_batch_if_needed,
        ..Default::default()
    }
    .extract_symbols_rec(circuit, &symbols, traversal)?;

    let extracted_circ = if use_elim_no_input_modules {
        SimpFnSubset::none()
            .include(vec!["elim_no_input_module".to_owned()])
            .unwrap()
            .simp(extracted_circ)
    } else {
        extracted_circ
    };

    let (nodes, arg_specs) = extracted_syms.into_values().unzip();

    Ok(Module::new(
        nodes,
        ModuleSpec {
            circuit: extracted_circ,
            arg_specs,
        },
        name,
    ))
}

#[pyfunction]
#[pyo3(signature=(
    circuit,
    get,
    use_elim_no_input_modules = true,
    conform_batch_if_needed = false,
    traversal = IterativeMatcher::noop_traversal().rc()
))]
pub fn extract_symbols_get(
    circuit: CircuitRc,
    get: IterativeMatcherRc,
    use_elim_no_input_modules: bool,
    conform_batch_if_needed: bool,
    traversal: IterativeMatcherRc,
) -> Result<Module> {
    let syms = get
        .get(circuit.clone(), false)
        .context("get failed in extract symbols")?
        .into_iter()
        .map(|x| {
            x.as_symbol().cloned().ok_or_else(|| {
                ExtractSymbolsError::GetFoundNonSymbol {
                    circ: x,
                    get: (**get).clone(),
                }
                .into()
            })
        })
        .collect::<Result<_>>()?;
    extract_symbols(
        circuit,
        SymbolSetRc::new(syms),
        use_elim_no_input_modules,
        conform_batch_if_needed,
        traversal,
    )
}

const PREFIX_PAST: &str = "(Not currently supported) the encountered";
const MIDDLE_PAST: &str = "was bound by a pushed module, but the";

#[apply(python_error_exception)]
#[base_error_name(PushDownModule)]
#[base_exception(PyValueError)]
#[derive(Error, Debug, Clone)]
pub enum PushDownModuleError {
    #[error("the encountered symbol={symbol:?} was bound by a pushed module, but bind_encountered_symbols is false ({e_name})")]
    PushPastPreviouslyBoundSymbol { symbol: Symbol },

    #[error("{PREFIX_PAST} symbol={symbol:?} {MIDDLE_PAST} skipped_module={skipped_module:?} overrides this binding ({e_name})")]
    PushingPastModuleWhichOverridesSym {
        symbol: Symbol,
        skipped_module: Module,
    },
}

#[apply(python_error_exception)]
#[base_error_name(ExtractSymbols)]
#[base_exception(PyValueError)]
#[derive(Error, Debug, Clone)]
pub enum ExtractSymbolsError {
    #[error("non-symbol={circ:?} (get={get}) ({e_name})")]
    GetFoundNonSymbol {
        circ: CircuitRc,
        get: IterativeMatcher,
    },

    #[error("node_ndim={node_ndim}>symbol_ndim={symbol_ndim} which implies batching\n{}\nsymbol={symbol:?} node={node:?} ({e_name})",
        concat!("batched inputs aren't handled by default (at the momement) but ",
        "can be handled by conforming the module to the batch shape.",
        "\nIf you want this to be done automatically, pass conform_batch_if_needed=true.")
        )]
    BatchedInput {
        node_ndim: usize,
        symbol_ndim: usize,
        symbol: Symbol,
        node: CircuitRc,
    },

    #[error("old_arg_spec={old_arg_spec:?} != new_arg_spec={new_arg_spec:?} ({e_name})")]
    ArgSpecInconsistent {
        old_arg_spec: ModuleArgSpec,
        new_arg_spec: ModuleArgSpec,
    },

    #[error("for symbol={symbol:?} old_bound={old_bound:?} != new_bound={new_bound:?} ({e_name})")]
    BoundInputInconsistent {
        symbol: Symbol,
        old_bound: CircuitRc,
        new_bound: CircuitRc,
    },

    #[error("the bound_input={bound_input:?} (for symbol={symbol:?}) contains free_outer_symbols={free_outer_symbols:?} which is bound by an outer module ({e_name})")]
    HasBindingsFromOuterModule {
        bound_input: CircuitRc,
        symbol: Symbol,
        free_outer_symbols: Vec<Symbol>,
    },
}
