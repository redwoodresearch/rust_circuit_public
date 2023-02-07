use std::{
    collections::BTreeSet,
    hash::{Hash, Hasher},
    iter::{once, zip},
    ops::Deref,
    sync::Arc,
};

use anyhow::{bail, Context, Result};
use macro_rules_attribute::apply;
use pyo3::{exceptions::PyValueError, prelude::*, pyclass::CompareOp};
use rr_util::{
    cached_lambda, cached_method,
    caching::FastUnboundedCache,
    eq_by_big_hash, impl_eq_by_big_hash,
    name::Name,
    py_types::{use_rust_comp, PyCallable},
    pycall, python_error_exception,
    rearrange_spec::RearrangeSpec,
    symbolic_size::SymbolicSizeProduct,
    tensor_util::{right_align_shapes, Shape},
    util::{arc_unwrap_or_clone, counts_g_1, is_unique},
    IndexSet,
};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet, FxHasher};
use thiserror::Error;
use uuid::{uuid, Uuid};

use crate::{
    circuit_node_auto_impl, circuit_node_extra_impl,
    circuit_node_private::{CircuitNodeHashItems, CircuitNodeSetNonHashInfo},
    circuit_utils::OperatorPriority,
    deep_map, deep_map_op, deep_map_op_context_preorder_stoppable, deep_map_pre_new_children,
    deep_map_preorder_unwrap,
    expand_node::{ExpandError, ReplaceExpander, ReplaceMapRc},
    named_axes::{deep_strip_axis_names, propagate_named_axes},
    new_rc_unwrap,
    prelude::*,
    visit_circuit_unwrap, CachedCircuitInfo, HashBytes, PyCircuitBase, Rearrange, Symbol,
    TensorEvalError,
};

/// can also be thought of as lambda from lambda calculus (expression with free variables + list of these variables)
#[pyclass]
#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct ModuleSpec {
    #[pyo3(get, set)]
    pub circuit: CircuitRc,
    #[pyo3(get, set)]
    pub arg_specs: Vec<ModuleArgSpec>,
}

type ModuleSpecHashable = (HashBytes, Vec<ModuleArgSpecHashable>);

pub fn are_args_used<'a, I: IntoIterator<Item = &'a ModuleArgSpec>>(
    circuit: &CircuitRc,
    arg_specs: I,
) -> Vec<bool>
where
    <I as IntoIterator>::IntoIter: DoubleEndedIterator,
{
    // right most binding has precedence, prior bindings unused
    let mut out: Vec<_> = arg_specs
        .into_iter()
        .rev()
        .scan(HashSet::default(), |prior_syms, spec| {
            let out = (!prior_syms.contains(&spec.symbol)) && has_free_sym(&circuit, &spec.symbol);
            prior_syms.insert(&spec.symbol);
            Some(out)
        })
        .collect();
    out.reverse();
    out
}

#[derive(Clone, Debug)]
pub struct SymbolSetRc {
    set: Arc<BTreeSet<Symbol>>,
    hash: HashBytes,
}

impl<'source> pyo3::FromPyObject<'source> for SymbolSetRc {
    fn extract(from_py_obj: &'source pyo3::PyAny) -> pyo3::PyResult<Self> {
        let from: BTreeSet<Symbol> = from_py_obj.extract()?;
        Ok(Self::new(from))
    }
}

impl Default for SymbolSetRc {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl Deref for SymbolSetRc {
    type Target = BTreeSet<Symbol>;

    fn deref(&self) -> &Self::Target {
        &self.set
    }
}

impl SymbolSetRc {
    pub fn new(set: BTreeSet<Symbol>) -> Self {
        let mut hasher = blake3::Hasher::new();
        for v in set.iter() {
            hasher.update(&v.info().hash);
        }
        Self {
            set: Arc::new(set),
            hash: hasher.finalize().into(),
        }
    }

    pub fn into_map(self) -> <Self as Deref>::Target {
        arc_unwrap_or_clone(self.set)
    }
}

impl eq_by_big_hash::EqByBigHash for SymbolSetRc {
    fn hash(&self) -> HashBytes {
        self.hash
    }
}
impl_eq_by_big_hash!(SymbolSetRc);

impl ModuleSpec {
    pub const EXPAND_PLACEHOLDER_UUID: Uuid = uuid!("741ba404-eec3-4ac9-b6ce-062e903fb033");

    pub fn get_hashable(&self) -> ModuleSpecHashable {
        (
            self.circuit.info().hash,
            self.arg_specs.iter().map(|x| x.get_hashable()).collect(),
        )
    }

    pub fn get_spec_circuit_uuid(&self) -> Uuid {
        let x: [_; 16] = self.circuit.info().hash[..16].try_into().unwrap();
        Uuid::from_bytes(x)
    }

    pub fn batch_shapes<'a>(&self, nodes: &'a [CircuitRc]) -> Vec<&'a [usize]> {
        zip(self.arg_specs.iter(), nodes)
            .map(|(arg_spec, node)| &node.shape()[..node.ndim() - arg_spec.symbol.ndim()])
            .collect()
    }

    pub fn aligned_batch_shape(&self, nodes: &[CircuitRc]) -> Result<Shape> {
        right_align_shapes(&self.batch_shapes(nodes))
    }

    pub fn expand_raw(&self, nodes: &Vec<CircuitRc>) -> Result<CircuitRc> {
        if self.arg_specs.len() != nodes.len() {
            bail!(ConstructError::ModuleWrongNumberChildren {
                expected: self.arg_specs.len(),
                got: nodes.len(),
                arg_specs: self.arg_specs.clone(),
                nodes: nodes.clone(),
            });
        }
        for (arg_spec, node) in zip(self.arg_specs.iter(), nodes) {
            if node.info().rank() < arg_spec.symbol.info().rank() {
                bail!(ExpandError::ModuleRankReduced {
                    node_rank: node.rank(),
                    symbol_rank: arg_spec.symbol.rank(),
                    arg_spec: arg_spec.clone(),
                    node_shape: node.shape().clone(),
                    spec_circuit: self.circuit.clone()
                });
            }
            if !arg_spec.batchable && node.info().rank() > arg_spec.symbol.info().rank() {
                bail!(ExpandError::ModuleTriedToBatchUnbatchableInput {
                    node_rank: node.rank(),
                    symbol_rank: arg_spec.symbol.rank(),
                    arg_spec: arg_spec.clone(),
                    spec_circuit: self.circuit.clone()
                });
            }
            if !arg_spec.expandable
                && node.info().shape[node.info().rank() - arg_spec.symbol.info().rank()..]
                    != arg_spec.symbol.info().shape[..]
            {
                bail!(ExpandError::ModuleTriedToExpandUnexpandableInput {
                    node_shape: node.shape().clone(),
                    symbol_shape: arg_spec.symbol.shape().clone(),
                    arg_spec: arg_spec.clone(),
                    spec_circuit: self.circuit.clone()
                });
            }
            if arg_spec.ban_non_symbolic_size_expand {
                for (dim, (&new_size, &old_size)) in node.shape()
                    [node.rank() - arg_spec.symbol.rank()..]
                    .iter()
                    .zip(arg_spec.symbol.shape())
                    .enumerate()
                {
                    if new_size != old_size && !SymbolicSizeProduct::has_symbolic(old_size) {
                        bail!(ExpandError::ModuleTriedToExpandOnNonSymbolicSizeAndBanned {
                            old_size,
                            new_size,
                            dim,
                            node_shape: node.shape().clone(),
                            arg_spec: arg_spec.clone(),
                            spec_circuit: self.circuit.clone()
                        });
                    }
                }
            }
        }

        // TODO: maybe we should allow for inconsistent symbolic batch shapes?
        let aligned_batch_shape = self
            .aligned_batch_shape(nodes)
            .context("batch shapes didn't match for module")?;

        // TODO: fix this being uncached!
        let out = ReplaceExpander::new_noop()
            .replace_expand_with_map(
                self.circuit.clone(),
                &ReplaceMapRc::new(
                    self.arg_specs
                        .iter()
                        .zip(nodes)
                        .map(|(arg_spec, node)| (arg_spec.symbol.crc(), node.clone()))
                        .collect(),
                ),
            )
            .context("replace expand failed from substitute")?;

        assert!(out.ndim() >= self.circuit.ndim());
        let out_batch_rank = out.ndim() - self.circuit.ndim();
        assert!(out_batch_rank <= aligned_batch_shape.len());
        right_align_shapes(&[&out.shape()[..out_batch_rank], &aligned_batch_shape])
            .expect("output shape should be right aligned subset of batch shape");

        let out = if out_batch_rank < aligned_batch_shape.len() {
            let spec = RearrangeSpec::prepend_batch_shape(
                aligned_batch_shape[..(aligned_batch_shape.len() - out_batch_rank)].into(),
                out.ndim(),
            )
            .context("rank overflow in module expand")?;

            let rep_name: Option<Name> = out.info().name.map(|x| format!("{} rep_batch", x).into());
            Rearrange::nrc(out, spec, rep_name)
        } else {
            out
        };
        Ok(out)
    }

    pub fn expand_shape(&self, shapes: &Vec<Shape>) -> Result<(CircuitRc, Vec<HashBytes>, String)> {
        let key = (self.get_hashable(), shapes.clone());
        if let Some(result) = MODULE_EXPANSIONS_SHAPE.with(|cache| {
            let borrowed = cache.borrow();
            if let Some((w, shapes, n)) = borrowed.get(&key) {
                return Some((w.clone(), shapes.clone(), n.clone()));
            }
            None
        }) {
            return Ok(result);
        }
        let uuid = format!("{}", Uuid::new_v4());
        let symbols = shapes
            .iter()
            .enumerate()
            .map(|(i, s)| {
                Symbol::nrc(
                    s.clone(),
                    self.get_spec_circuit_uuid(),
                    Some(
                        format!(
                            "{}_internal_expand_shape_{}_arg{}",
                            uuid,
                            Name::str_maybe_empty(self.circuit.info().name),
                            i
                        )
                        .into(),
                    ),
                )
            })
            .collect();
        let result = self.expand_raw(&symbols)?;
        let out_bytes: Vec<_> = symbols.into_iter().map(|x| x.info().hash).collect();
        MODULE_EXPANSIONS_SHAPE.with(|cache| {
            cache.borrow_mut().insert(
                (self.get_hashable(), shapes.clone()),
                (result.clone(), out_bytes.clone(), uuid.clone()),
            )
        });
        Ok((result, out_bytes, uuid))
    }

    /// we could check this on module spec construct if we wanted to
    pub fn check_no_ident_matching_issues(&self) -> Result<()> {
        let my_idents: HashMap<_, _> = self
            .arg_specs
            .iter()
            .map(|arg_spec| (arg_spec.symbol.ident(), &arg_spec.symbol))
            .collect();
        let sym_single_map = self
            .circuit
            .as_symbol()
            .map(|x| [x.clone()].into_iter().collect());
        let syms = sym_single_map
            .as_ref()
            .unwrap_or(self.circuit.info().get_raw_free_symbols());

        for sym_here in syms {
            if let Some(&my_sym) = my_idents.get(&sym_here.ident()) {
                if my_sym != sym_here {
                    bail!(
                        SubstitutionError::FoundNEQFreeSymbolWithSameIdentification {
                            sym: my_sym.clone(),
                            matching_sym: sym_here.clone()
                        }
                    )
                }
            }
        }

        Ok(())
    }

    pub fn check_no_free_bound_by_nested(&self, nodes: &[CircuitRc]) -> Result<()> {
        let key: (_, Vec<_>) = (
            self.get_hashable(),
            nodes.iter().map(|x| x.info().hash).collect(),
        );
        if NO_FREE_BOUND_BY_NESTED.with(|cache| cache.borrow().contains(&key)) {
            return Ok(());
        }
        recur_check_no_free_bound_by_nested(
            self.circuit.clone(),
            SymsNodesInfo::new(
                &self.arg_specs.iter().map(|x| &x.symbol).collect::<Vec<_>>(),
                &nodes.iter().collect::<Vec<_>>(),
            ),
            &mut Default::default(),
        )?;
        NO_FREE_BOUND_BY_NESTED.with(|cache| cache.borrow_mut().insert(key));
        Ok(())
    }

    pub fn substitute(
        &self,
        nodes: &[CircuitRc],
        name_prefix: Option<String>,
        name_suffix: Option<String>,
    ) -> Result<CircuitRc> {
        let key: (_, Vec<HashBytes>, Option<String>, _) = (
            self.get_hashable(),
            nodes.iter().map(|x| x.info().hash).collect(),
            name_prefix.clone(),
            name_suffix.clone(),
        );

        if let Some(result) = MODULE_EXPANSIONS.with(|cache| {
            let borrowed = cache.borrow();
            if let Some(w) = borrowed.get(&key) {
                return Some(w.clone());
                // if let Some(w) = w.upgrade() {
                //     return Some(CircuitRc(w));
                // } else {
                //     drop(borrowed);
                //     cache.borrow_mut().remove(&key);
                // }
            }
            None
        }) {
            return Ok(result);
        }

        self.check_no_ident_matching_issues()?;
        self.check_no_free_bound_by_nested(nodes)?;

        let shapes = nodes.iter().map(|x| x.info().shape.clone()).collect();
        // TODO: maybe rep_hashes should really just be syms?
        let (expanded_shape, rep_hashes, clear_names_str) = self.expand_shape(&shapes)?;
        let node_mapping: HashMap<_, _> = rep_hashes.into_iter().zip(nodes).collect();
        // first update naming
        let result = if name_prefix.is_some() || name_suffix.is_some() {
            deep_map_op(expanded_shape.clone(), |x| {
                // we only update naming if the given node has a bound free symbol
                // (NOTE: these free symbols are now the *expanded* symbols from expand_shape)
                if let Some(n) = x.info().name && !x.is_symbol()
                    && x.info()
                        .get_raw_free_symbols()
                        .iter()
                        .any(|x| node_mapping.contains_key(&x.info().hash))
                {
                    let name = Some(
                        (name_prefix.as_deref().unwrap_or("").to_owned()
                            + n.into()
                            + name_suffix.as_deref().unwrap_or(""))
                        .into(),
                    );
                    return Some(x.rename(name));
                }
                None
            })
            .unwrap_or(expanded_shape)
        } else {
            expanded_shape
        };
        let result = deep_map(result, |x| {
            if let Some(r) = node_mapping.get(&x.info().hash) {
                Ok((*r).clone())
            } else if let Some(n) = x.info().name && n.contains(&clear_names_str) {
                // internal name + fixup stuff is necessary to be able to cache expansion shape
                let name = x.get_autoname().unwrap_or(None);
                Ok(x.rename(name))
            } else {
                Ok(x)
            }
        })
        .with_context(|| {
            format!(
                concat!(
                    "replacing symbols with nodes",
                    " failed in ModuleSpec substitute\n",
                    "module_spec={:?}\nnodes={:?}"
                ),
                self, nodes
            )
        })?;
        MODULE_EXPANSIONS.with(|cache| {
            cache.borrow_mut().insert(
                key,
                result.clone(), // Arc::downgrade(&result.0)
            )
        });
        Ok(result)
    }

    pub fn substitute_with_mod_name(
        &self,
        nodes: &[CircuitRc],
        name_prefix: Option<String>,
        name_suffix: Option<String>,
        mod_name: Option<Name>,
    ) -> Result<CircuitRc> {
        let new_circ = self.substitute(nodes, name_prefix, name_suffix)?;
        let out = if new_circ != self.circuit && self.circuit.num_children() > 0 {
            new_circ.rename(mod_name)
        } else {
            new_circ
        };
        Ok(out)
    }

    pub fn compute_non_children_hash(&self, hasher: &mut blake3::Hasher) {
        for arg_spec in &self.arg_specs {
            // this is fine because each item is fixed size and we delimit with node hashs (which are uu)
            hasher.update(&[
                arg_spec.batchable as u8,
                arg_spec.expandable as u8,
                arg_spec.ban_non_symbolic_size_expand as u8,
            ]);
            hasher.update(&arg_spec.symbol.info().hash);
        }
    }

    pub fn map_circuit<F>(&self, mut f: F) -> Result<Self>
    where
        F: FnMut(CircuitRc) -> Result<CircuitRc>,
    {
        Ok(ModuleSpec {
            circuit: f(self.circuit.clone())?,
            arg_specs: self.arg_specs.clone(),
        })
    }
    pub fn map_circuit_unwrap<F>(&self, mut f: F) -> Self
    where
        F: FnMut(CircuitRc) -> CircuitRc,
    {
        ModuleSpec {
            circuit: f(self.circuit.clone()),
            arg_specs: self.arg_specs.clone(),
        }
    }

    // we could probably make this faster...
    pub fn map_on_replaced_path<F>(&self, f: F) -> Result<Self>
    where
        F: Fn(CircuitRc) -> Result<CircuitRc>,
    {
        #[apply(cached_lambda)]
        #[key((circ.info().hash, symbols.hash), (HashBytes, HashBytes))]
        #[use_try]
        // Some iff some child bound
        fn recurse(circ: CircuitRc, symbols: &SymbolSetRc) -> Result<Option<CircuitRc>> {
            if let Some(sym) = circ.as_symbol() {
                if symbols.contains(sym) {
                    return Ok(Some(sym.crc()));
                }
            }
            // TODO: maybe extract method...
            let new = if let Some(m) = circ.as_module() {
                let mut removed: BTreeSet<_> = symbols
                    .iter()
                    .filter(|x| m.info().get_raw_free_symbols().contains(*x))
                    .cloned()
                    .collect();
                for bound in &m.spec.arg_specs {
                    removed.remove(&bound.symbol);
                }

                std::iter::once(recurse(m.spec.circuit.clone(), &SymbolSetRc::new(removed)))
                    .chain(m.args().map(|x| recurse(x.clone(), symbols)))
                    .collect::<Result<Vec<_>>>()
            } else {
                circ.children().map(|x| recurse(x, symbols)).collect()
            };
            let new = new?;
            assert_eq!(new.len(), circ.num_children());
            // note that bound symbol children are always returned as Some (see above)
            if new.iter().all(|x| x.is_none()) {
                return Ok(None);
            }
            let new_circ = circ.map_children_unwrap_enumerate(|i, c| new[i].clone().unwrap_or(c));

            f(new_circ).map(Some)
        }
        let symbols = SymbolSetRc::new(self.arg_specs.iter().map(|x| x.symbol.clone()).collect());
        self.map_circuit(|circ| recurse(circ.clone(), &symbols).map(|x| x.unwrap_or(circ)))
    }

    pub fn map_on_replaced_path_unwrap<F>(&self, f: F) -> Self
    where
        F: Fn(CircuitRc) -> CircuitRc,
    {
        self.map_on_replaced_path(|x| Ok(f(x))).unwrap()
    }
}

pub enum FreeSymbolsWrapper<'a> {
    Sym(IndexSet<Symbol>),
    Other(&'a IndexSet<Symbol>),
}

impl<'a> Deref for FreeSymbolsWrapper<'a> {
    type Target = IndexSet<Symbol>;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Sym(x) => x,
            Self::Other(x) => x,
        }
    }
}

pub fn get_free_symbols(circuit: &Circuit) -> FreeSymbolsWrapper<'_> {
    // returns syms in circuit visit order (order maybe matters here).
    // The fact the ordering matters here is why we use indexmap in free_symbols
    if let Some(sym) = circuit.as_symbol() {
        FreeSymbolsWrapper::Sym([sym.clone()].into_iter().collect())
    } else {
        FreeSymbolsWrapper::Other(&circuit.info().get_raw_free_symbols())
    }
}

#[pyfunction]
#[pyo3(name = "get_free_symbols")]
pub fn py_get_free_symbols(circuit: CircuitRc) -> Vec<Symbol> {
    get_free_symbols(&circuit).iter().cloned().collect()
}

#[pymethods]
impl ModuleSpec {
    #[new]
    #[pyo3(signature=(circuit, arg_specs, check_all_inputs_used = true, check_unique_arg_names = true))]
    pub fn new(
        circuit: CircuitRc,
        arg_specs: Vec<ModuleArgSpec>,
        check_all_inputs_used: bool,
        check_unique_arg_names: bool,
    ) -> Result<Self> {
        let out = Self { circuit, arg_specs };
        if check_all_inputs_used {
            out.check_all_inputs_used()?;
        }
        if check_unique_arg_names {
            out.check_unique_arg_names()?;
        }
        Ok(out)
    }

    pub fn check_all_inputs_used(&self) -> Result<()> {
        let missing_symbols: HashSet<_> = self
            .are_args_used()
            .into_iter()
            .zip(&self.arg_specs)
            .filter_map(|(used, arg_spec)| (!used).then(|| arg_spec.symbol.clone()))
            .collect();
        if !missing_symbols.is_empty() {
            bail!(ConstructError::ModuleSomeArgsNotPresent {
                spec_circuit: self.circuit.clone(),
                missing_symbols,
            })
        }
        Ok(())
    }

    pub fn check_unique_arg_names(&self) -> Result<()> {
        // TODO: maybe cache me (as needed)!
        if self
            .arg_specs
            .iter()
            .any(|x| x.symbol.info().name.is_none())
        {
            bail!(ConstructError::ModuleSomeArgsNamedNone {
                symbols_named_none: self
                    .arg_specs
                    .iter()
                    .filter_map(|x| x.symbol.info().name.is_none().then(|| x.symbol.clone()))
                    .collect()
            })
        }
        let names: Vec<Name> = self
            .arg_specs
            .iter()
            .map(|x| x.symbol.info().name.unwrap())
            .collect();
        if !is_unique(&names) {
            bail!(ConstructError::ModuleArgsDupNames {
                dup_names: counts_g_1(names.into_iter().map(|x| x.to_owned()))
            })
        }

        Ok(())
    }

    #[pyo3(name = "map_circuit")]
    pub fn map_circuit_py(&self, f: PyObject) -> Result<Self> {
        self.map_circuit(|x| pycall!(f, (x,), anyhow))
    }

    #[staticmethod]
    #[pyo3(signature=(circuit, check_unique_arg_names = true))]
    pub fn new_free_symbols(circuit: CircuitRc, check_unique_arg_names: bool) -> Result<Self> {
        let arg_specs = get_free_symbols(&circuit)
            .iter()
            .map(|x| ModuleArgSpec::new(x.clone(), true, true, true))
            .collect();
        let out =
            Self::new(circuit, arg_specs, true, false).expect("method should guarantee valid");
        if check_unique_arg_names {
            // check after instead of in new so we can .expect on other errors in new
            out.check_unique_arg_names()?;
        }
        // maybe we should use no_check_args for speed
        Ok(out)
    }

    #[staticmethod]
    #[pyo3(signature=(circuit, arg_specs, check_all_inputs_used = true, check_unique_arg_names = true))]
    pub fn new_extract(
        circuit: CircuitRc,
        arg_specs: Vec<(CircuitRc, ModuleArgSpec)>,
        check_all_inputs_used: bool,
        check_unique_arg_names: bool,
    ) -> Result<Self> {
        let mut new_arg_specs: Vec<Option<ModuleArgSpec>> = vec![None; arg_specs.len()];
        let spec_circuit = deep_map_op_context_preorder_stoppable(
            circuit.clone(),
            &|circuit,
              c: &mut (
                &mut Vec<Option<ModuleArgSpec>>,
                &Vec<(CircuitRc, ModuleArgSpec)>,
            )| {
                let (real_arg_specs, proposed_arg_specs) = c;
                if let Some(i) = proposed_arg_specs
                    .iter()
                    .position(|x| x.0.info().hash == circuit.info().hash)
                {
                    let mut arg_spec = proposed_arg_specs[i].1.clone();
                    arg_spec.symbol = Symbol::new(
                        circuit.info().shape.clone(),
                        arg_spec.symbol.uuid,
                        arg_spec.symbol.info().name.or(circuit.info().name),
                    );
                    real_arg_specs[i] = Some(arg_spec);
                    return (Some(real_arg_specs[i].as_ref().unwrap().symbol.crc()), true);
                }
                (None, false)
            },
            &mut (&mut new_arg_specs, &arg_specs),
            &mut Default::default(),
        )
        .unwrap_or(circuit);
        let new_arg_specs: Vec<ModuleArgSpec> = if check_all_inputs_used {
            let er = new_arg_specs
                .iter()
                .cloned()
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| ConstructError::ModuleExtractNotPresent {
                    subcirc: arg_specs[new_arg_specs.iter().position(|x| x.is_none()).unwrap()]
                        .0
                        .clone(),
                });
            er?
        } else {
            new_arg_specs
                .into_iter()
                // TODO: maybe instead of filter we're supposed to just have missing module args?
                .filter(|z| z.is_some())
                .collect::<Option<Vec<_>>>()
                .unwrap()
        };
        // maybe we should use no_check_args for speed
        let out = Self::new(spec_circuit, new_arg_specs, check_all_inputs_used, false)
            .expect("method should guarantee valid");
        if check_unique_arg_names {
            // check after instead of in new so we can .expect on other errors in new
            out.check_unique_arg_names()?;
        }
        Ok(out)
    }

    // TODO: add some naming options maybe
    pub fn resize(&self, shapes: Vec<Shape>) -> Result<Self> {
        let arg_specs: Vec<ModuleArgSpec> = zip(&self.arg_specs, shapes)
            .map(|(arg_spec, shape)| ModuleArgSpec {
                symbol: Symbol::new(shape, arg_spec.symbol.uuid, arg_spec.symbol.info().name),
                ..arg_spec.clone()
            })
            .collect();

        let circuit = self
            .substitute(
                &arg_specs.iter().map(|x| x.symbol.crc()).collect::<Vec<_>>(),
                None,
                None,
            )
            .context("substitute failed from resize")?;
        Ok(Self { circuit, arg_specs })
    }

    pub fn are_args_used(&self) -> Vec<bool> {
        are_args_used(&self.circuit, &self.arg_specs)
    }

    pub fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    fn __richcmp__(&self, object: &Self, comp_op: CompareOp) -> bool {
        use_rust_comp(&self, &object, comp_op)
    }

    fn __hash__(&self) -> u64 {
        let mut s = FxHasher::default();
        self.hash(&mut s);
        s.finish()
    }

    #[pyo3(name = "map_on_replaced_path")]
    pub fn py_map_on_replaced_path(&self, f: PyCallable) -> Result<Self> {
        self.map_on_replaced_path(|x| pycall!(f, (x,), anyhow))
    }

    pub fn rename_on_replaced_path(&self, prefix: Option<String>, suffix: Option<String>) -> Self {
        if prefix.is_none() && suffix.is_none() {
            return self.clone();
        }

        self.map_on_replaced_path_unwrap(|x| {
            if let Some(n) = x.info().name {
                let name = Some(
                    (prefix.as_deref().unwrap_or("").to_owned()
                        + n.into()
                        + suffix.as_deref().unwrap_or(""))
                    .into(),
                );
                x.rename(name)
            } else {
                x
            }
        })
    }
}

#[pyclass]
#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct ModuleArgSpec {
    #[pyo3(get, set)]
    pub symbol: Symbol,
    #[pyo3(get, set)]
    pub batchable: bool,
    #[pyo3(get, set)]
    pub expandable: bool,
    #[pyo3(get, set)]
    pub ban_non_symbolic_size_expand: bool,
}

impl Default for ModuleArgSpec {
    fn default() -> Self {
        Self {
            symbol: Symbol::new_with_none_uuid([].into_iter().collect(), None),
            batchable: true,
            expandable: true,
            ban_non_symbolic_size_expand: true,
        }
    }
}

pub type ModuleArgSpecHashable = (HashBytes, [bool; 3]);

impl ModuleArgSpec {
    pub fn get_hashable(&self) -> ModuleArgSpecHashable {
        (
            self.symbol.info().hash,
            [
                self.batchable,
                self.expandable,
                self.ban_non_symbolic_size_expand,
            ],
        )
    }
}

#[pymethods]
impl ModuleArgSpec {
    #[new]
    #[pyo3(signature=(
        symbol,
        batchable = Self::default().batchable,
        expandable = Self::default().expandable,
        ban_non_symbolic_size_expand = Self::default().ban_non_symbolic_size_expand
    ))]
    fn new(
        symbol: Symbol,
        batchable: bool,
        expandable: bool,
        ban_non_symbolic_size_expand: bool,
    ) -> Self {
        Self {
            symbol,
            batchable,
            expandable,
            ban_non_symbolic_size_expand,
        }
    }
    #[staticmethod]
    #[pyo3(signature=(
        circuit,
        batchable = Self::default().batchable,
        expandable = Self::default().expandable,
        ban_non_symbolic_size_expand = false, // I think this is right default for this?
    ))]
    pub fn just_name_shape(
        circuit: CircuitRc,
        batchable: bool,
        expandable: bool,
        ban_non_symbolic_size_expand: bool,
    ) -> Self {
        Self {
            symbol: Symbol::new_with_random_uuid(circuit.info().shape.clone(), circuit.info().name),
            batchable,
            expandable,
            ban_non_symbolic_size_expand,
        }
    }

    pub fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    fn __richcmp__(&self, object: &Self, comp_op: CompareOp) -> bool {
        use_rust_comp(&self, &object, comp_op)
    }

    fn __hash__(&self) -> u64 {
        let mut s = FxHasher::default();
        self.hash(&mut s);
        s.finish()
    }
}

/// can also be thought of as a lambda + it's arguments in lambda calculus (but not yet beta reduced)
/// aka call site
#[pyclass(extends=PyCircuitBase)]
#[derive(Clone)]
#[repr(C)] // for some reason rustc wants to put info last (it's first in all other nodes!), repr(C) stops that
pub struct Module {
    info: CachedCircuitInfo,
    // invariant: spec.circuit == info.children[0]
    #[pyo3(get)]
    pub spec: ModuleSpec,
}

impl Module {
    #[apply(new_rc_unwrap)]
    pub fn try_new(nodes: Vec<CircuitRc>, spec: ModuleSpec, name: Option<Name>) -> Result<Self> {
        // we need to check this here because we substitute modules below where other checks run
        spec.check_no_free_bound_by_nested(&nodes).with_context(|| {
            const MSG: &str = "(free sym bound by nested)";
            format!("module construction failed because substitution isn't valid {MSG} (name={name:?})")
        })?;
        let mut substitute_cached_info = ModuleSpec {
            // substituing all
            circuit: substitute_all_modules(spec.circuit.clone()),
            arg_specs: spec.arg_specs.clone(),
        }
        .substitute(&nodes, None, None)
        .with_context(|| format!("module construction failed on substitute name={name:?}"))?
        .info()
        .clone();

        // the substituted free symbols is *nearly* right, but has a few issues:
        // - nodes might be unused which would result in missing their free symbols (and we want to include them)
        // - If the spec.circuit is a single free symbol, it isn't counter
        // - the ordering isn't ideal
        // Due to these issues we recompute the free symbols 'by hand'
        let mut free_syms: IndexSet<_> = get_free_symbols(&spec.circuit)
            .iter()
            .filter(|x| !spec.arg_specs.iter().any(|arg_spec| &arg_spec.symbol == *x))
            .cloned()
            .collect();
        for n in &nodes {
            free_syms.extend(get_free_symbols(&**n).iter().cloned());
        }
        substitute_cached_info.free_symbols = Some(Arc::new(free_syms));

        let out = Self {
            info: CachedCircuitInfo {
                name,
                children: once(spec.circuit.clone()).chain(nodes).collect(),
                ..substitute_cached_info
            },
            spec,
        };
        out.initial_init_info()
    }

    pub fn args(&self) -> std::slice::Iter<'_, CircuitRc> {
        self.info.children[1..].iter()
    }

    pub fn args_cloned(&self) -> Vec<CircuitRc> {
        self.info.children[1..].to_vec()
    }

    pub fn args_slice(&self) -> &[CircuitRc] {
        &self.info.children[1..]
    }

    pub fn num_args(&self) -> usize {
        self.info.children.len() - 1
    }

    pub fn new_kwargs(
        kwargs: &HashMap<Name, CircuitRc>,
        spec: ModuleSpec,
        name: Option<Name>,
    ) -> Result<Self> {
        let mut nodes: Vec<_> = vec![None; spec.arg_specs.len()];
        spec.check_unique_arg_names()?;
        for (k, v) in kwargs {
            match spec
                .arg_specs
                .iter()
                .position(|x| x.symbol.info().name.expect("check_unique_arg_names above") == *k)
            {
                Some(i) => {
                    nodes[i] = Some(v.clone());
                }
                None => {
                    bail!(ConstructError::ModuleUnknownArgument {
                        argument: *k,
                        all_module_inputs: spec
                            .arg_specs
                            .iter()
                            .map(|x| x.symbol.info().name.unwrap())
                            .collect()
                    })
                }
            }
        }
        if nodes.iter().any(|x| x.is_none()) {
            bail!(ConstructError::ModuleMissingNames {
                missing_arguments: nodes
                    .iter()
                    .zip(spec.arg_specs)
                    .filter_map(|(n, arg_spec)| n.is_none().then(|| arg_spec
                        .symbol
                        .info()
                        .name
                        .expect("check_unique_arg_names above")))
                    .collect()
            });
        }

        Self::try_new(nodes.into_iter().map(Option::unwrap).collect(), spec, name)
    }

    fn child_axis_map_inputs_uncached(&self) -> Result<Vec<Vec<Option<usize>>>> {
        let spec_resized = self
            .spec
            .resize(self.args().map(|x| x.info().shape.clone()).collect())?;
        let spec_circuit_resized = spec_resized.circuit.clone();
        assert!(spec_circuit_resized.info().shape[..] == self.info().shape[..]);
        let stripped = deep_strip_axis_names(spec_circuit_resized, &None);
        let out_named = propagate_named_axes(
            stripped,
            (0..self.info().rank())
                .map(|i| (i as u8, i.to_string().into()))
                .collect(),
            true,
        );
        let mut result: Vec<Vec<Option<usize>>> =
            self.args().map(|c| vec![None; c.info().rank()]).collect();
        visit_circuit_unwrap(out_named, |x| {
            if let Some(sym) = x.as_symbol() {
                if let Some(i) = self.spec.arg_specs.iter().position(|x| {
                    x.symbol.uuid == sym.uuid && x.symbol.info().name == sym.info().name
                }) {
                    for (k, v) in &sym.info().named_axes {
                        result[i][*k as usize] = Some(v.parse::<usize>().unwrap());
                    }
                }
            }
        });
        Ok(result)
    }
}

circuit_node_extra_impl!(Module, self_hash_default);

impl CircuitNodeHashItems for Module {
    fn compute_hash_non_name_non_children(&self, hasher: &mut blake3::Hasher) {
        self.spec.compute_non_children_hash(hasher)
    }
}

impl CircuitNodeSetNonHashInfo for Module {
    fn set_non_hash_info(&mut self) -> Result<()> {
        Ok(())
    }
}

impl CircuitNode for Module {
    circuit_node_auto_impl!("6825f723-f178-4dab-b568-cd85eb6d2bf3");

    fn child_axis_map(&self) -> Vec<Vec<Option<usize>>> {
        let map_inputs = self.child_axis_map_inputs().unwrap();
        once(vec![None; self.spec.circuit.info().rank()])
            .chain(map_inputs)
            .collect()
    }

    fn _replace_children(&self, children: Vec<CircuitRc>) -> Result<Self> {
        let new_spec = if children[0] == self.spec.circuit {
            self.spec.clone()
        } else {
            ModuleSpec {
                circuit: children[0].clone(),
                arg_specs: self.spec.arg_specs.clone(),
            }
        };

        assert_eq!(
            self.spec.arg_specs.len(),
            self.num_args(),
            "guaranteed by constructor via expand"
        );

        Self::try_new(children[1..].to_vec(), new_spec, self.info().name)
    }

    fn num_free_children(&self) -> usize {
        1
    }

    fn eval_tensors(
        &self,
        _tensors: &[rr_util::py_types::Tensor],
    ) -> Result<rr_util::py_types::Tensor> {
        bail!(TensorEvalError::ModulesCantBeDirectlyEvalutedInternal {
            module: self.clone(),
        })
    }
}

impl CircuitNodeAutoName for Module {
    const PRIORITY: OperatorPriority = OperatorPriority::Function {};

    fn auto_name(&self) -> Option<Name> {
        if self.children().any(|x| x.info().name.is_none()) {
            None
        } else {
            Some(
                (self.spec.circuit.info().name.unwrap().string()
                    + "("
                    + &self
                        .args()
                        .map(|node| node.info().name.unwrap().into())
                        .collect::<Vec<String>>()
                        .join(", ")
                    + ")")
                    .into(),
            )
        }
    }
}

#[pymethods]
impl Module {
    #[new]
    #[pyo3(signature=(spec, name = None, **kwargs))]
    fn new_py(
        spec: ModuleSpec,
        name: Option<Name>,
        kwargs: Option<HashMap<Name, CircuitRc>>,
    ) -> PyResult<PyClassInitializer<Module>> {
        Ok(Module::new_kwargs(&kwargs.unwrap_or_else(HashMap::default), spec, name)?.into_init())
    }

    #[staticmethod]
    #[pyo3(signature=(spec, *nodes, name = None))]
    fn new_flat(spec: ModuleSpec, nodes: Vec<CircuitRc>, name: Option<Name>) -> Result<Self> {
        Self::try_new(nodes, spec, name)
    }

    /// TODO: fancier renaming technology?
    #[pyo3(name = "substitute")]
    #[pyo3(signature=(
        name_prefix = None,
        name_suffix = None,
        use_self_name_as_prefix = false
    ))]
    pub fn substitute_py(
        &self,
        name_prefix: Option<String>,
        name_suffix: Option<String>,
        use_self_name_as_prefix: bool,
    ) -> Result<CircuitRc> {
        let name_prefix = self.get_prefix(name_prefix, use_self_name_as_prefix)?;
        Ok(self.substitute(name_prefix, name_suffix))
    }

    #[pyo3(signature=(prefix = None, suffix = None, use_self_name_as_prefix = false))]
    pub fn rename_on_replaced_path(
        &self,
        prefix: Option<String>,
        suffix: Option<String>,
        use_self_name_as_prefix: bool,
    ) -> Result<Module> {
        let prefix = self.get_prefix(prefix, use_self_name_as_prefix)?;
        Ok(Self::new(
            self.args_cloned(),
            self.spec.rename_on_replaced_path(prefix, suffix),
            self.info().name,
        ))
    }

    #[getter(nodes)]
    pub fn nodes_py(&self) -> Vec<CircuitRc> {
        self.args_cloned()
    }

    pub fn map_on_replaced_path(&self, f: PyCallable) -> Result<Self> {
        Ok(Self::new(
            self.args_cloned(),
            self.spec.py_map_on_replaced_path(f)?,
            self.info().name,
        ))
    }

    pub fn aligned_batch_shape(&self) -> Shape {
        self.spec.aligned_batch_shape(self.args_slice()).unwrap()
    }

    /// None is conform all dims
    pub fn conform_to_input_batch_shape(&self, dims_to_conform: Option<usize>) -> Result<Self> {
        let current_aligned = self.aligned_batch_shape();
        let dims_to_conform = dims_to_conform.unwrap_or(current_aligned.len());

        if dims_to_conform > current_aligned.len() {
            // better error as needed
            bail!("dims to conform is larger than num batch dims!")
        }

        let batch_shapes = self.spec.batch_shapes(self.args_slice());

        Ok(Self::new(
            self.args_cloned(),
            self.spec
                .resize(
                    self.spec
                        .arg_specs
                        .iter()
                        .zip(&batch_shapes)
                        .map(|(arg_spec, current_batch_shape)| {
                            let batch_start =
                                current_batch_shape.len().saturating_sub(dims_to_conform);
                            current_batch_shape[batch_start..]
                                .iter()
                                .chain(arg_spec.symbol.shape())
                                .copied()
                                .collect()
                        })
                        .collect(),
                )
                .expect("constructor should ensure this works"),
            self.info().name,
        ))
    }

    // TODO: add some naming options maybe
    pub fn conform_to_input_shapes(&self) -> Self {
        Self::new(
            self.args_cloned(),
            self.spec
                .resize(self.args().map(|x| x.shape().clone()).collect())
                .expect("constructor should ensure this works"),
            self.info().name,
        )
    }

    fn child_axis_map_inputs(&self) -> Result<Vec<Vec<Option<usize>>>> {
        let key = (
            self.spec.get_hashable(),
            self.args().map(|x| x.info().shape.clone()).collect(),
        );
        if let Some(result) =
            MODULE_EXPANSIONS_AXIS_MAPS.with(|cache| cache.borrow().get(&key).cloned())
        {
            return Ok(result);
        }

        let result = self.child_axis_map_inputs_uncached()?;
        MODULE_EXPANSIONS_AXIS_MAPS.with(|cache| cache.borrow_mut().insert(key, result.clone()));
        Ok(result)
    }

    pub fn arg_items(&self) -> Vec<(CircuitRc, ModuleArgSpec)> {
        self.args()
            .cloned()
            .zip(self.spec.arg_specs.clone())
            .collect()
    }
}

impl Module {
    pub fn substitute_self_name_prefix(&self) -> CircuitRc {
        self.substitute(self.info().name.map(|x| x.string() + "."), None)
    }

    pub fn substitute_no_self_rename(
        &self,
        name_prefix: Option<String>,
        name_suffix: Option<String>,
    ) -> CircuitRc {
        self.spec
            .substitute(self.args_slice(), name_prefix, name_suffix)
            .expect("constructor should ensure this works")
    }

    pub fn substitute(
        &self,
        name_prefix: Option<String>,
        name_suffix: Option<String>,
    ) -> CircuitRc {
        self.spec
            .substitute_with_mod_name(
                self.args_slice(),
                name_prefix,
                name_suffix,
                self.info().name,
            )
            .expect("constructor should ensure this works")
    }

    fn get_prefix(
        &self,
        name_prefix: Option<String>,
        use_self_name_as_prefix: bool,
    ) -> Result<Option<String>> {
        if use_self_name_as_prefix {
            if let Some(name_prefix) = name_prefix {
                bail!(
                    ConstructError::ModulePassedNamePrefixAndUseSelfNameAsPrefix {
                        name_prefix,
                        module: self.clone()
                    }
                )
            }

            Ok(self.info().name.map(|x| x.string() + "."))
        } else {
            Ok(name_prefix)
        }
    }
}

#[derive(Default)]
struct AllModuleSubstituter {
    cache: FastUnboundedCache<HashBytes, CircuitRc>,
}

impl AllModuleSubstituter {
    #[apply(cached_method)]
    #[self_id(self_)]
    #[key(circuit.info().hash)]
    #[cache_expr(cache)]
    pub fn substitute_all_modules(&mut self, circuit: CircuitRc) -> CircuitRc {
        let old_children: Vec<CircuitRc> = circuit.children().collect();
        let mut new_children: Vec<_> = old_children
            .into_iter()
            .map(|c| self_.substitute_all_modules(c))
            .collect();
        if let Some(m) = circuit.as_module() {
            // it's important that we don't construct a module here - the
            // module constructor uses the global ALL_MODULE_SUBSTITUTOR so
            // this would panic on double borrowing the ref cell.
            let nodes = new_children.split_off(1);
            let new_spec_circuit = new_children.pop().unwrap();
            ModuleSpec {
                circuit: new_spec_circuit,
                arg_specs: m.spec.arg_specs.clone(),
            }
            // we could speed up this substitute invocation probably
            // currently this is overall O(m*n) where m is number of mods and n is number of circs
            // but cached to O(n) per each module
            // and we could instead choose to make this O(n) but with worse caching.
            // Not clear if this would be slower or faster overall.
            .substitute_with_mod_name(&nodes, None, None, m.info().name)
            .expect("module constructor should ensure this works")
        } else {
            circuit.map_children_unwrap_idxs(|i| new_children[i].clone())
        }
    }
}

#[pyfunction]
pub fn substitute_all_modules(circuit: CircuitRc) -> CircuitRc {
    ALL_MODULE_SUBSTITUTOR.with(|sub| sub.borrow_mut().substitute_all_modules(circuit))
}

#[pyfunction]
pub fn conform_all_modules(circuit: CircuitRc) -> CircuitRc {
    deep_map_preorder_unwrap(circuit, |c| match &**c {
        Circuit::Module(mn) => mn.conform_to_input_shapes().rc(),
        _ => c.clone(),
    })
}

#[pyfunction]
pub fn inline_single_callsite_modules(circuit: CircuitRc) -> CircuitRc {
    let mut module_callsites: HashMap<ModuleSpec, usize> = HashMap::default();
    visit_circuit_unwrap(circuit.clone(), |circ| {
        if let Some(module) = circ.as_module() {
            module_callsites.insert(
                module.spec.clone(),
                *module_callsites.get(&module.spec).unwrap_or(&0) + 1,
            );
        }
    });
    deep_map_pre_new_children(circuit, |c, children| match &**c {
        Circuit::Module(mn) => {
            if module_callsites
                .get(&mn.spec)
                .map(|i| *i == 1)
                .unwrap_or(false)
            {
                mn.map_children_idxs(|z| Ok(children[z].clone()))
                    .unwrap()
                    .substitute(None, None)
            } else {
                c.clone()
            }
        }
        _ => c.clone(),
    })
}

#[pyfunction]
pub fn get_children_with_symbolic_sizes(circuit: CircuitRc) -> HashSet<CircuitRc> {
    let mut circuits_with_symbolic_sizes = HashSet::default();
    visit_circuit_unwrap(circuit, |x| {
        if x.shape()
            .iter()
            .any(|s| SymbolicSizeProduct::has_symbolic(*s))
        {
            circuits_with_symbolic_sizes.insert(x);
        }
    });
    circuits_with_symbolic_sizes
}

#[pyfunction]
pub fn any_children_with_symbolic_sizes(circuit: CircuitRc) -> bool {
    !get_children_with_symbolic_sizes(circuit).is_empty()
}

pub fn has_free_sym(circuit: &CircuitRc, sym: &Symbol) -> bool {
    if let Some(other) = circuit.as_symbol() {
        other == sym
    } else {
        circuit.info().get_raw_free_symbols().contains(sym)
    }
}

pub fn is_intersecting_free_syms_iter<'a>(
    circuit: &CircuitRc,
    syms: impl IntoIterator<Item = &'a Symbol>,
) -> bool {
    if let Some(other) = circuit.as_symbol() {
        syms.into_iter().any(|x| x == other)
    } else {
        syms.into_iter()
            .any(|x| circuit.info().get_raw_free_symbols().contains(x))
    }
}

pub fn is_intersecting_free_syms(circuit: &CircuitRc, syms: &IndexSet<Symbol>) -> bool {
    if let Some(other) = circuit.as_symbol() {
        syms.contains(other)
    } else {
        !circuit.info().get_raw_free_symbols().is_disjoint(syms)
    }
}

#[derive(Copy, Debug, Clone)]
struct SymsNodesInfo<'a> {
    syms: &'a [&'a Symbol],
    nodes: &'a [&'a CircuitRc],
    hash: HashBytes,
}

impl<'a> SymsNodesInfo<'a> {
    pub fn new(syms: &'a [&'a Symbol], nodes: &'a [&'a CircuitRc]) -> Self {
        let mut hasher = blake3::Hasher::new();

        for sym in syms {
            hasher.update(&sym.info().hash);
        }
        for n in nodes {
            hasher.update(&n.info().hash);
        }

        Self {
            syms,
            nodes,
            hash: hasher.finalize().into(),
        }
    }
}

fn recur_check_no_free_bound_by_nested(
    circ: CircuitRc,
    info: SymsNodesInfo<'_>,
    seen: &mut HashSet<(HashBytes, HashBytes)>,
) -> Result<()> {
    if !seen.insert((circ.info().hash, info.hash)) {
        return Ok(());
    }

    if let Some(m) = circ.as_module() {
        // filter out bound symbols for recurring into
        let (syms, nodes): (Vec<_>, Vec<&CircuitRc>) = info
            .syms
            .iter()
            .zip(info.nodes)
            .filter(|(s, _)| {
                // if the module binds the sym or doesn't have this symbol free, filter out
                m.spec
                    .arg_specs
                    .iter()
                    .all(|arg_spec| &arg_spec.symbol != **s)
                    && has_free_sym(&m.spec.circuit, *s)
            })
            .unzip();

        if !syms.is_empty() {
            // fail earlier if possible
            recur_check_no_free_bound_by_nested(
                m.spec.circuit.clone(),
                SymsNodesInfo::new(&syms, &nodes),
                seen,
            )?;

            // then check valid
            for (&sub_for_sym, &sym) in nodes.iter().zip(&syms) {
                if m.spec
                    .arg_specs
                    .iter()
                    .any(|x| has_free_sym(sub_for_sym, &x.symbol))
                {
                    bail!(SubstitutionError::CircuitHasFreeSymsBoundByNestedModule {
                        circ: sub_for_sym.clone(),
                        sym: sym.clone(),
                        nested_module: m.clone(),
                        bound_by_nested: m
                            .spec
                            .arg_specs
                            .iter()
                            .filter_map(|arg_spec| has_free_sym(sub_for_sym, &arg_spec.symbol)
                                .then(|| arg_spec.symbol.clone()))
                            .collect()
                    })
                }
            }
        }
    }

    circ.non_free_children()
        .map(|child| recur_check_no_free_bound_by_nested(child, info, seen))
        .collect::<Result<Vec<_>>>()?;
    Ok(())
}

const SAME_IDENT_P1: &str = concat!(
    "Substituting for a symbol when there is a not-equal free symbol",
    " with the same name and uuid is not allowed (aka, a near miss)."
);
const SAME_IDENT_P2: &str = "This is caused by having a different shape (or different named axes).";

#[apply(python_error_exception)]
#[base_error_name(Substitution)]
#[base_exception(PyValueError)]
#[derive(Error, Debug, Clone)]
pub enum SubstitutionError {
    #[error("subbing circ={circ:?} for sym={sym:?} results in nested_module={nested_module:?} bound_by_nested={bound_by_nested:?} {} ({e_name})",
        "inside of circ (aka higher order function)")]
    CircuitHasFreeSymsBoundByNestedModule {
        circ: CircuitRc,
        sym: Symbol,
        nested_module: Module,
        bound_by_nested: Vec<Symbol>,
    },

    #[error(
        "sym={sym:?} matching_sym={matching_sym:?}\n{SAME_IDENT_P1}\n{SAME_IDENT_P2}\n({e_name})"
    )]
    FoundNEQFreeSymbolWithSameIdentification { sym: Symbol, matching_sym: Symbol },
}

use std::cell::RefCell;
thread_local! {
    static MODULE_EXPANSIONS: RefCell<
        HashMap<
            (ModuleSpecHashable, Vec<HashBytes>, Option<String>, Option<String>),
            CircuitRc,
        >,
    > = RefCell::new(HashMap::default());
    static MODULE_EXPANSIONS_SHAPE: RefCell<
        HashMap<
            (ModuleSpecHashable, Vec<Shape>),
            (CircuitRc, Vec<HashBytes>, String),
        >,
    > = RefCell::new(HashMap::default());
    static MODULE_EXPANSIONS_AXIS_MAPS: RefCell<
        HashMap<(ModuleSpecHashable, Vec<Shape>), Vec<Vec<Option<usize>>>>,
    > = RefCell::new(HashMap::default());
    static NO_FREE_BOUND_BY_NESTED: RefCell<
        HashSet<(ModuleSpecHashable, Vec<HashBytes>)>,
    > = RefCell::new(HashSet::default());
    static ALL_MODULE_SUBSTITUTOR: RefCell<AllModuleSubstituter> = RefCell::new(Default::default());
}

#[pyfunction]
pub fn clear_module_circuit_caches() {
    MODULE_EXPANSIONS.with(|f| f.borrow_mut().clear());
    MODULE_EXPANSIONS_SHAPE.with(|f| f.borrow_mut().clear());
    ALL_MODULE_SUBSTITUTOR.with(|f| *f.borrow_mut() = Default::default());
}
