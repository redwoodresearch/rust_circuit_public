// ! CircuitNodeInit and various helpers for nicely implementing it
use std::{collections::BTreeMap, iter::zip, sync::Arc};

use anyhow::Context;
use itertools::Itertools;
use rr_util::{
    name::Name,
    symbolic_size::{SymbolicSizeConstraint, SymbolicSizeConstraints},
    tensor_util::TorchDeviceDtypeOp,
    util::HashBytes,
    IndexSet,
};
use uuid::uuid;

use crate::{
    CachedCircuitInfo, CircuitFlags, CircuitNode, CircuitRc, ConstructError, NamedAxes, Result,
    Symbol,
};

pub trait CircuitNodeInit {
    fn initial_init_info_impl(self) -> Result<Self>
    where
        Self: Sized,
    {
        self.init_info_impl(true)
    }

    fn init_info_impl(self, is_initial: bool) -> Result<Self>
    where
        Self: Sized;

    fn rename_impl(self, new_name: Option<Name>) -> Self
    where
        Self: Sized;

    fn update_info_impl<F>(self, f: F) -> Result<Self>
    where
        Self: Sized,
        F: FnOnce(&mut CachedCircuitInfo);
}

pub trait CircuitNodePrivate {
    fn info_mut(&mut self) -> &mut CachedCircuitInfo;
}

pub trait CircuitNodeComputeInfoImpl: CircuitNode {
    fn compute_flags(&self) -> CircuitFlags {
        self.compute_flags_default()
    }

    fn device_dtype_extra(&self) -> Option<Vec<TorchDeviceDtypeOp>> {
        None
    }
    fn device_dtype_override(&self) -> Option<Result<TorchDeviceDtypeOp>> {
        None
    }
    fn symbolic_size_constraints_extra(&self) -> Result<Vec<SymbolicSizeConstraint>> {
        Ok(Vec::new())
    }
}

fn compute_device_dtype(
    children: &[CircuitRc],
    extra: Option<Vec<TorchDeviceDtypeOp>>,
) -> Result<TorchDeviceDtypeOp> {
    children
        .iter()
        .map(|c| c.info().device_dtype.clone())
        .chain(extra.clone().into_iter().flatten())
        .try_fold(TorchDeviceDtypeOp::NONE, |acc, new| {
            TorchDeviceDtypeOp::combine(acc, new)
        })
        .with_context(|| {
            format!(
                "Could not combine device_dtype for node with children {:?} and extra {:?}",
                children
                    .iter()
                    .map(|c| c.info().clone())
                    .collect::<Vec<_>>(),
                extra
            )
        })
}

fn compute_symbolic_size_constraints(
    children: &[CircuitRc],
    extra: Vec<SymbolicSizeConstraint>,
) -> Result<SymbolicSizeConstraints> {
    SymbolicSizeConstraints::new(
        children
            .iter()
            .flat_map(|c| {
                c.info()
                    .symbolic_size_constraints
                    .clone()
                    .into_constraints()
            })
            .chain(extra)
            .collect(),
    )
}

fn compute_named_axes(
    children: &[CircuitRc],
    child_axis_map: &Vec<Vec<Option<usize>>>,
) -> NamedAxes {
    let mut result: NamedAxes = BTreeMap::new();
    for (mp, child) in zip(child_axis_map, children) {
        for (ax, name) in &child.info().named_axes {
            if let Some(top_ax) = mp[(*ax) as usize] {
                result.insert(top_ax as u8, name.clone());
            }
        }
    }
    result
}

fn compute_free_symbols(children: &[CircuitRc]) -> Option<Arc<IndexSet<Symbol>>> {
    // ideally, symbol would just return itself rather than doing this check here
    // but, then the symbol would have a circular reference to itself
    // So we can only include children and then we special case syms in various places. : /
    children
        .iter()
        .filter_map(|x| {
            if let Some(sym) = x.as_symbol() {
                Some(Arc::new(IndexSet::from_iter([sym.clone()])))
            } else {
                x.info().free_symbols.clone()
            }
        })
        .at_most_one()
        .unwrap_or_else(|mut it| {
            let mut a = (*it.next().unwrap()).clone();
            it.for_each(|x| a.extend(x.iter().cloned()));
            Some(Arc::new(a))
        })
}

pub trait CircuitNodeSetNonHashInfo: CircuitNodePrivate {
    fn set_non_hash_info(&mut self) -> Result<()>;
}

impl<T: CircuitNodeComputeInfoImpl + CircuitNodePrivate> CircuitNodeSetNonHashInfo for T {
    fn set_non_hash_info(&mut self) -> Result<()> {
        self.info_mut().flags = self.compute_flags();
        self.info_mut().device_dtype = if let Some(z) = self.device_dtype_override() {
            z
        } else {
            compute_device_dtype(self.children_sl(), self.device_dtype_extra())
        }?;
        self.info_mut().symbolic_size_constraints = compute_symbolic_size_constraints(
            self.children_sl(),
            self.symbolic_size_constraints_extra()?,
        )?;
        self.info_mut().named_axes = if let Some(out) = self.already_set_named_axes() {
            out
        } else if self
            .children_sl()
            .iter()
            .all(|x| x.info().named_axes.is_empty())
        {
            BTreeMap::new()
        } else {
            compute_named_axes(self.children_sl(), &self.child_axis_map())
        };
        self.info_mut().free_symbols = compute_free_symbols(self.children_sl());
        Ok(())
    }
}

pub trait CircuitNodeHashItems {
    fn compute_hash_non_name_non_children(&self, _hasher: &mut blake3::Hasher) {}
}

pub trait CircuitNodeHashWithChildren {
    fn compute_hash_non_name(&self, hasher: &mut blake3::Hasher);
}

impl<T: CircuitNodeHashItems + CircuitNode> CircuitNodeHashWithChildren for T {
    fn compute_hash_non_name(&self, hasher: &mut blake3::Hasher) {
        self.compute_hash_non_name_non_children(hasher);
        for child in self.children() {
            hasher.update(&child.info().hash);
        }
    }
}

impl<T> CircuitNodeInit for T
where
    T: CircuitNodeHashWithChildren
        + CircuitNodePrivate
        + CircuitNodeSetNonHashInfo
        + CircuitNode
        + Sized,
{
    fn init_info_impl(mut self, is_initial: bool) -> Result<Self> {
        let mut hasher = blake3::Hasher::new();
        self.compute_hash_non_name(&mut hasher);
        hasher.update(&self.node_type_uuid());

        // note that we this might be set by a prior construction
        // Because we can be called by (e.g.) rename_impl or update_info_impl
        let prior_autoname = self.info().use_autoname();
        let already_set_named_axes = self.already_set_named_axes(); // have named axes already been set?

        self.set_non_hash_info()?;
        if let Some(already_set_named_axes) = already_set_named_axes {
            self.info_mut().named_axes = already_set_named_axes; // if already set, keep these named axes
        }
        if !prior_autoname {
            // if disabled on init, keep disabled
            self.info_mut().flags &= !CircuitFlags::USE_AUTONAME;
        }
        // we have to call maybe_auto_name after setting the flag
        if is_initial {
            self.info_mut().name = self.info().name.or_else(|| {
                self.info()
                    .use_autoname()
                    .then(|| self.get_autoname().ok())
                    .flatten()
                    .flatten()
            })
        }

        for (axis, _) in &self.info().named_axes {
            if *axis as usize >= self.info().shape.len() {
                return Err(ConstructError::NamedAxisAboveRank {}.into());
            }
        }

        self.info_mut().hash = compute_hash_info(self.info(), &mut hasher);
        Ok(self)
    }

    fn rename_impl(mut self, new_name: Option<Name>) -> Self {
        self.info_mut().name = new_name;
        self.init_info_impl(false).unwrap() // we could avoid recomputing some stuff if we wanted
    }

    fn update_info_impl<F>(mut self, f: F) -> Result<Self>
    where
        F: FnOnce(&mut CachedCircuitInfo),
    {
        f(self.info_mut());
        self.init_info_impl(false)
    }
}

fn compute_hash_info(info: &CachedCircuitInfo, hasher: &mut blake3::Hasher) -> HashBytes {
    hasher.update(&[info.name.is_some() as u8]);
    hasher.update(Name::str_maybe_empty(info.name).as_bytes());
    hasher.update(uuid!("025e9af4-1366-4211-aa5f-7c28fc6cdf9f").as_bytes()); // delimit name with uuid

    // it's important that we hash the final autoname and named axes
    // instead of prior/already set values. (otherwise stuff like renaming
    // to same name can change hash!)
    hasher.update(&[info.use_autoname() as u8]);
    for (axis, name) in &info.named_axes {
        hasher.update(&[*axis]);
        hasher.update(name.as_bytes());
        // delimit axis names with uuid
        hasher.update(uuid!("db6b1967-35f7-4571-a69c-82ba3340215d").as_bytes());
    }
    hasher.finalize().into()
}
