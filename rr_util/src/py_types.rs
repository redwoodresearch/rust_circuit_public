use std::{fmt, ops::Deref};

use anyhow::{Context, Result};
use macro_rules_attribute::apply;
use pyo3::{
    once_cell::GILLazyPy,
    prelude::*,
    pyclass::CompareOp,
    types::{IntoPyDict, PyTuple},
};
use rustc_hash::FxHashMap as HashMap;

use crate::{
    tensor_util::{Shape, TorchDeviceDtype, TorchDtype},
    util::{EinsumAxes, HashBytes, HashBytesToPy},
};
pub struct PyUtils {
    pub torch: PyObject,
    get_tensor_shape: PyObject,
    id: PyObject,
    pub cast_int: PyObject,
    scalar_to_tensor: PyObject,
    pub cast_tensor: PyObject,
    pub dtype_convert: HashMap<TorchDtype, PyObject>,
    un_flat_concat: PyObject,
    tensor_scale: PyObject,
    pub generalfunctions: std::collections::HashMap<String, PyObject>,
    pub gen_index_function: PyObject,
    pub explicit_gen_index_function: PyObject,
    pub pow: PyObject,
    einsum: PyObject,
    make_diagonal: PyObject,
    pub print: PyObject,
    pub random_indices: PyObject,
    pub conv: PyObject,
    pub tensor_to_bytes: PyObject,
    pub tensor_from_bytes: PyObject,
    pub assert_tensors_close: PyObject,
    pub builtins: Py<PyModule>,
    pub gc: Py<PyModule>,
    pub optimizing_symbolic_size_warning: PyObject,
    pub maybe_dtype_to_maybe_string: PyObject,
    pub random_i64: PyObject,
}

/// misc python utilities
pub static PY_UTILS: GILLazyPy<PyUtils> = GILLazyPy::new_py(|py| {
    let utils = PyModule::from_code(
        py,
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/rust_circuit_type_utils.py"
        )),
        concat!(env!("CARGO_MANIFEST_DIR"), "/rust_circuit_type_utils.py"),
        "rust_circuit_type_utils",
    )
    .unwrap();

    let get = |s: &str| utils.getattr(s).unwrap().into();
    let builtins = PyModule::import(py, "builtins").unwrap();
    let get_builtin = |s: &str| builtins.getattr(s).unwrap().into();

    PyUtils {
        torch: get("torch"),
        get_tensor_shape: get("get_tensor_shape"),
        id: get_builtin("id"),
        cast_int: get_builtin("int"),
        scalar_to_tensor: get("scalar_to_tensor"),
        cast_tensor: get("cast_tensor"),
        dtype_convert: utils
            .getattr("dtype_from_string")
            .unwrap()
            .extract()
            .unwrap(),
        un_flat_concat: get("un_flat_concat"),
        tensor_scale: get("tensor_scale"),
        generalfunctions: utils
            .getattr("generalfunctions")
            .unwrap()
            .extract()
            .unwrap(),
        gen_index_function: get("gen_index_function"),
        explicit_gen_index_function: get("explicit_gen_index_function"),
        pow: get("pow"),
        einsum: get("einsum"),
        make_diagonal: get("make_diagonal"),
        print: get_builtin("print"),
        random_indices: get("random_indices"),
        conv: get("conv"),
        tensor_to_bytes: get("tensor_to_bytes"),
        tensor_from_bytes: get("tensor_from_bytes"),
        assert_tensors_close: get("assert_tensors_close"),
        builtins: builtins.into(),
        gc: PyModule::import(py, "gc").unwrap().into(),
        optimizing_symbolic_size_warning: get("OptimizingSymbolicSizeWarning"),
        maybe_dtype_to_maybe_string: get("maybe_dtype_to_maybe_string"),
        random_i64: get("random_i64"),
    }
});

pub fn py_address(x: &PyObject) -> usize {
    Python::with_gil(|py| {
        PY_UTILS
            .id
            .call(py, (x,), None)
            .unwrap()
            .extract(py)
            .unwrap()
    })
}

#[macro_export]
macro_rules! make_py_func {
    {
        #[py_ident($py:ident)]
        $( #[$m:meta] )*
        $vi:vis fn $name:ident($($arg_name:ident : $arg_ty:ty),* $(,)?) -> $ret_ty:ty
        { $($tt:tt)* }
    } => {
        paste::paste!{
            $(#[$m])*
            $vi fn [<$name _py>]<'py>($py : Python<'py>, $($arg_name : $arg_ty,)*) -> $ret_ty {
                $($tt)*
            }
            $(#[$m])* // TODO: maybe shouldn't apply to both??
            $vi fn $name($($arg_name : $arg_ty,)*) -> $ret_ty {
                Python::with_gil(|py| [<$name _py>](py, $($arg_name,)*))
            }
        }

    };
}

// currently unused, but I'm leaving around because a bit nice to use for debugging etc sometimes.
pub fn collect_garbage() {
    Python::with_gil(|py| PY_UTILS.gc.call_method0(py, "collect")).unwrap();
}

#[apply(make_py_func)]
#[py_ident(py)]
pub fn assert_tensors_close(a: Tensor, b: Tensor) -> Result<()> {
    PY_UTILS.assert_tensors_close.call(py, (a, b), None)?;
    Ok(())
}

#[apply(make_py_func)]
#[py_ident(py)]
pub fn scalar_to_tensor(v: f64, shape: Shape, device_dtype: TorchDeviceDtype) -> Result<Tensor> {
    PY_UTILS
        .scalar_to_tensor
        .call(py, (v, shape, device_dtype), None)
        .context("scalar to tensor")?
        .extract::<Tensor>(py)
        .context("scalar to tensor extract")
}

#[apply(make_py_func)]
#[py_ident(py)]
pub fn i64_to_tensor(v: i64, shape: Shape, device_dtype: TorchDeviceDtype) -> Result<Tensor> {
    PY_UTILS
        .scalar_to_tensor
        .call(py, (v, shape, device_dtype), None)
        .context("scalar to tensor")?
        .extract::<Tensor>(py)
        .context("scalar to tensor extract")
}

#[apply(make_py_func)]
#[py_ident(py)]
pub fn einsum(items: Vec<(Tensor, EinsumAxes)>, out_axes: EinsumAxes) -> PyResult<Tensor> {
    PY_UTILS
        .einsum
        .call(py, (items, out_axes), None)?
        .extract(py)
}

#[apply(make_py_func)]
#[py_ident(py)]
pub fn make_diagonal(
    non_diag: &Tensor,
    out_axes_deduped: EinsumAxes,
    out_axes: EinsumAxes,
) -> PyResult<Tensor> {
    PY_UTILS
        .make_diagonal
        .call(py, (non_diag, out_axes_deduped, out_axes), None)?
        .extract(py)
}

#[apply(make_py_func)]
#[py_ident(py)]
pub fn einops_repeat(
    tensor: &Tensor,
    op: String,
    sizes: impl IntoIterator<Item = (String, usize)>,
) -> Result<Tensor> {
    PY_EINOPS
        .repeat
        .call(py, (tensor, op), Some(sizes.into_py_dict(py)))
        .context("einops repeat")?
        .extract(py)
        .context("einops repeat extract")
}

#[apply(make_py_func)]
#[py_ident(py)]
pub fn un_flat_concat(tensor: &Tensor, split_shapes: Vec<Shape>) -> PyResult<Vec<Tensor>> {
    PY_UTILS
        .un_flat_concat
        .call(py, (tensor, split_shapes), None)
        .context("un flat concat")?
        .extract(py)
}

#[apply(make_py_func)]
#[py_ident(py)]
pub fn tensor_scale(tensor: &Tensor) -> PyResult<f64> {
    PY_UTILS.tensor_scale.call(py, (tensor,), None)?.extract(py)
}

#[apply(make_py_func)]
#[py_ident(py)]
pub fn random_torch_i64() -> i64 {
    PY_UTILS.random_i64.call0(py).unwrap().extract(py).unwrap()
}

pub struct PyEinops {
    pub einops: Py<PyModule>,
    pub repeat: PyObject,
}

pub static PY_EINOPS: GILLazyPy<PyEinops> = GILLazyPy::new_py(|py| {
    let einops = PyModule::import(py, "einops").unwrap();
    let get = |s: &str| einops.getattr(s).unwrap().into();

    PyEinops {
        einops: einops.into(),
        repeat: get("repeat"),
    }
});

pub static HASH_TENSOR: GILLazyPy<(PyObject, PyObject)> = GILLazyPy::new_py(|py| {
    let module = PyModule::from_code(
        py,
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/tensor_hash.py")),
        concat!(env!("CARGO_MANIFEST_DIR"), "/tensor_hash.py"),
        "tensor_hash",
    )
    .unwrap();
    (
        module.getattr("hash_tensor").unwrap().into(),
        module.getattr("pop_cuda_context").unwrap().into(),
    )
});

#[macro_export]
macro_rules! pycall {
    ($f:expr,$args:expr) => {
        pyo3::Python::with_gil(|py| $f.call(py, $args, None).unwrap().extract(py).unwrap())
    };
    ($f:expr,$args:expr,raw) => {
        pyo3::Python::with_gil(|py| $f.call(py, $args, None).unwrap())
    };
    ($f:expr, $args:expr, anyhow) => {
        pyo3::Python::with_gil(|py| {
            $f.call(py, $args, None)
                .map_err(|z| anyhow::Error::from(z))?
                .extract(py)
                .map_err(|z| anyhow::Error::from(z))
        })
    };
}

#[macro_export]
macro_rules! atr {
    ($x:expr,$name:ident) => {
        Python::with_gil(|py| {
            $x.getattr(py, stringify!($name))
                .unwrap()
                .extract(py)
                .unwrap()
        })
    };
    ($x:expr,$name:ident,raw) => {
        Python::with_gil(|py| $x.getattr(py, stringify!($name)).unwrap())
    };
    ($x:expr,$name:ident, anyhow) => {
        Python::with_gil(|py| Ok($x.getattr(py, stringify!($name))?.extract(py)?))
    };
}

macro_rules! generate_extra_py_ops {
    [$($op:ident),*] => {
        paste::paste! {
            struct PyOperators {
                $($op: PyObject,)*
            }

            static PY_OPERATORS: GILLazyPy<PyOperators> = GILLazyPy::new_py(|py| {
                let operator = PyModule::import(py, "operator").unwrap();

                PyOperators {
                    $( $op : operator.getattr(stringify!($op)).unwrap().into(),)*
                }
            });


            /// Trait for python operator methods when they return the same type.
            /// Used for tensors.
            ///
            /// Not useful when an operator returns a different type: this will
            /// always raise an error (e.g. dict).
            ///
            /// # Example
            ///
            /// ```
            /// # use pyo3::prelude::*;
            /// # use rr_util::py_types::ExtraPySelfOps;
            ///
            /// #[derive(Clone, Debug, FromPyObject)]
            /// struct WrapInt(i64);
            ///
            /// impl IntoPy<PyObject> for WrapInt {
            ///     fn into_py(self, py: Python<'_>) -> PyObject {
            ///         self.0.into_py(py)
            ///     }
            /// }
            ///
            /// impl ExtraPySelfOps for WrapInt {}
            ///
            /// pyo3::prepare_freethreaded_python();
            ///
            /// assert_eq!(
            ///     Python::with_gil(|py| WrapInt(8).py_add(py, 7)).unwrap().0,
            ///     7 + 8
            /// );
            /// assert_eq!(
            ///     Python::with_gil(|py| WrapInt(2).py_mul(py, 3)).unwrap().0,
            ///     2 * 3
            /// );
            /// ```
            pub trait ExtraPySelfOps
            where
                Self: IntoPy<PyObject>,
                for<'a> Self: FromPyObject<'a>,
            {
                $(
                    fn [<py_ $op>](self, py: Python<'_>, x: impl IntoPy<PyObject>) -> PyResult<Self> {
                        PY_OPERATORS.$op.call1(py, (self, x))?.extract(py)
                    }

                    // not sure if this method should exist
                    fn [<py_ $op _acquire>](self, x: impl IntoPy<PyObject>) -> PyResult<Self> {
                        Python::with_gil(|py| self.[<py_ $op>](py, x))
                    }
                )*
            }
        }
    }
}

// add more as needed
generate_extra_py_ops!(add, getitem, mul);

#[derive(Clone)]
pub struct Tensor {
    tensor: PyObject,
    shape: Shape, /* cache shape so doesn't have to be recomputed on reconstruct etc (not uber efficient I think) */
    hash: Option<HashBytes>,
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("tensor", &self.tensor)
            .field("shape", &self.shape)
            .finish_non_exhaustive()
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        if let Some(a)=self.hash && let Some(b)=other.hash{
            a==b
        }else{
            false
        }
    }
}

impl Eq for Tensor {}

impl<'source> FromPyObject<'source> for Tensor {
    fn extract(tensor: &'source PyAny) -> PyResult<Self> {
        let shape =
            Python::with_gil(|py| PY_UTILS.get_tensor_shape.call1(py, (tensor,))?.extract(py))?;

        Ok(Self {
            tensor: tensor.into(),
            shape,
            hash: None,
        })
    }
}

impl IntoPy<PyObject> for &Tensor {
    fn into_py(self, _py: Python<'_>) -> PyObject {
        self.tensor.clone()
    }
}
impl IntoPy<PyObject> for Tensor {
    fn into_py(self, _py: Python<'_>) -> PyObject {
        self.tensor
    }
}

impl ExtraPySelfOps for Tensor {}

impl Tensor {
    pub fn tensor(&self) -> &PyObject {
        &self.tensor
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn hash(&self) -> Option<&HashBytes> {
        self.hash.as_ref()
    }

    pub fn hash_base16(&self) -> Option<String> {
        self.hash.as_ref().map(|x| base16::encode_lower(x))
    }

    pub fn flatten(&self) -> Tensor {
        Python::with_gil(|py| {
            self.tensor
                .call_method(py, "flatten", (), None)
                .unwrap()
                .extract(py)
                .unwrap()
        })
    }

    /// DANGEROUS
    pub fn set_hash(&mut self, hash: Option<HashBytes>) {
        self.hash = hash;
    }

    pub fn hash_usize(&self) -> Option<usize> {
        self.hash.as_ref().map(|x| {
            let mut hash_prefix: [u8; 8] = Default::default();
            hash_prefix.copy_from_slice(&x[..8]);
            usize::from_le_bytes(hash_prefix)
        })
    }

    pub fn hashed(&self) -> Result<Tensor> {
        if self.hash.is_some() {
            Ok(self.clone())
        } else {
            Ok(Self {
                tensor: self.tensor.clone(),
                shape: self.shape.clone(),
                hash: Python::with_gil(|py| {
                    Ok::<Option<HashBytes>, anyhow::Error>(
                        HASH_TENSOR
                            .0
                            .call(py, (self.tensor.clone(),), None)?
                            .extract(py)
                            .unwrap(),
                    )
                })?,
            })
        }
    }
}

#[pyfunction]
pub fn hash_tensor(t: Tensor) -> Result<HashBytesToPy> {
    Ok((*t.hashed()?.hash().unwrap()).into())
}

#[derive(FromPyObject)]
pub struct PyShape(pub Shape);

impl IntoPy<PyObject> for PyShape {
    fn into_py(self, py: Python<'_>) -> PyObject {
        PyTuple::new(py, self.0).into_py(py)
    }
}

#[derive(FromPyObject)]
pub struct PyEinsumAxes(pub EinsumAxes);

impl IntoPy<PyObject> for PyEinsumAxes {
    fn into_py(self, py: Python<'_>) -> PyObject {
        PyTuple::new(py, &*self.0).into_py(py)
    }
}

#[derive(Debug, Clone)]
pub struct PyCallable(PyObject);

impl fmt::Display for PyCallable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Deref for PyCallable {
    type Target = PyObject;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl PyCallable {
    pub fn new(callable: PyObject) -> PyResult<Self> {
        if !Python::with_gil(|py| callable.as_ref(py).is_callable()) {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Value isn't callable!",
            ));
        }
        Ok(Self(callable))
    }
}

impl<'source> FromPyObject<'source> for PyCallable {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        Self::new(ob.into())
    }
}

impl IntoPy<PyObject> for PyCallable {
    fn into_py(self, _py: Python<'_>) -> PyObject {
        self.0
    }
}

pub fn use_rust_comp<T: PartialOrd>(l: &T, r: &T, comp_op: CompareOp) -> bool {
    match comp_op {
        CompareOp::Lt => l < r,
        CompareOp::Gt => l > r,
        CompareOp::Le => l <= r,
        CompareOp::Ge => l >= r,
        CompareOp::Eq => l == r,
        CompareOp::Ne => l != r,
    }
}

/// macro instead of generic type because of pyo3 bs
#[macro_export]
macro_rules! make_single_many {
    ($name:ident, $type:ty, $container:ident, into_py) => {
        make_single_many!($name, $type, $container);

        impl IntoPy<PyObject> for $name {
            fn into_py(self, py: Python<'_>) -> PyObject {
                match self {
                    Self::Many(x) => x.into_py(py),
                    Self::Single(x) => x.into_py(py),
                }
            }
        }
    };
    ($name:ident, $type:ty, $container:ident) => {
        #[derive(Clone, Debug, FromPyObject)]
        pub enum $name {
            Many($container<$type>),
            Single($type),
        }

        impl $name {
            pub fn into_many(self) -> $container<$type> {
                match self {
                    Self::Single(x) => [x].into_iter().collect(),
                    Self::Many(many) => many,
                }
            }

            pub fn is_many(&self) -> bool {
                matches!(self, Self::Many(_))
            }
            pub fn is_single(&self) -> bool {
                matches!(self, Self::Single(_))
            }

            pub fn map<F: FnMut($type) -> $type>(self, mut f: F) -> Self {
                match self {
                    Self::Single(x) => Self::Single(f(x)),
                    Self::Many(x) => Self::Many(x.into_iter().map(f).collect()),
                }
            }

            pub fn len(&self) -> usize {
                match self {
                    Self::Single(_) => 1,
                    Self::Many(many) => many.len(),
                }
            }
        }
    };
}

make_single_many!(PyOpAtAxes, i64, Vec);

pub fn reduction_to_ints(x: Option<PyOpAtAxes>, ndim: usize) -> Vec<i64> {
    x.map(|x| x.into_many())
        .unwrap_or_else(|| (0..ndim as i64).collect())
}

pub struct PyCircuitItems {
    pub computational_node: PyObject,
    pub constant: PyObject,
    pub interop: PyObject,
    pub rust_to_py: PyObject,
    pub circ_compiler_util: PyObject,
}

pub static PY_CIRCUIT_ITEMS: GILLazyPy<PyCircuitItems> = GILLazyPy::new_py(|py| {
    let module = PyModule::from_code(
        py,
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/py_circuit_interop.py"
        )),
        concat!(env!("CARGO_MANIFEST_DIR"), "/py_circuit_interop.py"),
        "py_circuit_interop",
    )
    .unwrap(); // TODO: fail more gracefully?

    let get = |s: &str| module.getattr(s).unwrap().into();

    let interop = get("interop");

    PyCircuitItems {
        computational_node: get("computational_node"),
        constant: get("constant"),
        rust_to_py: interop.getattr(py, "rust_to_py").unwrap(),
        interop,
        circ_compiler_util: get("circ_compiler_util"),
    }
});

pub static SELF_MODULE: GILLazyPy<Py<PyModule>> = GILLazyPy::new_py(|py| {
    PyModule::import(py, "rust_circuit").unwrap().into() // obviously a bit cursed : (
});

#[pyclass]
#[derive(Copy, Clone)]
pub struct NotSet;

#[derive(FromPyObject, Clone)]
pub enum MaybeNotSetImpl<T> {
    NotSet(NotSet),
    Val(T),
}

#[derive(Default, Clone)]
pub struct MaybeNotSet<T>(pub Option<T>);

impl<'source, T: FromPyObject<'source>> FromPyObject<'source> for MaybeNotSet<T> {
    fn extract(inp: &'source PyAny) -> PyResult<Self> {
        let x: MaybeNotSetImpl<T> = inp.extract()?;
        let out = match x {
            MaybeNotSetImpl::Val(x) => Some(x),
            MaybeNotSetImpl::NotSet(_) => None,
        };
        Ok(MaybeNotSet(out))
    }
}

pub fn is_python_running() -> bool {
    // I assume this is fine?
    // possibly not thread safe...
    unsafe { pyo3::ffi::Py_IsInitialized() != 0 }
}
