#[macro_export]
macro_rules! setup_wrap {
    (@full_inside $impl_struct_name:ident, $rc_struct_name:ident, $to_impl_struct_name:ident, $to_rc_struct_name:ident, $enum_name:ty, $from_py:ty) => {
        use $crate::{
            util::HashBytes, eq_by_big_hash::EqByBigHash, impl_both_by_big_hash, simple_from,
            py_types::use_rust_comp,
        };
        use pyo3::pyclass::CompareOp;
        use std::{
            fmt,
            ops::{Deref, DerefMut},
            prelude::rust_2021::*,
            sync::Arc,
        };

        /// NOTE: it's *not* valid to cache by bytes. The hash maybe depends on object
        /// pointer equality, so if you let a given item be deallocated, you can get
        /// collisions!
        #[pyo3::pyclass]
        #[derive(Clone)]
        pub struct $impl_struct_name {
            pub(super) data: $enum_name,
            pub(super) hash: HashBytes,
        }

        impl fmt::Debug for $impl_struct_name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.data.fmt(f)
            }
        }

        impl EqByBigHash for $impl_struct_name {
            fn hash(&self) -> HashBytes {
                self.hash
            }
        }
        impl_both_by_big_hash!($impl_struct_name);

        impl $impl_struct_name {
            pub fn data(&self) -> &$enum_name {
                &self.data
            }

            pub fn new(data: $enum_name) -> Self {
                let mut hasher = blake3::Hasher::new();
                hasher.update(&data.uuid());
                data.item_hash(&mut hasher);

                Self {
                    data,
                    hash: hasher.finalize().into(),
                }
            }

            pub fn rc(self) -> $rc_struct_name {
                self.into()
            }
            pub fn crc(&self) -> $rc_struct_name {
                self.clone().rc()
            }
        }

        simple_from!(|x: $enum_name| -> $impl_struct_name { Self::new(x) });
        simple_from!(|x: $enum_name| -> $rc_struct_name { Self(Arc::new(x.into())) });
        simple_from!(|x: $from_py| -> $rc_struct_name { Self(Arc::new(x.into())) });
        simple_from!(|x: $impl_struct_name| -> $rc_struct_name { Self(Arc::new(x)) });

        impl $enum_name {
            pub fn $to_impl_struct_name(self) -> $impl_struct_name {
                self.into()
            }
            pub fn $to_rc_struct_name(self) -> $impl_struct_name {
                self.into()
            }
        }
        impl $from_py {
            pub fn $to_impl_struct_name(self) -> $impl_struct_name {
                self.into()
            }
            pub fn $to_rc_struct_name(self) -> $impl_struct_name {
                self.into()
            }
        }

        #[pyo3::pymethods]
        impl $impl_struct_name {
            fn __richcmp__(&self, object: &Self, comp_op: CompareOp) -> bool {
                use_rust_comp(&self, &object, comp_op)
            }
            pub fn __hash__(&self) -> u64 {
                self.first_u64()
            }
            fn debug_print_to_str(&self) -> String {
                format!("{:?}", self)
            }
        }

        #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $rc_struct_name(pub Arc<$impl_struct_name>);

        simple_from!(|x: Arc<$impl_struct_name>| -> $rc_struct_name { Self(x) });

        impl Deref for $rc_struct_name {
            type Target = Arc<$impl_struct_name>;

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }
        impl DerefMut for $rc_struct_name {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.0
            }
        }
        impl<'source> pyo3::FromPyObject<'source> for $rc_struct_name {
            fn extract(from_py_obj: &'source pyo3::PyAny) -> pyo3::PyResult<Self> {
                let from: $from_py = from_py_obj.extract()?;
                Ok(from.into())
            }
        }
        impl pyo3::IntoPy<pyo3::PyObject> for $rc_struct_name {
            fn into_py(self, py: pyo3::Python<'_>) -> pyo3::PyObject {
                (**self).clone().into_py(py)
            }
        }
    };
    ($struct_name:ident, $enum_name:ty, $from_py:ty) => {
        paste::paste! {
            mod [<__ $struct_name:snake module>] {
                $crate::setup_wrap!(@full_inside $struct_name, [<$struct_name Rc>], [<to_ $struct_name:snake>], [<to_ $struct_name:snake _rc>],
                    super::$enum_name, super::$from_py);
            }

            pub use [<__ $struct_name:snake module>]::{$struct_name, [<$struct_name Rc>]};
        }
    };
}

#[macro_export]
macro_rules! setup_callable {
    ($struct_name:ident, $enum_name:ty, $from_py:ty, $func_name:ident ($($arg_name:ident : $arg_ty:ty),*) -> $func_ret:ty,no_py_callable) => {

        $crate::setup_wrap!($struct_name, $enum_name, $from_py);

        paste::paste!{
            #[derive(Clone)]
            pub struct [<Raw $struct_name>](Arc<dyn Fn($($arg_ty,)*) -> anyhow::Result<$func_ret> + Send + Sync>);

            impl Debug for [<Raw $struct_name>] {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    f.debug_tuple(&("Raw".to_owned() + stringify!($struct_name))).finish()
                }
            }
        }

        impl $struct_name {
            pub fn new_func<F: Fn($($arg_ty,)*) -> $func_ret + Send + Sync + 'static>(func: F) -> Self {
                Self::new_func_err(move |n| Ok(func(n)))
            }

            pub fn new_func_err<F: Fn($($arg_ty,)*) -> anyhow::Result<$func_ret> + Send + Sync + 'static>(
                func: F,
            ) -> Self {
                Self::new_func_err_arc(Arc::new(func))
            }

            pub fn new_func_err_arc(func: Arc<dyn Fn($($arg_ty,)*) -> anyhow::Result<$func_ret> + Send + Sync>) -> Self {
                paste::paste! {
                    $enum_name::Raw([<Raw $struct_name>](func)).into()
                }
            }
        }
    };
    ($struct_name:ident, $enum_name:ty, $from_py:ty, $func_name:ident ($($arg_name:ident : $arg_ty:ty),*) -> $func_ret:ty) => {

        setup_callable!($struct_name, $enum_name, $from_py, $func_name ($($arg_name : $arg_ty),*) -> $func_ret,no_py_callable);

        #[pymethods]
        impl $struct_name {
            fn __call__(&self, _py: Python<'_>, $($arg_name : $arg_ty,)*) -> anyhow::Result<$func_ret> {
                self.$func_name($($arg_name,)*)
            }
        }

    };
}
