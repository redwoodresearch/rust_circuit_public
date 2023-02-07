use pyo3::{
    types::{PyModule, PyType},
    Py, PyResult, PyTypeInfo, Python,
};

pub fn py_get_type<T: PyTypeInfo>() -> Py<PyType> {
    Python::with_gil(|py| T::type_object(py).into())
}

pub trait HasPythonException {
    fn get_py_err(&self, string: String) -> pyo3::PyErr;
    fn register(py: Python<'_>, m: &PyModule) -> PyResult<()>;
    fn print_stub(py: Python<'_>) -> PyResult<String>;
}

#[macro_export]
macro_rules! python_error_exception {
    (
        #[base_error_name($e_name:ident)]
        #[base_exception($base_ty:ty)]
        // #[error_py_description($desc:literal)]
        $( #[$($meta_tt:tt)*] )*
        $vis:vis enum $name:ident {
            $(
                #[error($($error_tt:tt)*)]
                $(#[$($meta_tt_item:tt)*])*
                $err_name:ident {
                    $(
                        $arg:ident : $arg_ty:ty
                    ),*
                    $(,)?
                },
            )*
        }
    ) => {
        $( #[$($meta_tt)*] )*
        $vis enum $name {
            $(
                #[error($($error_tt)*, e_name = format!("{}{}Error", stringify!($e_name), stringify!($err_name)))]
                $(#[$($meta_tt_item)*])*
                $err_name {
                    $(
                        $arg : $arg_ty,
                    )*
                },
            )*
        }

        paste::paste! {
            type [<__ $name BaseExcept>] = $base_ty;


            $(
                #[pyo3::prelude::pyclass]
                struct [<$e_name $err_name Info>] {
                    #[pyo3(get)]
                    err_msg_string : String,
                    $(
                        #[pyo3(get)]
                        $arg : $arg_ty,
                    )*
                }

                #[pyo3::prelude::pymethods]
                impl [<$e_name $err_name Info>] {
                    fn __repr__(&self) -> &str {
                        &self.err_msg_string
                    }

                    fn items(&self) -> pyo3::Py<pyo3::types::PyDict> {
                        Python::with_gil(|py| {
                            let dict = pyo3::types::PyDict::new(py);
                            $(
                                dict.set_item(stringify!($arg), self.$arg.clone().into_py(py)).unwrap();
                            )*
                            dict.into()
                        })
                    }
                }
            )*


            mod [< __ $name:snake _python_exception_stuff>] {
                $(
                use super::[<$e_name $err_name Info>];
                )*
                $crate::python_error_exception! {
                    @in_mod $vis $name {
                        $(
                            $err_name [[<$e_name $err_name Error>]] [[<$e_name $err_name Info>]] {
                                $(
                                    $arg : $arg_ty,
                                )*
                            },
                        )*
                    }
                    ([<$e_name Error>] super::[<__ $name BaseExcept>]
                     // $desc
                     )
                }
            }

            #[allow(dead_code)]
            $vis type [<Py $e_name Error>] = [< __ $name:snake _python_exception_stuff>]::[<$e_name Error>];
            $(
            #[allow(dead_code)]
            $vis type  [<Py $e_name $err_name Error>] = [< __ $name:snake _python_exception_stuff>]::[<$e_name $err_name Error>];
            )*
        }
    };
    (@op_name [] $name:ident) => {
        $name
    };
    (@op_name [$e_name:ident] $name:ident) => {
        paste::paste! {
            [<$e_name Error>]
        }
    };
    (
        @in_mod $vis:vis $name:ident {
            $(
                $err_name:ident [$sub_excep_name:ident] [$sub_excep_info_name:ident] {
                    $(
                        $arg:ident : $arg_ty:ty,
                    )*
                },
            )*
        }
        ($excep_name:ident $base_ty:ty
         // $desc:literal
         )
    ) => {
        use pyo3::{
            create_exception, types::PyModule, PyErr, PyResult, PyTypeInfo, Python,
        };

        use $crate::errors_util::HasPythonException;

        create_exception!(
            rust_circuit,
            $excep_name,
            $base_ty
            //, $desc
        );
        $(
            create_exception!(
                rust_circuit,
                $sub_excep_name,
                $excep_name
            );
        )*


        impl HasPythonException for super::$name {
            fn get_py_err(&self, err_msg_string: String) -> pyo3::PyErr {
                use super::$name::*;
                match self {
                    $(
                        $err_name { $($arg,)* } => {
                            PyErr::new::<$sub_excep_name, _>($sub_excep_info_name { err_msg_string, $($arg : $arg.clone(),)* })
                        },
                    )*
                }
            }
            fn register(py: Python<'_>, m: &PyModule) -> PyResult<()> {
                m.add(
                    stringify!($excep_name),
                    py.get_type::<$excep_name>(),
                )?;
                $(
                    m.add(
                        stringify!($sub_excep_name),
                        py.get_type::<$sub_excep_name>(),
                    )?;
                    m.add_class::<$sub_excep_info_name>()?;
                )*

                Ok(())
            }
            fn print_stub(py : Python<'_>) -> PyResult<String> {
                let out = [
                    format!("class {}({}): ...", $excep_name::NAME, <$base_ty>::type_object(py).name()?),
                    $(
                        format!("class {}({}): ...", $sub_excep_name::NAME, $excep_name::NAME),
                    )*
                ].join("\n");
                Ok(out)
            }
        }
    }
}
