#[macro_export]
macro_rules! pycallable{
    {@handle_ty; $ty:ty; $ex:expr} => { $ex };
    {
        $(#[doc = $doc:literal])*
        #[pyo3(name=$name_string:literal)]
        pub fn $name:ident<F>($arg_name:ident : $arg_ty:ty , $f:ident :F   $(,)?)-> Result<$ret_ty:ty>
        where
        F:$ftype:ident($( ($fn_arg_name:ident , $fn_arg_ty:ty)),*$(,)?)->Result<$fn_ret_ty:ty>$(,)?
        { $($tt:tt)* }
    }=>{
        $(#[doc = $doc])*
        pub fn $name<F> ($arg_name : $arg_ty , $f:F)-> anyhow::Result<$ret_ty>
        where
        F:$ftype($($fn_arg_ty),*)->anyhow::Result<$fn_ret_ty>
        { $($tt)* }
        paste::paste!{
            pub fn [<$name _unwrap>]<F> ($arg_name : $arg_ty , $f:F)-> $ret_ty
            where
            F:$ftype($($fn_arg_ty),*)->$fn_ret_ty
            {
                $name($arg_name,|$($fn_arg_name),*|Ok($f($($fn_arg_name),*))).unwrap()
            }

            #[pyfunction]
            #[pyo3(name=$name_string)]
            pub fn [<$name _py>]($arg_name : $arg_ty , o:rr_util::py_types::PyCallable)-> anyhow::Result<$ret_ty>{
                $name($arg_name , |$( $fn_arg_name :$fn_arg_ty),*| $crate::pycallable!(@handle_ty; $ret_ty; $crate::pycall!(o,($( $fn_arg_name),*,), anyhow)))
            }
        }
    };
    // copy but with mut :(
    {
        $(#[doc = $doc:literal])*
        #[pyo3(name=$name_string:literal)]
        pub fn $name:ident<F>($arg_name:ident : $arg_ty:ty , mut $f:ident :F   $(,)?)-> Result<$ret_ty:ty>
        where
        F:$ftype:ident($( ($fn_arg_name:ident , $fn_arg_ty:ty)),*$(,)?)->Result<$fn_ret_ty:ty>$(,)?
        { $($tt:tt)* }
    }=>{
        $(#[doc = $doc])*
        #[allow(unused_mut)]
        pub fn $name<F> ($arg_name : $arg_ty , mut $f:F)-> Result<$ret_ty>
        where
        F:$ftype($($fn_arg_ty),*)->Result<$fn_ret_ty>
        { $($tt)* }

        paste::paste!{
            pub fn [<$name _unwrap>]<F> ($arg_name : $arg_ty , mut $f:F)-> $ret_ty
            where
            F:$ftype($($fn_arg_ty),*)->$fn_ret_ty
            {
                $name($arg_name,|$($fn_arg_name),*|Ok($f($($fn_arg_name),*))).unwrap()
            }

            #[pyfunction]
            #[pyo3(name=$name_string)]
            pub fn [<$name _py>]($arg_name : $arg_ty , o:rr_util::py_types::PyCallable)-> Result<$ret_ty>{
                $name($arg_name , |$( $fn_arg_name :$fn_arg_ty),*|$crate::pycallable!(@handle_ty; $ret_ty; $crate::pycall!(o,($( $fn_arg_name),*,), anyhow)))
            }
        }
    };
}
