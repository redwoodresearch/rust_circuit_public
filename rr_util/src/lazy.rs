/// use via [macro_rules_attribute::apply](https://docs.rs/macro_rules_attribute/latest/macro_rules_attribute/)
#[macro_export]
macro_rules! make_lazy {
    {
        $( #[doc = $d:literal] )*
        $( #[lazy_ty($($which_lazy:tt)*)] )?
        $( #[lazy_name($lazy_name:ident)] )?
        $vi:vis fn $name:ident () -> $ret_ty:ty
        { $($tt:tt)* }
    } => {
        $(#[doc = $d])*
        $vi fn $name() -> $ret_ty {
            $($tt)*
        }


        $crate::make_lazy!(@lazy_impl $( #[doc = $d] )* $( #[lazy_ty($($which_lazy)*)] )* $( #[lazy_name($lazy_name)] )* {$vi $name; $ret_ty});
    };
    {
        @lazy_impl
        $( #[doc = $d:literal] )*
        #[lazy_ty($($which_lazy:tt)*)]
        #[lazy_name($lazy_name:ident)]
        {$vis:vis $name:ident; $ret_ty:ty}
    } => {
        $(#[doc = $d])*
        $vis static $lazy_name: $($which_lazy)*<$ret_ty> = $($which_lazy)*::new($name);
    };
    {
        @lazy_impl
        $( #[doc = $d:literal] )*
        #[lazy_ty($($which_lazy:tt)*)]
        {$vis:vis $name:ident; $ret_ty:ty}
    } => {
        paste::paste! {
            $crate::make_lazy!(@lazy_impl $(#[doc = $d])* #[lazy_ty($($which_lazy)*)] #[lazy_name([<$name:upper>])] {$vis $name; $ret_ty});
        }
    };
    {
        @lazy_impl
        $( #[doc = $d:literal] )*
        $( #[lazy_name($lazy_name:ident)] )?
        {$($t:tt)*}
    } => {
        $crate::make_lazy!(@lazy_impl $(#[doc = $d])* #[lazy_ty($crate::GILLazyPy)] $( #[lazy_name($lazy_name)] )* {$($t)*});
    };
}
