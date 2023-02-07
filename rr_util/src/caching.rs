use rustc_hash::FxHashMap as HashMap;

/// Sets up caching for a given method.
///
/// (used by [cached_lambda](crate::cached_lambda) macro, so those examples also applicable)
///
/// # Example
/// ```
/// 
/// use cached::Cached;
/// use macro_rules_attribute::apply;
/// use rr_util::{cached_method, caching::FastUnboundedCache}; // we use this to have a proc macro.
///
/// #[derive(Default)]
/// struct FunctionWithCache {
///     my_cache_name: FastUnboundedCache<u64, u64>,
/// }
///
/// impl FunctionWithCache {
///     #[apply(cached_method)] // we apply this as an attribute
///     #[self_id(self_)] // [optional] self_id attribute for self name in body (due to hygine bs)
///     #[key(a)] // [optional] the expression to produce a hash key
///     #[cache_expr(my_cache_name)] // this can take any expression to get a mutable reference to your cache from self
///     pub fn c(
///         &mut self,
///         a: u64,
///         depth: u64, // caching ignores this
///     ) -> u64 {
///         match a {
///             0 => 0,
///             1 => 1,
///             n => self_.c(n - 1, depth + 1) + self_.c(n - 2, depth + 1),
///         }
///     }
/// }
///
/// let mut func = FunctionWithCache::default();
///
/// assert_eq!(func.c(0, 0), 0);
/// assert_eq!(func.c(1, 0), 1);
/// assert_eq!(func.c(7, 0), 13);
/// assert_eq!(func.c(85, 0), 259695496911122585);
///
/// // no effect on caching!
/// assert_eq!(func.c(85, 3), 259695496911122585);
///
/// assert_eq!(func.my_cache_name.cache_size(), 86);
/// ```
#[macro_export]
macro_rules! cached_method {
    (
        #[self_id($self_id_n:ident)]
        $( #[key($overall_key_expr:expr)] )?
        $( #[__use_try($is_try:tt)] )?
        #[cache_expr($($access_cache:tt)*)]
        $( #[$($meta_tt:tt)*] )*
    //  ^~~~attributes~~~~^
        $vis:vis fn $name:ident (
            &mut self
            $(, $arg_name:ident : $arg_ty:ty )* $(,)?
    //      ^~~~~~~~~~~~~~~argument list!~~~~~~~~~~~^
            )
            $( -> $ret_ty:ty )?
    //      ^~~~return type~~~^
            { $($tt:tt)* }
     ) => {
        $(#[$($meta_tt)*])*
        $vis fn $name(&mut self, $($arg_name : $arg_ty,)* ) $(-> $ret_ty)* {
            use cached::Cached;

            #[allow(unused_macros)]
            macro_rules! make_key {
                (#[key($overall_key_expr_sub:expr)] $other:expr) => {
                    $overall_key_expr_sub
                };
                ($other:expr) => {
                    $other.clone()
                };
            }

            let key_val = make_key!($(#[key($overall_key_expr)])* ($($arg_name.clone(),)*));

            if let Some(out) = self.$($access_cache)*.cache_get(&key_val) {
                let out = out.clone();
                $(
                    let _ = $is_try;
                    let out = Ok(out);
                )*
                return out;
            }


            let $self_id_n = self;
            let out = (|| $(-> $ret_ty)* { $($tt)* })();

            $(
                let _ = $is_try;
                let out = out?;
            )*

            $self_id_n.$($access_cache)*.cache_set(key_val, out.clone());

            $(
                let _ = $is_try;
                let out = Ok(out);
            )*

            out
        }
    };

    (
        #[self_id($self_id_n:ident)]
        $( #[key($overall_key_expr:expr)] )?
        #[use_try]
        $($rest:tt)*
    ) => {
        cached_method! {
            #[self_id($self_id_n)]
            $( #[key($overall_key_expr)] )*
            #[__use_try(true)]
            $($rest)*
        }
    };

    (
        $( #[key($overall_key_expr:expr)] )?
        #[cache_expr($($access_cache:tt)*)]
        $($rest:tt)*
    ) => {
        cached_method! {
            #[self_id(self_name)]
            $( #[key($overall_key_expr)] )*
            #[cache_expr($($access_cache)*)]
            $($rest)*
        }
    }
}

/// Generates a (possibly recursive) cached lambda.
/// Uses `fn` syntax to allow proc macro parsing
///
/// Internally uses [cached_method](crate::cached_method).
///
/// Note that the lambda *must* be immutable, but it can capture.
///
/// # Example
///
/// ```
/// # use rr_util::cached_lambda;
/// # use cached::Cached;
/// use macro_rules_attribute::apply;
///
/// // # use rust_circuit::cached_lambda;
/// let zero_start = 0;
/// let one_start = 1;
///
/// #[apply(cached_lambda)]
/// fn func(a: u64) -> u64 {
///     match a {
///         0 => zero_start,
///         1 => one_start,
///         n => func(n - 1) + func(n - 2),
///     }
/// }
///
/// assert_eq!(func(85), 259695496911122585);
/// assert_eq!(func(10), 55);
///
/// #[apply(cached_lambda)]
/// #[key(a, u64)] // [optional] the expression to produce a hash key and the corresponding type
/// fn func_with_depth(a: u64, depth: u64) -> u64 {
///     match a {
///         0 => zero_start,
///         1 => one_start,
///         n => func_with_depth(n - 1, depth + 1) + func_with_depth(n - 2, depth + 1),
///     }
/// }
///
/// assert_eq!(func_with_depth(85, 7), 259695496911122585);
/// assert_eq!(func_with_depth(10, 7), 55);
///
/// // the input and output must be clonable
/// // and the input must implement Eq + Hash
/// #[apply(cached_lambda)]
/// fn rev_string(a: String) -> String {
///     a.chars().rev().collect()
/// }
///
/// assert_eq!(&rev_string("hi".to_owned()), "ih");
/// ```
///
/// Atm, this doesn't support caches other than unbounded, but would be easy to
/// implement. Also doesn't currently support accessing cache, but would also
/// be easy to add.
#[macro_export]
macro_rules! cached_lambda {
    {
        $(#[key($key_expr:expr, $key_ty:ty)])?
        $( #[__use_try($is_try:tt)] )?
        $( #[__cache_ty($cache_ty:ty)] )?
        $vis:vis fn $name:ident ($( $arg_name:ident : $arg_ty:ty ),* $(,)?) $( -> $ret_ty:ty )?
        { $($tt:tt)* }
    } => {
        paste::paste! {
            // TODO: hide this macros better!
            macro_rules! __make_key_type_impl {
                ({$sub_key_ty:ty} $sub_arg_ty:ty) => {
                    $sub_key_ty
                };
                ($sub_arg_ty:ty) => {
                    $sub_arg_ty
                };
            }

            macro_rules! __make_key_expr_impl {
                ({$sub_key_expr:expr} $sub_arg_expr:expr) => {
                    $sub_key_expr
                };
                ($sub_arg_expr:expr) => {
                    $sub_arg_expr
                };
            }

            macro_rules! __make_ret_type_impl {
                ($sub_cache_ty:ty) => { $sub_cache_ty };
                ($sub_cache_ty:ty; $sub_ret_ty:ty) => { $sub_cache_ty };
                (;  $sub_ret_ty:ty) => {
                    $sub_ret_ty
                };
                () => {
                    ()
                };
            }

            // Use lambda in holder pattern to allow for recursion.
            // Then, pass holder immutably so borrow checking can see that it's fine.

            // We can't avoid boxing because lambdas can't refer to themselves.
            // Hopefully inlined out?
            type [<$name:camel KeyType>] = __make_key_type_impl!($({$key_ty})* ($($arg_ty,)*));
            type [<$name:camel CacheType>] = __make_ret_type_impl!($($cache_ty)* $(; $ret_ty)*);
            type [<$name:camel RetType>] = __make_ret_type_impl!($(; $ret_ty)*);

            // TODO: it's possible to fix the lifetimes here so we can use references more easily
            // (but doesn't seem important for now...)
            struct [<$name:camel OtherFuncHolder>]<'a>(
                Box<dyn Fn(&mut [<$name:camel FuncHolder>], &Self, $($arg_ty,)*) -> [<$name:camel RetType>] + 'a>,
            );

            struct [<$name:camel FuncHolder>] {
                cache : $crate::caching::FastUnboundedCache<[<$name:camel KeyType>], [<$name:camel CacheType>]>,
            }

            impl [<$name:camel FuncHolder>] {
                #[apply($crate::cached_method)]
                #[self_id(self_)]
                #[key(__make_key_expr_impl!($({$key_expr})* ($($arg_name.clone(),)*)))]
                $(#[__use_try($is_try)])*
                #[cache_expr(cache)]
                fn call(&mut self, func_arg : &[<$name:camel OtherFuncHolder>],
                    $( $arg_name : $arg_ty,)*) $(-> $ret_ty)* {
                    func_arg.0(self_, func_arg, $($arg_name,)*)
                }
            }

            let other_holder = [<$name:camel OtherFuncHolder>](Box::new(
                |call_in: &mut [<$name:camel FuncHolder>], func_in: &[<$name:camel OtherFuncHolder>], $($arg_name,)*| {
                    #[allow(unused_mut, unused_variables)]
                    let mut $name = |$($arg_name,)*| call_in.call(func_in, $($arg_name,)*);
                    {
                        $($tt)*
                    }
                },
            ));

            // TODO: support other caching as needed
            let mut holder = [<$name:camel FuncHolder>] {
                cache: Default::default()
            };
            #[allow(unused_mut)]
            let mut $name = |$($arg_name,)*| holder.call(&other_holder, $($arg_name,)*);
        }
    };

    (
        $(#[key($key_expr:expr, $key_ty:ty)])?
        #[use_try]
        $vis:vis fn $name:ident ($( $arg_name:ident : $arg_ty:ty ),* $(,)?) -> Result<$ret_ty:ty>
        { $($tt:tt)* }
    ) => {
        cached_lambda! {
            $(#[key($key_expr, $key_ty)])*
            #[__use_try(true)]
            #[__cache_ty($ret_ty)]
            $vis fn $name ($( $arg_name : $arg_ty,)*) -> Result<$ret_ty>
            { $($tt)* }
        }
    };
}

/// faster because it uses FxHashMap instead of sip hash
#[derive(Clone, Debug)]
pub struct FastUnboundedCache<K, V>(pub HashMap<K, V>);

impl<K, V> Default for FastUnboundedCache<K, V> {
    fn default() -> Self {
        Self(HashMap::default())
    }
}

impl<K: std::hash::Hash + Eq, V> cached::Cached<K, V> for FastUnboundedCache<K, V> {
    fn cache_get(&mut self, k: &K) -> Option<&V> {
        self.0.get(k)
    }
    fn cache_get_mut(&mut self, k: &K) -> Option<&mut V> {
        self.0.get_mut(k)
    }
    fn cache_get_or_set_with<F: FnOnce() -> V>(&mut self, key: K, f: F) -> &mut V {
        self.0.entry(key).or_insert_with(f)
    }
    fn cache_set(&mut self, k: K, v: V) -> Option<V> {
        self.0.insert(k, v)
    }
    fn cache_remove(&mut self, k: &K) -> Option<V> {
        self.0.remove(k)
    }
    fn cache_clear(&mut self) {
        self.0.clear();
    }
    fn cache_reset(&mut self) {
        self.0 = HashMap::default();
    }
    fn cache_size(&self) -> usize {
        self.0.len()
    }
}
