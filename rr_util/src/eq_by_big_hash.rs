use std::mem;

use crate::util::HashBytes;

pub trait EqByBigHash {
    fn hash(&self) -> HashBytes;
    fn first_u64_bytes(&self) -> [u8; mem::size_of::<u64>()] {
        self.hash()[..mem::size_of::<u64>()].try_into().unwrap()
    }
    fn first_u64(&self) -> u64 {
        u64::from_le_bytes(self.first_u64_bytes())
    }
    fn first_i64(&self) -> i64 {
        i64::from_le_bytes(self.first_u64_bytes())
    }
}

#[macro_export]
macro_rules! impl_eq_by_big_hash {
    ($t:ty) => {
        impl PartialEq for $t {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                $crate::eq_by_big_hash::EqByBigHash::hash(self)
                    == $crate::eq_by_big_hash::EqByBigHash::hash(other)
            }
        }
        impl Eq for $t {}
        impl ::std::hash::Hash for $t {
            #[inline]
            fn hash<H: ::std::hash::Hasher>(&self, state: &mut H) {
                state.write(&$crate::eq_by_big_hash::EqByBigHash::first_u64_bytes(self));
            }
        }
    };
}

#[macro_export]
macro_rules! impl_ord_by_big_hash {
    ($t:ty) => {
        impl PartialOrd for $t {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<::std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for $t {
            #[inline]
            fn cmp(&self, other: &Self) -> ::std::cmp::Ordering {
                $crate::eq_by_big_hash::EqByBigHash::hash(self)
                    .cmp(&$crate::eq_by_big_hash::EqByBigHash::hash(other))
            }
        }
    };
}
#[macro_export]
macro_rules! impl_both_by_big_hash {
    ($t:ty) => {
        $crate::impl_eq_by_big_hash!($t);
        $crate::impl_ord_by_big_hash!($t);
    };
}
