#![feature(let_chains)]
//! Basic TODO: add more rust helpers + builders as needed!

pub mod iterative_matcher;
pub mod library;
pub mod matcher;
pub mod matcher_debug;
pub mod operations;
pub mod sampler;
pub mod transform;

pub use iterative_matcher::{
    new_traversal, restrict, restrict_sl, IterateMatchResults, IterativeMatcher,
    IterativeMatcherData, IterativeMatcherRc,
};
pub use matcher::{Matcher, MatcherData, MatcherFromPy, MatcherFromPyBase, MatcherRc};
pub use operations::{
    AnyFound, BoundAnyFound, BoundGetter, BoundUpdater, Expander, Getter, Updater,
};
pub use transform::{Transform, TransformData, TransformRc};
