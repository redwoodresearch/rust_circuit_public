#![feature(let_chains)]
#![feature(map_try_insert)]
#![feature(portable_simd)]
pub mod algebraic_rewrite;
pub mod batching;
pub mod canonicalize;
pub mod circuit_manipulation;
pub mod circuit_optimizer;
pub mod compiler_heuristics;
pub mod compiler_strip;
pub mod concat_rewrite;
pub mod debugging;
pub mod deep_rewrite;
pub mod diag_rewrite;
pub mod generalfunction_rewrite;
pub mod module_rewrite;
pub mod nb_rewrites;
pub mod sampling;
pub mod scatter_rewrite;
pub mod schedule_send;
pub mod scheduled_execution;
pub mod scheduling_alg;
pub mod scheduling_z3;
pub mod server;
