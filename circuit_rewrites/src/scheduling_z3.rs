use std::cmp::max;

use circuit_base::print::oom_fmt;
use rr_util::timed;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use z3::{
    self,
    ast::{Bool, Int},
    *,
};

use crate::{scheduled_execution::SchedulingOOMError, scheduling_alg::Dag};

pub fn schedule_dag(
    dag: &Dag,
    verbose: usize,
    mem_limit: usize,
    num_mem_chunks: usize,
) -> Result<Vec<usize>, SchedulingOOMError> {
    let cfg = Config::new();
    let ctx = Context::new(&cfg);
    let opt = Solver::new(&ctx);

    let n_time_slots = dag.node_costs.len() + 1;
    let mem_chunk_size = mem_limit / num_mem_chunks;
    let ctx_ref = &ctx;
    let liveness: HashMap<(usize, usize), z3::ast::Bool> = (0..dag.node_costs.len())
        .flat_map(|i| {
            (0..n_time_slots + 1).map(move |t| {
                (
                    (t, i),
                    Bool::new_const(ctx_ref, format!("live-{}-{}", t, i)),
                )
            })
        })
        .collect();
    let add_constraints = || {
        // key is (time,name)
        for i in 0..dag.node_costs.len() {
            opt.assert(&Bool::not(&liveness[&(0, i)]));
        }
        // Model that all variables are dead at time t = 0.
        // Model that all output are live at end
        for i in dag.get_outputs() {
            opt.assert(&liveness[&(n_time_slots - 1, i)]);
        }

        for t in 0..n_time_slots {
            for name in 0..dag.node_costs.len() {
                for child in &dag.children[name] {
                    opt.assert(&Bool::or(
                        &ctx,
                        &[
                            &liveness[&(t, *child as usize)],
                            &liveness[&(t, name)],
                            &Bool::not(&liveness[&(t + 1, name)]),
                        ],
                    ));
                    opt.assert(&Bool::or(
                        &ctx,
                        &[
                            &liveness[&(t + 1, *child as usize)],
                            &liveness[&(t, name)],
                            &Bool::not(&liveness[&(t + 1, name)]),
                        ],
                    ))
                }
            }
            opt.assert(&Bool::pb_le(
                &ctx,
                &dag.node_costs
                    .iter()
                    .enumerate()
                    .map(|(i, cost)| (&liveness[&(t, i)], (cost / mem_chunk_size) as i32))
                    .filter(|(_var, cost)| cost > &0)
                    .collect::<Vec<_>>(),
                num_mem_chunks as i32,
            ));
        }
    };
    timed!(add_constraints(), 10, verbose >= 2);

    // opt.minimize(&tend);
    timed!(opt.check(), 10, verbose >= 2);
    opt.get_model()
        .map(|model| {
            let mut result: Vec<usize> = vec![];
            let mut placed: HashSet<usize> = HashSet::default();
            for t in 1..n_time_slots + 1 {
                for i in 0..dag.node_hashes.len() {
                    let var = &liveness[&(t, i)];
                    let truth = model.eval(var, true).unwrap().as_bool().unwrap();
                    if truth && placed.insert(i) {
                        result.push(i)
                    }
                }
                // println!(
                //     "{}",
                //     (0..n_time_slots)
                //         .map(|t| {
                //             if liveness_result[&(t, i)] {
                //                 'O'
                //             } else {
                //                 '-'
                //             }
                //         })
                //         .collect::<String>()
                // )
            }

            result
        })
        .ok_or_else(|| {
            let mut costs_sorted: Vec<usize> = dag
                .node_costs
                .iter()
                .map(|x| x / mem_chunk_size)
                .filter(|x| x > &0)
                .collect();
            costs_sorted.sort();
            costs_sorted.reverse();
            SchedulingOOMError::Many {
                max_memory: mem_limit,
                memory_chunks: num_mem_chunks,
                node_memories: costs_sorted.clone(),
                string: format!(
                    "limit {} num_chunks {} node_chunks {:?}",
                    oom_fmt(mem_limit),
                    num_mem_chunks,
                    costs_sorted
                ),
            }
        })
}

pub fn schedule_dag_strategy_ints(
    dag: &Dag,
    verbose: usize,
    mem_limit: usize,
    num_mem_chunks: usize,
    timeout: usize,
    required_first: Option<usize>,
) -> Result<Vec<usize>, SchedulingOOMError> {
    let mut cfg = Config::new();
    cfg.set_timeout_msec(timeout as u64);
    let ctx = Context::new(&cfg);
    let opt = Solver::new(&ctx);

    let n_time_slots = dag.node_costs.len() + 1;
    let mem_chunk_size = max(mem_limit / num_mem_chunks, 1); // even if mem limit is 0, chunk size 1
    let ctx_ref = &ctx;

    let start_times: Vec<Int> = (0..dag.node_costs.len())
        .map(|i| Int::new_const(ctx_ref, format!("start {}", i)))
        .collect();
    let outputs = dag.get_outputs();
    let zero = Int::from_u64(ctx_ref, 0);
    let time_end = Int::from_u64(ctx_ref, (n_time_slots - 1) as u64);
    let end_times: Vec<Int> = (0..dag.node_costs.len())
        .map(|i| {
            if outputs.contains(&i) {
                time_end.clone()
            } else {
                Int::new_const(ctx_ref, format!("end {}", i))
            }
        })
        .collect();
    if let Some(rf) = required_first {
        opt.assert(&Int::le(&start_times[rf], &zero));
    }
    let add_constraints = || {
        // end after start, start ge 0
        for i in 0..dag.node_costs.len() {
            opt.assert(&Int::lt(&start_times[i], &end_times[i]));
            opt.assert(&Int::ge(&start_times[i], &zero));
        }

        for name in 0..dag.node_costs.len() {
            for child in &dag.children[name] {
                // for each start, ends of children must be after and starts must be earlier
                opt.assert(&start_times[name].lt(&end_times[*child as usize]));
                opt.assert(&start_times[name].gt(&start_times[*child as usize]));
            }
        }
        for t in 0..n_time_slots {
            let t_var = Int::from_u64(ctx_ref, t as u64);
            let boolies: Vec<(Bool, i32)> = dag
                .node_costs
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    (
                        Bool::and(
                            ctx_ref,
                            &[
                                &Int::le(&start_times[i], &t_var),
                                &Int::gt(&end_times[i], &t_var),
                            ],
                        ),
                        (*c / mem_chunk_size) as i32,
                    )
                })
                .collect();
            opt.assert(&Bool::pb_le(
                ctx_ref,
                &boolies.iter().map(|(b, c)| (b, *c)).collect::<Vec<_>>(),
                num_mem_chunks as i32,
            ));
        }
    };
    timed!(add_constraints(), 10, verbose >= 2);

    timed!(opt.check(), 10, verbose >= 2);
    opt.get_model()
        .map(|model| {
            let mut start_steps: Vec<(u64, bool, usize)> = (0..dag.node_costs.len())
                .map(|i| {
                    (
                        model.eval(&start_times[i], true).unwrap().as_u64().unwrap(),
                        required_first.is_none() || i != required_first.unwrap(),
                        i,
                    )
                })
                .collect();
            start_steps.sort();
            if verbose >= 5 {
                dbg!(&start_steps);
            }
            let result: Vec<usize> = start_steps.iter().map(|x| x.2).collect();
            if let Some(rf)=required_first && result[0]!=rf{
                panic!("required first not first, {:?} {:?}",required_first,&result);
            }
            result
        })
        .ok_or_else(|| {
            let mut costs_sorted: Vec<usize> = dag
                .node_costs
                .iter()
                .map(|x| x / mem_chunk_size)
                .filter(|x| x > &0)
                .collect();
            costs_sorted.sort();
            costs_sorted.reverse();
            SchedulingOOMError::Many {
                max_memory: mem_limit,
                memory_chunks: num_mem_chunks,
                string: format!(
                    "limit {} num_chunks {} node_chunks {:?}",
                    oom_fmt(mem_limit),
                    num_mem_chunks,
                    &costs_sorted
                ),
                node_memories: costs_sorted,
            }
        })
}
