use std::{
    collections::BTreeSet,
    fmt::{Display, Write},
    iter::{once, zip},
};

use anyhow::Result;
use circuit_base::{
    clicolor,
    print::{color, last_child_arrows, CliColor, PrintOptions},
    CircuitNode, CircuitNodeUnion, CircuitRc,
};
use itertools::Itertools;
use once_cell::unsync::Lazy;
use pyo3::prelude::*;
use rr_util::{
    python_println,
    util::{indent, indent_butfirst},
};
use rustc_hash::FxHashSet as HashSet;

use crate::{
    iterative_matcher::{per_child, ChainItem, IterativeMatcher},
    matcher::{Matcher, MatcherData},
    IterateMatchResults, IterativeMatcherData, IterativeMatcherRc,
};

const MAX_SELF_SINGLE_LINE_WIDTH: Lazy<usize> = Lazy::new(|| {
    std::env::var("MATCHER_PRINT_SELF_WIDTH")
        .ok()
        .and_then(|x| x.parse::<usize>().ok())
        .unwrap_or(80)
});
const MATCHER_PRINT_INDENT_WIDTH: usize = 4;

fn indented_if_multiline(x: &str) -> String {
    if x.contains("\n") {
        return format!("\n{}\n", indent(x.to_owned(), MATCHER_PRINT_INDENT_WIDTH));
    }
    x.to_owned()
}
fn maybe_indented_list(
    strings: Vec<String>,
    sep: &str,
    sep_after_line: bool,
    surrounding_newlines: bool,
) -> String {
    let total_length: usize = strings.iter().map(|s| s.len() + 3).sum();
    if total_length < *MAX_SELF_SINGLE_LINE_WIDTH && !strings.iter().any(|s| s.contains("\n")) {
        return strings.iter().join(sep);
    }
    let sep: String = if sep_after_line {
        format!("{}\n", sep)
    } else {
        "\n".to_owned()
    };
    let mut result = indent(
        strings
            .iter()
            .map(|s| indent_butfirst(s.clone(), MATCHER_PRINT_INDENT_WIDTH))
            .join(&sep),
        MATCHER_PRINT_INDENT_WIDTH,
    );
    if surrounding_newlines {
        result = format!("\n{}\n", result);
    }
    result
}

impl Display for IterativeMatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self.data() {
            IterativeMatcherData::Match(matcher) => format!("{}", matcher),
            IterativeMatcherData::Raw(_) => "RawIterativeMatcher".to_owned(),
            IterativeMatcherData::Term(term) => {
                if *term {
                    "Term".to_owned()
                } else {
                    "term next".to_owned()
                }
            }
            IterativeMatcherData::PyFunc(pyobject) => {
                format!(
                    "PyIterativeMatcher({})",
                    indented_if_multiline(&format!("{}", pyobject))
                )
            }
            IterativeMatcherData::Restrict(filter) => {
                let mut entries: Vec<String> = vec![];
                if filter.start_depth.is_some() || filter.end_depth.is_some() {
                    entries.push(format!(
                        "depth_range: {}:{}",
                        op_debug(&filter.start_depth),
                        op_debug(&filter.end_depth)
                    ));
                }
                if &**filter.iterative_matcher != &Matcher::true_matcher().into() {
                    entries.push(format!("matcher: {}", filter.iterative_matcher.0));
                }
                if &**filter.term_early_at != &Matcher::false_matcher() {
                    entries.push(format!("term_at: {}", filter.term_early_at.0));
                }
                if filter.term_if_matches {
                    entries.push("term_if_matches: true".to_owned());
                }
                if filter.depth != 0 {
                    entries.push(format!("cur_depth:{}", filter.depth));
                }
                format!(
                    "Restrict{{{}}}",
                    maybe_indented_list(entries, ", ", false, true)
                )
            }
            IterativeMatcherData::Children(childmatcher) => format!(
                "Children({}, [{}])",
                indented_if_multiline(&format!("{}", &childmatcher.iterative_matcher.0)),
                childmatcher
                    .child_numbers
                    .iter()
                    .map(|x| x.to_string())
                    .join(","),
            ),
            IterativeMatcherData::ModuleArg(module_arg_matcher) => format!(
                "ModuleArg({}, {})",
                indented_if_multiline(&format!("{}", &module_arg_matcher.module_matcher.0)),
                indented_if_multiline(&format!("{}", &module_arg_matcher.arg_sym_matcher.0)),
            ),
            IterativeMatcherData::SpecCircuit(matcher) => format!(
                "SpecCircuit({})",
                indented_if_multiline(&format!("{}", &matcher.0))
            ),
            IterativeMatcherData::NoModuleSpec(matcher) => format!(
                "NoModuleSpec({})",
                indented_if_multiline(&format!("{}", &matcher.0))
            ),
            IterativeMatcherData::Chains(chains) => {
                if chains.len() == 1 {
                    repr_chain_item(chains.iter().next().unwrap())
                } else {
                    format!(
                        "Any({})",
                        maybe_indented_list(
                            chains.iter().map(repr_chain_item).collect(),
                            ", ",
                            false,
                            true
                        )
                    )
                }
            }
            IterativeMatcherData::All(matchers) => {
                format!(
                    "All({})",
                    maybe_indented_list(
                        matchers.iter().map(|x| x.to_string()).collect(),
                        ", ",
                        false,
                        true
                    )
                )
            }
        };
        f.write_str(&s)
    }
}

#[pymethods]
impl IterativeMatcher {
    fn __repr__(&self) -> String {
        format!("{}", &self)
    }
}

fn repr_chain_item(chain_item: &ChainItem) -> String {
    if chain_item.rest.is_empty() {
        return format!("{}", &chain_item.first.0);
    }
    format!(
        "[{}]",
        maybe_indented_list(
            once(chain_item.first.clone())
                .chain(chain_item.rest.iter().cloned())
                .map(|x| format!("{}", &x.0))
                .collect(),
            " -> ",
            true,
            true
        )
    )
}

fn op_debug<T: std::fmt::Debug>(x: &Option<T>) -> String {
    match x {
        None => "".to_owned(),
        Some(s) => format!("{:?}", s),
    }
}

fn repr_single_many<T, F>(x: &BTreeSet<T>, f: F) -> String
where
    F: Fn(&T) -> String,
{
    if x.len() == 1 {
        return f(x.iter().next().unwrap());
    }
    format!(
        "{{{}}}",
        maybe_indented_list(x.iter().map(f).collect(), ", ", false, true)
    )
}

pub fn circuit_short_print(circuit: &CircuitRc) -> String {
    format!(
        "{} {} {}",
        match circuit.info().name {
            None => "None".to_owned(),
            Some(s) => format!("'{}'", s),
        },
        circuit.variant_string(),
        &circuit.hash_base16()[..6],
    )
}

impl Display for Matcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self.data() {
                MatcherData::Always(bool) =>
                    if *bool {
                        "Always".to_owned()
                    } else {
                        "Never".to_owned()
                    },
                MatcherData::Name(names) => repr_single_many(names, |x| format!("\"{}\"", x)),
                MatcherData::Type(types) => repr_single_many(types, |x| format!("{}", x)),
                MatcherData::Regex(regex) => format!(
                    "re{}'{}'",
                    if regex.escape_dot() { "-escdot" } else { "" },
                    regex.pattern()
                ),
                MatcherData::EqM(circuits) => repr_single_many(circuits, circuit_short_print),
                MatcherData::PyFunc(pyobject) => format!("{}", pyobject),
                MatcherData::Not(matcher) => format!("Not {}", &matcher.0),
                MatcherData::Any(any) => format!(
                    "Any({})",
                    maybe_indented_list(
                        any.iter().map(|x| format!("{}", &x.0)).collect(),
                        ", ",
                        false,
                        true
                    ),
                ),
                MatcherData::All(all) => format!(
                    "All({})",
                    maybe_indented_list(
                        all.iter().map(|x| format!("{}", &x.0)).collect(),
                        ", ",
                        false,
                        true
                    ),
                ),
                // MatcherData::Raw(RawMatcher),
                _ => "".to_owned(),
            }
        )
    }
}

#[pymethods]
impl Matcher {
    fn __repr__(&self) -> String {
        format!("{}", &self)
    }
}

// pub struct MatcherTreeifyItem {
//     circuit: CircuitRc,
//     matchers: BTreeSet<IterativeMatcherRc>,
//     children: Vec<Arc<MatcherTreeifyItem>>,
// }
// ///
// pub fn matcher_treeify_by_later_behavior(
//     circuit: CircuitRc,
//     matcher: IterativeMatcherRc,
// ) -> Result<MatcherTreeifyItem> {
//     type Key = (CircuitRc, IterativeMatcherRc);
//     let mut later_matched: HashMap<Key, BTreeSet<CircuitRc>> = Default::default();
//     fn recurse(
//         circuit: CircuitRc,
//         matcher: IterativeMatcherRc,
//         later_matched: &mut HashMap<Key, BTreeSet<CircuitRc>>,
//         upstream_stack: &mut Vec<Key>,
//     ) -> Result<()> {
//         let key = (circuit.clone(), matcher.clone());
//         if later_matched.contains_key(&key) {
//             return Ok(());
//         }
//         later_matched.insert(key.clone(), Default::default());
//         let matched: IterateMatchResults = matcher.match_iterate(circuit.clone())?;
//         if matched.found {
//             for us in upstream_stack.iter() {
//                 later_matched.get_mut(us).unwrap().insert(circuit.clone());
//             }
//         }
//         let child_matchers =
//             per_child(matched.updated, matcher.clone(), circuit.num_children());

//         upstream_stack.push(key);
//         for (i, (child, child_matcher)) in zip(circuit.children(), child_matchers).enumerate() {
//             if let Some(child_matcher) = child_matcher {
//                 let child_key = (child.clone(), child_matcher.clone());
//                 recurse(child, child_matcher, later_matched, upstream_stack)?;
//             }
//         }
//         upstream_stack.pop();

//         Ok(())
//     }
//     recurse(
//         circuit.clone(),
//         matcher.clone(),
//         &mut later_matched,
//         &mut vec![],
//     )?;
//     let topo = toposort_circuit(circuit);
//     let mut set_to_key: HashMap<BTreeSet<CircuitRc>, Vec<Key>> = Default::default();
//     for (k, v) in &later_matched {
//         set_to_key
//             .entry(v.clone())
//             .or_insert(Default::default())
//             .push(k.clone());
//     }
//     /// for
//     fn recurse_2(
//         key: Key,
//         matcher_set: &BTreeSet<CircuitRc>,
//         later_matched: &HashMap<Key, BTreeSet<CircuitRc>>,
//         set_to_key: &HashMap<BTreeSet<CircuitRc>, Vec<Key>>,
//     ) -> MatcherTreeifyItem {
//         let setty = later_matched[&key];
//     }
//     Ok(recurse_2(
//         (circuit.clone(), matcher.clone()),
//         &later_matched[&(circuit.clone(), matcher.clone())],
//         &later_matched,
//         &set_to_key,
//     ))
// }

#[pyfunction]
#[pyo3(signature=(circuit, matcher, print_options = Default::default()))]
pub fn repr_matcher_debug(
    circuit: CircuitRc,
    matcher: IterativeMatcherRc,
    print_options: PrintOptions,
) -> Result<String> {
    let mut result = "".to_owned();
    let mut seen: HashSet<(CircuitRc, IterativeMatcherRc)> = Default::default();
    fn recurse(
        circuit: CircuitRc,
        matcher: IterativeMatcherRc,
        last_child_stack: Vec<bool>,
        print_options: &PrintOptions,
        result: &mut String,
        seen: &mut HashSet<(CircuitRc, IterativeMatcherRc)>,
    ) -> Result<()> {
        let key = (circuit.clone(), matcher.clone());
        if !seen.insert(key) {
            write!(
                result,
                "{}'{}'...\n",
                last_child_arrows(&last_child_stack, true, print_options.arrows),
                circuit.info().name.unwrap_or("".into())
            )
            .unwrap();
            return Ok(());
        }
        let matched: IterateMatchResults = matcher.match_iterate(circuit.clone())?;
        let line_repr: String = format!(
            "{}{}{}{}\n",
            last_child_arrows(&last_child_stack, true, print_options.arrows),
            print_options.repr_line_info(circuit.clone())?,
            if matched.found {
                color(&" # Found", clicolor!(Green))
            } else {
                "".to_owned()
            },
            color(
                &format!(" # {}", &matcher.0),
                CliColor::new(matcher.__hash__() as usize)
            ),
        );
        result.push_str(&line_repr);

        let child_matchers = per_child(matched.updated, matcher.clone(), circuit.num_children());
        for (i, (child, child_matcher)) in zip(circuit.children(), child_matchers).enumerate() {
            if let Some(child_matcher) = child_matcher {
                let new_last_child_stack = last_child_stack
                    .iter()
                    .cloned()
                    .chain(once(i == circuit.num_children() - 1))
                    .collect();
                recurse(
                    child,
                    child_matcher,
                    new_last_child_stack,
                    print_options,
                    result,
                    seen,
                )?;
            } else {
                // reuslt.push_str("")
            }
        }
        Ok(())
    }
    recurse(
        circuit,
        matcher,
        vec![],
        &print_options,
        &mut result,
        &mut seen,
    )?;
    Ok(result)
}

#[pyfunction]
pub fn print_matcher_debug(
    circuit: CircuitRc,
    matcher: IterativeMatcherRc,
    print_options: PrintOptions,
) -> Result<()> {
    python_println!("{}", repr_matcher_debug(circuit, matcher, print_options)?);
    Ok(())
}

#[pyfunction]
#[pyo3(signature=(circuit, matcher, discard_old_name = false))]
pub fn append_matchers_to_names(
    circuit: CircuitRc,
    matcher: IterativeMatcherRc,
    discard_old_name: bool,
) -> Result<CircuitRc> {
    fn recurse(
        circuit: CircuitRc,
        matcher: Option<IterativeMatcherRc>,
        discard_old_name: bool,
    ) -> Result<CircuitRc> {
        if matcher.is_none() || circuit.is_symbol() {
            return Ok(circuit);
        }
        let matcher = matcher.unwrap();
        let matched = matcher.match_iterate(circuit.clone())?;
        let child_matchers = per_child(matched.updated, matcher.clone(), circuit.num_children());
        let new_children: Vec<CircuitRc> = zip(circuit.children(), child_matchers)
            .map(|(a, b)| recurse(a, b, discard_old_name))
            .collect::<Result<Vec<CircuitRc>>>()?;
        let result = circuit.map_children_unwrap_idxs(|i| new_children[i].clone());
        if discard_old_name {
            return Ok(result.rename(Some(format!("{}", &matcher.0).into())));
        }
        Ok(result.rename(
            circuit
                .info()
                .name
                .map(|n| format!("{} Matcher {}", n, &matcher.0).into()),
        ))
    }
    recurse(circuit, Some(matcher), discard_old_name)
}
// Filter{
//     m: Filter{
//         m: Chain(
//             Filter{ m: Always, term_if_matches true }
//             ->
//             "m1_pos0"
//         ),
//         term: {"a0.out_pos0", "a1.out_pos0", "a2.out_pos0", "m0_pos0", "m2_pos0"}
//     } -> "a0.out_pos0_head_1_sum",
//     term: {"a0.out_pos0", "a1.out_pos0", "a2.out_pos0", "m0_pos0", "m1_pos0", "m2_pos0"},
// }
