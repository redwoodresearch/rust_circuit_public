use std::{
    fmt::Write,
    iter::{once, zip},
};

use anyhow::Result;
use circuit_base::{
    clicolor,
    print::{color, CliColor, PrintOptions},
    CircuitNode, CircuitNodeSelfOnlyHash, CircuitRc,
};
use itertools::Itertools;
use pyo3::prelude::*;
use rr_util::{
    name::Name,
    util::{indent, HashBytes},
};
use rustc_hash::FxHashMap as HashMap;
use uuid::uuid;

#[pyfunction]
#[pyo3(signature=(
    circuit,
    hash_child_count = true,
    hash_child_shapes = false,
    hash_name = true
))]
pub fn compute_self_hash(
    circuit: CircuitRc,
    hash_child_count: bool,
    hash_child_shapes: bool,
    hash_name: bool,
) -> HashBytes {
    let mut m = blake3::Hasher::new();
    for l in &circuit.info().shape {
        m.update(&l.to_le_bytes());
    }
    m.update(uuid!("92814fb9-3aa0-489c-b016-0f936941535b").as_bytes());
    if hash_name {
        m.update(Name::str_maybe_empty(circuit.info().name).as_bytes());
    }
    m.update(uuid!("6261daa8-0085-46f7-9f38-b085601fa628").as_bytes());
    if hash_child_count {
        m.update(&circuit.num_children().to_le_bytes());
    }
    m.update(uuid!("c7034aef-2179-4afa-9b90-c9abfcd1405d").as_bytes());
    if hash_child_shapes {
        for x in circuit.children() {
            for l in x.shape() {
                m.update(&l.to_le_bytes());
            }
            m.update(uuid!("17519b66-2332-450e-bdb2-bf893f8ed699").as_bytes());
        }
    }
    m.update(uuid!("e95b4d23-0077-4f57-a993-224454cb8570").as_bytes());
    circuit.compute_self_only_hash(&mut m);
    m.finalize().into()
}

#[pyfunction]
#[pyo3(signature=(
    new,
    old,
    options = Default::default(),
    require_child_count_same = true,
    require_child_shapes_same = false,
    require_name_same = true,
    print_legend = true,
    same_self_color = clicolor!(Blue),
    same_color = CliColor::NONE,
    new_color = clicolor!(Green),
    removed_color = clicolor!(Red)
))]
pub fn diff_circuits(
    new: CircuitRc,
    old: CircuitRc,
    options: PrintOptions,
    require_child_count_same: bool,
    require_child_shapes_same: bool,
    require_name_same: bool,
    print_legend: bool,
    same_self_color: CliColor,
    same_color: CliColor,
    new_color: CliColor,
    removed_color: CliColor,
) -> Result<String> {
    let mut options = options;
    options.bijection = false;
    let mut result = "".to_owned();
    let mut seen_diffs: HashMap<(CircuitRc, CircuitRc), String> = HashMap::default();
    let mut seen_independent: HashMap<CircuitRc, String> = HashMap::default();
    fn recurse(
        new: CircuitRc,
        old: CircuitRc,
        result: &mut String,
        options: &PrintOptions,
        last_child_stack: Vec<bool>,
        require_child_count_same: bool,
        require_child_shapes_same: bool,
        require_name_same: bool,
        seen_diffs: &mut HashMap<(CircuitRc, CircuitRc), String>,
        seen_indep: &mut HashMap<CircuitRc, String>,
        same_self_color: CliColor,
        same_color: CliColor,
        new_color: CliColor,
        removed_color: CliColor,
    ) -> Result<()> {
        let diffkey = (new.clone(), old.clone());
        if let Some(id) = seen_diffs.get(&diffkey) {
            result.push_str(&indent(id.clone(), last_child_stack.len() * 2));
            result.push('\n');
            return Ok(());
        }
        let id = seen_diffs.len();

        if new == old {
            let mut new_options = options.clone();
            new_options.colorer = Some(PrintOptions::fixed_color(same_color));
            let idstr = color(&format!(" # same {}", id), same_color);
            let child_printed = new_options.repr(new.clone())?;
            let child_printed =
                once(&(child_printed.lines().next().unwrap().to_owned() + &idstr) as &str)
                    .chain(child_printed.lines().skip(1))
                    .join("\n");
            result.push_str(&indent(child_printed, last_child_stack.len() * 2));
            result.push_str("\n");
            seen_diffs.insert(diffkey.clone(), idstr);
            return Ok(());
        }
        if compute_self_hash(
            new.clone(),
            require_child_count_same,
            require_child_shapes_same,
            require_name_same,
        ) == compute_self_hash(
            old.clone(),
            require_child_count_same,
            require_child_shapes_same,
            require_name_same,
        ) {
            let line_prefix = if let Some(name) = new.info().name {
                name.string() + " "
            } else {
                "".to_owned()
            };

            result.push_str(&indent(
                color(
                    &format!("{}{}", line_prefix, options.repr_line_info(new.clone())?),
                    same_self_color,
                ),
                last_child_stack.len() * 2,
            ));
            let idstr = color(&format!(" # changed {}", id), same_self_color);
            result.push_str(&idstr);
            seen_diffs.insert(diffkey.clone(), idstr);
            result.push_str("\n");

            assert_eq!(new.num_children(), old.num_children());
            for (i, (new_child, old_child)) in zip(new.children(), old.children()).enumerate() {
                let new_child_stack: Vec<bool> = last_child_stack
                    .iter()
                    .cloned()
                    .chain(std::iter::once(i == new.num_children()))
                    .collect();
                recurse(
                    new_child,
                    old_child,
                    result,
                    options,
                    new_child_stack,
                    require_child_count_same,
                    require_child_shapes_same,
                    require_name_same,
                    seen_diffs,
                    seen_indep,
                    same_self_color,
                    same_color,
                    new_color,
                    removed_color,
                )?;
            }
            return Ok(());
        }
        let mut new_options = options.clone();
        new_options.colorer = Some(PrintOptions::fixed_color(new_color));
        result.push_str(&indent(
            new_options.repr(new.clone())?,
            last_child_stack.len() * 2,
        ));
        result.push_str("\n");

        let mut new_options = options.clone();
        new_options.colorer = Some(PrintOptions::fixed_color(removed_color));
        result.push_str(&indent(
            new_options.repr(old.clone())?,
            last_child_stack.len() * 2,
        ));
        result.push_str("\n");

        Ok(())
    }
    recurse(
        new,
        old,
        &mut result,
        &options,
        vec![],
        require_child_count_same,
        require_child_shapes_same,
        require_name_same,
        &mut seen_diffs,
        &mut seen_independent,
        same_self_color,
        same_color,
        new_color,
        removed_color,
    )?;
    if print_legend {
        if &result[result.len() - 1..] != "\n" {
            result.push('\n');
        }
        write!(
            result,
            "{} {} {} {}\n",
            color(&format!("{}: Same as original", same_color), same_color),
            color(
                &format!("{}: Children changed", same_self_color),
                same_self_color
            ),
            color(&format!("{}: New", new_color), new_color),
            color(&format!("{}: Removed", removed_color), removed_color)
        )
        .unwrap();
    }
    Ok(result)
}
