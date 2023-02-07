use std::fmt::Write;

use circuit_rewrites::deep_rewrite::SimpFnSubset;
use itertools::Itertools;

fn indent(x: String) -> String {
    let tab = " ".repeat(4);

    x.lines().map(|s| tab.clone() + s).join("\n")
}

fn to_doc_string(x: String) -> String {
    indent(format!("\"\"\"\n{}\n\"\"\"", x))
}

pub fn main() {
    let mut out = String::new();
    writeln!(
        &mut out,
        "# to generate below 3 fns, `cargo run -p circuit_rewrites --bin print_simp_fn_stub_part`"
    )
    .unwrap();
    writeln!(
        &mut out,
        "@staticmethod\ndef compiler_default() -> SimpFnSubset:\n{}",
        to_doc_string(format!(
            "Get compiler default simp fns. This is::\n{}",
            indent(SimpFnSubset::compiler_default().none_repr())
        ))
    )
    .unwrap();
    writeln!(
        &mut out,
        "@staticmethod\ndef default() -> SimpFnSubset:\n{}",
        to_doc_string(format!(
            "Get default simp fns. This is::\n{}",
            indent(SimpFnSubset::default().none_repr())
        ))
    )
    .unwrap();

    let all_args = SimpFnSubset::arg_fmt(|_| Some(": Optional[bool] = None".to_owned()), false);
    let tab = " ".repeat(4);
    writeln!(
        &mut out,
        "def set(\n{}self,\n{}\n) -> SimpFnSubset: ...",
        tab, all_args
    )
    .unwrap();
    println!("{}", indent(out));
}
