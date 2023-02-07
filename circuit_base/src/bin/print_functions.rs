use circuit_base::generalfunction::BASIC_SPEC_ITEMS;

pub fn main() {
    println!("# to generate below fns, `cargo run -p circuit_base --bin print_functions`");
    for (name, _, _) in BASIC_SPEC_ITEMS {
        println!(
            "def {}(circuit: Circuit, name: Optional[str] = None) -> GeneralFunction: ...",
            name
        );
    }
}
