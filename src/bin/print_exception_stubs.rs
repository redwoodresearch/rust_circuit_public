use pyo3::{PyResult, Python};
use rust_circuit::error::print_exception_stubs;
pub fn main() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();
    println!("# to generate below exception stubs, `cargo run --bin print_exception_stubs`");
    Python::with_gil(|py| -> PyResult<()> {
        println!("{}", print_exception_stubs(py)?);
        Ok(())
    })
}
