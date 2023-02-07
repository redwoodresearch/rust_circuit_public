use std::fs;

use circuit_base::parsing::Parser;
use circuit_rewrites::circuit_optimizer::{
    optimize_circuit, OptimizationContext, OptimizationSettings,
};
use rr_util::rrfs::get_rrfs_dir;

#[test]
#[ignore]
fn benchmarking_circs() {
    fn item() {
        pyo3::prepare_freethreaded_python();

        dbg!("pre");
        let circuits: Vec<_> = fs::read_dir(format!("{}/ryan/compiler_benches/", get_rrfs_dir()))
            .unwrap()
            .filter_map(|d| {
                Parser {
                    tensors_as_random: true,
                    ..Default::default()
                }
                .parse_circuit(&fs::read_to_string(d.unwrap().path()).unwrap(), &mut None)
                .ok()
            })
            .collect();
        dbg!("post");

        let mut settings: OptimizationSettings = Default::default();
        settings.verbose = 2;
        settings.log_simplifications = true;
        let mut context = OptimizationContext::new_settings(settings);
        for circuit in circuits {
            let _result = optimize_circuit(circuit, &mut context);
            println!("{}", context.stringify_logs());
            // println!("{:?}", result.info().hash);
        }
    }
    std::thread::Builder::new()
        .stack_size(1024usize.pow(2) * 128)
        .spawn(item)
        .unwrap()
        .join()
        .unwrap();
}
