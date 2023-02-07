use circuit_base::{deep_map_unwrap, parsing::Parser, CircuitNode, CircuitRc};
use circuit_rewrites::circuit_optimizer::{
    optimize_circuit, OptimizationContext, OptimizationSettings,
};
use mimalloc::MiMalloc;
use rr_util::{
    name::NameInterner,
    rrfs::get_rrfs_dir,
    tensor_util::{TorchDevice, TorchDeviceDtypeOp, TorchDtype},
    timed,
};
use uuid::Uuid;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub fn main() {
    pyo3::prepare_freethreaded_python();
    let paths: Vec<_> = std::fs::read_dir(format!(
        // "{}/ryan/compiler_benches_easy",
        // "{}/ryan/compiler_benches",
        "{}/ryan/compiler_benches_paren",
        get_rrfs_dir()
    ))
    .unwrap()
    .map(|d| d.unwrap().path())
    .collect();
    let circuits: Vec<_> = paths
        .iter()
        .take(1)
        .map(|p| {
            Parser {
                tensors_as_random: true,
                tensors_as_random_device_dtype: TorchDeviceDtypeOp {
                    device: Some(TorchDevice::Cuda1),
                    dtype: Some(TorchDtype::float16),
                },
                allow_hash_with_random: true,
                ..Default::default()
            }
            .parse_circuit(&std::fs::read_to_string(p).unwrap(), &mut None)
            .unwrap()
        })
        .collect();
    // timed!(PrintHtmlOptions::default().repr(circuits.clone()));
    // timed!(PrintHtmlOptions::default().repr(circuits));
    let circuits_allnamed: Vec<CircuitRc> = circuits
        .iter()
        .map(|z| {
            deep_map_unwrap(z.clone(), |c| {
                c.rename(Some(Uuid::new_v4().to_string().into()))
            })
        })
        .collect();
    let mut opsettings = OptimizationSettings::default();
    opsettings.verbose = 2;
    opsettings.log_simplifications = true;
    opsettings.keep_all_names = true;
    for _ in 0..3 {
        let mut ctx = OptimizationContext::new_settings(opsettings.clone());
        for circ in &circuits_allnamed {
            println!("{:?}", circ.info().name);
            timed!(optimize_circuit(circ.clone(), &mut ctx).unwrap());
        }
        println!("{}", ctx.stringify_logs());
    }
    println!(
        "interned num {} len {}",
        NameInterner::num_interned_strings(),
        NameInterner::total_interned_string_len()
    );
}
