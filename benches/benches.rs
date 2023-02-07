#![feature(test)]
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

// #[cfg(test)]

use std::fs;

use circuit_base::{parsing::Parser, CircuitRc};
use circuit_rewrites::circuit_optimizer::{
    optimize_circuit, OptimizationContext, OptimizationSettings,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use itertools::izip;
use rr_util::{opt_einsum::EinsumSpec, rrfs::get_rrfs_dir, tensor_util::TorchDeviceDtypeOp, timed};
extern crate test;
use circuit_rewrites::scheduled_execution::circuit_to_schedule;
use rr_util::tensor_util::{TorchDevice, TorchDtype};
#[allow(unused)]
fn bench_schedule(circuits: &[CircuitRc]) {
    let mut settings: OptimizationSettings = Default::default();
    settings.verbose = 2;
    settings.max_single_tensor_memory = 20_000_000_000;
    settings.max_memory = 40_000_000_000;
    settings.max_memory_fallback = Some(140_000_000_000);
    settings.log_simplifications = true;

    for circuit in circuits {
        let mut context = OptimizationContext::new_settings(settings.clone());
        let result = optimize_circuit(circuit.clone(), &mut context);
        black_box(result);
        // println!("{}", context.stringify_logs());
        // println!("{:?}", result.info().hash);
    }
}

fn get_optimized(circuits: &[CircuitRc]) -> Vec<(CircuitRc, OptimizationContext)> {
    let mut settings: OptimizationSettings = Default::default();
    settings.verbose = 1;
    settings.max_single_tensor_memory = 20_000_000_000;
    settings.max_memory = 40_000_000_000;
    settings.max_memory_fallback = Some(140_000_000_000);
    settings.log_simplifications = true;

    circuits
        .iter()
        .map(|c| {
            let mut context = OptimizationContext::new_settings(settings.clone());
            (optimize_circuit(c.clone(), &mut context).unwrap(), context)
        })
        .collect()
}

// Used in commented-out code below
#[allow(dead_code)]
fn test_einsum_specs(einsum_specs: &Vec<EinsumSpec>) {
    for spec in einsum_specs {
        let _ = black_box(spec.optimize(None, None, None, None));
    }
}
// 2.138
fn criterion_benchmark(c: &mut Criterion) {
    pyo3::prepare_freethreaded_python();
    let paths: Vec<_> = fs::read_dir(format!(
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
            .parse_circuit(&fs::read_to_string(p).unwrap(), &mut None)
            .unwrap()
        })
        .collect();

    // Used in commented-out code below
    #[allow(unused_variables)]
    #[rustfmt::skip]
    let einsumspecs = vec![EinsumSpec{input_ints:vec![vec![0,1,2],vec![0,3,4],vec![0,3,5],vec![6,2,7],vec![8,9,10],vec![11,7,12],vec![4,13],vec![14,15],vec![16,14],vec![5,13],vec![13,9],vec![17,14],vec![0],vec![12,18],vec![17,18],vec![10,18],vec![16,18],vec![17],vec![16],vec![5],vec![4],vec![5],vec![4],vec![17],vec![16],vec![8,19],vec![11,20],vec![3,21],vec![1,22],vec![19,23],vec![20,24],vec![6,25],vec![1,25],vec![21,26],vec![22,27],vec![23,26],vec![24,27],vec![],vec![]], output_ints:vec![15], int_sizes:vec![32768,32,384,32,384,384,32,384,9,384,384,9,384,768,768,384,384,384,60,32,32,9,9,9,9,5,4,4]},EinsumSpec{input_ints:vec![vec![0,1,2],vec![0,3,4],vec![5,1,6],vec![7,4,8],vec![9,10,11],vec![12,8,13],vec![2,14],vec![15,16],vec![17,15],vec![6,14],vec![14,10],vec![18,15],vec![5],vec![13,19],vec![18,19],vec![11,19],vec![17,19],vec![18],vec![17],vec![6],vec![2],vec![6],vec![2],vec![18],vec![17],vec![9,20],vec![12,21],vec![1,22],vec![3,23],vec![20,24],vec![21,25],vec![7,26],vec![3,26],vec![22,27],vec![23,28],vec![24,27],vec![25,28],vec![],vec![],vec![]], output_ints:vec![16], int_sizes:vec![32768,32,384,32,384,32768,384,32,384,9,384,384,9,384,768,768,384,384,384,60,32,32,9,9,9,9,5,4,4]},EinsumSpec{input_ints:vec![vec![0,1,2],vec![0,1,3],vec![4,5,6],vec![7,6,8],vec![3,9],vec![2,9],vec![9,10],vec![11,12,13],vec![14,15],vec![11,16,17],vec![11,12,14],vec![4],vec![1,11,5],vec![13,18],vec![8,19],vec![17,18],vec![10,19],vec![13],vec![17],vec![2],vec![3],vec![11,16],vec![2],vec![3],vec![13],vec![17],vec![7,20],vec![5,20],vec![1],vec![1],vec![18,21],vec![19,21],vec![],vec![],vec![]], output_ints:vec![15], int_sizes:vec![32768,32,384,384,32768,32,384,32,384,768,384,8,48,384,384,384,48,384,2,2,5,2]},EinsumSpec{input_ints:vec![vec![0,1,2],vec![0,3,4],vec![0,3,5],vec![6,2,7],vec![4,8],vec![5,8],vec![8,9],vec![10,11,12],vec![13,14],vec![10,15,16],vec![10,11,13],vec![0],vec![3,10,1],vec![12,17],vec![7,18],vec![16,17],vec![9,18],vec![12],vec![16],vec![5],vec![4],vec![10,15],vec![5],vec![4],vec![12],vec![16],vec![6,19],vec![1,19],vec![3],vec![3],vec![17,20],vec![18,20],vec![],vec![]], output_ints:vec![14], int_sizes:vec![32768,32,384,32,384,384,32,384,768,384,8,48,384,384,384,48,384,2,2,5,2]},EinsumSpec{input_ints:vec![vec![0,1,2],vec![0,1,3],vec![4,5,6],vec![7,6,8],vec![9,10,11],vec![12,8,13],vec![3,14],vec![15,16],vec![17,15],vec![2,14],vec![14,10],vec![18,15],vec![4],vec![13,19],vec![18,19],vec![11,19],vec![17,19],vec![18],vec![17],vec![2],vec![3],vec![2],vec![3],vec![18],vec![17],vec![9,20],vec![12,21],vec![1,22],vec![5,23],vec![20,24],vec![21,25],vec![7,26],vec![5,26],vec![22,27],vec![23,28],vec![24,27],vec![25,28],vec![],vec![],vec![]], output_ints:vec![16], int_sizes:vec![32768,32,384,384,32768,32,384,32,384,9,384,384,9,384,768,768,384,384,384,60,32,32,9,9,9,9,5,4,4]},EinsumSpec{input_ints:vec![vec![0,1,2],vec![0,3,4],vec![5,1,6],vec![7,4,8],vec![9,10,11],vec![12,8,13],vec![6,14],vec![15,16],vec![17,15],vec![2,14],vec![14,10],vec![18,15],vec![5],vec![13,19],vec![18,19],vec![11,19],vec![17,19],vec![18],vec![17],vec![2],vec![6],vec![2],vec![6],vec![18],vec![17],vec![9,20],vec![12,21],vec![1,22],vec![3,23],vec![20,24],vec![21,25],vec![7,26],vec![3,26],vec![22,27],vec![23,28],vec![24,27],vec![25,28],vec![],vec![],vec![]], output_ints:vec![16], int_sizes:vec![32768,32,384,32,384,32768,384,32,384,9,384,384,9,384,768,768,384,384,384,60,32,32,9,9,9,9,5,4,4]},EinsumSpec{input_ints:vec![vec![0,1,2],vec![0,3,4],vec![5,1,6],vec![7,4,8],vec![2,9],vec![6,9],vec![9,10],vec![11,12,13],vec![14,15],vec![11,16,17],vec![11,12,14],vec![5],vec![1,11,3],vec![13,18],vec![8,19],vec![17,18],vec![10,19],vec![13],vec![17],vec![6],vec![2],vec![11,16],vec![6],vec![2],vec![13],vec![17],vec![7,20],vec![3,20],vec![1],vec![1],vec![18,21],vec![19,21],vec![],vec![],vec![]], output_ints:vec![15], int_sizes:vec![32768,32,384,32,384,32768,384,32,384,768,384,8,48,384,384,384,48,384,2,2,5,2]},EinsumSpec{input_ints:vec![vec![0,1,2],vec![0,3,4],vec![5,1,6],vec![7,4,8],vec![6,9],vec![2,9],vec![9,10],vec![11,12,13],vec![14,15],vec![11,16,17],vec![11,12,14],vec![5],vec![1,11,3],vec![13,18],vec![8,19],vec![17,18],vec![10,19],vec![13],vec![17],vec![2],vec![6],vec![11,16],vec![2],vec![6],vec![13],vec![17],vec![7,20],vec![3,20],vec![1],vec![1],vec![18,21],vec![19,21],vec![],vec![],vec![]], output_ints:vec![15], int_sizes:vec![32768,32,384,32,384,32768,384,32,384,768,384,8,48,384,384,384,48,384,2,2,5,2]},EinsumSpec{input_ints:vec![vec![0,1,2],vec![0,3,4],vec![5,4,6],vec![7,2,8],vec![9,8,10],vec![11,6,12],vec![13,14],vec![15,13],vec![16,13],vec![0],vec![12,17],vec![16,17],vec![10,17],vec![15,17],vec![14],vec![16],vec![15],vec![14],vec![16],vec![15],vec![14],vec![9,18],vec![11,19],vec![1,20],vec![3,21],vec![18,22],vec![19,23],vec![5,24],vec![3,24],vec![7,25],vec![1,25],vec![20,26],vec![21,27],vec![22,26],vec![23,27],vec![],vec![]], output_ints:vec![], int_sizes:vec![32768,32,384,32,384,32,384,32,384,9,384,9,384,768,384,384,384,60,32,32,9,9,9,9,5,5,4,4]}];

    let mut group = c.benchmark_group("all");
    // group.measurement_time(Duration::from_millis(10000));
    group.sample_size(10);
    // let dummydata: Vec<u8> = vec![2, 3, 122, 255, 240, 1, 1, 1, 1, 70];
    // group.bench_function("u8s", |b| {
    //     b.iter(|| {
    //         black_box(
    //             black_box(dummydata.clone())
    //                 .into_iter()
    //                 .collect::<U8Set>()
    //                 .into_iter()
    //                 .collect::<Vec<u8>>(),
    //         )
    //     })
    // });
    // group.bench_function("hsu8", |b| {
    //     b.iter(|| {
    //         black_box(
    //             black_box(dummydata.clone())
    //                 .into_iter()
    //                 .collect::<HashSet<u8>>()
    //                 .into_iter()
    //                 .collect::<Vec<u8>>(),
    //         )
    //     })
    // });
    // group.bench_function("einsum_opt", |b| {
    //     b.iter(|| {
    //         for _i in 0..1 {
    //             test_einsum_specs(black_box(&einsumspecs));
    //         }
    //     })
    // });

    // we could cache this to disk, but we instead run to ensure we optimize
    // scheduling against the latest version of everything else.
    // If this got too slow, we could change this to be cached.
    let optimized = timed!(get_optimized(&circuits[..]));

    for (c, (o, ctx), p) in izip!(circuits, optimized, paths) {
        println!("anything");
        let f_name = p.file_name().unwrap().to_str().unwrap();
        group.bench_function(format!("optimize_circuit_{}", f_name), |b| {
            b.iter(|| {
                for _i in 0..1 {
                    black_box(get_optimized(black_box(std::array::from_ref(&c))));
                }
            })
        });
        group.bench_function(format!("schedule_circuit_{}", f_name), |b| {
            b.iter(|| {
                black_box(circuit_to_schedule(
                    black_box(o.clone()),
                    &mut black_box(ctx.clone()),
                ))
                .unwrap();
            })
        });
    }

    // // cursed
    // Python::with_gil(|py| HASH_TENSOR.1.call(py, (), None)).unwrap();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

#[bench]
fn the_benchmarks(_b: &mut Bencher) {
    benches();

    Criterion::default().configure_from_args().final_summary();
}
