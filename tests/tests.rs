use circuit_base::{
    circuit_utils::total_flops, Add, Array, CircuitNode, Einsum, GeneralFunction, Symbol,
};
use circuit_rewrites::{
    algebraic_rewrite::einsum_nest_path,
    circuit_optimizer::optimize_and_evaluate,
    diag_rewrite::{diags_intersection, diags_union},
};
use rand::Rng;
use rr_util::{
    opt_einsum::{get_disconnected_sugraphs, get_int_to_tensor_appearance, EinsumSpec},
    sv,
    tensor_util::TorchDeviceDtypeOp,
    timed_value, tu8v,
};
use uuid::Uuid;

// real tests are in python calling rust, these are for debugging
#[test]
fn test_basic() {
    let examples = [
        // EinsumSpec {
        //     input_ints: vec![vec![0, 1, 2], vec![0, 1], vec![2, 3], vec![0], vec![3, 4]],
        //     output_ints: vec![4],
        //     int_sizes: vec![2, 3, 4, 5, 6, 7],
        // },
        // EinsumSpec {
        //     input_ints: vec![vec![0, 1, 2], vec![0, 1, 3, 2], vec![3]],
        //     output_ints: vec![0, 1, 3],
        //     int_sizes: vec![30000, 8, 32, 35],
        // },
        // EinsumSpec {
        //     input_ints: vec![vec![0, 1, 2], vec![0, 1, 3, 2], vec![3]],
        //     output_ints: vec![0, 1, 3],
        //     int_sizes: vec![30000, 8, 32, 35],
        // },
        // EinsumSpec {
        //     input_ints: vec![vec![0, 0, 0], vec![],vec![1, 0, 2, 0]],
        //     output_ints: vec![0,],
        //     int_sizes: vec![4,2,2],
        // },
        EinsumSpec {
            input_ints: vec![vec![0], vec![1], vec![], vec![2], vec![1, 1], vec![3]],
            output_ints: vec![1],
            int_sizes: vec![2, 4, 1, 5],
        },
    ];
    // (((0,), (1,), (), (2,), (1, 1), (3,)), (1,), (torch.Size([2]), torch.Size([4]), (), torch.Size([1]), torch.Size([4, 4]), (5,)))
    // (((0, 0, 0), (), (1, 0, 2, 0)), (0,), (torch.Size([4, 4, 4]), torch.Size([]), (2, 4, 2, 4)))
    // ('abc,abdc,d->abd', {0: 32768, 1: 8, 2: 32, 3: 35})
    for example in examples.iter() {
        let opted = example.optimize_dp(None, None, None);

        println!("result {:?}", opted);
    }
}

#[test]
fn test_bloom_filter() {
    let mut random_data: Vec<u64> = vec![0; 128];
    let random_data_2: &mut [u64] = &mut random_data;
    rand::thread_rng().fill(random_data_2);
    let mut bloomy = circuit_rewrites::deep_rewrite::CircBloomFilter::default();
    for (i, r) in random_data.iter().enumerate() {
        bloomy.insert(*r);
        let mut false_positives = 0;
        for (j, r2) in random_data.iter().enumerate() {
            if j <= i {
                assert!(bloomy.contains(*r2));
            } else if bloomy.contains(*r2) {
                false_positives += 1;
            }
        }
        let ratio: f64 = (false_positives as f64) / i as f64;
        println!("n {} ratio {}", i, ratio);
    }
}

#[test]
#[ignore] // test is slow
fn test_worst_case() {
    // the test is that this halts
    assert!(core::mem::size_of::<usize>() == 8);
    let n_operands = 40;
    let operand_width = 40;
    let n_ints = 40;
    let example = EinsumSpec {
        input_ints: (0..n_operands)
            .map(|_i| {
                (0..operand_width)
                    .map(|_j| rand::thread_rng().gen_range(0..n_ints))
                    .collect()
            })
            .collect(),
        output_ints: vec![0, 1, 1, 2, 3, 4],
        int_sizes: (0..n_ints).collect(),
    };
    println!("have einspec {:?}", example);
    // ('abc,abdc,d->abd', {0: 32768, 1: 8, 2: 32, 3: 35})
    let opted = example.optimize_dp(None, None, Some(500));

    println!("result {:?}", opted);
}

#[test]
fn test_subgraph() {
    let example: Vec<Vec<usize>> = vec![
        vec![0],
        vec![],
        vec![0],
        vec![0, 1, 2],
        vec![0, 3, 4],
        vec![5],
        vec![6],
        vec![],
        vec![6, 1],
        vec![7],
        vec![2, 8],
        vec![8],
        vec![9, 10, 8],
        vec![9, 10],
        vec![11],
        vec![],
        vec![11, 3],
        vec![12],
        vec![4, 13],
        vec![13],
        vec![9, 14, 13],
    ];
    let appear = get_int_to_tensor_appearance(&example);
    println!("appear {:?}", appear);
    let sg = get_disconnected_sugraphs(&example, &appear);
    println!("sg {:?}", sg);
}

#[test]
fn test_generalfuction() {
    pyo3::prepare_freethreaded_python();
    let circuit = Einsum::nrc(
        vec![(
            Add::nrc(
                vec![
                    Array::randn_named(sv![2, 3, 4], None, TorchDeviceDtypeOp::default()).rc(),
                    GeneralFunction::new_by_name(
                        vec![
                            Array::randn_named(sv![2, 3, 4], None, TorchDeviceDtypeOp::default())
                                .rc(),
                        ],
                        "sigmoid".into(),
                        None,
                    )
                    .unwrap()
                    .rc(),
                ],
                None,
            ),
            tu8v![0, 1, 2],
        )],
        tu8v![0, 1],
        None,
    );
    optimize_and_evaluate(circuit, Default::default()).unwrap();
}

#[test]
fn test_diags_intersection_union() {
    let ex = vec![tu8v![0, 1, 0, 0], tu8v![0, 1, 0, 1]];
    let inter = diags_intersection(&ex);
    dbg!(&inter);
    let union = diags_union(&ex);
    dbg!(&union);
}

#[test]
// #[ignore]
fn test_sweep_settings_einsum_opt() {
    #[rustfmt::skip]
    let einsum_specs = vec![EinsumSpec{input_ints:vec![vec![0,1,2],vec![0,3,4],vec![0,3,5],vec![6,2,7],vec![8,9,10],vec![11,7,12],vec![4,13],vec![14,15],vec![16,14],vec![5,13],vec![13,9],vec![17,14],vec![0],vec![12,18],vec![17,18],vec![10,18],vec![16,18],vec![17],vec![16],vec![5],vec![4],vec![5],vec![4],vec![17],vec![16],vec![8,19],vec![11,20],vec![3,21],vec![1,22],vec![19,23],vec![20,24],vec![6,25],vec![1,25],vec![21,26],vec![22,27],vec![23,26],vec![24,27],vec![],vec![]], output_ints:vec![15], int_sizes:vec![32768,32,384,32,384,384,32,384,9,384,384,9,384,768,768,384,384,384,60,32,32,9,9,9,9,5,4,4]},EinsumSpec{input_ints:vec![vec![0,1,2],vec![0,3,4],vec![5,1,6],vec![7,4,8],vec![9,10,11],vec![12,8,13],vec![2,14],vec![15,16],vec![17,15],vec![6,14],vec![14,10],vec![18,15],vec![5],vec![13,19],vec![18,19],vec![11,19],vec![17,19],vec![18],vec![17],vec![6],vec![2],vec![6],vec![2],vec![18],vec![17],vec![9,20],vec![12,21],vec![1,22],vec![3,23],vec![20,24],vec![21,25],vec![7,26],vec![3,26],vec![22,27],vec![23,28],vec![24,27],vec![25,28],vec![],vec![],vec![]], output_ints:vec![16], int_sizes:vec![32768,32,384,32,384,32768,384,32,384,9,384,384,9,384,768,768,384,384,384,60,32,32,9,9,9,9,5,4,4]},EinsumSpec{input_ints:vec![vec![0,1,2],vec![0,1,3],vec![4,5,6],vec![7,6,8],vec![3,9],vec![2,9],vec![9,10],vec![11,12,13],vec![14,15],vec![11,16,17],vec![11,12,14],vec![4],vec![1,11,5],vec![13,18],vec![8,19],vec![17,18],vec![10,19],vec![13],vec![17],vec![2],vec![3],vec![11,16],vec![2],vec![3],vec![13],vec![17],vec![7,20],vec![5,20],vec![1],vec![1],vec![18,21],vec![19,21],vec![],vec![],vec![]], output_ints:vec![15], int_sizes:vec![32768,32,384,384,32768,32,384,32,384,768,384,8,48,384,384,384,48,384,2,2,5,2]},EinsumSpec{input_ints:vec![vec![0,1,2],vec![0,3,4],vec![0,3,5],vec![6,2,7],vec![4,8],vec![5,8],vec![8,9],vec![10,11,12],vec![13,14],vec![10,15,16],vec![10,11,13],vec![0],vec![3,10,1],vec![12,17],vec![7,18],vec![16,17],vec![9,18],vec![12],vec![16],vec![5],vec![4],vec![10,15],vec![5],vec![4],vec![12],vec![16],vec![6,19],vec![1,19],vec![3],vec![3],vec![17,20],vec![18,20],vec![],vec![]], output_ints:vec![14], int_sizes:vec![32768,32,384,32,384,384,32,384,768,384,8,48,384,384,384,48,384,2,2,5,2]},EinsumSpec{input_ints:vec![vec![0,1,2],vec![0,1,3],vec![4,5,6],vec![7,6,8],vec![9,10,11],vec![12,8,13],vec![3,14],vec![15,16],vec![17,15],vec![2,14],vec![14,10],vec![18,15],vec![4],vec![13,19],vec![18,19],vec![11,19],vec![17,19],vec![18],vec![17],vec![2],vec![3],vec![2],vec![3],vec![18],vec![17],vec![9,20],vec![12,21],vec![1,22],vec![5,23],vec![20,24],vec![21,25],vec![7,26],vec![5,26],vec![22,27],vec![23,28],vec![24,27],vec![25,28],vec![],vec![],vec![]], output_ints:vec![16], int_sizes:vec![32768,32,384,384,32768,32,384,32,384,9,384,384,9,384,768,768,384,384,384,60,32,32,9,9,9,9,5,4,4]},EinsumSpec{input_ints:vec![vec![0,1,2],vec![0,3,4],vec![5,1,6],vec![7,4,8],vec![9,10,11],vec![12,8,13],vec![6,14],vec![15,16],vec![17,15],vec![2,14],vec![14,10],vec![18,15],vec![5],vec![13,19],vec![18,19],vec![11,19],vec![17,19],vec![18],vec![17],vec![2],vec![6],vec![2],vec![6],vec![18],vec![17],vec![9,20],vec![12,21],vec![1,22],vec![3,23],vec![20,24],vec![21,25],vec![7,26],vec![3,26],vec![22,27],vec![23,28],vec![24,27],vec![25,28],vec![],vec![],vec![]], output_ints:vec![16], int_sizes:vec![32768,32,384,32,384,32768,384,32,384,9,384,384,9,384,768,768,384,384,384,60,32,32,9,9,9,9,5,4,4]},EinsumSpec{input_ints:vec![vec![0,1,2],vec![0,3,4],vec![5,1,6],vec![7,4,8],vec![2,9],vec![6,9],vec![9,10],vec![11,12,13],vec![14,15],vec![11,16,17],vec![11,12,14],vec![5],vec![1,11,3],vec![13,18],vec![8,19],vec![17,18],vec![10,19],vec![13],vec![17],vec![6],vec![2],vec![11,16],vec![6],vec![2],vec![13],vec![17],vec![7,20],vec![3,20],vec![1],vec![1],vec![18,21],vec![19,21],vec![],vec![],vec![]], output_ints:vec![15], int_sizes:vec![32768,32,384,32,384,32768,384,32,384,768,384,8,48,384,384,384,48,384,2,2,5,2]},EinsumSpec{input_ints:vec![vec![0,1,2],vec![0,3,4],vec![5,1,6],vec![7,4,8],vec![6,9],vec![2,9],vec![9,10],vec![11,12,13],vec![14,15],vec![11,16,17],vec![11,12,14],vec![5],vec![1,11,3],vec![13,18],vec![8,19],vec![17,18],vec![10,19],vec![13],vec![17],vec![2],vec![6],vec![11,16],vec![2],vec![6],vec![13],vec![17],vec![7,20],vec![3,20],vec![1],vec![1],vec![18,21],vec![19,21],vec![],vec![],vec![]], output_ints:vec![15], int_sizes:vec![32768,32,384,32,384,32768,384,32,384,768,384,8,48,384,384,384,48,384,2,2,5,2]},EinsumSpec{input_ints:vec![vec![0,1,2],vec![0,3,4],vec![5,4,6],vec![7,2,8],vec![9,8,10],vec![11,6,12],vec![13,14],vec![15,13],vec![16,13],vec![0],vec![12,17],vec![16,17],vec![10,17],vec![15,17],vec![14],vec![16],vec![15],vec![14],vec![16],vec![15],vec![14],vec![9,18],vec![11,19],vec![1,20],vec![3,21],vec![18,22],vec![19,23],vec![5,24],vec![3,24],vec![7,25],vec![1,25],vec![20,26],vec![21,27],vec![22,26],vec![23,27],vec![],vec![]], output_ints:vec![], int_sizes:vec![32768,32,384,32,384,32,384,32,384,9,384,9,384,768,384,384,384,60,32,32,9,9,9,9,5,5,4,4]}];
    for spec in &einsum_specs {
        let multiple = 70;
        let mut r = vec![];
        for hash_map_cap in 1..10 {
            let einsum = einspec_to_symbol_circuit(&spec);
            let (path, time) =
                timed_value!(spec.optimize(None, None, Some(hash_map_cap * multiple), None));
            let path = path.unwrap();
            let circ = einsum_nest_path(&einsum, path);
            let cost = total_flops(circ.rc()).to_u64_digits()[0];
            r.push((time, cost));
            // println!("cap {} cost {}", hash_map_cap * multiple, cost);
        }
        let max_time = r.iter().map(|x| x.0.as_nanos()).max().unwrap();
        let max_cost = r.iter().map(|x| x.1).max().unwrap();
        let r2: Vec<_> = r
            .into_iter()
            .map(|(time, cost)| {
                (
                    time.as_nanos() as f64 / max_time as f64,
                    cost as f64 / max_cost as f64,
                )
            })
            .collect();
        println!("{:?}", r2);
    }
}
pub fn einspec_to_symbol_circuit(einspec: &EinsumSpec) -> Einsum {
    Einsum::new(
        einspec
            .input_ints
            .iter()
            .map(|ints| {
                (
                    Symbol::nrc(
                        ints.iter().map(|i| einspec.int_sizes[*i]).collect(),
                        Uuid::new_v4(),
                        None,
                    ),
                    ints.iter().map(|i| *i as u8).collect(),
                )
            })
            .collect(),
        einspec.output_ints.iter().map(|i| *i as u8).collect(),
        None,
    )
}
