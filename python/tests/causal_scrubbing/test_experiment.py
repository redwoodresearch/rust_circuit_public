from typing import Callable, Optional

import pytest
import torch

from interp.tools.indexer import TORCH_INDEXER as I
from rust_circuit import (
    Add,
    Array,
    Circuit,
    DiscreteVar,
    Einsum,
    Index,
    IterativeMatcher,
    PrintOptions,
    Regex,
    Scalar,
    TorchDeviceDtypeOp,
    cast_circuit,
    restrict,
)
from rust_circuit.causal_scrubbing.dataset import Dataset
from rust_circuit.causal_scrubbing.experiment import Experiment, ExperimentEvalSettings
from rust_circuit.causal_scrubbing.hypothesis import Correspondence, ExactSampler
from rust_circuit.causal_scrubbing.hypothesis import FuncSampler as FS
from rust_circuit.causal_scrubbing.hypothesis import (
    IllegalCorrespondenceError,
    InterpNode,
    UncondSampler,
    corr_root_matcher,
)
from rust_circuit.causal_scrubbing.testing_utils import IntDataset, loss_fn

label_cond = FS(lambda d: d.labels.value)


def run_experiment(
    c: Circuit,
    ds: Dataset,
    corr: Correspondence,
    assert_loss_fn: Optional[Callable[[float], bool]] = None,
    num_examples=1,
    num_seeds=10,
    treeify=True,
    check=True,
):
    for seed in range(num_seeds):
        ex = Experiment(c, ds, corr, random_seed=seed, check=check)
        eval_settings = ExperimentEvalSettings(device_dtype="cpu")
        scrubbed = ex.scrub(treeify=treeify, ref_ds_or_num=num_examples)
        out = scrubbed.evaluate(eval_settings)
        if assert_loss_fn is not None:
            loss = loss_fn(scrubbed.ref_ds, out)
            assert assert_loss_fn(loss), loss


def datum_loss(c: Circuit, dataset: Dataset):
    datum = Array(dataset.xs.value[0], name="ref")
    c = IterativeMatcher("xs").update(c, lambda _: datum)
    out = cast_circuit(c, TorchDeviceDtypeOp(dtype="float32", device="cpu")).evaluate()
    return loss_fn(dataset[0], out)


def num_input_tags(c: Circuit):
    m = IterativeMatcher(Regex(r"\.*_scrub_\.*"))
    c.print()
    return len(m.get(c))


# === Corr checks ===
def test_corr_well_defined_matches_many():
    xs = Array(torch.tensor([0.0, 1.0]), name="xs")
    first = Index(xs, I[0], name="x0")
    second = Index(xs, I[1], name="x0")
    C = Add(first, second, name="C")

    C_t = InterpNode(name="C", cond_sampler=UncondSampler())
    x0 = C_t.make_descendant(name="x0", cond_sampler=UncondSampler())

    to_C = corr_root_matcher

    corr = Correspondence()
    corr.add(C_t, to_C)
    corr.add(x0, to_C.chain("x0"))

    dataset = IntDataset.of_shape((1000, 2), lambda x: 2 * x[0] + 3 * x[1])

    run_experiment(C, dataset, corr)


def test_corr_well_defined_matches_none():
    xs = Array(torch.tensor([0.0, 1.0]), name="not xs")
    first = Index(xs, I[0], name="x0")
    C = Add(first, Scalar(5), name="C")

    C_t = InterpNode(name="Ct", cond_sampler=UncondSampler())
    corr = Correspondence()
    corr.add(C_t, corr_root_matcher)

    dataset = IntDataset.of_shape((1000, 2), lambda x: 2 * x[0] + 3 * x[1])
    with pytest.raises(IllegalCorrespondenceError):
        run_experiment(C, dataset, corr)

    xs = Array(torch.tensor([0.0, 1.0]), name="not xs")
    first = Index(xs, I[0], name="x0")
    C = Add(first, Scalar(5), name="C")

    C_t = InterpNode(name="Ct", cond_sampler=UncondSampler())
    corr = Correspondence()
    corr.add(C_t, corr_root_matcher)
    x0 = C_t.make_descendant(name="x0", cond_sampler=UncondSampler())
    corr.add(x0, corr_root_matcher.chain("not_x0"))

    with pytest.raises(IllegalCorrespondenceError):
        run_experiment(C, dataset, corr)


def test_corr_injective_simple():
    xs = Array(torch.tensor([0.0, 1.0]), name="xs")
    first = Index(xs, I[0], name="x0")
    second = Index(xs, I[1], name="x1")
    C = Add(first, second, name="C")

    C_t = InterpNode(name="C", cond_sampler=UncondSampler())
    x0_1 = C_t.make_descendant(name="x0_1", cond_sampler=UncondSampler())
    x0_2 = C_t.make_descendant(name="x0_2", cond_sampler=UncondSampler())

    to_C = corr_root_matcher

    corr = Correspondence()
    corr.add(C_t, to_C)
    corr.add(x0_1, to_C.chain("x0"))
    corr.add(x0_2, to_C.chain("x0"))

    dataset = IntDataset.of_shape((1000, 2), lambda x: 2 * x[0] + 3 * x[1])
    with pytest.raises(IllegalCorrespondenceError):
        run_experiment(C, dataset, corr)


def test_corr_injective_subsumed():
    xs = Array(torch.tensor([0.0, 1.0]), name="xs")
    first = Index(xs, I[0], name="x0")
    A = Einsum.scalar_mul(first, 2)
    C = Add(first, A, name="C")

    C_t = InterpNode(name="C", cond_sampler=UncondSampler())
    A = C_t.make_descendant(name="A", cond_sampler=UncondSampler())
    A_x0 = A.make_descendant(name="A_x0", cond_sampler=UncondSampler())
    x0 = C_t.make_descendant(name="x0", cond_sampler=UncondSampler())

    to_C = corr_root_matcher

    corr = Correspondence()
    corr.add(C_t, to_C)
    corr.add(A, to_C.chain("A"))
    corr.add(x0, to_C.chain("x0"))
    corr.add(A_x0, restrict(to_C.chain("x0"), term_early_at="A"))

    dataset = IntDataset.of_shape((1000, 2), lambda x: 2 * x[0] + 3 * x[1])
    with pytest.raises(IllegalCorrespondenceError):
        run_experiment(C, dataset, corr)


def test_corr_injective_checking_leaves_insufficient():
    # An example where the leaves of the interpretation map to different nodes in
    # the treeified model, but where intermediate interp nodes map to the same one!
    xs = Array(torch.tensor([0.0, 1.0]), name="xs")
    first = Index(xs, I[0], name="x0")
    second = Index(xs, I[1], name="x1")
    added = Add(first, second, name="x2")
    C = Add(added, name="C")

    C_t = InterpNode(name="C", cond_sampler=UncondSampler())
    added_1 = C_t.make_descendant(name="x2_1", cond_sampler=UncondSampler())
    added_2 = C_t.make_descendant(name="x2_2", cond_sampler=UncondSampler())
    x0 = added_1.make_descendant(name="x0", cond_sampler=UncondSampler())
    x1 = added_2.make_descendant(name="x1", cond_sampler=UncondSampler())

    to_C = corr_root_matcher
    to_added_1 = to_C.chain("x2")
    to_added_2 = restrict(to_C.chain("x2"), term_early_at={"x3"})

    corr = Correspondence()
    corr.add(C_t, to_C)
    corr.add(added_1, to_added_1)
    corr.add(added_2, to_added_2)
    corr.add(x0, to_added_1.chain("x0"))
    corr.add(x1, to_added_2.chain("x1"))

    dataset = IntDataset.of_shape((1000, 2), lambda x: 2 * x[0] + 3 * x[1])
    with pytest.raises(IllegalCorrespondenceError):
        run_experiment(C, dataset, corr)


def test_corr_injective_unused_direct():
    xs = Array(torch.tensor([0.0, 1.0]), name="xs")
    first = Index(xs, I[0], name="x0")
    second = Index(xs, I[1], name="x1")
    C = Add(first, second, name="C")

    C_t = InterpNode(name="C", cond_sampler=UncondSampler())
    x0_1 = C_t.make_descendant(name="x0_1", cond_sampler=UncondSampler())
    x0_2 = C_t.make_descendant(name="x0_2", cond_sampler=UncondSampler())

    to_C = corr_root_matcher

    corr = Correspondence()
    corr.add(C_t, to_C)
    corr.add(x0_1, to_C.chain("x0"))
    corr.add(x0_2, restrict(to_C.chain("x0"), term_early_at="x1"))

    dataset = IntDataset.of_shape((1000, 2), lambda x: 2 * x[0] + 3 * x[1])
    with pytest.raises(IllegalCorrespondenceError):
        run_experiment(C, dataset, corr)


def test_corr_tree_structure_same_paths_to_inputs():
    xs = Array(torch.tensor([0.0, 1.0]), name="xs")
    first = Index(xs, I[0], name="x0")
    C = Add(first, name="C")

    # Backwards!
    x0 = InterpNode(UncondSampler(), "x0")
    C_t = x0.make_descendant(name="C", cond_sampler=UncondSampler())

    to_C = corr_root_matcher

    corr = Correspondence()
    corr.add(C_t, to_C)
    corr.add(x0, to_C.chain("x0"))

    dataset = IntDataset.of_shape((1000, 2), lambda x: 2 * x[0] + 3 * x[1])
    with pytest.raises(IllegalCorrespondenceError):
        run_experiment(C, dataset, corr)


def test_corr_injective_separate_paths():
    xs = Array(torch.tensor([0.0, 1.0]), name="xs")
    first = Index(xs, I[0], name="x0")
    A = Add(first, name="A")
    B = Einsum.scalar_mul(first, 2, name="B")
    C = Add(A, B, name="C")

    C_t = InterpNode(name="C", cond_sampler=UncondSampler())
    x0_A = C_t.make_descendant(name="x0_A", cond_sampler=UncondSampler())
    x0_B = C_t.make_descendant(name="x0_B", cond_sampler=UncondSampler())

    to_C = corr_root_matcher

    corr = Correspondence()
    corr.add(C_t, to_C)
    corr.add(x0_A, to_C.chain("A").chain("x0"))
    corr.add(x0_B, to_C.chain("B").chain("x0"))

    dataset = IntDataset.of_shape((1000, 2), lambda x: 2 * x[0] + 3 * x[1])

    ex = Experiment(C, dataset, corr)
    ex.scrub(1, treeify=True, check=True)


def test_corr_tree_structure_respects_paths():
    xs = Array(torch.tensor([0.0, 1.0]), name="xs")
    first = Index(xs, I[0], name="x0")
    A = Add(first, name="A")
    B = Einsum.scalar_mul(first, 2, name="B")
    C = Add(A, B, name="C")

    C_t = InterpNode(name="C", cond_sampler=UncondSampler())
    A = C_t.make_descendant(name="A", cond_sampler=UncondSampler())
    x0 = A.make_descendant(name="x0", cond_sampler=UncondSampler())

    to_C = corr_root_matcher

    corr = Correspondence()
    corr.add(C_t, to_C)
    corr.add(A, to_C.chain("A"))
    corr.add(x0, to_C.chain("B").chain("x0"))  # not via A!

    dataset = IntDataset.of_shape((1000, 2), lambda x: 2 * x[0] + 3 * x[1])
    with pytest.raises(IllegalCorrespondenceError):
        run_experiment(C, dataset, corr)


def test_corr_tree_structure_only_if():
    # Check that an interp node's paths are extensions of the other's *only if* it is a child of the other
    xs = Array(torch.tensor([0.0, 1.0]), name="xs")
    first = Index(xs, I[0], name="x0")
    A = Einsum.scalar_mul(first, 2, name="A")
    B = Einsum.scalar_mul(A, 3, name="B")
    C = Einsum.scalar_mul(B, 5, name="C")

    # Construct interpretation
    C_t = InterpNode(name="C", cond_sampler=UncondSampler())
    A_t = C_t.make_descendant(name="At", cond_sampler=UncondSampler())
    B_t = C_t.make_descendant(name="Bt", cond_sampler=UncondSampler())

    to_C = corr_root_matcher

    corr = Correspondence()
    corr.add(C_t, to_C)
    to_B = to_C.chain("B")
    corr.add(B_t, to_B)
    corr.add(A_t, to_B.chain("A"))

    dataset = IntDataset.of_shape((1000, 2), lambda x: 2 * x[0] + 3 * x[1])
    with pytest.raises(IllegalCorrespondenceError):
        run_experiment(C, dataset, corr)


# === Unit tests ===

# filter out warnings about check = False
@pytest.mark.filterwarnings("ignore:You're")
def test_treeify():
    # Construct model
    xs = Array(torch.tensor([0.0, 1.0]), name="xs")
    first = Index(xs, I[0])
    second = Index(xs, I[1])
    A = Einsum.scalar_mul(first, 2, name="A")

    # unused
    dataset = IntDataset.of_shape((5, 2), lambda x: 2 * x[0] + 3 * x[1])

    # Case: child subsumes parent entirely
    C = Einsum.scalar_mul(A, 3, name="C")
    C_t = InterpNode(name="Ct", cond_sampler=ExactSampler())
    out = corr_root_matcher
    to_A = out.chain("A")
    corr = Correspondence()
    corr.add(C_t, out)
    corr.add(C_t.make_descendant(name="A", cond_sampler=ExactSampler()), to_A)
    ex = Experiment(C, dataset, corr, check=False)
    ref_ds = ex.make_ref_ds(1)
    ex.sample(ref_ds)
    circuit = ex.treeified()
    circuit.print()
    assert num_input_tags(circuit) == 1

    # Case: only some children are mapped
    B = Einsum.scalar_mul(second, 3, name="B")
    C = Add(A, B, name="C")
    ex = Experiment(C, dataset, corr)
    ref_ds = ex.make_ref_ds(1)
    ex.sample(ref_ds)
    circuit = ex.treeified()
    assert num_input_tags(circuit) == 2

    # Case: multiple children are mapped
    to_B = out.chain("B")
    corr.add(C_t.make_descendant(name="B", cond_sampler=ExactSampler()), to_B)
    ex = Experiment(C, dataset, corr)
    ref_ds = ex.make_ref_ds(1)
    ex.sample(ref_ds)
    circuit = ex.treeified()
    assert num_input_tags(circuit) == 2


def test_set_inputs():
    # setup
    xs = Array(torch.tensor([0.0, 1.0]), name="xs")
    first = Index(xs, I[0])
    second = Index(xs, I[1])
    A = Einsum.scalar_mul(first, 2, name="A")
    B = Einsum.scalar_mul(second, 3, name="B")
    C = Add(A, B, name="C")
    dataset = IntDataset.of_shape((5, 2), lambda x: 2 * x[0] + 3 * x[1])

    # Construct correspondence
    out = corr_root_matcher
    to_A = out.chain("A")
    corr = Correspondence()
    C_t = InterpNode(name="C", cond_sampler=UncondSampler())
    corr.add(C_t, out)
    corr.add(C_t.make_descendant(name="A", cond_sampler=UncondSampler()), to_A)

    # First treeify, then set inputs
    ex = Experiment(C, dataset, corr, random_seed=1)
    ref_ds = ex.make_ref_ds(1)
    sampled_inputs = ex.sample(ref_ds)
    circuit = ex.treeified()
    assert num_input_tags(circuit) == 2

    # Note this number depends on the sampler returning different datasets for C and A
    circuit = ex.wrap_in_var(circuit, ref_ds, DiscreteVar.uniform_probs_and_group(1))
    circuit = ex.replace_inputs(circuit, sampled_inputs)
    assert len(IterativeMatcher("xs").get(circuit)) == 2

    # Set inputs without treeifying
    ex = Experiment(C, dataset, corr, random_seed=1)
    ref_ds = ex.make_ref_ds(1)
    sampled_inputs = ex.sample(ref_ds)
    circuit = ex.wrap_in_var(ex.base_circuit, ref_ds, DiscreteVar.uniform_probs_and_group(1))
    circuit = ex.replace_inputs(circuit, sampled_inputs)
    assert len(IterativeMatcher("xs").get(circuit)) == 2


# === E2E tests ===


def test_identical_no_check():
    # Construct model
    xs = Array(torch.tensor([0.0, 1.0]), name="xs")
    first = Index(xs, I[0])
    second = Index(xs, I[1])
    A = Einsum.scalar_mul(first, 2.0, name="A")
    B = Einsum.scalar_mul(second, 3.0, name="B")
    C = Add(A, B, name="C")

    # Ground truth
    dataset = IntDataset.of_shape((100, 2), lambda x: 2 * x[0] + 3 * x[1])

    # Construct interpretation
    C_t = InterpNode(name="Ct", cond_sampler=label_cond)
    A_t = C_t.make_descendant(name="At", cond_sampler=FS(lambda d: IntDataset.unwrap(d).inp_tensor[:, 0]))
    B_t = C_t.make_descendant(name="Bt", cond_sampler=FS(lambda d: IntDataset.unwrap(d).inp_tensor[:, 1]))

    to_C = corr_root_matcher

    corr = Correspondence()
    corr.add(C_t, to_C)
    corr.add(A_t, to_C.chain("A"))
    corr.add(B_t, to_C.chain("B"))

    run_experiment(C, dataset, corr, lambda l: l == 0, check=False, num_examples=1)


def test_identical():
    # Construct model
    xs = Array(torch.tensor([0.0, 1.0]), name="xs")
    first = Index(xs, I[0])
    second = Index(xs, I[1])
    A = Einsum.scalar_mul(first, 2.0, name="A")
    B = Einsum.scalar_mul(second, 3.0, name="B")
    C = Add(A, B, name="C")

    # Ground truth
    dataset = IntDataset.of_shape((100, 2), lambda x: 2 * x[0] + 3 * x[1])

    # Construct interpretation
    C_t = InterpNode(name="Ct", cond_sampler=label_cond)
    A_t = C_t.make_descendant(name="At", cond_sampler=FS(lambda d: IntDataset.unwrap(d).inp_tensor[:, 0]))
    B_t = C_t.make_descendant(name="Bt", cond_sampler=FS(lambda d: IntDataset.unwrap(d).inp_tensor[:, 1]))

    to_C = corr_root_matcher

    corr = Correspondence()
    corr.add(C_t, to_C)
    corr.add(A_t, to_C.chain("A"))
    corr.add(B_t, to_C.chain("B"))

    run_experiment(C, dataset, corr, lambda l: l == 0, num_examples=1)


def test_explicit_inputs_okay():
    # Construct model
    xs = Array(torch.tensor([0.0, 1.0]), name="xs")
    first = Index(xs, I[0], "x0")
    second = Index(xs, I[1], "x1")
    C = Add(first, second, name="C")

    # Ground truth
    dataset = IntDataset.of_shape((100, 2), lambda x: x[0] + x[1])

    # Construct interpretation
    C_t = InterpNode(name="Ct", cond_sampler=label_cond)
    x0_t = C_t.make_descendant(name="x0t", cond_sampler=FS(lambda d: IntDataset.unwrap(d).inp_tensor[:, 0]))
    x1_t = C_t.make_descendant(name="x1t", cond_sampler=FS(lambda d: IntDataset.unwrap(d).inp_tensor[:, 1]))
    xs_x0_t = x0_t.make_descendant(name="x0_xs", cond_sampler=FS(lambda d: IntDataset.unwrap(d).inp_tensor[:, 0]))
    xs_x1_t = x1_t.make_descendant(name="x1_xs", cond_sampler=FS(lambda d: IntDataset.unwrap(d).inp_tensor[:, 1]))

    to_C = corr_root_matcher

    corr = Correspondence()
    corr.add(C_t, to_C)
    corr.add(x0_t, to_C.chain("x0"))
    corr.add(x1_t, to_C.chain("x1"))
    corr.add(xs_x0_t, to_C.chain("x0").chain("xs"))
    corr.add(xs_x1_t, to_C.chain("x1").chain("xs"))

    run_experiment(
        C, dataset, corr, lambda l: l == 0, num_examples=1, treeify=False
    )  # explicit inputs lead to failing treeified checks!


def test_spread_over_residual():
    # Construct model
    xs = Array(torch.tensor([0.0, 1.0]), name="xs")
    first = Index(xs, I[0], name="x1")
    second = Index(xs, I[1], name="x2")
    A = Add(first, second, name="A")
    C = Add(A, second, name="C")

    # Ground truth
    dataset = IntDataset.of_shape((1000, 2), lambda x: x[0] + 2 * x[1])

    # Check that just running model fails to find issue
    non_scrub_loss = datum_loss(C, dataset)
    assert non_scrub_loss < 0.01

    # Construct interpretation
    C_t = InterpNode(name="Ct", cond_sampler=label_cond)
    A_t = C_t.make_descendant(name="A", cond_sampler=FS(lambda d: IntDataset.unwrap(d).inp_tensor[:, 0]))
    x1 = A_t.make_descendant(name="x1", cond_sampler=FS(lambda d: IntDataset.unwrap(d).inp_tensor[:, 0]))
    x2 = C_t.make_descendant(name="x2", cond_sampler=FS(lambda d: IntDataset.unwrap(d).inp_tensor[:, 1]))

    to_C = corr_root_matcher
    to_A = to_C.chain("A")

    corr = Correspondence()
    corr.add(C_t, to_C)
    corr.add(A_t, to_A)
    corr.add(x1, to_A.chain("x1"))
    corr.add(x2, restrict(to_C.chain("x2"), term_early_at={"A"}))

    C.print()
    print(C.shape)

    run_experiment(C, dataset, corr, lambda l: l > 1, num_examples=50)


def test_spread_over_two_nodes():
    num_examples = 50

    # Construct model
    xs = Array(torch.tensor([0.0, 1.0]), name="xs")
    first = Index(xs, I[0])
    A = Einsum.scalar_mul(first, 2, name="A")
    B = Einsum.scalar_mul(first, 3, name="B")
    C = Add(A, B, name="C")

    # Ground truth
    dataset = IntDataset.of_shape((1000, 2), lambda x: 5 * x[0])

    # Check that just running model fails to find issue
    assert datum_loss(C, dataset) < 0.01

    # Construct interpretation
    C_t = InterpNode(name="Ct", cond_sampler=FS(lambda d: IntDataset.unwrap(d).inp_tensor[:, 0]))
    A_t = C_t.make_descendant(name="A", cond_sampler=FS(lambda d: IntDataset.unwrap(d).inp_tensor[:, 0]))

    to_C = corr_root_matcher
    to_A = to_C.chain("A")

    corr = Correspondence()
    corr.add(C_t, to_C)
    corr.add(A_t, to_A)

    run_experiment(C, dataset, corr, lambda l: l > 0.1, num_examples=num_examples)


def test_large_deviation():
    # Construct model
    xs = Array(torch.tensor([0.0, 1.0]), name="xs")
    first = Index(xs, I[0])
    second = Index(xs, I[1])
    A = Add(Einsum.scalar_mul(first, 2), second, name="A")
    B = Einsum.scalar_mul(second, 3, name="B")
    C = Add(A, B, name="C")

    # Ground truth
    dataset = IntDataset.of_shape((1000, 2), lambda x: 2 * x[0] + 3 * x[1])

    # Construct interpretation
    C_t = InterpNode(name="Ct", cond_sampler=label_cond)
    A_t = C_t.make_descendant(name="At", cond_sampler=FS(lambda d: IntDataset.unwrap(d).inp_tensor[:, 0]))
    B_t = C_t.make_descendant(name="Bt", cond_sampler=FS(lambda d: IntDataset.unwrap(d).inp_tensor[:, 1]))

    to_C = corr_root_matcher

    corr = Correspondence()
    corr.add(C_t, to_C)
    corr.add(A_t, to_C.chain("A"))
    corr.add(B_t, to_C.chain("B"))

    run_experiment(C, dataset, corr, lambda l: l > 1, num_examples=50)


def test_tiny_deviation():
    # Construct model
    xs = Array(torch.tensor([0.0, 1.0]), name="xs")
    first = Index(xs, I[0])
    second = Index(xs, I[1])
    A = Add(Einsum.scalar_mul(first, 2), Einsum.scalar_mul(second, 0.001), name="A")
    B = Einsum.scalar_mul(second, 3, name="B")
    C = Add(A, B, name="C")

    # Ground truth
    dataset = IntDataset.of_shape((1000, 2), lambda x: 2 * x[0] + 3 * x[1])

    # Construct interpretation
    C_t = InterpNode(name="Ct", cond_sampler=label_cond)
    A_t = C_t.make_descendant(name="At", cond_sampler=FS(lambda d: IntDataset.unwrap(d).inp_tensor[:, 0]))
    B_t = C_t.make_descendant(name="Bt", cond_sampler=FS(lambda d: IntDataset.unwrap(d).inp_tensor[:, 1]))

    to_C = corr_root_matcher

    corr = Correspondence()
    corr.add(C_t, to_C)
    corr.add(A_t, to_C.chain("A"))
    corr.add(B_t, to_C.chain("B"))

    run_experiment(C, dataset, corr, lambda l: l < 0.01, num_examples=1)


def test_intermediates_cancel():
    # Construct model
    xs = Array(torch.tensor([0.0, 1.0, 2.0]), name="xs")
    # TODO: fix indexes
    first = Index(xs, I[0])
    second = Index(xs, I[1])
    third = Index(xs, I[2])
    A = Add(first, third, name="A")
    B = Add.minus(second, third, name="B")
    C = Add(A, B, name="C")

    # Ground truth
    dataset = IntDataset.of_shape((1000, 3), lambda x: x[0] + x[1])

    # Check that just running model fails to find issue
    assert datum_loss(C, dataset) < 0.0001

    # Construct interpretation
    C_t = InterpNode(
        name="Ct",
        cond_sampler=FS(lambda d: IntDataset.unwrap(d).inp_tensor[:, 0] + IntDataset.unwrap(d).inp_tensor[:, 1]),
    )
    A_t = C_t.make_descendant(name="At", cond_sampler=FS(lambda d: IntDataset.unwrap(d).inp_tensor[:, 0]))
    B_t = C_t.make_descendant(name="Bt", cond_sampler=FS(lambda d: IntDataset.unwrap(d).inp_tensor[:, 1]))

    to_C = corr_root_matcher

    corr = Correspondence()
    corr.add(C_t, to_C)
    corr.add(A_t, to_C.chain("A"))
    corr.add(B_t, to_C.chain("B"))

    run_experiment(C, dataset, corr, lambda l: l > 1, num_examples=50)


def test_swapped_intermediates():
    # Construct model
    xs = Array(torch.tensor([0.0, 1.0]), name="xs")
    first = Index(xs, I[0])
    second = Index(xs, I[1])
    A = Add(Einsum.scalar_mul(first, 2), second, name="A")
    B = Add(first, Einsum.scalar_mul(second, 2), name="B")
    C = Add(A, B, name="C")

    # Ground truth
    dataset = IntDataset.of_shape((1000, 2), lambda x: 3 * x[0] + 3 * x[1])

    # Check that just running model fails to find issue TODO sample
    assert datum_loss(C, dataset) == 0

    # Construct interpretation
    C_t = InterpNode(
        name="Ct",
        cond_sampler=FS(
            lambda d: 2 * IntDataset.unwrap(d).inp_tensor[:, 0] + 3 * IntDataset.unwrap(d).inp_tensor[:, 1]
        ),
    )
    A_t = C_t.make_descendant(
        name="At",
        cond_sampler=FS(lambda d: IntDataset.unwrap(d).inp_tensor[:, 0] + 2 * IntDataset.unwrap(d).inp_tensor[:, 1]),
    )
    B_t = C_t.make_descendant(
        name="Bt",
        cond_sampler=FS(lambda d: IntDataset.unwrap(d).inp_tensor[:, 0] + IntDataset.unwrap(d).inp_tensor[:, 1]),
    )

    to_C = corr_root_matcher

    corr = Correspondence()
    corr.add(C_t, to_C)
    corr.add(A_t, to_C.chain("A"))
    corr.add(B_t, to_C.chain("B"))

    run_experiment(C, dataset, corr, lambda l: l > 1, num_examples=50)


def test_data_correlation():
    # Construct model
    xs = Array(torch.tensor([0.0, 1.0]), name="xs")
    first = Index(xs, I[0], name="x0")
    second = Index(xs, I[1], name="x1")
    C = Add(first, second, name="C")

    # Ground truth
    rands = torch.randint(10, size=(1000, 1))
    dataset = IntDataset({"xs": Array(torch.hstack((rands, 2 * rands)), "xs"), "labels": Array(3 * rands, "labels")})

    # Check that just running model fails to find issue
    assert datum_loss(C, dataset) == 0

    # Construct interpretation
    C_t = InterpNode(name="Ct", cond_sampler=FS(lambda d: IntDataset.unwrap(d).inp_tensor[:, 0]))
    x0_t = C_t.make_descendant(name="x0", cond_sampler=FS(lambda d: IntDataset.unwrap(d).inp_tensor[:, 0]))

    to_C = corr_root_matcher

    corr = Correspondence()
    corr.add(C_t, to_C)
    corr.add(x0_t, to_C.chain("x0"))

    run_experiment(C, dataset, corr, lambda l: l > 1, num_examples=50)


# filter out warnings about insufficient data
@pytest.mark.filterwarnings("ignore:not enough")
def test_unspecified_nodes():
    # Construct model
    xs = Array(torch.tensor([0.0, 1.0]), name="xs")
    first = Index(xs, I[0])
    second = Index(xs, I[1])
    A = Einsum.scalar_mul(first, 2, name="A")
    B = Einsum.scalar_mul(second, 3, name="B")
    C = Add(A, B, name="C")

    # Ground truth
    dataset = IntDataset.of_shape((5, 2), lambda x: 2 * x[0] + 3 * x[1])

    # Construct interpretation, omitting the path through B
    C_t = InterpNode(name="Ct", cond_sampler=label_cond)
    A_t = C_t.make_descendant(name="At", cond_sampler=FS(lambda d: IntDataset.unwrap(d).inp_tensor[:, 0]))

    to_C = corr_root_matcher

    corr = Correspondence()
    corr.add(C_t, to_C)
    corr.add(A_t, to_C.chain("A"))

    # By default the unspecified nodes will be sampled randomly
    # so our scrubbed model will do poorly
    run_experiment(C, dataset, corr, lambda l: l > 1, num_examples=50)

    # But we can specify that by default nodes are not resampled...
    C_t.other_inputs_sampler = ExactSampler()
    # in which case we recover perfect performance
    run_experiment(C, dataset, corr, lambda l: l == 0, num_examples=1)


def test_print():
    # Construct model
    xs = Array(torch.tensor([0.0, 1.0]), name="xs")
    first = Index(xs, I[0])
    A = Einsum.scalar_mul(first, 2, name="A")
    B = Einsum.scalar_mul(first, 3, name="B")
    C = Add(A, B, name="C")

    # Deterministic ds
    generator = torch.Generator()
    generator.manual_seed(11)
    dataset = IntDataset(
        (
            Array(torch.randint(high=10, size=(100, 2), generator=generator), "xs"),
            Array(torch.arange(100, 200), "labels"),
        )
    )

    # Construct interpretation
    C_t = InterpNode(name="Ct", cond_sampler=FS(lambda d: IntDataset.unwrap(d).inp_tensor[:, 0]))
    A_t = C_t.make_descendant(name="A", cond_sampler=FS(lambda d: IntDataset.unwrap(d).inp_tensor[:, 0]))

    to_C = corr_root_matcher
    to_A = to_C.chain("A")

    corr = Correspondence()
    corr.add(C_t, to_C)
    corr.add(A_t, to_A)

    ex = Experiment(C, dataset, corr, random_seed=11)
    scrubbed_ex = ex.scrub(1)

    assert (
        scrubbed_ex.print(PrintOptions(), color_by_data=False, repr=True)
        == """C Add # cond_sampler=FuncSampler(d=IntDatum(xs=[0, 5], label=185), f(d)=0), other_inputs_sampler=UncondSampler(d=IntDatum(xs=[2, 3], label=167))
  A Einsum ,-> ... # cond_sampler=FuncSampler(d=IntDatum(xs=[0, 1], label=153), f(d)=0)
  B Einsum ,-> ..."""
    )

    scrubbed_ex.print()  # check coloring doesn't crash, without checking the result
