import pytest
import torch

from rust_circuit import Add, Array, Getter, PrintOptions
from rust_circuit.causal_scrubbing.hypothesis import (
    Correspondence,
    ExactSampler,
    FixedOtherSampler,
    IllegalCorrespondenceError,
    InterpNode,
    UncondSampler,
    UncondTogetherSampler,
    corr_root_matcher,
    to_inputs,
)
from rust_circuit.causal_scrubbing.testing_utils import IntDataset


def test_uncond_sampler():
    dataset = IntDataset.of_shape((100,), lambda x: x)
    sampler = UncondSampler()
    sampled: IntDataset = sampler(dataset, dataset)  # type: ignore
    corr = torch.corrcoef(torch.stack((dataset.inp_tensor, sampled.inp_tensor)))[0, 1]
    assert corr < 0.5

    sampled_2: IntDataset = sampler(dataset, dataset)  # type: ignore
    corr_2 = torch.corrcoef(torch.stack((sampled_2.inp_tensor, sampled.inp_tensor)))[0, 1]
    assert corr_2 < 0.5

    assert sampler.ds_eq_class(dataset) == sampler.ds_eq_class(IntDataset.of_shape((100,), lambda x: x))


def test_exact_sampler():
    dataset = IntDataset.of_shape((100, 2), lambda x: x[1])
    other = IntDataset.of_shape((100, 2), lambda x: x[1])
    sampler = ExactSampler()
    sampled = sampler(dataset, other)
    assert sampled == dataset

    assert sampler.ds_eq_class(dataset) != sampler.ds_eq_class(other)


def test_uncond_together_sampler():
    dataset = IntDataset.of_shape((100,), lambda x: x)

    inode = InterpNode(ExactSampler(), "A")
    b0 = inode.make_descendant(UncondTogetherSampler(), "B0")
    b1 = inode.make_descendant(UncondTogetherSampler(), "B1")
    sampled_inputs = inode.sample(None, dataset, dataset)
    assert len(sampled_inputs.sampler_pools) == 1

    sampled: IntDataset = sampled_inputs.datasets[b0]  # type: ignore
    corr = torch.corrcoef(torch.stack((dataset.inp_tensor, sampled.inp_tensor)))[0, 1]
    assert corr < 0.5

    sampled_2 = sampled_inputs.datasets[b1]
    assert sampled_2 == sampled

    other_dataset = IntDataset.of_shape((100,), lambda x: x)
    assert UncondTogetherSampler().ds_eq_class(dataset) == UncondTogetherSampler().ds_eq_class(other_dataset)


def test_fixed_other_sampler():
    dataset = IntDataset.of_shape((100,), lambda x: x)
    other_dataset = IntDataset.of_shape((100,), lambda x: x)

    inode = InterpNode(ExactSampler(), "A")
    b = inode.make_descendant(FixedOtherSampler(other_dataset), "B")
    sampled_inputs = inode.sample(None, dataset, dataset)

    sampled: IntDataset = sampled_inputs.datasets[b]  # type: ignore
    assert sampled == other_dataset


def test_implicit_inputs():
    dataset = IntDataset.of_shape((100,), lambda x: x)
    xs = Array(
        torch.tensor(
            [
                0.0,
            ]
        ),
        name="xs",
    )
    labels = Array(
        torch.tensor(
            [
                0.0,
            ]
        ),
        name="labels",
    )
    C = Add(xs, labels, name="C")

    to_C = corr_root_matcher

    to_inputs_found = Getter().get(C, to_inputs(to_C, dataset))

    assert len(to_inputs_found) == 2
    assert xs in to_inputs_found
    assert labels in to_inputs_found

    dataset = IntDataset(dataset.arrs, input_names={"xs"})

    to_inputs_found = Getter().get(C, to_inputs(to_C, dataset))

    assert len(to_inputs_found) == 1
    assert xs in to_inputs_found
    assert labels not in to_inputs_found


def test_corr_duplicate_keys():
    C_t = InterpNode(name="C", cond_sampler=UncondSampler())

    to_C = corr_root_matcher

    corr = Correspondence()
    with pytest.raises(ValueError):
        corr.replace(C_t, to_C)
    corr.add(C_t, to_C)
    with pytest.raises(ValueError):
        corr.add(C_t, to_C)
    corr.replace(C_t, to_C)


def test_corr_get_root():
    A = InterpNode(name="C", cond_sampler=UncondSampler())
    B = A.make_descendant(name="B", cond_sampler=UncondSampler())

    to_A = corr_root_matcher
    to_B = corr_root_matcher

    corr = Correspondence()
    corr.add(B, to_B)
    corr.add(A, to_A)
    assert A == corr.get_root()


def test_corr_must_be_connected():
    A = InterpNode(name="A", cond_sampler=UncondSampler())
    B = InterpNode(name="B", cond_sampler=UncondSampler())

    to_A = corr_root_matcher
    to_B = corr_root_matcher

    corr = Correspondence()
    corr.add(A, to_A)
    corr.add(B, to_B)
    with pytest.raises(IllegalCorrespondenceError):
        corr.get_root()


def test_corr_must_be_acyclic():
    A = InterpNode(name="C", cond_sampler=UncondSampler())
    B = A.make_descendant(name="B", cond_sampler=UncondSampler())
    B._children = B.children + (A,)  # very bad, no one should ever do this!

    to_A = corr_root_matcher
    to_B = corr_root_matcher

    corr = Correspondence()
    corr.add(A, to_A)
    corr.add(B, to_B)

    with pytest.raises(IllegalCorrespondenceError):
        corr.get_root()


def test_corr_complete():
    A = InterpNode(name="C", cond_sampler=UncondSampler())
    B = A.make_descendant(name="B", cond_sampler=UncondSampler())

    to_A = corr_root_matcher
    to_B = corr_root_matcher

    corr = Correspondence()
    corr.add(A, to_A)
    with pytest.raises(IllegalCorrespondenceError):
        corr.check_complete()

    corr.add(B, to_B)
    corr.check_complete()

    corr = Correspondence()
    corr.add(B, to_B)
    corr.check_complete()


def test_can_sample_interp_graph():
    dataset = IntDataset.of_shape((100,), lambda x: x)

    C_t = InterpNode(name="C", cond_sampler=UncondSampler())
    C_t.make_descendant(name="x0_A", cond_sampler=UncondSampler())
    C_t.make_descendant(name="x0_B", cond_sampler=UncondSampler())

    C_t.sample(source_ds=dataset, parent_ds=dataset, rng=torch.Generator())


def test_can_sample_corr():
    dataset = IntDataset.of_shape((100,), lambda x: x)

    C_t = InterpNode(name="C", cond_sampler=UncondSampler())
    x0_A = C_t.make_descendant(name="x0_A", cond_sampler=UncondSampler())
    x0_B = C_t.make_descendant(name="x0_B", cond_sampler=UncondSampler())

    to_C = corr_root_matcher

    corr = Correspondence()
    corr.add(C_t, to_C)
    corr.add(x0_A, to_C.chain("A").chain("x0"))
    corr.add(x0_B, to_C.chain("B").chain("x0"))

    corr.sample(source_ds=dataset, ref_ds=dataset, rng=torch.Generator())


def test_print():
    C_t = InterpNode(name="C", cond_sampler=ExactSampler())
    C_t.make_descendant(name="x0_A", cond_sampler=ExactSampler())
    C_t.make_descendant(name="x0_B", cond_sampler=UncondSampler())

    before_sampling = """\x1b[90mC GeneralFunction\x1b[0m # cond_sampler=ExactSampler, other_inputs_sampler=UncondSampler
  \x1b[90mx0_A GeneralFunction\x1b[0m # cond_sampler=ExactSampler
  \x1b[90mx0_B GeneralFunction\x1b[0m # cond_sampler=UncondSampler"""

    assert C_t.print(PrintOptions(), repr=True) == before_sampling

    ds = IntDataset((Array(torch.arange(10), "xs"), Array(torch.arange(10), "labels")))
    generator = torch.Generator()
    generator.manual_seed(11)
    sampled_inputs = C_t.sample(generator, ds, ds)

    C_t.print(PrintOptions(), sampled_inputs=sampled_inputs, color_by_data=False)
    after_sampling = """C GeneralFunction # cond_sampler=ExactSampler(d=IntDatum(xs=0, label=0)), other_inputs_sampler=UncondSampler(d=IntDatum(xs=1, label=1))
  x0_A GeneralFunction # cond_sampler=ExactSampler(d=IntDatum(xs=0, label=0))
  x0_B GeneralFunction # cond_sampler=UncondSampler(d=IntDatum(xs=9, label=9))"""

    assert C_t.print(PrintOptions(), sampled_inputs=sampled_inputs, color_by_data=False, repr=True) == after_sampling

    # check coloring doesn't crash, without checking the result. probably we actually want a deterministic colorer arg for testing.
    C_t.print(sampled_inputs=sampled_inputs)
