# %%
"""
A pool is a group of sampled datasets. If possible, datasets in the pool will
be (randomly) reused when scrubbing instead of sampling new datasets.

What does "if possible" mean? First of all, it has to be enabled by creating a CondSampler
with a PoolAnnotation (by default it is disabled). Then, it has to make semantic sense.
Recall that sampling always happens in the context of 3 things:
- a CondSampler
- a reference datum; the sampler is responsible for sampling a new, "equivalent" datum
- a source dataset to sample from
Suppose a given CondSampler is called on the same ref and source multiple times. Its sample
from the first call could potentially be reused for some of the following calls.

Specifically, we can save the result of calling the CondSampler some number of times, say 4
(this is the pool size), on this ref and source to create a pool. Then, whenever
it's called on the same ref and source, we'll randomly pick a sample from the pool.
"""
from rust_circuit.causal_scrubbing.hypothesis import (
    PNP,
    ExactSampler,
    FuncSampler,
    InterpNode,
    MaybePoolAnnotation,
    PoolAnnotation,
    UncondSampler,
    UncondTogetherSampler,
)
from rust_circuit.causal_scrubbing.testing_utils import IntDataset


class FirstSampler(FuncSampler):
    func = lambda x: IntDataset.unwrap(x).xs.value[:, 0]

    def __init__(self, pool_annot: MaybePoolAnnotation = PNP):
        super().__init__(FirstSampler.func, pool_annot)


dataset = IntDataset.of_shape((1000, 2), lambda x: x[0])

# root node's sampler doesn't have pools enabled
inode = InterpNode(ExactSampler(), "A")
# this sampler has a pool of size 4
first_sampler = FirstSampler(pool_annot=PoolAnnotation(4))
# 40 nodes, each with the same sampler and with the same parent node.
for j in range(40):
    inode.make_descendant(first_sampler, f"B{j}")

sampled_inputs = inode.sample(None, dataset, dataset)

# only one sampler with pools
assert len(sampled_inputs.sampler_pools) == 1
first_sampler_pools = sampled_inputs.sampler_pools[(first_sampler, dataset)]
# only one parent ds this sampler was called on
assert len(first_sampler_pools) == 1
first_sampler_pool_for_parent_ds = list(first_sampler_pools.values())[0]
# pool has size 4
assert len(first_sampler_pool_for_parent_ds) == 4
# so overall we have 5 different sampled datasets: one for the root, 4 for the rest
assert len(set(sampled_inputs.datasets.values())) == 5

inode.print(sampled_inputs=sampled_inputs)
# %%
"""
Why use pools?

This tames the explosion of the treeified model, while only failing to scrub high-order
correlations (that probably you didn't care about anyway). So, this is recommended if your
treeified model is too big (e.g for deep networks with non-trivial samplers).
"""
# %%
"""
In fact, there are additional cases where we could use samples from a pool. The first is:
given a different (by id) sampler, which is the same by attrs eq. This is for convenience, so you
don't have to keep track of all your CondSamplers. As long as you use the same PoolAnnotation
whenever you create CondSamplers, any two that are identical will share a pool (keep in mind
that lambdas are not equal, though, for when you are writing FuncSamplers).
"""
shared_pool = PoolAnnotation(1)
inode = InterpNode(ExactSampler(), "A")
b0 = inode.make_descendant(FirstSampler(pool_annot=shared_pool), "B0")
b1 = inode.make_descendant(FirstSampler(pool_annot=shared_pool), "B1")
assert b0.cond_sampler == b1.cond_sampler

sampled_inputs = inode.sample(None, dataset, dataset)

assert len(sampled_inputs.sampler_pools) == 1
assert len(set(sampled_inputs.datasets.values())) == 2  # one from the root, one from the other two nodes
inode.print(sampled_inputs=sampled_inputs)
# %%
# Meanwhile, samplers with different pool annotations are not equal, and don't share a pool:
inode = InterpNode(ExactSampler(), "A")
b0 = inode.make_descendant(FirstSampler(pool_annot=PoolAnnotation(1)), "B0")
b1 = inode.make_descendant(FirstSampler(pool_annot=PoolAnnotation(1)), "B1")
assert b0.cond_sampler != b1.cond_sampler

sampled_inputs = inode.sample(None, dataset, dataset)

assert len(sampled_inputs.sampler_pools) == 2
assert len(set(sampled_inputs.datasets.values())) == 3  # one from each node
inode.print(sampled_inputs=sampled_inputs)
# %%
# And just for completeness, here's what we get with no pools enabled:
first_sampler = FirstSampler()
inode = InterpNode(ExactSampler(), "A")
b0 = inode.make_descendant(first_sampler, "B0")
b1 = inode.make_descendant(first_sampler, "B1")

sampled_inputs = inode.sample(None, dataset, dataset)

assert len(sampled_inputs.sampler_pools) == 0
assert len(set(sampled_inputs.datasets.values())) == 3  # one from each node
inode.print(sampled_inputs=sampled_inputs)
# %%
"""
The other case where we could reuse samples is: if our sampler is called on a different reference,
which is nonetheless "the same, according to this sampler." Suppose you wrote a sampler which only
cared about the first element in a tuple, and the ref was (1, 2) so the sampler sampled (1, 5). If
the sampler then sees (1, 3), (1, 5) is just as likely to be sampled as it was the first time!

Without this, we might find we rarely make use of our pools, as it might be unlikely for a
sampler to see the same dataset in the course of a causal scrubbing run.

By default, datasets are only equivalent to themselves, i.e. if you extend CondSampler without
overriding ds_eq_class this sort of reuse won't happen. ExactSampler keeps this behavior.
Meanwhile, FuncSampler puts all datasets with the same value of its func in one equivalence class;
and UncondSampler puts all datasets into a single equivalence class.
"""
first_sampler = FirstSampler(pool_annot=PoolAnnotation(1))

inode = InterpNode(ExactSampler(), "A")
# B and C will separately sample data with the same first element
inode_b = inode.make_descendant(FirstSampler(), "B")
inode_c = inode.make_descendant(FirstSampler(), "C")
# So Db and Dc will see different parent ds but with the same first element:
inode_b.make_descendant(first_sampler, "Db")
inode_c.make_descendant(first_sampler, "Dc")

sampled_inputs = inode.sample(None, dataset, dataset)

# only one sampler with pools
assert len(sampled_inputs.sampler_pools) == 1
first_sampler_pools = sampled_inputs.sampler_pools[(first_sampler, dataset)]
# one equivalence class of ds this sampler was called on
eq_class = first_sampler.ds_eq_class(sampled_inputs.datasets[inode_b])
assert eq_class == first_sampler.ds_eq_class(sampled_inputs.datasets[inode_c])
assert len(first_sampler_pools) == 1
first_sampler_pool_for_parent_ds = first_sampler_pools[eq_class]
# pool has size 1
assert len(first_sampler_pool_for_parent_ds) == 1
# so overall we have 4 different sampled datasets: 3 from A B and C, 1 from the pool.
assert len(set(sampled_inputs.datasets.values())) == 4

inode.print(sampled_inputs=sampled_inputs)
# %%
# # Meanwhile if the datasets are "not the same", they do not share a pool:
inode = InterpNode(ExactSampler(), "A")
# B and C will sample totally random data:
inode_b = inode.make_descendant(UncondSampler(), "B")
inode_c = inode.make_descendant(UncondSampler(), "C")
# So Db and Dc will see different parent ds:
inode_b.make_descendant(first_sampler, "Db")
inode_c.make_descendant(first_sampler, "Dc")

sampled_inputs = inode.sample(None, dataset, dataset)

# only one sampler with pools
assert len(sampled_inputs.sampler_pools) == 1
first_sampler_pools = sampled_inputs.sampler_pools[(first_sampler, dataset)]
# two parent ds this sampler was called on
eq_class_b = first_sampler.ds_eq_class(sampled_inputs.datasets[inode_b])
eq_class_c = first_sampler.ds_eq_class(sampled_inputs.datasets[inode_c])
assert eq_class_b != eq_class_c
assert len(first_sampler_pools) == 2
first_sampler_pool_for_b_parent_ds = first_sampler_pools[eq_class_b]
first_sampler_pool_for_c_parent_ds = first_sampler_pools[eq_class_c]
# each pool has size 1
assert len(first_sampler_pool_for_b_parent_ds) == 1
assert len(first_sampler_pool_for_c_parent_ds) == 1
# so overall we have 5 different sampled datasets: 1 for each node
assert len(set(sampled_inputs.datasets.values())) == 5

inode.print(sampled_inputs=sampled_inputs)
# %%
"""
To sum up, we have:
- for each CondSampler hash and source ds:
  - for each ds eq class, according to the sampler:
    - as many samples as the pool size in the sampler's PoolAnnotation specified (in practice
      we sample these lazily, so there's no overhead to using pools when e.g. you only sample once).
"""
# %%
"""
Why use pools? pt 2

Additionally, pools can be used semantially; in particular, a pool of size 1 enforces
things are sampled together.

(And, of course, pools can be used for other_inputs_samplers.)
"""
inode = InterpNode(ExactSampler(), "A")
inode_b = inode.make_descendant(FirstSampler(), "B", other_inputs_sampler=UncondTogetherSampler())
inode_c = inode.make_descendant(FirstSampler(), "C", other_inputs_sampler=UncondTogetherSampler())
# This sampler is always the same by default; but you can pass different uuids to it if desired
assert inode_b.other_inputs_sampler == inode_c.other_inputs_sampler
inode_b.make_descendant(FirstSampler(), "Db")
inode_c.make_descendant(FirstSampler(), "Dc")

sampled_inputs = inode.sample(None, dataset, dataset)

# only one sampler with pools
assert len(sampled_inputs.sampler_pools) == 1
uncond_tog_sampler = UncondTogetherSampler()
uncond_sampler_pools = sampled_inputs.sampler_pools[(uncond_tog_sampler, dataset)]
# according to this sampler, all datasets are in the same equivalence class
eq_class_b = uncond_tog_sampler.ds_eq_class(sampled_inputs.datasets[inode_b])
assert eq_class_b == uncond_tog_sampler.ds_eq_class(sampled_inputs.datasets[inode_c])
assert len(uncond_sampler_pools) == 1
uncond_sampler_pool_for_parent_ds = uncond_sampler_pools[eq_class_b]

# pool has size 1
assert len(uncond_sampler_pool_for_parent_ds) == 1
# so overall we have 2 other_inputs_ds sampled: one from the root, one from the pool
assert len(set(sampled_inputs.other_inputs_datasets.values())) == 2

inode.print(sampled_inputs=sampled_inputs)
# %%
