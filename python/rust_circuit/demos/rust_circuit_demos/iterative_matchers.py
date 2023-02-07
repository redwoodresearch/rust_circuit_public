#%%
from rust_circuit import (
    FINISHED,
    Add,
    Circuit,
    IterateMatchResults,
    IterativeMatcher,
    Matcher,
    Scalar,
    add_flatten,
    new_traversal,
    restrict,
)

#%% [markdown]
# # Understanding Iterative Matchers
# ## 1. Matchers
# In brief, you can use them to find circuits in a parent circuit, regardless of their position in the parent.
# %%
# An example
parent = Add(Add(Scalar(3.14, name="pi"), Scalar(2.7, name="e")), Add(name="empty"))
parent.print()
# %%
# Here is how to find all circuits in the parent of type "Add"
Matcher(Add).get(parent)

# %%
# you can also do
parent.get(Add)
# which is equivalent to the above

# %%
# Here is how to find the circuit which name is "pi"
Matcher("pi").get_unique(parent)
# %%
# or
parent.get_unique("pi")
# %% [markdown]
# How does it work?
# A Matcher is a function: Circuit -> bool (with a few helper method to use it in other contexts), and ".get" while apply it to every circuit in the parent.
# %%
# You can match using many other attributes. See test_rust_get_update.py for more examples.
#%% [markdown]
# ## 2. IterativeMatchers (for finding circuits)
# In brief, you can use them to find circuits in a parent circuit, using their relation to the parent.
#%%
# Example 1: finding all Adds at depth 1 from the parent
restrict(IterativeMatcher(Add), start_depth=1, end_depth=2).get(parent)
# %%
# Example 2: finding the second child of the parent
restrict(IterativeMatcher(True), term_if_matches=True).children_matcher({1}).get(parent)
# Details:
# IterativeMatcher(True) -> match every node encountered
# .filter(term_if_matches=True) -> finish if match a node (will match only the parent)
# .children_matcher({1}) -> match the first child of every circuit matched by the callee
# Note: this is equivalent to `IterativeMatcher.term(match_next=True).children_matcher({1}).get(parent)`
#%%
# Example 3: Matching every Scalar, no matter their position (behaves like a Match)
IterativeMatcher(Scalar).get(parent)
# %% [markdown]
# How does it work?
# An IterativeMatcher is a function: Circuit -> IterateMatchResults (with a few helper method to use it in other contexts), where IterateMatchResults is an object holding 2 informations:
# - do we match the circuit? (named "found")
# - what do we do next? (named "updated")
# The IterativeMatcher is then called on the parent, and then it will propagate according to the answer to "what do we do next"
# %%
# what do we do next, option 1: finish
finish_immeditaly = IterativeMatcher.new_func(lambda circuit: IterateMatchResults(found=True, updated=FINISHED))
finish_immeditaly.get(parent)
# Details:
# - FINISHED is a constant provided by rust_circuit
# - to construct an IterativeMatcher by hand, we use IterativeMatcher.new_func, because we can't guess if a lambda returns a bool (which it would we passed a "Matcher"), or an IterateMatchResults.
# %%
# what do we do next, option 2: use the same matcher, for example here is the add matcher.
add_it_matcher = IterativeMatcher.new_func(
    lambda circuit: IterateMatchResults(found=isinstance(circuit, Add), updated=None)
)
add_it_matcher.get(parent)
# Detail: None means "use the same matcher"
# %%
# what do we do next, option 3: use a new IterativeMatcher for each child, for example, here is to return the second child of the first child

# this is equivalent to IterativeMatcher.term(match_next=True)
it_matcher_3 = IterativeMatcher.new_func(lambda _: IterateMatchResults(found=True, updated=FINISHED))

it_matcher_2 = IterativeMatcher.new_func(lambda _: IterateMatchResults(found=False, updated=[FINISHED, it_matcher_3]))
it_matcher_1 = IterativeMatcher.new_func(lambda _: IterateMatchResults(found=False, updated=[it_matcher_2, FINISHED]))
it_matcher_1.get(parent)
# %%
# You can also use the third option to store and modify a state in the IterativeMatcher
def countdown_factory(count: int) -> IterativeMatcher:
    def countdown_it_matcher(circuit: Circuit) -> IterateMatchResults:
        if count == 0:
            print("boom")
            return IterateMatchResults(FINISHED, found=True)
        if count % 2 == 0:
            print("tick")
        if count % 2 == 1:
            print("tack")
        updated_state = count - 1
        return IterateMatchResults([countdown_factory(updated_state) for _ in circuit.children], found=False)

    return IterativeMatcher.new_func(countdown_it_matcher)


chain = Add(name="root")
for i in range(100 + 1):
    chain = Add(chain, name=f"{i}")

countdown_factory(10).get_unique(chain).name
# %%
# ## 3. IterativeMatchers (for defining traversals)
# In brief, you can use them to define a subtree a parent circuit.
# Sometimes, it doesn't make sense to match any subset of the circuits of the parent.
# A traversal is what you will you give to a function need a subtree (that includes the root):
# it's just an IterativeMatcher, but instead of looking at what gets matched, the receiving function
# only looks at what get's traversed by the matcher until it finishes, and ignores the "found" term
# %%
# Example 1: flatten nested adds up to depth 2
one = Scalar(1.0, name="one")
parent = Add(
    Add(Add(one, name="bc"), Add(Add(one, name="aa"), name="c"), name="ab"),
    Add(Add(one, name="ae"), name="ad"),
    name="ac",
)
print("before")
parent.print()
# traversal is a constant equal to IterativeMatcher(True) (it can also be constructed via IterativeMatcher.noop_traversal())
my_traversal = new_traversal(end_depth=2 + 1)
print("after")
add_flatten(parent, my_traversal).print()
#%%
# The same thing happens if we match nothing instead of everything
my_traversal = restrict(IterativeMatcher(False), end_depth=2 + 1)
add_flatten(parent, my_traversal).print()
# %%
# Example 2: flatten nested adds that start with "a" connected to the root
def it_macher_fn(circuit: Circuit) -> IterateMatchResults:
    if not isinstance(circuit, Add):
        raise ValueError
    return IterateMatchResults(
        [
            (IterativeMatcher.new_func(it_macher_fn) if isinstance(c, Add) and c.name.startswith("a") else FINISHED)
            for c in circuit.children
        ]
    )  # Don't even explore nodes with don't start with "a"!


my_traversal = IterativeMatcher.new_func(it_macher_fn)
add_flatten(parent, my_traversal).print()
# %% [markdown]
# Traversals are also used for print, distribute, push_down_index, ...
# In general, you will not create the traversal by hand, and you will be able to use one using the different utilities at your disposal, either using filters, chains, ... or using a built-in traversal like traverse_until_depth.
# %% [markdown]
# ## 4. Updating nodes
# In addition to getting nodes we can also apply functions which update them
# %%

one = Scalar(1.0, name="one")
parent = Add(
    Add(Add(one, name="bc"), Add(Add(one, name="aa"), name="c"), name="ab"),
    Add(Add(one, name="ae"), name="ad"),
    name="ac",
)
print("before")
parent.print()
print()
print("after")
# update circuits with a 'c' anywhere in the name
updated = parent.update(Matcher.regex("c"), lambda x: x.rename(x.name + " has c"))
updated.print()

# %%

# you can also use
assert Matcher.regex("c").update(parent, lambda x: x.rename(x.name + " has c")) == updated

# %% [markdown]
# ## 5. Advanced subjects
#%% [markdown]
# You can chain iterators!
# When the head of the chain matches, the next element in the chain is started.
# When the tail of the chain matches, the matched circuit is considered matched by the chain.
# More info in the docstring for chain.
#%%
# Example 1: match all circuits which have a circuit "pi + e" as parent
parent = Add(Add(Scalar(3.14, name="pi"), Scalar(2.7, name="e")), Add(name="empty"))
Matcher("pi + e").chain(Matcher(True)).get(parent)
# Details:
# - using matcher.<some IterativeMatcher function> behaves as if the Matcher was case
#   to an IterativeMatcher which always returned updated=None
# - you can pass a Matcher constructor argument to the chain (this will construct the IterativeMatcher),
#   or a Matcher, or an IterativeMatcher.
#   This also works for most functions taking IterativeMatcher as argument. Read rust_circuit.pyi for more details.
#%%
# Example 2: return the second child of the first child
def n_th_child_iterative_matcher(n):
    return IterativeMatcher.term(True).children_matcher({n})


n_th_child_iterative_matcher(0).chain(n_th_child_iterative_matcher(1)).get(parent)
# %%
