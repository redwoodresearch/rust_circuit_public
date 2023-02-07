# %% [markdown]
# **Handcrafted parenthesis balancer and the cumulants approach**
#
# For background on the parenthesis balancing task (and interpretability on it for a DIFFERENT model) see Nix and Fabien's presentation.
#
# It's possible to build a 1L transformer (one attention head, one non-linearity), with hand-selected weights, that completes the parenthesis balancing task perfectly.
#
# This notebook builds such a model and then uses the cumulants approach to attribute its behaviour to different parts of the model.
#
# Two important things we do in this notebook are to use dataset modifications in order to make certain features independent, and also ReLU derivative cumulant approximations.
#
# Skip the next two cells: the first is imports, the second uses a different model to extract the tokens_var. The dataset used keeps the minimum elevation test and the total elevation test independent (look at get_independent_dataset if really curious about the dataset).
#
# Also ensure you have rust and have rust_circuit as in interp/circuit/rust_circuit/readme.md
# %%
# Uncomment to have module autoreload (grr mypy)
# %load_ext autoreload
# %autoreload 2
import os
from typing import Callable, Optional, Tuple

import jax
import numpy as np
import seaborn as sns
import torch
import torch as t

from adversarial.simple_task.dataset_utils import load_dataset
from adversarial.simple_task.model_architectures.simple_transformer import SimpleTokenizer
from interp.circuit import computational_node
from interp.circuit.algebric_rewrite import MUL_REST, rearrange_muls, residual_rewrite
from interp.circuit.circuit import Circuit
from interp.circuit.circuit_model_rewrites import basic_cum_expand_run
from interp.circuit.circuit_utils import cast_circuit
from interp.circuit.computational_node import Add, Einsum, GeneralFunction, Index
from interp.circuit.constant import ArrayConstant, One, Zero
from interp.circuit.cum_algo import cumulant_function_derivative_estim
from interp.circuit.cumulant import Cumulant
from interp.circuit.function_rewrites import get_relu_fake_derivatives
from interp.circuit.get_update_node import FunctionIterativeNodeMatcher as F
from interp.circuit.get_update_node import NameMatcher as NM
from interp.circuit.get_update_node import NodeUpdater as NU
from interp.circuit.get_update_node import Replace
from interp.circuit.print_circuit import PrintCircuit
from interp.circuit.projects.estim_helper import *
from interp.circuit.projects.interp_utils import *
from interp.circuit.projects.punct.utils import standard_pdi
from interp.circuit.scope_manager import ScopeManager
from interp.circuit.scope_rewrites import basic_factor_distribute
from interp.circuit.var import DiscreteVar
from interp.tools.indexer import TORCH_INDEXER as I

sns.set_theme()

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"  # Use 8 CPU devices
os.environ["RR_CIRCUITS_REPR_NAME"] = "true"

RRFS_DIR = os.path.expanduser("~/rrfs")
RRFS_INTERP_MODELS_DIR = f"{RRFS_DIR}/interpretability_models_jax/"
os.environ["INTERPRETABILITY_MODELS_DIR"] = os.environ.get(
    "INTERPRETABILITY_MODELS_DIR",
    os.path.expanduser("~/interp_models_jax/")
    if os.path.exists(os.path.expanduser("~/interp_models_jax/"))
    else RRFS_INTERP_MODELS_DIR,
)
jax.config.update("jax_platform_name", "cpu")

# %%
task = "balanced_parens"
tokenizer = SimpleTokenizer.task_tokenizer(task)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ParenBalanceDataSet:
    def __init__(self, data_list, tokenizer):
        self.vocab_size = len(tokenizer.i_to_t)

        strs, is_balanced = [s for s, a in data_list], [a for s, a in data_list]
        self.seqs = np.array(tokenizer.tokenize(strs))
        self.is_balanced = np.array(is_balanced)
        self.strs = strs

    def get_one_hots(self):
        return torch.nn.functional.one_hot(torch.LongTensor(self.seqs), self.vocab_size)


def get_random_paren_list(
    name: str, selector: Callable[[str, bool], bool] = lambda _1, _2: True
) -> Tuple[list[Tuple[str, bool]], list[Tuple[str, bool]]]:
    ds_train = [(s, r) for s, r in load_dataset(task, name)["train"] if selector(s, r)]
    ds_dev = [(s, r) for s, r in load_dataset(task, name)["dev"] if selector(s, r)]
    return ds_train, ds_dev


def all_quite_close(x, y):
    m = torch.max(torch.abs(x.detach().clone().cpu() - y.detach().clone().cpu()))
    return m.item() < 1e-4


def inverse_cumsum(a: torch.Tensor, axis=-1) -> torch.Tensor:
    return torch.flip(torch.flip(a, dims=[axis]).cumsum(dim=axis), dims=[axis])


def count_open_propotions(toks: torch.Tensor) -> torch.Tensor:
    device = toks.device
    if len(list(toks.shape)) == 0:
        return torch.FloatTensor([0.0]).to(device)
    open_parens_counts = inverse_cumsum(toks[..., 3])
    close_parens_counts = inverse_cumsum(toks[..., 4])
    return open_parens_counts / torch.maximum(
        open_parens_counts + close_parens_counts, torch.FloatTensor([1.0]).to(device)
    )


def feature_to_overall_elevation_wrong(x: torch.Tensor, bal_value: int = 0, unbal_value: int = 1) -> torch.Tensor:
    return (unbal_value - bal_value) * x[..., 1] + bal_value


def feature_to_neg_elevation(x: torch.Tensor, bal_value: int = 0, unbal_value: int = 1) -> torch.Tensor:
    return (unbal_value - bal_value) * torch.max(x, dim=-1).values + bal_value


def is_unbalanced(toks: torch.Tensor, func=torch.sigmoid) -> torch.Tensor:
    """
    Take a shape
    (b *) seq_length * vocab_size
    tensor (batch is optional) and calculate whether it is unbalanced (0) or not (1)
    """

    if len(list(toks.shape)) == 0:
        return torch.zeros(1, 1, 1)

    p = count_open_propotions(toks)
    x = feature_to_overall_elevation_wrong(torch.where(torch.abs(p - 0.5) > 0.01, 1.0, 0.0), -10, 20)
    y = feature_to_neg_elevation(torch.where(p - 0.5 > 0.01, 1.0, 0.0), -10, 20)
    return func(x + y)


def total_elev(toks: torch.Tensor) -> torch.Tensor:
    """
    Same as is_unbalanced, except measures whether something fails the total elevation test (1.0) or not (0.0)
    """

    if len(list(toks.shape)) == 0:
        return torch.zeros(1, 1, 1)

    p = count_open_propotions(toks)
    x = feature_to_overall_elevation_wrong(torch.where(torch.abs(p - 0.5) > 0.01, 1.0, 0.0), -10, 10)
    return torch.sigmoid(x)


def min_elev(toks: torch.Tensor, func=torch.sigmoid) -> torch.Tensor:
    """
    Same as is_unbalanced, except measures whether something fails the minimum elevation test (1.0) or not (0.0)
    """

    if len(list(toks.shape)) == 0:
        return torch.zeros(1, 1, 1)

    p = count_open_propotions(toks)
    y = feature_to_neg_elevation(torch.where(p - 0.5 > 0.01, 1.0, 0.0), -10, 10)
    return func(y)


def get_independent_dataset(
    tokenizer=tokenizer,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Return `INDPENDENT_MIX`: a dataset of
    length 40 parenthesis strings where failing the total elevation test is independent of failing the minimum elevation test. Note that three times as many data points are unbalanced strings compared to balanced strings
    """

    def string_is_good(s: str):
        return len(s) == 40 and s[0] == "("

    random_ds_name = "random_choice_len_40_extra_yeses_16"
    balanced_ds_name = "random_choice_len_40_balanced_advexes"

    unbalanced_selector = lambda s, r: not r and string_is_good(s)
    balanced_selector = lambda s, r: r and string_is_good(s)

    ds_unbalanced_train, _ = get_random_paren_list(random_ds_name, unbalanced_selector)
    ds_balanced_train, _ = get_random_paren_list(balanced_ds_name, balanced_selector)
    big_ds = ParenBalanceDataSet(ds_unbalanced_train, tokenizer)

    m = big_ds.get_one_hots().cpu()

    total_elev_failures = []
    min_elev_failures = []
    both_failures = []
    for i in range(m.shape[0]):
        if not all_quite_close(total_elev(m[i : i + 1]), min_elev(m[i : i + 1])):
            if all_quite_close(total_elev(m[i : i + 1]), torch.zeros(1)):
                min_elev_failures.append(m[i : i + 1])
            else:
                total_elev_failures.append(m[i : i + 1])
        else:
            both_failures.append(m[i : i + 1])

    total_fail_tens = torch.concat(total_elev_failures)
    min_fail_tens = torch.concat(min_elev_failures)
    both_fail_tens = torch.concat(both_failures)

    number_of_balance_to_load = max(len(min_elev_failures), len(total_elev_failures))
    balanced_sample_ds = ParenBalanceDataSet(ds_balanced_train[:number_of_balance_to_load], tokenizer)
    portion_sizes = min(len(min_elev_failures), len(total_elev_failures), len(both_failures))
    return (
        torch.cat(
            (
                both_fail_tens[:portion_sizes],
                total_fail_tens[:portion_sizes],
                min_fail_tens[:portion_sizes],
                balanced_sample_ds.get_one_hots()[:portion_sizes],
            ),
            0,
        ),
        None,
    )


def get_sub_circuit(circuit: Circuit, child_name: str):
    """
    Get a circuit leaf in the computational graph of `circuit`
    that has name `child_name`
    """

    assert isinstance(circuit, Circuit)
    assert isinstance(child_name, str)
    return F(NM(child_name)).g().get_unique_c(circuit)


def get_truth_for_cumulant(model: Circuit, function, name: str = "true") -> Circuit:
    """
    Return the circuit representing the input parenthesis sequence
    """

    tokens = get_sub_circuit(model, "tokens")

    return Index(
        GeneralFunction(
            tokens,
            lambda x: torch.broadcast_to(function(x)[:, None, None], x.shape),
            non_batch_dims=(-1, -2),
            name=f"{name}_function",
        ),
        I[0, 0],
        name=name,
    )


def replace_tokens(data: torch.Tensor, *circuits: Circuit):
    """
    Replace the tokens inside all circuits
    """

    new_tokens = ArrayConstant.from_converted(data, dtype=torch.float, name="tokens", device=DEVICE)

    new_node = DiscreteVar(new_tokens, name="tokens_var")

    return *[NU(Replace(new_node), F(NM("tokens")))(c) for c in circuits], new_node


do_cumulant_evaluations = True
sample_ds, val_ds = get_independent_dataset(tokenizer=tokenizer)
#%%
def make_handcrafted_transformer():
    """
    `tokens_var` is 42 * 5 in shape
    """
    # Embedding
    # We encode ( (3) as e_1, and ) (4) as e_2 in the 2D residual stream
    # also, encode the end of sequence token as 1 (this is needed for the total elevation test)
    m = torch.zeros(5, 2)
    m[3][0] = 1.0
    m[4][1] = 1.0
    m[0][0] = 1.0

    fake_tokens_var = Zero(shape=(42, 5), name="tokens")

    embed_matrix = ArrayConstant(m, name="w.embed")
    embedded = Einsum.from_einsum_str("lo,ox->lx", fake_tokens_var, embed_matrix, name="embed")

    # QKV weights
    # we make queries for both e_1 and e_2 be 10.0 (queries and keys are 1D)
    # on the other hand the value vectors are +1 for open brackets and -1 for close brackets
    qw = ArrayConstant(torch.ones(2, 1) * 10, name="q.w")
    kw = One(shape=(2, 1), name="k.w")
    m2 = t.zeros(2, 1)
    m2[0][0] = 1.0
    m2[1][0] = -1.0
    vw = ArrayConstant(m2, name="v.w")

    # QKV calculations
    # note we also use masking, so everything only pays attention to things before it
    q = Einsum.from_einsum_str("io,li->lo", qw, embedded, name="q")
    k = Einsum.from_einsum_str("io,li->lo", kw, embedded, name="k")
    v = Einsum.from_einsum_str("io,li->lo", vw, embedded, name="v")
    qk = Einsum.from_einsum_str("lo,Lo->lL", q, k, name="qk")
    upper_triangular_tensor = t.ones(42, 42).float() * (-1e6)
    for i in range(42):
        for j in range(42):
            if i <= j:
                upper_triangular_tensor[i][j] = 0
    upper_triangular = ArrayConstant(upper_triangular_tensor, name="upper_triangular")
    scores = Add.from_unweighted_list([qk, upper_triangular], name="masked")
    probs = computational_node.softmax(scores, name="probs")
    values = Einsum.from_einsum_str("lL,Lo->lo", probs, v, name="head.out")

    # this setup means (as you can verify!) if any of the last 39 positions are positive, the elevation test has failed. Additionally if the last position (which attend to all the sequence) is not <= 0, the string is unbalanced,
    # So we need reverse this positions sign, then take a ReLU and if ANYWHERE is positive then the string is unbalanced; this is pretty much an MLP!
    m3 = t.eye(42).float()
    m3[0, 0] = -1.0  # I thonk...
    m3[:, 41] = 0
    m3[41, :] = 0
    m1w = ArrayConstant(m3, name="m1.w")
    m1ed = Einsum.from_einsum_str("lL,lo->L", m1w, values, name="m1.out")
    relued = computational_node.relu(m1ed, name="m1.act")

    # then add up all the entries
    m2w = ArrayConstant(t.ones(42, 42), name="m2.w")
    m2ed = Einsum.from_einsum_str("lL,l->L", m2w, relued, name="m2.out")

    # taking sigmoid and the convention that 0.5 rounds down to 0, this model achieves perfect performance on length 40 sequences
    logits = Index(m2ed, I[0], name="logits")
    return logits


circuit = make_handcrafted_transformer()
truth = get_truth_for_cumulant(circuit, is_unbalanced)
circuit = cast_circuit(circuit, device=DEVICE)
truth = cast_circuit(truth, device=DEVICE)
circuit, truth, new_node = replace_tokens(sample_ds, circuit, truth)
# %%
# some tools for printing circuits
d1 = PrintCircuit(
    print_html=True, colorize=lambda _: True, max_depth=1, print_shape=True, copy_button=True
)  # depth 1 printing, but collapsibles mean things can be expanded

print("View the collapsible tree for the circuit:")
d1(circuit)
# %%
# Now let's define the covariance we want to explain
# (why is the covariance between the true answer and the difference between the two output logits high?)

cumulant_circuit = Cumulant((circuit, truth))
scopes = [ScopeManager(cumulant_circuit)]  # all scopes considered, as a list because checking how things change
d1(scopes[-1])
#%% [markdown]
# We now need to "push down" the cumulant. This is equivalent to noticing that the output logits $L$ are equal to (an entry in) $Mx$, where $M$ is the final matrix multiply.
# So we have $K(Mx, T) = MK(x, T)$, as $M$ is constant. So "pushing down" the cumulant means factoring out the $M$ term:

# %%
scopes.append(
    scopes[-1].u(
        NU(
            lambda x: basic_cum_expand_run(  # expand cumulant
                Cumulant.unwrap(x),
                cum_expand_through_factored_add_concat=True,
                until_func=NM("true"),  # do not expand through "true"
                child_suffix="logits",  # ... instead expand through "logits"
            ),
            F(NM(scopes[-1].unique().name)),  # the name of the cumulant that we want to expand
        )
    )
)
d1(scopes[-1])

#%%
print("Now push down the index in a similar way:")
scopes.append(scopes[-1].u(NU(lambda x: standard_pdi(x), F(NM(scopes[-1].unique().name)))))
# upto here, maybe delete ya extra type safety
#%%
print("Now let's see what the covariance is!")
estim = EstimHelper(Zero(), use_new_estim=True, device=DEVICE)
list(estim.estimate(scopes[-1].unique()).items())[0][1].item()
#%%
print("Now let's expand the derivative non-linearity by estimating with the first order term")


def relu_rewrite(
    scope: ScopeManager, cumulant_name: str, relu_name: str, highest_deriv: int = 6, std_deviation: float = 0.1
):
    """
    Expand a ReLU non-linearity (GeneralFunction) `relu_name`, which has a parent (Cumulant) `cumulant_name, into `highest_deriv` number of terms. Approximates ReLUs with the derivative of a normal PDF with std `std_deviation`
    """

    def run_expand_deriv_relu(x: Circuit, relu_name: str):
        """
        Function that updates ReLU nodes
        """

        pre_act = GeneralFunction.unwrap(F(NM(relu_name)).g().get_unique_c(x))

        estim = cumulant_function_derivative_estim(
            Cumulant.unwrap(x),
            pre_act,
            highest_deriv=highest_deriv,
            get_derivative=get_relu_fake_derivatives(pre_act, fake_dirac_std=std_deviation),
        )
        out, _, _ = residual_rewrite(x, estim, running_name="d_estim")
        return out

    return scope.u(NU(lambda x: run_expand_deriv_relu(x, relu_name), F(NM(cumulant_name))), no_eq_besides_name=False)


scopes.append(relu_rewrite(scopes[-1], cumulant_name="I k2 m1.act, true I", relu_name="m1.act"))
d1(scopes[-1])
#%%
scopes.append(basic_factor_distribute(scopes[-1], modify_inplace=False))
#%% The tree looks quite different now!
# Note the Jacobians here (which are large and mean we won't be able to evaluate many of these cumulants) are actually diagonal, as ReLU acts elementwise. See batched_deriv for a more efficient use (which we don't need here as we only need lower order terms)
scopes.append(
    scopes[-1].sub_get(F(NM("I k2 m1.act, true I_d_estim_out")))
)  # get derivative terms (we're now approximating the original cumulant)
#%%
scopes.append(
    scopes[-1].u(
        NU(
            lambda x: basic_cum_expand_run(  # expand cumulant
                Cumulant.unwrap(x),
                cum_expand_through_factored_add_concat=True,
                until_func=NM("true"),
                child_suffix="m1.out",
            ),
            F(NM("I k2 m1.out, true I")),
        ),
        no_eq_besides_name=False,  # now needed...
    )
)
#%%
scopes.append(basic_factor_distribute(scopes[-1], modify_inplace=False))
#%%
scopes.append(
    scopes[-1].sub_get(F(NM("I k2 m1.act, true I_d_estim_item_1_out")))
)  # now it is a scalar we get the first derivative
#%%
print("Let's check the cumulant between this approximation and the truth:")
print(list(estim.estimate(scopes[-1].unique()).items())[0][1].item())
print(
    "Note that this is less than half of the total effect size. It turns out it is possible to explain behaviour in this case though"
)
#%%
scopes.append(
    scopes[-1].u(
        NU(
            lambda x: basic_cum_expand_run(  # expand cumulant
                Cumulant.unwrap(x),
                cum_expand_through_factored_add_concat=True,
                until_func=NM("true"),
                child_suffix="head.out",
            ),
            F(NM("I k2 head.out, true I")),
        ),
        no_eq_besides_name=False,  # now needed...
    )
)
#%%
scopes.append(basic_factor_distribute(scopes[-1], modify_inplace=False))
#%%
circuits_to_estimate = Add.unwrap(scopes[-1].unique()).items.keys()
out = estim.estimate(*circuits_to_estimate)
for k, v in out.items():
    print(k.name, v.item())
print("This is not surprising since recall that the attention pattern was constant")
#%%
scopes.append(scopes[-1].sub_get(F(NM("I k2 true, v k1 probs I perm_out"))))
#%%
print("Finally we can rearrage the Einsums to get a product of the corellatios with the input, and another matrix:")
final_circuit = rearrange_muls(Einsum.unwrap(scopes[-1].unique()), (0, MUL_REST))
d1(final_circuit)

#%%
q = list(estim.estimate(get_sub_circuit(final_circuit, "I k2 tokens_var, true I")).items())[0][1][:, 3:]


def plot_matrix(m):
    if len(m.shape) < 2:
        m.unsqueeze(0)
    sns.heatmap(m.detach().cpu())


plot_matrix(q)
print("This basically shows us correlations between the input and the truth, so doesn't give much signal")
# %%
q2 = list(
    estim.estimate(
        get_sub_circuit(final_circuit, "m1.w m1.act_deriv_d_estim_expectation d_estim_recip_1! m2.w_idxed")
    ).items()
)[0][
    1
]  # [:, 3:]
plot_matrix(q2.unsqueeze(0))
print(
    "The high value at 0 is meaningless since it is timesed by 0 in the dataset corellation term. The rest of the plot shows that the final position accounts for most of the variance (the total elevation test) and the cyclical bands show the minimum elevation test occurs from right to left as the first minimum elevation failure will always be at an odd time"
)
