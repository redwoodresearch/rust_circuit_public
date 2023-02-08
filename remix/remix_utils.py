# %%
"""
REMIX Utilities

This file contains spoilers for Day 1 of REMIX!
"""

# %%
import os
import torch
import numpy as np
from torchvision import datasets, transforms
import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
from tqdm.notebook import tqdm
from einops import rearrange
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional
from typing import cast, Iterable, Union, Any, Callable
from matplotlib import pyplot as plt

MAIN = __name__ == "__main__"


def await_without_await(func: Callable[[], Any]):
    """We want solution files to be usable when run as a script from the command line (where a top level await would
    cause a SyntaxError), so we can do CI on the files. Avoiding top-level awaits also lets us use the normal Python
    debugger.
    Usage: instead of `await cui.init(port=6789)`, write `await_without_await(lambda: cui.init(port=6789))`
    """
    try:
        while True:
            func().send(None)
    except StopIteration:
        pass


def remove_all_hooks(module: nn.Module):
    module._forward_hooks.clear()
    module._forward_pre_hooks.clear()
    module._backward_hooks.clear()


def get_mnist(path="./remix_d2_data/mnist.pickle", force_download=False) -> tuple[TensorDataset, TensorDataset]:
    """Download MNIST, preprocess, and wrap in TensorDatasets. Cache preprocessed data."""
    if force_download or not os.path.exists(path):
        print("MNIST not found on disk, downloading...")
        mnist_train = datasets.MNIST("../data", train=True, download=True)
        mnist_test = datasets.MNIST("../data", train=False)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_stack = _flatten_image(
            torch.stack([transform(img) for img, label in tqdm(cast(Iterable, mnist_train), desc="Training data")])
        )
        test_stack = _flatten_image(
            torch.stack([transform(img) for img, label in tqdm(cast(Iterable, mnist_test), desc="Test data")])
        )
        train_dataset = TensorDataset(train_stack, mnist_train.targets)
        test_dataset = TensorDataset(test_stack, mnist_test.targets)
        t.save((train_dataset, test_dataset), path)
    else:
        train_dataset, test_dataset = t.load(path)
    return train_dataset, test_dataset


def _flatten_image(data: t.Tensor) -> t.Tensor:
    return rearrange(data, "batch 1 width height -> batch (width height)")


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target, label_smoothing=0.1)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(model: nn.Module, device: Union[str, t.device], loader: DataLoader) -> dict[str, float]:
    """Run the model on the provided data and return `loss` and `acc` keys."""
    model.eval()
    test_loss = 0.0
    correct = 0
    n_batches = 0
    count = 0  # track count in case we do drop_last in which case the length of the loader isn't accurate
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # default reduction is mean, so this is avg loss per example
            test_loss += torch.nn.functional.cross_entropy(output, target, label_smoothing=0.1).item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            count += len(data)
            n_batches += 1

    test_loss /= n_batches  # we summed n_batches terms so scale back to avg loss per example

    acc = correct / count
    # print("Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(test_loss, correct, count, 100.0 * acc))
    return dict(loss=test_loss, acc=acc)


def run_train_test(model, train_loader, test_loader, num_epochs=4, device="cpu", weight_decay=0.1, **kwargs):
    start = time.time()
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    for epoch in range(num_epochs):
        train(model, device, train_loader, optimizer, epoch, **kwargs)
        print("Test performance: ")
        test(model, device, test_loader)
        scheduler.step()
    print(f"Completed in {time.time() - start : .2f}s")


class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.last = nn.Linear(28 * 28, 10, bias=True)

    def forward(self, x):
        return self.last(x)


# %%
if MAIN:
    t.manual_seed(9876)
    device = "cuda"
    train_dataset, test_dataset = get_mnist(force_download=False)
    train_dataset = TensorDataset(*[tensor.to(device) for tensor in train_dataset.tensors])
    test_dataset = TensorDataset(*[tensor.to(device) for tensor in test_dataset.tensors])
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=512)
    test_loader = DataLoader(test_dataset, batch_size=512)

    print("Training logistic regression model (should reach 92%)")
    lrmodel = LogisticRegression()
    run_train_test(lrmodel, train_loader, test_loader, device="cuda", num_epochs=10)

# %%
# Check if the model looks reasonable
def plot_pixel_contributions_to_logits(coef: np.ndarray):
    import seaborn as sns

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 5))
    i = 0
    for row in axes:
        for ax in row:
            if i >= len(coef):
                continue
            sns.heatmap(
                coef[i].reshape(28, 28),
                ax=ax,
                center=0,
                xticklabels=False,
                yticklabels=False,
                vmin=-0.1,
                vmax=0.1,
                cbar=False,
            )
            i += 1
    fig.suptitle("Evidence in favor (red) or against (blue) each digit")
    return fig


if MAIN:
    coef = lrmodel.last.weight.detach().cpu()
    plot_pixel_contributions_to_logits(coef)


# %%
# Some pixels in MNIST are always black in training, so my understanding is that there should be zero gradient on weights that multiply them, so the weight should shrink to zero due to weight decay.

# TBD: investigate why I'm wrong about this
if MAIN:
    train_var = train_dataset.tensors[0].detach().cpu().var(axis=0)  # type: ignore
    test_var = test_dataset.tensors[0].detach().cpu().var(axis=0)  # type: ignore
    always_same_train = train_var == 0
    always_same_test = test_var == 0
    print("Number of always the same pixels in training: ", always_same_train.sum())
    print("Number of always the same pixels in test: ", always_same_test.sum())
    plt.figure()
    plt.imshow(always_same_train.reshape(28, 28))
    plt.title("Always black in training")

    plt.figure()
    plt.imshow(always_same_test.reshape(28, 28))
    plt.title("Always black in test")

    plt.figure()
    dist_shift = always_same_train & ~always_same_test
    plt.imshow((test_var - train_var).reshape(28, 28))
    plt.colorbar()
    plt.title("Var in test - var in train")

    plt.figure()
    plt.imshow(dist_shift.reshape(28, 28))
    # Distribution shift - may be able to make adversarial examples using these
    print("Distribution shift pixels: ", dist_shift.nonzero().flatten())

# %%
if MAIN:
    thres = 1e-3
    train_var = train_dataset.tensors[0].detach().cpu().var(axis=0)  # type: ignore
    test_var = test_dataset.tensors[0].detach().cpu().var(axis=0)  # type: ignore
    low_var_train = train_var < thres
    low_var_test = test_var < thres
    print("Number of always the same pixels in training: ", low_var_train.sum())
    print("Number of always the same pixels in test: ", low_var_test.sum())
    plt.figure()
    plt.imshow(low_var_train.reshape(28, 28))
    plt.title("Usually black in training")

    plt.figure()
    plt.imshow(always_same_test.reshape(28, 28))
    plt.title("Usually black in test")

    plt.figure()
    dist_shift = low_var_train & ~low_var_test
    plt.imshow((test_var - train_var).reshape(28, 28))
    plt.colorbar()
    plt.title("Var in test - var in train")

    plt.figure()
    plt.imshow(dist_shift.reshape(28, 28))
    # Distribution shift - may be able to make adversarial examples using these
    print("Distribution shift pixels: ", dist_shift.nonzero().flatten())


# %%
class TwoLayerSkip(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(28 * 28, 28 * 28, bias=True)
        self.last = nn.Linear(28 * 28, 10, bias=True)

    def forward(self, x: t.Tensor) -> t.Tensor:
        skip = x.clone()
        first_out = torch.nn.functional.relu(self.first(x))
        skip += first_out
        x = self.last(skip)
        return x


# Now we train a two layer with a skip, starting from the logistic intialization
# We zero out all the weights
if MAIN:
    t.manual_seed(1)
    twomodel = TwoLayerSkip()
    first_weight_init = twomodel.first.weight.detach().clone()
    # first_weight_init[~low_var_train, :] = 0
    first_weight_init[~low_var_train, :] = t.randn(low_var_train.shape) * 1e-5
    twomodel.first.weight = nn.Parameter(first_weight_init)
    last_weight = lrmodel.last.weight.detach().clone()
    # boost up the remaining weights - this is kinda obvious on the graph
    # but makes the adversarial single pixel thing stronger
    last_weight[:, low_var_train] *= 2
    twomodel.last.weight = nn.Parameter(last_weight)
    twomodel.last.bias = nn.Parameter(lrmodel.last.bias.detach().clone())
    run_train_test(twomodel, train_loader, test_loader, device="cuda", num_epochs=5, weight_decay=0)
    t.save(twomodel.state_dict(), "./remix_d2_data/model_b.pickle")

# %%
if MAIN:
    coef_twomodel = twomodel.last.weight.detach().cpu()
    plot_pixel_contributions_to_logits(coef_twomodel)
    plot_pixel_contributions_to_logits(coef_twomodel - coef)


# %%
"""
Sketch of adversarial example - maybe not in scope
"""
if False:
    plot_pixel_contributions_to_logits(twomodel.first.weight.detach().cpu()[12:22])
    plot_pixel_contributions_to_logits(twomodel.first.weight.detach().cpu()[[141, 645]])
    print("Pixel 141 will be used as evidence for: ", twomodel.last.weight[:, 141])
    print("Pixel 645 will be used as evidence for: ", twomodel.last.weight[:, 645])
    idx = (test_dataset.tensors[1] == 2).nonzero()[0].item()
    digit = test_dataset.tensors[0][idx]
    plt.imshow(digit.reshape(28, 28))
    logp_before = twomodel(digit.to(device)).softmax(-1)
    adv = digit.clone()
    adv[141] = 5
    # adv[645] = 1
    plt.imshow(adv.reshape(28, 28))
    logp_after = twomodel(adv.to(device)).softmax(-1)
    print("Before: ", (100 * logp_before).round())
    print("After: ", (100 * logp_after).round())


# %%
if MAIN:
    t.manual_seed(1)
    modela = TwoLayerSkip()
    run_train_test(modela, train_loader, test_loader, device="cuda", num_epochs=5, weight_decay=0.1)
    coef_a = modela.last.weight.detach().cpu()
    plot_pixel_contributions_to_logits(coef_a)
    t.save(modela.state_dict(), "./remix_d2_data/model_a.pickle")

# %%
