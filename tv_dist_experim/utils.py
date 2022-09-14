import torch
import numpy as np
from math import ceil
import itertools
import random

import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


def load_mnist(m, target_digit, device, dtype=torch.float32, seed=None):
    mnist = MNIST('..', train=False, transform=ToTensor(), download=True)
    indexes = (mnist.targets == target_digit).nonzero().flatten().tolist()
    if seed is not None:
        random.seed(seed)
    chosen = random.sample(indexes, m)
    cs = [mnist[i][0] for i in chosen]
    cs = torch.stack(cs).reshape(m, -1).type(dtype)
    return (cs / cs.sum(dim=-1, keepdims=True)).to(device)


def replace_zeros(arr, replace_val=1e-5, sumdim=-1):
    arr[arr < replace_val] = replace_val
    arr /= arr.sum(dim=sumdim, keepdim=True)
    return arr


def get_cost_mat(im_sz, device, dtype=torch.float32):
    partition = torch.linspace(0, 1, im_sz)
    couples = np.array(np.meshgrid(partition, partition)).T.reshape(-1, 2)
    x = np.array(list(itertools.product(couples, repeat=2)))
    x = torch.tensor(x, dtype=dtype, device=device)
    a = x[:, 0]
    b = x[:, 1]
    C = torch.linalg.norm(a - b, axis=1) ** 2
    return C.reshape((im_sz**2, -1))


def show_barycenter(r, fname, im_sz=8):
    img = r.cpu().numpy().reshape(im_sz, -1)
    plt.imshow(img, cmap='binary')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"out_files/{fname}.png", bbox_inches='tight')
    plt.close()


def show_barycenters(barycenters, img_sz, img_name, iterations=None, use_softmax=True, scaling='full', use_default_folder=True):
    """Display several barycenters across iterations."""
    n_bary = len(barycenters)
    if n_bary > 10:
        nrows, ncols = 2, ceil(n_bary / 2)
    else:
        nrows, ncols = 1, n_bary
    figsize = (ncols * 3, nrows * 4)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for i, ax in enumerate(axes.flat):
        if i < n_bary:
            z = barycenters[i]
            img = (torch.softmax(z, dim=-1) if use_softmax else z).cpu().numpy().reshape(img_sz, -1)
            # ax = next(axes.flat)  # axes[i] if n_bary > 1 else axes
            if np.allclose(img, img[0, 0]) or scaling == 'none':
                ax.imshow(img, cmap='binary', vmin=0, vmax=1)
            elif scaling == 'partial':
                ax.imshow(img, cmap='binary', vmin=0)
            else:
                ax.imshow(img, cmap='binary')

            if iterations is not None:
                it = iterations[i]
                title = f"Iteration {it}" if isinstance(it, int) else it
                ax.title.set_text(title)
        else:
            ax.axis('off')

        ax.set_xticks([])
        ax.set_yticks([])


    if use_default_folder:
        img_name = f"out_files/bary_{img_name}"
    plt.savefig(img_name + ".png", bbox_inches='tight')
    plt.close()


def get_fnames(folder, kappa, n_samples, prior_std, suffix):
    kappa_str = f'{round(kappa, 2)}_'.lstrip('0')
    n_samples_str = f"M{n_samples:.0e}".replace("e+0", "e")
    info = n_samples_str + f"_std{prior_std}_kappa" + kappa_str
    entities = ['red_mean', 'red_pot', 'obj_vals']
    return [folder + info + entity + f"_{suffix}.pt" for entity in entities]


def get_sample_generator(prior_mean, n_batches, prior_std, verbose=False, seed=0):
    def sample_generator():
        for i in range(n_batches):
            if verbose:
                print(f"sampling batch {i}")
            torch.manual_seed(seed + i)
            yield torch.normal(prior_mean, prior_std)

    return sample_generator


def plot_heatmap(fname):
    with open(fname + '.npy', 'rb') as f:
        table = np.load(f)
    im = plt.imshow(table, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar(im)
    plt.savefig(fname + '.png', bbox_inches='tight')
