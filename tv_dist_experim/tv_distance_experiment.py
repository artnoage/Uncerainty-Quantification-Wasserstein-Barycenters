import torch
import numpy as np
from ot import barycenter as pot_barycenter
import os.path
import time

from utils import load_mnist, replace_zeros, get_cost_mat, show_barycenters, show_barycenter, get_fnames, plot_heatmap
from potentials_computation import get_potentials
from reduced_posterior import get_reduced_posterior


def create_reduced_posterior(digit, m, seed_mnist, n_samples, prior_std, batch_sz, seed_prior, device='cuda',
                             n_iter=2000, constant=30, temp=1, test=False, load_poten=True, save_poten=True):
    # Pull 'm' mnist images of digit 'digit'
    cs = load_mnist(m, digit, device, seed=seed_mnist)  # shape (m, n)
    cs = replace_zeros(cs)
    out_name = f'digit{digit}_m{m}_seed{seed_mnist}'
    if test:  # It's stable: same seed repeatedly yields same image collections, different seeds yield different ones
        show_barycenters([c.to('cpu') for c in cs], 28, out_name, use_softmax=False, scaling='partial')

    # Run grad descent to find potentials
    pot_fname = './out_files/' + out_name + '_poten.pt'
    if load_poten and os.path.isfile(pot_fname):
        print("        Loading existing potentials")
        potentials = torch.load(pot_fname, map_location=device)
    else:
        print("        Computing potentials via gradient descent")
        potentials = get_potentials(cs, n_iter=n_iter, constant=constant)  # shape (m, n)
        if save_poten:
            torch.save(potentials, pot_fname)

    if test:  # Potentials give a barycenter similar to ot.barycenter but the latter is sharper
        bary_from_poten = torch.softmax(-potentials.sum(dim=0) * constant, dim=0)
        show_barycenter(bary_from_poten, out_name + '_frompoten', im_sz=28)
        cost_mat = get_cost_mat(28, cs.device, dtype=cs.dtype)
        r = pot_barycenter(cs.T, cost_mat, 1e-3)
        show_barycenter(r, out_name + '_sinkh', im_sz=28)

    # Run 'create_reduced_posterior'
    out_name = f'digit{digit}_m{m}_seeds{seed_mnist}_{seed_prior}'
    get_reduced_posterior(potentials, cs, n_samples, prior_std, batch_sz, 1. / constant, temp=temp,
                          suffix=out_name, seed=seed_prior, n_partitions=4)


def load_posterior(fnames, temp=1, device='cuda'):
    fname_mean, fname_sample, fname_obj_vals = fnames

    mean = torch.load(fname_mean, map_location=device)  # (n,)
    sample = torch.load(fname_sample, map_location=device)  # (n_samples, n,)
    obj_vals = torch.load(fname_obj_vals, map_location=device)  # (n_samples,)
    weights = torch.softmax(temp * obj_vals, dim=0)  # (n_samples,)
    return mean, sample, weights


def get_tv_distance(fnames1, fnames2, device='cuda', temp=1, bins=20):
    mean1, sample1, weights1 = load_posterior(fnames1, temp=temp, device=device)
    mean2, sample2, weights2 = load_posterior(fnames2, temp=temp, device=device)

    direction = mean2 - mean1
    projected_sample1 = (sample1 @ direction).cpu().numpy()
    projected_sample2 = (sample2 @ direction).cpu().numpy()

    x_min = min(projected_sample1.min(), projected_sample2.min())
    x_max = max(projected_sample1.max(), projected_sample2.max())
    x_range = (x_min, x_max)
    hist1, _ = np.histogram(projected_sample1, bins=bins, range=x_range, weights=weights1.cpu().numpy(), density=True)
    hist1 /= hist1.sum()
    hist2, _ = np.histogram(projected_sample2, bins=bins, range=x_range, weights=weights2.cpu().numpy(), density=True)
    hist2 /= hist2.sum()
    tv = 0.5 * np.linalg.norm(hist1 - hist2, ord=1)
    return tv


def tv_experiment(m, n_samples, prior_std, n_takes, constant, bins=20):
    digits = [0, 1, 2, 3, 4]
    folder = './out_files/'
    batch_sz = 25
    n_batches = n_samples // batch_sz

    for digit in digits:
        print(f"Creating posteriors for digit '{digit}'")
        for take in range(n_takes):
            print(f"    Image collection #{take+1}")
            seed_mnist = take
            seed_prior = (digit * n_takes + take) * n_batches
            create_reduced_posterior(digit, m, seed_mnist, n_samples, prior_std, batch_sz, seed_prior, constant=constant)

    print(f"Computing TV distances")
    tv_table = np.zeros((len(digits), len(digits)))
    for digit1 in digits:
        for take1 in range(n_takes // 2):
            seed_prior = (digit1 * n_takes + take1) * n_batches
            suffix = f'digit{digit1}_m{m}_seeds{take1}_{seed_prior}'
            fnames1 = get_fnames(folder, 1 / constant, n_samples, prior_std, suffix)

            for digit2 in digits:
                for take2 in range(n_takes // 2, n_takes):
                    seed_prior = (digit2 * n_takes + take2) * n_batches
                    suffix = f'digit{digit2}_m{m}_seeds{take2}_{seed_prior}'
                    fnames2 = get_fnames(folder, 1 / constant, n_samples, prior_std, suffix)

                    tv_table[digit1, digit2] += get_tv_distance(fnames1, fnames2, bins=bins)

    tv_table /= (n_takes // 2)**2
    fname = folder + time.strftime("%Y%m%d-%H%M%S")
    with open(fname + '.npy', 'wb') as f:
        np.save(f, tv_table)
    print(tv_table)
    plot_heatmap(fname)
    print(f"Saved heatmap to {fname}.png")


if __name__ == '__main__':
    m = 30
    n_samples = 10000
    prior_std = 0.03
    n_takes = 10
    constant = 30
    tv_experiment(m, n_samples, prior_std, n_takes, constant, bins=20)
