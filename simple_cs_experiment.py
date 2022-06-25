import torch
import numpy as np
import itertools
import ot
from math import ceil
from tqdm import tqdm
import os

from utils import get_cost_mat, replace_zeros, get_sample_generator


def get_c_concave(phi, cost_mat):
    # 'phi' has size (M, m*n), where M is sample size or 1
    M = phi.shape[0]
    n = cost_mat.shape[0]
    m = phi.shape[1] // n
    phi_c, _ = (cost_mat - phi.reshape(M, m, n, 1)).min(dim=2)  # (M, m, n)
    return phi_c


def objective_function(sample, cost_mat, cs, kappa):
    # 'sample' has size (M, m*n), where M is sample size or 1
    phi_c = get_c_concave(sample, cost_mat)  # (M, m, n)
    phi_bar = sample.reshape(*phi_c.shape).sum(dim=1)  # (M, n)

    logsumexp = -kappa * torch.logsumexp(-phi_bar / kappa, dim=-1)  # (M,)
    inner_prod = (phi_c * cs).sum(dim=(1, 2))  # (M,)
    return (logsumexp + inner_prod)


def experiment(n_samples, prior_std, batch_sz, kappa, store_n_batches, temp=1, calc_vals=True):
    device = 'cuda'
    folder = '/content/drive/MyDrive/simple_cs_experiment/'
    if not os.path.isdir(folder):
        folder = './'
    info = f"{n_samples}_{prior_std}_{kappa}_"

    mean_potentials = torch.load(folder + 'Mean.pt', map_location=device).detach()
    prior_mean = mean_potentials.reshape(1, -1).expand(batch_sz, -1)
    mean_measure = torch.softmax(-mean_potentials.sum(dim=0) / kappa, dim=0)
    cost_mat = get_cost_mat(28, device, dtype=mean_potentials.dtype)

    if calc_vals:
        assert n_samples % (batch_sz * store_n_batches) == 0
        cs = torch.load(folder + 'Archetypes.pt', map_location=device)

        n = cost_mat.shape[0]
        n_batches = ceil(n_samples / batch_sz)
        store_size = batch_sz * store_n_batches
        sample_generator = get_sample_generator(prior_mean, n_batches, prior_std)
        objective = lambda sample: objective_function(sample, cost_mat, cs, kappa)

        # Calculate objective values and distances from mean for all samples
        measures = torch.empty(store_size, n, device=device)
        obj_vals = torch.empty(n_samples, device=device)
        distances = torch.empty(n_samples, device=device)

        for i, batch in enumerate(tqdm(sample_generator(), total=n_batches)):
            position = (i % store_n_batches) * batch_sz
            measures[position:position + batch_sz] = torch.softmax(-batch.reshape(batch_sz, -1, n).sum(dim=1) / kappa, dim=1)

            if (i % store_n_batches) == (store_n_batches - 1):
                dst = replace_zeros(measures.T.contiguous(), replace_val=1e-7, sumdim=0)
                position = (i // store_n_batches) * store_size
                distances[position:position + store_size] \
                    = ot.sinkhorn2(mean_measure, dst, cost_mat, reg=2e-2)

            position = i * batch_sz
            obj_vals[position:position + batch_sz] = objective(batch)

        torch.save(obj_vals, folder + info + 'obj_vals.pt')
        torch.save(distances, folder + info + 'distances.pt')

    else:
        obj_vals = torch.load(folder + info + 'obj_vals.pt')
        distances = torch.load(folder + info + 'distances.pt')

    # Find radius
    weights = torch.softmax(temp * obj_vals, dim=0)
    distances, indices = torch.sort(distances, descending=True)
    weights = weights[indices]

    threshold = 0.05
    outside_weight = 0.
    for i in range(n_samples):
        outside_weight += weights[i]
        if outside_weight > threshold:
            if i:
                print(f"Prev. dist: {distances[i - 1]:.3f}")
            print(f"i = {i}, cur. dist: {distances[i]:.3f}")
            radius = (distances[i] + distances[i - 1]) / 2 if i else distances[i]
            print(f"radius: {radius:.3f}")
            break

    # Calculate distances for other barycenters
    barycenter_set = torch.load(folder + 'BarycenterSet.pt', map_location=device).detach()  # (10, 100, 784)
    n_digits, n_examples, _ = barycenter_set.shape
    bary_distances = torch.empty(n_digits, n_examples, device=device)

    for digit in tqdm(range(n_digits)):
        dst = replace_zeros(barycenter_set[digit].T.contiguous(), replace_val=1e-7, sumdim=0)
        bary_distances[digit] = ot.sinkhorn2(mean_measure, dst, cost_mat, reg=2e-2)

    avgs = bary_distances.mean(dim=1)
    mins, _ = bary_distances.min(dim=1)
    maxs, _ = bary_distances.max(dim=1)
    stats = {'digits': torch.arange(n_digits), 'avgs': avgs, 'mins': mins, 'maxs': maxs}
    for name, stat in stats.items():
        print(name)
        print(stat.tolist())


if __name__ == '__main__':
    n_samples = int(1e3)
    batch_sz = 25
    prior_std = 0.1
    kappa = 1. / 40.
    store_n_batches = 10
    temp = 10
    calc_vals = False

    experiment(n_samples, prior_std, batch_sz, kappa, store_n_batches, temp=temp, calc_vals=calc_vals)
