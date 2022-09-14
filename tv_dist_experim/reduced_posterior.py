import torch
from math import ceil
from tqdm import tqdm
import os.path

from utils import get_cost_mat, get_sample_generator, get_fnames
from potentials_computation import objective_function


def reduce_potentials(batch, n):
    # batch shape (batch_sz, mn)
    batch_sz = batch.shape[0]
    reduced_potential = batch.reshape(batch_sz, -1, n).mean(dim=1)  # (batch_sz, n)
    reduced_potential -= reduced_potential.mean(dim=1, keepdim=True)
    return reduced_potential


def get_reduced_posterior(mean_potentials, cs, n_samples, prior_std, batch_sz, kappa, temp=1, suffix='', seed=0, n_partitions=4):
    """
    Sample a batch from prior, calculate objective values and reduced potentials. Repeat until we used
    almost all GPU memory. Form a 'partition' from the obtained samples, transfer it to the CPU and continue
    the process. Compute weights and posterior mean. Save them and all reduced potentials to the disk.

    It's best if n_partitions divides n_samples and batch_sz divides partition_size (i.e., n_samples/n_partitions),
    otherwise behavior might be unexpected.
    """
    device = 'cuda'
    folder = './out_files/'
    fnames = get_fnames(folder, kappa, n_samples, prior_std, suffix)
    if all([os.path.isfile(fname) for fname in fnames]):
        print("        Posterior already exists")
        return

    print("        Sampling and calculating objective to create posterior")
    fname_red_mean, fname_red_pot, fname_obj_vals = fnames

    cost_mat = get_cost_mat(28, device, dtype=mean_potentials.dtype)
    n = cost_mat.shape[0]

    n_batches = ceil(n_samples / batch_sz)
    prior_mean = mean_potentials.reshape(1, -1).expand(batch_sz, -1)
    sample_generator = get_sample_generator(prior_mean, n_batches, prior_std, seed=seed)
    objective = lambda sample: objective_function(sample, cost_mat, cs, kappa)

    obj_vals = torch.empty(n_samples, device=device)
    partition_size = n_samples // n_partitions
    reduced_potentials = torch.empty(partition_size, n, device=device)
    reduced_potentials_cpu = []

    for i, batch in enumerate(tqdm(sample_generator(), total=n_batches)):
        position = i * batch_sz
        obj_vals[position:position + batch_sz] = objective(batch)
        position = position % partition_size
        reduced_potentials[position:position + batch_sz] = reduce_potentials(batch, n)
        if position + batch_sz == partition_size:
            reduced_potentials_cpu.append(reduced_potentials.cpu())

    torch.save(obj_vals, fname_obj_vals)  # filename of format 'M1e4_std0.1_kappa.03_obj_vals_{out_name}.pt'
    reduced_potentials_cpu = torch.vstack(reduced_potentials_cpu)
    torch.save(reduced_potentials_cpu, fname_red_pot)  # filename of format 'M1e4_std0.1_kappa.03_red_pot_{out_name}.pt'

    weights = torch.softmax(temp * obj_vals, dim=0)
    posterior_mean = weights.cpu() @ reduced_potentials_cpu
    torch.save(posterior_mean, fname_red_mean)  # filename of format 'M1e4_std0.1_kappa.03_red_mean_{out_name}.pt'
    print(f"        Posterior saved to files {fname_obj_vals}, {fname_red_pot}, {fname_red_mean}")
