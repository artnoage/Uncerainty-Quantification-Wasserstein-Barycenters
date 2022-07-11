import torch
from math import ceil
from tqdm import tqdm

from utils import get_cost_mat, get_sample_generator
from simple_cs_experiment import objective_function


def reduce_potentials(batch, n):
    # batch shape (batch_sz, mn)
    batch_sz = batch.shape[0]
    reduced_potential = batch.reshape(batch_sz, -1, n).sum(dim=1)  # (batch_sz, n)
    reduced_potential -= reduced_potential.sum(dim=1, keepdim=True)
    return reduced_potential


def experiment(n_samples, prior_std, batch_sz, kappa, temp=1):
    device = 'cuda'
    folder = './'
    info = f"{n_samples}_{prior_std}_{kappa}_"

    mean_potentials = torch.load(folder + 'Mean.pt', map_location=device).detach()
    cs = torch.load(folder + 'Archetypes.pt', map_location=device)
    cost_mat = get_cost_mat(28, device, dtype=mean_potentials.dtype)
    n = cost_mat.shape[0]

    n_batches = ceil(n_samples / batch_sz)
    prior_mean = mean_potentials.reshape(1, -1).expand(batch_sz, -1)
    sample_generator = get_sample_generator(prior_mean, n_batches, prior_std)
    objective = lambda sample: objective_function(sample, cost_mat, cs, kappa)

    obj_vals = torch.empty(n_samples, device=device)
    reduced_potentials = torch.empty(n_samples, n, device=device)

    for i, batch in enumerate(tqdm(sample_generator(), total=n_batches)):
        position = i * batch_sz
        obj_vals[position:position + batch_sz] = objective(batch)
        reduced_potentials[position:position + batch_sz] = reduce_potentials(batch, n)

    torch.save(obj_vals, folder + info + 'obj_vals.pt')
    torch.save(reduced_potentials, folder + info + f"reduced_potentials.pt")

    weights = torch.softmax(temp * obj_vals, dim=0)
    posterior_mean = weights @ reduced_potentials
    torch.save(posterior_mean, folder + info + f"reduced_posterior_mean.pt")


if __name__ == '__main__':
    n_samples = int(1e3)
    batch_sz = 5
    prior_std = 0.01
    kappa = 1.  # / 40.
    temp = 1

    experiment(n_samples, prior_std, batch_sz, kappa, temp=1)
