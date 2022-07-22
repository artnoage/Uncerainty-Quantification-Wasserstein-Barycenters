import torch
from math import ceil
from tqdm import tqdm

from utils import get_cost_mat, get_sample_generator
from simple_cs_experiment import objective_function


def reduce_potentials(batch, n):
    # batch shape (batch_sz, mn)
    batch_sz = batch.shape[0]
    reduced_potential = batch.reshape(batch_sz, -1, n).mean(dim=1)  # (batch_sz, n)
    reduced_potential -= reduced_potential.mean(dim=1, keepdim=True)
    return reduced_potential


def test_reduced_space():
    m = 11
    n = 28**2
    batch_sz = 13
    kappa = 1. / 3.
    batch = torch.randn(batch_sz, m * n)
    bary_true = torch.softmax(-batch.reshape(batch_sz, -1, n).sum(dim=1) / kappa, dim=1)
    reduced_poten = reduce_potentials(batch, n)
    bary_reduced = torch.softmax(-m * reduced_poten / kappa, dim=1)
    print(f"Correct?", torch.allclose(bary_true, bary_reduced, atol=1e-5))


def create_reduced_posterior(mean_potentials, cs, n_samples, prior_std, batch_sz, kappa, temp=1, out_name='', seed=0, n_partitions=4):
    device = 'cuda'
    folder = './'
    info = f"{n_samples}_{prior_std}_{round(kappa, 2)}_"

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

    torch.save(obj_vals, folder + info + f'obj_vals_{out_name}.pt')
    reduced_potentials_cpu = torch.vstack(reduced_potentials_cpu)
    torch.save(reduced_potentials_cpu, folder + info + f"reduced_potentials_{out_name}.pt")

    weights = torch.softmax(temp * obj_vals, dim=0)
    posterior_mean = weights @ reduced_potentials_cpu.cuda()
    torch.save(posterior_mean, folder + info + f"reduced_mean_{out_name}.pt")


def get_bounds(sample, mean, direction, weights):
    sample1d = sample @ direction
    mean1d = mean @ direction
    proximity = torch.abs(sample1d - mean1d)

    proximity, indices = torch.sort(proximity, descending=True)
    weights_sorted = weights[indices]

    i = 0
    weight_outside = 0.
    while weight_outside < 0.05:
        weight_outside += weights_sorted[i]
        i += 1

    dist = proximity[0] if i == 1 else (proximity[i-1] + proximity[i-2]) / 2.

    return dist.item(), mean1d.item()


def two_sample_test(target_names):
    device = 'cuda'

    n_samples = int(1e4)
    prior_std = 0.01
    kappa = 1. / 30.
    temp = 1

    save_reduced = False
    if save_reduced:
        batch_sz = 25
        out_name = 'target_5x30'
        mean_potentials = torch.load(target_names[0], map_location=device).detach().squeeze()
        cs = torch.load(target_names[1], map_location=device).squeeze()
        create_reduced_posterior(mean_potentials, cs, n_samples, prior_std, batch_sz, kappa, temp=temp, out_name=out_name, n_partitions=5)

        for m in [20, 30, 40]:
            pot_set = torch.load(f"PotentialSet_{m}.pt", map_location=device).detach()
            arc_set = torch.load(f"ArchetypeSet_{m}.pt", map_location=device).detach()
            for digit in range(10):
                mean_potentials = pot_set[digit, 0]
                cs = arc_set[digit, 0]
                out_name = f"{digit}x{m}"
                create_reduced_posterior(mean_potentials, cs, n_samples, prior_std, batch_sz, kappa, temp=temp, out_name=out_name, seed=n_samples*(m+digit), n_partitions=5)

    info = f"{n_samples}_{prior_std}_{round(kappa, 2)}_"
    posterior_mean = torch.load(info + f"reduced_mean_target_5x30.pt", map_location=device)  # (784,)
    target_sample = torch.load(info + f"reduced_potentials_target_5x30.pt", map_location=device)  # (1e4, 784,)
    target_obj_vals = torch.load(info + f"obj_vals_target_5x30.pt", map_location=device)  # (1e4,)
    target_weights = torch.softmax(temp * target_obj_vals, dim=0)  # (1e4,)
    for m in [20, 30, 40]:
        print('='*10, 'm =', m, '='*10)
        for digit in range(10):
            print(f"digit {digit}")
            candidate_mean = torch.load(info + f"reduced_mean_{digit}x{m}.pt", map_location=device)
            candidate_sample = torch.load(info + f"reduced_potentials_{digit}x{m}.pt", map_location=device)
            candidate_obj_vals = torch.load(info + f"obj_vals_{digit}x{m}.pt", map_location=device)
            candidate_weights = torch.softmax(temp * candidate_obj_vals, dim=0)
            direction = candidate_mean - posterior_mean
            print(f"Dist between posterior means: {round(torch.norm(direction).item(), 3)}")
            direction /= torch.norm(direction)

            target_dist, target_mean1d = get_bounds(target_sample, posterior_mean, direction, target_weights)
            candidate_dist, candidate_mean1d = get_bounds(candidate_sample, candidate_mean, direction, candidate_weights)
            if target_mean1d + target_dist < candidate_mean1d - candidate_dist:
                print(f"Posteriors are separated: mean1 {round(target_mean1d, 3)}, dist1 {round(target_dist, 3)}, mean2 {round(candidate_mean1d, 3)}, dist2 {round(candidate_dist, 3)}")
            else:
                print(f"Posteriors intersect: mean1 {round(target_mean1d, 3)}, dist1 {round(target_dist, 3)}, mean2 {round(candidate_mean1d, 3)}, dist2 {round(candidate_dist, 3)}")


if __name__ == '__main__':
    target_names = ["Potentials5_30.pt", "Archetypes5_30.pt"]
    two_sample_test(target_names)
