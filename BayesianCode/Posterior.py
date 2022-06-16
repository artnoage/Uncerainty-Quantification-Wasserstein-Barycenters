import torch
import ot
from math import ceil
import numpy as np
import itertools
from tqdm import tqdm


device=torch.device('cpu')
if torch.cuda.is_available():
  device=torch.device("cuda")


def get_cost_mat(im_sz, device, dtype=torch.float32):
    partition = torch.linspace(0, 1, im_sz)
    couples = np.array(np.meshgrid(partition, partition)).T.reshape(-1, 2)
    x = np.array(list(itertools.product(couples, repeat=2)))
    x = torch.tensor(x, dtype=dtype, device=device)
    a = x[:, 0]
    b = x[:, 1]
    C = torch.linalg.norm(a - b, axis=1) ** 2
    return C.reshape((im_sz**2, -1))


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
    phi_bar = phi_c.sum(dim=1)  # (M, n)

    logsumexp = -kappa * torch.logsumexp(-phi_bar / kappa, dim=-1)  # (M,)
    inner_prod = (phi_c * cs).sum(dim=(1, 2))  # (M,)
    return (logsumexp + inner_prod)


def get_sample_generator(prior_mean, n_batches, prior_std):
    def sample_generator():
        for i in range(n_batches):
            torch.manual_seed(i)
            yield torch.normal(prior_mean, prior_std)

    return sample_generator

def get_obj_vals_and_bary(n_samples, prior_std, mean_potentials, folder, batch_sz, cost_mat,
                          kappa, max_samples, sampling_params, device=device):
    prior_mean = mean_potentials.reshape(1, -1).expand(batch_sz, -1)
    n_batches = ceil(n_samples / batch_sz)
    n = cost_mat.shape[0]
    sample_generator = get_sample_generator(prior_mean, n_batches, prior_std)

    cs = torch.load(folder + 'Archetypes.pt', map_location=device)
    m = cs.shape[0]
    objective = lambda sample: objective_function(sample, cost_mat, cs, kappa)

    best = torch.empty(max_samples + batch_sz, n, device=device)
    obj_vals = torch.empty(max_samples + batch_sz, device=device)

    for i, batch in enumerate(tqdm(sample_generator(), total=n_batches)):
        position = i * batch_sz
        if position > (max_samples - batch_sz):
            best[max_samples:] = torch.softmax(-batch.reshape(batch_sz, -1, n).sum(dim=1) / kappa, dim=1)
            obj_vals[max_samples:] = objective(batch)
            obj_vals, indices = torch.sort(obj_vals,descending=True)
            best = best[indices]
        else:
            best[position:position + batch_sz] = torch.softmax(-batch.reshape(batch_sz, -1, n).sum(dim=1) / kappa, dim=1)
            obj_vals[position:position + batch_sz] = objective(batch)

    obj_vals, best = obj_vals[:max_samples], best[:max_samples]
    torch.save(obj_vals, folder + sampling_params + '_obj_vals.pt')
    torch.save(best, folder + sampling_params + '_best.pt')


def save_best(n_samples, batch_sz, folder, prior_std, kappa, device=device, max_samples=1000):
    sampling_params = f"{n_samples}_{max_samples}_{prior_std}"
    mean_potentials = torch.load(folder + 'Mean.pt', map_location=device).detach()
    cost_mat = get_cost_mat(28, device, dtype=mean_potentials.dtype)

    get_obj_vals_and_bary(
        n_samples, prior_std, mean_potentials, folder, batch_sz, cost_mat,
        kappa, max_samples, sampling_params, device=device)

def get_pairwise_dist(sample, cost_mat, dist_path, sinkhorn_reg=1e-2):
    # sample has shape (n, n_samples)
    n_samples = sample.shape[1]
    dist_mat = torch.zeros(n_samples, n_samples, device=sample.device)
    for i in tqdm(range(n_samples - 1)):
        src = sample[:, i]
        dst = sample[:, i + 1:]
        wass_dist = ot.sinkhorn2(src, dst, cost_mat, reg=sinkhorn_reg)
        dist_mat[i, i + 1:] = wass_dist

    dist_mat = dist_mat + dist_mat.T.contiguous()
    torch.save(dist_mat, dist_path)

def get_rho(x, sample, cost_mat, wght, sinkhorn_reg=1e-2):
    wass_dist = ot.sinkhorn2(x, sample, cost_mat, reg=sinkhorn_reg)
    rho = wght @ wass_dist
    return rho.item()

def replace_zeros(arr, replace_val=1e-6, sumdim=-1):
    arr[arr < replace_val] = replace_val
    arr /= arr.sum(dim=sumdim, keepdim=True)
    return arr

n_samples = 12000000
max_samples = 700
batch_sz = 75
prior_std = 0.01
kappa = 1. / 30.

folder = './'

# UNCOMMENT
#save_best(n_samples, batch_sz, folder, prior_std, kappa, device=device, max_samples=max_samples)



sinkhorn_reg = 0.02
replace_val = 1e-6

sampling_params = f"{n_samples}_{max_samples}_{prior_std}"
sample = torch.load(folder + sampling_params + '_best.pt')
cost_mat = get_cost_mat(28, sample.device, dtype=sample.dtype)
dist_path = folder + sampling_params + '_dist.pt'

sample = replace_zeros(sample, replace_val=replace_val, sumdim=-1)
sample = sample.T.contiguous()

# UNCOMMENT TO PERFORM COMPUTATION
#get_pairwise_dist(sample, cost_mat, dist_path, sinkhorn_reg=sinkhorn_reg)

temperature = 4
dist_mat = torch.load(folder + sampling_params + '_dist.pt')
obj_vals = torch.load(folder + sampling_params + '_obj_vals.pt')
weights = torch.softmax(temperature * obj_vals, dim=-1)
rhos = dist_mat @ weights
pairs = list(zip(rhos.tolist(), weights.tolist()))
sorted_pairs = sorted(pairs, key=lambda tup: tup[0])
threshold = 0.95
weight_in_CS = 0.
i = 0
while weight_in_CS < threshold:
    weight_in_CS += sorted_pairs[i][1]
    i += 1

idx = i if i < max_samples else max_samples - 1
r = sorted_pairs[idx][0]



mean_potentials = torch.load(folder + 'Mean.pt', map_location=sample.device).detach()
mean_poten_sum = mean_potentials.reshape(-1, cost_mat.shape[0]).sum(dim=0)  # (n,)
prior_mean_bary = torch.softmax(-mean_poten_sum / kappa, dim=0)  # (n,)
prior_mean_bary = replace_zeros(prior_mean_bary, replace_val=replace_val, sumdim=-1)
rho_prior_mean = get_rho(prior_mean_bary, sample, cost_mat, weights,
                         sinkhorn_reg=sinkhorn_reg)

barys_folder = './'

with open(folder  + sampling_params + f"_t{temperature}.txt", "w") as handle:
    if barys_folder:
        for target_digit in [5, 6]:
            handle.write(f"\nChecking barycenters of {target_digit}\n")
            barys = torch.load(barys_folder + f"/barys{target_digit}.pt", map_location=sample.device)
            barys = replace_zeros(barys, replace_val=replace_val, sumdim=-1)
            for i in range(barys.shape[0]):
                handle.write(f'{get_rho(barys[i], sample, cost_mat, weights, sinkhorn_reg=sinkhorn_reg):.2e} ')

    handle.write(f"\nConfidence set radius: {r:.2e}\nrho of prior_mean: {rho_prior_mean:.2e}")

with open(folder + sampling_params + f"_t{temperature}.txt", 'r') as handle:
    print(handle.read())