import ot
from math import ceil
from tqdm import tqdm
from utils import *


def get_pairwise_dist(sample, cost_mat, dist_path, sinkhorn_reg=2e-2):
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


def get_pairwise_dist_batched(sample, cost_mat, dist_path, batch_sz, sinkhorn_reg=2e-2):
    # sample has shape (n, n_samples)
    n_samples = sample.shape[1]
    if batch_sz >= n_samples - 1:
        get_pairwise_dist(sample, cost_mat, dist_path, sinkhorn_reg=sinkhorn_reg)

    dist_mat = torch.zeros(n_samples, n_samples, device=sample.device)
    n_batches = ceil((n_samples - 1) / batch_sz)

    for k in tqdm(range(n_batches)):
        for i in range(n_samples - k * batch_sz - 1):
            src = sample[:, i]
            dst = sample[:, i+1+k*batch_sz:i+1+(k+1)*batch_sz]
            wass_dist = ot.sinkhorn2(src, dst, cost_mat, reg=sinkhorn_reg)
            dist_mat[i, i+1+k*batch_sz:i+1+(k+1)*batch_sz] = wass_dist

    dist_mat = dist_mat + dist_mat.T.contiguous()
    torch.save(dist_mat, dist_path)


def test_batched_dist():
    # Accuracy test
    n_samples = 7
    batch_sz = 3
    im_sz = 28
    dist_path = 'dist_mat_test.pt'
    device = 'cuda'
    cost_mat = get_cost_mat(28, device)

    sample = torch.rand(im_sz ** 2, n_samples, device=device)
    sample /= sample.sum(dim=0, keepdim=True)

    get_pairwise_dist(sample, cost_mat, dist_path)
    get_pairwise_dist_batched(sample, cost_mat, 'batched_' + dist_path, batch_sz)

    pairwise_dist = torch.load(dist_path)
    pairwise_dist_batched = torch.load('batched_' + dist_path)
    print(f'Relative error: {100 * torch.norm(pairwise_dist - pairwise_dist_batched) / torch.norm(pairwise_dist):.2f}%')

    # Big size test
    n_samples = 3000
    batch_sz = 400
    dist_path = 'big_dist_mat_test.pt'

    sample = torch.rand(im_sz ** 2, n_samples, device=device)
    sample /= sample.sum(dim=0, keepdim=True)

    try:
        get_pairwise_dist(sample, cost_mat, dist_path)
        print("Basic function succeeded")
    except:
        print("Basic function failed")
        get_pairwise_dist_batched(sample, cost_mat, 'batched_' + dist_path, batch_sz)
        print("Batch function succeeded")


if __name__ == '__main__':
    test_batched_dist()
