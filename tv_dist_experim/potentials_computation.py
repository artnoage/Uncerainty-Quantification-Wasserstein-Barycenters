import torch
from utils import get_cost_mat


def get_c_concave(phi, cost_mat):
    # 'phi' has size (M, m*n), where M is sample size or 1
    M = phi.shape[0]
    n = cost_mat.shape[0]
    m = phi.shape[1] // n
    phi_c, _ = (cost_mat - phi.reshape(M, m, n, 1)).min(dim=2)  # (M, m, n)
    return phi_c


def objective_function(sample, cost_mat, cs, kappa, regterm=False):
    # 'sample' has size (M, m*n), where M is sample size or 1
    phi_c = get_c_concave(sample, cost_mat)  # (M, m, n)
    phi_bar = sample.reshape(*phi_c.shape).sum(dim=1)  # (M, n)

    logsumexp = -kappa * torch.logsumexp(-phi_bar / kappa, dim=-1)  # (M,)
    inner_prod = (phi_c * cs).sum(dim=(1, 2))  # (M,)
    obj_val = logsumexp + inner_prod
    if regterm:
        obj_val -= 0.1 * torch.norm(sample)**2
    return obj_val


def get_potentials(cs, n_iter=2000, constant=30):
    cost_mat = get_cost_mat(28, cs.device, dtype=cs.dtype)
    Mean = torch.zeros(1, cs.numel(), device=cs.device, dtype=cs.dtype, requires_grad=True)  # shape (m*n,)

    for k in range(n_iter):
        obj = -objective_function(Mean, cost_mat, cs, 1./constant, regterm=True)
        loss = obj
        loss.backward()
        with torch.no_grad():
            Mean.sub_(Mean.grad / (k + 1))
            Mean.grad.zero_()

    Mean.requires_grad = False
    return Mean.reshape(*cs.shape)  # shape (m, n)
