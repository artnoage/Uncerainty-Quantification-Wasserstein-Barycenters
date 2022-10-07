import torch
from utils import get_cost_mat, load_mnist, show_barycenter, replace_zeros
from tqdm import tqdm
from ot import barycenter as pot_barycenter
import matplotlib.pyplot as plt


def get_c_concave(phi, cost_mat):
    # 'phi' has size (M, m*n), where M is sample size or 1
    M = phi.shape[0]
    n = cost_mat.shape[0]
    m = phi.shape[1] // n
    phi_c, _ = (cost_mat - phi.reshape(M, m, n, 1)).min(dim=2)  # (M, m, n)
    return phi_c


def objective_function(sample, cost_mat, cs, kappa, regterm=False):
    """Old (non-smooth) objective function."""
    # 'sample' has size (M, m*n), where M is sample size or 1
    phi_c = get_c_concave(sample, cost_mat)  # (M, m, n)
    phi_bar = sample.reshape(*phi_c.shape).sum(dim=1)  # (M, n)

    logsumexp = -kappa * torch.logsumexp(-phi_bar / kappa, dim=-1)  # (M,)
    inner_prod = (phi_c * cs).sum(dim=(1, 2))  # (M,)
    obj_val = logsumexp + inner_prod
    if regterm:
        obj_val -= 0.1 * torch.norm(sample)**2
    return obj_val


def smooth_objective(sample, cost_mat, cs, kappa, gamma, regterm=False):
    """Smooth objective function."""
    # 'sample' has size (M, 2m*n), where M is sample size or 1
    n = cost_mat.shape[0]
    m = cs.shape[0]
    lam, mu = sample[:, :m*n], sample[:, m*n:]  # (M, m*n), (M, m*n)
    lam_sum = lam.reshape(-1, m, n).sum(dim=1)  # (M, n)

    logsumexp1 = kappa * torch.logsumexp(lam_sum / kappa, dim=-1)  # (M,)
    inner_prod = (mu.reshape(-1, m, n) * cs).sum(dim=(1, 2))  # (M,)
    # Matrix under Logsumexp has shape (M, m, n, n), Logsumexp itself has shape  # (M, m)
    logsumexp2 = gamma * torch.logsumexp(
        -(cost_mat + lam.reshape(-1, m, n, 1) + mu.reshape(-1, m, 1, n)) / gamma, dim=(2, 3)).sum(dim=1)  # (M,)
    obj_val = -logsumexp1 - inner_prod - logsumexp2
    if regterm:
        obj_val -= 0.1 * torch.norm(sample)**2
    return obj_val


def get_potentials(cs, n_iter=2000, constant=30, smooth=False, gamma=None, regterm=True, get_loss=False):
    cost_mat = get_cost_mat(28, cs.device, dtype=cs.dtype)
    variable_dim = cs.numel() * (2 if smooth else 1)  # (1, m*n) or (1, 2m*n)
    Mean = torch.zeros(1, variable_dim, device=cs.device, dtype=cs.dtype, requires_grad=True)
    if gamma is None:
        gamma = 1./constant
    objective = lambda x: -smooth_objective(x, cost_mat, cs, 1./constant, gamma, regterm=regterm) if smooth\
        else -objective_function(x, cost_mat, cs, 1./constant, regterm=regterm)

    losses = []
    for k in tqdm(range(n_iter)):
        obj = objective(Mean)
        losses.append(obj.item())
        loss = obj
        loss.backward()
        with torch.no_grad():
            Mean.sub_(Mean.grad / (k + 1))
            Mean.grad.zero_()

    Mean.requires_grad = False
    if get_loss:
        return Mean.reshape(-1, cs.shape[1]), losses
    else:
        return Mean.reshape(-1, cs.shape[1])  # shape (m, n) or (2m, n)


if __name__ == '__main__':
    m = 30
    constant = 30  # inverse 'kappa' coefficient
    digit = 5
    device = 'cuda'
    seed_mnist = 0
    n_iter = 500
    out_name = f'digit{digit}_m{m}_seed{seed_mnist}'
    cs = load_mnist(m, digit, device, seed=seed_mnist)  # shape (m, n)
    cs = replace_zeros(cs)

    # Run Grad Descent
    gamma = 0.1/constant
    lam_mu, losses = get_potentials(cs, n_iter=n_iter, constant=constant, smooth=True,
                                    gamma=gamma, regterm=False, get_loss=True)
    # lam_mu has shape (2m, n)

    # Plot objective function
    plt.plot(losses, label=fr'$\kappa^{-1}={constant}$')
    plt.yscale('log')
    plt.savefig(f"out_files/gd_convergence.png", bbox_inches='tight')
    plt.close()

    # Visualize corresponding barycenter
    lam_sum = lam_mu[:m, :].reshape(m, -1).sum(dim=0)  # (n,)
    bary_from_poten = torch.softmax(lam_sum * constant, dim=0)  # (n,)
    show_barycenter(bary_from_poten, out_name + f'_frompoten_{n_iter}iter', im_sz=28)

    # Visualize Sinkhorn barycenter for comparison
    cost_mat = get_cost_mat(28, cs.device, dtype=cs.dtype)
    r = pot_barycenter(cs.T, cost_mat, 2e-3)
    show_barycenter(r, out_name + '_sinkh', im_sz=28)
