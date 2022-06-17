from utils import  *
import torch
import numpy as np


device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


Dim=28
cost_mat = get_cost_mat(28, device, dtype=torch.float32)


def get_c_concave(phi, cost_mat):
    # 'phi' has size (M, m*n), where M is sample size or 1
    M = phi.shape[0]
    n = cost_mat.shape[0]
    m = phi.shape[1] // n
    phi_c, _ = (cost_mat - phi.reshape(M, m, n, 1)).min(dim=2)  # (M, m, n)
    return phi_c



def objective_function(sample, cost_mat, cs, kappa):
    # 'sample' has size (M, m*n), where M is sample size or 1
    sample=sample.unsqueeze(0)
    phi_c = get_c_concave(sample, cost_mat)  # (M, m, n)
    phi_bar = sample.reshape(*phi_c.shape).sum(dim=1)  # (M, n)

    logsumexp = -kappa * torch.logsumexp(-phi_bar / kappa, dim=-1)  # (M,)
    inner_prod = (phi_c * cs).sum(dim=(1, 2))  # (M,)
    return (logsumexp + inner_prod)


Archetypes= torch.load('Archetypes.pt').to(device).to(torch.float32)
BlurryArchetypes=Archetypes

constant=40
iterations=300

Mean = torch.zeros((len(Archetypes)*len(Archetypes[0]))).to(device).to(torch.float32)
Mean.requires_grad=True
for i in range(iterations):
    obj=-objective_function(Mean,cost_mat, BlurryArchetypes,1/constant)
    loss=obj
    loss.backward()
    with torch.no_grad():
        Mean.sub_(Mean.grad/(2*np.sqrt(i+1)))
        Mean.grad.zero_()
Mean.requires_grad=False

Mean=Mean.reshape(len(Archetypes),len(Archetypes[0]))
Barycenter=torch.softmax(-constant*((torch.sum(Mean,dim=0))),dim=0)
Barycenters=[Barycenter, Barycenter]
titles =  ['Barycenter'] + ['BlurryBarycenter']
show_barycenters(Barycenters, Dim,  'duals', use_softmax=False, iterations=titles, scaling='partial',use_default_folder=False)