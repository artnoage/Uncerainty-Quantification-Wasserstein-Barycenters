from utils import  get_sampler, show_barycenters 
import torch
import numpy as np
import itertools

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


Dim=28
Partition = torch.linspace(0, 1, Dim,dtype=torch.float32).to(dtype=torch.float32)
couples = np.array(np.meshgrid(Partition, Partition)).T.reshape(-1, 2)
x=np.array(list(itertools.product(couples, repeat=2)))
x = torch.from_numpy(x)
a = x[:, 0]
b = x[:, 1]
C = torch.linalg.norm(a - b, axis=1) ** 2
NumberOfAtoms= Dim**2
C=C.reshape(NumberOfAtoms,NumberOfAtoms).to(device)


def Transformation(cost,sample):
    if sample.dim()==2:
        lamext=sample.reshape(len(sample),len(sample[0]),1).expand(len(sample),len(sample[0]),len(sample[0])).transpose(2,1)
        lamstar=(cost-lamext).amin(dim=2)
    else:
        lamext=sample.reshape(len(sample),len(sample[0]),len(sample[0,1]),1).expand(len(sample),len(sample[0]),len(sample[0,1]),len(sample[0,1])).transpose(3,2)
        lamstar=(cost-lamext).amin(dim=3)
    del lamext
    return (lamstar)

def get_c_concave(phi, cost_mat):
    # 'phi' has size (M, m*n), where M is sample size or 1
    M = phi.shape[0]
    n = cost_mat.shape[0]
    m = phi.shape[1] // n
    phi_c, _ = (cost_mat - phi.reshape(M, m, n, 1)).min(dim=2)  # (M, m, n)
    return phi_c

def objective(input, cost,  Data, constant):
    input=input.reshape(len(Data),len(Data[0]))
    inputdual=Transformation(cost,input)
    Dataext=torch.broadcast_to(Data,(len(Data),len(Data[0])))
    regterm=torch.sum(torch.norm(input,dim=1)**2)
    estimation=torch.sum(torch.sum((Dataext*inputdual),dim=1),dim=0)-torch.logsumexp(-constant*(torch.sum(input,dim=0)),dim=0)/constant
    return (estimation)

def objective_function(sample, cost_mat, cs, kappa):
    # 'sample' has size (M, m*n), where M is sample size or 1
    sample=sample.reshape(1,len(sample))
    phi_c = get_c_concave(sample, cost_mat)  # (M, m, n)
    phi_bar = phi_c.sum(dim=1)  # (M, n)

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
    obj=-objective(Mean,C, Archetypes,1/constant)
    loss=obj
    loss.backward()
    print(i)
    with torch.no_grad():
        Mean.sub_(Mean.grad/(2*np.sqrt(i+1)))
        Mean.grad.zero_()
Mean.requires_grad=False

BlurryMean = torch.zeros((len(Archetypes)*len(Archetypes[0]))).to(device).to(torch.float32)
BlurryMean.requires_grad=True
for i in range(iterations):
    obj=-objective_function(BlurryMean,C, BlurryArchetypes,1/constant)
    loss=obj
    loss.backward()
    with torch.no_grad():
        BlurryMean.sub_(BlurryMean.grad/(2*np.sqrt(i+1)))
        BlurryMean.grad.zero_()
BlurryMean.requires_grad=False

Mean=Mean.reshape(len(Archetypes),len(Archetypes[0]))
BlurryMean=BlurryMean.reshape((len(Archetypes),len(Archetypes[0])))
print(torch.norm(torch.norm(Mean-BlurryMean,dim=1)))
Barycenter=torch.softmax(-constant*((torch.sum(Mean,dim=0))),dim=0)
BlurryBarycenter=torch.softmax(-constant*((torch.sum(BlurryMean,dim=0))),dim=0)
print(torch.norm(Barycenter-BlurryBarycenter))
Barycenters=[Barycenter, BlurryBarycenter]
print(objective(Mean,C, Archetypes,constant)-objective(BlurryMean,C, Archetypes,constant))
titles =  ['Barycenter'] + ['BlurryBarycenter']+['Archetype1'] + ['Archetype2'] +['BlurryArchetype1'] + ['BlurryArchetype2']
show_barycenters(Barycenters, Dim,  'duals', use_softmax=False, iterations=titles, scaling='partial',use_default_folder=False)