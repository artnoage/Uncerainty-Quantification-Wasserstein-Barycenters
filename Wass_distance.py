
import ot
import torch
import numpy as np
from utils import * 
import itertools

device=torch.device("cpu")
if torch.cuda.is_available():
  device=torch.device("cuda")





def Transformation(cost,sample):
    lamext=sample.reshape(len(sample),1).expand(len(sample),len(sample)).transpose(1,0)
    lamstar=(cost-lamext).amin(dim=1)
    return (lamstar)

def objective(firstmeasure, secondmeasure, cost, potential):
    potentialdual=Transformation(cost,potential)
    estimation=torch.sum((firstmeasure*potentialdual),dim=0) + torch.sum((secondmeasure*potential),dim=0)
    return (estimation)


A= torch.flatten(replace_zeros(load_mnist( 1, 3, device=device,size=(28,28))))
Dim=int(np.sqrt(len(A)))
cost_mat = get_cost_mat(28, device, dtype=torch.float32)

def wass_dis(mu,nu,cost):
    D = ot.sinkhorn(mu, nu,cost, reg=1e-3, numItermax=10000)
    a=torch.sum(cost*D)
    iterations=3000
    potential = torch.zeros(len(mu)).to(device)
    potential.requires_grad=True
    for i in range(iterations):
        obj=-objective(mu,nu,cost,potential)
        loss=obj
        loss.backward()
        with torch.no_grad():
            potential.sub_(potential.grad/(np.sqrt(i)+1))
            potential.grad.zero_()
    potential.requires_grad=False
    print(-loss,a)

for i in range(1000):
    A1= torch.flatten(replace_zeros(load_mnist( 1, 6, device=device,size=(28,28))))
    A2= torch.flatten(replace_zeros(load_mnist( 1, 7, device=device,size=(28,28))))
    wass_dis(A1,A2,cost_mat)