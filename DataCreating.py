from utils import *
import torch


device=torch.device("cpu")
if torch.cuda.is_available():
    device=torch.device("cuda")

Dim=28
NumberOfAtoms=Dim**2
cost_mat = get_cost_mat(28, device, dtype=torch.float32)

def Transformation(cost,sample):
    if sample.dim()==2:
        lamext=sample.reshape(len(sample),len(sample[0]),1).expand(len(sample),len(sample[0]),len(sample[0])).transpose(2,1)
        lamstar=(cost-lamext).amin(dim=2)
    else:
        lamext=sample.reshape(len(sample),len(sample[0]),len(sample[0,1]),1).expand(len(sample),len(sample[0]),len(sample[0,1]),len(sample[0,1])).transpose(3,2)
        lamstar=(cost-lamext).amin(dim=3)
    del lamext
    return (lamstar)

def objective(input, cost,  Data,constant):
    input=input.reshape(len(Data),len(Data[0]))
    inputdual=Transformation(cost,input)
    Dataext=torch.broadcast_to(Data,(len(Data),len(Data[0])))
    regterm=torch.sum(torch.norm(input,dim=1)**2)
    estimation=torch.sum(torch.sum((Dataext*inputdual),dim=1),dim=0)-torch.logsumexp(-constant*(torch.sum(input,dim=0)),dim=0)/constant-0.1*regterm
    return (estimation)

data_n=40
collector_size=100
constant=30
BarycenterSet=torch.zeros((10,collector_size,NumberOfAtoms))
PotentialSet=torch.zeros((10,collector_size,data_n,NumberOfAtoms))
for i in range(collector_size):
    for j in range (10):
        print(i,j)
        Archetypes= replace_zeros(load_mnist(data_n,target_digit=j, device=device,size=(28,28)))
        Mean = torch.zeros((data_n*NumberOfAtoms)).to(device).to(torch.float32)
        Mean.requires_grad=True
        for k in range(2000):
            obj=-objective(Mean,cost_mat, Archetypes,constant)
            loss=obj
            loss.backward()
            with torch.no_grad():
                Mean.sub_(Mean.grad/(k+1))
                Mean.grad.zero_()
        Mean.requires_grad=False
        Mean=Mean.reshape(data_n,NumberOfAtoms)
        Barycenter=torch.softmax(-constant*((torch.sum(Mean,dim=0))),dim=0)
        BarycenterSet[j,i]=Barycenter
        PotentialSet[j,i]=Mean

torch.save(BarycenterSet,"BarycenterSet.pt")
torch.save(PotentialSet,"PotentialSet.pt")
