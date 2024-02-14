import torch
from torchvision.models import resnet18, ResNet18_Weights

a = torch.tensor([2., 3.], requires_grad=True).reshape(-1,1)
b = torch.tensor([1., 5.], requires_grad=True).reshape(-1,1)
Q = 3*a**3 - b**2
grad_sum = torch.autograd.grad(outputs=Q.sum(), inputs=(a,b), create_graph=True)

second_grad0 = torch.autograd.grad(outputs=grad_sum[0].sum(), inputs= a, create_graph=True)[0]
second_grad = torch.autograd.grad(outputs=grad_sum[1].sum(), inputs= b, create_graph=True)[0]


# CORRETTO, DEVI ESEGUIRE LA DERIVATA RISPETTO ALLE COLONNE FIN DA SUBITO
a = torch.tensor([2., 3.], requires_grad=True).reshape(-1,1)
b = torch.tensor([1., 5.], requires_grad=True).reshape(-1,1)
variables = torch.cat((a,b), dim=1)
print(variables)

a1 = variables[:,0].clone().detach().requires_grad_(True).reshape(-1,1)
b1 = variables[:,1].clone().detach().requires_grad_(True).reshape(-1,1)
print(a1)
print(b1)
Q = 3*a1**3 - b1**2
print(Q)
grad_sum = torch.autograd.grad(outputs=Q.sum(), inputs=(a1,b1), create_graph=True)
print(grad_sum)
final_grad = torch.cat((grad_sum[0], grad_sum[1]), dim=1)
print(final_grad)

second_grad0 = torch.autograd.grad(outputs=grad_sum[0].sum(), inputs= a1, create_graph=True)[0]
second_grad = torch.autograd.grad(outputs=grad_sum[1].sum(), inputs= b1, create_graph=True)[0]
print(second_grad0)
print(second_grad)
print(second_grad0+second_grad)

f = torch.linalg.vector_norm(second_grad0, ord= 1)/2
print(f, f.data)