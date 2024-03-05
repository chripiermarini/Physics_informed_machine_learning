from neuralop.models import FNO
import torch
from neuralop.datasets import load_darcy_flow_small


"""
This script extracts data from the darcy_flow dataset and takes only the first sample
The PDE_model function takes a single sample as an input and returns the p_matrix (i.e. the value of the objective 
for the 16x16 samples), and the output of the neural network (i.e. the u(x) function, from which we extract the boundary
values for the constraints of the optimization problem)"""

device = 'cpu'
train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=1000, batch_size=32,
        test_resolutions=[16, 32], n_tests=[100, 50],
        test_batch_sizes=[32, 32],
        positional_encoding=True
)
data_processor = data_processor.to(device)
train_dataset = train_loader.dataset
train_sample = train_dataset[0]  ##single sample

def pde_model(train_sample):

    "Set-up of the FNO model and the preprocessing of data"
    model = FNO(n_modes=(16, 16), hidden_channels=64,
                in_channels=3, out_channels=1)
    model = model.to(device)
    data = data_processor.preprocess(train_sample, batched=False)

    "Extraction of the input matrices, differentiating nu, x1 and x2"
    input = data['x']
    nu = input[0].requires_grad_(True)
    x1 = input[1].requires_grad_(True)
    x2 = input[2].requires_grad_(True)

    "We perform the forward of our FNO model and then we compute the differential operator value for each point"
    out = model(torch.stack((nu, x1, x2), 0).unsqueeze(0)).requires_grad_(True)
    first_derivative = torch.autograd.grad(outputs=out.sum(),
                                           inputs=(x1, x2),
                                           create_graph=True,
                                           allow_unused=True)
    div_argument = [nu * first_derivative[i] for i in range(len(first_derivative))]
    second_derivative_x1 = torch.autograd.grad(outputs=div_argument[0].sum(), inputs=x1, create_graph=True)
    second_derivative_x2 = torch.autograd.grad(outputs=div_argument[1].sum(), inputs=x2, create_graph=True)
    p_matrix = - (second_derivative_x1[0] + second_derivative_x2[0]) - 1

    return p_matrix, out

p_matrix, out = pde_model(train_sample)