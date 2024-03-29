from nn_architecture import *
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from problem_base import BaseProblem

""" Problem statement Burgers' equation - TO BE CHECKED TOMORROW """

print("ciaoooo")
class Burgers(BaseProblem):
    name = 'Burgers'
    def __init__(self, device, n_obj_sample=100, n_constrs=3, reg=1, constraint_type='pde', t_index=0):

        # Initialize NN
        self.n_input = 2
        self.n_output = 1
        self.net = OneHiddenLayerFCNN(self.n_input, self.n_output, n_neurons = 16)
        # self.net = FCN(self.n_input, self.n_output, 32, 3)
        # self.net = TwoHiddenLayerFCNN(self.n_input, self.n_output,n_neurons = 64)
        self.net.to(device)
        self.n_parameters = self.count_parameters(self.net)
        self.reg = torch.tensor(reg)
        self.constraint_type = constraint_type
        self.t_index = t_index  # used in the generate samples

        # Generate sample
        self.n_obj_sample = n_obj_sample
        self.n_constrs = n_constrs
        self.pde_dataset, self.t_all, self.x_all, self.constr_row_select = self.generate_sample()
        self.mse_cost_function = torch.nn.MSELoss()  # Mean squared error

    def generate_sample(self):
        # all possible couples for the pde of the objective constraints
        t_all = torch.linspace(0, 1, self.n_obj_sample).view(-1, 1)
        x_all = torch.linspace(0, 1, self.n_obj_sample).view(-1, 1)

        pde_dataset = np.column_stack((np.meshgrid(x_all, t_all)[0].flatten(),
                                               np.meshgrid(x_all, t_all)[1].flatten()))
        perm_indices = torch.randperm(pde_dataset.shape[0])
        pde_dataset = torch.Tensor(pde_dataset[perm_indices][:self.n_obj_sample])

        ## first boundary constraint dataset must be built using t_all and x_all
        ## since we are using (x,0) couples for the first_boundary and (0,t) for the second_boundary
        constr_row_select = np.random.randint(0, self.n_obj_sample, size=self.n_constrs)

        return pde_dataset, t_all, x_all, constr_row_select

    def initial_condition(self, x):
        # Parameters
        A = 1.0  # Amplitude
        x0 = 0  # Center of the Gaussian
        sigma = 0.1  # Standard deviation

        # Initial condition
        u_initial = A * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
        return u_initial

    def pde(self, pde_dataset):
        pde_x = pde_dataset[:,0].requires_grad_(True)
        pde_t = pde_dataset[:,1].requires_grad_(True)

        output = self.net(torch.stack((pde_x, pde_t), 1)).requires_grad_(True) #u(x,t) function
        output_dt = torch.autograd.grad(outputs= output.sum(),
                                            inputs=pde_t,
                                            create_graph=True,
                                            allow_unused=True
                                            )[0]
        output_dx = torch.autograd.grad(outputs=output.sum(),
                                            inputs=pde_x,
                                            create_graph=True,
                                            allow_unused=True
                                            )[0]
        output_second_dx = torch.autograd.grad(outputs=output_dx.sum(),
                                                   inputs=pde_x,
                                                   create_graph=True,
                                                   allow_unused=True
                                                   )[0]
        pde_function = output_dt + torch.mul(output.squeeze(), output_dx) - 0.01*output_second_dx
        return pde_function

    def objective_func(self, return_multiple_f = False):
        #pde objective function
        pde_function = self.pde(self.pde_dataset)
        pde_loss = torch.mean(pde_function ** 2)

        #first boundary
        initial_values = self.initial_condition(self.x_all.squeeze())
        first_boundary_input = torch.stack((self.x_all.squeeze(), torch.zeros(self.n_obj_sample)), dim=1)
        first_boundary_value = self.net(first_boundary_input).squeeze() - initial_values
        first_boundary_loss = torch.mean(first_boundary_value**2)

        #second_boundary
        ### first we generate the (0,t) (1,t) couples
        lhs_dataset = torch.stack((torch.zeros(self.n_obj_sample), self.t_all.squeeze()), dim=1)
        rhs_dataset = torch.stack((torch.ones(self.n_obj_sample), self.t_all.squeeze()), dim=1)
        second_boundary_value = self.net(lhs_dataset).squeeze() - self.net(rhs_dataset).squeeze()
        second_boundary_loss = torch.mean(second_boundary_value**2)

        if return_multiple_f == True:
            return pde_loss, first_boundary_loss, second_boundary_loss
        else:
            f = pde_loss + first_boundary_loss + second_boundary_loss
            return f

    def constraint_func(self):
        """
        Discussing with Frank, we have decided to exclude the periodic boundary conditions, so there will be just 'pde'
        and the first boundary condition 'fitting'
        """

        if self.constraint_type == 'pde':
            pde_constr_input = self.pde_dataset[self.constr_row_select]
            c = self.pde(pde_constr_input)
        if self.constraint_type == 'fitting':
            constr_x = self.x_all[self.constr_row_select]
            boundary_constr_input = torch.stack((constr_x, torch.zeros(self.n_constrs)), dim=1)
            constr_initial_values = self.initial_condition(constr_x.squeeze())
            c = self.net(boundary_constr_input) - constr_initial_values
        return c

device = 'cpu'
problem = Burgers(device)
print(problem.objective_func())
print(problem.constraint_func())