from nn_architecture import *
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from problems.problem_base_formal import BaseProblemFormal
import random

""" Problem statement Burgers' equation"""

class Burgers(BaseProblemFormal):
    name = 'Burgers'
    def __init__(self, device, conf):

        self.conf = conf

        # Initialize NN
        self.net = eval(self.conf['nn_name'])(self.conf['nn_input'], self.conf['nn_output'],
                                              self.conf['nn_parameters']['n_hidden'],
                                              self.conf['nn_parameters']['n_layers'])
        self.net.to(device)

        self.n_parameters = self.count_parameters(self.net)

        self.regs = self.conf['regs']

        self.constraint_type = self.conf['constraint_type']

        self.n_constrs = self.conf['n_constrs']

        # Generate sample
        self.n_obj_sample = (self.conf['n_train_obj_samples_per_group']['pde'] +
                             self.conf['n_train_obj_samples_per_group']['boundary'] +
                             self.conf['n_train_obj_samples_per_group']['fitting'])

        self.n_constrs = self.conf['n_constrs']
        self.pde_dataset, self.t_all, self.x_all, self.constr_row_select = self.generate_sample()
        self.mse_cost_function = torch.nn.MSELoss()  # Mean squared error

    def generate_sample(self):
        # all possible couples for the pde of the objective constraints
        constr_row_select = np.random.randint(0, self.n_obj_sample, size=self.n_constrs)

        x_all = torch.linspace(0, 1, self.n_obj_sample).view(-1, 1)
        t_all = torch.linspace(0, 1, self.n_obj_sample).view(-1, 1)
        #t_all = torch.ones_like(x_all)

        pde_dataset = np.column_stack((np.meshgrid(x_all, t_all)[0].flatten(),
                                               np.meshgrid(x_all, t_all)[1].flatten()))

        if self.n_obj_sample == 1:
            return torch.Tensor(pde_dataset), t_all, x_all, constr_row_select

        perm_indices = torch.randperm(pde_dataset.shape[0])
        pde_dataset = torch.Tensor(pde_dataset[perm_indices][:self.n_obj_sample])

        return pde_dataset, t_all, x_all, constr_row_select

    def initial_condition(self, x):
        # smoout Gaussian condition
        #A = 1.0  # Amplitude
        #x0 = 0  # Center of the Gaussian
        #sigma = 0.1  # Standard deviation

        # Initial condition
        #u_initial = A * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

        #sinusoidal initial condition
        #Parameters
        N = 2

        array = x.detach().numpy()
        result = np.zeros_like(array)
        for wave_index in range(N):
            amp = random.uniform(0,1)
            phase = random.uniform(0, 2*np.pi)
            freq = random.uniform(1, 4/np.pi)

            result += amp * np.sin(2 * np.pi * freq * array + phase)
        result = torch.Tensor(result)
        return result

    def pde(self, input_dataset):
        pde_x = input_dataset[:,0].requires_grad_(True)
        pde_t = input_dataset[:,1].requires_grad_(True)

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
        pde_function = output_dt + torch.mul(output.squeeze(), output_dx) - 0.1*output_second_dx
        return pde_function

    def objective_func(self, return_multiple_f = False):
        #pde objective function
        pde_function = self.pde(self.pde_dataset)
        pde_loss = self.mse_cost_function(pde_function, torch.zeros_like(pde_function))

        #initial_condition
        initial_values = self.initial_condition(self.x_all.squeeze())
        first_boundary_input = torch.stack((self.x_all.squeeze(dim = 1), torch.zeros(self.n_obj_sample)), dim=1)
        first_boundary_value = self.net(first_boundary_input).squeeze()
        first_boundary_loss = self.mse_cost_function(first_boundary_value, initial_values)

        #periodic_boundary
        ### first we generate the (0,t) (1,t) couples
        lhs_dataset = torch.stack((torch.zeros(self.n_obj_sample), self.t_all.squeeze(dim =1)), dim=1)
        rhs_dataset = torch.stack((torch.ones(self.n_obj_sample), self.t_all.squeeze(dim = 1)), dim=1)
        second_boundary_value = self.net(lhs_dataset).squeeze() - self.net(rhs_dataset).squeeze()
        second_boundary_loss = self.mse_cost_function(second_boundary_value, torch.zeros_like(second_boundary_value))

        boundary_loss = first_boundary_loss + second_boundary_loss

        # boundary loss
        fitting_loss = torch.tensor(0)

        fs = {
            'pde': pde_loss,
            'boundary': boundary_loss,
            'fitting': fitting_loss
        }
        return fs

    def constraint_func(self):
        """
        Discussing with Frank, we have decided to exclude the periodic boundary conditions, so there will be just 'pde'
        and the first boundary condition 'fitting'
        """

        if self.constraint_type == 'pde':
            pde_constr_input = self.pde_dataset[self.constr_row_select]
            c = self.pde(pde_constr_input)

        if self.constraint_type == 'boundary': #AKA initial_condition
            constr_x = self.x_all[self.constr_row_select].squeeze(dim =1)
            boundary_constr_input = torch.stack((constr_x, torch.zeros(self.n_constrs)), dim=1)
            constr_initial_values = self.initial_condition(constr_x)
            c = self.net(boundary_constr_input).squeeze() - constr_initial_values
        return c

