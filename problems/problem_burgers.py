from nn_architecture import *
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from problems.problem_base_formal import BaseProblemFormal
import random
import pandas as pd

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
        self.n_obj_sample_pde = self.conf['n_train_obj_samples_per_group']['pde']
        self.n_obj_sample_boundary = self.conf['n_train_obj_samples_per_group']['boundary']
        self.n_obj_sample_fitting = self.conf['n_train_obj_samples_per_group']['fitting']

        self.n_constrs = self.conf['n_constrs']

        (self.pde_dataset,
         self.IC_dataset,
         self.periodic_condition_dataset,
         self.fitting_dataset,
         self.constr_row_select) = self.generate_sample()

        ### we select only a subset of the whole dataset
        self.pde_dataset = self.pde_dataset.sample(frac=1).reset_index(drop=True)[:self.n_obj_sample_pde] #shuffled dataset
        self.IC_dataset = self.IC_dataset[:self.n_obj_sample_boundary]
        self.periodic_condition_dataset = self.periodic_condition_dataset[:self.n_obj_sample_boundary]
        self.fitting_dataset = self.fitting_dataset.sample(frac=1).reset_index(drop=True)[:self.n_obj_sample_fitting]

        self.mse_cost_function = torch.nn.MSELoss()  # Mean squared error

    def generate_sample(self):
        # all possible couples for the pde of the objective constraints

        constr_row_select = np.random.randint(0, self.n_obj_sample_boundary, size=self.n_constrs)
        '''
        

        x_all = torch.linspace(0, 1, self.n_obj_sample).view(-1, 1)
        t_all = torch.linspace(0, 1, self.n_obj_sample).view(-1, 1)
        #t_all = torch.ones_like(x_all)

        pde_dataset = np.column_stack((np.meshgrid(x_all, t_all)[0].flatten(),
                                               np.meshgrid(x_all, t_all)[1].flatten()))

        if self.n_obj_sample == 1:
            return torch.Tensor(pde_dataset), t_all, x_all, constr_row_select

        perm_indices = torch.randperm(pde_dataset.shape[0])
        pde_dataset = torch.Tensor(pde_dataset[perm_indices][:self.n_obj_sample])
        '''
        data_folder = f'burgers_data_folder/'
        pde_dataset = pd.read_csv(str(data_folder + 'pde_dataset.csv'), sep=',', header=0)
        IC_dataset = pd.read_csv(str(data_folder + 'IC_dataset.csv'), sep=',', header=0)
        periodic_condition_dataset = pd.read_csv(str(data_folder + 'periodic_condition_dataset.csv'), sep=',', header=0)
        fitting_dataset = pd.read_csv(str(data_folder + 'fitting_dataset.csv'), sep=',', header=0)

        return pde_dataset, IC_dataset, periodic_condition_dataset, fitting_dataset, constr_row_select

    def initial_condition(self, x):
        random.seed(1776526)
        np.random.seed(1776526)
        # smoout Gaussian condition
        # A = 1.0  # Amplitude
        # x0 = 0  # Center of the Gaussian
        # sigma = 0.1  # Standard deviation

        # Initial condition
        # u_initial = A * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

        # sinusoidal initial condition
        # Parameters
        N = 2

        array = x
        result = np.zeros_like(array)
        for wave_index in range(N):
            amp = random.uniform(0, 1)
            phase = random.uniform(0, 2 * np.pi)
            freq = random.uniform(1, 4 / np.pi)

            result += amp * np.sin(2 * np.pi * freq * array + phase)
        return result

    def pde(self, input_dataset):
        x= input_dataset['x']

        pde_x = torch.Tensor(input_dataset['x'].values).requires_grad_(True)
        pde_t = torch.Tensor(input_dataset['t'].values).requires_grad_(True)
        pde_u_zero = torch.Tensor(input_dataset['u_zero'].values)

        output = self.net(torch.stack((pde_x, pde_t, pde_u_zero), 1)).requires_grad_(True) #u(x,t) function
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

        #initial_condition_boundary
        initial_values = torch.Tensor(self.IC_dataset['u_zero_initial'])
        first_boundary_value = self.net(torch.stack((torch.Tensor(self.IC_dataset['x_initial']),
                                                     torch.Tensor(self.IC_dataset['t_initial']),
                                                     torch.Tensor(self.IC_dataset['u_zero_initial'])), 1)).squeeze()
        ic_boundary_loss = self.mse_cost_function(first_boundary_value, initial_values)

        #periodic_boundary
        ### first we generate the (0,t) (1,t) couples
        lhs_dataset = torch.stack((torch.Tensor(self.periodic_condition_dataset['x_0']),
                                   torch.Tensor(self.periodic_condition_dataset['t']),
                                   torch.Tensor(self.periodic_condition_dataset['u_zero_0_t'])), dim=1)
        rhs_dataset = torch.stack((torch.Tensor(self.periodic_condition_dataset['x_1']),
                                   torch.Tensor(self.periodic_condition_dataset['t']),
                                   torch.Tensor(self.periodic_condition_dataset['u_zero_1_t'])), dim=1)
        second_boundary_value = self.net(lhs_dataset).squeeze() - self.net(rhs_dataset).squeeze()
        periodic_boundary_loss = self.mse_cost_function(second_boundary_value, torch.zeros_like(second_boundary_value))

        boundary_loss = ic_boundary_loss + periodic_boundary_loss

        # fitting_loss
        fitting_output = self.net(torch.stack((torch.Tensor(self.fitting_dataset['fitting_x']),
                                                     torch.Tensor(self.fitting_dataset['fitting_t']),
                                                     torch.Tensor(self.fitting_dataset['fitting_u_zero'])), 1)).squeeze()
        fitting_loss = self.mse_cost_function(fitting_output, torch.Tensor(self.fitting_dataset['fitting_u_labels']))

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
            pde_constr_input = self.pde_dataset.iloc[self.constr_row_select]
            c = self.pde(pde_constr_input)

        if self.constraint_type == 'boundary': #only the initial condition boundary

            constr_x = torch.Tensor(self.IC_dataset['x_initial'].iloc[self.constr_row_select].values)
            constr_t = torch.Tensor(self.IC_dataset['t_initial'].iloc[self.constr_row_select].values)
            constr_u_zero = torch.Tensor(self.IC_dataset['u_zero_initial'].iloc[self.constr_row_select].values)
            boundary_constr_input = torch.stack((constr_x, constr_t, constr_u_zero), dim=1)
            constr_initial_values = torch.Tensor(self.IC_dataset['u_zero_initial'].iloc[self.constr_row_select].values)
            c = self.net(boundary_constr_input).squeeze() - constr_initial_values
        return c

