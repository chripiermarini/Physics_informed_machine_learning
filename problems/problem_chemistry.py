from nn_architecture import *
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.integrate import odeint
from problems.problem_base_formal import BaseProblemFormal

""" Problem statement Chemistry equation"""

def tensor_from_file(file_path, final_row):
    """ Simple function that loads data from the .txt files and returns a tensor"""
    loaded_data = []
    with open(file_path, 'r') as file:
        i = 0
        for line in file:
            # Split line based on the delimiter and convert elements to integers
            elements = [float(elem) for elem in line.strip().split(',')]
            loaded_data.append(elements)
            i = i +1
            if i == final_row:
                break
    # Convert loaded data to tensor
    loaded_tensor = torch.tensor(loaded_data)
    return loaded_tensor

class Chemistry(BaseProblemFormal):
    name = 'Chemistry'

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

        self.rates = torch.Tensor([1, 0.5, 1, 1])

        self.t, self.initial_y, self.label_y, self.constr_row_select= self.generate_sample()
        self.mse_cost_function = torch.nn.MSELoss()  # Mean squared error

    def kinetic_kosir(self, y, t):
        """ This function computes the complex right hand-side of the kinetic_kosir PDE"""
        # A -> 2B; A <-> C; A <-> D
        rates = torch.tensor([8.566 / 2, 1.191, 5.743, 10.219, 1.535])
        tensor = torch.tensor([
            -(rates[0] + rates[1] + rates[3]) * y[0] + rates[2] * y[2] + rates[4] * y[3],
            2 * rates[0] * y[0],
            rates[1] * y[0] - rates[2] * y[2],
            rates[3] * y[0] - rates[4] * y[3]
        ])
        return tensor

    def create_dataset(self):
        ''' When called, this function creates new data to be stored in the .txt files
        It generates samples of the t ranging from 0 to 10 divide by the number of objective values constraints
        t until 10, noise_level is 1
        '''
        time_range = 10
        time_discretization = 10 / self.n_obj_sample
        time_span = np.arange(0, time_range, time_discretization)
        t = torch.Tensor(time_span).unsqueeze(1)

        file_path = 'problems/srinivas_input_t.txt'
        np.savetxt(file_path, t.numpy(), delimiter=',')
        print(1)

        noise_level = 1
        x_init = [np.random.uniform(-noise_level, noise_level, size=(4,)) for _ in range(self.n_obj_sample)]
        for e in range(self.n_obj_sample):
            x_init[e][0] = x_init[e][0] + 14.5467204
            x_init[e][1] = x_init[e][1] + 16.3351668
            x_init[e][2] = x_init[e][2] + 25.9473091
            x_init[e][3] = x_init[e][3] + 23.5250934
        y_initial = torch.Tensor(np.array(x_init))

        file_path = 'problems/srinivas_input_y_initial.txt'
        np.savetxt(file_path, y_initial.numpy(), delimiter=',')
        print(2)

        # Evaluate solution for each experiment
        solution = [odeint(func=self.kinetic_kosir, y0=xi, t=time_span) for xi in x_init]
        y_label = torch.Tensor(solution[0])
        file_path = 'problems/srinivas_input_y_label.txt'
        np.savetxt(file_path, y_label.numpy(), delimiter=',')
        print(3)
        return

    def generate_sample(self):
        """ This function reads the .txt files and extract data to feed the objective function
        Remember that y has dimension 4"""
        t = tensor_from_file('problems/srinivas_input_t.txt', self.n_obj_sample).requires_grad_(True)
        y_initial = tensor_from_file('problems/srinivas_input_y_initial.txt', self.n_obj_sample)
        y_label = tensor_from_file('problems/srinivas_input_y_label.txt', self.n_obj_sample)
        constr_row_select = np.random.randint(0, self.n_obj_sample, size=self.n_constrs)

        return t, y_initial, y_label, constr_row_select

    def pde(self, output, t):
        dt = torch.empty(output.size(0), 0)
        for i in range(output.size(1)):
            column_dt = torch.autograd.grad(outputs=output[:,i].sum(),
                                     inputs=t,
                                     create_graph=True,
                                     allow_unused=True
                                     )[0]
            dt=torch.cat((dt, column_dt), dim=1)

        rhs_pde= torch.stack([self.kinetic_kosir(row, t) for row in output])
        pde = dt - rhs_pde
        return pde,dt

    def objective_func(self, return_multiple_f = False):
        # PDE objective function
        output = self.net(torch.cat((self.initial_y, self.t), 1))
        pde_values, dt = self.pde(output, self.t)
        pde_loss = self.mse_cost_function(pde_values, torch.zeros_like(pde_values))

        # data fitting function
        fitting_loss = self.mse_cost_function(output, self.label_y)

        """ 
        ## 'other MSE' loss function
        rate_product_values = torch.matmul(dt, self.rates)
        rate_product_loss = self.mse_cost_function(rate_product_values, torch.zeros_like(rate_product_values))
        """

        # boundary loss
        boundary_loss = torch.tensor(0)

        fs = {
            'pde': pde_loss,
            'boundary': boundary_loss,
            'fitting': fitting_loss
        }
        return fs

    def constraint_func(self):
        assert self.constraint_type == 'other'
        if self.n_constrs == 0:
            c = torch.empty(size=(0,))
            return c
        else:
            initial_y_constraint_input = self.initial_y[self.constr_row_select]
            t_constraint_input = self.t[self.constr_row_select]
            output_constraint = self.net(torch.cat((initial_y_constraint_input, t_constraint_input), 1))
            #if self.constraint_type == 'other':
            _, constraint_dt = self.pde(output_constraint, t_constraint_input)
            c = torch.matmul(constraint_dt, self.rates)

            #if self.constraint_type == 'pde':
            #    c, _ = self.pde(output_constraint, t_constraint_input, initial_y_constraint_input)

            #if self.constraint_type == 'fitting':
            #   c = output_constraint - self.label_y[self.constr_row_select]

            return c