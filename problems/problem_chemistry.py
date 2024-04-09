from nn_architecture import *
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.integrate import odeint
from problems.problem_base_formal import BaseProblemFormal
import pandas as pd
import random
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)

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

        self.rates = torch.Tensor([1, 0.5, 1, 1]).to(DEVICE)

        self.n_discretization = self.conf['t_discretization']
        self.t_max = self.conf['t_max']
        self.n_initial_conditions = self.conf['n_initial_conditions']

        self.t, self.initial_y, self.label_y, self.constr_row_select= self.generate_training_data()
        self.test_t, self.test_t_tensor, self.test_label_y, self.test_initial_y = self.generate_test_data()
        
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

    def create_dataset(self, number_of_initial_cond, n_discretization = 100, name = 'train'):
        ''' When called, this function creates new data to be stored in the .txt files
        It generates samples of the t ranging from 0 to 10 divide by the number of objective values constraints
        t until 10, noise_level is 1
        '''
        random.seed(1776526)
        np.random.seed(1776526)
        if name == 'test':
          random.seed(1776)
          np.random.seed(1776)

        time_range = self.t_max
        time_discretization = time_range / n_discretization
        time_span = np.arange(0, time_range, time_discretization)

    # np.repeat(time_span, n_obj_sample, axis = 0)

        noise_level = 1
        x_init = [np.random.uniform(-noise_level, noise_level, size=(4,)) for _ in range(number_of_initial_cond)]
        for e in range(number_of_initial_cond):
            x_init[e][0] = x_init[e][0] + 14.5467204
            x_init[e][1] = x_init[e][1] + 16.3351668
            x_init[e][2] = x_init[e][2] + 25.9473091
            x_init[e][3] = x_init[e][3] + 23.5250934

        def kinetic_kosir_gen(x, t) -> np.ndarray:
    # A -> 2B; A <-> C; A <-> D
            rates = [8.566 / 2, 1.191, 5.743, 10.219, 1.535]
            return np.array([-(rates[0] + rates[1] + rates[3]) * x[0] + rates[2] * x[2] + rates[4] * x[3],
                     2 * rates[0] * x[0],
                     rates[1] * x[0] - rates[2] * x[2],
                     rates[3] * x[0] - rates[4] * x[3]])

    # Evaluate solution for each experiment
        solution_list = []
        for index, element in enumerate(x_init):
            solution = odeint(func=kinetic_kosir_gen, y0=element, t=time_span)
            solution_list.append(solution)

        final_solution = np.vstack(solution_list)
        time_span = np.tile(time_span, number_of_initial_cond).flatten()[:, np.newaxis]
        x_init = np.repeat(x_init, n_discretization, axis=0)
        final_training_dataset = pd.DataFrame(np.concatenate((time_span, x_init, final_solution), axis=1))

        file_path = f'/content/drive/MyDrive/SQPPIML/chemistry_data_folder/{name}_noise_1.txt'
        final_training_dataset.to_csv(file_path, sep = ',', index = False, header= False)
        return

    def generate_training_data(self):
        """ This function reads the .txt files and extract data to feed the objective function
        Remember that y has dimension 4""" 
        self.create_dataset(number_of_initial_cond = self.n_initial_conditions, 
                            n_discretization = self.n_discretization, 
                            name='train')

        final_training_dataset = pd.read_csv('/content/drive/MyDrive/SQPPIML/chemistry_data_folder/train_noise_1.txt', sep=',', header=None)
        
        t = torch.Tensor(final_training_dataset.iloc[:, 0].values).unsqueeze(1).to(DEVICE).requires_grad_(True)
        y_initial = torch.Tensor(final_training_dataset.iloc[:, 1:5].values).to(DEVICE)
        y_label = torch.Tensor(final_training_dataset.iloc[:, 5:].values).to(DEVICE)
        constr_row_select = np.random.randint(0, self.n_discretization*self.n_initial_conditions, size=self.n_constrs)

        return t, y_initial, y_label, constr_row_select
    
    def generate_test_data(self):
        self.create_dataset(number_of_initial_cond= self.conf['n_test_initial_conditions'],
                            n_discretization = self.n_discretization, 
                            name='test')

        test_dataset = pd.read_csv('/content/drive/MyDrive/SQPPIML/chemistry_data_folder/test_noise_1.txt', sep=',', header=None)
        t = test_dataset.iloc[:, 0].values
        test_label_y =test_dataset.iloc[:, 5:].values

        test_initial_y = torch.Tensor(test_dataset.iloc[:, 1:5].values).to(DEVICE)
        test_t_tensor = torch.Tensor(t).unsqueeze(1).to(DEVICE)
        return t, test_t_tensor, test_label_y, test_initial_y 

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
        # we select randomly for computing the fitting_loss
        fitting_rows_indices =  np.random.choice(output.shape[0], 
        size= int(output.shape[0]*self.conf['n_train_obj_samples_per_group']['fitting']), 
        replace=False)

        fitting_loss = self.mse_cost_function(output[fitting_rows_indices], self.label_y[fitting_rows_indices])
        
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

    def chemistry_plot(self, save_path):
      prediction = self.net(torch.cat((self.test_initial_y, self.test_t_tensor),1)).cpu().detach().numpy()
      # Create subplots
      fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

      # Plot 1
      for i in range(4):
          axs[0].plot(self.test_t, self.test_label_y[:, i], label = f'y_true_{i}') 
      axs[0].set_title('Test Label')  # Set subplot title
      axs[0].set_xlabel('t')  # Set xlabel
      axs[0].legend()  # Show legend

      # Plot 2
      for i in range(4):
          axs[1].plot(self.test_t, prediction[:, i], label = f'y_prediction_{i}')  # Example scatter plot
      axs[1].set_title('Prediction')  # Set subplot title
      axs[1].set_xlabel('t')  # Set xlabel
      axs[1].legend()  # Show legend

      plt.tight_layout()  # Adjust layout
      plt.savefig(save_path)
      return
      


