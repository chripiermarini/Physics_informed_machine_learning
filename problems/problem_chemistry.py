from nn_architecture import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from problems.problem_base import BaseProblem
import random
import os

class Chemistry(BaseProblem):
    """ It creates an instance of the Chemical engineering problem. The noise_level of the generated data
    is fixed as a class attribute."""

    name = 'Chemistry'
    noise_level = 1
    
    def __init__(self, device, conf):

        """ The problem and the neural network are initialized according to the settings described in the
        conf_chemistry.yaml file. If a dataset is not present, the code will create one with the
         features indicated below (rates, pde_rates) and the noise_level set as a class attribute.
          If a dataset is available, the code will simply load it to perform training."""

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

        self.rates = torch.tensor([1, 0.5, 1, 1]).to(device)
        self.pde_rates = torch.tensor([8.566 / 2, 1.191, 5.743, 10.219, 1.535]).to(device)

        self.n_discretization = self.conf['t_discretization']
        self.t_max = self.conf['t_max']

        self.train_sample_size = self.n_discretization * self.conf['n_initial_conditions']

        self.train, self.constr_row_select, self.test = self.generate_sample(device)
        
        self.n_fitting_sample = int(self.train_sample_size * self.conf['fitting_sample_percentage'])
        
        self.fitting_sample_indices = torch.randperm(self.train_sample_size)[:self.n_fitting_sample]
        
        self.mse_cost_function = torch.nn.MSELoss()  # Mean squared error

    def create_dataset(self, number_of_initial_cond, n_discretization = 100, save_name = None):

        ''' It creates a dataset if one is not available. The time range is discretized, and realistic
        x samples are created using the 'kinetic_kosir_gen' function.'''

        random.seed(number_of_initial_cond)
        np.random.seed(number_of_initial_cond)

        time_range = self.t_max
        time_discretization = time_range / n_discretization
        time_span = np.arange(0, time_range, time_discretization)

        x_init = [np.random.uniform(-self.noise_level, self.noise_level, size=(4,)) for _ in range(number_of_initial_cond)]
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
        samples = np.zeros((n_discretization * number_of_initial_cond, 9))
        for index, element in enumerate(x_init):
            solution = odeint(func=kinetic_kosir_gen, y0=element, t=time_span)
            samples[index * n_discretization : (index+1) * n_discretization] = np.concatenate((np.expand_dims(time_span,axis=1), np.repeat(np.expand_dims(element, axis=0), n_discretization, axis=0), solution), axis=1)

        #np.savetxt(save_name, samples, delimiter=',') 
        return samples.astype('float32')
    
    def generate_sample(self,device):

        ''' The training and test datasets are created ex-novo or read from an existing file.
        The 'constr_row-select' variable is computed to fix the indices of the samples selected
        to evaluate the problem constraints.'''

        # Generate training data
        if not os.path.exists(self.conf['train_file_path']):
            train_sample = self.create_dataset(number_of_initial_cond =self.conf['n_initial_conditions'], 
                                n_discretization = self.n_discretization, 
                                save_name=self.conf['train_file_path'])
        else:
            train_sample = np.loadtxt(self.conf['train_file_path'],delimiter=',')
 
        train = {
            't':torch.tensor(train_sample[:,0:1]).to(device).requires_grad_(True),
            'y_initial': torch.tensor(train_sample[:,1:5]).to(device),
            'y_label': torch.tensor(train_sample[:,5:]).to(device)
        }
        
        constr_row_select = torch.randperm(self.train_sample_size)[:self.n_constrs]
        
        # Generate testing data
        if not os.path.exists(self.conf['test_file_path']):
            test_sample = self.create_dataset(number_of_initial_cond = self.conf['n_test_initial_conditions'], 
                                n_discretization = self.n_discretization, 
                                save_name=self.conf['test_file_path'])
        else:
            test_sample = np.loadtxt(self.conf['test_file_path'],delimiter=',')
        test = {
            't':torch.tensor(test_sample[:,0:1]).to(device),
            'y_initial': torch.tensor(test_sample[:,1:5]).to(device),
            'y_label': torch.tensor(test_sample[:,5:]).to(device)
        }
        return train, constr_row_select, test
    
    def pde(self, output, t):

        """ This function computes the complex right hand-side of the kinetic_kosir PDE."""

        rhs = torch.zeros_like(output)
        rhs[:, 0] = (-(self.pde_rates[0] + self.pde_rates[1] + self.pde_rates[3]) * output[:,0] +
                     self.pde_rates[2] * output[:, 2] + self.pde_rates[4] * output[:,3])
        rhs[:, 1] = 2 * self.pde_rates[0] * output[:, 0]
        rhs[:, 2] = self.pde_rates[1] * output[:, 0] - self.pde_rates[2] * output[:,2]
        rhs[:, 3] = self.pde_rates[3] * output[:, 0] - self.pde_rates[4] * output[:,3]

        dt = torch.zeros(output.size())
        
        for i in range(output.size(1)):
            dt[:,i:(i+1)] = torch.autograd.grad(outputs=output[:,i].sum(),
                                        inputs=t,
                                        create_graph=True,
                                        allow_unused=True
                                        )[0]
        pde = dt - rhs
        return pde, dt

    def objective_func(self):

        """ It computes the neural network predictions, and evaluates the objective function value,
         split into residual PDE, mass-balance and fitting terms."""

        fitting_idx = self.fitting_sample_indices
        if self.conf['batch_size'] == 'full':
            batch_idx = torch.arange(self.train_sample_size)
        else:
            batch_idx = torch.randperm(self.train['t'].size(0))[:int(self.train['t'].size(0)*self.conf['batch_size'])]
        
        y_initial_pde = self.train['y_initial'][batch_idx]
        t_pde =  self.train['t'][batch_idx]
        output = self.net(torch.cat((y_initial_pde,t_pde), 1))
        pde_values, dt = self.pde(output, t_pde)
        
        # PDE residual term
        pde_loss = self.mse_cost_function(pde_values, torch.zeros_like(pde_values))

        # mass-balance term
        rate_product_values = torch.matmul(dt, self.rates)
        boundary_loss = self.mse_cost_function(rate_product_values, torch.zeros_like(rate_product_values))
        
        # data fitting term
        output_fitting = self.net(torch.cat((self.train['y_initial'][fitting_idx], self.train['t'][fitting_idx]), 1))
        fitting_loss = self.mse_cost_function(output_fitting, self.train['y_label'][fitting_idx])

        fs = {
            'pde': pde_loss,
            'boundary': boundary_loss,
            'fitting': fitting_loss
        }
        return fs

    def constraint_func(self):

        """The constraint function values are computed."""

        assert self.constraint_type == 'other'
        
        y_initial_constraint_input = self.train['y_initial'][self.constr_row_select]
        t_constraint_input = self.train['t'][self.constr_row_select]
        output_constraint = self.net(torch.cat((y_initial_constraint_input, t_constraint_input), 1))

        _, constraint_dt = self.pde(output_constraint, t_constraint_input)

        c = torch.matmul(constraint_dt, self.rates)

        return c

    def plot_prediction(self, save_path = None, epoch = None, save_label = False):

        """ This function plots the predictions generated by the neural network on test data."""

        # Plot label
        y_max = torch.max(self.test['y_label']).cpu()+5
        y_min = torch.min(self.test['y_label']).cpu()-5
        fig = plt.figure(figsize = self.figsize)
        ax = fig.add_subplot(111)
        if save_label == True:
            for i in range(4):
                ax.plot(self.test['t'].cpu(), self.test['y_label'][:, i].cpu(), label = f'True $u_{i}$') 
            title = 'True solution'
        else:
            prediction = self.net(torch.cat((self.test['y_initial'], self.test['t']),1)).cpu().detach().numpy() 
            for i in range(4):
                ax.plot(self.test['t'].cpu(), prediction[:, i], label = f'Predicted $u_{i}$')  # Example scatter plot
            if epoch != None:
                title = f'Prediction - Epoch {epoch}'
            else:
                title = 'Prediction'

        ax.set_ylim(ymin=y_min,ymax=y_max)
        plt.xticks([], [])
        plt.yticks([], [])
        #ax.legend(loc='upper left')  
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(save_path, format = 'png')
        plt.close()
      


