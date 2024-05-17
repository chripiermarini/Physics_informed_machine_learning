from nn_architecture import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from .problem_base import BaseProblem

"""
## Problem Statement

Follow https://github.com/benmoseley/harmonic-oscillator-pinn/blob/main/Harmonic%20oscillator%20PINN.ipynb
"""

def oscillator(d, w0, x):

    """This function defines the analytical solution to the 1D underdamped harmonic oscillator problem.
    Equations taken from: https://beltoforion.de/en/harmonic_oscillator/"""

    assert d < w0
    w = np.sqrt(w0**2-d**2)
    phi = np.arctan(-d/w)
    A = 1/(2*np.cos(phi))
    cos = torch.cos(phi+w*x)
    sin = torch.sin(phi+w*x)
    exp = torch.exp(-d*x)
    y  = exp*2*A*cos
    return y

class Spring(BaseProblem):

    """ It creates an instance of the Spring problem. The friction coefficient and the spring constant
    values are fixed as a class attribute."""

    name = "Spring"
    # Spring parameters
    mu = 4
    k = 400
    
    def __init__(self, device, conf):

        """ The problem and the neural network are initialized according to the settings described in the
        conf_spring.yaml file. The training and test datasets are generated once, when the problem is initialized.
        """
        
        self.conf = conf
        
        # Initialize NN
        self.net = eval(self.conf['nn_name'])(self.conf['nn_input'], self.conf['nn_output'], self.conf['nn_parameters']['n_hidden'],  self.conf['nn_parameters']['n_layers'])
        
        self.net.to(device)
        
        self.n_parameters = self.count_parameters(self.net)
        
        self.regs = self.conf['regs']
        
        self.constraint_type = self.conf['constraint_type']
        
        self.n_constrs = self.conf['n_constrs']

        self.t_fitting, self.u_fitting, self.t_pde, self.constr_row_select, self.t_test, self.u_test = self.generate_sample()
        
        self.mse_cost_function = torch.nn.MSELoss()

    def generate_sample(self):

        """It generates the training and test dataset. The 'constr_row-select' variable is computed to fix
        the indices of the samples selected to evaluate the problem constraints. """

        # get the analytical solution over the full domain
        t_test = torch.linspace(0, self.conf['t_max'],self.conf['t_discretization']).view(-1,1)
        u_test = oscillator(self.d, self.w0, t_test).view(-1,1)

        # slice out a small number of points from the LHS of the domain
        fitting_dist = int(np.ceil(self.conf['fitting_area_percent'] * self.conf['t_discretization'] / self.conf['n_train_obj_samples_per_group']['fitting']))
        fitting_range = fitting_dist * self.conf['n_train_obj_samples_per_group']['fitting']
        t_fitting = t_test[0:fitting_range:fitting_dist]
        u_fitting = u_test[0:fitting_range:fitting_dist]

        t_pde = torch.linspace(0, self.conf['t_max'],self.conf['n_train_obj_samples_per_group']['pde']).view(-1,1).requires_grad_(True)# sample locations over the problem domain
        
        # Generate sample for constraints
        # If constraints are some points on pde conditions, then the subsample are from domain_interior
        if self.constraint_type == 'pde':
            n_t_constr = t_pde.shape[0]
        elif self.constraint_type == 'fitting':
            n_t_constr = t_fitting.shape[0]
        constr_row_select = []
        for i in range(self.n_constrs):
            constr_idx = int(n_t_constr * self.conf['constrs_area_percent'][i])
            constr_row_select.append(constr_idx)
        
        return t_fitting, u_fitting, t_pde, constr_row_select, t_test, u_test
    
    def pde(self, u_pde_pred, t_pde):

        """ It computes the right-hand side of the harmonic oscillator PDE."""

        dt  = torch.autograd.grad(u_pde_pred, t_pde, torch.ones_like(u_pde_pred), create_graph=True)[0]# computes dy/dx
        dt2 = torch.autograd.grad(dt,  t_pde, torch.ones_like(dt),  create_graph=True)[0]# computes d^2y/dx^2
        pde_residual = dt2 + self.mu * dt + self.k * u_pde_pred# computes the residual of the 1D harmonic oscillator differential equation
        return pde_residual

    def objective_func(self):

        """ This function computes the neural network predictions, and evaluates the objective function value,
         split into residual PDE, boundary and fitting terms."""

        if self.conf['batch_size'] == 'full':
            batch_idx = torch.arange(self.t_pde.size(0))
        else:
            batch_idx = torch.randperm(self.t_pde.size(0))[:int(self.t_pde.size(0)*self.conf['batch_size'])]

        # fitting loss
        u_fitting_pred = self.net(self.t_fitting)
        fitting_loss = torch.mean((u_fitting_pred- self.u_fitting)**2)

        # pde loss
        t_pde_batch = self.t_pde[batch_idx]
        u_pde_pred = self.net(t_pde_batch)
        pde_residual = self.pde(u_pde_pred, t_pde_batch)
        pde_loss = torch.mean((pde_residual)**2)

        # boundary loss
        boundary_loss = torch.tensor(0)
        
        fs = {
            'pde': pde_loss,
            'boundary': boundary_loss,
            'fitting': fitting_loss
        }
        return fs

    
    def constraint_func(self):

        """The constraint function values are computed, by using the data points sampled using the constr_row_select
        variable."""

        if self.constraint_type == 'pde':
            t_pde_constr = self.t_pde[self.constr_row_select]
            u_pde_constr_pred = self.net(t_pde_constr)
            c = self.pde(u_pde_constr_pred, t_pde_constr)
        elif self.constraint_type == 'fitting':
            t_fitting_constr = self.t_fitting[self.constr_row_select]
            u_fitting_constr = self.u_fitting[self.constr_row_select]
            u_fitting_constr_pred = self.net(t_fitting_constr)
            c = u_fitting_constr_pred - u_fitting_constr 

        return c
    
    def plot_prediction(self,save_path, epoch):

        """ This function plots the predictions generated by the neural network on test data."""
        
        u_pred = self.net(self.t_test).detach().cpu()
        t = self.t_test.cpu()
        u_true = self.u_test.cpu()
        t_fitting = self.t_fitting.cpu()
        u_fitting = self.u_fitting.cpu()
        t_pde = self.t_pde.cpu().detach()
        plt.figure(figsize=self.figsize_rectangle)
        plt.plot(t,u_true, color="grey", linewidth=2, alpha=0.8, label="Exact solution")
        plt.plot(t,u_pred, color="tab:blue", linewidth=3, alpha=0.8, label="Neural network prediction")
        plt.scatter(t_fitting, u_fitting, s=30, color="tab:orange", alpha=0.7, label='Training data')
        if t_pde is not None:
            plt.scatter(t_pde, -0*torch.ones_like(t_pde), s=30, color="tab:green", alpha=0.7, 
                        label='Physics loss training locations')
        plt.xlim(-0.02, 1.02)
        plt.ylim(-0.9, 1.2)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.title("Epoch: %i"%(epoch))
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
        plt.close("all")

