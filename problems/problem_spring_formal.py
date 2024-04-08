from nn_architecture import *
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from .problem_base_formal import BaseProblemFormal

"""
## Problem Statement

Follow https://github.com/benmoseley/harmonic-oscillator-pinn/blob/main/Harmonic%20oscillator%20PINN.ipynb
"""

def oscillator(d, w0, x):
    """Defines the analytical solution to the 1D underdamped harmonic oscillator problem. 
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

class SpringFormal(BaseProblemFormal):
    name = "SpringFormal"
    # Sprint parameters
    d = 2
    w0 = 20
    mu, k = 2*d, w0**2
    
    def __init__(self, device, conf):
        '''
        Input: 
            constraint_type: str, only 'pde' or 'fitting'
        '''
        
        self.conf = conf
        
        # Initialize NN
        self.net = eval(self.conf['nn_name'])(self.conf['nn_input'], self.conf['nn_output'], self.conf['nn_parameters']['n_hidden'],  self.conf['nn_parameters']['n_layers'])
        
        self.net.to(device)
        
        self.n_parameters = self.count_parameters(self.net)
        
        self.regs = self.conf['regs']
        
        self.constraint_type = self.conf['constraint_type']
        
        self.n_constrs = self.conf['n_constrs']

        self.t_fitting, self.u_fitting, self.t_pde, self.constr_row_select, self.t_test, self.u_test = self.generate_sample()
        
        self.mse_cost_function = torch.nn.MSELoss() # Mean squared error

    def generate_sample(self):

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
        dt  = torch.autograd.grad(u_pde_pred, t_pde, torch.ones_like(u_pde_pred), create_graph=True)[0]# computes dy/dx
        dt2 = torch.autograd.grad(dt,  t_pde, torch.ones_like(dt),  create_graph=True)[0]# computes d^2y/dx^2
        pde_residual = dt2 + self.mu * dt + self.k * u_pde_pred# computes the residual of the 1D harmonic oscillator differential equation
        return pde_residual

    def objective_func(self):
        """
        Compute objective function value and gradient value
        Output: 
            objective function value and gradient value
        Arguments:
            optimizer: the optimizer object 
        """

        # fitting loss
        u_fitting_pred = self.net(self.t_fitting)
        fitting_loss = torch.mean((u_fitting_pred- self.u_fitting)**2)

        # pde loss
        u_pde_pred = self.net(self.t_pde)
        pde_residual = self.pde(u_pde_pred, self.t_pde)
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
        """
        Compute constraint function value and Jacobian value
        Output: 
            constraint function value and Jacobian value
        Arguments:
            optimizer: the optimizer object 
        """

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
    
    def plot_result(self,epoch, t,u_true, u_pred, t_fitting,u_fitting,t_pde=None, save_file=None):
        "Pretty plot training results"
        t = t.cpu()
        u_true = u_true.cpu()
        u_pred = u_pred.cpu()
        t_fitting = t_fitting.cpu()
        u_fitting = u_fitting.cpu()
        t_pde = t_pde.cpu()
        plt.figure(figsize=(10,4))
        plt.plot(t,u_true, color="grey", linewidth=2, alpha=0.8, label="Exact solution")
        plt.plot(t,u_pred, color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
        plt.scatter(t_fitting, u_fitting, s=60, color="tab:orange", alpha=0.4, label='Training data')
        if t_pde is not None:
            plt.scatter(t_pde, -0*torch.ones_like(t_pde), s=60, color="tab:green", alpha=0.4, 
                        label='Physics loss training locations')
        #l = plt.legend(loc=(1.01,0.34), frameon=False, fontsize="large")
        l = plt.legend(loc=(0.7,0), frameon=False, fontsize="small")
        plt.setp(l.get_texts(), color="k")
        plt.xlim(-0.05, 1.05)
        plt.ylim(-1.1, 1.1)
        #plt.text(1.065,0.7,"Training step: %i"%(epoch+1),fontsize="xx-large",color="k")
        plt.text(0.72,-0.5,"Training step: %i"%(epoch),fontsize="medium",color="k")
        #plt.axis("off")
        if save_file is not None:
            plt.savefig(save_file)

    def save_gif_PIL(self, outfile, files, fps=5, loop=0):
        "Helper function for saving GIFs"
        imgs = [Image.open(file) for file in files]
        imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)
