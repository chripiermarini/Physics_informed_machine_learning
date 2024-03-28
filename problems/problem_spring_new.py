from nn_architecture import *
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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

class SpringNew:
    name = "SpringNew"
    t_ub = 1
    n_discretization = 100
    d = 2
    w0 = 20
    mu, k = 2*d, w0**2
    def __init__(self, device, n_obj_sample = 1, n_constrs = 3, reg = 1, constraint_type='pde'):

        '''
        Input: 
            constraint_type: str, only 'pde' or 'fitting'
        '''
        # Initialize NN
        self.n_input = 1
        self.n_output = 1
        #self.net = OneHiddenLayerFCNN(self.n_input, self.n_output, n_neurons = 16) 
        self.net = FCN(self.n_input, self.n_output, 32,3)
        #self.net = TwoHiddenLayerFCNN(self.n_input, self.n_output,n_neurons = 64) 
        self.net.to(device)
        self.n_parameters = self.count_parameters(self.net)
        self.reg = torch.tensor(reg)
        self.constraint_type = constraint_type

        # Generate sample
        assert(n_obj_sample == 1)
        self.n_obj_sample = n_obj_sample
        self.n_constrs = n_constrs
        self.t_all, self.u_all, self.t_fitting, self.u_fitting, self.t_pde, self.constr_row_select = self.generate_sample()
        
        self.mse_cost_function = torch.nn.MSELoss() # Mean squared error

                
    def count_parameters(self, nn_net):
        return sum(p.numel() for p in nn_net.parameters() if p.requires_grad)

    def generate_sample(self):
        """
        Generate boundary points, interior points for both unconstrained and constrained case
        Be careful: when in unconstrained case, the number of samples for the objective function is
        double (S_B and S_i)

        n_obj_sample controls the number of x we have

        if constr is zero, we automatically generate 10 different t's
        If constr is not zero, we automatically generate n_constr different t's
        """

        # get the analytical solution over the full domain
        t_all = torch.linspace(0,1,500).view(-1,1)
        u_all = oscillator(self.d, self.w0, t_all).view(-1,1)

        # slice out a small number of points from the LHS of the domain
        t_fitting = t_all[0:200:20]
        u_fitting = u_all[0:200:20]

        t_pde = torch.linspace(0,1,30).view(-1,1).requires_grad_(True)# sample locations over the problem domain
        
        # Generate sample for constraints
        # If constraints are some points on pde conditions, then the subsample are from domain_interior
        if self.constraint_type == 'pde':
            n_t_pde = t_pde.shape[0]
            constr_row_select = np.random.randint(0, n_t_pde, size=(self.n_constrs))
        elif self.constraint_type == 'fitting':
            n_t_fitting = t_fitting.shape[0]
            constr_row_select = np.random.randint(0, n_t_fitting, size=(self.n_constrs))
        
        return t_all, u_all, t_fitting, u_fitting, t_pde, constr_row_select
    
    def pde(self, u_pde_pred, t_pde):
        dt  = torch.autograd.grad(u_pde_pred, t_pde, torch.ones_like(u_pde_pred), create_graph=True)[0]# computes dy/dx
        dt2 = torch.autograd.grad(dt,  t_pde, torch.ones_like(dt),  create_graph=True)[0]# computes d^2y/dx^2
        pde_residual = dt2 + self.mu * dt + self.k * u_pde_pred# computes the residual of the 1D harmonic oscillator differential equation
        return pde_residual

    def objective_func_and_grad(self, optimizer, no_grad = False, return_multiple_f=False):
        """
        Compute objective function value and gradient value
        Output: 
            objective function value and gradient value
        Arguments:
            optimizer: the optimizer object 
        """
        optimizer.zero_grad()

        u_fitting_pred = self.net(self.t_fitting)
        fitting_loss = torch.mean((u_fitting_pred- self.u_fitting)**2)

        u_pde_pred = self.net(self.t_pde)
        pde_residual = self.pde(u_pde_pred, self.t_pde)
        pde_loss = torch.mean((pde_residual)**2)

        ##compute objective function
        f = fitting_loss + self.reg * pde_loss

        # Compute objective value
        f_value = f.data

        # Backward of objective function
        
        f.backward()# (retain_graph=True)

        if no_grad is True:
            if return_multiple_f:
                return f_value, pde_loss.data, fitting_loss.data
            else:
                return f_value
        else:
            # Assign derivative to gradient value
            g_value = torch.zeros(self.n_parameters)
            i = 0
            for name, param in self.net.named_parameters():
                grad_l = len(param.grad.view(-1))
                g_value[i:i + grad_l] = param.grad.view(-1)
                i += grad_l
            if return_multiple_f:
                return f_value, pde_loss.data, fitting_loss.data, g_value
            else:
                return f_value, g_value

    
    def constraint_func_and_grad(self, optimizer, no_grad = False):
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

        c_value = c.data
        c_value = c_value.reshape(-1)

        if no_grad is True:
            return c_value
        else:
            # Compute Jacobian
            J_value = torch.zeros(self.n_constrs, self.n_parameters)
            for i in range(self.n_constrs):
                optimizer.zero_grad()

                # Backward of each constraint function
                c[i].backward(retain_graph=True)
                grads = torch.Tensor()  # dict()
                for name, param in self.net.named_parameters():
                    if param.grad is not None:
                        grads = torch.cat((grads, param.grad.view(-1)), 0)
                    else:
                        grads = torch.cat((grads, torch.zeros(param.view(-1).shape)), 0)
                J_value[i, :] = grads

            return c_value, J_value
        
        
    
    def save_net(self,path):
        torch.save(self.net.state_dict(), path)

    def load_net(self,path):
        self.net.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
        self.net.eval()
        

    
    def plot_result(self,epoch, t,u_true, u_pred, t_fitting,u_fitting,t_pde=None, save_file=None):
        "Pretty plot training results"
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
