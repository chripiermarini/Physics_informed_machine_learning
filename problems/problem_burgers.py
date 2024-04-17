from nn_architecture import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from problems.problem_base import BaseProblem
from scipy.integrate import odeint

""" Problem statement Burgers' equation"""

class Burgers(BaseProblem):
    name = 'Burgers'
    nu = 0.01   ##kinematic viscosity coefficient
    mu = 0.5    
    
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
        
        self.x_max = self.conf['x_max']
        
        self.x_discretization = self.conf['x_discretization']
        
        self.t_max = self.conf['t_max']
        
        self.t_discretization = self.conf['t_discretization']

        self.n_obj_sample_pde = self.conf['n_train_obj_samples_per_group']['pde']
        self.n_obj_sample_boundary = self.conf['n_train_obj_samples_per_group']['boundary']
        self.n_obj_sample_fitting = self.conf['n_train_obj_samples_per_group']['fitting']
        
        # Generate sample
        self.sample, self.constr_row_select = self.generate_sample(device)
        
        self.vmax=torch.max(self.sample['U_true'])
        self.vmin=torch.min(self.sample['U_true'])
        
        self.mse_cost_function = torch.nn.MSELoss()  # Mean squared error

    def initial_function(self, xs):
        # We use the fix initial function u_0(x) = 3*sin(2pi x)
        u0 = torch.sin(2*torch.pi * xs)
        return u0
    
    def true_burgers_solution(self, xs, ts, u0):
        #Wave number discretization
        k = 2 * torch.pi * torch.fft.fftfreq( self.x_discretization, d = xs[1] - xs[0])
        
        ############## EQUATION SOLVING ###############

        #Definition of ODE system (PDE ---(FFT)---> ODE system)
        def burg_system(u,t,k,mu,nu):
            #Spatial derivative in the Fourier domain
            u_hat = np.fft.fft(u)
            u_hat_x = 1j*k*u_hat
            u_hat_xx = -k**2*u_hat
            
            #Switching in the spatial domain
            u_x = np.fft.ifft(u_hat_x)
            u_xx = np.fft.ifft(u_hat_xx)
            
            #ODE resolution
            u_t = -mu*u*u_x + nu*u_xx
            return u_t.real
        
        
        #PDE resolution (ODE system resolution)
        U = odeint(burg_system, u0, ts, args=(k,self.mu,self.nu,))
        return U
    
    
    def generate_sample(self,device):
        sample = {}
        
        # Generate x and t discretization
        xs = torch.linspace(0,self.x_max,self.x_discretization)
        ts = torch.linspace(0,self.t_max,self.t_discretization)
        
        # Compute Initial condition
        u0 = self.initial_function(xs)
        
        # compute true solution
        u_true = self.true_burgers_solution(xs, ts, u0)
        u_true = torch.tensor(u_true).to(torch.float32).to(device)
        #self.plot_u(u_true, xs, ts)
        
        # Compose all sample input only for test
        X, T = torch.meshgrid(xs, ts, indexing='xy')
        X = X.flatten()
        T = T.flatten()
        test_input = torch.stack((X,T), axis=1)
        U_true = u_true.flatten()
        
        n_total = len(X)
        sample['test_input'] = test_input
        sample['U_true'] = U_true
        sample['u0'] = u0
        
        # Fitting sample
        fitting_idx = torch.randint(low=0,high=n_total,size=(self.n_obj_sample_fitting,))
        fitting_input = test_input[fitting_idx]
        fitting_u_true = U_true[fitting_idx]
        sample['fitting_input'] = fitting_input
        sample['fitting_u_true'] = fitting_u_true
                
        # boundary condition sample
        # u(0,t) = u(1,t) for some t
        assert(np.mod(self.t_discretization, self.n_obj_sample_boundary) == 0)
        step = self.t_discretization / self.n_obj_sample_boundary
        t_pc =  torch.arange(0, self.t_discretization, step)
        pc_input_0 = torch.zeros(self.n_obj_sample_boundary, 2)
        pc_input_1 = torch.ones(self.n_obj_sample_boundary, 2)
        pc_input_0[:,1] = ts[t_pc.int()]
        pc_input_1[:,1] = ts[t_pc.int()]
        sample['pc_input_0'] = pc_input_0
        sample['pc_input_1'] = pc_input_1
        
        # u(x,0) = u0(x) for some x
        assert(np.mod(self.x_discretization, self.n_obj_sample_boundary) == 0)
        x_step = self.x_discretization / self.n_obj_sample_boundary
        x_ic =  torch.arange(0, self.x_discretization, x_step)
        ic_input = torch.zeros(self.n_obj_sample_boundary, 2)
        ic_input[:,0] = xs[x_ic.int()]
        ic_u0 = u0[x_ic.int()]
        sample['ic_input'] = ic_input
        sample['ic_u0'] = ic_u0
        
        # PDE sample
        assert(np.mod(self.x_discretization, self.n_obj_sample_pde[0]) == 0)
        assert(np.mod(self.t_discretization, self.n_obj_sample_pde[1]) == 0)
        x_step = self.x_discretization / self.n_obj_sample_pde[0]
        t_step = self.t_discretization / self.n_obj_sample_pde[1]
        x_pde = torch.arange(0, self.x_discretization, x_step)
        t_pde =  torch.arange(0, self.t_discretization, t_step)
        X_pde, T_pde = torch.meshgrid(xs[x_pde.int()], ts[t_pde.int()], indexing='xy')
        X_pde = X_pde.flatten()
        T_pde = T_pde.flatten()
        pde_input = torch.stack((X_pde,T_pde), axis=1)
        sample['pde_input'] = pde_input
        
        # all possible couples for the pde of the objective constraints
        if self.constraint_type == 'pde':
            constr_row_select =  torch.randint(low=0,high=pde_input.shape[0],size=(self.n_constrs,))
        elif self.constraint_type == 'boundary':
            # half periodic condition and half initial condition
            half = int(self.n_constrs/2)
            constr_row_select={}
            constr_row_select['pc'] =  torch.randint(low=0,high=self.n_obj_sample_boundary,size=(half,))
            constr_row_select['ic'] =  torch.randint(low=0,high=self.n_obj_sample_boundary,size=(self.n_constrs - half,))
            
        return sample, constr_row_select

    def pde(self, output, x_pde, t_pde):
        output_dt = torch.autograd.grad(outputs= output.sum(),
                                            inputs=t_pde,
                                            create_graph=True,
                                            allow_unused=True
                                            )[0]
        output_dx = torch.autograd.grad(outputs=output.sum(),
                                            inputs=x_pde,
                                            create_graph=True,
                                            allow_unused=True
                                            )[0]
        output_second_dx = torch.autograd.grad(outputs=output_dx.sum(),
                                                   inputs=x_pde,
                                                   create_graph=True,
                                                   allow_unused=True
                                                   )[0]
        pde_function = output_dt + self.mu * torch.mul(output.squeeze(), output_dx) - self.nu*output_second_dx
        return pde_function

    def objective_func(self, return_multiple_f = False):

        # fitting loss
        u_fitting_pred = self.net(self.sample['fitting_input'])
        fitting_loss = self.mse_cost_function(u_fitting_pred.squeeze(), self.sample['fitting_u_true'])

        # pde loss
        x_pde = self.sample['pde_input'][:,0].requires_grad_(True) 
        t_pde = self.sample['pde_input'][:,1].requires_grad_(True) 
        u_pde_pred = self.net(torch.stack((x_pde, t_pde), axis=1)).requires_grad_(True)
        pde_residual = self.pde(u_pde_pred, x_pde, t_pde)
        pde_loss = self.mse_cost_function(pde_residual, torch.zeros_like(pde_residual))

        # boundary loss
        ## pc loss
        pc_0_pred = self.net(self.sample['pc_input_0'])
        pc_1_pred = self.net(self.sample['pc_input_1'])
        pc_loss = self.mse_cost_function(pc_0_pred, pc_1_pred)
        
        ## ic loss
        ic_pred = self.net(self.sample['ic_input'])
        ic_loss = self.mse_cost_function(ic_pred.squeeze(), self.sample['ic_u0'])
        
        boundary_loss = pc_loss + ic_loss
        
        fs = {
            'pde': pde_loss,
            'boundary': boundary_loss,
            'fitting': fitting_loss
        }
        
        return fs

    def constraint_func(self):

        if self.constraint_type == 'pde':
            x_pde = self.sample['pde_input'][self.constr_row_select,0].requires_grad_(True) 
            t_pde = self.sample['pde_input'][self.constr_row_select,1].requires_grad_(True) 
            u_pde_pred = self.net(torch.stack((x_pde, t_pde), axis=1)).requires_grad_(True)
            c = self.pde(u_pde_pred, x_pde, t_pde)

        if self.constraint_type == 'boundary':
            # half pc and half ic
            ## pc 
            pc_0_pred = self.net(self.sample['pc_input_0'][self.constr_row_select['pc']])
            pc_1_pred = self.net(self.sample['pc_input_1'][self.constr_row_select['pc']])
            c1 = (pc_0_pred - pc_1_pred).squeeze()
            
            ## ic 
            ic_pred = self.net(self.sample['ic_input'][self.constr_row_select['ic']])
            c2 = ic_pred.squeeze() - self.sample['ic_u0'][self.constr_row_select['ic']]
            
            c = torch.cat((c1,c2))
        return c


    def plot_u(self, save_path, u, title, is_true = False):
        
        # reshape u
        u = u.reshape(self.x_discretization,-1).detach().numpy()
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.imshow(u, cmap='viridis',vmin=self.vmin, vmax=self.vmax)  # Use 'viridis' colormap for better visualization
        plt.colorbar()  # Add a colorbar to show scale
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('t')
        if is_true:
            ax.scatter(self.sample['ic_input'][:,0]*self.x_discretization,self.sample['ic_input'][:,1], marker='o',s=3,c='blue')
            ax.scatter(self.sample['pc_input_0'][:,0]*self.x_discretization,self.sample['pc_input_0'][:,1]*self.t_discretization, marker='o',s=3,c='blue')
            ax.scatter(self.sample['pc_input_1'][:,0]*(self.x_discretization-1),self.sample['pc_input_1'][:,1]*self.t_discretization, marker='o',s=3,c='blue')
            ax.scatter(self.sample['fitting_input'][:,0]*(self.x_discretization-1),self.sample['fitting_input'][:,1]*(self.t_discretization-1), marker='o',s=3,c='red')
            ax.scatter(self.sample['pde_input'][:,0]*(self.x_discretization-1),self.sample['pde_input'][:,1]*(self.t_discretization-1), marker='o',s=3,c='black')
        ax.xaxis.tick_top()
        ax.set_xticks([0, u.shape[0]])
        ax.set_xticklabels([0, self.x_max])
        ax.set_yticks([0, u.shape[1]])
        ax.set_yticklabels([0, self.t_max])
        plt.tight_layout()
        fig.savefig(save_path)
        plt.close()
        return
    
    def plot(self,save_label=False,save_path=None,epoch=0):
        if save_label:
            title = 'True solution'
            self.plot_u(save_path, self.sample['U_true'], title, is_true = True)
        else:
            title = 'Prediction %s' %(epoch)
            # Compute prediction
            u_pred = self.net(self.sample['test_input'])
            self.plot_u(save_path, u_pred, title)