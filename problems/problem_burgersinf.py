from nn_architecture import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from problems.problem_base import BaseProblem
from scipy.integrate import odeint

""" Problem statement Burgers' equation"""

class Burgersinf(BaseProblem):
    name = 'Burgersinf'
    nu = 0.01   ##kinematic viscosity coefficient
    mu = 0.5    
    
    def __init__(self, device, conf):

        self.conf = conf

        # Initialize NN
        self.x_discretization = self.conf['x_discretization']
        self.conf['nn_input'] += self.x_discretization
        
        self.net = eval(self.conf['nn_name'])(self.conf['nn_input'], self.conf['nn_output'],
                                              self.conf['nn_parameters']['n_hidden'],
                                              self.conf['nn_parameters']['n_layers'])
        self.net.to(device)

        self.n_parameters = self.count_parameters(self.net)

        self.regs = self.conf['regs']

        self.constraint_type = self.conf['constraint_type']

        self.n_constrs = self.conf['n_constrs']
        
        self.x_max = self.conf['x_max']
        
        self.t_max = self.conf['t_max']
        
        self.t_discretization = self.conf['t_discretization']

        self.n_obj_sample_fitting = self.conf['n_obj_sample_fitting_per_group']

        self.n_group_pde_parameters = self.conf['n_group_pde_parameters']
        self.n_group_pde_parameters_test = self.conf['n_group_pde_parameters_test']
        
        # Generate sample
        self.sample, self.constr_row_select = self.generate_sample(device)
        
        self.mse_cost_function = torch.nn.MSELoss()  # Mean squared error

    def initial_function(self, xs, initial_rand_phase):
        # We use the fix initial function u_0(x) = 3*sin(2pi x)
        u0 = torch.sin(2*torch.pi * xs + initial_rand_phase)
        return u0
    
    def true_burgers_solution(self, xs, ts, u0):
        #Wave number discretization
        k = 2 * np.pi * np.fft.fftfreq( self.x_discretization, d = (xs[1] - xs[0]).cpu().detach().numpy() )
        
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
        U = odeint(burg_system, u0.cpu(), ts.cpu(), args=(k,self.mu,self.nu,))
        return U
    
    
    def generate_one_group(self, xs, ts, device):
        
        initial_rand_phase = torch.randint(100,(1,))/50* torch.pi
        
        # Compute Initial condition
        u0 = self.initial_function(xs,initial_rand_phase)
        
        # compute true solution
        u_true = self.true_burgers_solution(xs, ts, u0)
        u_true = torch.tensor(u_true).to(torch.float32).to(device)
        #self.plot_u(u_true, xs, ts)
        
        # Compose all sample input only for test
        X, T = torch.meshgrid(xs, ts, indexing='xy')
        X = X.flatten()
        T = T.flatten()
        input = torch.stack((X,T), axis=1)
        input = torch.cat((input,u0.repeat(input.shape[0],1)), axis=1)
        U_true = u_true.flatten()
        return input, U_true,u0
    
    
    def generate_sample(self,device):
        sample = {}
        
        # sample['test_input'] = test_input
        # sample['U_true'] = U_true
        # sample['u0'] = u0
        # Generate x and t discretization
        xs = torch.linspace(0,self.x_max,self.x_discretization)
        ts = torch.linspace(0,self.t_max,self.t_discretization)
        
        # Generate trainning data
        for k in range(self.n_group_pde_parameters):
            input, U_true, u0 = self.generate_one_group(xs, ts, device)
            
            # Fitting sample
            fitting_idx = torch.randperm(input.shape[0])[:self.n_obj_sample_fitting]
            fitting_input = input[fitting_idx]
            fitting_u_true = U_true[fitting_idx]
            if k == 0:
                sample['fitting_input'] = fitting_input
                sample['fitting_u_true'] = fitting_u_true
            else:
                sample['fitting_input'] = torch.cat((sample['fitting_input'], fitting_input), axis=0)
                sample['fitting_u_true'] = torch.cat((sample['fitting_u_true'], fitting_u_true), axis=0)
                
            # boundary condition sample
            # u(0,t) = u(1,t) for some t
            pc_input_0 = torch.zeros(self.t_discretization, 2)
            pc_input_1 = torch.ones(self.t_discretization, 2)
            pc_input_0[:,1] = ts
            pc_input_1[:,1] = ts
            pc_input_0 = torch.cat((pc_input_0,u0.repeat(pc_input_0.shape[0],1)), axis=1)
            pc_input_1 = torch.cat((pc_input_1,u0.repeat(pc_input_1.shape[0],1)), axis=1)
            if k == 0:
                sample['pc_input_0'] = pc_input_0
                sample['pc_input_1'] = pc_input_1
            else:
                sample['pc_input_0'] = torch.cat((sample['pc_input_0'], pc_input_0), axis=0)
                sample['pc_input_1'] = torch.cat((sample['pc_input_1'], pc_input_1), axis=0)
        
            # u(x,0) = u0(x) for some x
            ic_input = torch.zeros(self.x_discretization, 2)
            ic_input[:,0] = xs
            ic_input = torch.cat((ic_input,u0.repeat(ic_input.shape[0],1)), axis=1)
            ic_u0 = u0
            if k == 0:
                sample['ic_input'] = ic_input
                sample['ic_u0'] = ic_u0
            else:
                sample['ic_input'] = torch.cat((sample['ic_input'], ic_input), axis=0)
                sample['ic_u0'] = torch.cat((sample['ic_u0'], ic_u0), axis=0)
            
            # PDE sample
            if k == 0:
                sample['pde_input'] = input
            else:
                sample['pde_input'] = torch.cat((sample['pde_input'], input), axis=0)
        
        # Generate testing data
        sample['test'] = {k: {} for k in range(self.n_group_pde_parameters_test)} 
        for k in range(self.n_group_pde_parameters_test):
            input, U_true, u0 = self.generate_one_group(xs, ts, device)
            # PDE sample
            sample['test'][k]['input'] = input
            sample['test'][k]['u_true'] = U_true
        
        # all possible couples for the pde of the objective constraints
        if self.constraint_type == 'pde':
            constr_row_select =  torch.randperm(sample['pde_input'].shape[0])[:self.n_constrs]
        elif self.constraint_type == 'boundary':
            # half periodic condition and half initial condition
            half = int(self.n_constrs/2)
            constr_row_select={}
            constr_row_select['pc'] =  torch.randperm(self.sample['pc_input_0'].shape[0])[:half]
            constr_row_select['ic'] =  torch.randperm(self.sample['pc_input_0'].shape[0])[:(self.n_constrs - half)]
            
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

    def objective_func(self):

        if self.conf['batch_size'] == 'full':
            batch_idx = torch.arange(self.sample['pde_input'].size(0))
        else:
            batch_idx = torch.randperm(self.sample['pde_input'].size(0))[:int(self.sample['pde_input'].size(0)*self.conf['batch_size'])]

        # fitting loss
        u_fitting_pred = self.net(self.sample['fitting_input'])
        fitting_loss = self.mse_cost_function(u_fitting_pred.squeeze(), self.sample['fitting_u_true'])

        # pde loss
        x_pde = self.sample['pde_input'][batch_idx,0].requires_grad_(True) 
        t_pde = self.sample['pde_input'][batch_idx,1].requires_grad_(True)
        u0 =  self.sample['pde_input'][batch_idx,2:]
        pde_input = torch.cat((torch.stack((x_pde, t_pde), axis=1), u0), axis=1)
        u_pde_pred = self.net(pde_input).requires_grad_(True)
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
            u0 =  self.sample['pde_input'][self.constr_row_select,2:]
            pde_input = torch.cat((torch.stack((x_pde, t_pde), axis=1), u0), axis=1)
            u_pde_pred = self.net(pde_input).requires_grad_(True)
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

    def plot_prediction(self,save_label=False,save_path=None,epoch=0):
        fig = plt.figure(figsize=self.figsize_rectangle_vertical)
        for i in range(self.n_group_pde_parameters_test):
            ax = fig.add_subplot(3,1, i + 1)
            vmax=torch.max(self.sample['test'][i]['u_true'])
            vmin=torch.min(self.sample['test'][i]['u_true'])
            if save_label:
                u = self.sample['test'][i]['u_true']
                if i == 0:
                    ax.set_title('Test Label')
                plt.xticks([], [])
                plt.yticks([], [])
                plt.ylabel('sample # %s' %(i))
            else:
                u = self.net(self.sample['test'][i]['input'])
                if i == 0:
                    ax.set_title('Epoch %s' %(epoch))
                plt.xticks([], [])
                plt.yticks([], [])
            u = u.reshape(self.x_discretization,-1).cpu().detach().numpy()
            ax.imshow(u, cmap='viridis',vmin=vmin, vmax=vmax)  # Use 'viridis' colormap for better visualization

        plt.tight_layout()
        fig.savefig(save_path)
        plt.close("all")