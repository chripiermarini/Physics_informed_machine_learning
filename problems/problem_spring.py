from nn_architecture import *
import torch
from torch.autograd import Variable
import numpy as np
"""
## Problem Statement

We have been given a PDE: $\frac{du(x,t)}{dx}=2\frac{du(x,t)}{dt}+u(x,t)$
and boundary condition: $u(x,0)=6e^{-3x}$

- Independent variables (input): $(x,t)$ 
- pde solution (outputs): $u(x,t)$ 
- Let $\bar{u}(x,t)$ as the neural network-predicted pde solution at $(x,t)$


We want to use a neural network to accurately predict pde solution for all $x$ in range $[0,2]$ and $t$ in range $[0,1]$


When we solved this pde analytically, we found the solution: $u(x,t) = 6e^{-3x-2t}$

### Generate Sample
- Define $S_{B}:=\{(x_i,0)\}_{i=1}^{n_b}$ as a set of sample where, for $i \in [1,n_b]$, $(x_i,0)$ is a sample point on the boundary.
- Define $S_{I}:=\{(x_i,t_i)\}_{i=1}^{m}$ as a set of sample where, for $i \in [1,m]$, $(x_i,t_i)$ is a sample point in the interior of $[0,2]\times[0,1]$.


### Constrained Machine Learning
The objective is to minimize mse loss of predicted pde solution and true pde solution of boundary sample points, i.e.,  
$$
\frac{1}{n_b}\sum_{(x_i,0) \in S_B} \|\bar{u}(x_i,0) - u(x_i,0)\|^2.
$$
The constraints are pde is satisfied for all interior sample points, i.e.,
$$
\frac{d\bar{u}(x_i,t_i)}{dx}-2\frac{d\bar{u}(x_i,t_i)}{dt}-\bar{u}(x_i,t_i) = 0 \text{ for all } (x_i, t_i) \in S_I
$$

When forming the network, we have to keep in mind the number of inputs and outputs
In our case: #inputs = 2 (x,t)
and #outputs = 1

"""
class Spring:
    def __init__(self, device, n_obj_sample = 500, n_constrs = 10):

        # Initialize NN
        self.n_input = 2
        self.n_output = 1
        self.net = TwoHiddenLayerFCNN(self.n_input, self.n_output) 
        self.net.to(device)
        self.n_parameters = self.count_parameters(self.net)

        # Generate sample
        self.n_obj_sample = n_obj_sample
        self.n_constrs = n_constrs
        self.domain_obj, self.u_obj, self.domain_constr = self.generate_sample()

        # Construct tensor of generated sample 
        self.domain_obj_tensor = Variable(torch.from_numpy(self.domain_obj).float(), requires_grad=False).to(device)
        self.u_obj_tensor = Variable(torch.from_numpy(self.u_obj).float(), requires_grad=False).to(device)
        self.domain_constr_tensor = Variable(torch.from_numpy(self.domain_constr).float(), requires_grad=True).to(device)
        self.mse_cost_function = torch.nn.MSELoss() # Mean squared error
        
    def count_parameters(self, nn_net):
        return sum(p.numel() for p in nn_net.parameters() if p.requires_grad)

    def generate_sample(self):
        """
        Generate domain_obj, u_obj, domain_constr
        """ 
        # Boundary sample for objective
        x_obj = np.random.uniform(low=0.0, high=2.0, size=(self.n_obj_sample,1))
        t_obj = np.zeros((self.n_obj_sample,1))
        domain_obj = np.concatenate((x_obj, t_obj), axis = 1)
        u_obj = 6*np.exp(-3*x_obj)
        # Interior sample for constraints (no need pde true solution)
        x_constr = np.random.uniform(low=0.0, high=2.0, size=(self.n_constrs,1))
        t_constr = np.random.uniform(low=0.0, high=1.0, size=(self.n_constrs,1))
        domain_constr = np.concatenate((x_constr, t_constr), axis = 1)
        return domain_obj, u_obj, domain_constr
    

    def objective_func_and_grad(self, optimizer, no_grad = False):
        """
        Compute objective function value and gradient value
        Output: 
            objective function value and gradient value
        Arguments:
            optimizer: the optimizer object 
        """

        # NN forward to compute the output of sample for objective function
        u_obj_pred = self.net(self.domain_obj_tensor) 

        # Compute objective value
        f = self.mse_cost_function(u_obj_pred, self.u_obj_tensor)
        f_value = f.data

        # Backward of objective function
        optimizer.zero_grad()
        f.backward(retain_graph=True)

        if no_grad is True:
            return f_value
        else:
            # Assign derivative to gradient value
            g_value = torch.zeros(self.n_parameters)
            i = 0
            for name, param in self.net.named_parameters():
                grad_l = len(param.grad.view(-1))
                g_value[i:i + grad_l] = param.grad.view(-1)
                i += grad_l
            return f_value, g_value

    
    def constraint_func_and_grad(self, optimizer, no_grad = False):
        """
        Compute constraint function value and Jacobian value
        Output: 
            constraint function value and Jacobian value
        Arguments:
            optimizer: the optimizer object 
        """

        # NN forward to compute the output of sample for constraints
        u_constr_pred = self.net(self.domain_constr_tensor)

        # Compute the derivative of output with respect to domain
        u_derivative_to_domain = torch.autograd.grad(u_constr_pred.sum(), self.domain_constr_tensor, create_graph=True)[0]

        # Compute constraint value
        c = u_derivative_to_domain[:,0] - 2*u_derivative_to_domain[:,1] - u_constr_pred.reshape(-1)
        c_value = c.data

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
                    grads = torch.cat((grads, param.grad.view(-1)), 0)
                J_value[i, :] = grads
            return c_value, J_value

