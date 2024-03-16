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
    name = "Spring"

    n_discretization = 100
    def __init__(self, device, n_obj_sample = 10, n_constrs = 3, reg = 1, constraint_type='boundary'):

        '''
        Input: 
            constraint_type: str, either 'boundary' or 'pde'
        '''
        # Initialize NN
        self.n_input = 2
        self.n_output = 1
        #self.net = OneHiddenLayerFCNN(self.n_input, self.n_output, n_neurons = 512) 
        self.net = TwoHiddenLayerFCNN(self.n_input, self.n_output,n_neurons = 64) 
        self.net.to(device)
        self.n_parameters = self.count_parameters(self.net)
        self.reg = torch.tensor(reg)
        self.constraint_type = constraint_type

        # Generate sample
        self.n_obj_sample = n_obj_sample
        self.n_constrs = n_constrs
        x_obj, self.domain_boundary, self.u_boundary, self.domain_interior, self.constr_row_select = self.generate_sample()

        # Construct tensor of generated sample 
        self.domain_boundary_tensor = Variable(torch.from_numpy(self.domain_boundary).float(), requires_grad=False).to(device)
        self.u_boundary_tensor = Variable(torch.from_numpy(self.u_boundary).float(), requires_grad=False).to(device)
        self.domain_interior_tensor = Variable(torch.from_numpy(self.domain_interior).float(), requires_grad=True).to(device)
        
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

        # Boundary sample for objective
        x_obj = np.random.uniform(low=0.0, high=2.0, size=(self.n_obj_sample,1))
        t_boundary = np.zeros((self.n_obj_sample,1))
        domain_boundary = np.concatenate((x_obj, t_boundary), axis = 1) #S_B

        #boundary true pde solution
        u_boundary = 6* np.e ** (-3*x_obj)

        # Interior sample for objective
        step = float(1 / self.n_discretization)
        t_interior = np.arange(start=0.0, stop=1.0 * (1+step) , step=step)
        t_interior = t_interior[1:]
        t_interior = t_interior.reshape((self.n_discretization,1))
        domain_interior = np.column_stack((np.meshgrid(x_obj,t_interior)[0].flatten(),
                               np.meshgrid(x_obj,t_interior)[1].flatten()))
        
        # Generate sample for constraints
        
        # If constraints are some points on boundary conditions, then the subsample are from domain_boundary
        # If constraints are some points on pde conditions, then the subsample are from domain_interior
        if self.constraint_type == 'boundary':
            n_domain_boundary = domain_boundary.shape[0]
            constr_row_select = np.random.randint(0, n_domain_boundary, size=(self.n_constrs))
        elif self.constraint_type == 'pde':
            n_domain_interior = domain_interior.shape[0]
            constr_row_select = np.random.randint(0, n_domain_interior, size=(self.n_constrs))
        
        return x_obj, domain_boundary, u_boundary, domain_interior, constr_row_select
    

    def objective_func_and_grad(self, optimizer, no_grad = False, return_multiple_f=False):
        """
        Compute objective function value and gradient value
        Output: 
            objective function value and gradient value
        Arguments:
            optimizer: the optimizer object 
        """

        #### differential operator
        # NN forward to compute the output of sample for constraint
        u_interior_pred = self.net(self.domain_interior_tensor)

        # Compute the derivative of output with respect to domain
        u_derivative_to_domain = torch.autograd.grad(u_interior_pred.sum(), self.domain_interior_tensor, create_graph=True)[
            0]

        differential_operator = u_derivative_to_domain[:, 0] - 2 * u_derivative_to_domain[:, 1] - u_interior_pred.reshape(-1)
        interior_loss = self.mse_cost_function(differential_operator, torch.zeros(differential_operator.shape))

        ### boundary loss
        # NN forward to compute the output of sample for objective function
        u_boundary_pred = self.net(self.domain_boundary_tensor)
        boundary_loss = self.mse_cost_function(u_boundary_pred, self.u_boundary_tensor)

        ##compute objective function
        f = interior_loss + self.reg * boundary_loss

        # Compute objective value
        f_value = f.data

        # Backward of objective function
        optimizer.zero_grad()
        f.backward(retain_graph=True)

        if no_grad is True:
            if return_multiple_f:
                return f_value, interior_loss.data, boundary_loss.data
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
                return f_value, interior_loss.data, boundary_loss.data, g_value
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
        
        if self.constraint_type == 'boundary':
            domain_boundary_constr = self.domain_boundary_tensor[self.constr_row_select]
            u_boundary_constr = self.u_boundary_tensor[self.constr_row_select]
            u_boundary_pred = self.net(domain_boundary_constr)
            c = u_boundary_pred - u_boundary_constr
        elif self.constraint_type == 'pde':
            domain_interior_constr = self.domain_interior_tensor[self.constr_row_select]
            u_interior_pred = self.net(domain_interior_constr)
            u_derivative_to_domain = torch.autograd.grad(u_interior_pred.sum(), domain_interior_constr, create_graph=True)[
            0]
            differential_operator = u_derivative_to_domain[:, 0] - 2 * u_derivative_to_domain[:, 1] - u_interior_pred.reshape(-1)
            c = differential_operator

        c_value = c.data
        c_value = c_value.reshape(c_value.shape[0])

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
        
        
    
    def save_net(self,path):
        torch.save(self.net.state_dict(), path)

    def load_net(self,path):
        self.net.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
        self.net.eval()
        
    def get_u_true(self,domains):
        # domains.shape = [n, 2]
        # u(x,t) = 6e^{-3x-2t}
        u = 6 * np.e **  (-3 * domains[:,0] - 2 * domains[:,1])
        return u
