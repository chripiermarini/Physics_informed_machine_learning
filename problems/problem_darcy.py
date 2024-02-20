import random

from nn_architecture import *
import torch
from torch.autograd import Variable
import numpy as np

"""
## Problem Statement Darcy

Given a function $u(x)$, we define the residual on the 2D Darcy Flow PDE as the following:
\begin{equation}
    \mathcal{F}(u(x)) = -\nabla \cdot(\nu(x)\nabla u(x)) - f(x) = 0,
\end{equation}
where $\nu \in L^{\infty}((0,1)^{2}; \mathbb{R})$ is a diffusion coefficient and $f \in L^{2}((0,1)^{2}; \mathbb{R})$ is the forcing function. 
The boundary conditions are simply $u(x,0)= 0 \quad \forall {x_{i}} \in \partial(0,1)^{2}$.
- Independent variables (input): $(x1,x2) = x$ 
- pde solution (outputs): $u(x1,x2) = u(x)$ 
- Let $\bar{u}(x)$ as the neural network-predicted pde solution at $(x)$
- 
### Generate Sample
- Define $S_{x}:=\{(x1_i,x2_i)\}_{i=1}^{n_b}$ as a set of sample where, for $i \in [1,n_b]$, $(x1_i,x2_i)$ is a sample point in the box of $[0,1]\times[0,1]$.
- We have to be able to separate the interior from the boundary.

### Constrained Machine Learning
Given $N$ data points $x_{i}$ where $i=1, \dots, N$, the optimization problem through which the neural net can approximate a solution of 2D Darcy Flow PDE should be the following:
\begin{equation}
    \begin{aligned}

    &\min_{x_{i} \in (0,1)^{2}} & \frac{1}{N} \sum_{i=1}^{N}\lVert \mathcal{F}(u(x_{i}))\rVert^{2}_{2}\\
    &s.t. &u(x_{i}) = 0, \quad \forall {x_{i}} \in \partial(0,1)^{2}.       
    \end{aligned}
\end{equation}

When forming the network, we have to keep in mind the number of inputs and outputs
In our case: #inputs = 2 (x1,x2)
and #outputs = 1

"""

class Darcy:
    def __init__(self, device, n_obj_sample=500, n_constrs=10):
        # Initialize NN
        self.n_input = 2
        self.n_output = 1
        self.net = TwoHiddenLayerFCNN(self.n_input, self.n_output)
        self.net.to(device)
        self.n_parameters = self.count_parameters(self.net)

        # Generate sample
        self.n_obj_sample = n_obj_sample
        self.n_constrs = n_constrs
        self.domain_obj, self.domain_constr = self.generate_sample()

        # Construct tensor of generated sample
        self.domain_obj_tensor = Variable(torch.from_numpy(self.domain_obj).float(), requires_grad=True).to(device)
        self.domain_constr_tensor = Variable(torch.from_numpy(self.domain_constr).float(), requires_grad=True).to(
            device)

    def count_parameters(self, nn_net):
        return sum(p.numel() for p in nn_net.parameters() if p.requires_grad)

    def generate_sample(self):
        """
        Generate domain_obj, u_obj, domain_constr
        """
        # Interior points for the objective function
        x1_obj, x2_obj = (np.random.uniform(low=0, high=1, size=(self.n_obj_sample, 1)),
                          np.random.uniform(0, 1, size=(self.n_obj_sample, 1)))
        domain_obj = np.concatenate((x1_obj, x2_obj), axis=1)
        # Boundary points for constraints (x1 = 0, x2= 0, x1 = 1, x2= 1)
        constr1 = np.concatenate((np.zeros((self.n_constrs, 1)), np.random.uniform(low=0, high=1, size=(self.n_constrs, 1))), axis=1)
        constr2 = np.concatenate((np.random.uniform(low=0, high=1, size=(self.n_constrs, 1)), (np.zeros((self.n_constrs, 1)))), axis=1)
        constr3 = np.concatenate((np.ones((self.n_constrs, 1)), np.random.uniform(low=0, high=1, size=(self.n_constrs, 1))), axis=1)
        constr4 = np.concatenate((np.random.uniform(low=0, high=1, size=(self.n_constrs, 1)), (np.ones((self.n_constrs, 1)))), axis=1)
        # Complete constraints
        constr = np.concatenate((constr1, constr2, constr3, constr4), axis=0)
        domain_constr = constr[torch.randperm(constr.shape[0])][:self.n_constrs]

        return domain_obj, domain_constr

    def objective_func_and_grad(self, optimizer, no_grad = False):
        """
        Compute objective function value and gradient value
        Output:
            objective function value and gradient value
        Arguments:
            optimizer: the optimizer object
        CHECK THE OBJECTIVE FUNCTION
        """
        #Need to separate the two variables x1 and x2 to compute the divergent
        first_col_domain_obj_tensor = (self.domain_obj_tensor[:,0].clone()
                                       .requires_grad_(True).reshape(-1,1)) #x1 of all data points
        second_col_domain_obj_tensor = (self.domain_obj_tensor[:,1].clone()
                                       .requires_grad_(True).reshape(-1,1)) #x2 of all data points

        # NN forward to compute the output of sample for objective function
        u_obj_pred = self.net(torch.cat((first_col_domain_obj_tensor, second_col_domain_obj_tensor), dim=1))

        # Compute the argument of the divergent
        first_u_derivative = torch.autograd.grad(outputs=u_obj_pred.sum(),
                                                 inputs= (first_col_domain_obj_tensor, second_col_domain_obj_tensor),
                                                 create_graph=True)
        u_obj_derivative = torch.cat((first_u_derivative[0], first_u_derivative[1]), dim=1)

        v = 2  # diffusion coefficient
        divergent_argument = torch.mul(v,u_obj_derivative)

        # compute the divergent (second derivative of x1 + second derivative of x2)
        second_derivative_first_column = torch.autograd.grad(divergent_argument[0].sum(),
                                                      first_col_domain_obj_tensor, create_graph=True)[0]
        second_derivative_second_column = torch.autograd.grad(divergent_argument[1].sum(),
                                                       second_col_domain_obj_tensor, create_graph=True)[0]
        divergent = second_derivative_first_column + second_derivative_second_column

        #compute the obj function
        forcing_function= torch.ones(size=(self.n_obj_sample,1))
        pde = -divergent - forcing_function

        #f = torch.div((torch.linalg.vector_norm(pde[:,0], ord= 1)), self.n_obj_sample)
        #f = torch.linalg.vector_norm(pde[:,0], ord= 1)
        f = torch.div((pde[:,0]**2).sum(), self.n_obj_sample)
        f_value = f.data

        # Backward of objective function
        optimizer.zero_grad()
        f.backward()

        if no_grad is True:
            return f_value
        else:
            # Assign derivative to gradient value
            g_value = torch.zeros(self.n_parameters)
            i = 0
            for name, param in self.net.named_parameters():
                # print(f"{i}:", name, param, param.grad)
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

        # Compute constraint value
        c = u_constr_pred.reshape(-1)
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


