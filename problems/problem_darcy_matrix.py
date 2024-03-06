import random
from nn_architecture import *
import torch
from torch.autograd import Variable
import numpy as np
from neuralop.datasets import load_darcy_flow_small

"""
## Problem Statement Darcy
"""

class DarcyMatrix:
    n_discretize = 16
    hidden_channels=16
    def __init__(self, device, n_obj_sample=500, n_constrs=10, n_test_sample=1):
        """
        Input: 
            n_obj_sample:   int, the number of pictures for training sample, each picture will have n_discretize * n_discretize pixel samples
            n_constrs:      int, the number of pixels on the boundary used for constraints. These pixels will be sampled from the boundary pixels of all the pictures.
        """
        # Set problem name
        self.name = 'DarcyMatrix(' + 'n_discretize' + str(self.n_discretize) +  ', hidden_channels' + str(self.hidden_channels) +')'
        
        # Initialize NN
        self.net = FNOLocal(n_discretize = self.n_discretize, hidden_channels=self.hidden_channels)
        self.net.to(device)
        self.n_parameters = self.count_parameters(self.net)

        # Generate sample tensor and put to device
        self.n_obj_sample = n_obj_sample
        self.n_constrs = n_constrs
        self.n_test_sample = n_test_sample
        
        # shape of input : [batch_size, 3, n_discretize, n_discretize], in the 3 channels, the first channel is nu(x), second is x1, and third is x2 
        self.input = self.generate_sample(device)
        
        # Set indices pixels for constraints and fixed it
        self.constr_pixel_idx = self.set_constraint_pixel_idx()

    def count_parameters(self, nn_net):
        return sum(p.numel() for p in nn_net.parameters() if p.requires_grad)

    def generate_sample(self, device):
        """
        Generate input tensor
        """
        train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=self.n_obj_sample, batch_size=self.n_obj_sample,
        test_resolutions=[self.n_discretize], n_tests=[self.n_test_sample],
        test_batch_sizes=[self.n_test_sample],
        positional_encoding=True,
        grid_boundaries=[[0, self.n_discretize /(self.n_discretize-1) ], [0, self.n_discretize /(self.n_discretize-1)]] # ensure boundaries are 0 or 1
)
        data_processor = data_processor.to(device)
        res = data_processor.preprocess(train_loader.dataset[:])
        input_data = res['x']  

        return input_data

    def set_constraint_pixel_idx(self):
        # randomly select self.n_constrs number of pixels from the boundary of all the samples
        
        sample_idx = torch.randint(low=0,high=self.n_obj_sample,size=(self.n_constrs,))
        n_boundary_pixels = 4*(self.n_discretize-1)
        boundary_idx = torch.zeros(n_boundary_pixels,2)

        # Assign x1 idx
        # full bottom row
        boundary_idx[:self.n_discretize,0] = 0
        # full top row
        boundary_idx[self.n_discretize:2*self.n_discretize,0] = self.n_discretize-1
        # center right column
        boundary_idx[2*self.n_discretize:(3*self.n_discretize - 2),0] = torch.arange(1,self.n_discretize-1) 
        # center left column
        boundary_idx[(3*self.n_discretize - 2):,0] = torch.arange(1,self.n_discretize-1) 

        # Assign x2 idx
        # full bottom row
        boundary_idx[:self.n_discretize,1] = torch.arange(self.n_discretize) 
        # full top row
        boundary_idx[self.n_discretize:2*self.n_discretize,1] = torch.arange(self.n_discretize) 
        # center right column
        boundary_idx[2*self.n_discretize:(3*self.n_discretize - 2),1] = 0
        # center left column
        boundary_idx[(3*self.n_discretize - 2):,1] = self.n_discretize-1
        
        # For checking boundary_idx are assigned correctly
        #import matplotlib.pyplot as plt
        #b = boundary_idx.numpy()
        #plt.plot(b[:,0],b[:,1],'.')

        pixel_select = torch.randint(low=0,high=n_boundary_pixels,size=(self.n_constrs,))
        pixel_idx = boundary_idx[pixel_select]
        constr_pixel_idx = torch.concatenate((sample_idx.reshape(self.n_constrs,-1), pixel_idx),dim=1)
        constr_pixel_idx = constr_pixel_idx.to(torch.int32)
        return constr_pixel_idx

    def objective_func_and_grad(self, optimizer, no_grad = False):
        """
        Compute objective function value and gradient value
        Output:
            objective function value and gradient value
        Arguments:
            optimizer: the optimizer object
        CHECK THE OBJECTIVE FUNCTION
        """
        
        # separate x1 and x2 from sample and set they require_grad
        nu = self.input[:,0,:,:]
        x1 = self.input[:,1,:,:].requires_grad_(True)
        x2 = self.input[:,2,:,:].requires_grad_(True)
        
        # NN forward
        out = self.net(torch.stack((nu, x1, x2), 1)).requires_grad_(True)
        
        # Compute objective function value, where the Laplace operator need second order derivatives
        first_derivative = torch.autograd.grad(outputs=out.sum(),
                                           inputs=(x1, x2),
                                           create_graph=True,
                                           allow_unused=True)
        # TODO: confirm this is correct. When do the  Laplace operation, shall we consider the derivative of nu to domain? Otherwise, for pixels of nu is zero, their divergence are just zero 
        div_argument = [nu * first_derivative[i] for i in range(len(first_derivative))]
        second_derivative_x1 = torch.autograd.grad(outputs=div_argument[0].sum(), inputs=x1, create_graph=True)
        second_derivative_x2 = torch.autograd.grad(outputs=div_argument[1].sum(), inputs=x2, create_graph=True)
        
        # PDE residual of size torch.Size([n_obj_sample, n_discretize, n_discretize])
        p_matrix = - (second_derivative_x1[0] + second_derivative_x2[0]) - 1

        # Compute objective value: MSE over all interior pixels
        n_sample_pixel_interior = self.n_obj_sample * (self.n_discretize-2) * (self.n_discretize-2)
        f = (torch.linalg.matrix_norm(p_matrix[:,1:(self.n_discretize-1), 1:(self.n_discretize-1)])**2).sum()
        f = f / n_sample_pixel_interior
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
                grad_l = len(param.view(-1))
                #print(name, param.view(-1).shape)
                if param.grad is not None:
                    g_value[i:i + grad_l] = param.grad.view(-1)
                else:
                    #print('objective_func_and_grad', name, 'grad is none') # The last layer bias grad is none, I don't know why.
                    pass
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
        # NN forward to compute the output of all input  torch.Size([n_obj_sample, 1, n_discretize, n_discretize])
        out = self.net(self.input)     

        # Take the output of pixels for constraints      torch.Size([n_constr])
        c = out[self.constr_pixel_idx[:,0],0,self.constr_pixel_idx[:,1],self.constr_pixel_idx[:,2]]
        c_value = c.data
        
        if no_grad is True:
            return c_value
        else:
            # Compute Jacobian
            J_value = torch.zeros(self.n_constrs, self.n_parameters)
            for j in range(self.n_constrs):

                # Backward of each constraint function
                optimizer.zero_grad()
                c[j].backward(retain_graph=True)
                
                # Assign derivative to J_value
                i = 0
                for name, param in self.net.named_parameters():
                    grad_l = len(param.view(-1))
                    if param.grad is not None:
                        J_value[j,i:i + grad_l] = param.grad.view(-1)
                    else:
                        print('constraint_func_and_grad', name, 'grad is none') # Not happen in Jacobian. It's strange. TODO: to be checked
                    i += grad_l

            return c_value, J_value
