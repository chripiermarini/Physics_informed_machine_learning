import random
from .problem_base import BaseProblem
from nn_architecture import *
import torch
from neuralop.datasets import load_darcy_flow_small

"""
## Problem Statement Darcy
"""

class DarcyMatrix(BaseProblem):
    n_discretize = 16
    hidden_channels= 4
    def __init__(self, device, n_obj_sample=500, n_constrs=10, n_test_sample=1, reg=0):
        """
        Input:
            n_obj_sample:   int, the number of pictures for training sample, each picture will have n_discretize * n_discretize pixel samples
            n_constrs:      int, the number of pixels on the boundary used for constraints. These pixels will be sampled from the boundary pixels of all the pictures.
            n_test_sample:  int, the number of pictures for test sample
            reg:            float (>=0), multiplier for the term of boundary conditions violation that will be added to the objective function
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
        self.input, self.ground_truth, self.test_input_data, self.test_ground_truth = self.generate_sample(device)

        # Set boundary pixel index
        self.n_boundary_pixels_each_sample = 4*(self.n_discretize-1)
        self.boundary_idx = self.set_boundary_pixel_idx()
        
        # Set indices pixels for constraints and fixed it
        self.constr_pixel_idx = self.set_constraint_pixel_idx()
        
        # Assign regularization multiplier 
        self.reg = torch.tensor(reg)

    def generate_sample(self, device):
        """
        Generate input tensor
        """
        train_loader, test_loaders, data_processor = load_darcy_flow_small(
            n_train=self.n_obj_sample,
            batch_size=self.n_obj_sample,
            test_resolutions=[self.n_discretize],
            n_tests=[self.n_test_sample],
            test_batch_sizes=[self.n_test_sample],
            positional_encoding=True,
            grid_boundaries=[[0, self.n_discretize /(self.n_discretize-1) ], [0, self.n_discretize /(self.n_discretize-1)]] # ensure boundaries are 0 or 1
        )
        data_processor = data_processor.to(device)
        res = data_processor.preprocess(train_loader.dataset[:])
        input_data = res['x']
        ground_truth = res['y']

        test_loader = test_loaders[self.n_discretize]
        res_test = data_processor.preprocess(test_loader.dataset[:])
        test_input_data = res_test['x']
        test_ground_truth = res_test['y']

        return input_data, ground_truth, test_input_data, test_ground_truth

    def set_boundary_pixel_idx(self):
        boundary_idx = torch.zeros(self.n_boundary_pixels_each_sample,2)

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
        boundary_idx = boundary_idx.to(torch.int32)
        return boundary_idx

    def set_constraint_pixel_idx(self):
        # randomly select self.n_constrs number of pixels from the boundary of all the samples

        sample_idx = torch.randint(low=0,high=self.n_obj_sample,size=(self.n_constrs,))
        pixel_select = torch.randint(low=0,high=self.n_boundary_pixels_each_sample,size=(self.n_constrs,))
        pixel_idx = self.boundary_idx[pixel_select]
        constr_pixel_idx = torch.concatenate((sample_idx.reshape(self.n_constrs,1), pixel_idx),dim=1)
        constr_pixel_idx = constr_pixel_idx.to(torch.int32)
        return constr_pixel_idx

    def objective_func(self,return_multiple_f=False):
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
        div_argument = [nu * first_derivative[i] for i in range(len(first_derivative))]
        second_derivative_x1 = torch.autograd.grad(outputs=div_argument[0].sum(), inputs=x1, create_graph=True)[0]
        second_derivative_x2 = torch.autograd.grad(outputs=div_argument[1].sum(), inputs=x2, create_graph=True)[0]
        
        mse_func = torch.nn.MSELoss()
        
        # PDE residual of size torch.Size([n_obj_sample, n_discretize, n_discretize])
        p_matrix = - (second_derivative_x1 + second_derivative_x2) - 1
        
        # Indexing the P matrix of interior points
        p_matrix = p_matrix[:,1:(self.n_discretize-1), 1:(self.n_discretize-1)]
        
        # Compute PDE residual: MSE over all interior pixels       
        f_interior = mse_func(p_matrix, torch.zeros(p_matrix.shape))
        
        # Compute boundary conditions residual over all boundary pixels
        out_boundary = out[:,0,self.boundary_idx[:,0],self.boundary_idx[:,1]]
        out_boundary_true = torch.zeros(out_boundary.shape)
        f_boundary = mse_func(out_boundary,out_boundary_true)
        
        # Combine the objective function
        f = f_interior + self.reg * f_boundary
        if return_multiple_f:
            return f, f_interior.data, f_boundary.data
        else:
            return f

    def constraint_func(self):      
        # NN forward to compute the output of all input  torch.Size([n_obj_sample, 1, n_discretize, n_discretize])
        out = self.net(self.input[self.constr_pixel_idx[:,0]])     

        # Take the output of pixels for constraints      torch.Size([n_constr])
        c = out[list(range(0,out.shape[0])),0,self.constr_pixel_idx[:,1],self.constr_pixel_idx[:,2]]
        
        return c
    