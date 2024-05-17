from .problem_base import BaseProblem
from nn_architecture import *
import torch
from neuralop.datasets import load_darcy_flow_small
import matplotlib.pyplot as plt

"""
## Problem Statement Darcy
"""

class Darcy(BaseProblem):

    name = 'Darcy'
    f = 1
    
    def __init__(self, device, conf):
  
            
        self.conf = conf
        
        self.regs = self.conf['regs']
        
        self.constraint_type = self.conf['constraint_type']
        
        self.n_constrs = self.conf['n_constrs']

        self.n_discretize = self.conf['x_discretization']
        
        self.n_train_group_pde_parameters = self.conf['n_train_group_pde_parameters']
        
        self.n_test_group_pde_parameters = self.conf['n_test_group_pde_parameters']

        self.input, self.ground_truth, self.test_input_data, self.test_ground_truth = self.generate_sample(device)
        # shape of input : [batch_size, 3, n_discretize, n_discretize], in the 3 channels, the first channel is nu(x), second is x1, and third is x2
        
        # ensure the x max boundary is x_max
        assert(self.input[0,1,-1,0] == self.conf['x_max'])
        
        # Set boundary pixel index
        self.n_boundary_pixels_each_sample = 4*(self.n_discretize-1)
        self.n_interior_pixels_each_sample = (self.n_discretize-2)**2
        self.boundary_idx, self.interior_idx = self.set_boundary_and_interior_pixel_idx()
        
        # Set indices pixels for constraints and fixed it
        self.constr_pixel_idx = self.set_constraint_pixel_idx()
        
        # Initialize NN
        if self.conf['nn_name'] == 'FCN':
            self.conf['nn_input'] = self.input[0].reshape(-1).shape[0]
            self.conf['nn_output'] = self.ground_truth[0].reshape(-1).shape[0]
            self.net = eval(self.conf['nn_name'])(self.conf['nn_input'], self.conf['nn_output'],
                                                self.conf['nn_parameters']['n_hidden'],
                                                self.conf['nn_parameters']['n_layers'],is_darcy=True)
        elif self.conf['nn_name'] == 'FNOLocal':
            self.conf['nn_parameters']['n_discretize'] = 16
            self.conf['nn_parameters']['hidden_channels'] = 4
            self.net = eval(self.conf['nn_name'])( self.conf['nn_parameters']['n_discretize'],  self.conf['nn_parameters']['hidden_channels'])
        
        self.net.to(device)
        
        self.n_parameters = self.count_parameters(self.net)
        
        self.mse_func = torch.nn.MSELoss() # Mean squared error

    def generate_sample(self, device):
        """
        Generate input tensor

        input_data is composed by n_obj points, which are 3 channels of 16x16 pixels
        the first channel is nu, the second are values from 0 to 1 from left to right,
        the third are values from 0 to 1 from top to bottom

        ground_truth are labels, following the same structure (only one image for each data points)
        """
        train_loader, test_loaders, data_processor = load_darcy_flow_small(
            n_train=self.n_train_group_pde_parameters,
            batch_size=self.n_train_group_pde_parameters,
            test_resolutions=[self.n_discretize],
            n_tests=[self.n_test_group_pde_parameters],
            test_batch_sizes=[self.n_test_group_pde_parameters],
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

    def set_boundary_and_interior_pixel_idx(self):
        """ This function creates all the indexes to select the points at the boundaries
        of the 16x16 pixel image"""
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
        
        x1 = torch.arange(1,self.n_discretize-1)
        x2 = torch.arange(1,self.n_discretize-1)
        grid_x1, grid_x2 = torch.meshgrid(x1, x2, indexing='ij')
        interior_idx = torch.cat((grid_x1.reshape(1,-1), grid_x2.reshape(1,-1)),dim=0).T
        
        return boundary_idx, interior_idx

    def set_constraint_pixel_idx(self):
        # Selected sample for constraints
        sample_idx = torch.randperm(self.n_train_group_pde_parameters)[:self.n_constrs]
        
        if self.constraint_type == 'pde':
            pixel_select = torch.randperm(self.n_interior_pixels_each_sample)[:self.n_constrs]
            pixel_idx = self.interior_idx[pixel_select]
        elif self.constraint_type == 'boundary':        
            pixel_select = torch.randperm(self.n_boundary_pixels_each_sample)[:self.n_constrs]
            pixel_idx = self.boundary_idx[pixel_select]
        
        constr_pixel_idx = torch.concatenate((sample_idx.reshape(self.n_constrs,1), pixel_idx),dim=1)
        constr_pixel_idx = constr_pixel_idx.to(torch.int32)
        return constr_pixel_idx

    def apply_frame(self, tensor, discretize):
        # Create a 16x16 tensor filled with zeros
        tensor_16x16 = torch.zeros((tensor.shape[0], discretize, discretize))

        # Place the 14x14 tensor at the center of the 16x16 tensor
        tensor_16x16[:, 1:15, 1:15] = tensor[:]

        # Set the first row, last row, first column, and last column to ones
        tensor_16x16[:, 0, :] = 1  # Top row
        tensor_16x16[:, -1, :] = 1  # Bottom row
        tensor_16x16[:, :, 0] = 1  # Leftmost column
        tensor_16x16[:, :, -1] = 1  # Rightmost column

        return torch.Tensor(tensor_16x16)
    
    def pde(self, nu, x1, x2, out):
        # Compute objective function value, where the Laplace operator need second order derivatives
        first_derivative = torch.autograd.grad(outputs=out.sum(),
                                           inputs=(x1, x2),
                                           create_graph=True,
                                           allow_unused=True)
        div_argument = [nu * first_derivative[i] for i in range(len(first_derivative))]

        first_derivative_nu_x1 = torch.zeros_like(x1)
        first_derivative_nu_x1[:,:(self.n_discretize-1),:] = (nu[:,1:self.n_discretize,:] - nu[:,:(self.n_discretize-1),:])  / (self.conf['x_max'] /  (self.n_discretize - 1))
        
        first_derivative_nu_x2 = torch.zeros_like(x2)
        first_derivative_nu_x2[:,:,:(self.n_discretize-1)] = (nu[:,:,1:self.n_discretize] - nu[:,:,:(self.n_discretize-1)]) / (self.conf['x_max'] /  (self.n_discretize - 1))

        second_derivative_x1 = torch.autograd.grad(outputs=div_argument[0].sum(), inputs=x1, create_graph=True)[0]
        second_derivative_x2 = torch.autograd.grad(outputs=div_argument[1].sum(), inputs=x2, create_graph=True)[0]

        p_matrix = -(first_derivative_nu_x1 * first_derivative[0] + second_derivative_x1 +
                    first_derivative_nu_x2 * first_derivative[1] + second_derivative_x2)-self.f

        return p_matrix

    def objective_func(self):
        
        if self.conf['batch_size'] == 'full':
            batch_idx = torch.arange(self.input.size(0))
        else:
            batch_idx = torch.randperm(self.input.size(0))[:int(self.input.size(0)*self.conf['batch_size'])]
        
        # separate x1 and x2 from sample and set they require_grad
        nu = self.input[batch_idx,0,:,:] #1 and 0 values
        x1 = self.input[batch_idx,1,:,:].requires_grad_(True) #from 0 to 1 starting from left to right
        x2 = self.input[batch_idx,2,:,:].requires_grad_(True) #from 0 to 1 starting from top to bottom
        
        # NN forward
        out = self.net(torch.stack((nu, x1, x2), 1)).requires_grad_(True)
        
        # compute pde
        p_matrix = self.pde(nu, x1, x2, out)
        # Indexing the P matrix of interior points
        p_matrix = p_matrix[:,1:(self.n_discretize-1), 1:(self.n_discretize-1)]
        
        # Compute PDE residual: MSE over all interior pixels       
        pde_loss = self.mse_func(p_matrix, torch.zeros(p_matrix.shape))
        
        # Compute boundary conditions residual over all boundary pixels
        out_boundary = out[:,0,self.boundary_idx[:,0],self.boundary_idx[:,1]]
        out_boundary_true = torch.zeros(out_boundary.shape)
        boundary_loss = self.mse_func(out_boundary,out_boundary_true)
        
        # fitting loss
        n_fitting_sample = int(len(batch_idx) * self.conf['fitting_sample_group_percent'])
        if n_fitting_sample == 0:
            n_fitting_sample = 1
        fitting_sample_idx = batch_idx[:n_fitting_sample]            
        out_fitting = out[:n_fitting_sample]
        out_fitting_true = self.ground_truth[fitting_sample_idx]
        fitting_loss = self.mse_func(out_fitting,out_fitting_true)
        
        fs = {
            'pde': pde_loss,
            'boundary': boundary_loss,
            'fitting': fitting_loss
        }
        return fs

    def constraint_func(self):      
        if self.n_constrs == 0:
          return torch.tensor([])
        # separate x1 and x2 from sample and set they require_grad
        nu = self.input[self.constr_pixel_idx[:,0],0,:,:] #1 and 0 values
        x1 = self.input[self.constr_pixel_idx[:,0],1,:,:].requires_grad_(True) #from 0 to 1 starting from left to right
        x2 = self.input[self.constr_pixel_idx[:,0],2,:,:].requires_grad_(True) #from 0 to 1 starting from top to bottom
        
        # NN forward
        out = self.net(torch.stack((nu, x1, x2), 1)).requires_grad_(True) 

        if self.constraint_type == 'pde':    
            # compute pde
            p_matrix = self.pde(nu, x1, x2, out)
            c = p_matrix[list(range(0,out.shape[0])),self.constr_pixel_idx[:,1],self.constr_pixel_idx[:,2]]
        elif self.constraint_type == 'boundary':        
            # Take the output of pixels for constraints      torch.Size([n_constr])
            c = out[list(range(0,out.shape[0])),0,self.constr_pixel_idx[:,1],self.constr_pixel_idx[:,2]]
        
        return c
    
    def plot_prediction(self, save_path=None, epoch=None, save_label=False):
        print_indices = self.conf['print_test_indices']
        # Ground-truth
        y = self.test_ground_truth[print_indices]
        y = y.cpu()
        if save_label == True:
            # Input x
            x = self.test_input_data[print_indices]
            x = x.cpu()
            
            fig = plt.figure(figsize=(4.2, 2.2))
            
            for i in range(len(print_indices)):

                vmax=torch.max(y[i])
                vmin=torch.min(y[i])

                ax = fig.add_subplot(len(print_indices), 2, i*2 + 1)
                ax.imshow(x[i,0], cmap='gray')
                if i == 0:
                    ax.set_title(r'Input $\nu$')
                plt.xticks([], [])
                plt.yticks([], [])

                ax = fig.add_subplot(len(print_indices), 2, i*2 + 2)
                ax.imshow(y[i].squeeze(),vmin=vmin, vmax=vmax)
                if i == 0:
                    ax.set_title('True Solution')
                plt.xticks([], [])
                plt.yticks([], [])
        else:
            # Prediction
            out = self.net(self.test_input_data[print_indices])
            out = out.cpu()
            
            fig = plt.figure(figsize=(2.2, 2.2))
            for i in range(len(print_indices)):

                vmax=torch.max(y[i])
                vmin=torch.min(y[i])
                
                ax = fig.add_subplot(len(print_indices), 1, i + 1)
                ax.imshow(out[i].squeeze().detach().numpy(),vmin=vmin, vmax=vmax)
                if i == 0:
                    ax.set_title('Epoch %s' %(epoch))
                plt.xticks([], [])
                plt.yticks([], [])

        plt.tight_layout()
        fig.savefig(save_path)
        plt.close("all")
