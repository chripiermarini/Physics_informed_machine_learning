import torch
from abc import ABC, abstractmethod
from PIL import Image
"""
## Base Problem Class
"""

class BaseProblem(ABC):
    """
        Base problem class for physics inform machine learning problems. 
        All the abstract methods are required to be redefined in the child classes. 
    """
    figsize=(4.2, 3.2) 
    figsize_rectangle =(4.2, 2.2)
    figsize_square =(2.2, 2.2)
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def generate_sample(self):
        """
        Method for generating sample. 
        The sample usually includes sample for computing data-fitting loss, pde/ode residual loss, boundary residual loss,
        and other residual loss. There is no restriction for the input and output args.
        """
        pass

    def count_parameters(self, nn_net):
        """
        Method for counting number of trainable pytorch neural network parameters.
        
        Parameters
        ----------
        nn_net:  torch.nn.Module
        
        Returns
        -------
        number of trainable nn parameters: int
        """
        return sum(p.numel() for p in nn_net.parameters() if p.requires_grad)
        
    def save_net(self,path):
        """
        Method for saving the neural network parameters
        
        Parameters
        ----------
        path:  str
        """
        torch.save(self.net.state_dict(), path)

    def load_net(self,path,device):
        """
        Method for loading the neural network parameters to self.net
        
        Parameters
        ----------
        path:  str
        device: torch.device
        """
        try:
            self.net.load_state_dict(torch.load(path,map_location=device))
        except:
            self.net.load_state_dict(torch.load(path+'.zip',map_location=device))
        self.net.eval()
        
    def objective_func_and_grad(self, optimizer, no_grad = False):
        """
        Compute objective function value and gradient value
        Parameters
        ----------
        optimizer: torch.optim.optimizer.Optimizer
        no_grad: bool
            True: gradient will not be computed, nor returned
            False: gradient will be returned
        
        Returns
        -------
            fs: dict
                contain four keys: 'pde', 'boundary', 'fitting', and 'f', where the value of 'f' 
                is the total objective value (loss), and the values of 'pde', 'boundary', are 'fitting' corresponding losses. The values are torch.Tensor type. 
            g_value: torch.Tensor 
                gradient of objective function 'f'. Only return when no_grad = False
        """
        optimizer.zero_grad()

        fs = self.objective_func()
        f = 0
        for loss_type, reg in self.conf['regs'].items():
            f += reg * fs[loss_type]
        fs['f'] = f.data
        
        if no_grad is True:
            return fs
        else:
            # Backward of objective function
            f.backward()
            # Assign derivative to gradient value
            g_value = torch.zeros(self.n_parameters)
            i = 0

            for name, param in self.net.named_parameters():
                grad_l = len(param.view(-1))
                #print(name, param.view(-1).shape)
                if param.grad is not None:
                    g_value[i:i + grad_l] = param.grad.view(-1)
                else:
                    #print('objective_func_and_grad', name, 'grad is none') # For some nn, the last layer bias grad is none.
                    pass
                i += grad_l
                
            return fs, g_value
        
    def constraint_func_and_grad(self, optimizer, no_grad = False):
        """
        Compute constraint function value and Jacobian value
        Parameters
        ----------
        optimizer: torch.optim.optimizer.Optimizer
        no_grad: bool
            True: gradient will not be computed, nor returned
            False: gradient will be returned
        
        Returns
        -------
            c_value: torch.Tensor
                constraint value
            J_value: torch.Tensor 
                Jacobian value. Only return when no_grad = False
        """
        c = self.constraint_func()
        c_value = c.data
        c_value = c_value.reshape(-1)

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
                        #print('constraint_func_and_grad', name, 'grad is none')
                        pass
                    i += grad_l

            return c_value, J_value
        
    @abstractmethod
    def objective_func(self):
        """
        Compute each objective function values. This should be implemented in each Child class method
        
        Returns
        -------
        fs: dict of form fs = {
            'pde': torch.Tensor,
            'boundary': torch.Tensor,
            'fitting': torch.Tensor
            }
        }
        """
        pass

    @abstractmethod
    def constraint_func(self):
        """
        Compute constraint function value. This should be implemented in each Child class method
        
        Returns
        -------
        c: torch.Tensor
        """
        pass

    def save_gif_PIL(self, outfile, files, fps=5, loop=0):
        """
        Method for saving GIFs
        """
        imgs = [Image.open(file) for file in files]
        imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)
        
    def set_torch_random_seed(self, seed):
        """
        Method for setting torch random seed
        """
        torch.manual_seed(seed)