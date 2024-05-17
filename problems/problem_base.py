import torch
from abc import ABC, abstractmethod
from PIL import Image
"""
## Base Problem Class
"""

class BaseProblem(ABC):
    figsize=(4.2, 3.2) 
    figsize_rectangle =(4.2, 2.2)
    figsize_square =(2.2, 2.2)
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def generate_sample(self):
        pass

    def count_parameters(self, nn_net):
        return sum(p.numel() for p in nn_net.parameters() if p.requires_grad)
        
    def save_net(self,path):
        torch.save(self.net.state_dict(), path)

    def load_net(self,path,device):
        try:
            self.net.load_state_dict(torch.load(path,map_location=device))
        except:
            self.net.load_state_dict(torch.load(path+'.zip',map_location=device))
        self.net.eval()
        
    def objective_func_and_grad(self, optimizer, no_grad = False):
        """
        Compute objective function value and gradient value
        Output:
            objective function value and gradient value
        Arguments:
            optimizer: the optimizer object
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
                    #print('objective_func_and_grad', name, 'grad is none') # The last layer bias grad is none, I don't know why.
                    pass
                i += grad_l
                
            return fs, g_value
        
    def constraint_func_and_grad(self, optimizer, no_grad = False):
        """
        Compute constraint function value and Jacobian value
        Output:
            constraint function value and Jacobian value
        Arguments:
            optimizer: the optimizer object
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
        pass

    @abstractmethod
    def constraint_func(self):
        pass

    def save_gif_PIL(self, outfile, files, fps=5, loop=0):
        "Helper function for saving GIFs"
        imgs = [Image.open(file) for file in files]
        imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)
        
    def set_torch_random_seed(self, seed):
        torch.manual_seed(seed)