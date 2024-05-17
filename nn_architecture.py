import torch.nn as nn
from neuralop.models import FNO

class FNOLocal(nn.Module):
    def __init__(self, n_discretize=16, hidden_channels=16):
        super(FNOLocal, self).__init__()
        self.FNO = FNO(n_modes = (n_discretize, n_discretize), 
                       hidden_channels=hidden_channels,
                       in_channels=3,
                       out_channels=1)
        
    def forward(self, inputs):
        out = self.FNO(inputs)
        return out

class FCN(nn.Module):
    "Defines a connected network"
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS,is_darcy=False):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        self.is_darcy=is_darcy
        self.n_input = N_INPUT
        self.n_output = N_OUTPUT
        
    def forward(self, x):
        if self.is_darcy:
            x_shape = x.shape
            x = x.reshape(-1, self.n_input)
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        if self.is_darcy:
            x = x.reshape((x_shape[0],1, x_shape[2], x_shape[3]))
        return x
