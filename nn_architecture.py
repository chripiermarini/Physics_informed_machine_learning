import torch
import torch.nn as nn
from neuralop.models import FNO

class TwoHiddenLayerFCNN(nn.Module):
    def __init__(self, n_input, n_output, n_neurons=16):
        super(TwoHiddenLayerFCNN, self).__init__()
        self.n_input = n_input
        self.n_neurons = n_neurons
        self.n_output = n_output
        # the layer parameters are initialized by default
        self.hidden_layer1 = nn.Linear(self.n_input,self.n_neurons)
        self.hidden_layer2 = nn.Linear(self.n_neurons,self.n_neurons)
        self.output_layer = nn.Linear(self.n_neurons,self.n_output, bias = False)

    def forward(self, inputs):
        layer1_out = torch.sigmoid(self.hidden_layer1(inputs))
        layer2_out = torch.sigmoid(self.hidden_layer2(layer1_out))
        output = self.output_layer(layer2_out) ## For regression, no activation is used in output layer
        return output

class OneHiddenLayerFCNN(nn.Module):
    def __init__(self, n_input, n_output, n_neurons=1024):
        super(OneHiddenLayerFCNN, self).__init__()
        self.n_input = n_input
        self.n_neurons = n_neurons
        self.n_output = n_output
        # the layer parameters are initialized by default
        self.hidden_layer = nn.Linear(self.n_input,self.n_neurons)
        self.output_layer = nn.Linear(self.n_neurons,self.n_output, bias = False)

    def forward(self, inputs):
        out = torch.sigmoid(self.hidden_layer(inputs))
        output = self.output_layer(out) ## For regression, no activation is used in output layer
        return output

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
