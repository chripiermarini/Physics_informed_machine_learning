import torch
import torch.nn as nn

class TwoHiddenLayerFCNN(nn.Module):
    def __init__(self, n_input, n_output, n_neurons=16):
        super(TwoHiddenLayerFCNN, self).__init__()
        self.n_input = n_input
        self.n_neurons = n_neurons
        self.n_output = n_output
        self.hidden_layer1 = nn.Linear(self.n_input,self.n_neurons)
        self.hidden_layer2 = nn.Linear(self.n_neurons,self.n_neurons)
        self.output_layer = nn.Linear(self.n_neurons,self.n_output, bias = False)

    def forward(self, inputs):
        layer1_out = torch.sigmoid(self.hidden_layer1(inputs))
        layer2_out = torch.sigmoid(self.hidden_layer2(layer1_out))
        output = self.output_layer(layer2_out) ## For regression, no activation is used in output layer
        return output