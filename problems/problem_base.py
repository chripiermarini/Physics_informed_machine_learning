from nn_architecture import *
import torch
from torch.autograd import Variable
import numpy as np

class BaseProblem:
    def __init__(self, device, n_obj_sample, n_constrs):

        # Initialize NN
        pass