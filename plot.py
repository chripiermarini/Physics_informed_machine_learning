import torch
from stochasticsqp import *
from problems.problem_darcy_matrix import DarcyMatrix
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np
torch.set_default_device(device)
torch.manual_seed(22)
np.random.seed(22)
import sys
import matplotlib.pyplot as plt

def plot(problem,sample_type='train'):
    fig = plt.figure(figsize=(7, 7))
    for index in range(3):
        if sample_type == 'train':
            # Input x
            x = problem.input[index]
            # Ground-truth
            y = problem.ground_truth[index]
            # Model prediction
            out = problem.net(problem.input)[index]
        elif sample_type == 'test':
            # Input x
            x = problem.test_input_data[index]
            # Ground-truth
            y = problem.test_ground_truth[index]
            # Model prediction
            out = problem.net(problem.test_input_data)[index]
        x = x.cpu()
        y = y.cpu()
        out = out.cpu()
        vmax=torch.max(y)
        vmin=torch.min(y)

        ax = fig.add_subplot(3, 3, index*3 + 1)
        ax.imshow(x[0], cmap='gray')
        if index == 0:
            ax.set_title('Input x')
        plt.xticks([], [])
        plt.yticks([], [])

        ax = fig.add_subplot(3, 3, index*3 + 2)
        ax.imshow(y.squeeze(),vmin=vmin, vmax=vmax)
        if index == 0:
            ax.set_title('Ground-truth y')
        plt.xticks([], [])
        plt.yticks([], [])
        

        ax = fig.add_subplot(3, 3, index*3 + 3)
        ax.imshow(out.squeeze().detach().numpy(),vmin=vmin, vmax=vmax)
        if index == 0:
            ax.set_title('Model prediction')
        plt.xticks([], [])
        plt.yticks([], [])
        

    fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
    plt.tight_layout()
    fig.savefig('test.png')

if __name__ == '__main__':
    problem_name = "DarcyMatrix"  # "Spring" #sys.argv[1]
    problem = eval(problem_name)(device, n_obj_sample = 10, n_constrs = 30, reg=1)
    path='mdl/nn_epoch200_DarcyMatrix'
    problem.load_net(path)
    plot(problem,sample_type='train')
