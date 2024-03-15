import torch
from stochasticsqp import *
from problems.problem_darcy_matrix_old import DarcyMatrixOld
from problems.problem_darcy_matrix import DarcyMatrix
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np
torch.set_default_device(device)
torch.manual_seed(22)
np.random.seed(22)
import sys


def run(optimizer, problem, max_iter = 10000, save_every=10):
    optimizer.printerHeader()

    optimizer.initialize_param(0.05)

    for epoch in range(max_iter+1):
        # Compute f, g, c, J
        f, f_interior, f_boundary, g = problem.objective_func_and_grad(optimizer, return_multiple_f=True)
        c, J = problem.constraint_func_and_grad(optimizer)

        # Update f, g, c, J to optimizer
        optimizer.state['f'] = f
        optimizer.state['f_interior'] = f_interior
        optimizer.state['f_boundary'] = f_boundary
        optimizer.state['g'] = g
        optimizer.state['c'] = c
        optimizer.state['J'] = J
        optimizer.state["f_g_hand"] = problem.objective_func_and_grad
        optimizer.state["c_J_hand"] = problem.constraint_func_and_grad

        # Take a step inside optimizer
        optimizer.step()

        # Print out
        optimizer.printerIteration(every=1)

        if np.mod(epoch,save_every) == 0:
            # path for saving trained NN
            path='mdl/nn_epoch%s_%s' %(epoch, problem_name)
            problem.save_net(path)
    

if __name__ == '__main__':
    ## Initialize optimizer
    problem_name = "DarcyMatrix"  # "Spring" #sys.argv[1]
    problem = eval(problem_name)(device, n_obj_sample = 10, n_constrs = 3, reg=1)
    
    optimizer = StochasticSQP(problem.net.parameters(),
                          lr= 0.5,
                          n_parameters = problem.n_parameters, 
                          n_constrs = problem.n_constrs,
                          merit_param_init = 1, 
                          ratio_param_init = 1,
                          step_opt= 2,
                          problem = problem,
                         )
    run(optimizer, problem,  max_iter = int(200), save_every=20)
