import torch
from stochasticsqp import *
from problem_spring import Spring
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np
torch.manual_seed(10000)
np.random.seed(10000)

def run(optimizer, problem, max_iter = 10000):

    optimizer.printerHeader()

    for epoch in range(max_iter):
        
        # Compute f, g, c, J
        f,g = problem.objective_func_and_grad(optimizer)
        c,J = problem.constraint_func_and_grad(optimizer)

        # Update f, g, c, J to optimizer
        optimizer.state['f'] = f
        optimizer.state['g'] = g
        optimizer.state['c'] = c
        optimizer.state['J'] = J

        # Take a step inside optimizer
        optimizer.step()

        # Print out
        optimizer.printerIteration(every=1)
    

if __name__ == '__main__':
    ## Initialize optimizer

    problem = Spring(device, n_obj_sample = 500, n_constrs = 10)

    optimizer = StochasticSQP(problem.net.parameters(),
                          lr=0.001,
                          n_parameters = problem.n_parameters, 
                          n_constrs = problem.n_constrs,
                          merit_param_init = 1, 
                          ratio_param_init = 1,
                         )
    
    run(optimizer, problem,  max_iter = 10)
