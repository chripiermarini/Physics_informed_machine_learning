import torch
from stochasticsqp import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np
torch.manual_seed(28)
np.random.seed(28)
import sys

# Import all problems fro directory `problems`
import os
import importlib.util
def import_all_classes_from_directory(directory):
    classes = {}
    for filename in os.listdir(directory):
        if filename.endswith('.py'):
            module_name = filename[:-3]  # Remove '.py' to get the module name
            module_path = os.path.join(directory, filename)
            
            # Import the module
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Iterate through attributes of the module
            for attribute_name in dir(module):
                attribute = getattr(module, attribute_name)
                if isinstance(attribute, type):  # Check if it's a class
                    classes[attribute_name] = attribute
    return classes 
directory_path = './problems'
all_problems = import_all_classes_from_directory(directory_path)
# Now all_problems is a dictionary where keys are names of problem classes and values are problem objects


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
        optimizer.state["f_g_hand"]= problem.objective_func_and_grad
        optimizer.state["c_J_hand"] = problem.constraint_func_and_grad

        # Take a step inside optimizer
        optimizer.step()

        # Print out
        optimizer.printerIteration(every=1)
    

if __name__ == '__main__':
    ## Initialize optimizer

    problem_name = "DarcyMatrix" #"Spring" #sys.argv[1]

    problem = all_problems[problem_name](device, n_obj_sample = 1000, n_constrs = 30)    
    #print(problem.input[problem.constr_pixel_idx[:,0],1:3,problem.constr_pixel_idx[:,1],problem.constr_pixel_idx[:,2]])

    optimizer = StochasticSQP(problem.net.parameters(),
                          lr= 0.1,
                          n_parameters = problem.n_parameters, 
                          n_constrs = problem.n_constrs,
                          merit_param_init = 1, 
                          ratio_param_init = 1,
                          step_opt= 2,
                          problem = problem
                         )
    #f,g = problem.objective_func_and_grad(optimizer)
    #c,J = problem.constraint_func_and_grad(optimizer)
    run(optimizer, problem,  max_iter = int(1e4))
