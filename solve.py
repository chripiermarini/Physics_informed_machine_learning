import torch
from stochasticsqp import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np
torch.manual_seed(10000)
np.random.seed(10000)
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

        # Take a step inside optimizer
        optimizer.step()

        # Print out
        optimizer.printerIteration(every=1)
    

if __name__ == '__main__':
    ## Initialize optimizer

    problem_name = sys.argv[1]

    problem = all_problems[problem_name](device, n_obj_sample = 100, n_constrs = 1)

    optimizer = StochasticSQP(problem.net.parameters(),
                          lr=1.0,
                          n_parameters = problem.n_parameters, 
                          n_constrs = problem.n_constrs,
                          merit_param_init = 1, 
                          ratio_param_init = 1,
                         )
    
    run(optimizer, problem,  max_iter = 100)
