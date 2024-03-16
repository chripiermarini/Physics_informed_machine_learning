import torch
from stochasticsqp import *
from problems.problem_darcy_matrix_old import DarcyMatrixOld
from problems.problem_darcy_matrix import DarcyMatrix
from problems.problem_spring import Spring
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np
torch.set_default_device(device)
torch.manual_seed(22)
np.random.seed(22)
import sys
torch.set_printoptions(precision=8)
import matplotlib.pyplot as plt

def plot(u_true, u_pred, t, save_file_name):
    # Data for plotting
 
    fig, (ax0, ax1) = plt.subplots(2, 1, layout='constrained')
    ax0.plot(t, u_true)
    ax0.set_ylabel('u_true')
    ax1.plot(t, u_pred)
    ax1.set_ylabel('u_pred')
    fig.savefig(save_file_name)


def check_gradient(optimizer, problem):
    
    f, g_ori = problem.objective_func_and_grad(optimizer)
    
    max_abs_diff = 0
    for name, param in problem.net.named_parameters():
        g_param = param.grad.view(-1)
        for i in range(len(param.view(-1))):
            
            #print('Before-------')
            #for name_cur, param_cur in problem.net.named_parameters():
            #    print(param_cur.data)
            
            param.view(-1).data[i] += 1e-4
            
            #print('After-------')
            #for name_cur, param_cur in problem.net.named_parameters():
            #    print(param_cur.data)
                
            f_i,g = problem.objective_func_and_grad(optimizer)
            d_i = (f_i - f)/1e-4
            g_i = g_param.data[i]
            abs_diff = abs(d_i - g_i)
            max_abs_diff = max(max_abs_diff, abs_diff)
            re_diff =  abs(d_i - g_i) / max(abs(d_i), abs(g_i))
            msg = ''
            if re_diff > 1e-3:
                msg = 'large error'
            print(d_i, g_i, abs_diff, re_diff, msg)
            param.view(-1).data[i] -= 1e-4
    print(max_abs_diff)

def run(optimizer, problem, max_iter = 10000, save_every=10):
    optimizer.printerHeader()

    optimizer.initialize_param(0.1)

    #check_gradient(optimizer, problem)

    for epoch in range(max_iter+1):
        # Compute f, g, c, J
        f,f_interior,f_boundary, g = problem.objective_func_and_grad(optimizer,return_multiple_f = True)
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
        evaluate(problem, epoch)


def evaluate(problem,epoch):
    u_true = problem.get_u_true(problem.domain_interior)
    u_pred = problem.net(problem.domain_interior_tensor)
    u_pred = u_pred.reshape(-1)
    u_pred = u_pred.detach().numpy()
    err = np.linalg.norm(u_true - u_pred,2)
    t_np = problem.domain_interior_tensor[:,1].detach().numpy()
    if epoch == 200:
        plot(u_true, u_pred, t_np, '%s_%s.png' %(problem.name, epoch))
    print(err)

if __name__ == '__main__':
    ## Initialize optimizer
    problem_name = "Spring"  # "Spring" #sys.argv[1]
    problem = eval(problem_name)(device, n_obj_sample = 1, n_constrs = 0, constraint_type='boundary')
    
    optimizer = StochasticSQP(problem.net.parameters(),
                          lr= 0.5,
                          n_parameters = problem.n_parameters, 
                          n_constrs = problem.n_constrs,
                          merit_param_init = 1, 
                          ratio_param_init = 1,
                          step_opt= 2,
                          problem = problem,
                         )
    
    run(optimizer, problem,  max_iter = int(200), save_every=1)
    
    
