import torch
from stochasticsqp import *
from problems.problem_darcy_matrix_old import DarcyMatrixOld
from problems.problem_darcy_matrix import DarcyMatrix
from problems.problem_spring import Spring
from problems.problem_spring_new import SpringNew
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np
torch.set_default_device(device)
torch.manual_seed(22)
np.random.seed(22)
torch.manual_seed(123)
import sys
torch.set_printoptions(precision=8)
import matplotlib.pyplot as plt
import time
import os

MDL_DIR = './mdl'
PLOTS_DIR = './plots'
LOG_DIR = './log'

def create_dir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir) 

def get_mdl_path(epoch, suffix):
    path='%s/nn_%s_%s' %(MDL_DIR,epoch, suffix)
    return path

def get_optim_path(epoch, suffix):
    optim_path = '%s/optim_%s_%s.pt' %(MDL_DIR,epoch, suffix)
    return optim_path

def get_full_file_suffix(optimizer_name, problem_name, n_constrs, constraint_type, lr, mu, beta2, file_suffix):
    suffix='%s_%s_%s_%s_%s_%s_%s' %(optimizer_name, problem_name, n_constrs, constraint_type,
                                    lr, mu, beta2)
    if file_suffix is not None:
        suffix = '%s_%s' %(suffix, file_suffix)
    return suffix

def get_x(problem):
    res = []
    for name, param in problem.net.named_parameters():
        res.append(param.view(-1))
    res = torch.cat(res)
    return res

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


def run(optimizer_name, problem_name,  n_constrs, constraint_type,
        lr=1e-3, mu = 1e-7, beta2 = 0.999, 
        max_iter = 100000, 
        save_model_every=100, save_plot_every=100, 
        pretrain={},
        file_suffix=None,
        stdout = 0):     

    # Create directories if they do not exist
    create_dir(MDL_DIR)
    create_dir(PLOTS_DIR)
    create_dir(LOG_DIR)

    # Load problem instance
    problem = eval(problem_name)(device, n_obj_sample = 1, n_constrs = n_constrs, constraint_type=constraint_type, reg=1e-4)
    
    # Load optimizer
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(problem.net.parameters(),lr=lr)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(problem.net.parameters(),lr=lr)
    elif optimizer_name == 'sqp':
        optimizer = StochasticSQP(problem.net.parameters(),
                            lr= lr,
                            mu = mu,
                            beta2=beta2,
                            n_parameters = problem.n_parameters, 
                            n_constrs = problem.n_constrs,
                            merit_param_init = 1, 
                            ratio_param_init = 1,
                            step_opt= 2,
                            problem = problem,)
        
    suffix = get_full_file_suffix(optimizer_name, problem_name, n_constrs, constraint_type, lr, mu, beta2, file_suffix)
    
    # Log file IO
    if stdout == 1:
        log_file_name = '%s/%s.txt' %(LOG_DIR,suffix)
        log_f = open(log_file_name,'w')
    elif stdout == 0:
        log_f = None
        
    # reload the model and optimizer. Now only apply to sqp optimizer
    if pretrain != {} and optimizer_name == 'sqp':
        epoch_start = pretrain['epoch_start']
        pretrain_suffix = get_full_file_suffix(optimizer_name, problem_name, pretrain['n_constrs'], 
                                               pretrain['constraint_type'],pretrain['lr'],
                                               pretrain['mu'],pretrain['beta2'],pretrain['file_suffix'])
        mdl_path=get_mdl_path(epoch_start, pretrain_suffix)
        problem.load_net(mdl_path)
        optim_path = get_optim_path(epoch_start, pretrain_suffix)
        optimizer.load_pretrain_state(optim_path)
    else:
        epoch_start=0
        
    #optimizer.printerHeader()
    print('%11s: %10s' %('Problem',problem.name), file=log_f)
    print('%11s: %10s' %('Optimizer',optimizer_name), file=log_f)
    print('%11s: %10s' %('Pretrained?', len(pretrain) > 0 ) , file=log_f)
    print('%11s: %10s' %('Epoch start', epoch_start), file=log_f)
    print('%11s: %10s\n%11s: %10s\n%11s: %10s' %('lr',lr, 'mu',mu, 'beta2',beta2), file=log_f)
    print('-'*40, file=log_f)
    print('%5s %11s %11s %11s %11s %11s' %('epoch', 'f', 'f_interior', 'f_boundary', 'alpha_max', 'alpha_min'), file=log_f)

    #optimizer.initialize_param(0.1)

    #check_gradient(optimizer, problem)
    #x0 = get_x(problem)
    
    files = []
    
    # plot the initial predition
    u_pred = problem.net(problem.t_all).detach()
    problem.plot_result(epoch_start,problem.t_all,problem.u_all, u_pred, problem.t_fitting, problem.u_fitting,problem.t_pde.detach(), save_file=None)
    file = '%s/nn_%.8i_%s.png' %(PLOTS_DIR,epoch_start, suffix) 
    plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
    files.append(file)
    
    t_start = time.time()

    for epoch in range(epoch_start, epoch_start+max_iter+1):
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
        
        # if epoch == 0:
        #     x1 = get_x(problem)
        #     diff = (x1 - x0)*1000 
        #     with open('g_test_%s.txt' %(optimizer_name),'w') as f_g_test: 
        #         for i in diff:
        #             f_g_test.write(str(i.detach().numpy())+'\n')
                
        # Print out for SQP Optimizer
        #optimizer.printerIteration(every=100)

        # get max and min step size
        if optimizer_name == 'adam':
            alpha_adam_init = optimizer.param_groups[0]['lr']
            beta1_adam,beta2_adam = optimizer.param_groups[0]['betas']
            eps_adam = optimizer.param_groups[0]['eps']
            alphak = alpha_adam_init * np.sqrt(1-beta2_adam**(epoch+1)) / (1-beta1_adam**(epoch+1)) 
            vt = torch.tensor([])
            mt = torch.tensor([])
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    vt = torch.concat((vt,state['exp_avg_sq'].view(-1)))
                    mt = torch.concat((mt,state['exp_avg'].view(-1)))
            alpha_adam = alphak / (torch.sqrt(vt) + eps_adam)
            alpha_max = torch.max(alpha_adam)
            alpha_min = torch.min(alpha_adam)
            # x0 - x1 should be equal to alpha_max * mt. They have small difference now. 
        elif optimizer_name == 'sqp':
            alpha_sqp = optimizer.state['alpha_sqp'] / optimizer.state['H_diag']
            alpha_max = torch.max(alpha_sqp)
            alpha_min = torch.min(alpha_sqp)

        # Save model and optimizer parameters
        if np.mod(epoch+1-epoch_start,save_model_every) == 0:
            # path for saving trained NN
            mdl_path=get_mdl_path(epoch+1, suffix)
            problem.save_net(mdl_path)
            
            # path for saving optimizer state
            if optimizer_name == 'sqp':
                optim_path = get_optim_path(epoch+1, suffix)
                optimizer.save_pretrain_state(optim_path)
                
        if np.mod(epoch-epoch_start,save_model_every) == 0:
            # Printout
            print('%5s %11.4e %11.4e %11.4e %11.4e %11.4e ' %(epoch, f, f_interior, f_boundary, alpha_max, alpha_min),file=log_f)
                
        # plot the result as training progresses
        if np.mod(epoch-epoch_start,save_plot_every) == 0:
            u_pred = problem.net(problem.t_all).detach()
            problem.plot_result(epoch+1,problem.t_all,problem.u_all, u_pred, problem.t_fitting, problem.u_fitting,problem.t_pde.detach(), save_file=None)
            file = '%s/nn_%.8i_%s.png' %(PLOTS_DIR,epoch+1, suffix) 
            plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
            files.append(file)
            plt.close("all")
    t_end = time.time() - t_start
    print('Running time: %s' %(t_end), file=log_f)
    problem.save_gif_PIL("%s/pinn_%s_%s.gif" %(PLOTS_DIR,max_iter,suffix), files, fps=20, loop=0)

def evaluate(problem,epoch):
    #u_true = problem.get_u_true(
    u_pred = problem.net(problem.domain_interior_tensor)
    u_pred = u_pred.reshape(-1)
    u_pred = u_pred.detach().numpy()
    err = np.linalg.norm(u_true - u_pred,2)
    t_np = problem.domain_interior_tensor[:,1].detach().numpy()
    if epoch == 200:
        plot(u_true, u_pred, t_np, '%s_%s.png' %(problem.name, epoch))
    print(err)
    

if __name__ == '__main__':

    problem_name = "SpringNew"  
    optimizer_name = 'sqp'          # adam or sgd or sqp 
    n_constrs = 0
    constraint_type='pde'
    lr    = 1e-3
    mu    = 1e-7
    beta2 = 0.999
    file_suffix="type1"             # None or a str, like "type1", etc.
    stdout = 1                      # 0: print to screen, 1: print to a .log/ directory
    maxiter=100000
    save_model_every=100
    save_plot_every=1000             # maxiter / save_plot_every better not exceed 500
    
    pretrain = {                        
        'epoch_start':10000,          
        'n_constrs'  :0,
        'constraint_type':'pde',
        'lr'         :1e-3,
        'mu'         :1e-7,
        'beta2'      :0.999,
        'file_suffix':'type1',
    }
    
    pretrain = {}
        
    # train
    run(optimizer_name, problem_name,  n_constrs, constraint_type,
        lr=lr, mu = mu, beta2 = beta2, 
        max_iter = int(maxiter), 
        save_model_every=save_model_every, save_plot_every=save_plot_every, 
        pretrain=pretrain, file_suffix=file_suffix, stdout = stdout)
    
    
