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

def run(optimizer_name, problem, max_iter = 10000, save_every=10, save_plot_every=100, lr=1e-3, mu = 100, beta2=0.999):
    
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
    
    #optimizer.printerHeader()
    print('%10s: %10s\n%10s: %10s' %('Problem',problem.name, 'Optimizer',optimizer_name))
    print('%10s: %10s\n%10s: %10s\n%10s: %10s' %('lr',lr, 'mu',mu, 'beta',beta2))
    print('-'*40)
    print('%4s %11s %11s %11s ' %('epoch', 'f', 'f_interior', 'f_boundary'))

    #optimizer.initialize_param(0.1)

    #check_gradient(optimizer, problem)
    x0 = get_x(problem)
    
    files = []
    
    # plot the result as training progresses
    u_pred = problem.net(problem.t_all).detach()
    problem.plot_result(0,problem.t_all,problem.u_all, u_pred, problem.t_fitting, problem.u_fitting,problem.t_pde.detach(), save_file=None)
    file = 'plots/nn_%.8i_%s_%s.png' %(0, problem.name, optimizer_name) 
    plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
    files.append(file)
    
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
        
        if epoch == 0:
            x1 = get_x(problem)
            diff = (x1 - x0)*1000 
            with open('g_test_%s.txt' %(optimizer_name),'w') as f_g_test: 
                for i in diff:
                    f_g_test.write(str(i.detach().numpy())+'\n')
                
        # Print out
        #optimizer.printerIteration(every=100)

        if np.mod(epoch,save_every) == 0:
            # path for saving trained NN
            print('%4s %11.4e %11.4e %11.4e ' %(epoch, f, f_interior, f_boundary))
            # path='mdl/nn_epoch%s_%s_%s' %(epoch, problem.name, optimizer_name)
            # problem.save_net(path)
            # evaluate_spring_new(problem, epoch)
            # u_pred = problem.net(problem.t_all).detach()
            # problem.plot_result(epoch,problem.t_all,problem.u_all, u_pred, problem.t_fitting, problem.u_fitting,problem.t_pde.detach(), save_file='plots/nn_epoch%s_%s_%s' %(epoch, problem.name, optimizer_name))
            
        # plot the result as training progresses
        if np.mod(epoch,save_plot_every) == 0:
            u_pred = problem.net(problem.t_all).detach()
            problem.plot_result(epoch+1,problem.t_all,problem.u_all, u_pred, problem.t_fitting, problem.u_fitting,problem.t_pde.detach(), save_file=None)
            file = 'plots/nn_%.8i_%s_%s.png' %(epoch+1, problem.name, optimizer_name) 
            plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
            files.append(file)
            plt.close("all")
    problem.save_gif_PIL("plots/pinn_%s_%s.gif" %(problem.name, optimizer_name), files, fps=20, loop=0)

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
    
    
def evaluate_spring_new(problem,epoch):
    
    u_pred = u_pred.reshape(-1)
    u_pred = u_pred.detach().numpy()
    u_true = problem.u_all.reshape(-1).detach().numpy()
    err = np.linalg.norm(u_true - u_pred,ord=np.inf)
    print(err)

if __name__ == '__main__':
    ## Initialize optimizer
    problem_name = "SpringNew"  # "Spring" #sys.argv[1]
    problem = eval(problem_name)(device, n_obj_sample = 1, n_constrs = 0, constraint_type='pde', reg=1e-4)

    optimizer_name = sys.argv[1] #'sqp'  # adam or sgd or sqp

    # Try different lr, mu, beta2    
    lr = 1e-1
    mu = 100
    beta2 = 0.999
    run(optimizer_name, problem,  lr=lr, mu = mu, beta2 = beta2, max_iter = int(100000), save_every=100, save_plot_every=1000)
    
    
