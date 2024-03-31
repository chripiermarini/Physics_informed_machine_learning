import os
import torch
import matplotlib.pyplot as plt
import numpy as np

def create_dir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir) 

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
    