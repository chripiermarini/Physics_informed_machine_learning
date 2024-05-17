import torch
from stochasticsqp import *
from problems.problem_darcy import Darcy
from problems.problem_spring import Spring
from problems.problem_burgers import Burgers
from problems.problem_chemistry import Chemistry
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np
torch.set_default_device(DEVICE)
import sys
torch.set_printoptions(precision=8)
import time
import yaml
from utils import create_dir

def create_output_folders(output_folder, sub_folders, problem_name):
    create_dir(output_folder)
    folders = {}
    for k,v in sub_folders.items():
        sub_dir = output_folder + '/' + v 
        create_dir(sub_dir)
        sub_sub_dir = output_folder + '/' + v + '/' + problem_name
        create_dir(sub_sub_dir)
        folders[k] = sub_sub_dir
    return folders
    

def get_mdl_path(folders, epoch, suffix):
    path='%s/nn_%s_%s' %(folders['model'], suffix, epoch)
    return path

def get_optim_path(folders, epoch, suffix):
    optim_path = '%s/optim_%s_%s.pt' %(folders['model'],suffix, epoch)
    return optim_path

def get_plot_path(folders, epoch, suffix):
    path = '%s/plot_%s_%.8i.png' %(folders['plot'], suffix, epoch) 
    return path

def get_gif_path(folders, suffix):
    path = '%s/animation_%s.gif' %(folders['plot'],suffix)
    return path
    
def printRow(log_f, type='header', headers=[],values={}):
    for i, ele in enumerate(headers):
        if ele == 'epoch':
            if type == 'header':
                p_format = '{:>7s}'
            elif type == 'value':
                p_format = '{:7d}'
        elif ele == 'iter':
            if type == 'header':
                p_format = '{:>7s}'
            elif type == 'value':
                p_format = '{:7d}'
        elif ele == 'elapse':
            if type == 'header':
                p_format = '{:>12s}'
            elif type == 'value':
                p_format = '{:12d}'
        else:
            if type == 'header':
                p_format = '{:>12s}'
            elif type == 'value':
                p_format = '{:12.4e}'
                
        if type == 'header':
            value = headers[i]
        elif type == 'value':
            value = values[headers[i]]
            
        # Print
        if i == len(headers) - 1:
            print(p_format.format(value), sep=' ', file=log_f)
        else:
            print(p_format.format(value), sep=' ', end = '', file=log_f)

def printerBeginningSummary(config, log_f):
    print('-'*60, file=log_f)
    print(yaml.dump(config), file=log_f)
    headers = ['epoch', 'iter', 'f', 'f_pde', 'f_boundary', 'f_fitting',
                '||c||inf', '||c||1' ,'elapse']
    if config['optimizer']['name'] == 'sqp':
        headers = headers + ['H_max', 'H_min', 'merit_f', 'alpha', 'tau']
    print('-'*60, file=log_f)
    printRow(log_f, type='header', headers=headers)
    return headers

def save_model(folders, epoch, problem, optimizer, config):
    # path for saving trained NN
    mdl_path=get_mdl_path(folders, epoch, config['file_suffix'])
    problem.save_net(mdl_path)
    # path for saving optimizer state
    if config['optimizer']['name'] == 'sqp':
        optim_path = get_optim_path(folders,epoch, config['file_suffix'])
        optimizer.save_pretrain_state(optim_path)

def plot_prediction(folders, epoch, problem, config):
    file = get_plot_path(folders, epoch, config['file_suffix'])
    if problem.name in ['Darcy', 'Burgers', 'Chemistry'] and epoch == 0:
        problem.plot_prediction(save_label = True, save_path = file.replace('plot_', 'label_'))    
    problem.plot_prediction(save_path = file, epoch=epoch)
    return file

def plot_gif(folders, problem, config, files):
    gif_path = get_gif_path(folders, config['file_suffix'])
    problem.save_gif_PIL(gif_path, files, fps=20, loop=0)

def run(config):     
    if config['problem']['batch_size'] == 'full':# 'full' or percentage of sample 
        batch_seed = 0
    else:
        batch_seed = config['batch_seed'] 
    file_suffix = config['file_suffix']
    
    np.random.seed(22)
    torch.manual_seed(123)

    # Adjust file_suffix by batch_seed
    config['file_suffix'] = file_suffix + '_' + str(batch_seed)

    # Create output folders
    folders = create_output_folders(config['output_folder'],config['sub_folders'],config['problem']['name'])

    # Load problem instance
    problem = eval(config['problem']['name'])(DEVICE,config['problem'])
    
    # Set_torch_random_seed(self, seed)
    problem.set_torch_random_seed(batch_seed)
    
    # Add problem number of parameters to config
    config['problem']['n_parameters'] = problem.n_parameters

    # Load optimizer
    if config['optimizer']['name'] == 'adam':
        optimizer = torch.optim.Adam(problem.net.parameters(),lr=config['optimizer']['lr'])
    elif config['optimizer']['name'] == 'sgd':
        optimizer = torch.optim.SGD(problem.net.parameters(),lr=config['optimizer']['lr'])
    elif config['optimizer']['name'] == 'sqp':
        optimizer = StochasticSQP(problem.net.parameters(),
                            lr = config['optimizer']['lr'],
                            config = config['optimizer'],
                            n_parameters = problem.n_parameters, 
                            n_constrs = problem.n_constrs,
                            merit_param_init = 1, 
                            ratio_param_init = 1,
                            step_opt= 2,
                            problem = problem,)
    
    # Open log file IO
    if config['stdout'] == 1:
        log_file_name = '%s/%s.txt' %(folders['log'],config['file_suffix'])
        log_f = open(log_file_name,'w')
    elif config['stdout'] == 0:
        log_f = None


    # reload the model and optimizer. Now only apply to sqp optimizer
    if (config['optimizer']['pretrain'] is not None) and (config['optimizer']['name'] == 'sqp'):
        epoch_start = config['optimizer']['pretrain']['epoch_start']
        pretrain_suffix = config['optimizer']['pretrain']['file_suffix']
        mdl_path=get_mdl_path(folders, epoch_start, pretrain_suffix)
        problem.load_net(mdl_path,DEVICE)
        optim_path = get_optim_path(folders, epoch_start, pretrain_suffix)
        optimizer.load_pretrain_state(optim_path,DEVICE)
        iter = optimizer.state['iter']
    else:
        epoch_start=0
        iter = 0

        
    #printer header
    headers = printerBeginningSummary(config, log_f)
    values = {k:-1 for k in headers}

    
    files = []
    
    # plot the initial prediction
    file = plot_prediction(folders, epoch_start, problem, config)
    files.append(file)
    
    # Set starting time
    t_start = time.time()

    # Main Loop
    for epoch in range(epoch_start, epoch_start+config['n_epoch']+1):

        batch_size = config['problem']['batch_size'] # 'full' or percentage of sample 
        if batch_size == 'full':
            n_batch = 1
        else:
            n_batch = int(1/batch_size)
        
        # Add printer values    
        values['epoch'] = epoch
            
        for batch_i in range(n_batch):

            # Compute f, g, c, J
            fs, g = problem.objective_func_and_grad(optimizer)
            c, J = problem.constraint_func_and_grad(optimizer)
            
            # Add printer values
            values['iter'] = iter
            values['f'] = fs['f'].data
            values['f_pde'] = fs['pde'].data
            values['f_boundary'] = fs['boundary'].data
            values['f_fitting'] = fs['fitting'].data
            if config['problem']['n_constrs'] > 0:
                values['||c||inf'] = torch.norm(c,p=float('inf'))
            else:
                values['||c||inf'] = torch.tensor(0.)
            values['||c||1'] = torch.norm(c,p=1)
            
            # Update f, g, c, J to optimizer
            optimizer.state['f'] = fs['f'].data
            optimizer.state['g'] = g
            optimizer.state['c'] = c
            optimizer.state['J'] = J
            optimizer.state["f_g_hand"] = problem.objective_func_and_grad
            optimizer.state["c_J_hand"] = problem.constraint_func_and_grad

            # Take a step inside optimizer
            optimizer.step()

            # get max and min step size
            if config['optimizer']['name'] == 'adam':
                values['alpha'] = optimizer.param_groups[0]['lr']
                beta1_adam,beta2_adam = optimizer.param_groups[0]['betas']
                eps_adam = optimizer.param_groups[0]['eps']
                H = np.sqrt(1-beta2_adam**(iter+1)) / (1-beta1_adam**(iter+1)) 
                vt = torch.tensor([])
                mt = torch.tensor([])
                for group in optimizer.param_groups:
                    for p in group['params']:
                        state = optimizer.state[p]
                        vt = torch.concat((vt,state['exp_avg_sq'].view(-1)))
                        mt = torch.concat((mt,state['exp_avg'].view(-1)))
                H = H / (torch.sqrt(vt) + eps_adam)
                values['H_max'] = torch.max(H)
                values['H_min'] = torch.min(H) 
            elif config['optimizer']['name'] == 'sqp':
                values['alpha'] = optimizer.state['alpha_sqp']
                values['merit_f'] = optimizer.state['cur_merit_f']
                values['tau'] = optimizer.state['merit_param']
                values['H_max'] = torch.max(optimizer.state['H_diag'])
                values['H_min'] = torch.min(optimizer.state['H_diag'])
        
            # Print Initial Information
            if epoch==epoch_start and batch_i == 0:
                printRow(log_f,type='value',headers=headers,values=values)
        
            iter += 1
            
        # Add time elapse
        t_end = time.time() - t_start
        values['elapse'] = int(t_end)

        # Save model and optimizer parameters
        if np.mod(epoch+1-epoch_start,config['save_plot_model_every']) == 0:
            save_model(folders, epoch+1, problem, optimizer, config)

        # Print Iteration Information   
        if np.mod(epoch-epoch_start,config['save_loss_every']) == 0:
            printRow(log_f,type='value',headers=headers,values=values)

        # Plot prediction
        if np.mod(epoch-epoch_start,config['save_plot_model_every']) == 0:
            file = plot_prediction(folders, epoch+1, problem, config)
            files.append(file)         
            
    # Plot GIF
    plot_gif(folders, problem, config, files)

    # Close file IO
    if config['stdout'] == 1:
        log_f.close()

if __name__ == '__main__':

    conf_name = sys.argv[1] 
    with open('conf/'+conf_name+'.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # train
    run(config)
