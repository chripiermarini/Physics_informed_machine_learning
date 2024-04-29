import yaml
from solve import run 
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

plot = True
train = False
#problems = ['spring', 'chemistry', 'darcy', 'burgersinf']
problems = ['spring']
output_folder = '../test0428'
n_run = 3

plt.style.use("fast")

def main():

    for problem_name in problems:
        
        if train == True:
            
            #lrs = [1e-3, 5e-4, 1e-4]
            lrs = [1e-3]
            if problem_name == 'darcy':
                lrs = [1e-2]#, 5e-3, 1e-3, 5e-4, 1e-4]
            
            for lr_i, lr in enumerate(lrs):
            
                settings = {
                    1: {'alpha_type':'adam',
                        'is_constrained':0,
                        'is_full_batch':1},
                    2: {'alpha_type':'adam',
                        'is_constrained':1,
                        'is_full_batch':1},
                    3: {'alpha_type':'c_adam',
                        'is_constrained':1,
                        'is_full_batch':1},
                    4: {'alpha_type':'adam',
                        'is_constrained':0,
                        'is_full_batch':0},
                    5: {'alpha_type':'adam',
                        'is_constrained':1,
                        'is_full_batch':0},
                    6: {'alpha_type':'c_adam',
                        'is_constrained':1,
                        'is_full_batch':0},
                    }
                
                for k,setting in settings.items():
                    with open('conf/conf_'+problem_name+'.yaml') as f:
                        config = yaml.load(f, Loader=yaml.FullLoader)
                    
                    config['n_run'] = 3    
                    # config['n_epoch'] = 10
                    # config['save_model_every'] = 5
                    # config['save_plot_every'] = 5
                    
                    config['optimizer']['lr'] = lr
                
                    config['optimizer']['alpha_type'] = setting['alpha_type']
                    
                    if setting['is_constrained'] == 0:
                        config['problem']['n_constrs'] = 0    
                    
                    if setting['is_full_batch'] == 1:
                        config['problem']['batch_size'] = 'full'    
                    
                    config['output_folder'] = output_folder
                    
                    config['file_suffix'] = '%ssetting%s_lr%s' %(problem_name,k,lr_i)
                    
                    run(config)
                
        if plot == True:
            plot_f(problem_name)


def find_header_row_number(file, first_header):
    with open(file, 'r') as f:
        i=0
        while line := f.readline():
            line = line.strip().split()
            if len(line)> 0 and line[0] == first_header:
                break
            else:
                i += 1
    return i

def plot_f(problem):
    
    lr = 0
    batch='batch'

    plot_columns = {'Residual loss': 'f_pde', 
                'Data-fitting loss': 'f_fitting'}    
    
    if problem == 'spring':
        log_folder = 'Spring'
    elif problem == 'darcy':
        log_folder = 'Darcy'
        plot_columns ['Boundary residual loss'] = 'f_boundary'
    elif problem == 'burgersinf':
        log_folder = 'Burgersinf'
        plot_columns ['Boundary residual loss'] = 'f_boundary'
    elif problem == 'chemistry':
        log_folder = 'Chemistry'
        plot_columns['Mass balance residual loss'] = 'f_boundary'
    
    log_dir = '%s/log/%s' %(output_folder, log_folder)
    
    if batch == 'full':
    
        files = {
                'Adam(unconstrained)-full'       : '%ssetting1_lr%s' %(problem, lr),
                'Adam(constrained)-full'       : '%ssetting2_lr%s' %(problem, lr),
                'P-Adam(constrained)-full'       : '%ssetting3_lr%s' %(problem, lr),
                }
    elif batch == 'batch':
            
        files = {
                'Adam(unconstrained)-batch'      : '%ssetting4_lr%s' %(problem,lr),
                'Adam(constrained)-batch'        : '%ssetting5_lr%s' %(problem,lr),
                'P-Adam(constrained)-batch'      : '%ssetting6_lr%s' %(problem,lr),
                }
        
    first_header = 'epoch'
    x_column_name = 'epoch'
    
    suffix = '%s_lr%s' %(batch,lr) 
    
    dfs = {}
    for k, file in files.items():
        dfs[k] = {}
        filenames = [filename for filename in os.listdir(log_dir) if filename.startswith(file)]
        for run_i, filename in enumerate(filenames):
            full_file = '%s/%s' %(log_dir, filename)
            i = find_header_row_number(full_file,first_header)
            df = pd.read_csv(full_file,skiprows=i, sep='\s+',)
            dfs[k][run_i] = df
    for loss_name,i in plot_columns.items():
        # plot the data
        fig = plt.figure(figsize=(4.2,3.2))
        ax = fig.add_subplot(1, 1, 1)
        for k, runs in dfs.items():
            x = runs[0][x_column_name]
            ys = np.zeros((len(runs), len(x)))
            for run_i, df in runs.items():
                ys[run_i] = df[i]
            ys = np.log10(ys)               # use log scale y
            y_mean = np.mean(ys,axis=0)
            y_std = np.std(ys, axis=0)
            ax.plot(x, y_mean, label=k)
            ax.fill_between(x, (y_mean - y_std), ((y_mean + y_std)), alpha=0.3)
        #plt.yscale('log')
        ax.set_yticks(ax.get_yticks(), [r'$10^{%s}$'%int(t) for t in ax.get_yticks()])
        plt.locator_params(axis='y',nbins=4)
        ax.set_title('%s %s' %(loss_name, suffix))
        ax.legend()
        plt.xlabel('Epoch')
        plt.tight_layout()
        # display the plot
        plt.savefig('%s/%s_%s_%s.png' %(log_dir, problem, i,suffix))
        plt.close()


if __name__ == '__main__':
    main()
    try:
      import google.colab
      IN_COLAB = True
    except:
      IN_COLAB = False
    if IN_COLAB:
      from google.colab import runtime
      runtime.unassign()