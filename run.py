import yaml
from solve import run 
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from utils import create_dir


# --------------------- Modify settings below if required --------------------------------------
debug = False                                             # True or False. If it is True, then only a few epochs will be run.
plot = True                                               # Generate plots of losses over epochs using results from `output_folder`. The generated plots will be saved under `output_folder/loss_plots`
train = True                                             # True or False. If it is True, then the model will be trained and produce output and saved
                                                          #   in `output_folder`
problems = ['spring', 'chemistry', 'burgers', 'darcy']    # list of problems
output_folder = 'results'                                 # Output folder, under current directory

# Dict of settings. 1: full batch, adam(unc); 2: full batch, adam(con); 3: full batch, P-adam(con); 
#                   4: mini-batch, adam(unc); 5: mini-batch, adam(con); 6: mini-batch, P-adam(con); 
settings = {                                              
    1: {'alpha_type':'adam',
        'is_constrained':0,
        'is_full_batch':1},
    2: {'alpha_type':'adam',
        'is_constrained':1,
        'is_full_batch':1},
    3: {'alpha_type':'p_adam',
        'is_constrained':1,
        'is_full_batch':1},
    4: {'alpha_type':'adam',
        'is_constrained':0,
        'is_full_batch':0},
    5: {'alpha_type':'adam',
        'is_constrained':1,
        'is_full_batch':0},
    6: {'alpha_type':'p_adam',
        'is_constrained':1,
        'is_full_batch':0},
    }

# Dict of learning rates. Keys are indices and values are learning rate values.
lrs = {
  1:1e-3,
  2:5e-4,
}

# Dict of batch seeds. Key are indices and values are random seed for problem which distinguishes mini-batch generation
batch_seeds = {
  0:0,
  1:1,
  2:2,
  3:3,
  4:4
}

# Batch type for generating loss plots.
plot_batch_settings =['mini-batch', 'full batch']

# --------------------------------------------------------------------------------------------------------

plt.style.use("fast")

def main():

    if train == True:
        
        # loop over batch_seed, problems, learning rate, and setting to run all the experiments.
        
        for batch_seed in batch_seeds.values():
            for problem_name in problems:
                for lr_i, lr in lrs.items():

                    for k,setting in settings.items():
                        print('Running batch_seed:%s, problem: %s, lr: %s, setting: %s' %(batch_seed, problem_name, lr, k))
                        # full batch setting only have one run
                        if k in [1,2,3] and batch_seed != 0:
                            continue
                        
                        # Load default configuration of each problem
                        with open('conf/conf_'+problem_name+'.yaml') as f:
                            config = yaml.load(f, Loader=yaml.FullLoader)
                        
                        # Assign several configuration parameters
                        config['batch_seed'] = batch_seed    

                        if problem_name == 'chemistry' and k in [1,2,3]:   # full batch for chemistry, run 100000 epochs
                            config['n_epoch'] = 100000
                        elif problem_name == 'darcy' and k in [1,2,3]:   # full batch for darcy, run 4000 epochs
                            config['n_epoch'] = 4000
                        
                        config['optimizer']['lr'] = lr
                    
                        config['optimizer']['alpha_type'] = setting['alpha_type']
                        
                        if setting['is_constrained'] == 0:
                            config['problem']['n_constrs'] = 0    
                        
                        if setting['is_full_batch'] == 1:
                            config['problem']['batch_size'] = 'full'    
                        
                        if debug:
                            config['n_epoch'] = 100
                            config['save_loss_every'] = 10
                            config['save_plot_model_every'] = 50

                        config['output_folder'] = output_folder
                        
                        config['file_suffix'] = '%ssetting%s_lr%s' %(problem_name,k,lr_i)
                        
                        # config['optimizer']['pretrain'] = {
                        #     'epoch_start': 10000,
                        #     'file_suffix': '%ssetting%s_lr%s_0' %(problem_name,k,lr_i)
                        # }
                        
                        run(config)
                
    if plot == True:
        # Generate plots of losses over epochs
        for problem_name in problems:
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
    
    create_dir('%s/loss_plots' %(output_folder))
    
    for lr_i, lr in lrs.items():
        for batch in plot_batch_settings:

            plot_columns = {'Residual loss': 'f_pde', 
                        'Data-fitting loss': 'f_fitting',
                        'Loss': 'f'}    
            
            if problem == 'spring':
                log_folder = 'Spring'
            elif problem == 'darcy':
                log_folder = 'Darcy'
                plot_columns ['Boundary residual loss'] = 'f_boundary'
            elif problem == 'burgers':
                log_folder = 'Burgers'
                plot_columns ['Boundary residual loss'] = 'f_boundary'
            elif problem == 'chemistry':
                log_folder = 'Chemistry'
                plot_columns['Mass balance residual loss'] = 'f_boundary'
            
            log_dir = '%s/log/%s' %(output_folder, log_folder)
            
            if batch == 'full batch':
            
                files = {
                        'Adam(unc)'       : '%ssetting1_lr%s' %(problem, lr_i),
                        'Adam(con)'       : '%ssetting2_lr%s' %(problem, lr_i),
                        'P-Adam(con)'       : '%ssetting3_lr%s' %(problem, lr_i),
                        }
            elif batch == 'mini-batch':
                    
                files = {
                        'Adam(unc)'      : '%ssetting4_lr%s' %(problem,lr_i),
                        'Adam(con)'        : '%ssetting5_lr%s' %(problem,lr_i),
                        'P-Adam(con)'      : '%ssetting6_lr%s' %(problem,lr_i),
                        }
            
            markers = {
                'Adam(unc)': 's', 
                'Adam(con)': 'o',
                'P-Adam(con)': 'v'
            }
                
            first_header = 'epoch'
            x_column_name = 'epoch'
            
            suffix = '%s_lr%s' %(batch,lr) 
            
            dfs = {}
            for k, file in files.items():
                dfs[k] = {}
                filenames = [filename for filename in os.listdir(log_dir) if filename.startswith(file+'_')]
                for run_i, filename in enumerate(filenames):
                    full_file = '%s/%s' %(log_dir, filename)
                    i = find_header_row_number(full_file,first_header)
                    df = pd.read_csv(full_file,skiprows=i, sep=r'\s+',)
                    dfs[k][run_i] = df
            for loss_name,i in plot_columns.items():
                # plot the data
                fig = plt.figure(figsize=(4.2,3.2))
                ax = fig.add_subplot(1, 1, 1)
                
                for k, runs in dfs.items():
                    x = runs[0][x_column_name]
                    ys = np.zeros((len(runs), len(x)))
                    # Gather data for all batch_seed if there are more than one.
                    for run_i, df in runs.items():
                        ys[run_i] = df[i]
                    ys = np.log10(ys)               # use log scale y
                    y_mean = np.mean(ys,axis=0)
                    y_std = np.std(ys, axis=0)
                    ax.plot(x, y_mean, label=k, marker = markers[k], markevery=int(0.1 * len(x)), markersize=5, alpha=0.9)
                    # Plot mean and error bar over area between mean +- std
                    ax.fill_between(x, (y_mean - y_std), (y_mean + y_std), alpha=0.3)

                # Set yticks labels
                if problem == 'burgers':
                    y_ticks = [-4, -2, 0, 2]
                elif problem == 'chemistry':
                    y_ticks = [-4, -2, 0, 2, 4]
                elif problem == 'spring':
                    y_ticks = [-4, -2, 0]
                    ax.set_ylim([-4.5, 0.5])
                elif problem == 'darcy':
                    y_ticks = [-2, -1, 0, 1]
                ax.set_yticks(y_ticks)
                ax.set_yticklabels([r'$10^{%s}$'%(t) for t in y_ticks ])
                ax.set_title('%s (%s, $%s$)' %(loss_name, batch, lr))
                ax.legend()
                plt.xlabel('Epoch')
                plt.tight_layout()
                # display the plot
                plt.savefig('%s/loss_plots/%s_%s_%s.png' %(output_folder, problem, i,suffix))
                plt.close()


if __name__ == '__main__':
    main()
    try:
      import google.colab
      IN_COLAB = True
    except:
      IN_COLAB = False
    #if IN_COLAB:
    #  from google.colab import runtime
    #  runtime.unassign()
