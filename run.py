import yaml
from solve import run 
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from utils import create_dir

plot = False
train = True
problems = ['burgers']  # Qi runs chemistry and burgers
#problems = ['spring', 'darcy']         # Christian runs spring and darcy
output_folder = '../resultburgerstest' # '../result0506chemistry'

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
    # 4: {'alpha_type':'adam',
    #     'is_constrained':0,
    #     'is_full_batch':0},
    # 5: {'alpha_type':'adam',
    #     'is_constrained':1,
    #     'is_full_batch':0},
    # 6: {'alpha_type':'c_adam',
    #     'is_constrained':1,
    #     'is_full_batch':0},
    }

lrs_all = {
  0:1e-3,
  1:5e-4,
  2:1e-4
}


lrs_darcy = {
  0:1e-2, 
  1:5e-3, 
  2:1e-3, 
  3:5e-4, 
  4:1e-4
  }

n_run = {
  0:0,
#   1:1,
#   2:2,
#   3:3,
#   4:4
}

plt.style.use("fast")

def main():

    if train == True:
        for batch_seed in n_run.values():
            for problem_name in problems:
                if problem_name == 'darcy':
                    lrs = lrs_darcy
                else:
                    lrs = lrs_all
                    
                for lr_i, lr in lrs.items():

                    for k,setting in settings.items():
                        print('Running batch_seed:%s, problem: %s, lr: %s, setting: %s' %(batch_seed, problem_name, lr, k))
                        # full batch setting only have one run
                        if k in [1,2,3] and batch_seed != 0:
                            continue
                        
                        with open('conf/conf_'+problem_name+'.yaml') as f:
                            config = yaml.load(f, Loader=yaml.FullLoader)
                        
                        config['batch_seed'] = batch_seed    
                        
                        config['n_epoch'] = 0
                        #config['save_model_every'] = 1
                        #config['save_plot_every'] = 1
                        
                        config['optimizer']['lr'] = lr
                    
                        config['optimizer']['alpha_type'] = setting['alpha_type']
                        
                        if setting['is_constrained'] == 0:
                            config['problem']['n_constrs'] = 0    
                        
                        if setting['is_full_batch'] == 1:
                            config['problem']['batch_size'] = 'full'    
                        
                        config['output_folder'] = output_folder
                        
                        config['file_suffix'] = '%ssetting%s_lr%scontd' %(problem_name,k,lr_i)
                        
                        config['optimizer']['pretrain'] = {
                            'epoch_start': 10000,
                            'file_suffix': '%ssetting%s_lr%s_0' %(problem_name,k,lr_i)
                        }
                        
                        run(config)
                
    if plot == True:
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
    
    if problem == 'darcy':
        lrs = lrs_darcy
    else:
        lrs = lrs_all
    
    batch_setting =['full batch'] #[ 'mini-batch', 'full batch']
    for lr_i, lr in lrs.items():
        for batch in batch_setting:

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
                    ax.plot(x, y_mean, label=k, marker = markers[k], markevery=int(0.1 * len(x)), markersize=5, alpha=0.9)
                    ax.fill_between(x, (y_mean - y_std), (y_mean + y_std), alpha=0.3)
                #plt.yscale('log')
                # log_tick_label = []
                # for t in ax.get_yticks():
                #     t = str(round(t,1))
                #     if t[-1] == '0':
                #         t = t[:-2]
                #     log_tick_label.append(r'$10^{%s}$' %t)
                # ax.set_yticks(ax.get_yticks(), log_tick_label)
                # plt.locator_params(axis='y',nbins=4)
                if problem == 'burgers':
                    y_ticks = [-4, -2, -0, 2]
                else:
                    #TODO
                    pass
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
