import yaml
from solve import run 
import pandas as pd
import matplotlib.pyplot as plt

plot = False
train = True
problems = ['chemistry', 'spring', 'darcy']
output_folder = './test2'

plt.style.use("fast")

def main():

    for problem_name in problems:
        
        if train == True:
        
            with open('conf/conf_'+problem_name+'.yaml') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            
            settings = {
                1: {},
                }
            
            for k,setting in settings.items():
                config['maxiter'] = 100
                config['save_model_every'] = 10
                config['save_plot_every'] = 10
                config['output_folder'] = output_folder
                config['file_suffix'] = '%s%s%s' %(problem_name, 'test', k)
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

    plot_columns = {'PDE loss': 'f_pde', 
                'Data fitting loss': 'f_fitting'}    
    
    if problem == 'spring':
        log_folder = 'Spring'
    elif problem == 'darcy':
        log_folder = 'Darcy'
        plot_columns ['Boundary'] = 'f_boundary'
    elif problem == 'chemistry':
        log_folder = 'Chemistry'
        plot_columns['Other MSE'] = 'f_boundary'
    
    log_dir = '%s/log/%s' %(output_folder, log_folder)
    files = {'method1'     : '%ssetting1.txt' %(problem),
            'method2'      : '%ssetting2.txt' %(problem),
            }
    first_header = 'epoch'
    x_column_name = 'epoch'
    
    dfs = {}
    for k, file in files.items():
        full_file = '%s/%s' %(log_dir, file)
        i = find_header_row_number(full_file,first_header)
        df = pd.read_csv(full_file,skiprows=i, sep='\s+',)
        dfs[k] = df
    for loss_name,i in plot_columns.items():
        # plot the data
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        for k, df in dfs.items():
            ax.plot(df[x_column_name], df[i], label=k)
        plt.yscale('log')
        ax.set_title(loss_name)
        ax.legend()
        plt.tight_layout()
        # display the plot
        plt.savefig('%s/%s_%s.png' %(log_dir, problem, i))
        plt.close()


if __name__ == '__main__':
    main()
