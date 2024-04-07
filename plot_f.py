import pandas as pd
import matplotlib.pyplot as plt


log_dir = 'results/log/SpringFormal'
files = {'Adam'     : 'adam.txt', 
         'Adagrad'  : 'ada.txt', 
         'PAdam'    : 'c_adam.txt', 
         'PAdagrad' : 'c_ada.txt'}
first_header = 'epoch'
plot_columns = {'PDE loss': 'f_pde', 
                'Data fitting loss': 'f_fitting'}
x_column_name = 'epoch'

plt.style.use("fast")

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

def plot():
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
        plt.savefig('%s/%s.png' %(log_dir, i))


if __name__ == '__main__':
    plot()
