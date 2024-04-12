"""
This file was built to solve numerically 1D Burgers' equation wave equation with the FFT. The equation corresponds to :

$\dfrac{\partial u}{\partial t} + \mu u\dfrac{\partial u}{\partial x} = \nu \dfrac{\partial^2 u}{\partial x^2}$

where
 - u represent the signal
 - x represent the position
 - t represent the time
 - nu and mu are constants to balance the non-linear and diffusion terms.

Copyright - © SACHA BINDER - 2021
"""

############## MODULES IMPORTATION ###############
import numpy as np
import torch
from scipy.integrate import odeint
import random
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', None)

def initial_condition(x):

    # smoout Gaussian condition
    # A = 1.0  # Amplitude
    # x0 = 0  # Center of the Gaussian
    # sigma = 0.1  # Standard deviation

    # Initial condition
    # u_initial = A * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

    # sinusoidal initial condition
    # Parameters
    N = 2

    array = x
    result = np.zeros_like(array)
    for wave_index in range(N):
        amp = random.uniform(0, 1)
        phase = random.uniform(0, 2 * np.pi)
        freq = random.uniform(1, 4 / np.pi)

        result += amp * np.sin(2 * np.pi * freq * array + phase)
    return result
def true_burgers_solution(X, T):
    ############## SET-UP THE PROBLEM ###############

    mu = 0.5 #the 1/2 in Burgers' PDE
    nu = 0.01  # kinematic viscosity coefficient

    dx = (X[-1] -X[0])/(X.shape[0])
    # Wave number discretization
    k = 2 * np.pi * np.fft.fftfreq(X.shape[0], d=dx)

    u0 = initial_condition(X)

    ############## EQUATION SOLVING ###############

    # Definition of ODE system (PDE ---(FFT)---> ODE system)
    def burg_system(u, t, k, mu, nu):
        # Spatial derivative in the Fourier domain
        u_hat = np.fft.fft(u)
        u_hat_x = 1j * k * u_hat
        u_hat_xx = -k ** 2 * u_hat

        # Switching in the spatial domain
        u_x = np.fft.ifft(u_hat_x)
        u_xx = np.fft.ifft(u_hat_xx)

        # ODE resolution
        u_t = -mu * u * u_x + nu * u_xx
        return u_t.real

    # PDE resolution (ODE system resolution)
    U = odeint(burg_system, u0, T, args=(k, mu, nu,))
    return U

def create_burgers_dataset(n_points, test = False):
    x = np.linspace(0, 1, n_points)
    t = np.linspace(0, 1, n_points)
    u_zero = initial_condition(x)

    #PDE dataset
    # structure: x | t | u_zero
    meshgrid = np.column_stack((np.meshgrid(x, t)[0].flatten(),
                                   np.meshgrid(x, t)[1].flatten()))
    repeated_u_zero = np.tile(u_zero, n_points)

    pde_dataset = pd.DataFrame({'x': meshgrid[:,0],
                                't': meshgrid[:,1],
                                'u_zero':repeated_u_zero})

    #initial condition
    x_initial = np.linspace(0, 1, n_points)
    u_zero_initial = initial_condition(x_initial)
    IC_dataset = pd.DataFrame({'x_initial': x_initial,
                               't_initial': np.zeros_like(x_initial),
                               'u_zero_initial': u_zero_initial})

    #periodic_condition
    u_zero_x_0 = np.tile(initial_condition(0.0), n_points)
    u_zero_x_1 = np.tile(initial_condition(1.0), n_points)
    periodic_condition_dataset = pd.DataFrame({'x_0': torch.zeros(n_points),
                                               'x_1': torch.ones(n_points),
                                               't': t,
                                               'u_zero_0_t': u_zero_x_0,
                                               'u_zero_1_t': u_zero_x_1})

    #fitting condition
    fitting_meshgrid = meshgrid
    fitting_u_zero = repeated_u_zero
    u_labels = true_burgers_solution(x,t).flatten()
    if test == True:
        fitting_dataset = pd.DataFrame({'test_x': fitting_meshgrid[:, 0],
                                        'test_t': fitting_meshgrid[:, 1],
                                        'test_u_zero': fitting_u_zero,
                                        'test_u_labels': u_labels})
    else:
        fitting_dataset = pd.DataFrame({'fitting_x': fitting_meshgrid[:, 0],
                                        'fitting_t': fitting_meshgrid[:, 1],
                                        'fitting_u_zero': fitting_u_zero,
                                        'fitting_u_labels': u_labels})

    return pde_dataset, IC_dataset, periodic_condition_dataset, fitting_dataset


def crea_nuovo_dataset(n_points = 100):
    range = (0,1)
    x = np.linspace(*range, n_points)
    t = np.linspace(*range, n_points)
    ticks = np.linspace(0, 1, 5) * (range[1] - range[0]) + range[0]

    u_zero = initial_condition(x) #same dimension of x

    meshgrid = np.column_stack((np.meshgrid(x, t)[0].flatten(),
                                   np.meshgrid(x, t)[1].flatten()))

    transposed_array = np.transpose(u_zero)
    stacked_u_zero = np.tile(transposed_array, (n_points**2, 1))
    final_dataset = np.concatenate((meshgrid, stacked_u_zero), axis = 1)

    u_labels = true_burgers_solution(x,t)  #(len(t), len(x))
    # Plot meshgrid
    plt.figure(figsize=(8, 6))
    plt.imshow(u_labels, extent=[*range, *range], cmap='viridis')  # Use 'viridis' colormap for better visualization
    plt.colorbar()  # Add a colorbar to show scale
    plt.title('True solution')
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.show()

    return

crea_nuovo_dataset()



''' 
pde_dataset, IC_dataset, periodic_condition_dataset, fitting_dataset = create_burgers_dataset(30, test= True)
print(fitting_dataset, fitting_dataset.shape)
data_folder = f'burgers_data_folder/'
fitting_dataset.to_csv(data_folder + 'test_dataset.csv', sep = ',', index = False)

pde_dataset, IC_dataset, periodic_condition_dataset, fitting_dataset = create_burgers_dataset(30)
df = pd.read_csv(str(data_folder +'pde_dataset.csv'), sep=',', header=0)
print(df)
# Create a folder to save DataFrames
data_folder = f'burgers_data_folder/'
pde_dataset.to_csv(data_folder +'pde_dataset.csv', sep = ',', index = False)
IC_dataset.to_csv(data_folder + 'IC_dataset.csv', sep = ',', index = False)
periodic_condition_dataset.to_csv(data_folder + 'periodic_condition_dataset.csv', sep = ',', index = False)
fitting_dataset.to_csv(data_folder + 'fitting_dataset.csv', sep = ',', index = False)
'''

def kinetic_kosir(x, t) -> np.ndarray:
    # A -> 2B; A <-> C; A <-> D
    rates = [8.566 / 2, 1.191, 5.743, 10.219, 1.535]
    return np.array([-(rates[0] + rates[1] + rates[3]) * x[0] + rates[2] * x[2] + rates[4] * x[3],
                     2 * rates[0] * x[0],
                     rates[1] * x[0] - rates[2] * x[2],
                     rates[3] * x[0] - rates[4] * x[3]])

def create_dataset(n_obj_sample = 100, test = False):
    random.seed(1776526)
    np.random.seed(1776526)

    time_range = 10
    time_discretization = time_range / 100
    time_span = np.arange(0, time_range, time_discretization)

    # np.repeat(time_span, n_obj_sample, axis = 0)

    noise_level = 1
    x_init = [np.random.uniform(-noise_level, noise_level, size=(4,)) for _ in range(n_obj_sample)]
    for e in range(n_obj_sample):
        x_init[e][0] = x_init[e][0] + 14.5467204
        x_init[e][1] = x_init[e][1] + 16.3351668
        x_init[e][2] = x_init[e][2] + 25.9473091
        x_init[e][3] = x_init[e][3] + 23.5250934

    # Evaluate solution for each experiment
    solution_list = []
    for index, element in enumerate(x_init):
        solution = odeint(func=kinetic_kosir, y0=element, t=time_span)
        solution_list.append(solution)

    final_solution = np.vstack(solution_list)
    time_span = np.tile(time_span, n_obj_sample).flatten()[:, np.newaxis]
    x_init = np.repeat(x_init, 100, axis=0)
    final_training_dataset = pd.DataFrame(np.concatenate((time_span, x_init, final_solution), axis=1))

    t = torch.Tensor(final_training_dataset.iloc[:, 0].values).unsqueeze(1).requires_grad_(True)
    y_initial = torch.Tensor(final_training_dataset.iloc[:, 1:5].values)
    y_label = torch.Tensor(final_training_dataset.iloc[:, 5:].values)

    file_path = 'chemistry_data_folder/train_noise_1.txt'
    final_training_dataset.to_csv(file_path, sep = ',', index = False, header= False)

    return

#create_dataset(1000, test= True)

