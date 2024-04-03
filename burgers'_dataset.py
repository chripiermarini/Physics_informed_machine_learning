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
np.set_printoptions(threshold=np.inf)
import pandas as pd

def initial_condition(x):
    random.seed(1776526)
    np.random.seed(1776526)
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

def create_burgers_dataset(n_points):
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
    fitting_dataset = pd.DataFrame({'fitting_x': fitting_meshgrid[:,0],
                                    'fitting_t': fitting_meshgrid[:,1],
                                    'fitting_u_zero': fitting_u_zero,
                                    'fitting_u_labels': u_labels})
    return pde_dataset, IC_dataset, periodic_condition_dataset, fitting_dataset


pde_dataset, IC_dataset, periodic_condition_dataset, fitting_dataset = create_burgers_dataset(30)

# Create a folder to save DataFrames
data_folder = f'burgers_data_folder/'
pde_dataset.to_csv(data_folder +'pde_dataset.csv', sep = ',', index = False)
IC_dataset.to_csv(data_folder + 'IC_dataset.csv', sep = ',', index = False)
periodic_condition_dataset.to_csv(data_folder + 'periodic_condition_dataset.csv', sep = ',', index = False)
fitting_dataset.to_csv(data_folder + 'fitting_dataset.csv', sep = ',', index = False)

df = pd.read_csv(str(data_folder +'fitting_dataset.csv'), sep=',', header=0)
