import torch
from torch.optim.optimizer import Optimizer, required
import copy
import numpy as np

class StochasticSQP(Optimizer):
    r"""Implements the algorithm proposed in
    [Sequential Quadratic Optimization for Nonlinear Equality Constrained Stochastic Optimization]
    Example:
    !    >>> from stochasticsqp import *
    !    >>> optimizer = StochasticSQP(model.parameters(), lr=0.1)

    Input:
        params          : torch neural network paratemetrs to be optimized
        lr              : initial stepsize \alpha
        n_parameters    : number of parameters to be optimized
        n_constrs       : number of constraints
        merit_param_init: initial \tau 
        ratio_param_init: initial \xi
        step_size_decay : factor for decreasing stepsize, i.e., alpha = step_size_decay * alpha
        sigma :
        epsilon: ratio parameter for numerical stability
        step_opt: option variable to set specific stepsize selection ( 1 for simple function reduction, 2 for Armijo)
    """
    
    sigma=0.5
    epsilon=1e-6
    eta=1e-4 # line search parameter
    buffer=0
    
    def __init__(self, params, lr=required, 
                 n_parameters = 0, 
                 n_constrs = 0, 
                 merit_param_init = 1,
                 ratio_param_init = 1, 
                 step_size_decay = 0.5,
                 step_opt= 1,
                 problem=None,
                 mu=100,
                 beta2 = 0.999):
        defaults = dict()
        super(StochasticSQP, self).__init__(params, defaults)
        self.n_parameters = n_parameters
        self.n_constrs = n_constrs
        self.merit_param = merit_param_init
        self.ratio_param = ratio_param_init
        self.step_size = lr
        self.step_size_init = lr
        self.step_size_decay = step_size_decay
        self.trial_merit = 1.0
        self.trial_ratio = 1.0
        self.norm_d = 0.0
        self.initial_params = params
        self.step_opt = step_opt
        self.problem = problem
        self.mu = mu
        self.beta2 = beta2 
        self.printerBeginningSummary()

    def __setstate__(self, state):
        super(StochasticSQP, self).__setstate__(state)

    def initialize_param(self, initial_value = 1):
        ## Update parameters
        assert len(self.param_groups) == 1
        group = self.param_groups[0]
        for p in group['params']:
            p.data.add_(initial_value, alpha=1)
        return

    def step(self, closure=None):
        """ Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        Before call this method, you have to assign the value of jacobian, gradient, 
            constraint, and objective to the state of keys: J, g, c and f.
        Here, we also define an example of the Hessian matrix //
        The H matrix is set as torch.eye(self.n_parameters)
        """
        
        ## Compute step
        J = self.state['J']
        g = self.state['g']
        c = self.state['c']
        f = self.state['f']
        
        if 'iter' not in self.state:
            self.state['iter'] = 0
            self.state['g_square_sum'] = (1-self.beta2) * g**2
        else:
            self.state['iter'] += 1
            self.state['g_square_sum'] = self.beta2 * self.state['g_square_sum'] +  (1-self.beta2) * g**2
        
        loss = None

        H_diag = torch.sqrt(self.state['g_square_sum'] + self.mu)
        H =  torch.diag(H_diag) #torch.eye(self.n_parameters)
        self.state['H_diag'] = H_diag

        ls_matrix = torch.cat((torch.cat((H, torch.transpose(J,0,1)), 1),
                               torch.cat((J, torch.zeros(self.n_constrs,self.n_constrs)), 1)), 0)
        ls_rhs = -torch.cat((g,c), 0)

        # the line below is time-consuming if the linear system is large
        # the computed system_solution
        # remember to split the primal solution d (is the step that will be used to update 'param')
        # and the dual solution y

        system_solution = torch.linalg.solve(ls_matrix, ls_rhs)
        d = system_solution[:self.n_parameters]
        y = system_solution[self.n_parameters:]
        self.norm_d = torch.norm(d)
        
        self.kkt_norm = torch.norm(g + torch.matmul(torch.transpose(J,0,1), y), float('inf'))

        dHd = torch.matmul(torch.matmul(d, H), d)
        gd = torch.matmul(g, d)
        gd_plus_max_dHd_0 = gd + torch.max(dHd, torch.tensor(0))
        c_norm_1 = torch.linalg.norm(c, ord=1)

        ## norm of d_k equal to 0 exception
        if torch.linalg.norm(d, ord = 2) <= 10**(-8):
            self.trial_merit = 10 **(10)
            self.trial_ratio = 10 **(10)
            self.step_size = 1
        else:
            ## Update merit parameter
            # define trial merit parameter
            if gd_plus_max_dHd_0 <= 0:
                self.trial_merit = 10 ** 10
            else:
                self.trial_merit = ((1 - self.sigma) * c_norm_1) / gd_plus_max_dHd_0

            if self.merit_param > self.trial_merit:
                self.merit_param = self.trial_merit * (1 - self.epsilon)

            ## Update ratio parameter
            # since d is the solution of the linear system, it entails delta_q defined through the formula (2.4)
            delta_q = - self.merit_param * (gd + 0.5 * torch.max(dHd, torch.tensor(0))) + c_norm_1
            self.trial_ratio = delta_q / (self.merit_param * self.norm_d ** (2))
            if self.ratio_param > self.trial_ratio:
                self.ratio_param = self.trial_ratio * (1 - self.epsilon) 

        
        """
        Line Search step-size

        """
        #get current values of parameters, f is the current phi
        self.state['cur_merit_f'] = self.merit_param * f + torch.linalg.norm(c, 1)
        self.state['phi_new'] = self.state['cur_merit_f']
        self.state['search_rhs'] = 0
        
        alpha_pre = 0.0
        phi = self.merit_param * f + torch.linalg.norm(c, 1)
        self.ls_k = 0
        self.step_size = self.step_size_init
        for self.ls_k in range(1):

            ## Update parameters
            assert len(self.param_groups) == 1
            group = self.param_groups[0]
            d_p_i_start = 0
            for p in group['params']:
                #print(p.view(-1).shape)
                # TODO: check of p.grad is None
                #if p.grad is None:
                #    continue
                d_p_i_end = d_p_i_start + len(p.view(-1))
                d_p = d[d_p_i_start:d_p_i_end].reshape(p.shape)
                p.data.add_(d_p, alpha=self.step_size-alpha_pre)
                d_p_i_start = d_p_i_end

            #compute objective, constraints in new point using new values
            f_new = self.state["f_g_hand"](self, no_grad = True)
            c_new = self.state["c_J_hand"](self, no_grad = True)
            self.state['phi_new']= self.merit_param*f_new + torch.linalg.norm(c_new, ord= 1)
            self.state['alpha_sqp'] = self.step_size-alpha_pre
            
            #perfrom linesearch procedure
            if self.step_opt == 1: #simple decrease
                self.state['search_rhs'] = self.state['cur_merit_f']
            elif self.step_opt == 2: #armijo
                delta_q = - self.merit_param * (gd + 0.5 * torch.max(dHd, torch.tensor(0))) + c_norm_1
                self.state['search_rhs'] = self.state['cur_merit_f'] - self.eta*self.step_size*delta_q
            
            #either accept the new point or reduce step_size, add buffer for computation precision
            if self.state['phi_new'] < self.state['search_rhs'] + self.buffer:
                break
            else:
                alpha_pre = self.step_size
                self.step_size = self.step_size * self.step_size_decay
            #self.printerIteration()

        return loss
    
    def printerBeginningSummary(self):
        print('-----------------------------------StochasticSQPOptimizer--------------------------------')
        print('Problem name:          ', self.problem.name)
        print('Number of parameters:  ', self.n_parameters)
        print('Number of constraints: ', self.n_constrs)
        print('Sample size:           ', self.problem.n_obj_sample)
        print('-----------------------------------------------------------------------------------------')
    
    def printerHeader(self):
        
        print('{:>8s} {:>11s} {:>11s} {:>11s} {:>11s} {:>11s} {:>11s} {:>11s} {:>11s} {:>11s} {:>11s} {:>11s} {:>11s} {:>11s} {:>11s} {:>11s}'
              .format('Iter', 'f', 'f_interior', 'f_boundary', '||c||_1', 'merit_f', 'phi_new', 'search_rhs','stepsize','merit_param','ratio_param',
                      'trial_merit', 'trial_ratio', 'norm_d','kkt_norm', 'ls_Iter'))

    def printerIteration(self,every=1):
        if np.mod(self.state['iter'],every) == 0:
            print('{:8d} {:11.4e} {:11.4e} {:11.4e} {:11.4e} {:11.4e} {:11.4e} {:11.4e} {:11.4e} {:11.4e} {:11.4e} {:11.4e} {:11.4e} {:11.4e} {:11.4e} {:11d}'.format(
                self.state['iter'], 
                self.state['f'], 
                self.state['f_interior'], 
                self.state['f_boundary'], 
                torch.linalg.norm(self.state['c'], 1), 
                self.state['cur_merit_f'],
                self.state['phi_new'],
                self.state['search_rhs'],
                self.step_size,
                self.merit_param,
                self.ratio_param,
                self.trial_merit,
                self.trial_ratio,
                self.norm_d,
                self.kkt_norm,
                self.ls_k
                ))
            if np.mod(self.state['iter'],every*20) == 0 and (self.state['iter'] != 0):
                self.printerHeader()
