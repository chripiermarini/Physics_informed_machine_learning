import torch
from torch.optim.optimizer import Optimizer, required
import copy
import numpy as np

class StochasticSQP(Optimizer):
    r"""Implements the algorithm proposed in  [Sequential Quadratic Optimization for Nonlinear Equality Constrained Stochastic Optimization]
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
    """

    def __init__(self, params, lr=required, 
                 n_parameters = 0, 
                 n_constrs = 0, 
                 merit_param_init = 1,
                 ratio_param_init = 1, 
                 step_size_decay = 0.5):
        defaults = dict()
        super(StochasticSQP, self).__init__(params, defaults)
        self.n_parameters = n_parameters
        self.n_constrs = n_constrs
        self.merit_param = merit_param_init
        self.ratio_param = ratio_param_init
        self.step_size = lr
        self.step_size_decay = step_size_decay

    def __setstate__(self, state):
        super(StochasticSQP, self).__setstate__(state)

    def step(self, closure=None):
        """ Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        Before call this method, you have to assign the value of jacobian, gradient, 
            constraint, and objective to the state of keys: J, g, c and f.
        """
        loss = None

        ## Compute step
        J = self.state['J']
        g = self.state['g']
        c = self.state['c']
        f = self.state['f'] 
        ls_matrix = torch.cat((torch.cat((torch.eye(self.n_parameters), torch.transpose(J,0,1)), 1),
                               torch.cat((J, torch.zeros(self.n_constrs,self.n_constrs)), 1)), 0)
        ls_rhs = -torch.cat((g,c), 0)

        # the line below is timeconsuming if the linear system is large
        # the computed d is the step that will be used to update 'param'
        d = torch.linalg.solve(ls_matrix, ls_rhs)

        ## Update merit parameter
        # the meric parameter is different from the paper. It is not changed

        ## Update ratio parameter
        # the meric parameter is different from the paper. It is not changed

        ## Update stepsize
        # the stepsize is different from the paper, but a simple one:
        # if the current merit function value is greater than previous merit function, then
        #   decrease the stepsize 
        # otherwise
        #   do not change stepsize
        self.state['cur_merit_f'] = self.merit_param * f + torch.linalg.norm(c, 1)
        if 'iter' not in self.state:
            self.state['iter'] = 0
        else:
            self.state['iter'] += 1
            if self.state['cur_merit_f'] > self.state['pre_merit_f']:
                self.step_size = self.step_size_decay * self.step_size
        self.state['pre_merit_f'] = self.state['cur_merit_f']

        ## Update parameters
        assert len(self.param_groups) == 1
        group = self.param_groups[0]
        d_p_i_start = 0
        for p in group['params']:
            if p.grad is None:
                continue
            d_p_i_end = d_p_i_start + len(p.view(-1))
            d_p = d[d_p_i_start:d_p_i_end].reshape(p.shape)
            p.data.add_(d_p, alpha=self.step_size)
            d_p_i_start = d_p_i_end
            
        return loss
    
    def printerHeader(self):
        print('{:>8s} {:>11s} {:>11s} {:>11s} {:>11s} {:>11s} {:>11s}'.format('Iter', 'Loss', '||c||', 'merit_f','stepsize','merit_param','ratio_param'))

    def printerIteration(self,every=1):
        if np.mod(self.state['iter'],every) == 0:
            print('{:8d} {:11.4e} {:11.4e} {:11.4e} {:11.4e} {:11.4e} {:11.4e}'.format(
                self.state['iter'], 
                self.state['f'], 
                torch.linalg.norm(self.state['c'], 1), 
                self.state['cur_merit_f'],
                self.step_size,
                self.merit_param,
                self.ratio_param,
                ))
