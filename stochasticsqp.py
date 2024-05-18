import torch
from torch.optim.optimizer import Optimizer, required

class StochasticSQP(Optimizer):
    """
    Input:
        params          : <class 'generator'>
            torch neural network parameters to be optimized
        lr              : float
            stepsize 
        n_parameters    : int
            number of parameters to be optimized
        n_constrs       : int
            number of constraints
        merit_param_init: float
            initial merit parameter 
        ratio_param_init: float
            initial ratio parameter
        problem         : class BaseProblem
        config          : dict
            contain keys of 'mu', 'beta2', 'alpha_type', 'beta1'
    """
    
    sigma=0.5               # parameter for computing merit parameter
    epsilon=1e-6            # parameter for computing merit parameter
    eps_singular_matrix = 1 # parameter for modifying matrix to be nonsingular
    
    def __init__(self, params, lr=required, 
                 n_parameters = 0, 
                 n_constrs = 0, 
                 merit_param_init = 1,
                 ratio_param_init = 1, 
                 problem=None,
                 config=None,
                 ):
        defaults = dict()
        super(StochasticSQP, self).__init__(params, defaults)
        self.n_parameters = n_parameters
        self.n_constrs = n_constrs
        self.merit_param = merit_param_init
        self.ratio_param = ratio_param_init
        self.step_size = lr
        self.trial_merit = 1.0
        self.trial_ratio = 1.0
        self.norm_d = 0.0
        self.problem = problem
        self.mu = config['mu']
        self.beta2 = config['beta2']
        self.alpha_type = config['alpha_type']
        self.beta1 = config['beta1']   

    def __setstate__(self, state):
        super(StochasticSQP, self).__setstate__(state)

    def solve_linsys(self, A,b, i = 0):
        """
        Solve linear system Ax = b, if A is singular, a loop will begin and self.eps_singular_matrix will be added to the diagonal elements of A until the linear system is solved.
        
        Parameters
        ----------
        A and b: tensor of A and b for computing Ax = b
        
        Returns
        -------
        x : tensor
            solution of Ax = b
        i : int
            value that is added to the diagonal of A for ensuring A to be nonsingular
        """
        solved = False
        i = i
        while not solved:
            try: 
                if i > 0:
                    A = A + torch.diag(i * self.eps_singular_matrix * torch.ones(A.shape[0]))
                x = torch.linalg.solve(A,b)
                break
            except Exception as e:
                pass
            finally:
                i = i + 1
        return x,i

    def step(self, closure=None):
        """ Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        Before call this method, you have to assign the value of jacobian, gradient, 
            constraint, and objective to the state of keys: J, g, c and f.
        """
        
        ## obtain value of J, q, c, f
        J = self.state['J']
        g = self.state['g']
        c = self.state['c']
        f = self.state['f']
        
        ## compute right hand side value of p_k as in Algorithm 2 Line 6 in the paper
        ## for Adam, p_k(grad) is gradient
        ## for P-Adam, p_k(grad) is projected gradient
        if self.alpha_type == 'adam':
            grad = g
            singular_add_i = 0
        elif self.alpha_type == 'p_adam':
            Pg,singular_add_i = self.solve_linsys(J @ J.T, J@g)
            grad = g - J.T @ Pg
        
        ## Compute accumulated sum and sum of square of grad 
        if 'iter' not in self.state:
            self.state['iter'] = 0
            self.state['g_square_sum'] = (1-self.beta2) * grad**2
            self.state['g_sum'] = (1-self.beta1) * grad**2
        else:
            self.state['iter'] += 1
            self.state['g_square_sum'] = self.beta2 * self.state['g_square_sum'] +  (1-self.beta2) * grad**2
            self.state['g_sum'] = self.beta1 * self.state['g_sum'] +  (1-self.beta1) * grad
        
        ## Compute p_k^had and q_k^had as in Algorithm Line 4 and 5 in the paper
        m_hat = self.state['g_sum'] / (1 - self.beta1**(self.state['iter'] + 1))
        v_hat = self.state['g_square_sum'] / (1 - self.beta2**(self.state['iter'] + 1))
        
        loss = None
        
        H_diag = torch.sqrt(v_hat + self.mu)
        
        self.state['H_diag'] = H_diag
        
        if 0:
            # Solve linear system in Algorithm 2 Line 6 directly. 
            H =  torch.diag(H_diag) 
            ls_matrix = torch.cat((torch.cat((H, J.T), 1),
                                torch.cat((J, torch.zeros(self.n_constrs,self.n_constrs)), 1)), 0)
            ls_rhs = -torch.cat((m_hat,c), 0)
            # the line below is time-consuming if the linear system is large
            system_solution = torch.linalg.solve(ls_matrix, ls_rhs)
            d = system_solution[:self.n_parameters]
        else:
            # Solve linear system by decomposed step u and v. We use this method for efficiency.
            v = - J.T @ self.solve_linsys(J @ J.T, c, i = singular_add_i)[0] 
            JHinvJT = J*(1/H_diag) @ J.T
            Hinvm_plus_v = 1/H_diag * m_hat + v
            Pv = J.T @ self.solve_linsys(JHinvJT, J@Hinvm_plus_v, i = singular_add_i)[0] 
            u = - Hinvm_plus_v +  1/H_diag * Pv 
            d = u + v
        
        self.norm_d = torch.norm(d)

        dHd = torch.matmul(d*H_diag, d)
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
            delta_q = - self.merit_param * (gd + 0.5 * torch.max(dHd, torch.tensor(0))) + c_norm_1
            self.trial_ratio = delta_q / (self.merit_param * self.norm_d ** (2))
            if self.ratio_param > self.trial_ratio:
                self.ratio_param = self.trial_ratio * (1 - self.epsilon) 

        
        self.state['merit_param'] = self.merit_param
        self.state['cur_merit_f'] = self.merit_param * f + torch.linalg.norm(c, 1)

        ## Update parameters
        assert len(self.param_groups) == 1
        group = self.param_groups[0]
        d_p_i_start = 0
        for p in group['params']:
            #print(p.view(-1).shape)
            d_p_i_end = d_p_i_start + len(p.view(-1))
            d_p = d[d_p_i_start:d_p_i_end].reshape(p.shape)
            p.data.add_(d_p, alpha=self.step_size)
            d_p_i_start = d_p_i_end
        self.state['alpha_sqp'] = self.step_size

        return loss

    def load_pretrain_state(self,optim_path,device):
        """
        load pretrained optimizer state
        """
        state = torch.load(optim_path,map_location=device)
        self.state['iter'] = state['iter']
        self.state['g_square_sum'] = state['g_square_sum']
        self.state['g_sum'] = state['g_sum']
        
    def save_pretrain_state(self,optim_path):
        """
        load optimizer state
        """
        state = {'iter': self.state['iter'], 
                'g_square_sum': self.state['g_square_sum'],
                'g_sum':self.state['g_sum'],}
        torch.save(state, optim_path)
