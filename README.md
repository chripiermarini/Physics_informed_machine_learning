# SQPPIML
Using Stochastic SQP algorithm to solve physics-informed machine learning problem

Solve a problem :
``
python solve Spring
``


`stochasticsqp.py` is the optimizer, where the `step` method is to compute the step to update neural network parameters. This `step` method is a simplified version of Stochastic SQP method in [Berahas, Albert S., et al. "Sequential quadratic optimization for nonlinear equality constrained stochastic optimization." SIAM Journal on Optimization 31.2 (2021): 1352-1379.]

`test_stochasicsqp_pde_example.ipynb` is a demo of how to use the above optimizer to solve pde constrained machine learning problem. Try to run each block of this jupyter notebook file.

