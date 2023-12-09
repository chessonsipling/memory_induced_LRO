# Memory-induced long-range order in neural activity
This repository contains source code for reproducing the simulation results in the paper *Memory-induced long-range order in neural activity*. Paper coming (very) soon! 

Requirements: NumPy 
Recommended:  CuPy (for large system size simulations)

Usage: 

`python data_generator_np.py` computes the avalanche size distribution with specific parameters. One can change the parameters by modifying the file `neuron_ODE_np.py`. 

`python data_generator_cp.py` computes the avalanche size distribution with specific parameters. One can change the parameters by modifying the file `neuron_ODE_np.py`. 
