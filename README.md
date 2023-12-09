# Memory-induced long-range order in neural activity
This repository contains source code for reproducing the simulation results in the paper *Memory-induced long-range order in neural activity*. Paper coming (very) soon! 

Requirements: NumPy 

Recommended:  CuPy (for large system size simulations)

Usage: 

`python data_generator_np.py` simulates the neural activity with specific parameters with NumPy. One can change the simulation parameters by modifying the file `neuron_ODE_np.py`. 

`python data_generator_cp.py` simulates the neural activity with specific parameters with CuPy, suitable for large lattice size simulation (up to 512 x 512). One can change the simulation parameters by modifying the file `neuron_ODE_np.py`. 
