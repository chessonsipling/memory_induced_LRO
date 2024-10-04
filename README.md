# Memory in neural activity: long-range order without criticality
This repository contains source code for reproducing the simulation results in the paper *Memory in neural activity: long-range order without criticality*. [Paper now available on arxiv!](https://arxiv.org/abs/2409.16394)


Requirements: NumPy >= 1.23.1, SciPy >= 1.13.1, PyTorch >= 2.3.0

Usage: 

`python data_generator_pytorch.py` simulates the neural activity with specific parameters with PyTorch.

`python correlation_length.py` computes and plots the avalanche size distributions using the data generated in the previous step. 
