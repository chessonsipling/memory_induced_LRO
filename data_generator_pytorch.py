import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import os
from tqdm import tqdm, trange

import torch
import torch.nn.functional as F
import sys
from cluster_finding import find_cluster_graph
sys.setrecursionlimit(3000)

torch.set_default_tensor_type(torch.cuda.FloatTensor
                              if torch.cuda.is_available()
                              else torch.FloatTensor)


def ODE(current_Rho, current_R, delta, td, a, b, h, D, sigma, L):
    # current_Rho, current_R: (batch, L, L)
    batch = current_Rho.shape[0]
    laplacian_rho = current_Rho.roll(1, 1) + current_Rho.roll(-1, 1) \
                  + current_Rho.roll(1, 2) + current_Rho.roll(-1, 2) - 4 * current_Rho
    Rho_deriv = (-a + current_R) * current_Rho + b * current_Rho ** 2 - current_Rho ** 3 \
                + h + D * laplacian_rho + sigma * torch.randn(batch, L, L)
    R_deriv = delta - (current_R * current_Rho - sigma * torch.randn(batch, L, L)) / td
    return torch.stack([Rho_deriv, R_deriv], dim=0)


def Euler(func, x, h):
    return h * func(x)


def RK4(func, x, h):
    k1 = func(x)
    k2 = func(x + h / 2 * k1)
    k3 = func(x + h / 2 * k2)
    k4 = func(x + h * k3)
    return h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def step(x, delta_t):
    # x: (2, batch, L, L)
    dx = Euler(lambda w: ODE(w[0], w[1], delta, td, a, b, h, D, sigma, L), x, delta_t)
    # dx = RK4(lambda w: ODE(w[0], w[1], delta, td, a, b, h, D, sigma, L), x, delta_t)
    return (x + dx).clamp_(0)


if __name__ == '__main__':

    compiled_step = None
    a = 1
    b = 1.5
    h = 1e-7
    D = 1                    # 1e-2, 1e-1, 1, 2, 4, Discussed in Supp. Info. 2
    sigma = 1
    delta = 4e-3
    delta_t = 0.01     ##  delta_t simulation (different from delta_t binning)
    delta_t_transient = 0.01
    Ls = [64, 128, 256, 512]
    time_spans = [5000, 2000, 500, 125]
    ensemble_sizes = [160, 400, 2000, 4800]
    # Ls = [512]
    # time_spans = [125]
    # ensemble_sizes = [4800]
    # transient_time = 1000
    #  = int(transient_time / delta_t_transient)
    minibatch_ensemble = 80

    window_length = 30
    window_dt = 0.3
    threshold = 0.5
    # td_list = [17, 20, 23, 25, 34, 51, 68, 77, 80, 82, 85, 88]
    # td_list = [7, 15, 17, 20, 82, 85, 88, 91, 97]
    # td_list = [82, 85, 88, 91, 97]
    td_list = [51]

    for L_idx, L in enumerate(Ls):

        lattice_idx = torch.arange(L * L).reshape(L, L)
        pad_op = torch.nn.CircularPad2d((0, 1, 0, 1))
        lattice_idx = pad_op(lattice_idx.reshape(1, L, L)).reshape(L + 1, L + 1)
        edges = [torch.stack([lattice_idx[:-1, :].reshape(-1), lattice_idx[1:, :].reshape(-1)], dim=1),
                 torch.stack([lattice_idx[:, :-1].reshape(-1), lattice_idx[:, 1:].reshape(-1)], dim=1)]
        edges = torch.cat(edges, dim=0)

        time_span = time_spans[L_idx]
        ensemble_size = ensemble_sizes[L_idx]
        n_steps = int(time_span / delta_t)
        n_minibatch_ensemble = int(ensemble_size / minibatch_ensemble)

        for td in td_list:
            result_folder = f"../correlation_length_no_limit/td_{td}/"
            graph_folder = f'graphs/td_{td}/'
            os.makedirs(result_folder, exist_ok=True)
            os.makedirs(graph_folder, exist_ok=True)

            for i in trange(n_minibatch_ensemble):
                current_Rho = torch.abs(torch.randn(minibatch_ensemble, L, L) * 0.1 + 0.24)
                current_R = torch.abs(torch.randn(minibatch_ensemble, L, L) * 0.1 + 0.29)
                s_last = None
                x = torch.stack([current_Rho, current_R], dim=0)

                if compiled_step is None:
                    compiled_step = torch.compile(step)

                length = int(np.ceil(n_steps / window_length))
                t_idx = 0
                flip = torch.zeros(minibatch_ensemble, L * L, length, dtype=torch.bool)
                print(f'Simulating dynamics for td={td}, ensemble {i+1}/{n_minibatch_ensemble}')
                transient_steps = np.random.randint(50000, 100000)
                for t_step in trange(transient_steps):
                    x = compiled_step(x, delta_t_transient)
                s_last = x[0] > 0.5
                for t_step in trange(n_steps):
                    x = compiled_step(x, delta_t)
                    if t_step % window_length == 0:
                        s = x[0] > 0.5
                        flip[:, :, t_idx] = (s ^ s_last).reshape(minibatch_ensemble, L * L)
                        # flip.append((s ^ s_last).cpu().numpy())  # move data to CPU to save GPU memory
                        s_last = s.clone()
                        t_idx += 1
                flip = flip.to('cpu')

                # flip = np.stack(flip, axis=-1).reshape((minibatch_ensemble, L * L, length))
                print(f'Finding clusters for td={td}, ensemble {i+1}/{n_minibatch_ensemble}')
                for j in trange(minibatch_ensemble):
                    idx = i * minibatch_ensemble + j
                    flip_j = flip[j].to(torch.device('cuda'))
                    # flip_j = torch.tensor(flip[j])  # move data back to GPU
                    print(f'{L}_{td}_{window_length}_{idx}')
                    label, cluster_sizes, duration = find_cluster_graph(flip_j, edges)
                    save_string = f'{L}_{td}_{window_length}_{idx}'
                    torch.save(cluster_sizes, f'{result_folder}cluster_sizes_{save_string}.pt')
                    torch.save(duration, f'{result_folder}duration_{save_string}.pt')

