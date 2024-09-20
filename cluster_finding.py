import numpy as np
import os
from tqdm import tqdm, trange

import torch
import sys
sys.setrecursionlimit(3000)

torch.set_default_tensor_type(torch.cuda.FloatTensor
                              if torch.cuda.is_available()
                              else torch.FloatTensor)



def find(vertex, parent):
    if parent[vertex] != vertex:
        parent[vertex] = find(parent[vertex], parent)
    return parent[vertex]


def union(u, v, parent):
    root_u = find(u, parent)
    root_v = find(v, parent)
    if root_u != root_v:
        parent[root_u] = root_v


def find_cluster_graph(lattice, edges):
    # lattice: (n, length)
    n, length = lattice.shape
    if length < 2:
        return lattice, torch.zeros(1)
    label = torch.zeros_like(lattice, dtype=torch.int)
    label = label.reshape(-1)
    label[lattice.reshape(-1) > 0] = torch.arange(1, (lattice.reshape(-1) > 0).sum() + 1, dtype=torch.int)
    label = label.reshape(n, length)

    equivalence = []
    for i in range(length):
        equivalence.append(label[:, i][edges])
    for i in range(length - 1):
        equivalence.append(torch.stack([label[:, i], label[:, i + 1]], dim=1))
        equivalence.append(torch.stack([label[:, i][edges[:, 0]], label[:, i + 1][edges[:, 1]]], dim=1))
        equivalence.append(torch.stack([label[:, i][edges[:, 1]], label[:, i + 1][edges[:, 0]]], dim=1))
    equivalence = torch.cat(equivalence, dim=0)

    nonzero_mask = (equivalence > 0).all(dim=1)
    equivalence = equivalence[nonzero_mask]
    equivalence = torch.unique(equivalence, dim=0)

    # find connected components of the equivalence graph
    graph_edges = equivalence.cpu().numpy()
    nodes = np.arange(1, lattice.sum().cpu().numpy().item() + 1)
    parent = {key: value for key, value in zip(nodes, nodes)}

    for edge in graph_edges:
        union(edge[0], edge[1], parent)

    value_map = torch.tensor([find(node, parent) for node in nodes])
    unique_labels = torch.unique(value_map)
    if unique_labels.numel() == 0:
        return label, torch.zeros(1), torch.zeros(1)
    else:
        relabeled = torch.arange(1, len(unique_labels) + 1, dtype=torch.int)
        relabel_map = torch.zeros(unique_labels.max() + 1, dtype=torch.int)
        relabel_map[unique_labels] = relabeled
        value_map = relabel_map[value_map]

        value_map = torch.cat([torch.zeros(1, dtype=torch.int), value_map], dim=0)

        # relabel the lattice
        label = value_map[label].to(torch.int64)

        # # Optional: each site can only be counted once within each cluster, remove the duplicates
        # new_label, indices = label.sort(dim=1)
        # new_label[:, 1:] *= (torch.diff(new_label, dim=1) != 0).to(torch.int64)
        # indices = indices.sort(dim=1)[1]
        # label = torch.gather(new_label, 1, indices)
        index = label.reshape(-1).to(torch.int64)

        # Compute the duration of each avalanche
        min_times = torch.zeros(len(unique_labels) + 1, dtype=torch.int64)
        max_times = torch.zeros(len(unique_labels) + 1, dtype=torch.int64)
        time_indices = torch.arange(label.shape[1]).unsqueeze(0).expand(label.shape).reshape(-1)
        min_times.scatter_reduce_(0, label.reshape(-1), time_indices, reduce='amin', include_self=False)
        max_times.scatter_reduce_(0, label.reshape(-1), time_indices, reduce='amax', include_self=False)
        duration = max_times - min_times + 1
        duration = duration[1:]

        weight = lattice.reshape(-1)
        cluster_sizes = torch.zeros(value_map.max() + 1, dtype=torch.float)
        cluster_sizes.scatter_add_(0, index, weight.to(torch.float))
        cluster_sizes = cluster_sizes[1:]

        return label, cluster_sizes, duration