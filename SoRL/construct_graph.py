import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def knn_graph_from_similarity(sim_matrix, k):
    N = sim_matrix.shape[0]
    graph = np.zeros((N, N))
    for i in range(N):
        row = sim_matrix[i].copy()
        row[i] = -np.inf
        knn_idx = np.argsort(row)[-k:]
        graph[i, knn_idx] = 1
    graph = np.maximum(graph, graph.T)
    return graph


def normalize_adj(adj, backend='torch'):
    if backend == 'torch':
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        deg = adj.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        return D_inv_sqrt @ adj @ D_inv_sqrt

    elif backend == 'numpy':
        adj = adj + np.eye(adj.shape[0])
        deg = np.sum(adj, axis=1)
        deg_inv_sqrt = np.power(deg, -0.5)
        deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.
        D_inv_sqrt = np.diag(deg_inv_sqrt)
        return D_inv_sqrt @ adj @ D_inv_sqrt


def get_graph(data, k, method):
    if method == 'euclidean':
        data = StandardScaler().fit_transform(data)
        dist = euclidean_distances(data)
        sim = -dist  # 越小距离，越相似

    elif method == 'pearson':
        # data: n_samples x n_features
        sim = np.corrcoef(data)  # 输出 shape: n_samples x n_samples
        np.fill_diagonal(sim, 0)  # 去除对角线
        sim = np.clip(sim, 0, 1)  # 只保留正相关（可改为 np.abs(sim)）
        np.abs(sim)

    else:
        raise ValueError("method must be 'euclidean' or 'pearson'")

    graph = knn_graph_from_similarity(sim, k)
    return torch.Tensor(normalize_adj(graph, 'numpy'))



