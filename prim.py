import numpy as np
import torch


def mst(distance_matrix, not_visited):
    """Determine the minimum spanning tree for a set of points represented
    :  by their inter-point distances... ie their 'W'eights
    :Requires:
    :--------
    :  W - edge weights (distance, time) for a set of points. W needs to be
    :      a square array or a np.triu perhaps
    :Returns:
    :-------
    :  pairs - the pair of nodes that form the edges
    """
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix needs to be square matrix of edge weights")

    dm = torch.index_select(
        torch.index_select(distance_matrix, 0, not_visited),
        1, not_visited)

    n_vertices = dm.shape[0]
    pairs = []
    visited_vertices = [0]  # Add the first point
    num_visited = 1
    # exclude self connections by assigning inf to the diagonal
    dm.fill_diagonal_(np.inf)
    #
    mst_val = 0
    while num_visited != n_vertices:
        new_edge = torch.argmin(dm[visited_vertices])
        mst_val += dm[visited_vertices].view(-1)[new_edge]

        new_edge = divmod(new_edge.item(), n_vertices)
        new_edge = [visited_vertices[new_edge[0]], new_edge[1]]
        pairs.append(new_edge)
        visited_vertices.append(new_edge[1])
        dm[visited_vertices, new_edge[1]] = np.inf
        dm[new_edge[1], visited_vertices] = np.inf
        num_visited += 1
    return mst_val



def mst_np(distance_matrix, prefix):
    """
    Determine the minimum spanning tree for a set of points represented
    :  by their inter-point distances... ie their 'W'eights
    :Requires:
    :--------
    :  W - edge weights (distance, time) for a set of points. W needs to be
    :      a square array or a np.triu perhaps
    :Returns:
    :-------
    :  pairs - the pair of nodes that form the edges
    """

    W = distance_matrix.copy()

    if W.shape[0] != W.shape[1]:
        raise ValueError("W needs to be square matrix of edge weights")
    W = np.delete(np.delete(W, prefix[1:], 0), prefix[1:], 1)
    Np = W.shape[0]
    pairs = []
    pnts_seen = [0]  # Add the first point                    
    n_seen = 1
    # exclude self connections by assigning inf to the diagonal
    diag = np.arange(Np)
    W[diag, diag] = np.inf
    # 
    mst_val = 0
    while n_seen != Np:                                     
        new_edge = np.argmin(W[pnts_seen], axis=None)
        mst_val += W[pnts_seen].reshape(-1)[new_edge]

        new_edge = divmod(new_edge, Np)

        new_edge = [pnts_seen[new_edge[0]], new_edge[1]]
        pairs.append(new_edge)
        pnts_seen.append(new_edge[1])
        W[pnts_seen, new_edge[1]] = np.inf
        W[new_edge[1], pnts_seen] = np.inf
        n_seen += 1
    return mst_val


def greedy_path(distance_matrix, prefix):
    M = np.inf
    cost = []
    path = []
    dest = prefix[0]
    current = prefix[-1]
    dm_copy = distance_matrix.copy()

    np.fill_diagonal(dm_copy, M)
    dm_copy[:, prefix[:-1]] = M
    dm_copy[prefix[:-1], :] = M

    while np.any(dm_copy != M):
        greedy = np.argmin(dm_copy[current])
        cost.append(dm_copy[current][greedy])
        path.append(greedy)
        dm_copy[:, current] = M
        dm_copy[current, :] = M
        current = greedy

    cost.append(distance_matrix[path[-1], dest])
    return np.sum(cost)

