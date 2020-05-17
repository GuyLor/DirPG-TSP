import numpy as np
import torch
import kruskals_cpp



def torch_divmod(x,y):
    return x//y, x%y
def prim_pytorch(distance_matrix, not_visited=None):
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
    """
    dm = torch.index_select(
        torch.index_select(distance_matrix, 0, not_visited),
        1, not_visited) if len(not_visited) - 2 > 0 else distance_matrix
    """
    dm = distance_matrix.clone()
    device = dm.device
    n_vertices = torch.tensor(dm.shape[0], device=device)
    visited_vertices = torch.tensor([0], device=device)  # Add the first point
    num_visited = 1
    # exclude self connections by assigning inf to the diagonal
    dm.fill_diagonal_(np.inf)

    mst_edges = torch.zeros(n_vertices, n_vertices, dtype=torch.bool, device=dm.device)
    while num_visited != n_vertices:
        new_edge = torch.argmin(dm[visited_vertices])
        new_edge = torch_divmod(new_edge, n_vertices)
        new_edge = [visited_vertices[new_edge[0]], new_edge[1]]

        mst_edges[new_edge[0], new_edge[1]] = True
        visited_vertices = torch.cat([visited_vertices,new_edge[1].unsqueeze(0)], dim=0)
        dm[visited_vertices, new_edge[1]] = np.inf
        dm[new_edge[1], visited_vertices] = np.inf
        num_visited += 1
    return mst_edges*distance_matrix




def prim_np(distance_matrix, prefix):
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
    if len(prefix)>1:
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

class kruskals:
    def __init__(self, distance_matrix):
        self.weights_and_edges = self.sort_edges(self.convert_distance_matrix_to_batched_edges(distance_matrix))

    def compute_mst(self, not_visited):
        reduced_sorted_wae = self.weights_and_edges[:,not_visited]
        edges_idx = kruskals_cpp.get_tree(reduced_sorted_wae[:, :, 1:].int(), n, False)
        return reduced_sorted_wae[:, :, 0][edges_idx.bool()].sum()



    @staticmethod
    def sort_edges(weights_and_edges):
        sorted_weights = torch.argsort(weights_and_edges[:, :, 0], -1, descending=False)
        dummy = sorted_weights.unsqueeze(2).expand(*(sorted_weights.shape + (weights_and_edges.size(2),)))
        # sorted_edges is shape (batch_size, n * (n - 1) / 2, 2)
        return torch.gather(weights_and_edges, 1, dummy)

    @staticmethod
    def convert_distance_matrix_to_batched_edges(distance_matrix):
        """distance_matrix: batch of distance matrices. size: [batch, n, n]
        returns weights_and_edges: in shape (batch_size, n * (n - 1) / 2, 3), where
        weights_and_edges[.][i] = [weight_i, node1_i, node2_i] for edge i."""

        n = distance_matrix.shape[0]
        weights_and_edges = torch.zeros(1, n * (n - 1) // 2, 3)

        upper_trg_ind = np.triu_indices(n, k=1)
        edges_heap = distance_matrix[upper_trg_ind].tolist()

        for i, (edges, n1, n2) in enumerate(zip(edges_heap, *upper_trg_ind)):
            weights_and_edges[0, i] = torch.tensor([edges, n1, n2])

        return weights_and_edges

