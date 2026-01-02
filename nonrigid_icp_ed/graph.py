from dataclasses import dataclass
from typing import Literal

import torch

from nonrigid_icp_ed.knn import find_nearest_neighbors_faiss


# class Graph:
#     def __init__(
#         self,
#         poss: torch.Tensor | None = None,
#         edges: torch.Tensor | None = None,
#         weights: torch.Tensor | None = None,
#     ):
#         """
#         edges: (E, 2) LongTensor of edges (undirected)
#         num_nodes: number of nodes in the graph
#         """
#         if poss is None:
#             poss = torch.empty((0, 3), dtype=torch.float32)
#         if edges is None:
#             edges = torch.empty((0, 2), dtype=torch.long)
#         if weights is None:
#             weights = torch.empty((0,), dtype=torch.float32)
#         assert edges.dim() == 2 and edges.shape[1] == 2, "Edges must be of shape (E, 2)"
#         assert edges.dtype == torch.long, "Edges must be of dtype torch.long"
#         self.edges = edges
#         self.poss = poss
#         self.weights = weights

#     def make_edges(self, K: int, eps: float = 1e-8):
#         """
#         Construct K-NN graph edges based on current node positions.
#         Updates self.edges and self.weights.
#         """
#         src2tgt, tgt2src, s2t_dist_np, t2s_dist_np = find_nearest_neighbors_faiss(
#             self.poss, self.poss, K=K + 1
#         )
#         # exclude self-loop (first column)
#         knn_indices = src2tgt[:, 1:]  # (N, K)
#         knn_distances = torch.from_numpy(s2t_dist_np[:, 1:]).to(
#             device=self.poss.device, dtype=self.poss.dtype
#         )  # (N, K)

#         N = self.poss.shape[0]
#         edge_list = []
#         weight_list = []
#         for i in range(N):
#             weight_sum = 0.0
#             weights_per_node = []
#             for k in range(K):
#                 j = knn_indices[i, k].item()
#                 if i < j:  # to avoid duplicate edges
#                     edge_list.append([i, j])
#                     #edge_list.append([j, i])  # undirected
#                     dist_ij = knn_distances[i, k].item()
#                     weight = 1.0 / (dist_ij + eps)
#                     # weight_list.append(weight)
#                     weights_per_node.append(weight)
#                     #weights_per_node.append(weight)
#                     weight_sum += weight
#             # normalize weights for edges from node i
#             for w in weights_per_node:
#                 w /= weight_sum
#                 weight_list.append(w)
#         self.edges = torch.tensor(edge_list, dtype=torch.long, device=self.poss.device)
#         self.weights = torch.tensor(
#             weight_list, dtype=self.poss.dtype, device=self.poss.device
#         )
#         assert (
#             self.edges.shape[0] == self.weights.shape[0]
#         ), f"Edges and weights size mismatch: {self.edges.shape[0]} vs {self.weights.shape[0]}"
#         assert self.edges.shape[1] == 2, "Edges must be of shape (E, 2)"

#     def assign_nodes_to_points_by_knn(
#         self, points: torch.Tensor, K: int, eps: float = 1e-8
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         """
#         Assign each point to its K nearest graph nodes.

#         Returns:
#             indices: (P, K) LongTensor of node indices for each point
#             weights: (P, K) FloatTensor of weights for each point
#         """
#         src2tgt, tgt2src, s2t_dist_np, t2s_dist_np = find_nearest_neighbors_faiss(
#             points, self.poss, K=K
#         )
#         distances = torch.from_numpy(s2t_dist_np).to(
#             device=points.device, dtype=points.dtype
#         )  # (P, K)
#         if K == 1:
#             weights_per_point = torch.ones_like(distances)
#         else:
#             inv_distances = 1.0 / (distances + eps)  # (P, K)
#             weights_per_point = inv_distances / inv_distances.sum(
#                 dim=-1, keepdim=True
#             )  # (P, K)
#         return src2tgt, weights_per_point


@dataclass
class Graph:
    poss: torch.Tensor | None = None          # (N,3) float
    edges: torch.Tensor | None = None         # (E,2) long
    weights: torch.Tensor | None = None       # (E,)  float

    def __post_init__(self):
        if self.poss is None:
            self.poss = torch.empty((0, 3), dtype=torch.float32)
        if self.edges is None:
            self.edges = torch.empty((0, 2), dtype=torch.long)
        if self.weights is None:
            self.weights = torch.empty((0,), dtype=torch.float32)
        assert self.edges.ndim == 2 and self.edges.shape[1] == 2, "Edges must be (E,2)"
        assert self.edges.dtype == torch.long, "Edges must be torch.long"

    @property
    def device(self) -> torch.device:
        return self.poss.device

    def make_edges(
        self,
        K: int,
        *,
        weight_type: Literal["inv", "gaussian"] = "inv",
        sigma: float | None = None,
        eps: float = 1e-8,
        knn_func=None,  # expects: (src, tgt, k) -> (idx:(N,k), dist:(N,k))
    ):
        """
        Build a directed KNN graph (no self-loops) and row-normalized weights.
        edges: (E,2) with pairs [i, j] for each of the K neighbors of node i.
        weights: (E,), row-normalized over outgoing edges of i.
        """
        assert self.poss.ndim == 2 and self.poss.shape[1] == 3
        N = self.poss.shape[0]
        if N == 0:
            self.edges = torch.empty((0, 2), dtype=torch.long, device=self.device)
            self.weights = torch.empty((0,), dtype=self.poss.dtype, device=self.device)
            return

        if knn_func is None:
            knn_func = find_nearest_neighbors_faiss
        # K+1 to include self, then drop self neighbor (distance 0)
        idx, dist = knn_func(self.poss, self.poss, K + 1)  # (N, K+1)
        # Drop the first column (self)
        knn_idx  = idx[:, 1:]                              # (N, K)
        knn_dist = dist[:, 1:]                             # (N, K)

        # Compute per-edge weights
        if weight_type == "inv":
            w = 1.0 / (knn_dist.clamp_min(0.0) + eps)      # (N, K)
        elif weight_type == "gaussian":
            if sigma is None:
                # heuristic sigma: median distance over all edges
                sigma = knn_dist.median().item() + eps
            w = torch.exp(- (knn_dist ** 2) / (2.0 * (sigma ** 2)))
        else:
            raise ValueError("weight_type must be 'inv' or 'gaussian'")

        # Row-normalize: sum of outgoing weights per node i is 1
        w = w / (w.sum(dim=1, keepdim=True) + 1e-12)       # (N, K)

        # Build directed edge list: (i -> j) for all neighbors
        i_idx = torch.arange(N, device=self.device).unsqueeze(1).expand(-1, knn_idx.size(1))  # (N, K)
        edges = torch.stack((i_idx.reshape(-1), knn_idx.reshape(-1)), dim=1)                  # (N*K, 2)
        weights = w.reshape(-1)                                                                # (N*K,)

        self.edges = edges.to(dtype=torch.long)
        self.weights = weights.to(dtype=self.poss.dtype)

    def assign_nodes_to_points_by_knn(
        self,
        points: torch.Tensor,     # (P,3)
        K: int,
        *,
        weight_type: Literal["inv", "gaussian"] = "inv",
        sigma: float | None = None,
        eps: float = 1e-8,
        knn_func=None,            # expects: (src, tgt, k) -> (idx:(P,k), dist:(P,k))
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        For each point, pick K nearest graph nodes and return:
          indices: (P,K) long
          weights: (P,K) float (row-normalized)
        """
        assert points.ndim == 2 and points.shape[1] == 3
        if self.poss.numel() == 0:
            return (
                torch.empty((points.shape[0], K), dtype=torch.long, device=points.device),
                torch.zeros((points.shape[0], K), dtype=points.dtype, device=points.device),
            )

        idx, dist = knn_func(points, self.poss, K)         # (P, K)
        if weight_type == "inv":
            w = 1.0 / (dist.clamp_min(0.0) + eps)          # (P, K)
        elif weight_type == "gaussian":
            if sigma is None:
                sigma = dist.median().item() + eps
            w = torch.exp(- (dist ** 2) / (2.0 * (sigma ** 2)))
        else:
            raise ValueError("weight_type must be 'inv' or 'gaussian'")

        w = w / (w.sum(dim=1, keepdim=True) + 1e-12)       # row-normalize
        return idx.to(dtype=torch.long), w.to(dtype=points.dtype)