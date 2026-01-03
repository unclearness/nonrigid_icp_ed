from dataclasses import dataclass, field
from typing import Literal

import torch

from nonrigid_icp_ed.knn import find_nearest_neighbors_faiss


def _neighbor_offsets(connectivity: int, device):
    if connectivity == 6:
        offsets = torch.tensor(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            device=device,
            dtype=torch.long,
        )
    elif connectivity == 26:
        o = torch.tensor([-1, 0, 1], device=device)
        grid = torch.stack(torch.meshgrid(o, o, o, indexing="ij"), dim=-1).reshape(
            -1, 3
        )
        offsets = grid[(grid != 0).any(dim=1)]
    else:
        raise ValueError("connectivity must be 6 or 26")
    return offsets  # (M,3)


def _make_grid_edges_vectorized(
    div_x: int,
    div_y: int,
    div_z: int,
    *,
    connectivity: int = 6,
    device: torch.device,
):
    # grid indices
    ix = torch.arange(div_x, device=device)
    iy = torch.arange(div_y, device=device)
    iz = torch.arange(div_z, device=device)

    gx, gy, gz = torch.meshgrid(ix, iy, iz, indexing="ij")
    gx = gx.reshape(-1)
    gy = gy.reshape(-1)
    gz = gz.reshape(-1)

    # linear index
    i = gx * (div_y * div_z) + gy * div_z + gz  # (N,)

    offsets = _neighbor_offsets(connectivity, device)  # (M,3)
    M = offsets.shape[0]

    # broadcast neighbors
    jx = gx[:, None] + offsets[None, :, 0]
    jy = gy[:, None] + offsets[None, :, 1]
    jz = gz[:, None] + offsets[None, :, 2]

    valid = (
        (jx >= 0) & (jx < div_x) & (jy >= 0) & (jy < div_y) & (jz >= 0) & (jz < div_z)
    )

    j = jx * (div_y * div_z) + jy * div_z + jz

    # flatten
    i_flat = i[:, None].expand_as(j)[valid]
    j_flat = j[valid]

    # avoid duplicates (undirected graph)
    mask = i_flat < j_flat
    edges = torch.stack([i_flat[mask], j_flat[mask]], dim=1)

    return edges  # (E,2)


@dataclass
class Graph:
    poss: torch.Tensor = field(
        default_factory=lambda: torch.empty((0, 3), dtype=torch.float32)
    )  # (N,3)
    Rs: torch.Tensor = field(
        default_factory=lambda: torch.empty((0, 3, 3), dtype=torch.float32)
    )  # (N,3,3)
    ts: torch.Tensor = field(
        default_factory=lambda: torch.empty((0, 3), dtype=torch.float32)
    )  # (N,3)
    edges: torch.Tensor = field(
        default_factory=lambda: torch.empty((0, 2), dtype=torch.long)
    )  # (E,2)
    weights: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.float32)
    )  # (E,)

    def __post_init__(self):
        assert self.poss.ndim == 2 and self.poss.shape[1] == 3, "poss must be (N,3)"
        assert self.edges.ndim == 2 and self.edges.shape[1] == 2, "edges must be (E,2)"
        assert self.edges.dtype == torch.long, "edges must be torch.long"
        assert self.weights.ndim == 1, "weights must be (E,)"

    @property
    def device(self) -> torch.device:
        return self.poss.device

    def to_dict(self) -> dict:
        return {
            "poss": self.poss,
            "Rs": self.Rs,
            "ts": self.ts,
            "edges": self.edges,
            "weights": self.weights,
        }

    @staticmethod
    def from_dict(d: dict) -> "Graph":
        return Graph(
            poss=d["poss"],
            Rs=d["Rs"],
            ts=d["ts"],
            edges=d["edges"],
            weights=d["weights"],
        )

    def to(self, device: torch.device) -> "Graph":
        return Graph(
            poss=self.poss.to(device),
            Rs=self.Rs.to(device),
            ts=self.ts.to(device),
            edges=self.edges.to(device),
            weights=self.weights.to(device),
        )

    def init_as_grid(
        self,
        size_x: float,
        size_y: float,
        size_z: float,
        div_x: int,
        div_y: int,
        div_z: int,
        *,
        connectivity: Literal[6, 26] = 6,
        weight_type: Literal["inv", "gaussian"] = "inv",
        sigma: float | None = None,
        eps: float = 1e-8,
        translation: torch.Tensor | None = None,
        rotation: torch.Tensor | None = None,
    ) -> None:
        """
        Initialize graph nodes, edges, and weights on a regular 3D grid.

        Nodes:
            Regular grid in [0,size_x] x [0,size_y] x [0,size_z]

        Edges:
            6-neighborhood (axis-aligned) or 26-neighborhood

        Weights:
            Distance-based (inverse or Gaussian)

        Sets:
            self.poss    : (N,3)
            self.edges   : (E,2)  (undirected, i < j)
            self.weights : (E,)
        """

        device = self.device

        # ------------------------------------------------------------
        # 1. Create grid nodes
        # ------------------------------------------------------------
        xs = torch.linspace(0.0, size_x, div_x, device=device)
        ys = torch.linspace(0.0, size_y, div_y, device=device)
        zs = torch.linspace(0.0, size_z, div_z, device=device)

        grid_x, grid_y, grid_z = torch.meshgrid(xs, ys, zs, indexing="ij")
        poss = torch.stack(
            [grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)],
            dim=1,
        )  # (N,3)
        self.poss = poss

        if translation is not None:
            assert translation.shape == (3,)
            self.poss += translation

        if rotation is not None:
            assert rotation.shape == (3, 3)
            self.poss = self.poss @ rotation.T

        self.Rs = (
            torch.eye(3, device=device).unsqueeze(0).expand(self.poss.shape[0], -1, -1)
        )  # (N,3,3)
        self.ts = torch.zeros(self.poss.shape[0], 3, device=device)  # (N,3)

        # ------------------------------------------------------------
        # 2. Build edges
        # ------------------------------------------------------------
        edges = _make_grid_edges_vectorized(
            div_x=div_x,
            div_y=div_y,
            div_z=div_z,
            connectivity=connectivity,
            device=self.device,
        )
        self.edges = edges

        # ------------------------------------------------------------
        # 3. Compute edge weights
        # ------------------------------------------------------------
        p0 = poss[edges[:, 0]]  # (E,3)
        p1 = poss[edges[:, 1]]  # (E,3)
        d = torch.linalg.norm(p1 - p0, dim=1)  # (E,)

        if weight_type == "inv":
            weights = 1.0 / (d + eps)
        elif weight_type == "gaussian":
            if sigma is None:
                # heuristic: grid spacing
                dx = size_x / max(div_x - 1, 1)
                dy = size_y / max(div_y - 1, 1)
                dz = size_z / max(div_z - 1, 1)
                sigma = 0.5 * (dx + dy + dz)
            weights = torch.exp(-(d**2) / (2.0 * sigma**2))
        else:
            raise ValueError("weight_type must be 'inv' or 'gaussian'")

        self.weights = weights

    def init_edges_and_weights_by_knn_from_poss(
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
            self.Rs = torch.empty((0, 3, 3), dtype=self.poss.dtype, device=self.device)
            self.ts = torch.empty((0, 3), dtype=self.poss.dtype, device=self.device)
            return

        if knn_func is None:
            knn_func = find_nearest_neighbors_faiss
        # K+1 to include self, then drop self neighbor (distance 0)
        idx, dist = knn_func(self.poss, self.poss, K + 1)  # (N, K+1)
        # Drop the first column (self)
        knn_idx = idx[:, 1:]  # (N, K)
        knn_dist = dist[:, 1:]  # (N, K)

        # Compute per-edge weights
        if weight_type == "inv":
            w = 1.0 / (knn_dist.clamp_min(0.0) + eps)  # (N, K)
        elif weight_type == "gaussian":
            if sigma is None:
                # heuristic sigma: median distance over all edges
                sigma = knn_dist.median().item() + eps
            w = torch.exp(-(knn_dist**2) / (2.0 * (sigma**2)))
        else:
            raise ValueError("weight_type must be 'inv' or 'gaussian'")

        # Row-normalize: sum of outgoing weights per node i is 1
        w = w / (w.sum(dim=1, keepdim=True) + 1e-12)  # (N, K)

        # Build directed edge list: (i -> j) for all neighbors
        i_idx = (
            torch.arange(N, device=self.device).unsqueeze(1).expand(-1, knn_idx.size(1))
        )  # (N, K)
        edges = torch.stack((i_idx.reshape(-1), knn_idx.reshape(-1)), dim=1)  # (N*K, 2)
        weights = w.reshape(-1)  # (N*K,)

        self.edges = edges.to(dtype=torch.long)
        self.weights = weights.to(dtype=self.poss.dtype)
        self.Rs = (
            torch.eye(3, device=self.device)
            .unsqueeze(0)
            .expand(self.poss.shape[0], -1, -1)
        )  # (N,3,3)
        self.ts = torch.zeros(self.poss.shape[0], 3, device=self.device)  # (N,3)

    def assign_nodes_to_points_by_knn(
        self,
        points: torch.Tensor,  # (P,3)
        K: int,
        *,
        weight_type: Literal["inv", "gaussian"] = "inv",
        sigma: float | None = None,
        eps: float = 1e-8,
        knn_func=None,  # expects: (src, tgt, k) -> (idx:(P,k), dist:(P,k))
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        For each point, pick K nearest graph nodes and return:
          indices: (P,K) long
          weights: (P,K) float (row-normalized)
        """
        assert points.ndim == 2 and points.shape[1] == 3
        if self.poss.numel() == 0:
            return (
                torch.empty(
                    (points.shape[0], K), dtype=torch.long, device=points.device
                ),
                torch.zeros(
                    (points.shape[0], K), dtype=points.dtype, device=points.device
                ),
            )

        if knn_func is None:
            knn_func = find_nearest_neighbors_faiss

        idx, dist = knn_func(points, self.poss, K)  # (P, K)
        if weight_type == "inv":
            w = 1.0 / (dist.clamp_min(0.0) + eps)  # (P, K)
        elif weight_type == "gaussian":
            if sigma is None:
                sigma = dist.median().item() + eps
            w = torch.exp(-(dist**2) / (2.0 * (sigma**2)))
        else:
            raise ValueError("weight_type must be 'inv' or 'gaussian'")

        w = w / w.sum(dim=1, keepdim=True)  # row-normalize
        return idx.to(dtype=torch.long), w.to(dtype=points.dtype)
