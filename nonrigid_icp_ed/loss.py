from typing import Literal

import torch
import torch.nn.functional as F

from nonrigid_icp_ed.warp import warp_embedded_deformation


def compute_landmark_loss(
    src_landmarks: torch.Tensor,
    tgt_landmarks: torch.Tensor,
    anchor_poss: torch.Tensor,
    node_Rs: torch.Tensor,
    node_ts: torch.Tensor,
    anchor_indices: torch.Tensor,
    anchor_weights: torch.Tensor,
) -> torch.Tensor:
    warped_src_landmarks = warp_embedded_deformation(
        src_landmarks, anchor_poss, node_Rs, node_ts, anchor_indices, anchor_weights
    )
    landmark_loss = torch.mean(
        torch.sum((warped_src_landmarks - tgt_landmarks) ** 2, dim=-1)
    )
    return landmark_loss


def compute_arap_loss(
    node_Rs: torch.Tensor,  # (N, 3, 3)
    node_ts: torch.Tensor,  # (N, 3)
    graph_nodes: torch.Tensor,  # (N, 3)
    graph_edges: torch.Tensor,  # (E, 2)  [src=i, tgt=j]
    graph_edges_weights: torch.Tensor,  # (E,)
) -> torch.Tensor:
    """
    ARAP on ED graph: sum_ij w_ij || R_i (g_j - g_i) + t_i - t_j - (g_j - g_i) ||^2 / sum w_ij
    """
    i = graph_edges[:, 0]
    j = graph_edges[:, 1]

    d_ij = graph_nodes[j] - graph_nodes[i]  # (E, 3)
    R_i = node_Rs[i]  # (E, 3, 3)
    t_i = node_ts[i]  # (E, 3)
    t_j = node_ts[j]  # (E, 3)

    Rd = torch.bmm(R_i, d_ij.unsqueeze(-1)).squeeze(-1)  # (E, 3)
    resid = Rd + t_i - t_j - d_ij  # (E, 3)

    sq = (resid * resid).sum(dim=-1)  # (E,)
    w = graph_edges_weights
    loss = (w * sq).sum() / (w.sum().clamp_min(1e-12))
    return loss


def compute_truncated_chamfer_distance_without_reduction(
    src_points: torch.Tensor,  # (S, 3)
    tgt_points: torch.Tensor,  # (T, 3)
    src2tgt_correspondence: torch.Tensor,  # (S, K) long
    tgt2src_correspondence: torch.Tensor,  # (T, K) long
    trunc_th: float = -1.0,  # <0: no truncation; >=0: clamp L2 distance by trunc_th
    ignore_index: int = -99999,  # used to skip invalid indices (e.g., -1)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Chamfer distance in Open3D style:
        CD = mean_p min_q ||p - q||_2  +  mean_q min_p ||q - p||_2
    - Non-squared L2 distance.
    - Each direction is averaged separately, then summed (no 1/2 factor).
    - If K>1 correspondences are provided, the minimum (nearest neighbor) is used.
    """

    # --- sanity checks ---
    assert (
        src2tgt_correspondence.dim() == 2 and tgt2src_correspondence.dim() == 2
    ), "Correspondence tensors must be 2D"
    S, K_s = src2tgt_correspondence.shape
    T, K_t = tgt2src_correspondence.shape
    assert S == src_points.size(
        0
    ), f"Source correspondence size does not match source points: {S} vs {src_points.size(0)}"
    assert T == tgt_points.size(
        0
    ), f"Target correspondence size does not match target points: {T} vs {tgt_points.size(0)}"
    assert (
        src_points.shape[-1] == 3 and tgt_points.shape[-1] == 3
    ), f"Points must be of shape (N, 3): got {src_points.shape} and {tgt_points.shape}"
    assert (
        src2tgt_correspondence.dtype == torch.long
        and tgt2src_correspondence.dtype == torch.long
    ), "Correspondence indices must be torch.long"

    K = K_s

    # --- optional ignore index handling ---
    has_ignore = (src2tgt_correspondence == ignore_index).any() or (
        tgt2src_correspondence == ignore_index
    ).any()
    s_idx = src2tgt_correspondence
    t_idx = tgt2src_correspondence
    s_mask, t_mask = None, None
    if has_ignore:
        s_idx = src2tgt_correspondence.clone()
        t_idx = tgt2src_correspondence.clone()
        s_mask = s_idx == ignore_index
        t_mask = t_idx == ignore_index
        if s_mask.any():
            s_idx[s_mask] = 0
        if t_mask.any():
            t_idx[t_mask] = 0

    # --- gather corresponding points ---
    # (S, K, 3)
    tgt_corr_points = tgt_points[s_idx.reshape(-1)].reshape(S, K, 3)
    # (T, K, 3)
    src_corr_points = src_points[t_idx.reshape(-1)].reshape(T, K, 3)

    # --- compute squared distances ---
    d2_s2t = ((src_points[:, None, :] - tgt_corr_points) ** 2).sum(dim=-1)
    d2_t2s = ((tgt_points[:, None, :] - src_corr_points) ** 2).sum(dim=-1)

    # Mask invalid indices as +inf to exclude them from min operation
    if has_ignore:
        assert s_mask is not None and t_mask is not None
        if s_mask.any():
            d2_s2t = d2_s2t.masked_fill(s_mask, float("inf"))
        if t_mask.any():
            d2_t2s = d2_t2s.masked_fill(t_mask, float("inf"))

    # --- take nearest neighbor distance (min over K) ---
    min_d2_s2t = d2_s2t.amin(dim=-1)  # (S,)
    min_d2_t2s = d2_t2s.amin(dim=-1)  # (T,)

    # Convert squared distance to L2 distance
    min_d_s2t = torch.sqrt(min_d2_s2t)
    min_d_t2s = torch.sqrt(min_d2_t2s)

    # --- truncation (applied on L2 distances) ---
    if trunc_th >= 0.0:
        min_d_s2t = torch.clamp(min_d_s2t, max=float(trunc_th))
        min_d_t2s = torch.clamp(min_d_t2s, max=float(trunc_th))

    # --- no reduction: return both directions separately ---

    return min_d_s2t, min_d_t2s


def compute_truncated_chamfer_distance(
    src_points: torch.Tensor,  # (S, 3)
    tgt_points: torch.Tensor,  # (T, 3)
    src2tgt_correspondence: torch.Tensor,  # (S, K) long
    tgt2src_correspondence: torch.Tensor,  # (T, K) long
    trunc_th: float = -1.0,  # <0: no truncation; >=0: clamp L2 distance by trunc_th
    reduction: Literal["mean", "sum"] = "mean",
    ignore_index: int = -99999,  # used to skip invalid indices (e.g., -1)
) -> torch.Tensor:

    min_d_s2t, min_d_t2s = compute_truncated_chamfer_distance_without_reduction(
        src_points,
        tgt_points,
        src2tgt_correspondence,
        tgt2src_correspondence,
        trunc_th,
        ignore_index,
    )

    # --- reduction (Open3D: mean per direction, sum of both) ---
    if reduction == "mean":
        loss_s = torch.nanmean(
            min_d_s2t.masked_fill(torch.isinf(min_d_s2t), float("nan"))
        )
        loss_t = torch.nanmean(
            min_d_t2s.masked_fill(torch.isinf(min_d_t2s), float("nan"))
        )
    elif reduction == "sum":
        loss_s = torch.sum(min_d_s2t)
        loss_t = torch.sum(min_d_t2s)
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")

    # Final Chamfer distance = mean(src→tgt) + mean(tgt→src)
    return loss_s + loss_t


def compute_unique_edges(F: torch.Tensor) -> torch.Tensor:
    # --- build unique edges ---
    e01 = F[:, [0, 1]]
    e12 = F[:, [1, 2]]
    e20 = F[:, [2, 0]]
    edges = torch.cat([e01, e12, e20], dim=0)      # (3F,2)
    edges = torch.sort(edges, dim=1).values
    edges = torch.unique(edges, dim=0)             # (E,2)
    return edges


def compute_edge_length_uniform_loss(
    V: torch.Tensor,          # (N,3), requires_grad=True
    *,
    F: torch.Tensor | None = None,          # (F,3), torch.long
    edges : torch.Tensor | None = None,  # (E,2), torch.long
    target_length: float | None = None,
    target_mode: str = "mean",   # "median" or "mean"
) -> torch.Tensor:
    """
    Edge length uniformization loss.

    Returns:
        scalar torch.Tensor
    """
    assert V.ndim == 2 and V.shape[1] == 3
    if F is not None:
        assert F.ndim == 2 and F.shape[1] == 3
        assert F.dtype == torch.long
    assert (edges is None) != (F is None), "Either edges or F must be provided, but not both."

    if edges is None and F is not None:
        edges = compute_unique_edges(F)

    assert edges is not None
    assert edges.ndim == 2 and edges.shape[1] == 2
    assert edges.dtype == torch.long
    vi = V[edges[:, 0]]
    vj = V[edges[:, 1]]

    # edge lengths
    lengths = torch.norm(vi - vj, dim=1)            # (E,)

    # --- target length ---
    if target_length is None:
        if target_mode == "median":
            L = lengths.detach().median()
        elif target_mode == "mean":
            L = lengths.detach().mean()
        else:
            raise ValueError("target_mode must be 'median' or 'mean'")
    else:
        L = torch.tensor(
            target_length,
            device=V.device,
            dtype=V.dtype,
        )

    # squared deviation
    loss = ((lengths - L) ** 2).mean()

    return loss


@torch.no_grad()
def build_adjacent_face_pairs(F: torch.Tensor) -> torch.Tensor:
    """
    F: (T,3) long
    Returns:
        adj: (A,2) long, each row is (face_id0, face_id1) sharing an edge
        (boundary edges are ignored)
    """
    assert F.dtype == torch.long and F.ndim == 2 and F.shape[1] == 3
    T = F.shape[0]

    # collect undirected edges -> face id
    # key = (min(u,v), max(u,v))
    edge2faces = {}
    F_cpu = F.detach().cpu()

    for fid in range(T):
        a, b, c = F_cpu[fid].tolist()
        edges = [(a, b), (b, c), (c, a)]
        for u, v in edges:
            if u > v:
                u, v = v, u
            key = (u, v)
            if key not in edge2faces:
                edge2faces[key] = [fid]
            else:
                edge2faces[key].append(fid)

    pairs = []
    for faces in edge2faces.values():
        if len(faces) == 2:      # internal edge
            pairs.append((faces[0], faces[1]))
        # non-manifold (>2) は必要ならここで全部組み合わせる等、用途に応じて

    if len(pairs) == 0:
        return torch.empty((0, 2), dtype=torch.long, device=F.device)

    return torch.tensor(pairs, dtype=torch.long, device=F.device)


def compute_adjacent_normal_consistency_loss(
    V: torch.Tensor,                 # (N,3) requires_grad=True
    F_tri: torch.Tensor,             # (T,3) long
    adj_faces: torch.Tensor,         # (A,2) long (from build_adjacent_face_pairs)
    *,
    cos_margin: float = 0.0,         # 0 => forbid >90deg flips, 0.2=>stricter
    beta: float = 50.0,              # softplus hardness
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Encourage adjacent face normals to have non-negative dot product:
        dot(n0, n1) >= cos_margin
    This prevents local fold-over / "inside-out" between neighboring triangles.

    Returns scalar loss.
    """
    if adj_faces.numel() == 0:
        return V.new_zeros(())

    i0, i1, i2 = F_tri[:, 0], F_tri[:, 1], F_tri[:, 2]
    v0, v1, v2 = V[i0], V[i1], V[i2]

    # face normals (area-weighted)
    n = torch.cross(v1 - v0, v2 - v0, dim=1)  # (T,3)
    n = n / torch.norm(n, dim=1, keepdim=True).clamp_min(eps)  # unit normals

    f0 = adj_faces[:, 0]
    f1 = adj_faces[:, 1]
    dot = (n[f0] * n[f1]).sum(dim=1)  # (A,)

    # barrier: penalize if dot < cos_margin
    loss = F.softplus((cos_margin - dot) * beta).mean() / beta
    return loss