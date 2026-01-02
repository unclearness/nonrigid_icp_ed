import torch


def warp_embedded_deformation(
    points: torch.Tensor,          # (P, 3)
    anchor_poss: torch.Tensor,     # (N, 3)
    node_Rs: torch.Tensor,         # (N, 3, 3)
    node_ts: torch.Tensor,         # (N, 3)
    anchor_indices: torch.Tensor,  # (P, K)
    anchor_weights: torch.Tensor,  # (P, K)
) -> torch.Tensor:
    """
    Warp points using embedded deformation with per-point anchor indices.
    Based on Sumner et al. 2007 (Embedded Deformation for Shape Manipulation).
    Each point is influenced by K nearby graph nodes.
    """
    # gather anchor attributes for each point
    g_k = anchor_poss[anchor_indices]    # (P, K, 3)
    R_k = node_Rs[anchor_indices]        # (P, K, 3, 3)
    t_k = node_ts[anchor_indices]        # (P, K, 3)

    # (P, K, 3): p - g_k
    diff = points[:, None, :] - g_k

    # (P, K, 3): R_k * (p - g_k)
    rotated = torch.einsum('pkij,pkj->pki', R_k, diff)

    # (P, K, 3): transformed anchors
    deformed_per_anchor = rotated + g_k + t_k

    # (P, 3): weighted blend
    deformed_points = (deformed_per_anchor * anchor_weights[..., None]).sum(dim=1)

    return deformed_points
