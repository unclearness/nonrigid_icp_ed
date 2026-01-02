import os
import random

import open3d as o3d
import torch
import torch.nn.functional as F
import numpy as np


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def umeyama(src, dst, estimate_scale):
    """Estimate N-D similarity transformation (R, t, s) with or without scaling.
    Parameters
    ----------
    src : (M, N) array_like
        Source coordinates.
    dst : (M, N) array_like
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
    Returns
    -------
    T : (N + 1, N + 1)
        Homogeneous similarity transformation matrix.
    R : (N, N)
        Rotation matrix (orthonormal).
    t : (N,)
        Translation vector.
    s : float
        Isotropic scale factor.
    References
    ----------
    .. [1] Umeyama, "Least-squares estimation of transformation parameters
           between two point patterns", PAMI 1991.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)

    num = src.shape[0]
    dim = src.shape[1]

    # Means
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Demean
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38)
    A = (dst_demean.T @ src_demean) / num

    # Eq. (39)
    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1.0

    # SVD
    U, S, Vh = np.linalg.svd(A)
    # Note: numpy returns Vh = V^T

    # Rotation (unscaled)
    rank = np.linalg.matrix_rank(A)
    T = np.eye(dim + 1, dtype=np.float64)

    if rank == 0:
        R = np.full((dim, dim), np.nan, dtype=np.float64)
        t = np.full((dim,), np.nan, dtype=np.float64)
        s = np.nan
        return T * np.nan, R, t, s
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(Vh) > 0:
            R = U @ Vh
        else:
            d2 = d.copy()
            d2[dim - 1] = -1.0
            R = U @ np.diag(d2) @ Vh
    else:
        R = U @ np.diag(d) @ Vh

    # Scale
    if estimate_scale:
        # Eq. (41) and (42)
        var_src = src_demean.var(axis=0).sum()
        s = (S @ d) / var_src
    else:
        s = 1.0

    # Translation
    t = dst_mean - s * (R @ src_mean)

    # Assemble T (scaled rotation + translation)
    T[:dim, :dim] = s * R
    T[:dim, dim] = t

    return T, R, t, s


def set_random_seed(seed: int = 42) -> None:
    # --- Python ---
    random.seed(seed)

    # --- Python Hash ---
    os.environ["PYTHONHASHSEED"] = str(seed)

    # --- NumPy and Scipy ---
    np.random.seed(seed)

    # --- Open3D ---
    o3d.utility.random.seed(seed)

    # --- PyTorch ---
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # --- CuDNN ---
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # --- CuBLAS ---
    # https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # --- PyTorch 2.0+ (optional) ---
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)
