import numpy as np
import torch
import faiss
import open3d as o3d

def find_nearest_neighbors_faiss(
    src: torch.Tensor, tgt: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    # Use the FAISS 1.13.x-safe code we wrote earlier, but return distances too.
    S, T = src.shape[0], tgt.shape[0]
    if S == 0 or T == 0:
        dev, dt = src.device, src.dtype
        return (
            torch.empty((S, k), dtype=torch.long, device=dev),
            torch.empty((S, k), dtype=dt, device=dev),
        )

    src_np = np.ascontiguousarray(src.detach().to(torch.float32).cpu().numpy())
    tgt_np = np.ascontiguousarray(tgt.detach().to(torch.float32).cpu().numpy())
    d = tgt_np.shape[1]

    gpu_ok = (
        (src.is_cuda or tgt.is_cuda)
        and hasattr(faiss, "get_num_gpus")
        and faiss.get_num_gpus() > 0
        and hasattr(faiss, "GpuIndexFlatL2")
    )
    if gpu_ok:
        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.device = 0
        index = faiss.GpuIndexFlatL2(res, d, cfg)
    else:
        index = faiss.IndexFlatL2(d)

    index.add(tgt_np)  # base
    dist_np, idx_np = index.search(src_np, k)  # (S,k)
    idx = torch.from_numpy(idx_np).to(device=src.device, dtype=torch.long)
    dist = (
        torch.from_numpy(dist_np).to(device=src.device, dtype=src.dtype).sqrt()
    )  # L2 (Open3D-style)
    return idx, dist


def _torch_to_o3d_tensor(x: torch.Tensor, o3d_device: o3d.core.Device, dtype=o3d.core.Dtype.Float32):
    """
    Torch Tensor -> Open3D core Tensor.
    Uses DLPack when possible (zero-copy). Falls back to copy if needed.
    """
    # Ensure contiguous
    if not x.is_contiguous():
        x = x.contiguous()

    # If device matches and dtype is float32, try dlpack zero-copy
    try:
        if dtype == o3d.core.Dtype.Float32 and x.dtype == torch.float32:
            # Open3D device must match
            if (x.is_cuda and o3d_device.get_type() == o3d.core.Device.DeviceType.CUDA) or (
                (not x.is_cuda) and o3d_device.get_type() == o3d.core.Device.DeviceType.CPU
            ):
                return o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(x))
    except Exception:
        pass

    # Copy fallback: move to CPU/GPU to match and cast to float32
    x2 = x.detach().to(torch.float32)
    if o3d_device.get_type() == o3d.core.Device.DeviceType.CUDA:
        x2 = x2.to("cuda")
    else:
        x2 = x2.cpu()
    return o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(x2.contiguous()))


def _o3d_to_torch(x: o3d.core.Tensor, torch_device: torch.device, torch_dtype: torch.dtype):
    """
    Open3D core Tensor -> Torch Tensor via DLPack (zero-copy where possible).
    """
    t = torch.utils.dlpack.from_dlpack(x.to_dlpack())
    return t.to(device=torch_device, dtype=torch_dtype)


def find_nearest_neighbors_open3d(
    src: torch.Tensor, tgt: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Open3D(core.nns) KNN that is (practically) equivalent to the FAISS version:
      - base = tgt, query = src
      - returns (idx, dist) with shapes (S, k)
      - dist is L2 (sqrt of squared L2), cast to src.dtype, on src.device
      - if S==0 or T==0: returns empty tensors like the FAISS code
      - if k > T: pads with idx=-1 and dist=+inf (FAISS-style behavior)
    """
    S, T = int(src.shape[0]), int(tgt.shape[0])
    if S == 0 or T == 0:
        dev, dt = src.device, src.dtype
        return (
            torch.empty((S, k), dtype=torch.long, device=dev),
            torch.empty((S, k), dtype=dt, device=dev),
        )

    if src.ndim != 2 or tgt.ndim != 2:
        raise ValueError(f"src/tgt must be 2D (N, d). got src={tuple(src.shape)}, tgt={tuple(tgt.shape)}")
    if src.shape[1] != tgt.shape[1]:
        raise ValueError(f"dim mismatch: src.shape[1]={src.shape[1]} vs tgt.shape[1]={tgt.shape[1]}")
    if k <= 0:
        raise ValueError("k must be positive")

    # Choose Open3D device (prefer CUDA if either input is CUDA and Open3D CUDA is available)
    use_cuda = (src.is_cuda or tgt.is_cuda) and o3d.core.cuda.is_available()
    o3d_dev = o3d.core.Device("CUDA:0") if use_cuda else o3d.core.Device("CPU:0")

    # Open3D NNS expects float tensors
    src_o3d = _torch_to_o3d_tensor(src, o3d_dev, dtype=o3d.core.Dtype.Float32)
    tgt_o3d = _torch_to_o3d_tensor(tgt, o3d_dev, dtype=o3d.core.Dtype.Float32)

    # Build index on tgt, query src
    nns = o3d.core.nns.NearestNeighborSearch(tgt_o3d)
    nns.knn_index()

    k_eff = min(k, T)
    idx_o3d, dist2_o3d = nns.knn_search(src_o3d, k_eff)  # dist2 is squared L2

    # Convert to torch
    idx_eff = _o3d_to_torch(idx_o3d, src.device, torch.int64)         # (S, k_eff)
    dist2_eff = _o3d_to_torch(dist2_o3d, src.device, torch.float32)   # (S, k_eff)

    # Pad to (S, k) if needed (FAISS IndexFlatL2 returns k columns; missing neighbors -> -1 / inf)
    if k_eff < k:
        pad_n = k - k_eff
        idx_pad = torch.full((S, pad_n), -1, device=src.device, dtype=torch.long)
        dist2_pad = torch.full((S, pad_n), float("inf"), device=src.device, dtype=torch.float32)
        idx_eff = torch.cat([idx_eff, idx_pad], dim=1)
        dist2_eff = torch.cat([dist2_eff, dist2_pad], dim=1)

    # Return L2 distances (Open3D-style), cast to src.dtype
    dist = torch.sqrt(dist2_eff).to(device=src.device, dtype=src.dtype)
    return idx_eff, dist