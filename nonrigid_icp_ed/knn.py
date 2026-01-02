import numpy as np
import torch
import faiss


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
