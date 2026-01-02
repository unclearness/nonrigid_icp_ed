import numpy as np
import torch
import faiss


# def find_nearest_neighbors(src_pcd: torch.Tensor, tgt_pcd: torch.Tensor, tgt_kdtree : o3d.geometry.KDTreeFlann | None = None) -> tuple[torch.Tensor, torch.Tensor]:
#     """
#     Find nearest neighbors from src_pcd to tgt_pcd and vice versa.
#     Args:
#         src_pcd: (S, 3) source point cloud
#         tgt_pcd: (T, 3) target point cloud
#     Returns:
#         src2tgt_indices: (S,) indices of nearest neighbors in tgt_pcd for each point in src_pcd
#         tgt2src_indices: (T,) indices of nearest neighbors
#     """
#     src_o3d = o3d.geometry.PointCloud()
#     src_o3d.points = o3d.utility.Vector3dVector(src_pcd.cpu().numpy())

#     tgt_o3d = o3d.geometry.PointCloud()
#     tgt_o3d.points = o3d.utility.Vector3dVector(tgt_pcd.cpu().numpy())
#     if tgt_kdtree is None:
#         tgt_kdtree = o3d.geometry.KDTreeFlann(tgt_o3d)

#     src2tgt_indices = []
#     for point in src_o3d.points:
#         [_, idx, _] = tgt_kdtree.search_knn_vector_3d(point, 1)
#         src2tgt_indices.append(idx[0])
#     src2tgt_indices = torch.tensor(src2tgt_indices, device=src_pcd.device, dtype=torch.long)

#     src_kdtree = o3d.geometry.KDTreeFlann(src_o3d)
#     tgt2src_indices = []
#     for point in tgt_o3d.points:
#         [_, idx, _] = src_kdtree.search_knn_vector_3d(point, 1)
#         tgt2src_indices.append(idx[0])
#     tgt2src_indices = torch.tensor(tgt2src_indices, device=tgt_pcd.device, dtype=torch.long)

#     return src2tgt_indices, tgt2src_indices


# def find_nearest_neighbors_faiss(
#     src_pcd: torch.Tensor,  # (S, 3)
#     tgt_pcd: torch.Tensor,  # (T, 3)
#     K: int = 1,
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     """
#     FAISS-based exact 1-NN in both directions (src->tgt and tgt->src), CPU or GPU.
#     - Uses IndexFlatL2 (exact brute-force L2).
#     - If FAISS-GPU is available *and* any input is on CUDA, it uses a GPU index.
#     - Returns indices only (to match your original signature).
#       Distances can be added easily if you need them later.
#     Notes:
#       * Inputs are converted to contiguous float32 NumPy arrays for FAISS.
#       * Returned tensors are placed on the same device as the corresponding source cloud,
#         i.e., src2tgt -> src_pcd.device, tgt2src -> tgt_pcd.device.
#       * `tgt_kdtree` param is ignored.
#     """
#     # --- sanity checks ---
#     assert src_pcd.ndim == 2 and src_pcd.shape[1] == 3, "src_pcd must be (S,3)"
#     assert tgt_pcd.ndim == 2 and tgt_pcd.shape[1] == 3, "tgt_pcd must be (T,3)"
#     S, T = src_pcd.shape[0], tgt_pcd.shape[0]
#     if S == 0 or T == 0:
#         # Empty-safe return
#         return (
#             torch.empty((S,), dtype=torch.long, device=src_pcd.device),
#             torch.empty((T,), dtype=torch.long, device=tgt_pcd.device),
#         )

#     # --- prepare NumPy float32 (FAISS expects float32, row-major) ---
#     # (FAISS works on CPU arrays; GPU wrapper will copy as needed.)
#     src_np = np.ascontiguousarray(src_pcd.detach().to("cpu", torch.float32).numpy())
#     tgt_np = np.ascontiguousarray(tgt_pcd.detach().to("cpu", torch.float32).numpy())

#     # --- helper: build (CPU or GPU) FlatL2 index and search ---
#     def faiss_search(query_np: np.ndarray, base_np: np.ndarray, gpu: bool):
#         d = base_np.shape[1]
#         index = faiss.IndexFlatL2(d)  # exact L2
#         if gpu and hasattr(faiss, "StandardGpuResources"):
#             res = faiss.StandardGpuResources()
#             # Use GPU 0 by default; customize if you have multi-GPU
#             index = faiss.index_cpu_to_gpu(res, 0, index)
#         index.add(base_np)  # add base vectors
#         dist, idx = index.search(query_np, K)  # k=1
#         # idx: (N,1) int64; dist: (N,1) float32
#         return idx.reshape(-1)

#     # Decide GPU use: if any tensor is on CUDA and FAISS-GPU is available
#     use_gpu = (src_pcd.is_cuda or tgt_pcd.is_cuda) and hasattr(
#         faiss, "StandardGpuResources"
#     )

#     # --- src -> tgt ---
#     s2t_idx_np = faiss_search(src_np, tgt_np, gpu=use_gpu)
#     # --- tgt -> src ---
#     t2s_idx_np = faiss_search(tgt_np, src_np, gpu=use_gpu)

#     # --- to torch, on corresponding devices ---
#     src2tgt_indices = torch.from_numpy(s2t_idx_np).to(
#         device=src_pcd.device, dtype=torch.long
#     )
#     tgt2src_indices = torch.from_numpy(t2s_idx_np).to(
#         device=tgt_pcd.device, dtype=torch.long
#     )

#     return src2tgt_indices, tgt2src_indices


# def find_nearest_neighbors_faiss(
#     src_pcd: torch.Tensor,  # (S, 3)
#     tgt_pcd: torch.Tensor,  # (T, 3)
#     K: int = 1,
# ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
#     """
#     FAISS-based exact K-NN (src→tgt and tgt→src) supporting FAISS ≥1.13.x.
#     • Uses GpuIndexFlatL2 if FAISS-GPU and CUDA device available, else IndexFlatL2 (CPU).
#     • Returns indices only, shaped (S,K) and (T,K).
#     """
#     assert src_pcd.ndim == 2 and src_pcd.shape[1] == 3, f"src_pcd must be (S,3): {src_pcd.shape}"
#     assert tgt_pcd.ndim == 2 and tgt_pcd.shape[1] == 3, f"tgt_pcd must be (T,3): {tgt_pcd.shape}"
#     assert K >= 1

#     S, T = src_pcd.shape[0], tgt_pcd.shape[0]
#     if S == 0 or T == 0:
#         return (
#             torch.empty((S, K), dtype=torch.long, device=src_pcd.device),
#             torch.empty((T, K), dtype=torch.long, device=tgt_pcd.device),
#             np.empty((S, K), dtype=np.float32),
#             np.empty((T, K), dtype=np.float32),
#         )

#     # Prepare float32 contiguous arrays
#     src_np = np.ascontiguousarray(src_pcd.detach().to(torch.float32).cpu().numpy())
#     tgt_np = np.ascontiguousarray(tgt_pcd.detach().to(torch.float32).cpu().numpy())

#     def _flatL2_knn(
#         query_np: np.ndarray, base_np: np.ndarray, use_gpu: bool
#     ) -> np.ndarray:
#         d = base_np.shape[1]
#         gpu_ok = (
#             use_gpu
#             and hasattr(faiss, "get_num_gpus")
#             and faiss.get_num_gpus() > 0
#             and hasattr(faiss, "GpuIndexFlatL2")
#         )

#         if gpu_ok:
#             # Modern GPU path (FAISS 1.13+)
#             res = faiss.StandardGpuResources()
#             cfg = faiss.GpuIndexFlatConfig()
#             cfg.device = 0
#             index = faiss.GpuIndexFlatL2(res, d, cfg)
#         else:
#             # CPU fallback
#             index = faiss.IndexFlatL2(d)

#         index.add(base_np)
#         distacnes, idx = index.search(query_np, K)
#         return idx, distacnes  # (N, K)

#     want_gpu = src_pcd.is_cuda or tgt_pcd.is_cuda
#     s2t_idx_np, s2t_dist_np = _flatL2_knn(src_np, tgt_np, use_gpu=want_gpu)
#     t2s_idx_np, t2s_dist_np = _flatL2_knn(tgt_np, src_np, use_gpu=want_gpu)

#     src2tgt = torch.from_numpy(s2t_idx_np).to(device=src_pcd.device, dtype=torch.long)
#     tgt2src = torch.from_numpy(t2s_idx_np).to(device=tgt_pcd.device, dtype=torch.long)
#     return src2tgt, tgt2src, s2t_dist_np, t2s_dist_np


def find_nearest_neighbors_faiss(src: torch.Tensor, tgt: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Use the FAISS 1.13.x-safe code we wrote earlier, but return distances too.
    S, T = src.shape[0], tgt.shape[0]
    if S == 0 or T == 0:
        dev, dt = src.device, src.dtype
        return (torch.empty((S, k), dtype=torch.long, device=dev),
                torch.empty((S, k), dtype=dt, device=dev))

    src_np = np.ascontiguousarray(src.detach().to(torch.float32).cpu().numpy())
    tgt_np = np.ascontiguousarray(tgt.detach().to(torch.float32).cpu().numpy())
    d = tgt_np.shape[1]

    gpu_ok = (src.is_cuda or tgt.is_cuda) and hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0 \
             and hasattr(faiss, "GpuIndexFlatL2")
    if gpu_ok:
        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig(); cfg.device = 0
        index = faiss.GpuIndexFlatL2(res, d, cfg)
    else:
        index = faiss.IndexFlatL2(d)

    index.add(tgt_np)                          # base
    dist_np, idx_np = index.search(src_np, k)  # (S,k)
    idx = torch.from_numpy(idx_np).to(device=src.device, dtype=torch.long)
    dist = torch.from_numpy(dist_np).to(device=src.device, dtype=src.dtype).sqrt()  # L2 (Open3D-style)
    return idx, dist