from pathlib import Path

import open3d as o3d
import numpy as np
import torch

from nonrigid_icp_ed.graph import Graph
from nonrigid_icp_ed.io import (
    export_graph_as_lines,
    export_graph_as_mesh,
    import_wrap3_json,
)
from nonrigid_icp_ed.util import umeyama, set_random_seed
from nonrigid_icp_ed.registration import NonRigidICP, OptimizationHistory
from nonrigid_icp_ed.config import NonrigidIcpEdConfig

# ------------------------------------------------------------
# Fade function
# Same quintic polynomial as Ken Perlin's original implementation:
#   fade(t) = 6t^5 - 15t^4 + 10t^3
# This guarantees C1 continuity across lattice cell boundaries.
# ------------------------------------------------------------
def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

def lerp(a, b, t):
    return a + t * (b - a)

# ------------------------------------------------------------
# Gradient generation
#
# DIFFERENCE FROM CLASSIC PERLIN:
# - Classic Perlin uses a fixed permutation table (size 256)
#   and a small predefined set of gradient directions.
# - Here, gradients are generated procedurally using a hash
#   of integer lattice coordinates -> angle -> (cos, sin).
#
# PRO:
# - No permutation table needed
# - Unlimited domain size
# - Very compact implementation
#
# CONS:
# - Not bitwise-identical to Ken Perlin's reference
# - Periodicity (tiling) is NOT guaranteed unless enforced explicitly
# ------------------------------------------------------------
def grad_hash(ix, iy, seed=0):
    # Use uint64 to avoid overflow warnings.
    # The wrap-around behavior is intentional and preserved.
    h = (
        np.uint64(ix) * np.uint64(374761393)
        + np.uint64(iy) * np.uint64(668265263)
        + np.uint64(seed) * np.uint64(2246822519)
    )
    h = (h ^ (h >> np.uint64(13))) * np.uint64(1274126177)
    h = h ^ (h >> np.uint64(16))

    # Explicitly reduce to 32-bit space (intentional wrap)
    h = h & np.uint64(0xFFFFFFFF)

    angle = (h.astype(np.float64) / np.float64(2**32)) * (2.0 * np.pi)
    return np.cos(angle), np.sin(angle)

# ------------------------------------------------------------
# 2D Gradient Noise (Perlin-style)
#
# SAME CORE IDEA AS PERLIN:
# - Integer lattice
# - Gradient vectors at lattice points
# - Dot product with displacement vectors
# - Fade-weighted bilinear interpolation
#
# DIFFERENCE:
# - Gradient selection method (hash vs permutation table)
# - Output range is approximate, not analytically normalized
# ------------------------------------------------------------
def perlin2(x, y, seed=0):
    # Integer lattice coordinates
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    # Local coordinates inside cell [0,1)
    sx = x - x0
    sy = y - y0

    # Gradients at the four corners
    g00x, g00y = grad_hash(x0, y0, seed)
    g10x, g10y = grad_hash(x1, y0, seed)
    g01x, g01y = grad_hash(x0, y1, seed)
    g11x, g11y = grad_hash(x1, y1, seed)

    # Displacement vectors
    dx0 = sx
    dy0 = sy
    dx1 = sx - 1.0
    dy1 = sy - 1.0

    # Dot products (core of gradient noise)
    n00 = g00x * dx0 + g00y * dy0
    n10 = g10x * dx1 + g10y * dy0
    n01 = g01x * dx0 + g01y * dy1
    n11 = g11x * dx1 + g11y * dy1

    # Smooth interpolation
    u = fade(sx)
    v = fade(sy)

    nx0 = lerp(n00, n10, u)
    nx1 = lerp(n01, n11, u)
    nxy = lerp(nx0, nx1, v)

    # Note:
    # Classic Perlin often scales the result to a specific range.
    # Here we keep the raw value (~[-0.7, 0.7]).
    return nxy

# ------------------------------------------------------------
# Fractal Brownian Motion (fBm)
#
# This is NOT part of Perlin Noise itself.
# It is a standard technique for terrain synthesis:
#   sum_i amplitude_i * noise(frequency_i * x)
#
# This is exactly how Perlin noise is typically used in practice
# for terrain generation.
# ------------------------------------------------------------
def fbm2(x, y, seed=0, octaves=6, lacunarity=2.0, gain=0.5):
    amp = 1.0
    freq = 1.0
    s = np.zeros_like(x, dtype=np.float64)
    norm = 0.0

    for o in range(octaves):
        s += amp * perlin2(x * freq, y * freq, seed + o * 101)
        norm += amp
        amp *= gain
        freq *= lacunarity

    return s / max(norm, 1e-9)


def make_faces_vectorized(W, H):
    """
    Vectorized triangle index generation for a regular grid mesh.

    Vertices are assumed to be laid out as:
        index = r * W + c
    with r in [0, H), c in [0, W)
    """

    # Grid of top-left vertices for each quad
    r = np.arange(H - 1)
    c = np.arange(W - 1)
    rr, cc = np.meshgrid(r, c, indexing="ij")

    v00 = rr * W + cc
    v10 = v00 + 1
    v01 = v00 + W
    v11 = v01 + 1

    # Two triangles per quad
    tri1 = np.stack([v00, v01, v10], axis=-1)
    tri2 = np.stack([v10, v01, v11], axis=-1)

    # Concatenate and reshape
    F = np.concatenate([tri1, tri2], axis=0).reshape(-1, 3).astype(np.int32)
    return F


# ------------------------------------------------------------
# Terrain mesh generation
#
# Height = fBm(Perlin-style gradient noise)
# Grid mesh (two triangles per cell)
# ------------------------------------------------------------
def make_terrain_mesh(
    W=256, H=256,
    size_x=10.0, size_z=10.0,
    height=2.0,
    noise_scale=5.0,
    seed=1,
    octaves=6,
    center_around_origin=True
):
    xs = np.linspace(0.0, size_x, W, endpoint=True)
    zs = np.linspace(0.0, size_z, H, endpoint=True)
    X, Z = np.meshgrid(xs, zs, indexing="xy")

    # Noise domain coordinates
    nx = (X / size_x) * noise_scale
    nz = (Z / size_z) * noise_scale

    # Height field
    Y = fbm2(nx, nz, seed=seed, octaves=octaves) * height

    # Vertices
    V = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float64)

    if center_around_origin:
        V[:, 0] -= size_x / 2.0
        V[:, 2] -= size_z / 2.0

    # Triangle indices
    F = make_faces_vectorized(W, H)

    return V, F



def main():
    set_random_seed()

    output_dir = Path(__file__).parent / "output" / "terrain"
    output_dir.mkdir(parents=True, exist_ok=True)

    size_x = 10.0
    size_z = 20.0
    V, F = make_terrain_mesh(size_x=size_x, size_z=size_z, center_around_origin=True)
    src_mesh = o3d.geometry.TriangleMesh()
    src_mesh.vertices = o3d.utility.Vector3dVector(V)
    src_mesh.triangles = o3d.utility.Vector3iVector(F)
    src_mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(
        str(output_dir / "src.ply"),
        src_mesh,
    )

    graph = Graph()
    graph.init_as_grid(
        size_x=size_x,
        size_y=1.0,
        size_z=size_z,
        div_x=int(size_x) * 2,
        div_y=1,
        div_z=int(size_z) * 2,
        connectivity=6,
        weight_type="inv",
        translation =torch.tensor([ -size_x / 2.0, 0.0, -size_z / 2.0 ])
    )
    export_graph_as_lines(graph, str(output_dir / "graph_lines_start.obj"))
    raidus_node = size_x / 50.0
    raidus_edge = raidus_node / 10.0
    export_graph_as_mesh(
        graph,
        str(output_dir / "graph_mesh_start.ply"),
        radius_node=raidus_node,
        radius_edge=raidus_edge,
    )

    deg = 15
    rad = np.deg2rad(deg)
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler("x", rad).as_matrix()
    #rotated_z = R @ np.array([0.0, 0.0, -size_z / 2.0])
    #rotated_z = R @ graph.poss[:, 2].numpy().reshape(3, -1)
    R = torch.from_numpy(R).float()
    rotated_z = (R @ graph.poss.T).T
    # Apply rotation to the z-end of the graph
    fix_mask = graph.poss[:, 2] < 0 * (size_z / 2.0)
    update_mask = ~fix_mask
    graph.Rs[fix_mask] = R#torch.from_numpy(R).float()
    graph.ts[fix_mask] = rotated_z[fix_mask] - graph.poss[fix_mask]

    graph = graph.to("cuda" if torch.cuda.is_available() else "cpu")
    indices_nodes, weights_nodes = graph.assign_nodes_to_points_by_knn(
        graph.poss,
        K=4,
        weight_type="inv",
    )
    indices_v, weights_v = graph.assign_nodes_to_points_by_knn(
        torch.from_numpy(np.asarray(src_mesh.vertices)).float().to(graph.poss.device),
        K=4,
        weight_type="inv",
    )
    from nonrigid_icp_ed.registration import warp_embedded_deformation
    from nonrigid_icp_ed.loss import compute_arap_loss
    from torch import nn
    import loguru
    warped = warp_embedded_deformation(
        torch.from_numpy(np.asarray(src_mesh.vertices)).float().to(graph.poss.device),
        graph.poss,
        graph.Rs,
        graph.ts,
        indices_v,
        weights_v,
    )
    warped = warped.detach().cpu().numpy()
    warped_mesh = o3d.geometry.TriangleMesh()
    print(warped.shape)
    warped_mesh.vertices = o3d.utility.Vector3dVector(warped)
    warped_mesh.triangles = src_mesh.triangles
    warped_mesh.compute_vertex_normals()
    print(warped)
    o3d.io.write_triangle_mesh(
        str(output_dir / "tgt_before.ply"),
        warped_mesh,
    )

    max_iter = 1000
    update_Rs = graph.Rs
    update_ts = graph.ts
    update_Rs = nn.Parameter(update_Rs)
    update_ts = nn.Parameter(update_ts)
    print(update_Rs)
    print(update_ts)
    optimizer = torch.optim.Adam(
        [update_Rs, update_ts],
        lr=0.001,
    )
    for iter in range(max_iter):
        optimizer.zero_grad()
        loss_arap = compute_arap_loss(
            update_Rs,
            update_ts,
            graph.poss,
            graph.edges,
            graph.weights,
        )
        loss = loss_arap
        loguru.logger.debug(f"Iter {iter+1}/{max_iter}, ARAP Loss: {loss_arap.item():.6f}")
        loss.backward()
        optimizer.step()

    graph.Rs = update_Rs.detach()
    graph.ts = update_ts.detach()

    warped = warp_embedded_deformation(
        torch.from_numpy(np.asarray(src_mesh.vertices)).float().to(graph.poss.device),
        graph.poss,
        update_Rs,
        update_ts,
        indices_v,
        weights_v,
    )
    warped = warped.detach().cpu().numpy()
    warped_mesh = o3d.geometry.TriangleMesh()
    print(warped.shape)
    warped_mesh.vertices = o3d.utility.Vector3dVector(warped)
    warped_mesh.triangles = src_mesh.triangles
    warped_mesh.compute_vertex_normals()
    print(warped)
    o3d.io.write_triangle_mesh(
        str(output_dir / "tgt_after.ply"),
        warped_mesh,
    )
    
    graph = Graph()
    expand_ratio = 1.1
    graph.init_as_grid(
        size_x=size_x*expand_ratio,
        size_y=1.0,
        size_z=size_z*expand_ratio,
        div_x=5,
        div_y=1,
        div_z=int(size_z) // 2,
        connectivity=6,
        weight_type="inv",
        translation =torch.tensor([ -size_x * expand_ratio / 2.0, 0.0, -size_z*expand_ratio / 2.0 ])
    )
    # graph = Graph()
    # graph.init_as_grid(
    #     size_x=size_x,
    #     size_y=1.0,
    #     size_z=size_z,
    #     div_x=int(size_x) * 2,
    #     div_y=1,
    #     div_z=int(size_z) * 2,
    #     connectivity=6,
    #     weight_type="inv",
    #     translation =torch.tensor([ -size_x / 2.0, 0.0, -size_z / 2.0 ])
    # )
    export_graph_as_mesh(
        graph,
        str(output_dir / "graph_mesh_end.ply"),
        radius_node=raidus_node,
        radius_edge=raidus_edge,
    )
    indices_v, weights_v = graph.assign_nodes_to_points_by_knn(
        torch.from_numpy(np.asarray(src_mesh.vertices)).float().to(graph.poss.device),
        K=len(graph.poss),
        weight_type="inv",
    )
    config = NonrigidIcpEdConfig()
    config.minimization_conf.w_landmark = 0.0
    trunc_th = -1
    config.minimization_conf.trunc_th = trunc_th
    config.minimization_conf.learning_rate = 0.001
    config.minimization_conf.w_arap = 10.0
    config.minimization_conf.max_iters = 20
    config.num_iterations = 20
    config.keep_history_on_memory = False
    #config.write_history_dir = None
    src_pcd = torch.from_numpy(np.asarray(src_mesh.vertices)).float()
    tgt_pcd = torch.from_numpy(np.asarray(warped_mesh.vertices)).float()
    nricp = NonRigidICP(
        src_pcd,
        tgt_pcd,
        graph,
        config,
        weights_v,
        indices_v,
    )
    nricp = nricp.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    warped_src_pcd = nricp.run()

    warped_src_pcd_np = warped_src_pcd.detach().cpu().numpy()
    warped_src_mesh = o3d.geometry.TriangleMesh()
    warped_src_mesh.vertices = o3d.utility.Vector3dVector(warped_src_pcd_np)
    warped_src_mesh.triangles = src_mesh.triangles
    o3d.io.write_triangle_mesh(
        str(output_dir / "warped_src.obj"), warped_src_mesh
    )


if __name__ == "__main__":
    main()
