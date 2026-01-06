import json

import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from nonrigid_icp_ed.graph import Graph


def import_wrap3_json(filepath: str) -> dict:
    """
    Import a Wrap3 JSON file and return its contents as a dictionary.
    """
    with open(filepath, "r") as f:
        data = json.load(f)
    if type(data) is list:
        dict_data = {str(i): v for i, v in enumerate(data)}
    elif type(data) is dict:
        dict_data = data
    else:
        raise ValueError("Unsupported Wrap3 JSON format.")
    return dict_data


def export_graph_as_lines(
    graph: Graph,
    filepath: str,
) -> None:
    """
    Export the graph structure to an OBJ file for visualization.
    Nodes are exported as vertices, and edges as lines.
    """
    with open(filepath, "w") as f:
        # Write vertices
        for v in graph.poss.cpu().numpy():
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        # Write edges as lines
        for edge in graph.edges.cpu().numpy():
            # OBJ format uses 1-based indexing
            f.write(f"l {edge[0] + 1} {edge[1] + 1}\n")


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


def _R_from_a_to_b(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Return rotation matrix R such that R @ a == b (both 3D).
    Uses correct Rodrigues formula with sinθ and (1 - cosθ).
    Handles parallel and anti-parallel cases robustly.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a / (np.linalg.norm(a) + eps)
    b = b / (np.linalg.norm(b) + eps)

    v = np.cross(a, b)
    c = float(np.dot(a, b))  # cosθ
    s = np.linalg.norm(v)  # ‖v‖ = sinθ

    if s < eps:
        # parallel or anti-parallel
        if c > 0.0:
            return np.eye(3, dtype=np.float64)  # θ ≈ 0
        # θ ≈ π: rotate about any axis orthogonal to a
        u = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            u = np.array([0.0, 1.0, 0.0])
        u = u - a * np.dot(u, a)
        u = u / (np.linalg.norm(u) + eps)
        K = _skew(u)
        # Rodrigues with θ=π → sinθ=0, 1-cosθ=2 → R = I + 0*K + 2*K^2
        return np.eye(3) + 2.0 * (K @ K)

    # general case
    k = v / s
    K = _skew(k)
    # R = I + K*sinθ + K^2*(1 - cosθ)  with sinθ = s, cosθ = c
    return np.eye(3) + K * s + (K @ K) * (1.0 - c)


def export_graph_as_mesh(
    graph,
    filepath: str,
    radius_node: float = 0.01,
    radius_edge: float = 0.005,
    resolution: int = 10,
    min_edge_radius: float = 1e-4,
) -> None:
    """
    Export graph as a mesh: nodes = spheres, edges = cylinders.
    Cylinders are aligned to edge direction and centered at the segment midpoint.
    """
    assert graph.poss.ndim == 2 and graph.poss.shape[1] == 3
    poss_np = graph.poss.detach().cpu().numpy().astype(np.float64)
    edges_np = graph.edges.detach().cpu().numpy().reshape(-1, 2)
    w_np = (
        graph.weights.detach().cpu().numpy().astype(np.float64)
        if getattr(graph, "weights", None) is not None
        and graph.weights.numel() == edges_np.shape[0]
        else np.ones((edges_np.shape[0],), dtype=np.float64)
    )

    geoms = []

    # spheres (nodes)
    for v in poss_np:
        sph = o3d.geometry.TriangleMesh.create_sphere(
            radius=radius_node, resolution=resolution
        )
        sph.translate(v)
        sph.paint_uniform_color([1.0, 0.0, 0.0])
        geoms.append(sph)

    # cylinders (edges)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    for e_idx, (i, j) in enumerate(edges_np):
        start = poss_np[i]
        end = poss_np[j]
        seg = end - start
        h = float(np.linalg.norm(seg))
        if h <= 1e-12:
            continue  # skip degenerate

        r = max(min_edge_radius, float(radius_edge) * float(abs(w_np[e_idx])))

        cyl = o3d.geometry.TriangleMesh.create_cylinder(
            radius=r, height=h, resolution=resolution
        )

        # align +Z to segment direction
        R = _R_from_a_to_b(z_axis, seg)
        cyl.rotate(R, center=np.array([0.0, 0.0, 0.0]))

        # translate to segment midpoint (Open3D cylinder is centered at origin)
        mid = (start + end) * 0.5
        cyl.translate(mid)

        cyl.paint_uniform_color([0.0, 1.0, 0.0])
        geoms.append(cyl)

    if not geoms:
        return

    mesh = o3d.geometry.TriangleMesh()
    for g in geoms:
        mesh += g
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(filepath, mesh)


def export_correspondences_as_lines(
    src_pcd: torch.Tensor | np.ndarray,
    tgt_pcd: torch.Tensor | np.ndarray,
    filepath: str,
    src_color: tuple = (1.0, 0.0, 0.0),
    tgt_color: tuple = (0.0, 1.0, 0.0)
) -> None:
    if isinstance(src_pcd, torch.Tensor):
        src_pcd = src_pcd.detach().cpu().numpy()
    if isinstance(tgt_pcd, torch.Tensor):
        tgt_pcd = tgt_pcd.detach().cpu().numpy()

    with open(filepath, "w") as f:
        for v in src_pcd:
            f.write(f"v {v[0]} {v[1]} {v[2]} {src_color[0]} {src_color[1]} {src_color[2]}\n")
        for v in tgt_pcd:
            f.write(f"v {v[0]} {v[1]} {v[2]} {tgt_color[0]} {tgt_color[1]} {tgt_color[2]}\n")
        n_src = src_pcd.shape[0]
        for i in range(n_src):
            f.write(f"l {i + 1} {i + 1 + n_src}\n")


def export_correspondences_as_mesh(
    src_pcd: torch.Tensor | np.ndarray,
    tgt_pcd: torch.Tensor | np.ndarray,
    filepath: str,
    radius: float = 0.002,
    resolution: int = 5,
) -> None:
    if isinstance(src_pcd, torch.Tensor):
        src_pcd = src_pcd.detach().cpu().numpy()
    if isinstance(tgt_pcd, torch.Tensor):
        tgt_pcd = tgt_pcd.detach().cpu().numpy()

    geoms = []

    n_src = src_pcd.shape[0]
    for i in range(n_src):
        start = src_pcd[i]
        end = tgt_pcd[i]
        seg = end - start
        h = float(np.linalg.norm(seg))
        if h <= 1e-12:
            continue  # skip degenerate

        cyl = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius, height=h, resolution=resolution
        )

        # align +Z to segment direction
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        R = _R_from_a_to_b(z_axis, seg)
        cyl.rotate(R, center=np.array([0.0, 0.0, 0.0]))

        # translate to segment midpoint (Open3D cylinder is centered at origin)
        mid = (start + end) * 0.5
        cyl.translate(mid)

        cyl.paint_uniform_color([0.0, 1.0, 0.0])
        geoms.append(cyl)

    if not geoms:
        return

    mesh = o3d.geometry.TriangleMesh()
    for g in geoms:
        mesh += g
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(filepath, mesh)