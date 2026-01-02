import os
import random
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
from nonrigid_icp_ed.util import umeyama
from nonrigid_icp_ed.registration import NonRigidICP
from nonrigid_icp_ed.config import NonrigidICPEDConfig


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


def extract_vertex_ids_from_tri_landmarks(
    tri_landmarks: dict,
    triangles: np.ndarray,
) -> np.ndarray:
    vertex_ids = []
    for name, tri_uv in tri_landmarks.items():
        tri_id = int(tri_uv[0])
        uv = tri_uv[1:3]
        tri = triangles[tri_id]
        vid0 = tri[0]
        vertex_ids.append(vid0)
    return np.array(vertex_ids, dtype=int)


def main():
    set_random_seed()

    print("Hello from nonrigid-icp!")
    graph = Graph()

    input_dir = Path(__file__).parent / "data" / "face"
    output_dir = Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)

    mediapipe_tri_landmarks = import_wrap3_json(
        str(input_dir / "mediapipe_face_landmarks.json")
    )
    mediapipe_mesh = o3d.io.read_triangle_mesh(str(input_dir / "mediapipe_face.obj"))

    lpshead_tri_landmarks = import_wrap3_json(
        str(input_dir / "lpshead/head_triangulated_landmarks.json")
    )
    lpshead_mesh = o3d.io.read_triangle_mesh(
        str(input_dir / "lpshead/head_triangulated.obj")
    )

    mediapipe_vertex_ids = extract_vertex_ids_from_tri_landmarks(
        mediapipe_tri_landmarks, np.asarray(mediapipe_mesh.triangles)
    )
    lpshead_vertex_ids = extract_vertex_ids_from_tri_landmarks(
        lpshead_tri_landmarks, np.asarray(lpshead_mesh.triangles)
    )

    mediapipe_points = np.asarray(mediapipe_mesh.vertices)
    lpshead_points = np.asarray(lpshead_mesh.vertices)

    # Pre-align the two meshes using Umeyama
    T, R, t, s = umeyama(
        mediapipe_points[mediapipe_vertex_ids],
        lpshead_points[lpshead_vertex_ids],
        estimate_scale=True,
    )

    aligned_mediapipe_points = (s * mediapipe_points @ R.T) + t

    aligned_mediapipe_mesh = o3d.geometry.TriangleMesh()
    aligned_mediapipe_mesh.vertices = o3d.utility.Vector3dVector(
        aligned_mediapipe_points
    )
    aligned_mediapipe_mesh.triangles = mediapipe_mesh.triangles
    o3d.io.write_triangle_mesh(
        str(output_dir / "aligned_mediapipe_face.obj"), aligned_mediapipe_mesh
    )

    config = NonrigidICPEDConfig()
    node_num = 50
    rand_indices = np.random.choice(
        len(aligned_mediapipe_points), size=node_num, replace=False
    )

    node_poss = aligned_mediapipe_points[rand_indices]

    graph.poss = torch.from_numpy(node_poss).float()
    graph.make_edges(K=config.graph_conf.edges_k)
    export_graph_as_lines(graph, str(output_dir / "graph_lines_start.obj"))
    aligned_src_length = np.linalg.norm(
        aligned_mediapipe_points.max(axis=0) - aligned_mediapipe_points.min(axis=0)
    )
    radius_node = aligned_src_length / 50.0
    radius_edge = radius_node / 2.0
    export_graph_as_mesh(
        graph,
        str(output_dir / "graph_mesh_start.ply"),
        radius_node=radius_node,
        radius_edge=radius_edge,
    )

    config.minimization_conf.trunc_th = aligned_src_length * 0.05

    src_pcd = torch.from_numpy(aligned_mediapipe_points).float()
    tgt_pcd = torch.from_numpy(lpshead_points).float()
    src_node_indices, src_node_weights = graph.assign_nodes_to_points_by_knn(
        src_pcd,
        K=config.graph_conf.num_nodes_per_point,
        weight_type=config.graph_conf.weight_type,
        sigma=config.graph_conf.sigma,
        eps=config.graph_conf.eps,
    )
    nricp = NonRigidICP(
        src_pcd,
        tgt_pcd,
        graph,
        config,
        src_node_weights,
        src_node_indices,
        src_landmark_idxs=torch.from_numpy(mediapipe_vertex_ids),
        tgt_landmark_idxs=torch.from_numpy(lpshead_vertex_ids),
    )
    nricp = nricp.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    warped_src_pcd = nricp.run()

    warped_src_pcd_np = warped_src_pcd.detach().cpu().numpy()
    warped_mediapipe_mesh = o3d.geometry.TriangleMesh()
    warped_mediapipe_mesh.vertices = o3d.utility.Vector3dVector(warped_src_pcd_np)
    warped_mediapipe_mesh.triangles = mediapipe_mesh.triangles
    o3d.io.write_triangle_mesh(
        str(output_dir / "warped_mediapipe_face.obj"), warped_mediapipe_mesh
    )

    export_graph_as_lines(nricp.graph, str(output_dir / "graph_lines_end.obj"))
    export_graph_as_mesh(
        nricp.graph,
        str(output_dir / "graph_mesh_end.ply"),
        radius_node=radius_node,
        radius_edge=radius_edge,
    )


if __name__ == "__main__":
    main()
