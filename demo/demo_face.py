import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

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
from nonrigid_icp_ed.registration import NonRigidIcp, OptimizationHistory
from nonrigid_icp_ed.config import NonrigidIcpConfig


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

    graph = Graph()

    input_dir = Path(__file__).parent.parent / "data" / "face"
    output_dir = Path(__file__).parent.parent / "output" / "face"
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

    config_dir = Path(__file__).parent.parent / "config"
    config = NonrigidIcpConfig.load_yaml(config_dir / "demo_face.yaml")
    node_num = 50
    rand_indices = np.random.choice(
        len(aligned_mediapipe_points), size=node_num, replace=False
    )

    node_poss = aligned_mediapipe_points[rand_indices]

    graph.poss = torch.from_numpy(node_poss).float()
    graph.init_edges_and_weights_by_knn_from_poss(K=config.graph_conf.edges_k)
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

    # Set truncation threshold based on the bounding box diagonal length
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
    nricp = NonRigidIcp(
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
    torch.save(
        [h.to_dict() for h in nricp.optimization_histories],
        str(output_dir / "opt_history.pt"),
    )

    if True:
        histories = [
            OptimizationHistory.from_dict(h)
            for h in torch.load(str(output_dir / "opt_history.pt"), map_location="cpu")
        ]
    else:
        histories = []
        for history_pt in sorted(
            (output_dir / "optimization_histories").glob(
                "optimization_history_iter_*.pt"
            )
        ):
            print(f"Loading optimization history from: {history_pt}")
            histories.append(
                OptimizationHistory.from_dict(
                    torch.load(history_pt, map_location="cpu")
                )
            )

    reconstructed_warped_src_pcd = NonRigidIcp.reconstruct_from_optimization_histories(
        histories, torch.from_numpy(aligned_mediapipe_points).float(), config.global_deform
    )
    reconstructed_warped_src_pcd_np = (
        reconstructed_warped_src_pcd.detach().cpu().numpy()
    )
    reconstructed_warped_mediapipe_mesh = o3d.geometry.TriangleMesh()
    reconstructed_warped_mediapipe_mesh.vertices = o3d.utility.Vector3dVector(
        reconstructed_warped_src_pcd_np
    )
    reconstructed_warped_mediapipe_mesh.triangles = mediapipe_mesh.triangles
    o3d.io.write_triangle_mesh(
        str(output_dir / "reconstructed_warped_mediapipe_face.obj"),
        reconstructed_warped_mediapipe_mesh,
    )


if __name__ == "__main__":
    main()
