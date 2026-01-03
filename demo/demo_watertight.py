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
)
from nonrigid_icp_ed.util import set_random_seed
from nonrigid_icp_ed.registration import NonRigidICP
from nonrigid_icp_ed.config import NonrigidIcpEdConfig
from nonrigid_icp_ed.obj_io import load_obj_as_open3d


def main():
    set_random_seed()

    graph = Graph()

    input_dir = Path(__file__).parent.parent / "data"
    output_dir = Path(__file__).parent.parent / "output" / "watertight"
    output_dir.mkdir(parents=True, exist_ok=True)

    src_mesh = load_obj_as_open3d(str(input_dir / "sphere" / "icosphere5_smart_uv.obj"))
    tgt_mesh = load_obj_as_open3d(str(input_dir / "spot" / "spot_triangulated.obj"))

    src_points = np.asarray(src_mesh.vertices)
    # tgt_points = np.asarray(tgt_mesh.vertices)
    # tgt_sampled_pcd = tgt_mesh.sample_points_poisson_disk(number_of_points=len(src_points) * 15)
    tgt_sampled_pcd = tgt_mesh.sample_points_uniformly(
        number_of_points=len(src_points) * 15
    )

    tgt_points = np.asarray(tgt_sampled_pcd.points)

    config_dir = Path(__file__).parent.parent / "config"
    config = NonrigidIcpEdConfig.load_yaml(
        config_dir / "demo_watertight.yaml"
    )
    node_num = 1500
    rand_indices = np.random.choice(len(src_points), size=node_num, replace=False)

    node_poss = src_points[rand_indices]

    graph.poss = torch.from_numpy(node_poss).float()
    graph.init_edges_and_weights_by_knn_from_poss(K=config.graph_conf.edges_k)
    export_graph_as_lines(graph, str(output_dir / "graph_lines_start.obj"))
    src_length = np.linalg.norm(src_points.max(axis=0) - src_points.min(axis=0))
    radius_node = src_length / 50.0
    radius_edge = radius_node / 2.0
    export_graph_as_mesh(
        graph,
        str(output_dir / "graph_mesh_start.ply"),
        radius_node=radius_node,
        radius_edge=radius_edge,
    )

    # Set truncation threshold based on the bounding box diagonal length
    config.minimization_conf.trunc_th = -1
    config.write_history_dir = str(output_dir / "optimization_histories")

    src_pcd = torch.from_numpy(src_points).float()
    tgt_pcd = torch.from_numpy(tgt_points).float()
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
        src_triangles=torch.from_numpy(np.asarray(src_mesh.triangles)).long(),
    )
    nricp = nricp.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # Run NRICP
    warped_src_pcd = nricp.run()

    # Refine further with loosen regularization
    nricp.config.minimization_conf.w_normal_consistency *= 0.1
    nricp.config.minimization_conf.w_edge_length_uniform *= 0.1
    nricp.config.minimization_conf.w_arap *= 0.1
    nricp.config.minimization_conf.learning_rate *= 0.1
    nricp.config.minimization_conf.max_iters //= 2
    nricp.config.num_iterations //= 2
    nricp.src_pcd = warped_src_pcd.detach().clone()
    warped_src_pcd = nricp.run()

    warped_src_pcd_np = warped_src_pcd.detach().cpu().numpy()
    warped_src_mesh = o3d.geometry.TriangleMesh()
    warped_src_mesh.vertices = o3d.utility.Vector3dVector(warped_src_pcd_np)
    warped_src_mesh.triangles = src_mesh.triangles
    o3d.io.write_triangle_mesh(
        str(output_dir / "warped_sphere_mesh.obj"), warped_src_mesh
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
