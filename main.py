import open3d as o3d
import numpy as np
import torch

from nonrigid_icp_ed.graph import Graph
from nonrigid_icp_ed.io import export_graph_as_lines, export_graph_as_mesh


def main():
    print("Hello from nonrigid-icp!")
    graph = Graph()

    mesh = o3d.io.read_triangle_mesh("face/mediapipe_face.obj")
    points = np.array(mesh.vertices)

    node_poss = points[np.random.randint(low=0, high=len(points), size=20, dtype=int)]
    graph.poss = torch.from_numpy(node_poss).float()
    graph.make_edges(K=6)
    export_graph_as_lines(graph, "graph_lines.obj")
    radius_node = np.linalg.norm(points.max(axis=0) - points.min(axis=0)) / 50.0
    radius_edge = radius_node / 2.0
    export_graph_as_mesh(
        graph, "graph_mesh.ply", radius_node=radius_node, radius_edge=radius_edge
    )


if __name__ == "__main__":
    main()
