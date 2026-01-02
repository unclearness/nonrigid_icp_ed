import torch
import open3d as o3d
import numpy as np


from nonrigid_icp_ed.graph import Graph
from nonrigid_icp_ed.warp import warp_embedded_deformation
from nonrigid_icp_ed.util import rotation_6d_to_matrix, matrix_to_rotation_6d
from nonrigid_icp_ed.loss import (
    compute_truncated_chamfer_distance,
    compute_arap_loss,
    compute_landmark_loss,
)
from nonrigid_icp_ed.knn import find_nearest_neighbors_faiss
from nonrigid_icp_ed.config import NonrigidICPEDConfig


def optimize_embeded_deformation_with_correspondences(
    graph_nodes: torch.Tensor,
    graph_edges: torch.Tensor,
    graph_edge_weights: torch.Tensor,
    src_node_weights: torch.Tensor,
    src_node_indices: torch.Tensor,
    max_iters: int,
    learning_rate: float,
    src_pcd: torch.Tensor,
    tgt_pcd: torch.Tensor,
    src2tgt_correspondence: torch.Tensor,
    tgt2src_correspondence: torch.Tensor,
    src_landmarks: torch.Tensor,
    tgt_landmarks: torch.Tensor,
    w_chamfer: float,
    w_landmark: float,
    w_arap: float,
    trunc_th: float,
    device: torch.device,
    eps: float = 1e-7,
    init_ts: torch.Tensor | None = None,
    init_Rs: torch.Tensor | None = None,
    fix_anchors: bool = False,
):

    assert w_chamfer > 0, "Chamfer weight must be positive."

    ts = init_ts
    if ts is None:
        node_translations = torch.zeros_like(graph_nodes, device=device)
        ts = torch.nn.Parameter(node_translations)

    Rs = init_Rs
    if Rs is None:
        Rs = torch.eye(3, device=device).unsqueeze(0).repeat(graph_nodes.size(0), 1, 1)
        rot6_vecs = torch.nn.Parameter(matrix_to_rotation_6d(Rs))
    else:
        rot6_vecs = torch.nn.Parameter(matrix_to_rotation_6d(Rs))

    optimizer = torch.optim.Adam([rot6_vecs, ts], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

    anchor_poss = graph_nodes.detach().clone()

    warped_src_pcd = src_pcd
    for i in range(max_iters):

        if not fix_anchors:
            # if not fix_anchors: Update anchor positions per iteration
            anchor_poss = graph_nodes + ts.detach()

        Rs = rotation_6d_to_matrix(rot6_vecs)
        warped_src_pcd = warp_embedded_deformation(
            src_pcd, anchor_poss, Rs, ts, src_node_indices, src_node_weights
        )

        if w_arap > 0:
            arap_loss = compute_arap_loss(Rs, ts, graph_nodes, graph_edges, graph_edge_weights)
        else:
            arap_loss = 0

        if w_landmark > 0:
            landmark_loss = compute_landmark_loss(
                src_landmarks,
                tgt_landmarks,
                anchor_poss,
                Rs,
                ts,
                src_node_indices,
                src_node_weights,
            )
        else:
            landmark_loss = 0

        chamfer_distance = compute_truncated_chamfer_distance(
            warped_src_pcd,
            tgt_pcd,
            src2tgt_correspondence,
            tgt2src_correspondence,
            trunc_th=trunc_th,
            reduction="mean",
            ignore_index=-1,
        )

        loss = (
            arap_loss * w_arap
            + landmark_loss * w_landmark
            + chamfer_distance * w_chamfer
        )

        if loss.item() < eps:
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return warped_src_pcd, Rs, ts, anchor_poss


class NonRigidICP:
    def __init__(
        self,
        src_pcd: torch.Tensor,
        tgt_pcd: torch.Tensor,
        graph: Graph,
        config: NonrigidICPEDConfig,
    ):
        self.initialize(src_pcd, tgt_pcd, graph, config)

    def initialize(
        self,
        src_pcd: torch.Tensor,
        tgt_pcd: torch.Tensor,
        graph: Graph,
        config: NonrigidICPEDConfig,
    ):
        self.src_pcd = src_pcd
        self.tgt_pcd = tgt_pcd
        self.graph = graph
        self.config = config

    def run(self):
        for i in range(self.config.num_iterations):
            pass
            # src2tgt_indices, tgt2src_indices =  find_nearest_neighbors_faiss(
            #    self.src_pcd, self.tgt_pcd, self.config.knn_conf.k
            # )

            # self.warped_src_pcd, self.Rs, self.ts, self.anchor_poss = optimize_embeded_deformation_with_correspondences(
