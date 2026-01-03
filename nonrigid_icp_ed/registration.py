from dataclasses import dataclass
from pathlib import Path
import math

import torch
import loguru

from nonrigid_icp_ed.graph import Graph
from nonrigid_icp_ed.warp import warp_embedded_deformation
from nonrigid_icp_ed.util import rotation_6d_to_matrix, matrix_to_rotation_6d
from nonrigid_icp_ed.loss import (
    compute_truncated_chamfer_distance,
    compute_arap_loss,
    compute_landmark_loss,
    compute_unique_edges,
    compute_edge_length_uniform_loss,
    build_adjacent_face_pairs,
    compute_adjacent_normal_consistency_loss,
)
from nonrigid_icp_ed.knn import find_nearest_neighbors_faiss
from nonrigid_icp_ed.config import NonrigidIcpEdConfig


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
    src_landmark_idxs: torch.Tensor | None,
    tgt_landmark_idxs: torch.Tensor | None,
    w_chamfer: float,
    w_landmark: float,
    w_arap: float,
    w_edge_length_uniform: float,
    w_normal_consistency: float,
    trunc_th: float,
    device: torch.device,
    eps: float = 1e-7,
    init_Rs: torch.Tensor | None = None,
    init_ts: torch.Tensor | None = None,
    fix_anchors: bool = False,
    src_triangles: torch.Tensor | None = None,
    report_interval: int = 10,
):

    assert w_chamfer > 0, "Chamfer weight must be positive."

    anchor_ts = init_ts
    if anchor_ts is None:
        node_translations = torch.zeros_like(graph_nodes, device=device)
        anchor_ts = torch.nn.Parameter(node_translations)

    anchor_Rs = init_Rs
    if anchor_Rs is None:
        anchor_Rs = (
            torch.eye(3, device=device).unsqueeze(0).repeat(graph_nodes.size(0), 1, 1)
        )
        anchor_rot6_vecs = torch.nn.Parameter(matrix_to_rotation_6d(anchor_Rs))
    else:
        anchor_rot6_vecs = torch.nn.Parameter(matrix_to_rotation_6d(anchor_Rs))

    optimizer = torch.optim.Adam([anchor_rot6_vecs, anchor_ts], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

    anchor_poss = graph_nodes.detach().clone()

    if (src_landmark_idxs is None or tgt_landmark_idxs is None) and w_landmark > 0:
        w_landmark = 0.0
        loguru.logger.warning("No landmarks provided, setting landmark weight to 0.")

    src_edges = None
    if src_triangles is None and w_edge_length_uniform > 0:
        w_edge_length_uniform = 0
        loguru.logger.warning(
            "No source triangles provided, setting edge length uniform weight to 0."
        )
    elif src_triangles is not None and w_edge_length_uniform > 0:
        src_edges = compute_unique_edges(src_triangles)

    adj = None
    if src_triangles is None and w_normal_consistency > 0:
        w_normal_consistency = 0
        loguru.logger.warning(
            "No source triangles provided, setting normal consistency weight to 0."
        )
    elif src_triangles is not None and w_normal_consistency > 0:
        adj = build_adjacent_face_pairs(src_triangles)

    warped_src_pcd = src_pcd
    for i in range(max_iters):

        if not fix_anchors:
            # if not fix_anchors: Update anchor positions per iteration
            anchor_poss = graph_nodes + anchor_ts.detach()

        anchor_Rs = rotation_6d_to_matrix(anchor_rot6_vecs)
        warped_src_pcd = warp_embedded_deformation(
            src_pcd,
            anchor_poss,
            anchor_Rs,
            anchor_ts,
            src_node_indices,
            src_node_weights,
        )

        if w_arap > 0:
            arap_loss = compute_arap_loss(
                anchor_Rs, anchor_ts, graph_nodes, graph_edges, graph_edge_weights
            )
        else:
            arap_loss = 0

        if w_landmark > 0:
            src_landmarks = warped_src_pcd[src_landmark_idxs]
            tgt_landmarks = tgt_pcd[tgt_landmark_idxs]
            src_landmarks_node_indices = src_node_indices[src_landmark_idxs]
            src_landmarks_node_weights = src_node_weights[src_landmark_idxs]
            landmark_loss = compute_landmark_loss(
                src_landmarks,
                tgt_landmarks,
                anchor_poss,
                anchor_Rs,
                anchor_ts,
                src_landmarks_node_indices,
                src_landmarks_node_weights,
            )
        else:
            landmark_loss = 0

        if w_edge_length_uniform > 0 and src_edges is not None:
            edge_length_uniform_loss = compute_edge_length_uniform_loss(
                warped_src_pcd, edges=src_edges
            )
        else:
            edge_length_uniform_loss = 0

        if w_normal_consistency > 0 and adj is not None and src_triangles is not None:
            cos_margin = math.cos(math.radians(60.0))
            normal_consistency_loss = compute_adjacent_normal_consistency_loss(
                warped_src_pcd,
                src_triangles,
                adj,
                cos_margin=cos_margin,
            )
        else:
            normal_consistency_loss = 0

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
            + edge_length_uniform_loss * w_edge_length_uniform
            + normal_consistency_loss * w_normal_consistency
        )

        if i == 0 or (i + 1) % report_interval == 0 or i == max_iters - 1:
            loguru.logger.debug(
                f"Iter {i+1}/{max_iters}: Total Loss={loss.item():.6f}, "
                f"Chamfer={chamfer_distance.item():.6f}, "
                f"Landmark={landmark_loss.item() if w_landmark > 0 else 0:.6f}, "
                f"ARAP={arap_loss.item() if w_arap > 0 else 0:.6f}, "
                f"EdgeLengthUniform={edge_length_uniform_loss.item() if w_edge_length_uniform > 0 else 0:.6f}, "
                f"NormalConsistency={normal_consistency_loss.item() if w_normal_consistency > 0 else 0:.6f}"
            )

        if loss.item() < eps or max_iters - 1 <= i:
            # No backward if converged or the last iteration
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return warped_src_pcd, anchor_Rs, anchor_ts, anchor_poss


@dataclass
class OptimizationHistory:
    graph: Graph
    warped_src_pcd: torch.Tensor
    src_node_indices: torch.Tensor
    src_node_weights: torch.Tensor

    def to_dict(self):
        return {
            "graph": self.graph.to_dict(),
            "warped_src_pcd": self.warped_src_pcd,
            "src_node_indices": self.src_node_indices,
            "src_node_weights": self.src_node_weights,
        }

    @staticmethod
    def from_dict(d):
        return OptimizationHistory(
            graph=Graph.from_dict(d["graph"]),
            warped_src_pcd=d["warped_src_pcd"],
            src_node_indices=d["src_node_indices"],
            src_node_weights=d["src_node_weights"],
        )


class NonRigidICP:
    def __init__(
        self,
        src_pcd: torch.Tensor,
        tgt_pcd: torch.Tensor,
        graph: Graph,
        config: NonrigidIcpEdConfig,
        src_node_weights: torch.Tensor,
        src_node_indices: torch.Tensor,
        src_landmark_idxs: torch.Tensor | None = None,
        tgt_landmark_idxs: torch.Tensor | None = None,
        src_triangles: torch.Tensor | None = None,
    ):
        self.initialize(
            src_pcd,
            tgt_pcd,
            graph,
            config,
            src_node_weights,
            src_node_indices,
            src_landmark_idxs,
            tgt_landmark_idxs,
            src_triangles,
        )

    def initialize(
        self,
        src_pcd: torch.Tensor,
        tgt_pcd: torch.Tensor,
        graph: Graph,
        config: NonrigidIcpEdConfig,
        src_node_weights: torch.Tensor,
        src_node_indices: torch.Tensor,
        src_landmark_idxs: torch.Tensor | None = None,
        tgt_landmark_idxs: torch.Tensor | None = None,
        src_triangles: torch.Tensor | None = None,
    ):
        self.src_pcd = src_pcd
        self.tgt_pcd = tgt_pcd
        self.graph = graph
        self.config = config
        self.src_node_weights = src_node_weights
        self.src_node_indices = src_node_indices
        self.src_landmark_idxs = src_landmark_idxs
        self.tgt_landmark_idxs = tgt_landmark_idxs
        self.src_triangles = src_triangles
        self.optimization_histories = []

    def to(self, device: torch.device):
        self.src_pcd = self.src_pcd.to(device)
        self.tgt_pcd = self.tgt_pcd.to(device)
        self.graph.poss = self.graph.poss.to(device)
        self.graph.edges = self.graph.edges.to(device)
        self.graph.weights = self.graph.weights.to(device)
        self.src_node_weights = self.src_node_weights.to(device)
        self.src_node_indices = self.src_node_indices.to(device)
        if self.src_landmark_idxs is not None:
            self.src_landmark_idxs = self.src_landmark_idxs.to(device)
        if self.tgt_landmark_idxs is not None:
            self.tgt_landmark_idxs = self.tgt_landmark_idxs.to(device)
        if self.src_triangles is not None:
            self.src_triangles = self.src_triangles.to(device)
        return self

    def run(self):
        loguru.logger.info("Starting Non-Rigid ICP with Embedded Deformation...")
        self.warped_src_pcd = self.src_pcd
        self.optimization_histories = []
        for i in range(self.config.num_iterations):
            loguru.logger.info(
                f"Non-Rigid ICP Iteration {i+1}/{self.config.num_iterations}"
            )
            src_pcd = self.warped_src_pcd.detach().clone()

            loguru.logger.info("Finding nearest neighbor correspondences...")
            src2tgt_indices, src2tgt_dists = find_nearest_neighbors_faiss(
                src_pcd, self.tgt_pcd, self.config.correspondence_conf.k
            )
            tgt2src_indices, tgt2src_dists = find_nearest_neighbors_faiss(
                self.tgt_pcd, src_pcd, self.config.correspondence_conf.k
            )

            loguru.logger.info("Optimizing embedded deformation...")
            if self.config.init_with_previous_Rt:
                init_Rs = self.graph.Rs.detach().clone()
                init_ts = self.graph.ts.detach().clone()
            else:
                init_Rs = None
                init_ts = None
            warped_src_pcd, node_Rs, node_ts, anchor_poss = (
                optimize_embeded_deformation_with_correspondences(
                    graph_nodes=self.graph.poss,
                    graph_edges=self.graph.edges,
                    graph_edge_weights=self.graph.weights,
                    src_node_weights=self.src_node_weights,
                    src_node_indices=self.src_node_indices,
                    max_iters=self.config.minimization_conf.max_iters,
                    learning_rate=self.config.minimization_conf.learning_rate,
                    src_pcd=src_pcd,
                    tgt_pcd=self.tgt_pcd,
                    src2tgt_correspondence=src2tgt_indices,
                    tgt2src_correspondence=tgt2src_indices,
                    src_landmark_idxs=self.src_landmark_idxs,
                    tgt_landmark_idxs=self.tgt_landmark_idxs,
                    w_chamfer=self.config.minimization_conf.w_chamfer,
                    w_landmark=self.config.minimization_conf.w_landmark,
                    w_arap=self.config.minimization_conf.w_arap,
                    w_edge_length_uniform=self.config.minimization_conf.w_edge_length_uniform,
                    w_normal_consistency=self.config.minimization_conf.w_normal_consistency,
                    trunc_th=self.config.minimization_conf.trunc_th,
                    device=self.src_pcd.device,
                    eps=self.config.minimization_conf.eps,
                    init_Rs=init_Rs,
                    init_ts=init_ts,
                    fix_anchors=self.config.minimization_conf.fix_anchors,
                    src_triangles=self.src_triangles,
                    report_interval=self.config.minimization_conf.report_interval,
                )
            )

            self.warped_src_pcd = warped_src_pcd.detach().clone()
            self.graph.poss = anchor_poss.detach().clone()
            self.graph.Rs = node_Rs.detach().clone()
            self.graph.ts = node_ts.detach().clone()
            history = OptimizationHistory(
                graph=Graph(
                    poss=self.graph.poss.detach().clone(),
                    Rs=self.graph.Rs.detach().clone(),
                    ts=self.graph.ts.detach().clone(),
                    edges=self.graph.edges.detach().clone(),
                    weights=self.graph.weights.detach().clone(),
                ),
                warped_src_pcd=self.warped_src_pcd.detach().clone(),
                src_node_indices=self.src_node_indices.detach().clone(),
                src_node_weights=self.src_node_weights.detach().clone(),
            )
            if self.config.keep_history_on_memory:
                self.optimization_histories.append(history)
            if self.config.write_history_dir is not None:
                output_dir = Path(self.config.write_history_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    history.to_dict(),
                    output_dir / f"optimization_history_iter_{i+1:06d}.pt",
                )

        loguru.logger.info("Non-Rigid ICP completed.")
        return self.warped_src_pcd

    @staticmethod
    def reconstruct_from_optimization_histories(
        histories: list[OptimizationHistory],
        src_pcd: torch.Tensor,
    ) -> torch.Tensor:
        warped_src_pcd = src_pcd.detach().clone()
        for history in histories:
            warped_src_pcd = warp_embedded_deformation(
                warped_src_pcd,
                history.graph.poss,
                history.graph.Rs,
                history.graph.ts,
                history.src_node_indices,
                history.src_node_weights,
            )
            warped_src_pcd = warped_src_pcd.detach().clone()
        return warped_src_pcd
