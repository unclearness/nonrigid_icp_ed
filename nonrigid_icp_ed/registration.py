from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math
from typing import Callable

import torch
import loguru

from nonrigid_icp_ed.graph import Graph
from nonrigid_icp_ed.warp import warp_embedded_deformation
from nonrigid_icp_ed.util import rotation_6d_to_matrix, matrix_to_rotation_6d
from nonrigid_icp_ed.loss import (
    compute_truncated_chamfer_distance,
    compute_truncated_l2,
    compute_arap_loss,
    compute_landmark_loss,
    compute_unique_edges,
    compute_edge_length_uniform_loss,
    build_adjacent_face_pairs,
    compute_adjacent_normal_consistency_loss,
)
from nonrigid_icp_ed.knn import find_nearest_neighbors_open3d
from nonrigid_icp_ed.config import NonrigidIcpConfig


def rebuild_optimizer_with_state(old_opt, new_params, opt_ctor):
    """
    old_opt: existing optimizer
    new_params: new Parameter iterable to optimize
    opt_ctor: lambda params: torch.optim.AdamW(params, lr=..., betas=..., ...)
    """
    # Make a new optimizer
    new_opt = opt_ctor(new_params)

    # Copy state if it matches by Parameter object
    old_sd = old_opt.state_dict()
    new_sd = new_opt.state_dict()

    def params_in_opt(opt):
        ps = []
        for g in opt.param_groups:
            ps.extend(g["params"])
        return ps

    old_ps = params_in_opt(old_opt)
    new_ps = params_in_opt(new_opt)

    old_ids = []
    for g in old_sd["param_groups"]:
        old_ids.extend(g["params"])
    new_ids = []
    for g in new_sd["param_groups"]:
        new_ids.extend(g["params"])

    old_map = {id(p): old_id for p, old_id in zip(old_ps, old_ids)}
    new_map = {id(p): new_id for p, new_id in zip(new_ps, new_ids)}

    # Copy
    for p in new_ps:
        oid = old_map.get(id(p), None)
        nid = new_map.get(id(p), None)
        if oid is not None and nid is not None and oid in old_sd["state"]:
            new_sd["state"][nid] = old_sd["state"][oid]

    new_opt.load_state_dict(new_sd)
    return new_opt


def optimize_embeded_deformation_with_correspondences(
    graph_nodes: torch.Tensor,
    graph_edges: torch.Tensor,
    graph_edge_weights: torch.Tensor,
    src_node_weights: torch.Tensor,
    src_node_indices: torch.Tensor,
    max_iters: int,
    lr_translation: float,
    lr_rotation: float,
    inherit_lr_state: bool,
    prev_optimizer_Rs: torch.optim.Optimizer | None,
    prev_optimizer_ts: torch.optim.Optimizer | None,
    src_pcd: torch.Tensor,
    tgt_pcd: torch.Tensor,
    src2tgt_correspondence: torch.Tensor,
    tgt2src_correspondence: torch.Tensor,
    src_landmark_idxs: torch.Tensor | None,
    tgt_landmark_idxs: torch.Tensor | None,
    w_chamfer: float,
    w_src2t_dists: float,
    w_tgt2s_dists: float,
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

    assert (
        w_chamfer > 0 or w_src2t_dists > 0 or w_tgt2s_dists > 0
    ), "At least one distance weight must be positive."

    anchor_ts = init_ts
    if anchor_ts is None:
        anchor_ts = torch.zeros_like(graph_nodes, device=device)
    anchor_ts = torch.nn.Parameter(anchor_ts)

    anchor_Rs = init_Rs
    if anchor_Rs is None:
        anchor_Rs = (
            torch.eye(3, device=device).unsqueeze(0).repeat(graph_nodes.size(0), 1, 1)
        )
    anchor_rot6_vecs = torch.nn.Parameter(matrix_to_rotation_6d(anchor_Rs))

    optimizer_Rs, optimizer_ts = None, None
    if inherit_lr_state:
        loguru.logger.info("Inheriting optimizer states...")
        if prev_optimizer_Rs is not None:
            optimizer_Rs = rebuild_optimizer_with_state(
                old_opt=prev_optimizer_Rs,
                new_params=[anchor_rot6_vecs],
                opt_ctor=lambda params: torch.optim.Adam(params, lr=lr_rotation),
            )
        if prev_optimizer_ts is not None:
            optimizer_ts = rebuild_optimizer_with_state(
                old_opt=prev_optimizer_ts,
                new_params=[anchor_ts],
                opt_ctor=lambda params: torch.optim.Adam(params, lr=lr_translation),
            )
    else:
        loguru.logger.info("Creating new optimizers...")
        if 0 < lr_rotation:
            optimizer_Rs = torch.optim.Adam([anchor_rot6_vecs], lr=lr_rotation)
        if 0 < lr_translation:
            optimizer_ts = torch.optim.Adam([anchor_ts], lr=lr_translation)
    if optimizer_Rs is None:
        loguru.logger.warning(
            "Rotation learning rate is set to 0 or negative, optimizer for rotations is not created."
        )
    if optimizer_ts is None:
        loguru.logger.warning(
            "Translation learning rate is set to 0 or negative, optimizer for translations is not created."
        )

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

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

        if w_src2t_dists > 0:
            src2tgt_dists_trunc = compute_truncated_l2(
                warped_src_pcd,
                tgt_pcd,
                src2tgt_correspondence,
                trunc_th=trunc_th,
            )
        else:
            src2tgt_dists_trunc = 0

        if w_tgt2s_dists > 0:
            tgt2src_dists_trunc = compute_truncated_l2(
                tgt_pcd,
                warped_src_pcd,
                tgt2src_correspondence,
                trunc_th=trunc_th,
            )
        else:
            tgt2src_dists_trunc = 0

        if w_chamfer > 0:
            chamfer_distance = compute_truncated_chamfer_distance(
                warped_src_pcd,
                tgt_pcd,
                src2tgt_correspondence,
                tgt2src_correspondence,
                trunc_th=trunc_th,
                reduction="mean",
                ignore_index=-1,
            )
        else:
            chamfer_distance = 0

        loss = (
            arap_loss * w_arap
            + landmark_loss * w_landmark
            + chamfer_distance * w_chamfer
            + src2tgt_dists_trunc * w_src2t_dists
            + tgt2src_dists_trunc * w_tgt2s_dists
            + edge_length_uniform_loss * w_edge_length_uniform
            + normal_consistency_loss * w_normal_consistency
        )

        if i == 0 or (i + 1) % report_interval == 0 or i == max_iters - 1:
            loguru.logger.debug(
                f"Iter {i+1}/{max_iters}: Total Loss={loss.item():.6f}, "
                f"Chamfer={chamfer_distance.item() if w_chamfer > 0 else 0:.6f}, "
                f"S2T_Dists={src2tgt_dists_trunc.item() if w_src2t_dists > 0 else 0:.6f}, "
                f"T2S_Dists={tgt2src_dists_trunc.item() if w_tgt2s_dists > 0 else 0:.6f}, "
                f"Landmark={landmark_loss.item() if w_landmark > 0 else 0:.6f}, "
                f"ARAP={arap_loss.item() if w_arap > 0 else 0:.6f}, "
                f"EdgeLengthUniform={edge_length_uniform_loss.item() if w_edge_length_uniform > 0 else 0:.6f}, "
                f"NormalConsistency={normal_consistency_loss.item() if w_normal_consistency > 0 else 0:.6f}"
            )

        if loss.item() < eps or max_iters - 1 <= i:
            # No backward if converged or the last iteration
            break

        if optimizer_Rs is not None:
            optimizer_Rs.zero_grad()
        if optimizer_ts is not None:
            optimizer_ts.zero_grad()
        loss.backward()
        if optimizer_Rs is not None:
            optimizer_Rs.step()
        if optimizer_ts is not None:
            optimizer_ts.step()

        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # scheduler.step()

    return (
        warped_src_pcd,
        anchor_Rs,
        anchor_ts,
        anchor_poss,
        optimizer_Rs,
        optimizer_ts,
    )


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


class NonRigidIcp:
    def __init__(
        self,
        src_pcd: torch.Tensor,
        tgt_pcd: torch.Tensor,
        graph: Graph,
        config: NonrigidIcpConfig,
        src_node_weights: torch.Tensor,
        src_node_indices: torch.Tensor,
        src_landmark_idxs: torch.Tensor | None = None,
        tgt_landmark_idxs: torch.Tensor | None = None,
        src_triangles: torch.Tensor | None = None,
        src_normals: torch.Tensor | None = None,
        tgt_normals: torch.Tensor | None = None,
        callback_after_correspondence_search: (
            Callable[
                [
                    NonRigidIcp,
                    int,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                ],
                None,
            ]
            | None
        ) = None,
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
            src_normals,
            tgt_normals,
            callback_after_correspondence_search,
        )

    def initialize(
        self,
        src_pcd: torch.Tensor,
        tgt_pcd: torch.Tensor,
        graph: Graph,
        config: NonrigidIcpConfig,
        src_node_weights: torch.Tensor,
        src_node_indices: torch.Tensor,
        src_landmark_idxs: torch.Tensor | None = None,
        tgt_landmark_idxs: torch.Tensor | None = None,
        src_triangles: torch.Tensor | None = None,
        src_normals: torch.Tensor | None = None,
        tgt_normals: torch.Tensor | None = None,
        callback_after_correspondence_search: (
            Callable[
                [
                    NonRigidIcp,
                    int,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                ],
                None,
            ]
            | None
        ) = None,
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
        self.src_normals = src_normals
        self.tgt_normals = tgt_normals
        self.callback_after_correspondence_search = callback_after_correspondence_search

        self.optimization_histories = []

    def to(self, device: torch.device):
        self.src_pcd = self.src_pcd.to(device)
        self.tgt_pcd = self.tgt_pcd.to(device)
        self.graph = self.graph.to(device)
        self.src_node_weights = self.src_node_weights.to(device)
        self.src_node_indices = self.src_node_indices.to(device)
        if self.src_landmark_idxs is not None:
            self.src_landmark_idxs = self.src_landmark_idxs.to(device)
        if self.tgt_landmark_idxs is not None:
            self.tgt_landmark_idxs = self.tgt_landmark_idxs.to(device)
        if self.src_triangles is not None:
            self.src_triangles = self.src_triangles.to(device)
        if self.src_normals is not None:
            self.src_normals = self.src_normals.to(device)
        if self.tgt_normals is not None:
            self.tgt_normals = self.tgt_normals.to(device)
        return self

    def run(self):
        loguru.logger.info("Starting Non-Rigid ICP with Embedded Deformation...")
        self.warped_src_pcd = self.src_pcd.detach().clone()
        self.warped_src_normals = (
            self.src_normals.detach().clone() if self.src_normals is not None else None
        )
        self.optimization_histories = []
        prev_optimizer_Rs = None
        prev_optimizer_ts = None
        for i in range(self.config.num_iterations):
            loguru.logger.info(
                f"Non-Rigid ICP Iteration {i+1}/{self.config.num_iterations}"
            )

            loguru.logger.info("Finding nearest neighbor correspondences...")
            src2tgt_indices, src2tgt_dists = find_nearest_neighbors_open3d(
                self.warped_src_pcd, self.tgt_pcd, self.config.correspondence_conf.k
            )
            tgt2src_indices, tgt2src_dists = find_nearest_neighbors_open3d(
                self.tgt_pcd, self.warped_src_pcd, self.config.correspondence_conf.k
            )

            if self.callback_after_correspondence_search is not None:
                self.callback_after_correspondence_search(
                    self,
                    i,
                    src2tgt_indices,
                    tgt2src_indices,
                    src2tgt_dists,
                    tgt2src_dists,
                )

            loguru.logger.info("Optimizing embedded deformation...")
            if self.config.global_deform:
                # Optimize difference from the initial source
                init_Rs = self.graph.Rs.detach().clone()
                init_ts = self.graph.ts.detach().clone()
                src_pcd = self.src_pcd
            else:
                # Optimize difference from the last iteration
                # ARAP loss is computed on the deformed graph from the last iteration, so it will be reset at each iteration
                init_Rs = None
                init_ts = None
                src_pcd = self.warped_src_pcd.detach().clone()
            if i == 0:
                inherit_lr_state = False
            else:
                inherit_lr_state = (self.config.minimization_conf.inherit_lr_state,)

            (
                warped_src_pcd,
                node_Rs,
                node_ts,
                anchor_poss,
                prev_optimizer_Rs,
                prev_optimizer_ts,
            ) = optimize_embeded_deformation_with_correspondences(
                graph_nodes=self.graph.poss,
                graph_edges=self.graph.edges,
                graph_edge_weights=self.graph.weights,
                src_node_weights=self.src_node_weights,
                src_node_indices=self.src_node_indices,
                max_iters=self.config.minimization_conf.max_iters,
                lr_translation=self.config.minimization_conf.lr_translation,
                lr_rotation=self.config.minimization_conf.lr_rotation,
                inherit_lr_state=inherit_lr_state,
                prev_optimizer_Rs=prev_optimizer_Rs,
                prev_optimizer_ts=prev_optimizer_ts,
                src_pcd=src_pcd,
                tgt_pcd=self.tgt_pcd,
                src2tgt_correspondence=src2tgt_indices,
                tgt2src_correspondence=tgt2src_indices,
                src_landmark_idxs=self.src_landmark_idxs,
                tgt_landmark_idxs=self.tgt_landmark_idxs,
                w_chamfer=self.config.minimization_conf.w_chamfer,
                w_src2t_dists=self.config.minimization_conf.w_src2t_dists,
                w_tgt2s_dists=self.config.minimization_conf.w_tgt2s_dists,
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
    def reassign_indices_and_weights(
        src_pcd: torch.Tensor,
        graph_nodes: torch.Tensor,
        num_nodes_per_point: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src_node_indices, src_node_dists = find_nearest_neighbors_open3d(
            src_pcd, graph_nodes, num_nodes_per_point
        )
        # Convert distances to weights using Gaussian kernel
        sigma = torch.mean(torch.sqrt(src_node_dists))
        src_node_weights = torch.exp(-src_node_dists / (2 * sigma * sigma))
        # Normalize weights
        src_node_weights = src_node_weights / (
            torch.sum(src_node_weights, dim=1, keepdim=True) + 1e-8
        )
        return src_node_indices, src_node_weights

    @staticmethod
    def reconstruct_from_optimization_histories(
        histories: list[OptimizationHistory],
        src_pcd: torch.Tensor,
        global_deform: bool,
    ) -> torch.Tensor:
        warped_src_pcd = src_pcd.detach().clone()
        if global_deform:
            # Apply only the last history
            last_history = histories[-1]
            src_node_indices, src_node_weights = (
                last_history.src_node_indices,
                last_history.src_node_weights,
            )
            if warped_src_pcd.shape[0] != last_history.warped_src_pcd.shape[0]:
                loguru.logger.info(
                    "Reassigning node indices and weights for global deformation reconstruction..."
                )
                src_node_indices, src_node_weights = (
                    NonRigidIcp.reassign_indices_and_weights(
                        warped_src_pcd,
                        last_history.graph.poss,
                        last_history.src_node_indices.shape[1],
                    )
                )
            warped_src_pcd = warp_embedded_deformation(
                warped_src_pcd,
                last_history.graph.poss,
                last_history.graph.Rs,
                last_history.graph.ts,
                src_node_indices,
                src_node_weights,
            )
        else:
            # Apply all histories sequentially
            for history in histories:
                src_node_indices, src_node_weights = (
                    history.src_node_indices,
                    history.src_node_weights,
                )
                if warped_src_pcd.shape[0] != history.warped_src_pcd.shape[0]:
                    loguru.logger.info(
                        "Reassigning node indices and weights for sequential deformation reconstruction..."
                    )
                    src_node_indices, src_node_weights = (
                        NonRigidIcp.reassign_indices_and_weights(
                            warped_src_pcd,
                            history.graph.poss,
                            history.src_node_indices.shape[1],
                        )
                    )
                warped_src_pcd = warp_embedded_deformation(
                    warped_src_pcd,
                    history.graph.poss,
                    history.graph.Rs,
                    history.graph.ts,
                    src_node_indices,
                    src_node_weights,
                )
                warped_src_pcd = warped_src_pcd.detach().clone()

        return warped_src_pcd
