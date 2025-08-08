import sys
import os
from thirdpart.ma_sh.Model.mash import Mash
from thirdpart.mesh_graph_cut.mesh_graph_cut.Module.mesh_graph_cutter import MeshGraphCutter
from module.FLAMEDatasetGenerator import FLAMEDatasetGenerator
import trimesh
import torch
import numpy as np

from loguru import logger
class DHAFAnchorOptimizer(Mash):
    def __init__(self,
                 flame_dataset : FLAMEDatasetGenerator,
                 points_per_submesh: int = 1024,
                 anchor_num: int = 400, 
                 mask_degree_max: int = 3,
                 sh_degree_max: int = 2,
                 sample_phi_num: int = 40,
                 sample_theta_num: int = 40,
                 dtype=torch.float32,
                 device: str = "cuda",
                 flame_model_path='flamemodel/'):
        super().__init__(
            anchor_num, mask_degree_max, sh_degree_max,
            sample_phi_num, sample_theta_num, dtype, device
        )
        self.anchor_num = anchor_num
        self.flame_dataset = flame_dataset
        self.device = device
        self.points_per_submesh = points_per_submesh

        # canonical flame mesh
        self.canonical_vertices = self.flame_dataset.get_canonical_vertices()
        self.flame_faces = torch.from_numpy(self.flame_dataset.get_flame_faces()).to(device)
        self.num_vertices = self.canonical_vertices.shape[0]

        os.makedirs('output/tmp', exist_ok=True)

        self._initialize_anchor_positions()
        logger.info(f'DHAFAnchorOptimizer initialized with {anchor_num} anchors')
        logger.info(f'FLAME vertices: {self.num_vertices}, faces: {len(self.flame_faces)}')


    def _initialize_anchor_positions(self):
        """在标准FLAME模型上初始化锚点"""

        logger.info("Initialize anchor positions on canonical FLAME model...")

        self.flame_dataset.save_sample_as_mesh(self.canonical_vertices, 0, 'output/tmp/canonical.obj')
        mesh_graph_cutter = MeshGraphCutter('output/tmp/canonical.obj')
        mesh_graph_cutter.cutMesh(self.anchor_num, self.points_per_submesh)

        self.gt_points = mesh_graph_cutter.sub_mesh_sample_points
        fps_vertex_idxs = mesh_graph_cutter.fps_vertex_idxs
        
        fps_positions = mesh_graph_cutter.vertices[mesh_graph_cutter.fps_vertex_idxs]
        fps_normal = mesh_graph_cutter.vertex_normals[
            fps_vertex_idxs
        ]

        W0 = 0.5 * np.sqrt(1.0 / np.pi)
        self.surface_dist = 1e-4

        # 初始化SH参数
        sh_params = torch.zeros_like(self.sh_params)
        sh_params[:, 0] = self.surface_dist / W0

        with torch.no_grad():
            self.loadParams(
                sh_params=sh_params,
                positions=fps_positions + self.surface_dist * fps_normal,
                face_forward_vectors=-fps_normal,
            )
        
        self.anchor_vertex_indices = fps_vertex_idxs.to(self.device)
        logger.info(f"Initialized {self.anchor_num} anchors using FPS sampling")