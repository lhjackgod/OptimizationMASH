import torch
import numpy as np
from thirdpart.FLAME_PyTorch.flame_pytorch.config import get_config
from thirdpart.FLAME_PyTorch.flame_pytorch.flame import FLAME
from typing import List, Dict, Tuple, Optional
from loguru import logger 
import trimesh
class FLAMEDatasetGenerator:
    """生成多样化的FLAME模型用于锚点优化训练"""
    def __init__(self,
                 num_samples: int = 120,
                 seed: int = 42,
                 shape_param_std: float = 2.0,
                 shape_param_range: Tuple[float,float] = (-2.0, 2.0),
                 include_expressions: bool = False,
                 expression_param_std: float = 1.0,
                 include_poses: bool = False,
                 device: str = "cuda"):
        self.num_samples = num_samples
        self.device = device
        self.include_expressions = include_expressions
        self.include_poses = include_poses
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.config = get_config()
        self.flamelayer = FLAME(self.config)
        self.flamelayer.to(device)
        
        # 生成参数数据集
        self.shape_params = self._generate_shape_params(shape_param_std, shape_param_range)
        self.expression_params = self._generate_expression_params(expression_param_std)
        self.pose_params = self._generate_pose_params()
        
        self.neck_pose = None
        self.eye_pose = None
        
        logger.info(f'Generated FLAME dataset with {num_samples} samples')
        logger.info(f"Shape params: {self.shape_params.shape}")
        logger.info(f"Expression params: {self.expression_params.shape}")
        logger.info(f"Pose params: {self.pose_params.shape}")
    
    def _generate_shape_params(self, std: float, param_range: Tuple[float,float]) -> torch.Tensor:
        shape_params = torch.randn(self.num_samples, 100, device=self.device) * std
        
        shape_params = torch.clamp(shape_params, param_range[0], param_range[1])
        return shape_params
    
    def _generate_expression_params(self, std: float) -> torch.Tensor:
        """生成表情参数 (ψ)"""
        if self.include_expressions:
            # FLAME的表情参数通常是50维
            expression_params = torch.randn(self.num_samples, 50, device=self.device) * std
            expression_params = torch.clamp(expression_params, -2.0, 2.0)
        else:
            # 中性表情
            expression_params = torch.zeros(self.num_samples, 50, device=self.device)
            
        return expression_params
    
    def _generate_pose_params(self) -> torch.Tensor:
        """生成姿态参数 (θ)"""
        if self.include_poses:
            # 生成多样化的头部姿态
            # pose_params[:, :3] : 全局旋转
            # pose_params[:, 3:] : 下颚旋转
            pose_params = torch.zeros(self.num_samples, 6, device=self.device)
            
            # 全局旋转 (绕Y轴的头部转动为主)
            pose_params[:, 1] = torch.rand(self.num_samples, device=self.device) * 60 * np.pi / 180 - 30 * np.pi / 180  # -30° to +30°
            
            # 可选：添加少量X轴和Z轴旋转
            pose_params[:, 0] = torch.rand(self.num_samples, device=self.device) * 20 * np.pi / 180 - 10 * np.pi / 180  # -10° to +10°
            pose_params[:, 2] = torch.rand(self.num_samples, device=self.device) * 20 * np.pi / 180 - 10 * np.pi / 180  # -10° to +10°
            
            # 下颚旋转 (通常较小)
            pose_params[:, 3:] = torch.rand(self.num_samples, 3, device=self.device) * 10 * np.pi / 180 - 5 * np.pi / 180
        else:
            # 中性姿态
            pose_params = torch.zeros(self.num_samples, 6, device=self.device)
            
        return pose_params
    
    def get_canonical_vertices_and_landmarks(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取标准中性FLAME模型的顶点和关键点"""
        zero_shape = torch.zeros(1, 100, device=self.device)
        zero_exp = torch.zeros(1, 50, device=self.device)
        zero_pose = torch.zeros(1, 6, device=self.device)
        
        with torch.no_grad():
            if self.neck_pose is not None and self.eye_pose is not None:
                zero_neck = torch.zeros(1, 3, device=self.device)
                zero_eye = torch.zeros(1, 6, device=self.device)
                vertices, landmarks = self.flamelayer(
                    zero_shape, zero_exp, zero_pose, zero_neck, zero_eye
                )
            else:
                vertices, landmarks = self.flamelayer(
                    zero_shape, zero_exp, zero_pose
                )
        return vertices.squeeze(0), landmarks.squeeze(0)
    
    def get_canonical_vertices(self) -> torch.Tensor:
        """只获取标准中性FLAME模型的顶点"""
        vertices,_ = self.get_canonical_vertices_and_landmarks()
        return vertices
    
    def get_flame_vertices_and_landmarks(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取第idx个FLAME模型的顶点和关键点"""
        if idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.num_samples}")
        shape_param = self.shape_params[idx:idx+1]
        expression_param = self.expression_params[idx:idx+1]  # [1, 50]
        pose_param = self.pose_params[idx:idx+1]  # [1, 6]
        with torch.no_grad():
            if self.neck_pose is not None and self.eye_pose is not None:
                neck_pose = self.neck_pose[idx:idx+1]
                eye_pose = self.eye_pose[idx:idx+1]
                vertices, landmarks = self.flamelayer(
                    shape_param, expression_param, pose_param, neck_pose, eye_pose
                )
            else:
                vertices, landmarks = self.flamelayer(
                    shape_param, expression_param, pose_param
                )
        return vertices.squeeze(0), landmarks.squeeze(0)
    
    def get_batch_vertices(self, batch_size: int = 32, start_idx: int = 0) -> torch.Tensor:
        end_idx = min(start_idx + batch_size, self.num_samples)
        actual_batch_size = end_idx - start_idx
        
        shape_params_batch = self.shape_params[start_idx:end_idx]
        expression_params_batch = self.expression_params[start_idx:end_idx]
        pose_params_batch = self.pose_params[start_idx:end_idx]
        
        with torch.no_grad():
            if self.neck_pose is not None and self.eye_pose is not None:
                neck_pose_batch = self.neck_pose[start_idx:end_idx]
                eye_pose_batch = self.eye_pose[start_idx:end_idx]
                vertices_batch, _ = self.flamelayer(
                    shape_params_batch, expression_params_batch, pose_params_batch,
                    neck_pose_batch, eye_pose_batch
                )
            else:
                vertices_batch, _ = self.flamelayer(
                    shape_params_batch, expression_params_batch, pose_params_batch
                )
        return vertices_batch
    
    def get_total_vertices(self) -> torch.Tensor:
        total_vertices = []
        for i in range(0, self.num_samples, self.config.batch_size):
            vertices_batch = self.get_batch_vertices(self.config.batch_size, i)
            total_vertices.append(vertices_batch)
        return torch.cat(total_vertices, dim=0)

    def get_flame_faces(self):
        return self.flamelayer.faces
    
    def save_sample_as_mesh(self, vertices: torch.Tensor, idx: int, save_path: str):
        faces = self.get_flame_faces()
        
        vertices_up = vertices.detach().cpu().numpy()
        #vertex_color = np.ones([vertices_up.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
        mesh = trimesh.Trimesh(vertices_up, faces)
        mesh.export(save_path)
        logger.info(f"Saved sample {idx} as mesh: {save_path}")
    
    def __len__(self):
        return self.num_samples

if __name__ == '__main__':
    flame_dataset = FLAMEDatasetGenerator(
        num_samples=100,
        device='cuda',
        shape_param_std=2.0,
        include_expression=True,
        include_poses=False,
        seed=42
    )
    print(f"Dataset size: {len(flame_dataset)}")
    canonical_vertices = flame_dataset.get_canonical_vertices()
    print(f"Canonical vertices shape: {canonical_vertices.shape}")
    
    vertices_2, landmarks_2 = flame_dataset.get_flame_vertices_and_landmarks(2)
    print(f'Sample 2 - Vertices: {vertices_2.shape}, landmarks: {landmarks_2.shape}')

    # 保存第2个样本为网格文件
    flame_dataset.save_sample_as_mesh(2, 'sample_2.obj')
    