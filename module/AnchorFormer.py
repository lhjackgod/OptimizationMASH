import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List

class AnchorFormer(nn.Module):
    """
    从FLAME参数预测完整的MASH参数
    输入: FLAME shape参数 β 和 expression参数 ψ
    输出: 所有MASH参数 (mask_params, sh_params, ortho_poses, positions)
    """
    def __init__(self,
                num_anchors: int = 400,
                shape_dim: int = 100,
                expression_dim: int = 50,
                mask_degree_max: int = 3,
                sh_degree_max: int = 2,
                d_model: int = 768,
                nhead: int = 8,
                num_encoder_layers: int = 4,
                num_decoder_layer: int = 6,
                dim_feedforward: int = 2048,
                dropout: float = 0.1,
                activation: str = 'gelu',
                normalize_before: bool = True,
                flame_vertex_masks: Dict[str, torch.Tensor] = None
                ):
        super().__init__()
        self.num_anchors = num_anchors
        self.d_model = d_model
        self.mask_degree_max = mask_degree_max
        self.sh_degree_max = sh_degree_max

        # 计算参数维度
        self.sh_dim = (sh_degree_max + 1) ** 2  # 球谐系数数量
        self.mask_dim = 2 * mask_degree_max + 1  # 掩码参数维度

        # FLAME region defined
        self.region_names = [
            'forehead', 'nose', 'lips', 
            'left_eye_region', 'right_eye_region',
            'face_only', 'eye_region_only',
            'neck', 'boundary'
        ]

        self.num_regions = len(self.region_names)
        self.flame_vertex_masks = flame_vertex_masks

        # ========== 1. FLAME参数编码器 ==========
        self.flame_encoder = FlameParameterEncoder(shape_dim, expression_dim, d_model)
        # ========== 2. 区域感知编码器 ==========
        self.region_encoder = RegionAwareEncoder(
            d_model, self.num_regions, nhead,
            num_encoder_layers, dim_feedforward, dropout
        )
        # ========== 3. 锚点分配策略 ==========
        # 为每个区域预分配锚点数量（可学习）
        self.region_anchor_weights = nn.Parameter(torch.ones(self.num_regions) / self.num_regions)

        # ========== 4. 参数解码器 ==========
        # 为不同类型参数创建查询
        self.position_queries = nn.Parameter(torch.randn(num_anchors, d_model))
        self.ortho_queries = nn.Parameter(torch.randn(num_anchors, d_model))
        self.sh_queries = nn.Parameter(torch.randn(num_anchors, d_model))
        self.mask_queries = nn.Parameter(torch.randn(num_anchors, d_model))

        # 共享位置编码
        self.anchor_pos_encoding = nn.Parameter(torch.randn(num_anchors, d_model))
        
        #Transform decoder
        self.parameter_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model, nhead, dim_feedforward, dropout,
                activation = 'gelu', batch_first=True, norm_first=True
            ),
            num_layers=num_decoder_layer
        )

        # ========== 5. 输出头 ==========
        # position prediction
        self.position_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3)
        )

        # sh prediction
        self.sh_head = SHParameterHead(d_model, sh_degree_max)

        # 正交姿态预测
        self.ortho_head = OrthoParameterHead(d_model)
        
         # 掩码参数预测（各向异性视锥）
        self.mask_head = MaskParameterHead(d_model, mask_degree_max)
        
        self._reset_parameters()

    def _reset_parameters(self):
        for name, param in self.named_parameters():
            if 'queries' in name or 'encoding' in name:
                nn.init.nomal_(param, std=0.02)
            elif param.dim() > 1 and 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, 
                shape_params: torch.Tensor,
                expression_params: Optional[torch.Tensor] = None,
                ) -> Dict[str, torch.Tensor]:
        """
        Args:
            shape_params: [B, 100] FLAME shape参数
            expression_params: [B, 50] FLAME expression参数
        
        Returns:
            Dict包含:
                - positions: [B, N_anchors, 3]
                - sh_params: [B, N_anchors, sh_dim]
                - mask_params: [B, N_anchors, mask_dim]
                - ortho_poses: [B, N_anchors, 6]
        """
        batch_size = shape_params.shape[0]
        # 1. 编码FLAME参数
        flame_features = self.flame_encoder(shape_params, expression_params)# [B, d_model]
        # 2. 生成区域特征（如果有mask）
        if self.flame_vertex_masks is not None:
            region_features, region_importance = self.region_encoder(
                flame_features, self.flame_vertex_masks, self.region_names
            )
            memory = region_features# [B, num_regions, d_model]
        else: 
            # 没有mask时，直接使用全局特征
            memory = flame_features.unsqueeze(1)# [B, 1, d_model]
            region_importance = None

        # 3. 准备所有查询向量
        queries_list = []
        for q in [self.position_queries, self.sh_queries,
                  self.mask_queries, self.ortho_queries]:
            queries_list.append(q.unsqueeze(0).expand(batch_size, -1, -1))
        
        all_queries = torch.cat(queries_list, dim=1)  # [B, 4*num_anchors, d_model]
        # 位置编码
        pos_encoding = self.anchor_pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)
        pos_encoding = pos_encoding.repeat(1, 4, 1)  # 复制4次
        # 4. 解码参数
        decoded_features = self.parameter_decoder(
            tgt=all_queries + pos_encoding,
            memory=memory
        )
        # 5. 分离不同类型的特征
        chunk_size = self.num_anchors
        pos_feat, sh_feat, mask_feat, ortho_feat = decoded_features.chunk(4, dim=1)
        # 6. 预测各参数
        positions = self.position_head(pos_feat) * 0.2  # 缩放到合理范围
        sh_params = self.sh_head(sh_feat)
        mask_params = self.mask_head(mask_feat)
        ortho_poses = self.ortho_head(ortho_feat)
        
        outputs = {
            'positions': positions,        # [B, N_anchors, 3]
            'sh_params': sh_params,        # [B, N_anchors, sh_dim]
            'mask_params': mask_params,    # [B, N_anchors, mask_dim]
            'ortho_poses': ortho_poses,    # [B, N_anchors, 6]
        }
        
        if region_importance is not None:
            outputs['region_importance'] = region_importance
        
        return outputs

class FlameParameterEncoder(nn.Module):
    """FLAME参数编码器"""
    def __init__(self, shape_dim: int, expression_dim: int, d_model: int):
        super().__init__()
        
        # Shape encoder
        self.shape_encoder = nn.Sequential(
            nn.Linear(shape_dim, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model)
        )

        # Expression encoder
        self.expression_encoder = nn.Sequential(
            nn.Linear(expression_dim, d_model // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 4, d_model // 2),
            nn.LayerNorm(d_model // 2)
        )

        self.fusion = nn.Sequential(
            nn.Linear(d_model + d_model // 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

    def forward(self, shape_params: torch.Tensor, expression_params: Optional[torch.Tensor] = None) -> torch.Tensor:
        shape_feat = self.shape_encoder(shape_params)

        if expression_params is not None:
            expr_feat = self.expression_encoder(expression_params)

            combined = torch.cat([shape_feat, expr_feat], dim = -1)
            return self.fusion(combined)
        
        return shape_feat
    
class RegionAwareEncoder(nn.Module):
    """基于FLAME区域的特征编码器"""
    def __init__(self, d_model: int, num_regions: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.num_regions = num_regions
        self.d_model = d_model

        # region insert
        self.region_embeddings = nn.Parameter(torch.randn(num_regions, d_model))
        self.region_projections = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_regions)
        ])

        self.region_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout,
                activation='gelu', batch_first=True
            ),
            num_layers = num_layers
        )

        self.importance_head = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
    
    def forward(self, global_features: torch.Tensor,
               vertex_masks: Dict[str, torch.Tensor],
               region_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch_size = global_features.shape[0]
        # generate feature for ever regions
        region_features = []
        for i, region_name in enumerate(region_names):
            region_embed = self.region_embeddings[i].unsqueeze(0).expand(batch_size, -1)
            region_feat = global_features + region_embed

            # 区域特定的投影
            region_feat = self.region_projections[i](region_feat)
            region_features.append(region_feat)
        
        # 堆叠区域特征
        region_features = torch.stack(region_features, dim = 1)# [B, num_regions, d_model]
        # 区域交互
        region_features = self.region_transformer(region_features)
        # calculate the important of the region
        importance = self.importance_head(region_features).squeeze(-1) # [B, num_regions]
        
        return region_features, importance
    
class MaskParameterHead(nn.Module):
    """专门用于预测MASH掩码参数的模块"""
    def __init__(self, d_model: int, mask_degree_max: int = 3):
        super().__init__()
        self.mask_degree_max = mask_degree_max
        self.mask_dim = 2 * mask_degree_max + 1

        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.LayerNorm(d_model // 2),
            nn.Linear(d_model // 2, self.mask_dim)
        )

        self._init_weight()
    def _init_weight(self):
        # 特殊初始化，使初始视锥接近均匀
        with torch.no_grad():
            final_layer = self.projection[-1]
            # DC分量初始化为-0.4（经过sigmoid后约0.4）
            nn.init.constant_(final_layer.bias[0], -0.4)
            # 高频分量初始化为小值
            if self.mask_dim > 1:
                final_layer.weight.data[:, 1:] *= 0.1
                nn.init.zeros_(final_layer.bias[1:])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, N_anchors, d_model]
        Returns:
            mask_params: [B, N_anchors, mask_dim]
        """
        return self.projection(features)

class SHParameterHead(nn.Module):
    """预测球谐系数（距离场）"""
    def __init__(self, d_model: int, sh_degree_max: int = 2):
        super().__init__()
        self.sh_degree_max = sh_degree_max
        self.sh_dim = (sh_degree_max + 1) ** 2

        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.LayerNorm(d_model // 2),
            nn.Linear(d_model // 2, self.sh_dim)
        )
        self._init_weights()

    def _init_weights(self):
        # 初始化使得DC分量为正（表示有效距离）
        with torch.no_grad():
            final_layer = self.projection[-1]
            # DC分量初始化为小的正值
            nn.init.constant_(final_layer.bias[0],0.1)
            # 高阶系数初始化为0附近
            if self.sh_dim > 1:
                final_layer.weight.data[:, 1:] *= 0.1
                nn.init.zeros_(final_layer.bias[1:])
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        sh_params = self.projection(features)
        # 确保DC分量为正
        sh_params_reg = sh_params.clone()
        sh_params_reg[..., 0] = F.relu(sh_params[..., 0]) + 0.01
        return sh_params_reg

class OrthoParameterHead(nn.Module):
    """预测正交姿态参数"""
    def __init__(self, d_model: int):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 6)
        )
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        预测两个正交的3D向量
        Args:
            features: [B, N_anchors, d_model]
        Returns:
            ortho_poses: [B, N_anchors, 6]
        """
        ortho_raw = self.projection(features)
        # Gram-Schmidt正交化
        v1 = ortho_raw[..., :3]
        v2 = ortho_raw[..., 3:]
        
        # 第一个向量归一化
        v1 = F.normalize(v1, dim=-1)
        
        # 第二个向量正交化并归一化
        v2 = v2 - (v2 * v1).sum(dim=-1, keepdim=True) * v1
        v2 = F.normalize(v2, dim=-1)
        
        return torch.cat([v1, v2], dim=-1)