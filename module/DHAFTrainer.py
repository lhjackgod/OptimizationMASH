from module.DHAFAnchorOptimizer import DHAFAnchorOptimizer
from module.FLAMEDatasetGenerator import FLAMEDatasetGenerator
from module.AnchorFormer import AnchorFormer
from typing import Union, Dict, List, Tuple, Optional
import torch
import os
from loguru import logger
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
class DHAFTrainer:
    """DHAF锚点优化训练器"""
    def __init__(self,
                flame_dataset : FLAMEDatasetGenerator,
                # Anchor-Format Parameters
                use_anchor_former: bool = True,
                anchor_former_config: Optional[Dict] = None,
                # DHAF Parameters
                anchor_num: int = 400,
                points_per_submesh: int = 1024,
                mask_degree_max: int = 3,
                sh_degree_max: int = 2,
                # Train Paramters
                device: str = 'cuda',
                lr_anchor_former: float = 1e-4,
                lr_dhaf: float = 2e-3,
                min_lr: float = 1e-3,
                warmup_step_num: int = 80,
                factor: float = 0.8,
                patience: int = 2,
                batch_size: int = 8,
                num_epochs: int = 100,
                save_result_floder_path: Union[str, None] = None,
                save_log_floder_path: Union[str, None] = None):
        
        self.flame_dataset = flame_dataset
        self.batch_size = batch_size
        self.device = device
        self.num_epochs = num_epochs
        self.use_anchor_former = use_anchor_former

        # Training parameters from MASH
        self.lr_dhaf = lr_dhaf
        self.lr_anchor_former = lr_anchor_former
        self.min_lr = min_lr
        self.warmup_step_num = warmup_step_num
        self.factor = factor
        self.patience = patience

        # create floder
        if save_result_floder_path is not None:
            os.makedirs(save_result_floder_path, exist_ok = True)
            self.save_result_floder = save_result_floder_path
        else:
            self.save_result_floder = 'output/results'
            os.makedirs(self.save_result_floder, exist_ok = True)
        if save_log_floder_path is not None:
            os.makedirs(save_log_floder_path, exist_ok = True)
            self.save_log_floder = save_log_floder_path
        else:
            self.save_log_floder = 'output/logs'
            os.makedirs(self.save_log_floder, exist_ok = True)

        #initialize DHAF optimizer
        self.dhaf_optimizer = DHAFAnchorOptimizer(
            flame_dataset = flame_dataset,
            points_per_submesh = points_per_submesh,
            anchor_num = anchor_num,
            mask_degree_max = mask_degree_max,
            sh_degree_max = sh_degree_max,
            device = device
        )
        # initialize Anchor-former
        if self.use_anchor_former:
            if anchor_former_config is None:
                anchor_former_config= {
                    'num_anchors' : anchor_num,
                    'shape_dim' : 100,
                    'expression_dim' : 50,
                    'd_model' : 512,
                    'nhead' : 8,
                    'num_decoder_layers' : 6,
                    'dim_feedforward' : 2048,
                    'dropout' : 0.1
                }
            self.anchor_former = AnchorFormer(**anchor_former_config).to(device)
            logger.info("Initialize Anchor-Former for dynamic anchor generation")
        else:
            self.anchor_former = None
            logger.info("Using static anchor initialization")

        # set optimization
        self._setup_optimization(lr_anchor_former, lr_dhaf)

        self.epoch = 0
        self.step = 0
        self.loss_min = float('inf')
    
    def _setup_optimization(self, lr_anchor_former: float, lr_dhaf: float):
        """set optimization and lr """

        # DHAF Parameters optimizer
        dhaf_params = [
            self.dhaf_optimizer.mask_params,
            self.dhaf_optimizer.sh_params,
            self.dhaf_optimizer.ortho_poses,
        ]

        if not self.use_anchor_former:
            # if don't use the anchor-former, position also need to oprimize from the default way
            dhaf_params.append(self.dhaf_optimizer.positions)
        
        self.optimizer_dhaf = SGD(dhaf_params, lr=lr_dhaf)

        # Anchor-Former optimizer
        if self.use_anchor_former:
            self.optimizer_anchor_former = Adam(
                self.anchor_former.parameters(),
                lr = lr_anchor_former,
                betas=(0.9, 0.999),
                weight_decay=1e-4
            )

            self.scheduler_anchor_former = CosineAnnealingLR(
                self.optimizer_anchor_former,
                T_max = self.num_epochs,
                eta_min=lr_anchor_former * 0.01
            )

    def train(self):
        """major training cycle"""

        logger.info(f'Start warmup training for {self.warm_steps} epochs')
        


        logger.info(f'Start training for {self.num_epochs} epochs')
        dataset_size = len(self.flame_dataset)

        steps_per_epoch = dataset_size // self.batch_size
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            epoch_losses = []


