from module.FLAMEDatasetGenerator import FLAMEDatasetGenerator
from module.DHAFAnchorOptimizer import DHAFAnchorOptimizer
import os
from loguru import logger

script_path = os.path.abspath(__file__)
script_path = os.path.dirname(script_path)
os.chdir(script_path)

if __name__ == '__main__':
    flame_dataset = FLAMEDatasetGenerator(
        num_samples=120,
        shape_param_range=(-1.0, 1.0),
        device='cuda',
        shape_param_std=2.0,
        include_expressions=True,
        include_poses=False,
        seed=42
    )

    # vertices_100 = flame_dataset.get_total_vertices()
    # for i in range(vertices_100.shape[0]):
    #     vertices = vertices_100[i]
    #     flame_dataset.save_sample_as_mesh(vertices, i, f'output/flameModule/sample_{i}.obj')
    #     logger.info(f'Sample {i} - Vertices: {vertices.shape}')
    DHAFAnchorOptimizer = DHAFAnchorOptimizer(
        flame_dataset = flame_dataset,
        anchor_num = 400, 
        mask_degree_max = 3,
        sh_degree_max = 2,
        sample_phi_num = 40,
        sample_theta_num = 40,
        device = "cuda",
        flame_model_path = 'flamemodel/'
    )
