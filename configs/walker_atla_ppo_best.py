import os
import json

import sys
from utils import dict_product, iwt, generate_configs

with open("../src/MuJoCo.json") as f:
    BASE_CONFIG = json.load(f)





PARAMS = {
    "game": ["Walker2d-v2"],
    "mode": ["adv_ppo"],
    "out_dir": ["atla_ppo_walker/agents"],
    "norm_rewards": ["returns"],
    "initialization": ["orthogonal"],
    "anneal_lr": [True],
    "ppo_lr_adam": [4e-4],
    "val_lr": [3e-4],
    "entropy_coeff": [3e-4],
    "adv_clip_eps": [0.2],
    "adv_entropy_coeff": [1e-5],
    "adv_ppo_lr_adam": [1e-3],
    "adv_val_lr": [1e-4], 
    "save_iters": [50],
    "train_steps": [2441],
    "robust_ppo_eps": [0.05],  # used for attack
    "value_clipping": [True],
}

generate_configs(BASE_CONFIG, PARAMS)
