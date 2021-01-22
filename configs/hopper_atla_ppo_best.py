import os
import json

import sys
from utils import dict_product, iwt, generate_configs

with open("../src/MuJoCo.json") as f:
    BASE_CONFIG = json.load(f)





PARAMS = {
    "game": ["Hopper-v2"],
    "mode": ["adv_ppo"],
    "out_dir": ["atla_ppo_hopper/agents"],
    "norm_rewards": ["returns"],
    "initialization": ["orthogonal"],
    "anneal_lr": [True],
    "ppo_lr_adam": [3e-4],
    "val_lr": [2.5e-4],
    "entropy_coeff": [0.01],
    "adv_clip_eps": [0.2],
    "adv_entropy_coeff": [0.0],
    "adv_ppo_lr_adam": [3e-3],
    "adv_val_lr": [3e-5], 
    "save_iters": [50],
    "train_steps": [2441],
    "robust_ppo_eps": [0.075],  # used for attack
    "value_clipping": [False],
}

generate_configs(BASE_CONFIG, PARAMS)
